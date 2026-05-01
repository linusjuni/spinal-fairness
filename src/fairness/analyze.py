"""Fairness analysis orchestrator.

Loads per-case evaluation CSVs, joins demographics, applies grouping
strategies, computes fairness metrics, and writes an EDAReport.

Usage:
    uv run -m src.fairness.analyze \
        --evaluation-csvs eval_all.csv eval_gold.csv eval_silver.csv \
        --ruler-labels all gold silver \
        --mapping case_id_mapping.json \
        [--report-name fairness]
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.data.groups import (
    AgeStrategy,
    EthnicityStrategy,
    GroupingSpec,
    RaceStrategy,
    age,
    ethnicity,
    race,
)
from src.data.loader import load_metadata
from src.data.schemas import Col
from src.eda.report import EDAReport
from src.fairness.metrics import (
    apply_fdr,
    bootstrap_ci,
    compare_fairness_gaps,
    dir_widening,
    disparate_impact_ratio,
    fairness_gap,
    group_summary,
    kruskal_wallis_test,
    mann_whitney_test,
    ols_regression,
    permutation_test,
)
from src.fairness.plots import (
    bootstrap_forest,
    cross_ruler_dir,
    dir_bar_chart,
    violin_by_group,
)
from src.utils.logger import get_logger

logger = get_logger("fairness.analyze")

DICE_COLS = ["dice_vb", "dice_disc", "dice_macro"]
HD95_COLS = ["hd95_vb", "hd95_disc", "hd95_macro"]

Grouping = tuple[str, GroupingSpec | None, str, str]

GROUPINGS: list[Grouping] = [
    # (label, strategy_or_None, source_col, group_col)
    ("sex", None, Col.SEX, Col.SEX),
    ("race_wb", race[RaceStrategy.WHITE_VS_BLACK], Col.RACE, f"{Col.RACE}_group"),
    ("race_wbo", race[RaceStrategy.WHITE_VS_BLACK_VS_OTHER], Col.RACE, f"{Col.RACE}_group"),
    ("race_wn", race[RaceStrategy.WHITE_VS_NONWHITE], Col.RACE, f"{Col.RACE}_group"),
    ("age_3bin", age[AgeStrategy.THREE_BINS], Col.AGE, f"{Col.AGE}_group"),
    ("age_median", age[AgeStrategy.MEDIAN_SPLIT], Col.AGE, f"{Col.AGE}_group"),
    ("ethnicity", ethnicity[EthnicityStrategy.HISPANIC_VS_NOT], Col.ETHNICITY, f"{Col.ETHNICITY}_group"),
]


def _add_derived_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add macro-averaged metrics as derived columns."""
    df = df.with_columns(
        ((pl.col("dice_vb") + pl.col("dice_disc")) / 2.0).alias("dice_macro")
    )
    if "hd95_vb" in df.columns:
        df = df.with_columns(
            ((pl.col("hd95_vb") + pl.col("hd95_disc")) / 2.0).alias("hd95_macro")
        )
    return df


def _detect_score_cols(df: pl.DataFrame) -> list[str]:
    """Auto-detect which score columns are present."""
    cols = list(DICE_COLS)
    if "hd95_vb" in df.columns:
        cols.extend(HD95_COLS)
    return cols


def _apply_grouping(
    df: pl.DataFrame, spec: GroupingSpec | None, source_col: str
) -> pl.DataFrame:
    """Apply a grouping strategy, or return df unchanged for direct columns (e.g. sex)."""
    if spec is None:
        return df
    return spec.apply(df, source_col)


def _n_groups(df: pl.DataFrame, group_col: str) -> int:
    return df[group_col].n_unique()


def _analyze_single_ruler(
    eval_df: pl.DataFrame,
    ruler_label: str,
    metadata: pl.DataFrame,
    report: EDAReport,
) -> dict:
    """Run all fairness analyses for one ruler. Returns collected stats."""
    df = eval_df.join(metadata, on=Col.SERIES_SUBMITTER_ID, how="inner")
    logger.info(f"Ruler '{ruler_label}': joined {df.height} cases")

    score_cols = _detect_score_cols(df)
    all_p_values: list[float] = []
    all_p_labels: list[str] = []
    ruler_stats: dict = {"ruler": ruler_label, "n_cases": df.height}
    ruler_gaps: list[dict] = []
    ruler_gap_labels: list[str] = []
    ci_results: list[dict] = []
    ci_labels: list[str] = []

    for grouping_label, spec, source_col, group_col in GROUPINGS:
        grouped_df = _apply_grouping(df, spec, source_col)
        ng = _n_groups(grouped_df, group_col)
        if ng < 2:
            logger.warning(f"Skipping {grouping_label}: only {ng} group(s)")
            continue

        for score_col in score_cols:
            key = f"{ruler_label}__{score_col}__{grouping_label}"

            summary = group_summary(grouped_df, score_col, group_col)
            report.save_table(summary, f"summary_{key}")

            gap = fairness_gap(grouped_df, score_col, group_col)
            ruler_stats[f"gap_{key}"] = gap

            if score_col == "dice_macro":
                ruler_gaps.append(gap)
                ruler_gap_labels.append(grouping_label)

            if ng == 2:
                test_result = mann_whitney_test(grouped_df, score_col, group_col)
            else:
                test_result = kruskal_wallis_test(grouped_df, score_col, group_col)
            ruler_stats[f"test_{key}"] = test_result
            all_p_values.append(test_result["p"])
            all_p_labels.append(key)

            ci = bootstrap_ci(
                grouped_df, score_col, group_col,
                disparate_impact_ratio, seed=42,
            )
            ruler_stats[f"bootstrap_dir_{key}"] = ci

            if score_col == "dice_macro":
                ci_results.append(ci)
                ci_labels.append(grouping_label)

            perm = permutation_test(
                grouped_df, score_col, group_col,
                disparate_impact_ratio, seed=42,
            )
            ruler_stats[f"permtest_dir_{key}"] = perm

            violin_by_group(
                grouped_df, score_col, group_col, report,
                title=f"{ruler_label}: {score_col} by {grouping_label}",
            )

    if all_p_values:
        corrected = apply_fdr(all_p_values)
        fdr_table = pl.DataFrame({
            "test": all_p_labels,
            "p_raw": all_p_values,
            "p_fdr": corrected,
        })
        report.save_table(fdr_table, f"fdr_{ruler_label}")
        ruler_stats["fdr_n_significant_005"] = int(sum(1 for p in corrected if p < 0.05))

    if ruler_gaps:
        dir_bar_chart(ruler_gaps, ruler_gap_labels, report, fig_name=f"dir_bar_{ruler_label}")

    if ci_results:
        bootstrap_forest(ci_results, ci_labels, report, fig_name=f"forest_{ruler_label}")

    try:
        volume_covariates = [Col.SEX, Col.RACE, Col.AGE]
        ols_df = _apply_grouping(df, race[RaceStrategy.WHITE_VS_BLACK], Col.RACE)
        ols_result = ols_regression(ols_df, "dice_macro", volume_covariates)
        ruler_stats[f"ols_{ruler_label}"] = ols_result
    except Exception as e:
        logger.warning(f"OLS regression failed for {ruler_label}: {e}")

    return ruler_stats


def run(
    evaluation_csvs: list[Path],
    ruler_labels: list[str],
    mapping_path: Path,
    report_name: str = "fairness",
) -> None:
    """Main orchestrator: load CSVs, join demographics, compute fairness metrics."""
    if len(evaluation_csvs) != len(ruler_labels):
        msg = f"Got {len(evaluation_csvs)} CSVs but {len(ruler_labels)} labels"
        raise ValueError(msg)

    metadata = load_metadata()
    logger.info("Loaded metadata", n=metadata.height)

    all_ruler_stats: dict[str, dict] = {}
    all_ruler_gaps: dict[str, list[dict]] = {}

    with EDAReport(report_name, report_type="fairness") as report:
        for csv_path, ruler_label in zip(evaluation_csvs, ruler_labels):
            eval_df = pl.read_csv(csv_path)
            eval_df = _add_derived_columns(eval_df)
            logger.info(f"Loaded {ruler_label}", cases=eval_df.height, columns=eval_df.columns)

            ruler_stats = _analyze_single_ruler(eval_df, ruler_label, metadata, report)
            all_ruler_stats[ruler_label] = ruler_stats

            report.log_stat(f"ruler_{ruler_label}", ruler_stats)

            dice_macro_gaps = {
                k: v for k, v in ruler_stats.items()
                if k.startswith("gap_") and "__dice_macro__" in k
            }
            all_ruler_gaps[ruler_label] = list(dice_macro_gaps.values())

        if len(ruler_labels) > 1:
            _cross_ruler_comparison(all_ruler_gaps, ruler_labels, report)


def _cross_ruler_comparison(
    all_ruler_gaps: dict[str, list[dict]],
    ruler_labels: list[str],
    report: EDAReport,
) -> None:
    """Compare fairness gaps across rulers."""
    logger.info("Running cross-ruler comparison")

    for grouping_idx, (grouping_label, _, _, _) in enumerate(GROUPINGS):
        gaps_for_grouping = []
        labels_for_grouping = []
        for ruler in ruler_labels:
            ruler_gap_list = all_ruler_gaps.get(ruler, [])
            if grouping_idx < len(ruler_gap_list):
                gaps_for_grouping.append(ruler_gap_list[grouping_idx])
                labels_for_grouping.append(ruler)

        if len(gaps_for_grouping) > 1:
            comparison = compare_fairness_gaps(gaps_for_grouping, labels_for_grouping)
            report.save_table(comparison, f"comparison_{grouping_label}")

    if "gold" in ruler_labels and "silver" in ruler_labels:
        gold_gaps = all_ruler_gaps.get("gold", [])
        silver_gaps = all_ruler_gaps.get("silver", [])

        for i, (gl, sg) in enumerate(zip(gold_gaps, silver_gaps)):
            label = GROUPINGS[i][0] if i < len(GROUPINGS) else f"group_{i}"
            widening = dir_widening(gl["dir"], sg["dir"])
            report.log_stat(f"dir_widening_{label}", widening)
            logger.info(
                f"DIR widening ({label})",
                gold=f"{gl['dir']:.4f}",
                silver=f"{sg['dir']:.4f}",
                widening=f"{widening['widening_pct']:.1f}%",
            )

    grouping_labels = [g[0] for g in GROUPINGS[:len(next(iter(all_ruler_gaps.values()), []))]]
    if grouping_labels:
        cross_ruler_dir(all_ruler_gaps, grouping_labels, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fairness analysis of segmentation metrics")
    parser.add_argument("--evaluation-csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--ruler-labels", type=str, nargs="+", required=True)
    parser.add_argument("--mapping", type=Path, required=True, help="case_id_mapping.json")
    parser.add_argument("--report-name", type=str, default="fairness")
    args = parser.parse_args()

    run(
        evaluation_csvs=args.evaluation_csvs,
        ruler_labels=args.ruler_labels,
        mapping_path=args.mapping,
        report_name=args.report_name,
    )
