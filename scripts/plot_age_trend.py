"""Plot the biased-ruler age trend from the real Run 8 summary CSVs.

Mixed model (Dataset001) on the 76 gold-test images, macro Dice by age, graded
against the GOLD ruler (left) and the generated SILVER ruler (right). Same model,
same images, two reference labels.

The story: the age gap (60+ scores lower) is the same tiny size on both rulers,
but the gold ruler is noisy (tall boxes -> "can't tell", n.s.) while the silver
ruler collapses the variance (razor-thin boxes -> the same tiny gap becomes
statistically certain). That variance collapse is the biased-ruler effect.

Reads the per-group summary stats the analysis itself wrote, so it matches Run 8
exactly. No raw metadata needed (works off the synced outputs).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

RUN = Path("outputs/fairness/fairness_biased_ruler/20260607_210826")
OUT = Path("paper/figures/age_trend_gold_vs_silver.png")
AGE_ORDER = ["<40", "40-60", "60+"]
# FDR-corrected Kruskal-Wallis p for macro Dice ~ age_3bin (from fdr_{ruler}.csv)
FDR_P = {"gold": 0.270, "silver": 0.026}


def load_summary(ruler: str) -> dict[str, dict]:
    df = pl.read_csv(RUN / f"summary_{ruler}__dice_macro__age_3bin.csv")
    return {row["group"]: row for row in df.iter_rows(named=True)}


def main() -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    muted = sns.color_palette("muted")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    panels = [
        ("Gold ruler (expert labels)", "gold", muted[0]),  # blue
        (
            "Silver ruler ($M_{\\mathrm{gold}}$ predictions)",
            "silver",
            muted[1],
        ),  # orange
    ]

    for ax, (title, ruler, color) in zip(axes, panels):
        s = load_summary(ruler)
        xs = list(range(len(AGE_ORDER)))
        medians = []

        for x, g in zip(xs, AGE_ORDER):
            r = s[g]
            q25, med, q75 = r["q25"], r["median"], r["q75"]
            mean, std, n = r["mean"], r["std"], r["n"]
            medians.append(med)

            # IQR box
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.25, q25),
                    0.50,
                    q75 - q25,
                    facecolor=color,
                    alpha=0.25,
                    edgecolor=color,
                    linewidth=1.2,
                )
            )
            # Median line
            ax.plot(
                [x - 0.25, x + 0.25],
                [med, med],
                color=color,
                linewidth=2.2,
                solid_capstyle="round",
                zorder=5,
            )
            # ±1 std whiskers
            ax.plot(
                [x, x],
                [mean - std, mean + std],
                color=color,
                linewidth=1.0,
                alpha=0.7,
                zorder=3,
            )
            ax.plot(
                [x - 0.06, x + 0.06],
                [mean - std, mean - std],
                color=color,
                linewidth=1.0,
            )
            ax.plot(
                [x - 0.06, x + 0.06],
                [mean + std, mean + std],
                color=color,
                linewidth=1.0,
            )
            # Mean marker
            ax.plot(
                x,
                mean,
                "o",
                color="white",
                markersize=5,
                markeredgecolor=color,
                markeredgewidth=1.2,
                zorder=6,
            )

        # Connect medians
        ax.plot(xs, medians, "--", color=color, linewidth=1.2, alpha=0.7, zorder=4)

        # Annotation: std and verdict
        p = FDR_P[ruler]
        stds = [s[g]["std"] for g in AGE_ORDER]
        std_range = f"std: {min(stds):.3f}–{max(stds):.3f}"
        sig_label = f"$p_{{\\mathrm{{fdr}}}} = {p:.3f}$"
        if p < 0.05:
            sig_label += " (significant)"
        else:
            sig_label += " (n.s.)"

        ax.text(
            0.97,
            0.05,
            f"{std_range}\n{sig_label}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
        )

        ax.set_title(title, fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels(AGE_ORDER)
        ax.set_xlabel("Age group")
        ax.set_xlim(-0.6, len(AGE_ORDER) - 0.4)

    axes[0].set_ylabel("Macro Dice")
    axes[0].set_ylim(0.75, 1.0)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT}")

    # Console check
    for ruler in ("gold", "silver"):
        s = load_summary(ruler)
        line = "  ".join(
            f"{g}: med={s[g]['median']:.3f} std={s[g]['std']:.3f}" for g in AGE_ORDER
        )
        print(f"{ruler:6s} | {line}")


if __name__ == "__main__":
    main()
