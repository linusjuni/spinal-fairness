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
from matplotlib.patches import Patch

RUN = Path("outputs/fairness/fairness_biased_ruler/20260607_210826")
OUT = RUN / "age_trend_gold_vs_silver.png"
AGE_ORDER = ["<40", "40-60", "60+"]
# FDR-corrected Kruskal-Wallis p for macro Dice ~ age_3bin (from fdr_{ruler}.csv)
FDR_P = {"gold": 0.270, "silver": 0.026}


def load_summary(ruler: str) -> dict[str, dict]:
    df = pl.read_csv(RUN / f"summary_{ruler}__dice_macro__age_3bin.csv")
    return {row["group"]: row for row in df.iter_rows(named=True)}


def main() -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    muted = sns.color_palette("muted")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), sharey=True)

    panels = [
        ("Gold ruler (expert labels)", "gold", muted[8]),    # muted gold/yellow
        ("Silver ruler (twin-model labels)", "silver", muted[7]),  # muted grey
    ]

    for ax, (title, ruler, color) in zip(axes, panels):
        s = load_summary(ruler)
        xs = list(range(len(AGE_ORDER)))
        medians = []

        spread_bits = []
        for x, g in zip(xs, AGE_ORDER):
            r = s[g]
            q25, med, q75 = r["q25"], r["median"], r["q75"]
            mean, std, n = r["mean"], r["std"], r["n"]
            medians.append(med)
            spread_bits.append(f"{g}: n={n}, std={std:.3f}")

            # IQR box (q25-q75) with median line
            ax.add_patch(
                plt.Rectangle((x - 0.22, q25), 0.44, q75 - q25,
                              facecolor=color, alpha=0.30, edgecolor=color, linewidth=1.5)
            )
            ax.plot([x - 0.22, x + 0.22], [med, med], color=color, linewidth=2.6, zorder=5)
            # +/-1 std whisker (the "spread" = noise) centred on the mean
            ax.plot([x, x], [mean - std, mean + std], color=color, linewidth=1.3,
                    alpha=0.8, zorder=3)
            ax.plot([x - 0.05, x + 0.05], [mean - std, mean - std], color=color, linewidth=1.3)
            ax.plot([x - 0.05, x + 0.05], [mean + std, mean + std], color=color, linewidth=1.3)
            # mean diamond
            ax.plot(x, mean, "D", color="black", markersize=8, markerfacecolor="white",
                    markeredgewidth=1.6, zorder=6)

        # connect medians: the downward old-is-lower step
        ax.plot(xs, medians, "--", color=color, linewidth=1.6, alpha=0.9, zorder=4)

        p = FDR_P[ruler]
        verdict = "SIGNIFICANT after FDR" if p < 0.05 else "not significant after FDR"
        vcolor = "#c0392b" if p < 0.05 else "#2e7d32"
        ax.text(0.5, 0.30,
                "per-group spread (noise):\n" + "\n".join(spread_bits),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5,
                color="#333",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#bbb", alpha=0.95))
        ax.text(0.5, 0.10,
                f"Kruskal–Wallis (Dice ~ age):  FDR p = {p:.3f}\n→ {verdict}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5,
                fontweight="bold", color=vcolor,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=vcolor, alpha=0.95))

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(AGE_ORDER)
        ax.set_xlabel("Age group")
        ax.set_xlim(-0.6, len(AGE_ORDER) - 0.4)

    axes[0].set_ylabel("Macro Dice (vertebrae + disc)")
    axes[0].set_ylim(0.6, 1.0)

    legend = [
        Patch(facecolor="grey", alpha=0.30, edgecolor="grey", label="IQR (25–75%)"),
        plt.Line2D([0], [0], color="grey", lw=2.6, label="median"),
        plt.Line2D([0], [0], marker="D", color="black", markerfacecolor="white",
                   lw=0, markersize=8, label="mean"),
        plt.Line2D([0], [0], color="grey", lw=1.3, label="±1 std (spread / noise)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9.5,
               framealpha=0.95, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle(
        "Older spines score slightly lower on BOTH rulers — but only the low-noise "
        "silver ruler makes the gap statistically certain",
        fontsize=12.5, y=0.98,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")

    # console sanity check
    for ruler in ("gold", "silver"):
        s = load_summary(ruler)
        line = "  ".join(f"{g}: med={s[g]['median']:.3f} std={s[g]['std']:.3f}" for g in AGE_ORDER)
        print(f"{ruler:6s} | {line}")


if __name__ == "__main__":
    main()
