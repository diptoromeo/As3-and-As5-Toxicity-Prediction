import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
ratios = [0, 10, 20, 30]
x = np.array(ratios)

data = {
    r"As$^{3+}$ Day-10  ($N$=49)": {
        "acc":  [0.691, 0.753, 0.613, 0.693],
        "f1":   [0.679, 0.743, 0.590, 0.638],
        "bacc": [0.728, 0.790, 0.643, 0.695],
        "best": 1,
    },
    r"As$^{3+}$ Day-20  ($N$=35)": {
        "acc":  [0.629, 0.600, 0.543, 0.600],
        "f1":   [0.586, 0.545, 0.498, 0.584],
        "bacc": [0.685, 0.643, 0.620, 0.657],
        "best": 0,
    },
    r"As$^{5+}$ Day-10  ($N$=40)": {
        "acc":  [0.675, 0.725, 0.600, 0.475],
        "f1":   [0.539, 0.722, 0.573, 0.448],
        "bacc": [0.603, 0.797, 0.687, 0.513],
        "best": 1,
    },
    r"As$^{5+}$ Day-20  ($N$=31)": {
        "acc":  [0.605, 0.619, 0.414, 0.386],
        "f1":   [0.565, 0.538, 0.332, 0.356],
        "bacc": [0.675, 0.680, 0.430, 0.535],
        "best": 0,
    },
}

PANEL = ["(a)", "(b)", "(c)", "(d)"]

C_ACC  = "#2166ac"
C_F1   = "#4dac26"
C_BACC = "#d01c8b"
C_ZONE = "#f4a582"

plt.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            8,
    "axes.linewidth":       0.6,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "lines.solid_capstyle": "round",
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

fig = plt.figure(figsize=(7.20, 6.30))
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    hspace=0.60, wspace=0.38,
    left=0.09, right=0.97,
    top=0.83,  bottom=0.17,
)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

for idx, (cond, v) in enumerate(data.items()):
    ax = axes[idx]
    best_i = v["best"]

    ax.axvspan(17.5, 32, color=C_ZONE, alpha=0.18, lw=0, zorder=0)
    ax.axvline(x=20, color=C_ZONE, lw=0.9, ls=(0, (4, 3)), alpha=0.75, zorder=1)
    ax.axhline(0.50, color="#bbbbbb", lw=0.55, ls=":", zorder=0)

    kw = dict(lw=1.6, zorder=3, clip_on=False)
    ax.plot(x, v["acc"],  color=C_ACC,  marker="o", ms=5.0, **kw)
    ax.plot(x, v["f1"],   color=C_F1,   marker="s", ms=4.5, **kw)
    ax.plot(x, v["bacc"], color=C_BACC, marker="^", ms=5.5, **kw)

    ax.plot(x[best_i], v["bacc"][best_i],
            marker="^", ms=10, color=C_BACC,
            markerfacecolor="none", markeredgewidth=1.5,
            zorder=5, clip_on=False)

    delta = v["bacc"][3] - v["bacc"][0]
    sign  = "+" if delta >= 0 else ""
    ax.annotate(
        f"{sign}{delta:.3f}",
        xy=(30, v["bacc"][3]),
        xytext=(4, 1), textcoords="offset points",
        fontsize=6.8, color=C_BACC,
        va="bottom", ha="left", style="italic",
    )

    for xi, yi in zip(x, v["bacc"]):
        offset = 6 if yi == max(v["bacc"]) else -9
        ax.annotate(
            f"{yi:.3f}",
            xy=(xi, yi), xytext=(0, offset),
            textcoords="offset points",
            fontsize=6, color=C_BACC,
            ha="center", va="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["0%\n(baseline)", "10%", "20%", "30%"], fontsize=7.5)
    ax.set_xlim(-3, 34)
    ax.set_ylim(0.28, 0.92)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="y", which="major", color="#e0e0e0", lw=0.5, zorder=0)

    # Place panel label below each subplot, centered below the x-axis label
    ax.text(0.5, -0.35, PANEL[idx],
            transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="center")
    ax.set_title(cond, fontsize=8.5, fontweight="bold", pad=5, loc="left")

    # Ensure axis labels for each figure
    ax.set_xlabel("Missing-data ratio (%)", fontsize=8, labelpad=5, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=8, labelpad=4, fontweight='bold')

handles = [
    mlines.Line2D([], [], color=C_ACC,  marker="o", ms=5,   lw=1.6, label="Accuracy"),
    mlines.Line2D([], [], color=C_F1,   marker="s", ms=4.5, lw=1.6, label="Macro-F1"),
    mlines.Line2D([], [], color=C_BACC, marker="^", ms=5.5, lw=1.6, label="Balanced accuracy"),
    mlines.Line2D([], [], color=C_BACC, marker="^", ms=8,   lw=0,
                  markerfacecolor="none", markeredgewidth=1.4,
                  label="Best ratio (balanced acc.)"),
    mpatches.Patch(facecolor=C_ZONE, alpha=0.35, edgecolor="none",
                   label="Risk zone (>=\u200b20%)"),
]
fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.05),
    ncol=3,
    fontsize=7.8,
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    handlelength=2.2,
    handletextpad=0.55,
    columnspacing=1.4,
    borderpad=0.6,
)

# Caption removed as requested by the user
# caption = (
#     "Fig. X  |  Effect of missing-data ratio (0-30%) on GCN classification "
#     "performance for four arsenic-exposure conditions.\n"
#     "Metrics: accuracy (blue circles), macro-F1 (green squares), balanced accuracy (pink triangles). "
#     "Open triangle = best-performing ratio per condition.\n"
#     "Italic annotation = delta balanced accuracy at 30% vs. 0% baseline.  "
#     "Shaded region (>=\u200b20%) marks the operational risk zone."
# )
# fig.text(
#     0.50, 0.005,
#     s=caption,
#     ha="center", va="bottom",
#     fontsize=7.0, color="#333333",
#     linespacing=1.55,
# )

out = "fig_missing_v3.pdf"
fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
print("saved:", out)
