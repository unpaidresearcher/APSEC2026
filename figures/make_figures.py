"""
make_figures.py -- paper figures for BASK.
Font: Liberation Serif (metric-compatible Times New Roman substitute).
Outputs (vector PDF) into paper/figs/.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

HERE = os.path.dirname(__file__)
OUT  = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# ---- register Times-compatible font ----
for _f in ["LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"]:
    p = os.path.join(HERE, "fonts", _f)
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": "Liberation Serif",   # == Times New Roman metrics
    "font.size": 11,
    "mathtext.fontset": "stix",
})

# ---- pastel palette ----
PASTEL_RED  = "#e8a9a9"   # no bridge
PASTEL_BLUE = "#a9c9e8"   # + BASK
RED_EDGE    = "#b96b6b"
BLUE_EDGE   = "#5b86b0"

# ======================================================================
# RQ3: all four metrics, no-bridge vs +BASK, 2x2 grouped bars
# ======================================================================
llms = ["StarCoder\nBase-3B", "StarCoder2\n-3B", "DeepSeek\n-6.7B", "Qwen2.5\n-3B"]
data = {
    "VD-S FNR (lower better)": ([92.35,90.16,91.44,90.16], [89.07,89.25,88.89,90.16], (86,94)),
    "F1 (higher better)":      ([0.180,0.217,0.186,0.231], [0.226,0.236,0.199,0.240], (0.15,0.27)),
    "AUC (higher better)":     ([0.821,0.764,0.817,0.824], [0.808,0.834,0.802,0.844], (0.74,0.87)),
    "P-C (higher better)":     ([11.09,13.65,15.78,14.93], [15.14,16.84,13.43,15.57], (10,19)),
}
x = np.arange(len(llms)); w = 0.38
fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
for ax, (title, (nob, bask, ylim)) in zip(axes.flat, data.items()):
    b1 = ax.bar(x - w/2, nob,  w, label="Base model", color=PASTEL_RED,  edgecolor=RED_EDGE,  linewidth=0.7)
    b2 = ax.bar(x + w/2, bask, w, label="+ BASK",    color=PASTEL_BLUE, edgecolor=BLUE_EDGE, linewidth=0.7)
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(llms, fontsize=10.5, rotation=0)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_ylim(*ylim)
    ax.spines[["top","right"]].set_visible(False)
    # bold value labels on top of every bar
    fmt = "{:.2f}" if ylim[1] > 5 else "{:.3f}"
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(fmt.format(bar.get_height()),
                        (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
axes.flat[0].legend(frameon=False, fontsize=10.5, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT,"fig_rq3_bars_1x4.pdf")); plt.close(fig)

# ======================================================================
# Heatmaps: bold, larger numbers; white text on dark cells; no title.
# ======================================================================
def heatmap(matrix, rows, cols, fname, cmap, center=None,
            fmt="{:+.2f}", cbar_label="", black_cells=()):
    # black_cells: iterable of (i,j) that get BLACK text; all others WHITE.
    black = set(black_cells)
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    arr = np.array(matrix, dtype=float)
    if center is not None:
        vmax = np.abs(arr - center).max()
        vmin_, vmax_ = center - vmax, center + vmax
    else:
        vmin_, vmax_ = arr.min(), arr.max()
    im = ax.imshow(arr, cmap=cmap, vmin=vmin_, vmax=vmax_, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=11)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=11)
    for i in range(len(rows)):
        for j in range(len(cols)):
            txt_col = "black" if (i, j) in black else "white"
            ax.text(j, i, fmt.format(arr[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=txt_col)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

# Feature-group signed delta (diverging; negative = removing helps).
# White text everywhere except the "detail" column (col index 2).
heatmap(
    [[1.28, 2.19, -0.55], [0.55, 1.09, -0.18]],
    rows=["CodeBERT","UniXcoder"], cols=["complexity","security","detail"],
    fname="fig_ablation_feat_heat.pdf", cmap="RdBu_r", center=0.0,
    cbar_label="+ = group helps   |   - = group is noise",
    black_cells=[(0, 2), (1, 2)])

# Signal-source FNR (lower better).
# White text everywhere except the CodeBERT / Both cell (row 0, col 2).
heatmap(
    [[88.71, 88.34, 89.25], [87.98, 88.52, 88.34]],
    rows=["CodeBERT","UniXcoder"], cols=["KG only","Features only","Both"],
    fname="fig_ablation_signal_heat.pdf", cmap="Blues_r", center=None,
    fmt="{:.2f}", cbar_label="VD-S FNR",
    black_cells=[(0, 2)])

print("Saved figures to", OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".pdf"): print("  ", f)
