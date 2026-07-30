"""Generate a dot-grid visualization of the YOLOv3 embedding vector.

Produces:
  embedding_visual.png -- 3 YOLOv3 detection grids as dot grids
                          (dot size + color = activation strength)
                          + a 10x10 matrix of raw numbers below

Uses a real pre-extracted embedding from presentation/model/embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR.parent
EMB_DIR = ROOT / "presentation" / "model" / "embeddings"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HEAD_SHAPES = [
    (255, 13, 13),
    (255, 26, 26),
    (255, 52, 52),
]
HEAD_LENGTHS = [c * h * w for c, h, w in HEAD_SHAPES]


def load_embedding(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as pkg:
        emb = pkg["embedding"].astype(np.float32)
    if emb.ndim == 2 and emb.shape[0] == 1:
        emb = emb[0]
    return emb


def split_heads(emb: np.ndarray):
    segments = []
    offset = 0
    for channels, height, width in HEAD_SHAPES:
        length = channels * height * width
        seg = emb[offset : offset + length]
        segments.append(seg.reshape(channels, height, width))
        offset += length
    return segments


# -- Load real embedding --
emb_path = EMB_DIR / "test_emb_000000000036.npz"
if not emb_path.exists():
    print(f"ERROR: embedding file not found: {emb_path}")
    sys.exit(1)

emb = load_embedding(emb_path)
heads = split_heads(emb)

# ============================================================
# FIGURE: Dot grid + 10x10 number matrix
# Top half: 3 dot grids (13x13, 26x26, 52x52)
#   Each dot = one grid cell. Dot size = |max activation| across
#   all 255 channels. Dot color = sign (red=positive, blue=negative).
# Bottom: 10x10 raw number matrix from head13 objectness channel.
# ============================================================

fig = plt.figure(figsize=(20, 16), facecolor="white")
gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 2], hspace=0.35, wspace=0.3)

fig.suptitle(
    "YOLOv3 Embedding: 904,995 Values as Detection Grids\n"
    "3 anchors x (5 box params + 80 COCO classes) per cell -- no color, no pixels",
    fontsize=18, fontweight="bold", y=0.97,
)

head_names = [
    "head13: 255 x 13 x 13\n(43,095 values, large objects)",
    "head26: 255 x 26 x 26\n(172,380 values, medium objects)",
    "head52: 255 x 52 x 52\n(689,520 values, small objects)",
]

# -- Top row: dot grids --
for col, (head, name, (channels, h, w)) in enumerate(zip(heads, head_names, HEAD_SHAPES)):
    ax = fig.add_subplot(gs[0, col])
    ax.set_facecolor("#0a0a12")

    # Max absolute activation across 255 channels per cell
    max_vals = np.abs(head).max(axis=0)  # [h, w]
    sign_vals = head[np.abs(head).argmax(axis=0), np.arange(h)[:, None], np.arange(w)]  # signed value at max

    # Normalize dot sizes
    max_abs = max_vals.max() if max_vals.max() > 0 else 1.0
    sizes = (max_vals / max_abs) * 400  # max dot size in points^2

    # Colors: positive = warm, negative = cool
    colors = np.where(sign_vals >= 0, "#ff4444", "#4488ff")

    # Grid coordinates
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    ax.scatter(xs.ravel(), ys.ravel(), s=sizes.ravel(), c=colors.ravel(),
               alpha=0.8, edgecolors="white", linewidths=0.3)

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(name, fontsize=12, fontweight="bold", color="white", pad=10)
    ax.set_xlabel(f"grid x ({w})", fontsize=10, color="#aaaaaa")
    ax.set_ylabel(f"grid y ({h})", fontsize=10, color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa", labelsize=8)

    # Legend
    ax.scatter([], [], s=200, c="#ff4444", label="positive activation", edgecolors="white")
    ax.scatter([], [], s=200, c="#4488ff", label="negative activation", edgecolors="white")
    ax.scatter([], [], s=30, c="#888888", label="weak", edgecolors="none")
    ax.scatter([], [], s=200, c="#888888", label="strong", edgecolors="none")
    leg = ax.legend(loc="upper right", fontsize=7, framealpha=0.7, facecolor="#1a1a2e")
    for text in leg.get_texts():
        text.set_color("white")

# -- Bottom: 10x10 number matrix spanning all 3 columns --
ax_num = fig.add_subplot(gs[1, :])
ax_num.set_facecolor("white")
ax_num.set_title(
    "Raw values: 10x10 slice from head13, objectness channel (anchor 0)\n"
    "These are 100 of the 904,995 numbers the decoder uses to reconstruct the image",
    fontsize=13, fontweight="bold", pad=15,
)

obj_vals = heads[0][0, :10, :10]  # objectness, first 10x10

# Color the cells
vmin, vmax = obj_vals.min(), obj_vals.max()
im = ax_num.imshow(obj_vals, cmap="RdYlBu_r", interpolation="nearest", aspect="auto",
                   vmin=vmin, vmax=vmax)

# Print numbers in each cell
for i in range(10):
    for j in range(10):
        val = obj_vals[i, j]
        # Choose text color for readability
        mid = (vmin + vmax) / 2
        rng = (vmax - vmin) if (vmax - vmin) > 0 else 1e-6
        color = "white" if abs(val - mid) > rng * 0.35 else "black"
        ax_num.text(j, i, f"{val:+.5f}", ha="center", va="center",
                    fontsize=9, color=color, family="monospace", fontweight="bold")

ax_num.set_xticks(range(10))
ax_num.set_yticks(range(10))
ax_num.set_xlabel("grid x", fontsize=11)
ax_num.set_ylabel("grid y", fontsize=11)

# Colorbar
cbar = plt.colorbar(im, ax=ax_num, fraction=0.02, pad=0.02)
cbar.set_label("activation value (L2-normalized)", fontsize=10)

plt.savefig(OUTPUT_DIR / "embedding_visual.png", dpi=150, bbox_inches="tight",
            facecolor="white", pad_inches=0.3)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'embedding_visual.png'}")
