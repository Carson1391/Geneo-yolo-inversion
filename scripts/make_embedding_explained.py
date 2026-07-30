"""Generate a clean, publication-quality 2-panel figure:

Panel A: Conceptual flow diagram
  - Text path: "A pink umbrella" -> Tokenizer -> 248-dim vector
  - Image path: photo -> YOLOv3 -> 904,995-dim vector
  - Both converge to "Shared Latent Space"
  - Arrow to "Reconstruction Possible"

Panel B: Scatter plot of first 1,014 embedding values
  - Dots only, channel boundaries, labels, Box H spike annotation

Designed for non-technical readers in a data privacy research paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec

# -- Paths --
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent
EMB_PATH = OUTPUT_DIR / "model" / "embeddings" / "test_emb_000000000036.npz"
IMG_PATH = OUTPUT_DIR / "model" / "images" / "000000000036.jpg"

# -- Load embedding --
with np.load(EMB_PATH, allow_pickle=False) as pkg:
    emb = pkg["embedding"].astype(np.float32)
if emb.ndim == 2:
    emb = emb[0]

# -- Channel data --
boundaries = [0, 169, 338, 507, 676, 845, 1014]
channel_labels = [
    "Objectness",
    "Box X",
    "Box Y",
    "Box W",
    "Box H",
    "Class 0",
]
channel_sublabels = [
    "0-168",
    "169-338",
    "338-507",
    "507-676",
    "676-845",
    "845-1014",
]
region_colors = ["#d4e6f1", "#d5f5e3", "#fef9e7", "#fadbd8", "#f5b7b1", "#d6eaf8"]
region_dark = ["#2c6fbb", "#27ae60", "#f39c12", "#e67e22", "#c0392b", "#2980b9"]

# ================================================================
# Build figure: 2 panels (A on top, B on bottom)
# ================================================================

fig = plt.figure(figsize=(14, 16), facecolor="white")
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.1],
              hspace=0.12, left=0.06, right=0.94, top=0.96, bottom=0.04)

# ================================================================
# PANEL A: Conceptual flow diagram
# ================================================================

ax_a = fig.add_subplot(gs[0])
ax_a.set_xlim(0, 100)
ax_a.set_ylim(0, 100)
ax_a.set_aspect("equal")
ax_a.axis("off")

# Panel label
ax_a.text(2, 95, "A", fontsize=16, fontweight="bold", va="top", color="#1a1a2e")

# -- Layout grid --
# Text path:  left side, x=12-38, flowing top to bottom
# Image path: left side, x=12-38, flowing bottom to top
# Shared:     center-right, x=55-80
# Risk:       far right, x=85-98

# Colors
C_TEXT = "#2c6fbb"
C_TEXT_BG1 = "#e3effa"
C_TEXT_BG2 = "#c5ddf5"
C_IMG = "#c0563e"
C_IMG_BG1 = "#fce8e3"
C_IMG_BG2 = "#f5c8bc"
C_SHARED = "#5b2c8f"
C_SHARED_BG = "#ede3f5"
C_RISK = "#b03030"
C_RISK_BG = "#fce8e8"

# Uniform box dimensions
BOX_W = 28
BOX_H = 9
TEXT_X = 25  # center x for text/image path boxes
IMG_X = 25

def box(cx, cy, w, h, text, fontsize=10, bg="#f8f9fa", border="#adb5bd",
        text_color="#1a1a2e", bold=False, mono=True):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.3",
                          facecolor=bg, edgecolor=border, linewidth=1.8)
    ax_a.add_patch(rect)
    weight = "bold" if bold else "normal"
    family = "monospace" if mono else "sans-serif"
    ax_a.text(cx, cy, text, ha="center", va="center",
              fontsize=fontsize, color=text_color, fontweight=weight,
              family=family, linespacing=1.3)

def arrow(x1, y1, x2, y2, color="#666666", lw=2.5):
    ax_a.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=20,
        color=color, linewidth=lw,
    ))

# ---- TEXT PATH (top, flowing down) ----
ax_a.text(TEXT_X, 88, "Text Path", fontsize=12, fontweight="bold",
          ha="center", va="center", color=C_TEXT)

box(TEXT_X, 80, BOX_W, BOX_H, '"A pink umbrella"', fontsize=11,
    bg=C_TEXT_BG1, border=C_TEXT, bold=True)

arrow(TEXT_X, 75.5, TEXT_X, 71.5, C_TEXT)

box(TEXT_X, 67, BOX_W, BOX_H, 'Tokenizer\n["A", "pink", "umbrella"]',
    fontsize=9, bg=C_TEXT_BG2, border=C_TEXT)

arrow(TEXT_X, 62.5, TEXT_X, 58.5, C_TEXT)

box(TEXT_X, 54, BOX_W, BOX_H,
    '[0.84, -0.21, ...]\n248 values',
    fontsize=10, bg="#a8c8e8", border=C_TEXT)

# Arrow from text vector -> shared space
arrow(TEXT_X + BOX_W/2, 54, 55, 48, C_TEXT, lw=2.5)

# ---- IMAGE PATH (bottom, flowing up) ----
ax_a.text(IMG_X, 46, "Image Path", fontsize=12, fontweight="bold",
          ha="center", va="center", color=C_IMG)

# Show actual image
try:
    from PIL import Image as PILImage
    img = PILImage.open(IMG_PATH)
    img = img.resize((60, 60))
    img_ax = fig.add_axes([0.16, 0.26, 0.05, 0.04])
    img_ax.imshow(img)
    img_ax.axis("off")
    border = FancyBboxPatch((IMG_X - 5, 36), 10, 6,
                            boxstyle="round,pad=0.2",
                            facecolor="none", edgecolor=C_IMG, linewidth=1.8)
    ax_a.add_patch(border)
except Exception:
    box(IMG_X, 39, BOX_W, BOX_H, "[image]", fontsize=11,
        bg=C_IMG_BG1, border=C_IMG, bold=True)

arrow(IMG_X, 36, IMG_X, 32, C_IMG)

box(IMG_X, 27, BOX_W, BOX_H, 'YOLOv3 Detector\n[detection grids]',
    fontsize=9, bg=C_IMG_BG2, border=C_IMG)

arrow(IMG_X, 22.5, IMG_X, 18.5, C_IMG)

box(IMG_X, 14, BOX_W, BOX_H,
    '[0.84, -0.21, ...]\n904,995 values',
    fontsize=10, bg="#e8a898", border=C_IMG)

# Arrow from image vector -> shared space
arrow(IMG_X + BOX_W/2, 14, 55, 30, C_IMG, lw=2.5)

# ---- SHARED LATENT SPACE (center) ----
shared_w = 24
shared_h = 28
shared_cx = 67
shared_cy = 39
shared_rect = FancyBboxPatch(
    (shared_cx - shared_w/2, shared_cy - shared_h/2),
    shared_w, shared_h,
    boxstyle="round,pad=0.6",
    facecolor=C_SHARED_BG, edgecolor=C_SHARED, linewidth=2.5)
ax_a.add_patch(shared_rect)

ax_a.text(shared_cx, shared_cy + 8, "Shared\nLatent Space",
          fontsize=12, fontweight="bold", ha="center", va="center",
          color=C_SHARED, family="sans-serif")
ax_a.text(shared_cx, shared_cy - 2,
          "[0.84, -0.21, ...]\n\nSame format.\nSame meaning.",
          fontsize=9, ha="center", va="center", color="#444444",
          family="monospace", linespacing=1.4)

# ---- RISK (right) ----
risk_cx = 90
ax_a.text(risk_cx, 62, "The Risk", fontsize=12, fontweight="bold",
          ha="center", va="center", color=C_RISK)

box(risk_cx, 42, 16, 30,
    'Treated as\n"anonymous"\n\nBut contains\nenough meaning\nto reconstruct\nthe original\nimage.',
    fontsize=9, bg=C_RISK_BG, border=C_RISK, text_color=C_RISK, bold=True)

arrow(shared_cx + shared_w/2, 39, risk_cx - 8, 39, C_RISK, lw=2)

# ---- Caption below panel A ----
ax_a.text(50, 2,
          'Why does a 15-character phrase produce a 900,000-number fingerprint?',
          fontsize=11, ha="center", va="center", color="#555555",
          style="italic", fontweight="bold")

# ================================================================
# PANEL B: Scatter plot with channel boundaries
# ================================================================

ax_b = fig.add_subplot(gs[1])

# Panel label
ax_b.text(-0.02, 1.06, "B", fontsize=16, fontweight="bold",
          va="top", transform=ax_b.transAxes, color="#1a1a2e")

N = 1014
indices = np.arange(0, N, 2)  # 507 points
values = emb[indices]

# Shaded regions
for i in range(6):
    ax_b.axvspan(boundaries[i], boundaries[i+1],
                 alpha=0.15, color=region_colors[i], zorder=0)

# Vertical dashed lines
for b in boundaries[1:-1]:
    ax_b.axvline(x=b, color="#555555", linewidth=1.2, linestyle="--",
                 alpha=0.6, zorder=3)

# Scatter only
ax_b.scatter(indices, values, s=10, c="#1a1a2e", alpha=0.5,
             edgecolors="none", zorder=4)

# Channel labels at top
y_top = values.max() * 1.25
for i in range(6):
    mid = (boundaries[i] + boundaries[i+1]) / 2
    ax_b.text(mid, y_top, f"{channel_labels[i]}\n({channel_sublabels[i]})",
              fontsize=9, fontweight="bold", ha="center", va="bottom",
              color=region_dark[i],
              bbox=dict(boxstyle="round,pad=0.25",
                        facecolor=region_colors[i],
                        edgecolor="#bbbbbb", alpha=0.95))

# Box H spike annotation
box_h_vals = values[(indices >= 676) & (indices < 845)]
ax_b.annotate(
    "Box H spike\n'this image contains tall objects'\n(people are taller than wide)",
    xy=(760, box_h_vals.mean()),
    xytext=(930, box_h_vals.min() * 0.45),
    fontsize=9, fontweight="bold", color="#c1121f",
    arrowprops=dict(arrowstyle="-|>", color="#c1121f", lw=1.8),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5",
              edgecolor="#c1121f", alpha=0.95),
    ha="center", va="center",
)

ax_b.set_xlabel("Vector Index (0 - 1,014)", fontsize=11, fontweight="bold")
ax_b.set_ylabel("Embedding Value", fontsize=11, fontweight="bold")
ax_b.axhline(y=0, color="#888888", linewidth=0.5, linestyle="-", alpha=0.4)
ax_b.set_xlim(-15, 1030)
ax_b.grid(True, alpha=0.12, axis="y")
ax_b.tick_params(labelsize=10)

# Subtitle
ax_b.text(0.5, 1.02,
          "First 1,014 values of the 904,995-dim embedding: organized channels, not random noise",
          fontsize=11, ha="center", va="bottom", transform=ax_b.transAxes,
          color="#555555", style="italic")

# -- Save --
out_path = OUTPUT_DIR / "embedding_explained.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight",
            facecolor="white", pad_inches=0.4)
plt.close()
print(f"Saved: {out_path}")
print(f"Scatter points: {len(indices)}")
