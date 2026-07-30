"""Generate embedding_progression.png -- 3-panel figure showing
the structure of a single YOLOv3 embedding vector.

Left:   First 15 raw values as text in brackets (3 rows x 5)
Middle: Large arrow pointing left to right
Right:  Scatter plot of first 1,000 values (index vs value)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent
EMB_PATH = OUTPUT_DIR / "model" / "embeddings" / "test_emb_000000000036.npz"

# -- Load the single embedding --
with np.load(EMB_PATH, allow_pickle=False) as pkg:
    emb = pkg["embedding"].astype(np.float32)
if emb.ndim == 2 and emb.shape[0] == 1:
    emb = emb[0]

print(f"Loaded embedding: {emb.shape[0]} values")
print(f"  min={emb.min():.6f}, max={emb.max():.6f}")

# -- Create the 3-panel figure --
fig, (ax_left, ax_mid, ax_right) = plt.subplots(1, 3, figsize=(20, 6), facecolor="white",
                                                 gridspec_kw={"width_ratios": [1.2, 0.4, 2.4]})

fig.suptitle("YOLOv3 Embedding: From Raw Numbers to Visual Structure",
             fontsize=18, fontweight="bold", y=0.98)

# --- Left subplot: first 15 raw values as text ---
ax_left.set_axis_off()
ax_left.set_xlim(0, 1)
ax_left.set_ylim(0, 1)

first_15 = emb[:15]

# Build the text block: 3 rows of 5 values in brackets
lines = []
for row in range(3):
    vals = first_15[row * 5 : (row + 1) * 5]
    formatted = "  ".join(f"[{v:+.6f}]" for v in vals)
    lines.append(formatted)

text_block = "\n".join(lines)

ax_left.text(0.5, 0.65, "First 15 values",
             fontsize=14, fontweight="bold", ha="center", va="center",
             transform=ax_left.transAxes)
ax_left.text(0.5, 0.35, text_block,
             fontsize=11, ha="center", va="center",
             family="monospace", color="#1a1a2e",
             transform=ax_left.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f0", edgecolor="#cccccc"))

# --- Middle subplot: large arrow ---
ax_mid.set_axis_off()
ax_mid.set_xlim(0, 1)
ax_mid.set_ylim(0, 1)

ax_mid.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                                lw=4, mutation_scale=40))
ax_mid.text(0.5, 0.65, "904,995\nvalues", fontsize=12, ha="center", va="center",
            color="#666666", fontweight="bold")
ax_mid.text(0.5, 0.30, "structure\nemerges", fontsize=10, ha="center", va="center",
            color="#999999", style="italic")

# --- Right subplot: scatter of first 1,000 values ---
indices = np.arange(1000)
values = emb[:1000]

ax_right.scatter(indices, values, s=4, c="#2c3e50",
                 alpha=0.7, edgecolors="none")
ax_right.set_xlabel("vector index (0 - 1000)", fontsize=12)
ax_right.set_ylabel("value", fontsize=12)
ax_right.set_title("First 1,000 values of the embedding vector\n"
                   "Distinct peaks = structure, not random noise",
                   fontsize=13, fontweight="bold")
ax_right.axhline(y=0, color="#333333", linewidth=0.5, linestyle="--", alpha=0.5)
ax_right.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = OUTPUT_DIR / "embedding_progression.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3)
plt.close()
print(f"Saved: {out_path}")
