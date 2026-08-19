import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# MedVision AI - Module Performance Graph
# ---------------------------------------------------------

modules = [
    "RAG Context Retrieval\n(Top-1 Retrieval Accuracy)",
    "YOLO26 Detection\n(mAP@0.5)",
    "Qwen2.5 Clinical Reporting\n(Overall Reporting Score)"
]

# Experimentally obtained values
values = [92.31, 95.61, 97.92]

# Light professional colors
colors = [
    "#86C98A",   # light green
    "#7BAAF7",   # light blue
    "#F28E8E"    # light red
]

# Create figure
fig, ax = plt.subplots(figsize=(11, 5.5))

y = np.arange(len(modules))

bars = ax.barh(
    y,
    values,
    color=colors,
    height=0.48,
    edgecolor="none"
)

# ---------------------------------------------------------
# Y-axis labels
# ---------------------------------------------------------

ax.set_yticks(y)
ax.set_yticklabels(
    modules,
    fontsize=11,
    fontweight="bold"
)

# Put RAG at top
ax.invert_yaxis()

# ---------------------------------------------------------
# X-axis
# ---------------------------------------------------------

ax.set_xlim(0, 105)

ax.set_xlabel(
    "Performance Score (%)",
    fontsize=11,
    fontweight="bold"
)

ax.set_xticks(np.arange(0, 101, 20))

# Light vertical grid
ax.xaxis.grid(
    True,
    linestyle="--",
    alpha=0.25
)

ax.set_axisbelow(True)

# ---------------------------------------------------------
# Values at end of bars
# ---------------------------------------------------------

for bar, value in zip(bars, values):

    ax.text(
        value + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}%",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

ax.set_title(
    "MedVision AI: System Module Performance",
    fontsize=15,
    fontweight="bold",
    pad=18
)

# ---------------------------------------------------------
# Clean research-paper appearance
# ---------------------------------------------------------

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# ---------------------------------------------------------
# Save high-resolution figure
# ---------------------------------------------------------

output_file = "Figure5_Module_Performance.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print("=" * 60)
print("Figure 5 generated successfully")
print(f"Saved to: {output_file}")
print("=" * 60)

plt.show()