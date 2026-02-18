import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Add parent directory to sys.path for common imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import apply_plot_style

apply_plot_style()

# region constants
# ------------------------------------------------------------------------------
output_dir = "data"
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# region load data
# ------------------------------------------------------------------------------
forest_results_path = f"{output_dir}/forest_results.csv"
model_stats_path = f"{output_dir}/model_stats.csv"

if not os.path.exists(forest_results_path):
    print(
        f"Error: {forest_results_path} not found. "
        "Run the analysis script first."
    )
    sys.exit(1)

forest_df = pd.read_csv(forest_results_path)
stats_df = pd.read_csv(model_stats_path)

# region Forest plot
# ------------------------------------------------------------------------------
# Create forest plot with hazard ratios across datasets

fig, ax = plt.subplots(figsize=(10, 6))

variables = forest_df["Variable"].unique()
datasets_list = sorted(forest_df["Dataset"].unique(), key=str.lower)

# Color scheme for variables
colors = {"Lactate": "#e74c3c", "SOFA Score": "#3498db"}

y_pos = 0
y_ticks = []
y_labels = []
dataset_positions = {}

for i, dataset in enumerate(sorted(datasets_list, reverse=True, key=str.lower)):
    dataset_positions[dataset] = []
    start_y = y_pos
    for var in ["Lactate", "SOFA Score"]:
        data_point = forest_df[
            (forest_df["Dataset"] == dataset) & (forest_df["Variable"] == var)
        ]

        if data_point.empty:
            continue

        row = data_point.iloc[0]
        ax.plot(
            [row["CI_lower"], row["CI_upper"]],
            [y_pos, y_pos],
            color=colors[var],
            linewidth=2,
        )
        ax.scatter(
            row["HR"],
            y_pos,
            s=100,
            color=colors[var],
            zorder=3,
            edgecolors="black",
            linewidth=1,
        )
        dataset_positions[dataset].append(y_pos)

        y_pos += 1

    end_y = y_pos - 1
    # Add alternating shading
    if i % 2 == 1:
        ax.axhspan(start_y - 0.75, end_y + 0.75, color="lightgray", alpha=0.3)

    y_pos += 1  # Add space between datasets

# Add horizontal line at HR=1
ax.axvline(x=1, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xlim(0.95, 1.25)

ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=12, fontweight="bold")
ax.set_ylabel("Database", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")

# Create custom y-axis labels (one per dataset)
y_ticks_final = []
y_labels_final = []

y_pos = 0
for i, dataset in enumerate(sorted(datasets_list, reverse=True, key=str.lower)):
    start_y = y_pos
    y_pos += 2  # Since two variables
    end_y = y_pos - 1
    center_y = (start_y + end_y) / 2
    y_ticks_final.append(center_y)
    y_labels_final.append(dataset)
    y_pos += 1  # Space

ax.set_yticks(y_ticks_final)
ax.set_yticklabels(y_labels_final, fontsize=10)

# Add legend
legend_elements = [
    Patch(facecolor=colors["Lactate"], edgecolor="black", label="Lactate"),
    Patch(facecolor=colors["SOFA Score"], edgecolor="black", label="SOFA Score"),
] # fmt: skip
ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

plt.tight_layout()
plt.savefig(f"{plots_dir}/forest_plot.eps", dpi=300, bbox_inches="tight")
plt.savefig(f"{plots_dir}/forest_plot.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"\nForest plot saved to: {plots_dir}/forest_plot.*")

# region Markdown table
# ------------------------------------------------------------------------------
# Create markdown table with hazard ratios and 95% CI for each dataset

# Pivot forest_df to get one row per dataset
md_rows = []
datasets = sorted(forest_df["Dataset"].unique(), key=str.lower)
for dataset in datasets:
    row = {"Dataset": dataset}

    for var in ["Lactate", "SOFA Score"]:
        data_point = forest_df[
            (forest_df["Dataset"] == dataset) & (forest_df["Variable"] == var)
        ]

        if not data_point.empty:
            hr = data_point.iloc[0]["HR"]
            ci_lower = data_point.iloc[0]["CI_lower"]
            ci_upper = data_point.iloc[0]["CI_upper"]

            # Format as "HR (CI_lower-CI_upper)"
            hr_ci_str = f"{hr:.2f} ({ci_lower:.2f}-{ci_upper:.2f})"
            row[f"{var} HR (95% CI)"] = hr_ci_str

    md_rows.append(row)

# Create markdown table
md_df = pd.DataFrame(md_rows)
markdown_table = md_df.to_markdown(index=False)

# Save to file
with open(f"{output_dir}/hazard_ratios_table.md", "w") as f:
    f.write("# Hazard Ratios by Dataset\n\n")
    f.write(markdown_table)
    f.write("\n")

print(f"Markdown table saved to: {output_dir}/hazard_ratios_table.md")
print("\n" + markdown_table)

# region Summary
# ------------------------------------------------------------------------------
# Save model summary stats to a text file
with open(f"{output_dir}/model_summaries.txt", "w") as f:
    for _, row in stats_df.iterrows():
        dataset = row["Dataset"]
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {dataset}\n")
        f.write("=" * 80 + "\n")
        f.write(f"N subjects: {row['n_subjects']}\n")
        f.write(f"N events: {row['n_events']}\n")
        f.write(f"N data points: {row['n_datapoints']}\n")
        f.write("\n")

print(f"Model summaries saved to: {output_dir}/model_summaries.txt")
