# Case study IV: Causal effect of mechanical ventilation in the first 24 hours
# after respiratory failure on ICU mortality
################################################################################

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import apply_plot_style

apply_plot_style()

# region setup
# ------------------------------------------------------------------------------
# constants
output_dir = "data"
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# check if output directory exists
if not os.path.exists(output_dir):
    raise FileNotFoundError(
        f"Output directory '{output_dir}' not found. "
        "Please run the data processing and model training script first."
    )

# Load data
analysis_df_complete = pd.read_csv("data/ANALYSIS_DATA.csv")
sensitivity_df = pd.read_csv("data/SENSITIVITY_ANALYSIS_RESULTS.csv")
global_estimate_df = pd.read_csv("data/GLOBAL_ESTIMATE.csv")

or_estimate = global_estimate_df["or_estimate"].iloc[0]
or_ci_lower = global_estimate_df["or_ci_lower"].iloc[0]
or_ci_upper = global_estimate_df["or_ci_upper"].iloc[0]

treatment = "Mechanical Ventilation in first 24h"
T = analysis_df_complete[treatment].values.astype(int)
weights_truncated = analysis_df_complete["IPTW_truncated"].values


# region Love Plot
# ------------------------------------------------------------------------------
def calculate_smd(data, treatment, weights=None):
    """Calculate absolute standardized mean difference for a variable."""
    if weights is None:
        weights = np.ones_like(treatment, dtype=float)

    treated = data[treatment == 1]
    control = data[treatment == 0]
    w_treated = weights[treatment == 1]
    w_control = weights[treatment == 0]

    mean_treated = np.average(treated, weights=w_treated)
    mean_control = np.average(control, weights=w_control)
    var_treated = np.average((treated - mean_treated) ** 2, weights=w_treated)
    var_control = np.average((control - mean_control) ** 2, weights=w_control)

    pooled_std = np.sqrt((var_treated + var_control) / 2)
    smd = abs((mean_treated - mean_control) / (pooled_std + 1e-10))
    return smd


print("   -> plotting love plot")
variable_info = [
    ("Age", "Admission Age (years)"),
    ("is Female", "is Female"),
    ("BMI", "BMI"),
    ("SOFA", "SOFA Score"),
    ("GCS", "Worst GCS prior to T_0"),
    ("P/F ratio", "Worst PaO2/FiO2 Ratio prior to T_0"),
    ("Lactate", "Worst Lactate prior to T_0"),
    ("PaO2", "Worst PaO2 prior to T_0"),
    ("PaCO2", "Worst PaCO2 prior to T_0"),
]

smd_results = []
for name, col in variable_info:
    smd_b = calculate_smd(analysis_df_complete[col].values, T)
    smd_a = calculate_smd(
        analysis_df_complete[col].values, T, weights_truncated
    )
    smd_results.append({"name": name, "before": smd_b, "after": smd_a})

smd_results.sort(key=lambda x: x["before"])
variable_names = [r["name"] for r in smd_results]
smd_before = [r["before"] for r in smd_results]
smd_after = [r["after"] for r in smd_results]

fig, ax = plt.subplots(figsize=(8, 6))
y_pos = np.arange(len(variable_names))
ax.plot(smd_before, y_pos, color="red", marker="o", linestyle="-", zorder=1, label="Before weighting") # fmt: skip
ax.plot(smd_after, y_pos, color="green", marker="o", linestyle="-", zorder=1, label="After weighting") # fmt: skip
ax.axvline(0.0, color="k", linestyle="-")
ax.axvline(0.1, color="k", linestyle="--", label="|SMD| = 0.1")
ax.set_yticks(y_pos)
ax.set_yticklabels(variable_names)
ax.set_xlabel("Absolute Standardized Mean Difference")
ax.set_title("Love Plot: Covariate Balance Before and After IPTW")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/love_plot.png", dpi=300, bbox_inches="tight")
plt.close()


# region IPTW Diagnostics
# ------------------------------------------------------------------------------
def plot_iptw_diagnostics(axes, data, title_prefix, show_legend=True):
    """Helper to plot PS overlap and weight distribution."""
    hue_col = "MV in first 24h"

    # PS Overlap (Unweighted)
    sns.kdeplot(
        data=data,
        x="PS",
        hue=hue_col,
        fill=True,
        ax=axes[0],
        palette="magma",
        alpha=0.5,
        legend=show_legend,
    )
    axes[0].set_title(f"{title_prefix}: PS Overlap (Unweighted)")

    # PS Overlap (Weighted)
    sns.kdeplot(
        data=data,
        x="PS",
        hue=hue_col,
        weights="IPTW_truncated",
        fill=True,
        ax=axes[1],
        palette="magma",
        alpha=0.5,
        legend=False,
    )
    axes[1].set_title(f"{title_prefix}: PS Overlap (Weighted)")

    # IPTW Weight Distribution
    sns.kdeplot(
        data=data,
        x="IPTW_truncated",
        hue=hue_col,
        fill=True,
        ax=axes[2],
        palette="viridis",
        alpha=0.5,
        legend=False,
    )
    axes[2].set_title(f"{title_prefix}: IPTW Weight Distribution")

    avg_weight = data["IPTW_truncated"].mean()
    axes[2].axvline(
        avg_weight,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Avg: {avg_weight:.2f}",
    )
    if show_legend:
        axes[2].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)


print("   -> plotting iptw diagnostics")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_iptw_diagnostics(axes, analysis_df_complete, "Global", show_legend=True)
plt.tight_layout()
plt.savefig("plots/iptw_diagnostics.png", dpi=300, bbox_inches="tight")
plt.close()

# region Database-Specific Diagnostics
# ------------------------------------------------------------------------------
print("   -> plotting database-specific diagnostics")
for database in sensitivity_df["database"].unique():
    db_cohort = analysis_df_complete[
        analysis_df_complete["Source Dataset"] == database
    ]
    db_plots_dir = f"plots/{database}"
    os.makedirs(db_plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_iptw_diagnostics(axes, db_cohort, database)
    plt.tight_layout()
    plt.savefig(
        f"{db_plots_dir}/iptw_diagnostics.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

# region Combined Database Diagnostics
# ------------------------------------------------------------------------------
print("   -> plotting combined database-specific diagnostics")
databases = sensitivity_df["database"].unique()
n_dbs = len(databases)

fig, axes = plt.subplots(n_dbs, 3, figsize=(18, 5 * n_dbs))

for i, database in enumerate(databases):
    db_cohort = analysis_df_complete[
        analysis_df_complete["Source Dataset"] == database
    ]
    curr_axes = axes[i] if n_dbs > 1 else axes
    plot_iptw_diagnostics(
        curr_axes, db_cohort, database, show_legend=(i == n_dbs - 1)
    )

plt.tight_layout()
plt.savefig("plots/combined_iptw_diagnostics.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/combined_iptw_diagnostics.eps", dpi=300, bbox_inches="tight")
plt.close()

# region Forest Plot
# ------------------------------------------------------------------------------
print("   -> plotting forest plot")
fig, ax = plt.subplots(figsize=(10, 4))
databases = sensitivity_df["database"].tolist()
ors = sensitivity_df["OR"].tolist()
lower_cis = sensitivity_df["OR_95_CI_lower"].tolist()
upper_cis = sensitivity_df["OR_95_CI_upper"].tolist()

db_color, global_color = "#3498db", "#e74c3c"
y_positions = []
for i, (db, or_val, lower, upper) in enumerate(
    zip(databases, ors, lower_cis, upper_cis)
):
    ax.plot([lower, upper], [i, i], color=db_color, linewidth=2, zorder=2)
    ax.scatter(
        or_val,
        i,
        s=100,
        color=db_color,
        zorder=3,
        edgecolors="black",
        linewidth=1,
    )
    y_positions.append(i)

global_y_pos = len(databases) + 1
ax.scatter(
    or_estimate,
    global_y_pos,
    marker="D",
    s=100,
    color=global_color,
    zorder=3,
    edgecolors="black",
    linewidth=1.5,
)
ax.plot(
    [or_ci_lower, or_ci_upper],
    [global_y_pos, global_y_pos],
    color=global_color,
    linewidth=2.5,
    zorder=2,
)
ax.axvline(x=1, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.set_yticks(y_positions + [global_y_pos])
ax.set_yticklabels(databases + ["Global Estimate"], fontsize=10)
ax.set_xlabel("Odds Ratio (95% CI)", fontsize=12, fontweight="bold")
ax.set_ylabel("Database", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("plots/forest_plot_sensitivity.png", dpi=300, bbox_inches="tight")
plt.savefig("plots/forest_plot_sensitivity.eps", dpi=300, bbox_inches="tight")
plt.close()
