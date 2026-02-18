# Case study I: Development and external validation of an in-ICU mortality
# model for ICU patients
################################################################################

import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import shap
from scipy.interpolate import make_interp_spline

# Add common style
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import apply_plot_style

apply_plot_style()

# region constants
# ------------------------------------------------------------------------------
output_dir = "data"
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# check if output directory exists
if not os.path.exists(output_dir):
    raise FileNotFoundError(
        f"Output directory '{output_dir}' not found. "
        "Please run the data processing and model training script first."
    )

# region load data
# ------------------------------------------------------------------------------
print("Loading data for plotting...")

# Load evaluation data
roc_df = pl.read_parquet(f"{output_dir}/icu_mortality_roc_data.parquet")
cal_df = pl.read_parquet(f"{output_dir}/icu_mortality_cal_data.parquet")
dca_df = pl.read_parquet(f"{output_dir}/icu_mortality_dca_data.parquet")
metrics_df = pl.read_csv(f"{output_dir}/icu_mortality_db_metrics.csv")

prevalence_df = pl.read_csv(f"{output_dir}/icu_mortality_prevalence.csv")
prevalence_mimic_iii_train = prevalence_df["prevalence"][0]

# Professional color palette
db_colors = {
    "AmsterdamUMCdb": "#1f77b4",
    "eICU-CRD": "#ff7f0e",
    "HiRID": "#d62728",
    "MIMIC-III": "#2ca02c",
    "MIMIC-IV": "#9467bd",
    "NWICU": "#8c564b",
    "SICdb": "#7f7f7f",
}
sorted_dbs = sorted(db_colors.keys(), key=str.lower)

# region combined plot
# ------------------------------------------------------------------------------
print("Generating combined evaluation plot...")
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
ax_roc = fig.add_subplot(gs[0, 0])
ax_cal = fig.add_subplot(gs[0, 1])
ax_dca = fig.add_subplot(gs[1, :])
axs = [ax_roc, ax_cal, ax_dca]

# Global legend
global_legend_handles = []
global_legend_labels = []
for db in sorted_dbs:
    color = db_colors.get(db, "#808080")
    global_legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=3))
    global_legend_labels.append(db)

fig.legend(
    global_legend_handles,
    global_legend_labels,
    loc="lower center",
    ncol=len(sorted_dbs),
    frameon=True,
    framealpha=0.95,
    fontsize=10,
    handlelength=2,
    bbox_to_anchor=(0.5, 0.02),
)

# region ROC Plot
# ------------------------------------------------------------------------------
axs[0].set_title("a. ROC Curve", fontweight="bold", pad=15)
axs[0].plot(
    [0, 1],
    [0, 1],
    "k--",
    linewidth=1.5,
    alpha=0.6,
    label="No Discrimination",
)
axs[0].legend(["No Discrimination"], loc="lower right", framealpha=0.95)

for db in sorted_dbs:
    db_data = roc_df.filter(pl.col("db") == db)
    if db_data.height == 0:
        continue
    auc_stats = metrics_df.filter(pl.col("DBName") == db)
    auc_val = auc_stats["ROC AUC"][0]
    auc_ci_lower = auc_stats["AUC CI Lower"][0]
    auc_ci_upper = auc_stats["AUC CI Upper"][0]
    label = f"{db} (AUC: {auc_val:.3f} [{auc_ci_lower:.3f}–{auc_ci_upper:.3f}])"
    color = db_colors.get(db, "#808080")
    axs[0].plot(
        db_data["fpr"],
        db_data["tpr"],
        label=label,
        color=color,
        linewidth=2.0,
    )

axs[0].set_xlim([0.0, 1.0])
axs[0].set_ylim([0.0, 1.0])
axs[0].set_xlabel("False Positive Rate", fontweight="bold")
axs[0].set_ylabel("True Positive Rate", fontweight="bold")
axs[0].set_aspect("equal")

# region Calibration Plot
# ------------------------------------------------------------------------------
axs[1].set_title("b. Calibration Plot", fontweight="bold", pad=15)
axs[1].plot(
    [0, 1],
    [0, 1],
    "k--",
    linewidth=1.5,
    alpha=0.6,
    label="Perfect Calibration",
)
axs[1].legend(["Perfect Calibration"], loc="upper left", framealpha=0.95)

for db in sorted_dbs:
    db_data = cal_df.filter(pl.col("db") == db)
    if db_data.height == 0:
        continue
    x_new = np.linspace(
        db_data["predicted_prob"].min(),
        db_data["predicted_prob"].max(),
        300,
    )
    spl = make_interp_spline(
        db_data["predicted_prob"].to_numpy(),
        db_data["actual_prob"].to_numpy(),
        k=2,
    )
    y_smooth = spl(x_new)
    color = db_colors.get(db, "#808080")
    axs[1].plot(x_new, y_smooth, color=color, linewidth=2.0)

axs[1].set_xlim([0.0, 1.0])
axs[1].set_ylim([0.0, 1.0])
axs[1].set_xlabel("Predicted Probability", fontweight="bold")
axs[1].set_ylabel("Actual Probability", fontweight="bold")
axs[1].set_aspect("equal")

# region DCA Plot
# ------------------------------------------------------------------------------
axs[2].set_title("c. Decision Curve Analysis", fontweight="bold", pad=15)
axs[2].axhline(
    y=0,
    color="black",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="Treat None",
)

thresholds_dca_plot = np.linspace(0, 1, 101)
net_benefit_treat_all_plot = [
    prevalence_mimic_iii_train
    - (1 - prevalence_mimic_iii_train) * (thr / (1 - thr + 1e-9))
    for thr in thresholds_dca_plot
]
axs[2].plot(
    thresholds_dca_plot,
    net_benefit_treat_all_plot,
    color="black",
    linestyle=":",
    linewidth=2.0,
    label="Treat All",
)

prevalence_threshold_label_text = (
    f"Prevalence Threshold in MIMIC-IV ({prevalence_mimic_iii_train:.3f})"
)
axs[2].axvline(
    x=prevalence_mimic_iii_train,
    color="gray",
    linestyle="-.",
    linewidth=1.5,
    alpha=0.7,
    label=prevalence_threshold_label_text,
)

sns.lineplot(
    ax=axs[2],
    data=dca_df.to_pandas(),
    x="threshold",
    y="net_benefit",
    hue="db_name",
    hue_order=sorted_dbs,
    palette=db_colors,
    markers=False,
    dashes=False,
    linewidth=2.0,
    legend=False,
)

axs[2].legend(
    ["Treat None", "Treat All", prevalence_threshold_label_text],
    loc="upper right",
    framealpha=0.95,
)
axs[2].set_xlim([0.0, 1.0])
axs[2].set_ylim(
    [
        min(dca_df["net_benefit"].min(), -0.05),
        max(dca_df["net_benefit"].max(), 0.1),
    ]
)
axs[2].set_xlabel("Threshold Probability", fontweight="bold")
axs[2].set_ylabel("Net Benefit", fontweight="bold")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plot_filename = f"{plots_dir}/icu_mortality_evaluation"
fig.savefig(plot_filename + ".png", dpi=300, bbox_inches="tight")
fig.savefig(plot_filename + ".eps", dpi=300, bbox_inches="tight")
plt.close()

# region SHAP and Feature Importance
# ------------------------------------------------------------------------------
print("Generating SHAP and feature importance plots...")

# Load SHAP values
with open(f"{output_dir}/shap_values.pkl", "rb") as f:
    shap_data = pickle.load(f)

with plt.rc_context(
    {"font.family": "serif", "font.serif": ["Times New Roman"]}
):
    shap.summary_plot(
        shap_data["shap_values"],
        shap_data["X_train"],
        feature_names=shap_data["features"],
        show=False,
    )
    plt.savefig(
        f"{plots_dir}/shap_summary_plot.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

# Load Model for importance plots
with open(f"{output_dir}/lgbm_model.pkl", "rb") as f:
    model = pickle.load(f)

for imp_type in ["split", "gain"]:
    fig, ax = plt.subplots(figsize=(10, 6))
    lgb.plot_importance(
        model, max_num_features=20, importance_type=imp_type, ax=ax
    )
    ax.set_title(
        f"Feature Importance: {imp_type.capitalize()}",
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(
        f"{plots_dir}/feature_importance_{imp_type}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

print("All plots generated successfully.")
