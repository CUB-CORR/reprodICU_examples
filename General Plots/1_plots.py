# Plotting different relevant variables
################################################################################

import os
import time

import matplotlib.pyplot as plt
import polars as pl
import reprodICU
import seaborn as sns
from reprodICU.utils.laboratory.oxygenation import PaO2_FiO2_RATIO
from reprodICU.utils.scores import OASIS, SOFA, SOFA2, VIS

# region constants
# ==============================================================================
STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"
output_dir = "plots"
data_dir = "data"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

SECONDS_IN_24H = 24 * 60 * 60

# ------------------------------------------------------------------------------
# X. lazy-load the datasets
info: pl.LazyFrame = reprodICU.patient_information
vent: pl.LazyFrame = reprodICU.VENTILATION_DURATION
SOURCE_DATABASE = info.select(STAY_KEY, "Source Dataset")

################################################################################
################################################################################

# region scores
# ==============================================================================
print("\n=== COMPUTING SCORES (first 24h) ===")


# Helper function to load or compute score
def load_or_compute_score(name, score, score_col, alias):
    path = os.path.join(data_dir, f"{alias.lower()}_day0.parquet")
    if not os.path.exists(path):
        start = time.time()
        (
            score()
            .filter(pl.col("Days Relative to Admission") == 1)
            .sink_parquet(path)
        )
        print(f"{name} computed in {time.time() - start:.2f} seconds.")
    return pl.read_parquet(path).select(STAY_KEY, pl.col(score_col).alias(alias)) # fmt: skip


# fmt: off
sofa_scores  = load_or_compute_score("SOFA",   SOFA,  "SOFA Score",   "SOFA")
sofa2_scores = load_or_compute_score("SOFA-2", SOFA2, "SOFA-2 Score", "SOFA2")
oasis_scores = load_or_compute_score("OASIS",  OASIS, "OASIS Score",  "OASIS")
vis_scores   = load_or_compute_score("VIS",    VIS,   "Vasoactive-Inotropic Score (VIS)", "VIS")

# Join all scores along with Source Dataset
scores_df = (
    SOURCE_DATABASE.collect()
    .join(sofa_scores,  on=STAY_KEY, how="inner")
    .join(sofa2_scores, on=STAY_KEY, how="left")
    .join(oasis_scores, on=STAY_KEY, how="left")
    .join(vis_scores,   on=STAY_KEY, how="left")
    .sort(pl.col("Source Dataset").str.to_lowercase())
    .to_pandas()
)
# fmt: on


# region respiratory
# ==============================================================================
_VENT_START = "Ventilation Start Relative to Admission (seconds)"
_VENT_END   = "Ventilation End Relative to Admission (seconds)" # fmt: skip
_VENT = vent.filter(
    pl.col("Ventilation Type").is_in(["invasive ventilation", "tracheostomy"])
    | pl.col("Ventilation Type").is_null()
)

# Compute minimum PaO2/FiO2 ratio within the first 24h and label IMV vs no IMV
PF = (
    PaO2_FiO2_RATIO()
    .filter(pl.col(TIME_KEY) <= SECONDS_IN_24H)
    .join(_VENT, on=STAY_KEY, how="left")
    .with_columns(
        pl.when(pl.col(TIME_KEY).is_between(_VENT_START, _VENT_END))
        .then(pl.lit("IMV"))
        .otherwise(pl.lit("No IMV"))
        .alias("Ventilation")
    )
    .select(STAY_KEY, "Ventilation", "PaO2/FiO2 Ratio")
    .group_by(STAY_KEY, "Ventilation")
    .agg(pl.median("PaO2/FiO2 Ratio"))
    .collect()
)

pf_ratio_df = (
    SOURCE_DATABASE.collect()
    .join(PF, on=STAY_KEY, how="left")
    .sort(pl.col("Source Dataset").str.to_lowercase())
    .to_pandas()
)

################################################################################
################################################################################

# region plotting
# ==============================================================================
print("\n=== GENERATING PLOTS ===")
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

print("Plotting ridgeline for PaO2/FiO2 Ratio...")

plot_data = pf_ratio_df[["PaO2/FiO2 Ratio", "Ventilation", "Source Dataset"]].dropna() # fmt: skip
plot_data = plot_data[plot_data["PaO2/FiO2 Ratio"] <= 1000]

# Force a stable hue order and matching colors
hue_order = ["No IMV", "IMV"]  # change to match your exact labels
palette = {"No IMV": "#4C72B0", "IMV": "#DD8452"}

g = sns.FacetGrid(
    plot_data,
    row="Source Dataset",
    hue="Ventilation",
    hue_order=hue_order,
    palette=palette,
    aspect=7,
    height=1,
)

# Clip KDE support at x >= 0 so it does not bleed left of the axis
g.map(
    sns.kdeplot,
    "PaO2/FiO2 Ratio",
    bw_adjust=2,
    clip=(0, None),
    fill=True,
    alpha=0.5,
    linewidth=1.5,
)
g.refline(y=0, linewidth=1.5, linestyle="-", color="k", clip_on=False)
g.refline(x=0, linewidth=1.5, linestyle="-", color="k", clip_on=False)


def _label_row(x, color, label):
    ax = plt.gca()
    row_idx = list(g.axes.flatten()).index(ax)
    dataset_name = sorted(plot_data["Source Dataset"].unique())[row_idx]
    ax.text(
        -0.05,
        0.2,
        dataset_name,
        fontweight="bold",
        color="black",
        ha="right",
        va="center",
        transform=ax.transAxes,
    )


g.map(_label_row, "PaO2/FiO2 Ratio")
g.set_titles("")
g.set(yticks=[], ylabel="", xlim=(0, 1000))
g.despine(bottom=True, left=True)
g.figure.subplots_adjust(hspace=-0.25)

g.add_legend(title="Ventilation Type")
plt.savefig(
    os.path.join(output_dir, "ridgeline_PF_ratio.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Ridgeline plots for each score, faceted by dataset
# ------------------------------------------------------------------------------
# Create a ridgeline plot for each score
scores_to_plot = ["SOFA", "SOFA2", "OASIS", "VIS"]

# Simple FacetGrid-based ridgeline with slight vertical overlap
for score in scores_to_plot:
    print(f"Plotting ridgeline for {score}...")
    plot_data = scores_df[[score, "Source Dataset"]].dropna()
    
    # Filter out NWICU
    plot_data = plot_data[plot_data["Source Dataset"] != "NWICU"]

    if plot_data.empty:
        print(f"No data to plot for {score}")
        continue

    # For VIS, clip extreme outliers at the 99th percentile
    if score == "VIS":
        cutoff = plot_data[score].quantile(0.95)
        plot_data = plot_data[plot_data[score] <= cutoff]
        
    min_score = plot_data[score].min()

    g = sns.FacetGrid(
        plot_data,
        row="Source Dataset",
        hue="Source Dataset",
        aspect=7,
        height=1,
        palette="tab10",
    )

    # Plot the KDE for each dataset with some overlap
    g.map(
        sns.kdeplot,
        score,
        bw_adjust=2,
        clip=(0, None),
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1.5,
    )
    g.map(
        sns.kdeplot,
        score,
        bw_adjust=2,
        clip=(0, None),
        clip_on=False,
        color="w",
        linewidth=2,
    )
    g.refline(y=0, linewidth=1.5, linestyle="-", color=None, clip_on=False)
    g.refline(x=0, linewidth=1.5, linestyle="-", color="k")

    def _label_row(x, color, label):
        ax = plt.gca()
        ax.text(
            -0.02,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="right",
            va="center",
            transform=ax.transAxes,
        )

    g.map(_label_row, score)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    g.figure.subplots_adjust(hspace=-0.25)
    # plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"ridgeline_{score}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# Correlation subplot grid for SOFA vs SOFA2 (one panel per database, 2x3)
# ------------------------------------------------------------------------------
sns.set_theme(style="white")
print("Plotting SOFA vs SOFA-2 correlation (per-database subplots)...")

sofa_sofa2_data = scores_df[["SOFA", "SOFA2", "Source Dataset"]].dropna()
sofa_sofa2_data = sofa_sofa2_data[sofa_sofa2_data["Source Dataset"] != "NWICU"]

# Fixed tab10 palette — matches Mortality Prediction/_constants.py DB_PALETTE
DB_PALETTE = {
    "AmsterdamUMCdb": "#1f77b4",
    "eICU-CRD":       "#ff7f0e",
    "HiRID":          "#2ca02c",
    "MIMIC-III":      "#d62728",
    "MIMIC-IV":       "#9467bd",
    "NWICU":          "#8c564b",
    "SICdb":          "#e377c2",
}

plot_databases = sorted(sofa_sofa2_data["Source Dataset"].unique(), key=str.lower)

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
axes_flat = axes.flatten()

for ax_idx, db in enumerate(plot_databases):
    ax      = axes_flat[ax_idx]
    color   = DB_PALETTE[db]
    db_data = sofa_sofa2_data[sofa_sofa2_data["Source Dataset"] == db]

    # 2-D KDE contour lines
    sns.kdeplot(
        data=db_data, x="SOFA", y="SOFA2",
        color=color, fill=True,
        levels=5, bw_adjust=2.0,
        linewidths=1.2, alpha=0.5,
        ax=ax, 
    )
    
    # Scatter points (rasterised for large datasets)
    ax.scatter(
        db_data["SOFA"], db_data["SOFA2"],
        color=color, alpha=0.12, s=6, edgecolors="none", rasterized=True,
    )

    # Identity reference line
    upper = max(db_data["SOFA"].max(), db_data["SOFA2"].max()) + 0.5
    ax.plot([0, upper], [0, upper], linestyle="--", color="0.65", linewidth=0.8, zorder=0)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)

    ax.set_title(db, fontweight="bold", color=color, fontsize=11)
    ax.set_xlabel("SOFA", fontsize=10)
    ax.set_ylabel("SOFA-2", fontsize=10)
    ax.tick_params(labelsize=9)

# Hide unused axes (if fewer than 6 databases)
for ax_idx in range(len(plot_databases), len(axes_flat)):
    axes_flat[ax_idx].set_visible(False)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "correlation_SOFA_vs_SOFA2.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print(f"saved to {output_dir}/")
