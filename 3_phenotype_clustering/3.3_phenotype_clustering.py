# Case study III: Phenotype clustering of ICU patients with sepsis based on 
# temperature trajectories
################################################################################

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import reprodICU
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import apply_plot_style

apply_plot_style()

# region constants
# ------------------------------------------------------------------------------
HOURS_IN_DAY = 24

# Create output directory
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# region loading
# ------------------------------------------------------------------------------
temp = pl.scan_parquet("data/TEMPERATURE.parquet")

info: pl.LazyFrame = reprodICU.patient_information
sources = info.select("Global ICU Stay ID", "Source Dataset")


# region de-standardization
# ------------------------------------------------------------------------------
# Compute standardization parameters from the data
temp_col = "Temperature"
temp_mean = temp.collect().select(pl.col(temp_col).mean()).item()
temp_std = temp.collect().select(pl.col(temp_col).std()).item()


def destandardize(standardized_values, mean=temp_mean, std=temp_std):
    """Convert from standardized space to original temperature space"""
    return standardized_values * std + mean


# region trajectories
# ------------------------------------------------------------------------------
# Define trajectory coefficients in standardized space (a, b, c) for: a + b*t + c*t^2

MODELS = {
    # BHAVANI TEMPERATURE TRAJECTORIES
    # traj1 = -0.89548 - 0.00298 * adm + 0.00010 * (adm^2)
    # traj2 = -0.00667 + 0.00050 * adm - 0.00001 * (adm^2)
    # traj3 =  1.35157 - 0.06946 * adm + 0.00065 * (adm^2)
    # traj4 =  1.22203 - 0.00590 * adm - 0.00007 * (adm^2)
    "Bhavani": {
        "labels": ["LT", "NT", "HTFR", "HTSR"],
        "coefficients": {
            "LT": (-0.89548, -0.00298, 0.00010),
            "NT": (-0.00667, 0.00050, -0.00001),
            "HTFR": (1.35157, -0.06946, 0.00065),
            "HTSR": (1.22203, -0.00590, -0.00007),
        },
        "palette": {
            "LT": "#0000FF",
            "NT": "#990000",
            "HTFR": "#336600",
            "HTSR": "#FF9900",
        },
    },
    # REPRODICU TEMPERATURE TRAJECTORIES
    #                                     Standard       T for H0:
    # Group   Parameter        Estimate        Error     Parameter=0   Prob > |T|
    # 1       Intercept        -1.33798      0.02785         -48.034       0.0000
    #         Linear            0.04170      0.00148          28.210       0.0000
    #         Quadratic        -0.00042      0.00002         -21.909       0.0000
    # 2       Intercept        -0.26467      0.02073         -12.768       0.0000
    #         Linear            0.01975      0.00106          18.644       0.0000
    #         Quadratic        -0.00023      0.00001         -16.814       0.0000
    # 3       Intercept         0.53914      0.02761          19.527       0.0000
    #         Linear            0.01440      0.00159           9.067       0.0000
    #         Quadratic        -0.00018      0.00002          -8.880       0.0000
    # 4       Intercept        -3.49501      0.01491        -234.482       0.0000
    #         Linear            0.03405      0.00098          34.734       0.0000
    #         Quadratic         0.00048      0.00001          35.032       0.0000
    #         Sigma             0.80941      0.00191         424.093       0.0000
    # Group membership
    # 1       (%)              20.66031      1.05450          19.592       0.0000
    # 2       (%)              40.81293      1.15969          35.193       0.0000
    # 3       (%)              15.80097      0.95741          16.504       0.0000
    # 4       (%)              22.72579      0.86904          26.150       0.0000
    "AmsterdamUMCdb": {
        "labels": ["LT", "NT", "HT", "X"],
        "coefficients": {
            "LT": (-1.33798, 0.04170, -0.00042),  # 20.7 %
            "NT": (-0.26467, 0.01975, -0.00023),  # 40.8 %
            "HT": (0.53914, 0.01440, -0.00018),  # 15.8 %
            "X": (-3.49501, 0.03405, 0.00048),  # 22.7 %
        },
        "palette": {
            "LT": "#0000FF",
            "NT": "#990000",
            "HT": "#FF9900",
            "X": "#336600",
        },
    },
    #                                       Standard       T for H0:
    # Group   Parameter        Estimate        Error     Parameter=0   Prob > |T|
    # 1       Intercept        -0.33756      0.02928         -11.529       0.0000
    #         Linear            0.00917      0.00172           5.344       0.0000
    #         Quadratic        -0.00011      0.00002          -4.889       0.0000
    # 2       Intercept         0.19083      0.02754           6.928       0.0000
    #         Linear            0.01576      0.00183           8.609       0.0000
    #         Quadratic        -0.00020      0.00002          -8.773       0.0000
    # 3       Intercept         0.58388      0.03877          15.058       0.0000
    #         Linear            0.03429      0.00199          17.201       0.0000
    #         Quadratic        -0.00043      0.00003         -16.912       0.0000
    # 4       Intercept        -2.28424      0.07023         -32.525       0.0000
    #         Linear            0.02681      0.00452           5.930       0.0000
    #         Quadratic         0.00022      0.00006           3.505       0.0005
    #         Sigma             0.62466      0.00283         221.035       0.0000
    # Group membership
    # 1       (%)              34.22775      1.85307          18.471       0.0000
    # 2       (%)              41.99438      1.82247          23.043       0.0000
    # 3       (%)              21.18534      1.56170          13.566       0.0000
    # 4       (%)               2.59253      0.52504           4.938       0.0000
    "MIMIC-III": {
        "labels": ["LT", "NT", "HT", "X"],
        "coefficients": {
            "LT": (-0.33756, 0.00917, -0.00011),  # 34.2 %
            "NT": (0.19083, 0.01576, -0.00020),  # 42.0 %
            "HT": (0.58388, 0.03429, -0.00043),  # 21.2 %
            "X": (-2.28424, 0.02681, 0.00022),  #  2.6 %
        },
        "palette": {
            "LT": "#0000FF",
            "NT": "#990000",
            "HT": "#FF9900",
            "X": "#336600",
        },
    },
    #                                        Standard       T for H0:
    #  Group   Parameter        Estimate        Error     Parameter=0   Prob > |T|
    #  1       Intercept        -0.20866      0.01113         -18.749       0.0000
    #          Linear            0.01280      0.00064          19.968       0.0000
    #          Quadratic        -0.00016      0.00001         -18.218       0.0000
    #  2       Intercept         0.20886      0.00949          22.016       0.0000
    #          Linear            0.01847      0.00061          30.255       0.0000
    #          Quadratic        -0.00023      0.00001         -28.203       0.0000
    #  3       Intercept         0.74592      0.01475          50.558       0.0000
    #          Linear            0.02133      0.00086          24.682       0.0000
    #          Quadratic        -0.00028      0.00001         -24.593       0.0000
    #  4       Intercept        -1.86033      0.03291         -56.527       0.0000
    #          Linear            0.02736      0.00204          13.395       0.0000
    #          Quadratic         0.00008      0.00003           2.740       0.0062
    #          Sigma             0.56127      0.00118         475.083       0.0000
    #   Group membership
    #  1       (%)              34.78441      0.94007          37.002       0.0000
    #  2       (%)              45.78634      0.93428          49.007       0.0000
    #  3       (%)              16.66957      0.66868          24.929       0.0000
    #  4       (%)               2.75967      0.26926          10.249       0.0000
    "MIMIC-IV": {
        "labels": ["LT", "NT", "HT", "X"],
        "coefficients": {
            "LT": (-0.20866, 0.01280, -0.00016),  # 34.8 %
            "NT": (0.20886, 0.01847, -0.00023),  # 45.8 %
            "HT": (0.74592, 0.02133, -0.00028),  # 16.7 %
            "X": (-1.86033, 0.02736, 0.00008),  #  2.8 %
        },
        "palette": {
            "LT": "#0000FF",
            "NT": "#990000",
            "HT": "#FF9900",
            "X": "#336600",
        },
    },
}


# region subphenotypes
# ------------------------------------------------------------------------------
def assign_subphenotypes(traj_df, labels, trajectory_coefficients):
    """Assign subphenotypes based on minimum SSE for each trajectory."""
    traj_cols = [f"traj{i+1}" for i in range(len(labels))]

    # Create trajectory columns for this model
    adm = pl.col("Hours Relative to T_0")
    for i, label in enumerate(labels):
        a, b, c = trajectory_coefficients[label]
        traj_df = traj_df.with_columns(
            (a + b * adm + c * adm.pow(2)).alias(traj_cols[i])
        )

    # Assign subphenotypes based on minimum SSE
    return (
        traj_df.group_by("Global ICU Stay ID")
        .agg(
            pl.concat_list(
                (pl.col("Temperature (standardized)") - pl.col(col))
                .pow(2)
                .sum()
                for col in traj_cols
            ).alias("sse_list")
        )
        .with_columns(
            pl.col("sse_list").list.arg_min().alias("subphenotype_idx")
        )
        .with_columns(
            pl.col("subphenotype_idx")
            .map_elements(lambda i: labels[i], return_dtype=str)
            .alias("subphenotype")
        )
    )


# region plotting
# ------------------------------------------------------------------------------
def plot_trajectories_on_ax(ax, model_name, model_config, letter):
    """Plot trajectories for a single model on a given axis."""
    labels = model_config["labels"]
    coefficients = model_config["coefficients"]
    palette = model_config["palette"]

    # Assign subphenotypes
    subphenotypes = assign_subphenotypes(temp, labels, coefficients)

    # Prepare plotting data
    plotting_df = (
        subphenotypes.join(temp, on="Global ICU Stay ID", how="left")
        .join(sources, on="Global ICU Stay ID", how="left")
        .select(
            "Global ICU Stay ID",
            "Hours Relative to T_0",
            "Temperature",
            "subphenotype",
            "Source Dataset",
        )
        .collect()
        .to_pandas()
    )

    # Plot data trajectories (semi-transparent for background)
    sns.lineplot(
        data=plotting_df,
        x="Hours Relative to T_0",
        y="Temperature",
        hue="subphenotype",
        style="Source Dataset",
        palette=palette,
        ax=ax,
        lw=1.8,
        alpha=0.2,
        ci=None,
        legend=False,
    )

    # Plot de-standardized reference curves (opaque and prominent)
    time_range = np.arange(0, 3 * HOURS_IN_DAY, 0.5)
    for label in labels:
        a, b, c = coefficients[label]
        # Calculate in standardized space
        traj_std = a + b * time_range + c * (time_range**2)
        # De-standardize to original temperature space
        traj_original = destandardize(traj_std)

        ax.plot(
            time_range,
            traj_original,
            color=palette[label],
            lw=3.0,
            alpha=1.0,
            label=label,
            linestyle="-",
            zorder=10,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # Enhanced plot styling
    ax.set_title(
        f"{letter}. Temperature Trajectories - {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Hours Relative to ICU Admission", fontsize=11, fontweight="bold") # fmt: skip
    ax.set_ylabel("Temperature (°C)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 3 * HOURS_IN_DAY)
    ax.set_ylim(34.5, 39.5)

    # Add grid for enhanced readability
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Professional legend formatting
    ax.legend(
        loc="best",
        fontsize=8,
        ncol=2,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
        title="Phenotype",
        title_fontsize=10,
    )

    # Improve tick formatting
    ax.tick_params(axis="both", which="major", labelsize=10)


def create_trajectory_plot(model_name, model_config, letter):
    """Create individual trajectory plot for a model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_trajectories_on_ax(ax, model_name, model_config, letter)

    # Save figure with high quality
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.tight_layout()
    plt.savefig(
        f"plots/{safe_name}_trajectories.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        f"plots/{safe_name}_trajectories.eps",
        bbox_inches="tight",
    )
    plt.close()


def create_combined_trajectory_plot():
    """Create a 2x2 grid of trajectory plots for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (model_name, model_config) in enumerate(MODELS.items()):
        letter = chr(ord("a") + i)
        print(f"Plotting {model_name} onto grid...")
        plot_trajectories_on_ax(axes[i], model_name, model_config, letter)

    plt.tight_layout()
    plt.savefig(
        "plots/Combined_trajectories.eps",
        bbox_inches="tight",
        format="eps",
    )
    plt.savefig(
        "plots/Combined_trajectories.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Combined plot saved to plots/combined_trajectories.eps")


def export_assignment_counts(model_name, model_config):
    """Export absolute and relative assignment counts per database per subphenotype."""
    # Calculate counts in Polars
    counts = (
        assign_subphenotypes(
            temp, model_config["labels"], model_config["coefficients"]
        )
        .join(sources, on="Global ICU Stay ID", how="left")
        .select("subphenotype", "Source Dataset")
        .group_by("Source Dataset", "subphenotype")
        .len()
        .with_columns(pl.col("len").sum().over("Source Dataset").alias("Total"))
        .with_columns(
            (pl.col("len") / pl.col("Total") * 100).round(2).alias("Percentage")
        )
        .drop("Total")
        .sort("Source Dataset", "len", descending=[False, True])
    )

    # Save outputs
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    counts.sink_csv(f"data/{safe_name}_subphenotype_assignments.csv")

    counts_pd = counts.collect().to_pandas()
    with open(f"data/{safe_name}_subphenotype_assignments.md", "w") as f:
        f.write(
            f"# Subphenotype Assignments - {model_name}\n\n{counts_pd.to_markdown(index=False)}\n"
        )

    print(f"Exported assignment counts for {model_name}")


# region main
# ------------------------------------------------------------------------------
# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Generate individual plots for all models
for i, (model_name, model_config) in enumerate(MODELS.items()):
    letter = chr(ord("a") + i)
    print(f"Generating {model_name} individual plot...")
    create_trajectory_plot(model_name, model_config, letter)

# Generate combined plot for all models
create_combined_trajectory_plot()

# Export assignment counts for all models
for model_name, model_config in MODELS.items():
    export_assignment_counts(model_name, model_config)

print("All plots generated and assignment counts exported successfully!")
