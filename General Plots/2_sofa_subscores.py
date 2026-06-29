import os
import time

import matplotlib.pyplot as plt
import polars as pl
import reprodICU
import seaborn as sns
from reprodICU.utils.scores import SOFA

# region constants
# ==============================================================================
STAY_KEY = "Global ICU Stay ID"
SOURCE_KEY = "Source Dataset"
DAY_KEY = "Days Relative to Admission"

output_dir = "plots"
data_dir = "data"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

SUBSCORE_COLS_DICT = {
    "Respiration": "RESP",
    "Coagulation": "COAG",
    "Liver": "LIVER",
    "Cardiovascular (MAP)": "CV (MAP)",
    "Cardiovascular (VPs)": "CV (VP)",
    "Cardiovascular": "CV",
    "Central nervous system": "CNS",
    "Renal (creatinine)": "REN (SCr)",
    "Renal (urine output)": "REN (UO)",
    "Renal": "REN",
}
SUBSCORE_COLS = list(SUBSCORE_COLS_DICT.keys())
SUBSCORE_LABELS = [SUBSCORE_COLS_DICT[col] for col in SUBSCORE_COLS]
POINT_BINS = [0, 1, 2, 3, 4]

# ------------------------------------------------------------------------------
# X. lazy-load the datasets
info: pl.LazyFrame = reprodICU.patient_information
SOURCE_DATABASE = info.select(STAY_KEY, "Source Dataset")


# ------------------------------------------------------------------------------
def load_or_compute_sofa_subscores_day0() -> pl.DataFrame:
    """Load cached first 24h SOFA subscores or compute and cache them."""
    path = os.path.join(data_dir, "sofa_subscores_day0.parquet")
    if not os.path.exists(path):
        start = time.time()
        (
            SOFA()
            .filter(pl.col(DAY_KEY) == 1)
            .select(STAY_KEY, *SUBSCORE_COLS)
            .join(SOURCE_DATABASE, on=STAY_KEY, how="left")
            .sink_parquet(path)
        )
        print(f"SOFA subscores computed in {time.time() - start:.2f} seconds.")
    return pl.read_parquet(path)


def _facet_barplot(data, palette_map, **kwargs):
    """Draw one discrete bar chart in a FacetGrid cell."""
    ax = plt.gca()
    if data.empty:
        return

    ax._no_data_facet = False
    row_name = data[SOURCE_KEY].iloc[0]
    subscore_name = data["Subscore"].iloc[0]
    alpha = 0.3 if "(" in subscore_name else 1.0
    sns.barplot(
        data=data,
        x="Points",
        y="Count",
        order=POINT_BINS,
        color=palette_map[row_name],
        alpha=alpha,
        edgecolor="none",
        linewidth=0,
        ax=ax,
    )


# region plotting
# ==============================================================================
print("\n=== GENERATING FACETED SOFA SUBSCORE BARS ===")

counts_df = (
    SOURCE_DATABASE.collect()
    .join(load_or_compute_sofa_subscores_day0(), on=STAY_KEY, how="inner")
    .unpivot(
        index=[SOURCE_KEY],
        on=SUBSCORE_COLS,
        variable_name="Subscore",
        value_name="Points",
    )
    .drop_nulls([SOURCE_KEY, "Points"])
    .with_columns(pl.col("Points").cast(int))
    .group_by(SOURCE_KEY, "Subscore", "Points")
    .len()
    .rename({"len": "Count"})
    .sort(pl.col(SOURCE_KEY).str.to_lowercase(), "Subscore", "Points")
)

datasets = (
    counts_df.select(SOURCE_KEY)
    .unique()
    .sort(pl.col(SOURCE_KEY).str.to_lowercase())
    .get_column(SOURCE_KEY)
    .to_list()
)
plot_df = counts_df.to_pandas()
plot_df.to_csv(os.path.join(data_dir, "sofa_subscore_counts.csv"), index=False)
plot_df["Subscore"] = plot_df["Subscore"].map(SUBSCORE_COLS_DICT)

palette = sns.color_palette("tab10", n_colors=len(datasets))
palette_map = {dataset: palette[i] for i, dataset in enumerate(datasets)}

sns.set_theme(style="white")

g = sns.FacetGrid(
    plot_df,
    row=SOURCE_KEY,
    col="Subscore",
    row_order=datasets,
    col_order=SUBSCORE_LABELS,
    sharex=True,
    sharey="row",
    height=1.2,
    aspect=0.95,
)
for row_axes in g.axes:
    for ax in row_axes:
        ax._no_data_facet = True

g.map_dataframe(_facet_barplot, palette_map=palette_map)
g.set_titles("")

for col_idx, col_name in enumerate(SUBSCORE_LABELS):
    g.axes[0, col_idx].set_title(col_name, fontweight="bold")

for row_idx, row_axes in enumerate(g.axes):
    dataset_name = datasets[row_idx]
    for col_idx, ax in enumerate(row_axes):
        if getattr(ax, "_no_data_facet", False):
            ax.text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                va="center",
                fontweight="bold",
                color="0.3",
                transform=ax.transAxes,
            )
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
        ax.set_xticks(POINT_BINS)
        ax.set_xlabel("Points")
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.grid(False, axis="x")

    row_axes[0].text(
        -0.1,
        0.5,
        dataset_name,
        # fontsize=20,
        fontweight="bold",
        color=palette_map[dataset_name],
        ha="right",
        va="center",
        transform=row_axes[0].transAxes,
    )

g.despine(left=True, bottom=False)

g.figure.subplots_adjust(top=0.9, wspace=0.2, hspace=0.35)

png_path = os.path.join(output_dir, "sofa_subscores_facetgrid.png")
plt.savefig(png_path, dpi=300, bbox_inches="tight")

pdf_path = os.path.join(output_dir, "sofa_subscores_facetgrid.pdf")
plt.savefig(pdf_path, bbox_inches="tight")

plt.close()
print("Saved SOFA subscores FacetGrid plot.")
