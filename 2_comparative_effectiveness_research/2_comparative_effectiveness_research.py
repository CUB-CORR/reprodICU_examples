# Case study II: Association between lactate and ICU mortality
################################################################################

# Laboratory Results:
#   Lactate

# Scores:
#   SOFA score

# ------------------------------------------------------------------------------

# Model: Cox proportional hazards regression
# Outcome: ICU mortality
# Covariates: Lactate, SOFA score

################################################################################

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import reprodICU
from lifelines import CoxTimeVaryingFitter
from reprodICU.utils.scores.SOFA import SOFA

sys.path.append(str(Path(__file__).resolve().parent.parent))

# region constants
# ------------------------------------------------------------------------------
SECONDS_IN_A_DAY = 86400
HOURS_IN_A_DAY = 24

# Create output directory
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# region cohort
# ------------------------------------------------------------------------------
# X. lazy-load the datasets
info: pl.LazyFrame = reprodICU.patient_information
vitals: pl.LazyFrame = reprodICU.timeseries_vitals
labs: pl.LazyFrame = reprodICU.timeseries_labs
resp: pl.LazyFrame = reprodICU.timeseries_respiratory
inout: pl.LazyFrame = reprodICU.timeseries_intakeoutput
meds: pl.LazyFrame = reprodICU.medications
vent: pl.LazyFrame = reprodICU.VENTILATION_DURATION

# Prepare lactate data (blood/serum only), aggregated per hour
lactate = (
    labs.select(
        "Global ICU Stay ID",
        "Time Relative to Admission (seconds)",
        "Lactate",
    )
    .with_columns(
        pl.when(
            pl.col("Lactate")
            .struct.field("system")
            .is_in(["Blood", "Blood arterial"])
            | pl.col("Lactate").struct.field("system").is_null()
        )
        .then(pl.col("Lactate").struct.field("value"))
        .alias("Lactate"),
        pl.col("Time Relative to Admission (seconds)")
        .truediv(SECONDS_IN_A_DAY)
        .ceil()
        .alias("Days Relative to Admission"),
    )
    .filter(pl.col("Lactate").is_not_null())
    .group_by("Global ICU Stay ID", "Days Relative to Admission")
    .agg(pl.col("Lactate").mean())
)

# Prepare death information from patient info
death = info.select(
    "Global ICU Stay ID",
    pl.col("ICU Length of Stay (days)")
    .ceil()
    .alias("Days Relative to Admission"),
    pl.col("Mortality in ICU").fill_null(False).alias("Death"),
)

# Get SOFA scores at each timestep
if not os.path.exists(f"{output_dir}/SOFA.parquet"):
    (
        SOFA(
            window_size=SECONDS_IN_A_DAY,
            timeframe_name="Days Relative to Admission",
        ).sink_parquet(f"{output_dir}/SOFA.parquet")
    )
sofa = (
    pl.scan_parquet(f"{output_dir}/SOFA.parquet")
    .select(
        "Global ICU Stay ID",
        "Days Relative to Admission",
        "SOFA Score",
    )
    .cast({"Days Relative to Admission": float})
)

# Select cohort of patients aged 18 and up with ICU stay > 1
cohort = (
    info.select(
        "Global ICU Stay ID",
        "Source Dataset",
        pl.col("Admission Age (years)").alias("Age"),
        pl.col("ICU Length of Stay (days)").alias("ICU_LOS_days"),
        "Mortality in ICU",
    ).filter(
        pl.col("Age").ge(18),
        pl.col("ICU_LOS_days") > 1,
    )
    # Drop patients without valid lactate measurements (negative or extreme)
    .join(
        lactate.group_by("Global ICU Stay ID")
        .agg(pl.col("Lactate").is_between(0, 50).all().alias("valid_lactate"))
        .filter(pl.col("valid_lactate")),
        on="Global ICU Stay ID",
        how="inner",
    )
)

# Join data (keep lazy)
JOIN_ON = ["Global ICU Stay ID", "Days Relative to Admission"]
data = (
    sofa.join(lactate, on=JOIN_ON, how="full", coalesce=True)
    .join(death, on=JOIN_ON, how="full", coalesce=True)
    .join(cohort, on="Global ICU Stay ID", how="inner")
    # Filter data up to and including the time of death
    .filter(
        pl.col("Death")
        .cum_max()
        .shift(1)
        .over(
            partition_by="Global ICU Stay ID",
            order_by="Days Relative to Admission",
        )
        .fill_null(False)
        .not_()
    )
    # Forward fill Lactate and SOFA after filtering
    .with_columns(
        pl.col("Lactate", "SOFA Score")
        .forward_fill()
        .over(
            partition_by="Global ICU Stay ID",
            order_by="Days Relative to Admission",
        ),
        pl.col("Death").fill_null(False),
    )
    .select(
        "Global ICU Stay ID",
        "Source Dataset",
        "Days Relative to Admission",
        "Lactate",
        "SOFA Score",
        "Death",
    )
    .filter(pl.col("Days Relative to Admission") > 0)
    .unique(subset=["Global ICU Stay ID", "Days Relative to Admission"])
    .drop_nulls(subset=["Global ICU Stay ID", "Days Relative to Admission"])
    .sort("Global ICU Stay ID", "Days Relative to Admission")
)
data.sink_parquet(f"{output_dir}/cox_data.parquet")

INCLUDED = data.select("Global ICU Stay ID").unique()
SOURCE_DATABASE = info.select("Global ICU Stay ID", "Source Dataset")

print(
    INCLUDED.join(SOURCE_DATABASE, on="Global ICU Stay ID", how="left")
    .group_by("Source Dataset")
    .len()
    .sort("Source Dataset")
    .collect()
)

# region Cox model
# ------------------------------------------------------------------------------
# Fit Cox proportional hazards models for each dataset

datasets = (
    data.select(pl.col("Source Dataset").unique())
    .collect()
    .to_series()
    .to_list()
)
datasets = [ds for ds in datasets if ds is not None]
datasets.sort()

print("Datasets available:")
for ds in datasets:
    print(f"  - {ds}")

# Store results for each dataset
cox_results = {}

for dataset in datasets:
    print(f"\nProcessing {dataset}...")

    # Filter data for this dataset (keep lazy)
    data_ds = data.filter(pl.col("Source Dataset") == dataset)

    # Prepare data for Cox model: create intervals from hour to hour
    data_cox_lazy = data_ds.select(
        "Global ICU Stay ID",
        pl.col("Days Relative to Admission").sub(1).alias("Start"),
        pl.col("Days Relative to Admission").alias("End"),
        "Lactate",
        pl.col("SOFA Score"),
        pl.col("Death").alias("Event"),
    ).drop_nulls(subset=["Lactate", "SOFA Score"])

    # Collect data for this dataset
    data_cox = data_cox_lazy.collect().to_pandas()
    n_subjects = data_cox["Global ICU Stay ID"].nunique()
    n_events = data_cox["Event"].sum()

    # Fit Cox model with time-varying covariates
    cph = CoxTimeVaryingFitter(penalizer=0.1)
    cph.fit(
        data_cox,
        formula="Lactate + `SOFA Score`",
        id_col="Global ICU Stay ID",
        event_col="Event",
        start_col="Start",
        stop_col="End",
        show_progress=True,
    )

    cox_results[dataset] = {
        "model": cph,
        "n_subjects": n_subjects,
        "n_events": n_events,
        "data": data_cox,
    }

    print(f"  {n_subjects} subjects, {n_events} events, {len(data_cox)} data points") # fmt: skip
    print(cph.summary[["coef", "exp(coef)", "se(coef)"]].to_string())

# region save results
# ------------------------------------------------------------------------------
# Prepare data for forest plot and save to CSV

forest_data = []
model_stats = []

for dataset in datasets:
    model = cox_results[dataset]["model"]
    summary = model.summary

    # Store model stats
    model_stats.append(
        {
            "Dataset": dataset,
            "n_subjects": cox_results[dataset]["n_subjects"],
            "n_events": cox_results[dataset]["n_events"],
            "n_datapoints": len(cox_results[dataset]["data"]),
            "event_rate": (
                cox_results[dataset]["n_events"]
                / cox_results[dataset]["n_subjects"]
            ),
        }
    )

    for var in ["Lactate", "SOFA Score"]:
        if var in summary.index:
            row_data = summary.loc[var]
            forest_data.append(
                {
                    "Dataset": dataset,
                    "Variable": var,
                    "HR": row_data["exp(coef)"],
                    "CI_lower": np.exp(row_data["coef"] - 1.96 * row_data["se(coef)"]),
                    "CI_upper": np.exp(row_data["coef"] + 1.96 * row_data["se(coef)"]),
                } # fmt: skip
            )

# Save results to CSVs
forest_df = pd.DataFrame(forest_data)
forest_df.to_csv(f"{output_dir}/forest_results.csv", index=False)

stats_df = pd.DataFrame(model_stats)
stats_df.to_csv(f"{output_dir}/model_stats.csv", index=False)

print("\nResults saved to:")
print(f"  - {output_dir}/forest_results.csv")
print(f"  - {output_dir}/model_stats.csv")
