# Case study III: Phenotype clustering of ICU patients with sepsis based on 
# temperature trajectories
################################################################################

# Source:
#   Bhavani SV, Carey KA, Gilbert ER, Afshar M, Verhoef PA, Churpek MM.
#   Identifying Novel Sepsis Subphenotypes Using Temperature Trajectories.
#   Am J Respir Crit Care Med. 2019 Aug 1;200(3):327-335. doi: 10.1164/rccm.201806-1197OC. PMID: 30789749; PMCID: PMC6680307.

# ------------------------------------------------------------------------------

# Vital signs:
#   Temperature

# ------------------------------------------------------------------------------

# Model: Group-based trajectory modeling (GBTM)
# Covariates: Temperature

################################################################################

import os

import polars as pl
import reprodICU
from reprodICU.utils.sepsis import SEPSIS
from tableone import TableOne

# region constants
# ------------------------------------------------------------------------------
SECONDS_IN_1H = 60 * 60
SECONDS_IN_1D = 24 * SECONDS_IN_1H
SECONDS_IN_12H = 12 * SECONDS_IN_1H
SECONDS_IN_72H = 72 * SECONDS_IN_1H

# Create output directory
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# region lazyload
# ------------------------------------------------------------------------------
# X. lazy-load the datasets
info: pl.LazyFrame = reprodICU.patient_information
meds: pl.LazyFrame = reprodICU.medications
vitals: pl.LazyFrame = reprodICU.timeseries_vitals

# region 0. study design
################################################################################
# 0. Study design
print("0. Setting up the study design")
# ------------------------------------------------------------------------------
# 0.1 Defining time of eligibility assessment => (1)
# ------------------------------------------------------------------------------
# -> t_eligibility = [icu_admission] which is 0 by definition in this harmonized dataset

T_ELIGIBILITY = info.select(
    "Global ICU Stay ID",
    pl.lit(0).alias("T_ELIGIBILITY"),
)

# region t_0 / t_1
# ------------------------------------------------------------------------------
# 0.2 Defining exposure period (for 2.1) => (2/3)
# ------------------------------------------------------------------------------
# -> t_0 = t_eligibility
# -> t_1 = [t_0 + 72h]

T_0 = T_ELIGIBILITY.select(
    "Global ICU Stay ID", pl.col("T_ELIGIBILITY").alias("T_0")
)
T_1 = T_0.with_columns((pl.col("T_0") + SECONDS_IN_72H).alias("T_1"))


# region 1. initial cohort
################################################################################
# 1. Defining the primary study cohort
print("1. Defining the initial study cohort")
# ------------------------------------------------------------------------------
# Here, all variables used to select patients of interest should be created (considering the time of eligibility assessment).
# However, dropping patients of not interest for this study should be done below (## 5. Figure 1 / Study flow)
# ------------------------------------------------------------------------------
# Inclusion criteria:
# - first ICU admission
# - age >= 18 years at ICU admission
# - ICU admission <= 48 hours after hospitalization
# - ICU length of stay >= 3 days
# - sepsis within 24 hours of ICU admission (t_eligibility)

# Exclusion criteria:
# - no temperature measurements within the first 72 hours of ICU admission

# region 1.0 inclusion_exclusion
# ------------------------------------------------------------------------------
# 1.0 Defining data source(s) for inclusion/exclusion criteria
# ------------------------------------------------------------------------------

CASE_IDS = (
    info.select("Global ICU Stay ID", "Source Dataset")
    # .filter(pl.col("Source Dataset").str.contains("MIMIC"))
    .unique()
)

ICU_STAY_NUM = info.select(
    "Global Person ID",
    "Global ICU Stay ID",
    "ICU Stay Sequential Number (per Person ID)",
    "Table: Timeseries (Vitals)",
)
ICU_STAY_NUM_EICU = info.select(
    "Global Person ID",
    "Global Hospital Stay ID",
    "Global ICU Stay ID",
    "ICU Stay ID",
    "Admission Year",
    "Admission Age (years)",
    "Pre-ICU Length of Stay (days)",
    "Table: Timeseries (Respiratory data)",
)

AGES = info.select("Global ICU Stay ID", "Admission Age (years)")
ICU_LENGTH_OF_STAY = info.select(
    "Global ICU Stay ID",
    "ICU Length of Stay (days)",
    "Pre-ICU Length of Stay (days)",
)

if not os.path.exists(f"{output_dir}/SEPSIS.parquet"):
    print("   -> calculating sepsis / septic shock onset data", end="\r")
    SEPSIS(t_0_per_stay=T_0).sink_parquet(f"{output_dir}/SEPSIS.parquet")
    print("   -> calculated sepsis / septic shock onset data")

SEPSIS_ONSET = (
    pl.scan_parquet(f"{output_dir}/SEPSIS.parquet")
    .rename({"timeframe": "Hours Relative to ICU Admission"})
    .select(
        "Global ICU Stay ID",
        "Hours Relative to ICU Admission",
        # pl.col("SEPSIS_RHEE").alias("SEPSIS"),
        "SEPSIS",
    )
    .filter(pl.col("SEPSIS").is_not_null())
    .group_by("Global ICU Stay ID")
    .agg(pl.col("Hours Relative to ICU Admission").min().alias("SEPSIS_ONSET"))
)

if not os.path.exists(f"{output_dir}/NUM_TEMPERATURE.parquet"):
    print("   -> calculating number of temperature measurements in first 72h", end="\r") # fmt: skip
    (
        vitals.select(
            "Global ICU Stay ID",
            "Time Relative to Admission (seconds)",
            "Temperature",
        )
        .filter(
            pl.col("Time Relative to Admission (seconds)") >= 0,
            pl.col("Time Relative to Admission (seconds)") <= SECONDS_IN_72H,
            pl.col("Temperature").is_not_null(),
        )
        .group_by("Global ICU Stay ID")
        .agg(pl.len().alias("# of temperature measurements in first 72h"))
        .sink_parquet(f"{output_dir}/NUM_TEMPERATURE.parquet")
    )
    print("   -> calculated number of temperature measurements in first 72h")

NUM_TEMPERATURE = pl.scan_parquet(f"{output_dir}/NUM_TEMPERATURE.parquet")

# region 1.1 eligibility
# ------------------------------------------------------------------------------
# 1.1 Defining time of eligibility
# ------------------------------------------------------------------------------

FIRST_ADMIT = ICU_STAY_NUM.with_columns(
    # fill_null is for HiRID, where ICU stay numbers are not available
    pl.col("ICU Stay Sequential Number (per Person ID)")
    .eq(
        pl.col("ICU Stay Sequential Number (per Person ID)")
        # .filter(pl.col("Table: Timeseries (Vitals)").is_not_null())
        .min().over("Global Person ID")
    )
    .fill_null(True)
    .alias("Inclusion: 1st ICU admission"),
).select("Global ICU Stay ID", "Inclusion: 1st ICU admission")

# L27–46 from eicu/cohort.sql of Neto et al. (2018)
FIRST_ADMIT_EICU = (
    ICU_STAY_NUM_EICU.with_columns(
        pl.col("Global ICU Stay ID")
        .rank("ordinal")
        .over(
            partition_by="Global Person ID",
            order_by=[
                "Admission Year",
                "Admission Age (years)",
                "ICU Stay ID",
                "Pre-ICU Length of Stay (days)",
            ],
        )
        .alias("eicu_hospital_admission_rank"),
        pl.col("Global ICU Stay ID")
        .rank("ordinal")
        .over(
            partition_by="Global Hospital Stay ID",
            order_by=["Pre-ICU Length of Stay (days)"],
        )
        .alias("eicu_icu_admission_rank"),
    )
    .with_columns(
        (
            pl.col("eicu_hospital_admission_rank").eq(1)
            & pl.col("eicu_icu_admission_rank").eq(1)
            & pl.col("Table: Timeseries (Respiratory data)").is_not_null()
        ).alias("is_first_eicu_admission")
    )
    .select("Global ICU Stay ID", "is_first_eicu_admission")
)

INCLUSION_EXCLUSION = (
    CASE_IDS.join(FIRST_ADMIT, on="Global ICU Stay ID", how="left")
    .join(FIRST_ADMIT_EICU, on="Global ICU Stay ID", how="left")
    .with_columns(
        pl.when(pl.col("Source Dataset") == "eICU-CRD")
        .then(pl.col("is_first_eicu_admission"))
        .otherwise(pl.col("Inclusion: 1st ICU admission"))
        .alias("Inclusion: 1st ICU admission")
    )
)

# 1.2 Cleaning raw data for clinically implausible values
# 1.3 Quality check

# region 1.4 aggregation
# ------------------------------------------------------------------------------
# 1.4 Aggregation of variables
# ------------------------------------------------------------------------------

# region 1.4.I inclusion
# ------------------------------------------------------------------------------
# 1.4.I.1 Age >= 18
INCLUSION_EXCLUSION = (
    INCLUSION_EXCLUSION.join(AGES, on="Global ICU Stay ID", how="left")
    .with_columns(
        (pl.col("Admission Age (years)") >= 18).alias("Inclusion: Age > 18")
    )
    .drop("Admission Age (years)")
)

# 1.4.I.2 ICU admission <= 48 hours after hospitalization
# 1.4.I.3 ICU length of stay >= 3 days
INCLUSION_EXCLUSION = (
    INCLUSION_EXCLUSION.join(
        ICU_LENGTH_OF_STAY, on="Global ICU Stay ID", how="left"
    )
    .with_columns(
        (
            (pl.col("Pre-ICU Length of Stay (days)") <= 2)
            | pl.col("Pre-ICU Length of Stay (days)").is_null()
        ).alias("Inclusion: ICU admission <= 48 hours after hospitalization"),
        (pl.col("ICU Length of Stay (days)") >= 3).alias(
            "Inclusion: ICU LOS >= 3 days"
        ),
    )
    .drop("ICU Length of Stay (days)", "Pre-ICU Length of Stay (days)")
)

# 1.4.I.4 Sepsis within 24 hours of ICU admission
INCLUSION_EXCLUSION = (
    INCLUSION_EXCLUSION.join(SEPSIS_ONSET, on="Global ICU Stay ID", how="left")
    .with_columns(
        pl.col("SEPSIS_ONSET")
        .is_between(0, 24, closed="both")
        .alias("Inclusion: Sepsis within 24 hours of ICU admission")
    )
    .drop("SEPSIS_ONSET")
)

# region 1.4.E exclusion
# ------------------------------------------------------------------------------
# 1.4.E.1 exclude patients without temperature measurements within the first 72 hours of ICU admission
INCLUSION_EXCLUSION = (
    INCLUSION_EXCLUSION.join(
        NUM_TEMPERATURE, on="Global ICU Stay ID", how="left"
    )
    .with_columns(
        pl.col("# of temperature measurements in first 72h")
        .ge(3)
        .alias("Exclusion: No temperature measurements within first 72 hours")
    )
    .drop("# of temperature measurements in first 72h")
)


# region 1.5 wide format
# ------------------------------------------------------------------------------
# 1.5 Push aggregated variable to wide format DF for analyses
# ------------------------------------------------------------------------------
INCLUSION_CRITERIA = [
    # "Inclusion: 1st ICU admission",
    "Inclusion: Age > 18",
    "Inclusion: ICU admission <= 48 hours after hospitalization",
    "Inclusion: ICU LOS >= 3 days",
    "Inclusion: Sepsis within 24 hours of ICU admission",
    "Exclusion: No temperature measurements within first 72 hours",
]

INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.unique().with_columns(
    pl.all_horizontal(*INCLUSION_CRITERIA).alias("is_included")
)

INCLUDED = (
    INCLUSION_EXCLUSION.filter(pl.col("is_included"))
    .select("Global ICU Stay ID")
    .unique()
)

SOURCE_DATABASE = info.select("Global ICU Stay ID", "Source Dataset")

print(
    INCLUSION_EXCLUSION.join(
        SOURCE_DATABASE, on="Global ICU Stay ID", how="left"
    )
    .select("Source Dataset", "is_included")
    .group_by("Source Dataset")
    .sum()
    .sort("Source Dataset")
    .collect()
)

# | Source Dataset | is_included |
# | -------------- | ----------- |
# │ AmsterdamUMCdb ┆ 2076        │
# │ eICU-CRD       ┆ 199         │
# │ MIMIC-III      ┆ 1286        │
# │ MIMIC-IV       ┆ 3409        │


INCLUSION_EXCLUSION.join(
    info.select(
        "Global ICU Stay ID", "Global Hospital Stay ID", "Global Person ID"
    ),
    on="Global ICU Stay ID",
    how="left",
).sink_csv(f"{output_dir}/INCLUSION_COHORT.csv")

# ##############################################################################
# ##############################################################################
# ##############################################################################

temperature = (
    vitals.join(T_0, on="Global ICU Stay ID", how="left")
    .join(INCLUDED, on="Global ICU Stay ID", how="semi")
    .filter(
        pl.col("Time Relative to Admission (seconds)").is_between(
            pl.col("T_0"),
            pl.col("T_0") + SECONDS_IN_72H,
            closed="both",
        ),
        pl.col("Temperature").is_between(32, 44),
    )
    .drop_nulls("Temperature")
    # Scale temperature to standardize (remove the mean and divide by the standard deviation)
    .with_columns(
        (
            (pl.col("Temperature") - pl.mean("Temperature"))
            / pl.std("Temperature")
        ).alias("Temperature (standardized)")
    )
    # Select necessary columns for the next steps
    .select(
        "Global ICU Stay ID",
        pl.col("Time Relative to Admission (seconds)")
        .sub(pl.col("T_0"))
        .alias("Time Relative to T_0 (seconds)"),
        "Temperature",  # Original temperature
        "Temperature (standardized)",  # Standardized temperature
    )
    # Aggregate temperature measurements by hour
    .with_columns(
        pl.col("Time Relative to T_0 (seconds)")
        .floordiv(SECONDS_IN_1H)
        .alias("Hours Relative to T_0")
    )
    .group_by("Global ICU Stay ID", "Hours Relative to T_0")
    .agg(
        pl.col("Temperature", "Temperature (standardized)")
        .sort_by("Time Relative to T_0 (seconds)")
        .first()
    )
)


def save_temperature_trajectories(dataset: str):
    dataset_name = dataset.lower().replace("-", "_")
    (
        temperature.join(SOURCE_DATABASE, on="Global ICU Stay ID", how="left")
        .filter(pl.col("Source Dataset") == dataset)
        # Save the standardized temperature trajectories
        .sort("Hours Relative to T_0")
        .select(
            pl.col("Global ICU Stay ID").alias("id"),
            pl.col("Hours Relative to T_0").alias("time").cast(int),
            pl.col("Temperature (standardized)").alias("temp"),
        )
        .collect()
        .pivot(index="id", on="time", values=["temp", "time"])
        .write_csv(f"{output_dir}/TEMPERATURE_GBTM_{dataset_name}.csv")
    )


# Save temperature trajectories for fitting GBTM model
for ds in ["MIMIC-III", "MIMIC-IV", "AmsterdamUMCdb"]:
    save_temperature_trajectories(ds)

# Save the temperature trajectories
temperature.sink_parquet(f"{output_dir}/TEMPERATURE.parquet")

# ##############################################################################
# ##############################################################################
# ##############################################################################

# region TableOne
# ------------------------------------------------------------------------------
table1 = info.join(INCLUDED, on="Global ICU Stay ID", how="semi").select(
    "Global ICU Stay ID",
    "Source Dataset",
    "Mortality in Hospital",
    "Gender",
    "Admission Age (years)",
    "ICU Length of Stay (days)",
    pl.when(
        pl.col("Ethnicity").is_in(
            ["Asian", "Black or African American", "White"]
        )
    )
    .then(pl.col("Ethnicity"))
    .otherwise(pl.lit("Other"))
    .alias("Race"),
    pl.when(pl.col("Ethnicity") == "Hispanic or Latino")
    .then(pl.lit("Hispanic or Latino"))
    .otherwise(pl.lit("Not Hispanic or Latino"))
    .alias("Ethnicity"),
)

columns = [
    "Admission Age (years)",
    "Gender",
    "Race",
    "Ethnicity",
    "ICU Length of Stay (days)",
    "Mortality in Hospital",
]
categorical = [
    "Gender",
    "Race",
    "Ethnicity",
    "Mortality in Hospital",
]
continuous = [
    "Admission Age (years)",
    "ICU Length of Stay (days)",
]
groupby = "Source Dataset"
nonnormal = ["ICU Length of Stay (days)"]
table1 = TableOne(
    table1.collect().to_pandas(),
    columns=columns,
    categorical=categorical,
    continuous=continuous,
    groupby=groupby,
    nonnormal=nonnormal,
    # rename=rename,
    pval=False,
    missing=False,
    limit={
        "Mortality in Hospital": 1,
        "Gender": 1,
        "Race": 4,
    },
    order={
        "Mortality in Hospital": ["True", "False"],
        "Gender": ["Female", "Male"],
        "Race": ["Asian", "Black or African American", "White", "Other"],
    },
)

# Save table1 to file
# Save Table 1 to CSV and print
table1_output_file = f"{output_dir}/table1.csv"
table1.tableone.replace("0 (nan)", None, inplace=True)
table1.tableone.replace("0 (0.0)", None, inplace=True)
table1.to_csv(table1_output_file)
print(f"Table 1 saved to {table1_output_file}.")

# also save as markdown for easy inclusion in reports
table1_md = (
    table1.tableone.to_markdown()
    .replace("('Grouped by Source Dataset', '", " " * 31)
    .replace("True", " " * 4)
    .replace("')", " " * 2)
    .replace("('", " " * 2)
    .replace("', '", ", ")
    .replace(".0  ", " " * 4)
    .replace(",   ", " " * 4)
)
# write to file
with open(f"{output_dir}/table1.md", "w") as f:
    f.write(table1_md)
