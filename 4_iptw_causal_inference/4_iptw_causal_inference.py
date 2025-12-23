# Case study IV: Causal effect of mechanical ventilation in the first 24 hours
# after respiratory failure on ICU mortality
################################################################################

# Demographics:
#   Gender (1 for female, 0 for male), Admission Age (years), BMI

# Vital Signs:
#   Glasgow coma score

# Laboratory Results:
#   PaO2, PaCO2, PaO2/FiO2 ratio, Lactate

# Scores:
#   SOFA score

# ------------------------------------------------------------------------------

# Model: LightGBM Classifier with hyperparameter tuning via GridSearchCV
# Outcome: ICU mortality
# Covariates: demographics, vital signs, and lab results listed above

################################################################################
import os

import numpy as np
import polars as pl
import reprodICU
import statsmodels.api as sm
from reprodICU.utils.clinical import RESPIRATORY_FAILURE, PaO2_FiO2_RATIO
from reprodICU.utils.common import DROP_IMPLAUSIBLE_VALUES
from reprodICU.utils.mortality import COMMON_MORTALITY_MEASURES
from reprodICU.utils.scores.SOFA import SOFA
from tableone import TableOne

# region constants
# ------------------------------------------------------------------------------
STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"

SECONDS_IN_1H = 60 * 60
SECONDS_IN_24H = 24 * SECONDS_IN_1H

# Create output directory
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# region lazyload
# ------------------------------------------------------------------------------
# X. lazy-load the datasets
info: pl.LazyFrame = reprodICU.patient_information
meds: pl.LazyFrame = reprodICU.medications
vitals: pl.LazyFrame = reprodICU.timeseries_vitals
labs: pl.LazyFrame = reprodICU.timeseries_labs
resp: pl.LazyFrame = reprodICU.timeseries_respiratory
inout: pl.LazyFrame = reprodICU.timeseries_intakeoutput
diags: pl.LazyFrame = reprodICU.diagnoses

vent: pl.LazyFrame = reprodICU.VENTILATION_DURATION


# region 0. study design
################################################################################
# 0. Study design
print("0. Setting up the study design")
# ------------------------------------------------------------------------------
# 0.1 Defining time of eligibility assessment => (1)
# ------------------------------------------------------------------------------
# -> t_eligibility = [icu_admission] which is 0 by definition in this harmonized dataset

T_ELIGIBILITY = (
    info.select(STAY_KEY)
    .unique()
    .with_columns(pl.lit(0).alias("T_ELIGIBILITY"))
)

# region t_0 / t_1
# ------------------------------------------------------------------------------
# 0.2 Defining exposure period (for 2.1) => (2/3)
# ------------------------------------------------------------------------------
# -> t_0 = [respiratory failure]
# -> t_1 = 24 hours after t_0

if not os.path.exists("data/RESPIRATORY_FAILURE.parquet"):
    RESPIRATORY_FAILURE().sink_parquet("data/RESPIRATORY_FAILURE.parquet")
    print("   -> calculated RESPIRATORY_FAILURE data")
RESP_FAILURE = pl.scan_parquet("data/RESPIRATORY_FAILURE.parquet")

T_0 = (
    RESP_FAILURE.filter(
        pl.col(TIME_KEY) >= 0,
        pl.any_horizontal(
            pl.col(col).is_not_null() for col in ["PaO2", "PaCO2"]
        ),
    )
    .group_by(STAY_KEY)
    .agg(pl.col(TIME_KEY).min().alias("T_0"))
)
T_1 = T_0.with_columns((pl.col("T_0") + SECONDS_IN_24H).alias("T_1"))

# 0.3 Defining time of outcome evaluation (for 3.1) => (4)

# region t_baseline
# ------------------------------------------------------------------------------
# 0.4 Defining time frame for standard confounders (ideally lying between (1) and (2/3))
# ------------------------------------------------------------------------------
T_BASELINE = T_0.with_columns(
    pl.col("T_0").sub(SECONDS_IN_24H).alias("T_BASELINE"),
)

# region 1. initial cohort
################################################################################
# 1. Defining the primary study cohort
print("1. Defining the initial study cohort")
# ------------------------------------------------------------------------------
# Here, all variables used to select patients of interest should be created (considering the time of eligibility assessment).
# However, dropping patients of not interest for this study should be done below (## 5. Figure 1 / Study flow)
# ------------------------------------------------------------------------------
# Inclusion criteria:
# - age >= 18 years
# - first ICU stay
# - ICU length of stay >= 24 hours
# - in respiratory failure
#
# Exclusion criteria:
# - mechanically ventilated before respiratory failure / ICU admission
# - no respiratory data available
# - no ICU outcome data available

# region 1.0 inclusion_exclusion
# ------------------------------------------------------------------------------
# 1.0 Defining data source(s) for inclusion/exclusion criteria
# ------------------------------------------------------------------------------
CASE_IDS = info.select(STAY_KEY, "Source Dataset").unique()
FIRST_STAY = info.select(STAY_KEY, "ICU Stay Sequential Number (per Person ID)")
LENGTH_OF_STAY = info.select(STAY_KEY, "ICU Length of Stay (days)")
AGE = info.select(STAY_KEY, "Admission Age (years)")
MECHANICAL_VENTILATION = vent.filter(
    (pl.col("Ventilation Type").is_in(["invasive ventilation", "tracheostomy"]))
    | (pl.col(STAY_KEY).str.starts_with("eicu-"))
)
RESPIRATORY_DATA = info.select(STAY_KEY, "Table: Timeseries (Respiratory data)")

# region 1.1 eligibility
# ------------------------------------------------------------------------------
# 1.1 Defining time of eligibility
# ------------------------------------------------------------------------------
INCLUSION_EXCLUSION = CASE_IDS.filter(
    pl.col("Source Dataset") != "NWICU",  # NWICU has no respiratory data at all
    pl.col("Source Dataset") != "SICdb",  # SICdb has no GCS data at all
)

# 1.2 Cleaning raw data for clinically implausible values
# 1.3 Quality check

# region 1.4 aggregation
# ------------------------------------------------------------------------------
# 1.4 Aggregation of variables
# ------------------------------------------------------------------------------

# region 1.4.I inclusion
# ------------------------------------------------------------------------------
# 1.4.I.1 age >= 18
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    AGE.select(
        STAY_KEY,
        (pl.col("Admission Age (years)") >= 18).alias("Inclusion: age > 18"),
    ),
    on=STAY_KEY,
    how="left",
).with_columns(pl.col("Inclusion: age > 18").fill_null(False))

# 1.4.I.2 first ICU stay
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    FIRST_STAY.select(
        STAY_KEY,
        (
            (pl.col("ICU Stay Sequential Number (per Person ID)") == 1)
            | pl.col("ICU Stay Sequential Number (per Person ID)").is_null()
        ).alias("Inclusion: first ICU stay"),
    ),
    on=STAY_KEY,
    how="left",
).with_columns(pl.col("Inclusion: first ICU stay").fill_null(False))

# 1.4.I.3 ICU length of stay >= 24 hours
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    LENGTH_OF_STAY.select(
        STAY_KEY,
        (pl.col("ICU Length of Stay (days)") >= 1).alias(
            "Inclusion: ICU length of stay >= 24 hours"
        ),
    ),
    on=STAY_KEY,
    how="left",
).with_columns(
    pl.col("Inclusion: ICU length of stay >= 24 hours").fill_null(False)
)

# 1.4.I.4 in respiratory failure
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    RESP_FAILURE.filter(pl.col(TIME_KEY) >= 0)
    .select(
        STAY_KEY,
        pl.lit(True).alias("Inclusion: in respiratory failure"),
    )
    .unique(),
    on=STAY_KEY,
    how="left",
).with_columns(pl.col("Inclusion: in respiratory failure").fill_null(False))


# region 1.4.E exclusion
# ------------------------------------------------------------------------------
# 1.4.E.1 mechanically ventilated before respiratory failure / ICU admission
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    MECHANICAL_VENTILATION.join(T_0, on=STAY_KEY, how="left")
    .filter(
        pl.col("Ventilation Start Relative to Admission (seconds)")
        > pl.col("T_0")
    )
    .select(
        STAY_KEY,
        pl.lit(True).alias("Exclusion: mechanically ventilated before T_0"),
    ),
    on=STAY_KEY,
    how="left",
).with_columns(
    pl.col("Exclusion: mechanically ventilated before T_0")
    .fill_null(False)
    .not_()
)

# 1.4.E.2 no respiratory data available
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    RESPIRATORY_DATA.select(
        STAY_KEY,
        pl.col("Table: Timeseries (Respiratory data)")
        .is_null()
        .alias("Exclusion: no respiratory data available"),
    ),
    on=STAY_KEY,
    how="left",
).with_columns(
    pl.col("Exclusion: no respiratory data available").fill_null(False).not_()
)

# 1.4.E.3 no ICU outcome data available
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    COMMON_MORTALITY_MEASURES()
    .drop_nulls(subset=["Mortality in ICU"])
    .select(
        STAY_KEY,
        pl.lit(False).alias("Exclusion: no ICU outcome data available"),
    ),
    on=STAY_KEY,
    how="left",
).with_columns(
    pl.col("Exclusion: no ICU outcome data available").fill_null(True).not_()
)


# region 1.5 wide format
# ------------------------------------------------------------------------------
# 1.5 Push aggregated variable to wide format DF for analyses
# ------------------------------------------------------------------------------
INCLUSION_CRITERIA = [
    "Inclusion: age > 18",
    "Inclusion: first ICU stay",
    "Inclusion: ICU length of stay >= 24 hours",
    "Inclusion: in respiratory failure",
    "Exclusion: mechanically ventilated before T_0",
    "Exclusion: no respiratory data available",
    "Exclusion: no ICU outcome data available",
]

INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.unique().with_columns(
    pl.all_horizontal(INCLUSION_CRITERIA).alias("is_included")
)

INCLUDED = (
    INCLUSION_EXCLUSION.filter(pl.col("is_included")).select(STAY_KEY).unique()
)

SOURCE_DATABASE = info.select(STAY_KEY, "Source Dataset")

print(
    INCLUSION_EXCLUSION.join(SOURCE_DATABASE, on=STAY_KEY, how="left")
    .select("Source Dataset", "is_included")
    .group_by("Source Dataset")
    .sum()
    .sort(pl.col("Source Dataset").str.to_lowercase())
    .collect()
)

# ┌────────────────┬─────────────┐
# │ Source Dataset ┆ is_included │
# ╞════════════════╪═════════════╡
# │ AmsterdamUMCdb ┆ 10766       │
# │ eICU-CRD       ┆ 53578       │
# │ HiRID          ┆ 18467       │
# │ MIMIC-III      ┆ 10640       │
# │ MIMIC-IV       ┆ 14838       │
# │ NWICU          ┆ 0           │
# │ SICdb          ┆ 11093       │
# └────────────────┴─────────────┘

INCLUSION_EXCLUSION.join(
    info.select(STAY_KEY, "Global Hospital Stay ID", "Global Person ID"),
    on=STAY_KEY,
    how="left",
).sink_csv("data/INCLUSION_COHORT.csv")

# region 1.X flowchart
# ------------------------------------------------------------------------------
# 1.X Creating study flowchart
# ------------------------------------------------------------------------------
_INCLUSION_EXCLUSION = pl.read_csv("data/INCLUSION_COHORT.csv")
df = _INCLUSION_EXCLUSION.to_pandas()
for criterion in INCLUSION_CRITERIA:
    # ensure missing values are treated as False before boolean inversion
    idxRem = ~df[criterion].fillna(False).astype(bool)
    print(
        "{:6d} - removing {:6d} ({:5.2f}%) patients - {}.".format(
            df.shape[0], np.sum(idxRem), 100.0 * np.mean(idxRem), criterion
        )
    )
    df = df.loc[~idxRem, :]
print("{:6d} - final cohort.\n".format(df.shape[0]), end="\n\n")

for db, database in [
    ("eicu-", "eICU-CRD"),
    ("mimic3-", "MIMIC-III"),
    ("mimic4-", "MIMIC-IV"),
    ("nwicu-", "NWICU"),
    ("hirid-", "HiRID"),
    ("sicdb-", "SICdb"),
    ("umcdb-", "UMCdb"),
]:
    df = _INCLUSION_EXCLUSION.filter(
        pl.col(STAY_KEY).str.starts_with(db)
    ).to_pandas()
    for criterion in INCLUSION_CRITERIA:
        idxRem = ~df[criterion].fillna(False).astype(bool)
        print(
            "{:6d} - removing {:6d} ({:6.2f}%) patients from {} - {}.".format(
                df.shape[0],
                np.sum(idxRem),
                100.0 * np.mean(idxRem),
                database,
                criterion,
            )
        )
        df = df.loc[~idxRem, :]
    print("{:6d} - final cohort.".format(df.shape[0]), end="\n\n")

# region 2. primary exposure
################################################################################
# 2. Defining the primary exposure
print("2. Defining the primary exposure")
# ------------------------------------------------------------------------------
# 2.0 Defining data source(s) for exposure definitions

# region 2.1 PERIOD OF EXPOSURE
# ------------------------------------------------------------------------------
# 2.1 Defining "time 0" and exposure period
# ------------------------------------------------------------------------------
# We are interested in the baseline period (first 24 hours)
EXPOSURE_PERIOD = T_1

# 2.2 Cleaning raw data for clinically implausible values
# 2.3 Quality check

# region 2.4 aggregation
# ------------------------------------------------------------------------------
# 2.4 Aggregation of final primary exposure variable
# ------------------------------------------------------------------------------
MECHANICAL_VENTILATION_IN_FIRST_24H = (
    MECHANICAL_VENTILATION.join(EXPOSURE_PERIOD, on=STAY_KEY, how="inner")
    .filter(
        pl.col("Ventilation Start Relative to Admission (seconds)").is_between(
            0, SECONDS_IN_24H, closed="both"
        )
    )
    .select(
        STAY_KEY,
        pl.lit(True).alias("Mechanical Ventilation in first 24h"),
    )
    .unique()
)

# region 2.5 wide format
# ------------------------------------------------------------------------------
# 2.5 Push aggregated variable to wide format DF for analyses
# ------------------------------------------------------------------------------
EXPOSURE = EXPOSURE_PERIOD.join(
    MECHANICAL_VENTILATION_IN_FIRST_24H,
    on=STAY_KEY,
    how="left",
).with_columns(pl.col("Mechanical Ventilation in first 24h").fill_null(False))


# region 3. primary outcome
################################################################################
# 3. Defining the primary outcome
print("3. Defining the primary outcome")
# ------------------------------------------------------------------------------
# 3.0 Defining data source(s) for outcome definitions
MORTALITY_MEASURES = COMMON_MORTALITY_MEASURES()

# 3.1 Time of outcome evaluation
# 3.2 Cleaning raw data for clinically implausible values
# 3.3 Quality check

# region 3.4 aggregation
# ------------------------------------------------------------------------------
# 3.4 Aggregation of final primary outcome variable
# ------------------------------------------------------------------------------
OUTCOME = MORTALITY_MEASURES.join(
    INCLUSION_EXCLUSION.filter(pl.col("is_included")),
    on=STAY_KEY,
    how="right",
).select(STAY_KEY, "Mortality in ICU", "Mortality in Hospital")

# 3.5 Push aggregated outcome variable to wide format DF for analyses

# region 4. descriptives and confounders
################################################################################
# 4. Defining key descriptives and confounding variables for the primary analysis
print("4. Defining key descriptives and confounding variables for the primary analysis") # fmt: skip
# ------------------------------------------------------------------------------
# 4.1 Data source(s)
DEMOGRAPHICS = info.select(
    STAY_KEY,
    "Admission Age (years)",
    "Admission Height (cm)",
    "Admission Weight (kg)",
    "Gender",
    "Ethnicity",
    "Source Dataset",
)

HOROVITZ = PaO2_FiO2_RATIO(t_0_per_stay=T_0)

GCS = (
    vitals.select(STAY_KEY, TIME_KEY, "Glasgow coma score total")
    .drop_nulls()
    .collect()
    .lazy()
)

LABS = labs.select(
    STAY_KEY,
    TIME_KEY,
    # Lactate
    pl.when(
        pl.col("Lactate")
        .struct.field("system")
        .is_in(["Blood", "Blood arterial"])
        | pl.col("Lactate").struct.field("system").is_null()
    )
    .then(pl.col("Lactate").struct.field("value"))
    .alias("Lactate"),
    # PaO2
    pl.when(
        pl.col("Oxygen")
        .struct.field("system")
        .is_in(["Blood arterial", "Blood"])
        | pl.col("Oxygen").struct.field("system").is_null()
    )
    .then(pl.col("Oxygen").struct.field("value"))
    .otherwise(None)
    .alias("PaO2"),
    # PaCO2
    pl.when(
        pl.col("Carbon dioxide")
        .struct.field("system")
        .is_in(["Blood arterial", "Blood"])
        | pl.col("Carbon dioxide").struct.field("system").is_null()
    )
    .then(pl.col("Carbon dioxide").struct.field("value"))
    .otherwise(None)
    .alias("PaCO2"),
    # pH
    pl.when(
        pl.col("pH").struct.field("system").is_in(["Blood arterial", "Blood"])
        | pl.col("pH").struct.field("system").is_null()
    )
    .then(pl.col("pH").struct.field("value"))
    .otherwise(None)
    .alias("pH"),
    # SaO2
    pl.when(
        pl.col("Oxygen saturation")
        .struct.field("system")
        .is_in(["Blood arterial", "Blood"])
        | pl.col("Oxygen saturation").struct.field("system").is_null()
    )
    .then(pl.col("Oxygen saturation").struct.field("value"))
    .otherwise(None)
    .alias("SaO2"),
)

if not os.path.exists("data/SOFA.parquet"):
    SOFA(
        patient_information=info,
        timeseries_vitals=vitals,
        timeseries_labs=labs,
        timeseries_resp=resp,
        timeseries_inout=inout,
        medications=meds,
        ventilation=vent,
        t_0_per_stay=T_0,
        window_size=SECONDS_IN_24H,
        timeframe_name="Days Relative to T_0",
    ).sink_parquet("data/SOFA.parquet")
    print("   -> calculated SOFA score data")

SOFA_SCORES = (
    pl.scan_parquet("data/SOFA.parquet")
    .filter(pl.col("Days Relative to T_0") == 1)
    .select(STAY_KEY, "SOFA Score")
)

# region 4.2 PERIOD OF CONFOUNDING
# ------------------------------------------------------------------------------
# 4.2 Period of interest (e.g., Xh pre T0 OR Xh post admission)
# ------------------------------------------------------------------------------
# We define the covariate period as the first 24 hours
COVARIATE_PERIOD = T_BASELINE

# region 4.3 Cleaning raw data for clinically implausible values
# ------------------------------------------------------------------------------
# 4.3 Cleaning raw data for clinically implausible values
# ------------------------------------------------------------------------------

DEMOGRAPHICS_CLEANED = DEMOGRAPHICS.with_columns(
    pl.col(col).pipe(DROP_IMPLAUSIBLE_VALUES, col).alias(col)
    for col in ["Admission Height (cm)", "Admission Weight (kg)"]
)

LABS_CLEANED = LABS.with_columns(
    pl.col(col).pipe(DROP_IMPLAUSIBLE_VALUES, col_).alias(col)
    for col, col_ in zip(["PaO2", "PaCO2"], ["Oxygen", "Carbon dioxide"])
)

# 4.4 Quality check

# region 4.5 aggregation
# ------------------------------------------------------------------------------
# 4.5 Choose aggregation method and aggregate variable
# If multiple measurements were available for a variable during an epoch, the
# mean value was used.
# ------------------------------------------------------------------------------

HOROVITZ_BASELINE = (
    HOROVITZ.join(COVARIATE_PERIOD, on=STAY_KEY, how="inner")
    .filter(pl.col(TIME_KEY).is_between("T_BASELINE", "T_0", closed="both"))
    .group_by(STAY_KEY)
    .agg(
        pl.col("PaO2/FiO2 Ratio")
        .min()
        .alias("Worst PaO2/FiO2 Ratio prior to T_0")
    )
)

# ------------------------------------------------------------------------------

RESP_FAILURE_BASELINE = (
    RESP_FAILURE.join(COVARIATE_PERIOD, on=STAY_KEY, how="inner")
    .filter(pl.col(TIME_KEY) == pl.col("T_0"))
    .select(STAY_KEY, "Respiratory Failure Type")
    .with_columns(
        pl.col("Respiratory Failure Type")
        .is_in(["Acute Hypoxemic", "Mixed"])
        .cast(int)
        .alias("is hypoxemic"),
        pl.col("Respiratory Failure Type")
        .is_in(["Acute Hypercapnic", "Mixed"])
        .cast(int)
        .alias("is hypercapnic"),
    )
)

# ------------------------------------------------------------------------------

GCS_BASELINE = (
    GCS.join(COVARIATE_PERIOD, on=STAY_KEY, how="inner")
    .filter(pl.col(TIME_KEY).is_between("T_BASELINE", "T_0", closed="left"))
    .group_by(STAY_KEY)
    .agg(
        pl.col("Glasgow coma score total")
        .sort_by(TIME_KEY)
        .min()
        .fill_null(15)
        .alias("Worst GCS prior to T_0")
    )
)

# ------------------------------------------------------------------------------

LABS_BASELINE = (
    LABS_CLEANED.join(COVARIATE_PERIOD, on=STAY_KEY, how="inner")
    .filter(pl.col(TIME_KEY).is_between("T_BASELINE", "T_0", closed="both"))
    .group_by(STAY_KEY)
    .agg(
        pl.col("Lactate").max().alias("Worst Lactate prior to T_0"),
        pl.col("PaO2").min().alias("Worst PaO2 prior to T_0"),
        pl.col("PaCO2").max().alias("Worst PaCO2 prior to T_0"),
    )
)

# ------------------------------------------------------------------------------

DEMOGRAPHICS_BASELINE = DEMOGRAPHICS_CLEANED.with_columns(
    pl.col("Admission Weight (kg)")
    .truediv(pl.col("Admission Height (cm)").truediv(100).pow(2))
    .alias("BMI"),
    (pl.col("Gender") == "Female").cast(int).alias("is Female"),
).drop("Admission Height (cm)", "Admission Weight (kg)")


# region 4.6 wide format
# ------------------------------------------------------------------------------
# 4.6 Push aggregated variable to wide format DF for analyses
# ------------------------------------------------------------------------------
COVARIATES = (
    INCLUDED.join(DEMOGRAPHICS_BASELINE, on=STAY_KEY, how="left")
    .join(SOFA_SCORES, on=STAY_KEY, how="left")
    .join(HOROVITZ_BASELINE, on=STAY_KEY, how="left")
    .join(RESP_FAILURE_BASELINE, on=STAY_KEY, how="left")
    .join(GCS_BASELINE, on=STAY_KEY, how="left")
    .join(LABS_BASELINE, on=STAY_KEY, how="left")
)

# region 5. final cohort
################################################################################
# 5. Select study cohort and create Figure 1 / study flow
print("5. Creating final study cohort")
# ------------------------------------------------------------------------------
STUDY_COHORT = (
    INCLUDED.collect()
    .join(EXPOSURE.collect(), on=STAY_KEY, how="left")
    .join(OUTCOME.collect(), on=STAY_KEY, how="left")
    .join(COVARIATES.collect(), on=STAY_KEY, how="left")
    .unique()
)

# ------------------------------------------------------------------------------
# 5.2 save final study cohort
STUDY_COHORT.sort(STAY_KEY).lazy().sink_parquet("data/STUDY_COHORT.parquet")

# region 5.X flowchart
# ------------------------------------------------------------------------------
# Print complete case cohort size
COMPLETE_CASE_COLUMNS = [
    "Admission Age (years)",
    "is Female",
    "BMI",
    "SOFA Score",
    "is hypoxemic",
    "is hypercapnic",
    "Worst PaO2/FiO2 Ratio prior to T_0",
    "Worst GCS prior to T_0",
    "Worst Lactate prior to T_0",
    "Worst PaO2 prior to T_0",
    "Worst PaCO2 prior to T_0",
]

FINAL_INCLUSION_CRITERIA = [*INCLUSION_CRITERIA, "Exclusion: missing data"]
FINAL_INCLUSION_EXCLUSION = _INCLUSION_EXCLUSION.join(
    STUDY_COHORT.select(
        STAY_KEY,
        pl.all_horizontal(
            pl.col(col).is_not_null() for col in COMPLETE_CASE_COLUMNS
        )
        .fill_null(False)
        .alias("Exclusion: missing data"),
    ),
    on=STAY_KEY,
    how="left",
)

for db, database in [
    ("eicu-", "eICU-CRD"),
    ("mimic3-", "MIMIC-III"),
    ("mimic4-", "MIMIC-IV"),
    ("nwicu-", "NWICU"),
    ("hirid-", "HiRID"),
    ("sicdb-", "SICdb"),
    ("umcdb-", "UMCdb"),
]:
    df = FINAL_INCLUSION_EXCLUSION.filter(
        pl.col(STAY_KEY).str.starts_with(db)
    ).to_pandas()
    for criterion in FINAL_INCLUSION_CRITERIA:
        idxRem = ~df[criterion].fillna(False).astype(bool)
        print(
            "{:6d} - removing {:6d} ({:6.2f}%) patients from {} - {}.".format(
                df.shape[0],
                np.sum(idxRem),
                100.0 * np.mean(idxRem),
                database,
                criterion,
            )
        )
        df = df.loc[~idxRem, :]
    print("{:6d} - final cohort.".format(df.shape[0]), end="\n\n")

# region 6. quality check
################################################################################
# 6. Quality check of study cohort
# Create Data Quality Report based on Y-data profile, assess the following aspects:
# 6.1 Alerts highlighted by the report
# 6.2 Overall cohort size and feasibility, especially focusing on systematic patterns of missigness across years / ICUs / centers
# 6.3 Descriptives for all aggregated variables
# 6.4 Expected interactions and correlation of variables for sanity checks
# 6.5 Patterns of missingness
# 6.6 Select a sample of X observations for front-end double check by clinican (for relevant variables of primary)

# region 7. Table 1
################################################################################
# 7. Create Table 1 across exposure (or outcome) levels and assess carefully with clinicans
print("7. Creating Table 1")
# ------------------------------------------------------------------------------
columns = [
    "Mortality in ICU",
    "Mortality in Hospital",
    "Mechanical Ventilation in first 24h",
    "Admission Age (years)",
    "BMI",
    "Gender",
    "Ethnicity",
    "Source Dataset",
    "SOFA Score",
    "is hypoxemic",
    "is hypercapnic",
]
categorical = [
    "Mortality in ICU",
    "Mortality in Hospital",
    "Mechanical Ventilation in first 24h",
    "Gender",
    "Ethnicity",
    "is hypoxemic",
    "is hypercapnic",
]
continuous = [
    "Admission Age (years)",
    "BMI",
    "SOFA Score",
]
groupby = "Source Dataset"
nonnormal = [
    "Admission Age (years)",
    "SOFA Score",
]
table1 = TableOne(
    STUDY_COHORT.filter(
        pl.all_horizontal(
            pl.col(col).is_not_null() for col in COMPLETE_CASE_COLUMNS
        ).fill_null(False)
    )
    .with_columns(
        pl.when(
            pl.col("Ethnicity").is_in(
                [
                    "Asian",
                    "Black or African American",
                    "Hispanic or Latino",
                    "White",
                ]
            )
        )
        .then(pl.col("Ethnicity"))
        .otherwise(pl.lit("Other"))
        .alias("Ethnicity"),
        pl.col("is hypoxemic").cast(bool),
        pl.col("is hypercapnic").cast(bool),
    )
    .to_pandas(),
    columns=columns,
    categorical=categorical,
    # continuous=continuous,
    groupby=groupby,
    nonnormal=nonnormal,
    # rename=rename,
    pval=False,
    missing=False,
    include_null=True,
    limit={
        "Mortality in ICU": 1,
        "Mortality in Hospital": 1,
        "Mechanical Ventilation in first 24h": 1,
        "is hypoxemic": 1,
        "is hypercapnic": 1,
        "Gender": 1,
        "Ethnicity": 5,
    },
    order={
        "Mortality in ICU": ["True", "False"],
        "Mortality in Hospital": ["True", "False"],
        "Mechanical Ventilation in first 24h": ["True", "False"],
        "is hypoxemic": ["True", "False"],
        "is hypercapnic": ["True", "False"],
        "Gender": ["Female", "Male"],
        "Ethnicity": [
            "Asian",
            "Black or African American",
            "Hispanic or Latino",
            "White",
            "Other",
        ],
    },
)
table1.tableone.replace("0 (nan)", None, inplace=True)
table1.tableone.replace("0 (0.0)", None, inplace=True)
table1.tableone.replace("nan (nan)", None, inplace=True)
table1.tableone.replace("nan [nan,nan]", None, inplace=True)
table1.to_csv("data/table1.csv")

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
with open("data/table1.md", "w") as f:
    f.write(table1_md)

################################################################################
################################################################################
################################################################################


# region 8. IPTW causal outcome model
################################################################################
# 8. Inverse Probability of Treatment Weighting (IPTW) Analysis
print("8. Performing IPTW causal analysis")
# ------------------------------------------------------------------------------
# This section implements a simple propensity score-based IPTW approach to
# estimate the causal effect of mechanical ventilation on ICU mortality,
# following best practices from causal inference.
outcome = "Mortality in ICU"
treatment = "Mechanical Ventilation in first 24h"
covariates = COMPLETE_CASE_COLUMNS

# region 8.1 prepare data for analysis
# Convert to pandas for sklearn compatibility
analysis_df = STUDY_COHORT.with_columns(
    pl.col(outcome).cast(pl.Int32),
    pl.col(treatment).cast(pl.Int32),
).to_pandas()

# Handle missing values: keep only complete cases for propensity score
analysis_df_complete = analysis_df.dropna(
    subset=[outcome, treatment] + covariates
)

print(f"   -> {len(analysis_df_complete)} patients with complete data for IPTW")

# region 8.2 propensity score model
# Fit logistic regression to predict mechanical ventilation (treatment)
# Covariates: demographics, baseline severity
X = analysis_df_complete[covariates].values
T = analysis_df_complete[treatment].values.astype(int)
Y = analysis_df_complete[outcome].values.astype(int)

# Fit propensity score model using statsmodels
X_ps = sm.add_constant(X)
ps_model = sm.Logit(T, X_ps).fit(disp=0)
ps = ps_model.predict(X_ps)  # P(T=1|X)

print(f"   -> propensity score mean: {ps.mean():.2f}, std: {ps.std():.2f}")

# region 8.3 IPTW weight construction (stabilized)
# Marginal probability of treatment
p_treatment = T.mean()  # P(T=1)
p_control = 1 - p_treatment  # P(T=0)

# Calculate stabilized IPTW weights
weights = np.zeros_like(ps, dtype=float)
weights[T == 1] = p_treatment / ps[T == 1]  # P(T=1) / PS for treated
weights[T == 0] = p_control / (1 - ps[T == 0])  # P(T=0) / (1-PS) for control

print(f"   -> weight mean: {weights.mean():.2f}, std: {weights.std():.2f}")
print(f"   -> weight range: [{weights.min():.2f}, {weights.max():.2f}]")
print(
    f"   -> weight percentiles [1, 5, 95, 99]: "
    f"[{np.percentile(weights, 1):.2f}, "
    f"{np.percentile(weights, 5):.2f}, "
    f"{np.percentile(weights, 95):.2f}, "
    f"{np.percentile(weights, 99):.2f}]"
)

# region 8.4 weight truncation (optional but recommended)
# Truncate at 1st and 99th percentiles to reduce influence of extreme weights
p1 = np.percentile(weights, 1)
p99 = np.percentile(weights, 99)
weights_truncated = np.clip(weights, p1, p99)

print(
    "   -> after truncation at [1st, 99th]: "
    f"mean: {weights_truncated.mean():.2f}, "
    f"std: {weights_truncated.std():.2f}"
)

# region 8.5 propensity score positivity check
print("   -> positivity check (PS overlap by treatment group):")
ps_treated = ps[T == 1]
ps_control_group = ps[T == 0]
print(
    f"      Treated (MV=1): mean PS = {ps_treated.mean():.2f}, "
    f"range [{ps_treated.min():.2f}, {ps_treated.max():.2f}]"
)
print(
    f"      Control (MV=0): mean PS = {ps_control_group.mean():.2f}, "
    f"range [{ps_control_group.min():.2f}, {ps_control_group.max():.2f}]"
)

# region 8.6 outcome model: weighted logistic regression with covariates
# Fit logistic regression for mortality with IPTW weights
# Model: logit(Death) = α0 + α1 * MV + α2 * Age + α3 * BMI + α4 * SOFA + ...
analysis_df_complete["IPTW"] = weights
analysis_df_complete["IPTW_truncated"] = weights_truncated
analysis_df_complete["PS"] = ps
analysis_df_complete["MV in first 24h"] = analysis_df_complete[treatment].map({1: "True", 0: "False"}) # fmt: skip
analysis_df_complete.to_csv("data/ANALYSIS_DATA.csv")

# Fit logistic regression with statsmodels
# Using GLM with binomial family for weighted logistic regression
XT = analysis_df_complete[[treatment] + covariates]
XT = sm.add_constant(XT)

outcome_model = sm.GLM(
    Y,
    XT,
    family=sm.families.Binomial(),
    freq_weights=weights_truncated,
).fit()

# Print model summary
print("\n   -> IPTW Outcome Model Summary:")
print(outcome_model.summary())

# Extract treatment effect and CIs directly from model
treatment_coef = outcome_model.params[treatment]
treatment_ci = outcome_model.conf_int().loc[treatment]

# Convert to odds ratio scale
or_estimate = np.exp(treatment_coef)
or_ci_lower = np.exp(treatment_ci[0])
or_ci_upper = np.exp(treatment_ci[1])

print("\n   -> IPTW Outcome Model Results (Mechanical Ventilation Effect):")
print(
    f"      Treatment coefficient (log OR): {treatment_coef:.2f} "
    f"(95% CI: [{treatment_ci[0]:.2f}, {treatment_ci[1]:.2f}])"
)
print(
    f"      Odds Ratio (95% CI): {or_estimate:.2f} "
    f"[{or_ci_lower:.2f}, {or_ci_upper:.2f}]"
)

# Save global estimate results
pl.DataFrame(
    {
        "or_estimate": [or_estimate],
        "or_ci_lower": [or_ci_lower],
        "or_ci_upper": [or_ci_upper],
    }
).write_csv("data/GLOBAL_ESTIMATE.csv")
print("\n   -> saved GLOBAL_ESTIMATE.csv")

print("\n8. IPTW analysis complete.")

# region 9. sensitivity analysis
################################################################################
# 9. Sensitivity Analysis: Database-Specific IPTW Results
print("\n9. Performing sensitivity analysis: database-specific IPTW results")
# ------------------------------------------------------------------------------
# Re-run the IPTW analysis separately for each database to assess consistency
# of treatment effects across different healthcare systems and populations.

DATABASES = ["eICU-CRD", "MIMIC-III", "MIMIC-IV", "HiRID", "AmsterdamUMCdb"]
SENSITIVITY_RESULTS = []

for database in DATABASES:
    print(f"\n   Analyzing {database}...")

    # Filter cohort for current database
    db_cohort = analysis_df_complete
    db_cohort = db_cohort[db_cohort["Source Dataset"] == database]

    n_db = len(db_cohort)
    print(f"   -> {n_db} patients with complete data")

    if n_db < 30:
        print(f"   -> Insufficient sample size for {database}, skipping")
        continue

    # Prepare data for propensity score estimation
    X_db = db_cohort[covariates]
    X_db = sm.add_constant(X_db)
    T_db = db_cohort[treatment].astype(int)
    Y_db = db_cohort[outcome].astype(int)

    # Estimate propensity score
    ps_model = sm.Logit(T_db, X_db).fit(disp=0)
    ps_db = ps_model.predict(X_db)

    # Calculate IPTW
    weights_db = np.where(T_db == 1, 1 / ps_db, 1 / (1 - ps_db))

    # Truncate weights at 1st and 99th percentiles
    p1, p99 = np.percentile(weights_db, [1, 99])
    weights_truncated_db = np.clip(weights_db, p1, p99)
    db_cohort["PS"] = ps_db
    db_cohort["IPTW_truncated"] = weights_truncated_db

    print(
        f"   -> Propensity score mean: {ps_db.mean():.2f}, "
        f"std: {ps_db.std():.2f}"
    )
    print(
        f"   -> Weight range (pre-truncation): "
        f"[{weights_db.min():.2f}, {weights_db.max():.2f}]"
    )
    print(
        f"   -> Weight range (post-truncation): "
        f"[{weights_truncated_db.min():.2f}, {weights_truncated_db.max():.2f}]"
    )

    # Fit outcome model with IPTW
    XT_db = db_cohort[[treatment] + covariates]
    outcome_model = sm.GLM(
        Y_db,
        XT_db,
        family=sm.families.Binomial(),
        freq_weights=weights_truncated_db,
    ).fit()

    # Extract treatment coefficient and OR
    coef = outcome_model.params[treatment]
    se = outcome_model.bse[treatment]
    ci = outcome_model.conf_int().loc[treatment]
    ci_lower, ci_upper = ci[0], ci[1]

    or_point = np.exp(coef)
    or_lower, or_upper = np.exp(ci_lower), np.exp(ci_upper)
    p_value = outcome_model.pvalues[treatment]

    print(
        f"   -> Treatment effect (log OR): {coef:.2f} "
        f"(95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])"
    )
    print(
        f"   -> Odds Ratio (95% CI): {or_point:.2f} "
        f"[{or_lower:.2f}, {or_upper:.2f}], p={p_value:.2f}"
    )

    SENSITIVITY_RESULTS.append(
        {
            "database": database,
            "n": n_db,
            "log_OR": coef,
            "log_OR_95_CI_lower": ci_lower,
            "log_OR_95_CI_upper": ci_upper,
            "OR": or_point,
            "OR_95_CI_lower": or_lower,
            "OR_95_CI_upper": or_upper,
            "p_value": p_value,
            "mean_ps": ps_db.mean(),
            "mean_weight": weights_truncated_db.mean(),
        }
    )

# Save sensitivity analysis results
sensitivity_df = pl.DataFrame(SENSITIVITY_RESULTS)
sensitivity_df.write_csv("data/SENSITIVITY_ANALYSIS_RESULTS.csv")
print("\n   -> saved SENSITIVITY_ANALYSIS_RESULTS.csv")

print("\n9. Sensitivity analysis complete.")
