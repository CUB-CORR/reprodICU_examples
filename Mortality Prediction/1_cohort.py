# Cohort Extraction from reprodICU

# Demographics:
#   Age (years), Gender (1 female / 0 male), ICU mortality,
#   admission weight (kg), admission height (cm), BMI

# Vital Signs:
#   Heart rate (min/mean/max), Systolic blood pressure (min/mean/max),
#   Diastolic blood pressure (min/mean/max), Temperature (min/mean/max)

# Laboratory Results:
#   Hemoglobin (min/max), Hematocrit (min/max), Platelets (min/max), WBC (min/max),
#   Albumin (min/max), Anion gap (min/max), Bicarbonate (min/max), BUN (min/max),
#   Calcium (min/max), Chloride (min/max), Creatinine (min/max), Glucose (min/max),
#   Sodium (min/max), Potassium (min/max), Bilirubin (min/max),
#   Lactate (min/max), pH (min/max), SO2 (min/max), PO2 (min/max),
#   PCO2 (min/max), Base excess (min/max)

# ------------------------------------------------------------------------------

# Source databases: AmsterdamUMCdb, eICU-CRD, HiRID, MIMIC-III, MIMIC-IV, NWICU, SICdb
# Outcome: ICU mortality
# Inclusion criteria: age >= 18, ICU LOS >= 1 day
# Coverage: first 24 hours of ICU admission

################################################################################

import os

import numpy as np
import polars as pl
import reprodICU
import tableone

# region constants
# ==============================================================================
SECONDS_IN_6_HOURS  =  6 * 60 * 60
SECONDS_IN_12_HOURS = 12 * 60 * 60
SECONDS_IN_24_HOURS = 24 * 60 * 60
N_BOOTSTRAPS = 1000  # Number of bootstrap samples
CONFIDENCE_LEVEL = 0.95  # Confidence level for CI

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"

# Ensure output directory exists
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# X. lazy-load the datasets
info: pl.LazyFrame = reprodICU.patient_information
labs: pl.LazyFrame = reprodICU.timeseries_labs
vitals: pl.LazyFrame = reprodICU.timeseries_vitals

# region inclusion
# ==============================================================================
CASE_IDS = info.select(STAY_KEY, "Source Dataset").unique()
AGES = info.select(STAY_KEY, "Admission Age (years)")
ICU_LENGTH_OF_STAY = info.select(STAY_KEY, "ICU Length of Stay (days)")
SOURCE_DATABASE = info.select(STAY_KEY, "Source Dataset")
VITALS_AVAILABLE = info.select(STAY_KEY, "Table: Timeseries (Vitals)")
MORTALITY_AVAILABLE = info.select(STAY_KEY, "Mortality in ICU")

# 1.4.I.1 Age >= 18
INCLUSION_EXCLUSION = CASE_IDS.join(AGES, on=STAY_KEY, how="left").with_columns(
    pl.col("Admission Age (years)").ge(18).alias("Inclusion: Age >= 18")
)

# 1.4.I.2 ICU Length of Stay >= 1 days
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    ICU_LENGTH_OF_STAY, on=STAY_KEY, how="left"
).with_columns(
    # The LoS calculations
    # - in MIMIC-III have millisecond precision
    # - in UMCdb also have millisecond precision
    pl.when(
        pl.col(STAY_KEY).str.starts_with("mimic3")
        | pl.col(STAY_KEY).str.starts_with("umcdb")
    )
    .then(pl.col("ICU Length of Stay (days)"))
    # - in MIMIC-IV are rounded to two decimal places
    # - in eICU-CRD are rounded to full hours
    .otherwise(pl.col("ICU Length of Stay (days)").round(2))
    .ge(1)
    .alias("Inclusion: ICU LOS >= 1 days")
)

# 1.4.E.1 Has no vital signs
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    VITALS_AVAILABLE, on=STAY_KEY, how="left"
).with_columns(
    # in MIMIC-III, patients with no vitals explicitly excluded
    # -> `WHERE adm.has_chartevents_data = 1` in L102 in mimic-iii/concepts/icustay_detail.sql
    pl.when(pl.col(STAY_KEY).str.starts_with("mimic3"))
    .then(pl.col("Table: Timeseries (Vitals)").is_null())
    # in MIMIC-IV and eICU-CRD, those are not explicitly excluded
    .otherwise(False)
    .alias("Exclusion: Has no vital signs")
)

# 1.4.E.2 ICU mortality outcome undocumented (cannot be used as label)
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    MORTALITY_AVAILABLE, on=STAY_KEY, how="left"
).with_columns(
    (
        pl.col("Mortality in ICU").is_null()
        # HiRID & UMCdb explicitly treat patients with undocumented mortality as survivors
        & ~pl.col(STAY_KEY).str.starts_with("umcdb")
        & ~pl.col(STAY_KEY).str.starts_with("hirid")
        # eICU-CRD sometimes has nulls in ICU mortality, but known hospital survival
        # -> reprodICU infers ICU survival for these patients based on hospital survival
        # -> but these patients are excluded in the SQL cohort definition, so must be manually excluded here as well
        | pl.col(STAY_KEY).str.contains_any([
            "eicu-640242",  "eicu-1016442", "eicu-1047049", "eicu-1199340",
            "eicu-1723722", "eicu-2390466",
        ])
    ).alias("Exclusion: No ICU mortality documented")
) # fmt: skip

INCLUSION_CRITERIA = ["Inclusion: Age >= 18", "Inclusion: ICU LOS >= 1 days"]
EXCLUSION_CRITERIA = ["Exclusion: Has no vital signs", "Exclusion: No ICU mortality documented"] # fmt: skip

INCLUDED = (
    INCLUSION_EXCLUSION.unique()
    .filter(
        pl.all_horizontal(*INCLUSION_CRITERIA),
        ~pl.any_horizontal(*EXCLUSION_CRITERIA),
    )
    .select(STAY_KEY)
    .unique()
)

print(
    INCLUDED.join(SOURCE_DATABASE, on=STAY_KEY, how="left")
    .group_by("Source Dataset")
    .len()
    .sort("Source Dataset")
    .collect()
)

# region cohort
# ==============================================================================
# Demographics + Body Mass Index & Weight
demographics = (
    info.join(INCLUDED, on=STAY_KEY, how="semi")
    .with_columns(
        pl.when(pl.col("Admission Height (cm)").is_between(50, 250))
        .then(pl.col("Admission Height (cm)"))
        .otherwise(None)
        .alias("Admission Height (cm)")
    )
    .with_columns(
        pl.col("Admission Weight (kg)")
        .truediv(pl.col("Admission Height (cm)").truediv(100).pow(2))
        .round(1)
        .alias("BMI")
    )
    .select(
        STAY_KEY,
        "Source Dataset",
        # Explicitly treat null ICU mortality as False (survivors),
        # as per the data documentation for HiRID and UMCdb
        pl.col("Mortality in ICU").fill_null(False).cast(bool),
        pl.col("Mortality in Hospital").cast(bool),
        pl.col("Gender").eq("Female").cast(bool).alias("isfemale"),
        "Admission Age (years)",
        "BMI",
        "Admission Weight (kg)",
    )
)

# Vital Signs
vital_signs = (
    vitals.filter(
        pl.col(TIME_KEY).is_between(-SECONDS_IN_6_HOURS, SECONDS_IN_24_HOURS)
    )
    .select(
        STAY_KEY,
        "Heart rate",
        "Temperature",
        pl.coalesce(
            pl.col("Invasive systolic arterial pressure"),
            pl.col("Non-invasive systolic arterial pressure"),
        ).alias("Systolic blood pressure"),
        pl.coalesce(
            pl.col("Invasive diastolic arterial pressure"),
            pl.col("Non-invasive diastolic arterial pressure"),
        ).alias("Diastolic blood pressure"),
    )
    .with_columns(
        pl.when(pl.col(vital_sign).is_between(low, high))
        .then(pl.col(vital_sign))
        .otherwise(None)
        .alias(vital_sign)
        for vital_sign, low, high in [
            ("Heart rate", 0, 300),
            ("Temperature", 25, 45),
            ("Systolic blood pressure", 40, 320),
            ("Diastolic blood pressure", 20, 200),
        ]
    )
    # Group by stay ID and aggregate: min, mean, max for each vital sign
    .group_by(STAY_KEY)
    .agg(
        expr.alias(f"{vital_sign} ({suffix})")
        for vital_sign in [
            "Heart rate",
            "Temperature",
            "Systolic blood pressure",
            "Diastolic blood pressure",
        ]
        for expr, suffix in (
            (pl.col(vital_sign).min(), "min"),
            (pl.col(vital_sign).max(), "max"),
            (pl.col(vital_sign).mean(), "mean"),
        )
    )
)


# Laboratory Results
LABS = [
    # BLOOD COUNTS
    "Hemoglobin",
    "Erythrocyte/Blood",  # Hematocrit
    "Platelets",
    "Leukocytes",
    # CHEMISTRY
    "Albumin",
    "Anion gap",
    "Bicarbonate",
    "Urea nitrogen",
    "Calcium",
    "Chloride",
    "Glucose",
    "Potassium",
    "Sodium",
    "Creatinine",
    "Bilirubin",
    # BLOOD GASES
    "Lactate",
    "Oxygen",
    "Oxygen saturation",
    "Carbon dioxide",
    "pH",
    "Base excess",
]
LABS_hct = [x for x in LABS if x != "Erythrocyte/Blood"] + ["Hematocrit"]

laboratory_results = (
    labs.filter(
        pl.col(TIME_KEY).is_between(-SECONDS_IN_6_HOURS, SECONDS_IN_24_HOURS)
    )
    # Select and extract 'value' from struct, filtering by 'system'
    .select(
        STAY_KEY,
        TIME_KEY,
        *[
            pl.when(
                pl.col(lab).struct.field("system").ne_missing("Urine")
                | pl.col(lab).struct.field("system").is_null()
            )
            .then(pl.col(lab).struct.field("value"))
            .otherwise(None)
            .alias(lab)
            for lab in LABS
        ],
    )
    .rename({"Erythrocyte/Blood": "Hematocrit"})
    # Group by stay ID and aggregate: min, max for each lab
    .group_by(STAY_KEY)
    .agg(
        *[pl.col(lab).min().alias(f"{lab} (min)") for lab in LABS_hct],
        *[pl.col(lab).max().alias(f"{lab} (max)") for lab in LABS_hct],
    )
)

# Join all dataframes
icu_mortality = (
    demographics.join(vital_signs, on=STAY_KEY, how="left")
    .join(laboratory_results, on=STAY_KEY, how="left")
    .sort(STAY_KEY)
)

# Save the result
icu_mortality.sink_parquet(f"{output_dir}/cohort.parquet")

################################################################################
################################################################################

# region flowchart
# ==============================================================================
print("\n=== STUDY FLOWCHART ===")

INCLUSION_CRITERIA_FLOWCHART = INCLUSION_CRITERIA + EXCLUSION_CRITERIA

# Build inclusion/exclusion tracking using the finalized INCLUSION_EXCLUSION
# Use the pre-computed inclusion/exclusion flags and attach the source database.
flowchart_data = INCLUSION_EXCLUSION.unique().sort(STAY_KEY).collect()
df = flowchart_data.to_pandas()

# Overall flowchart
print("\nAll databases:")
print(f"{len(df):6d} - All patients")
for criterion in INCLUSION_CRITERIA_FLOWCHART:
    if criterion in EXCLUSION_CRITERIA:
        idxRem = df[criterion] == True  # Exclusion criterion: patients with True are removed
    else:
        idxRem = ~df[criterion].fillna(False)  # Inclusion criterion: False or NaN → excluded

    print(
        "{:6d} - {:6d} ({:5.2f}%) patients excluded - {}.".format(
            len(df) - np.sum(idxRem),
            np.sum(idxRem),
            100.0 * np.mean(idxRem),
            criterion,
        )
    )
    df = df.loc[~idxRem, :]
print("{:6d} - final cohort.\n".format(df.shape[0]), end="\n")

# Flowchart by major databases
SOURCE_DATABASE_COLLECTED = SOURCE_DATABASE.collect()
for db_name in [
    db
    for db in sorted(
        SOURCE_DATABASE_COLLECTED.get_column("Source Dataset")
        .unique()
        .to_list(),
        key=str.lower,
    )
]:
    df = (
        flowchart_data.join(SOURCE_DATABASE_COLLECTED, on=STAY_KEY, how="left")
        .filter(pl.col("Source Dataset") == db_name)
        .to_pandas()
    )

    print(f"{db_name}:")
    print(f"{len(df):6d} - All patients")
    for criterion in INCLUSION_CRITERIA_FLOWCHART:
        if criterion in EXCLUSION_CRITERIA:
            idxRem = df[criterion].fillna(False)  # Exclusion criterion: patients with True are removed
        else:
            idxRem = ~df[criterion].fillna(False) # Inclusion criterion: patients with False or NaN are removed

        print(
            "{:6d} - {:6d} ({:5.2f}%) patients excluded from {} - {}.".format(
                len(df) - np.sum(idxRem),
                np.sum(idxRem),
                100.0 * np.mean(idxRem),
                db_name,
                criterion,
            )
        )
        df = df.loc[~idxRem, :]
    print("{:6d} - final cohort.\n".format(df.shape[0]), end="\n")

################################################################################
################################################################################

# region table1
# ==============================================================================
table1 = info.select(
    STAY_KEY,
    "Source Dataset",
    "Mortality in ICU",
    "Mortality in Hospital",
    "Gender",
    "Admission Age (years)",
    "Admission Type",
    "Admission Urgency",
    pl.when(
        pl.col("Ethnicity").is_in([
            "Asian",
            "Black or African American",
            "Hispanic or Latino",
            "White",
        ])
    )
    .then(pl.col("Ethnicity"))
    .otherwise(pl.lit("Other"))
    .alias("Ethnicity"),
)

columns = [
    "Admission Age (years)",
    "Gender",
    "Ethnicity",
    "Admission Type",
    "Admission Urgency",
    "Mortality in ICU",
    "Mortality in Hospital",
]
categorical = [
    "Gender",
    "Ethnicity",
    "Admission Type",
    "Admission Urgency",
    "Mortality in ICU",
    "Mortality in Hospital",
]
continuous = [
    "Admission Age (years)",
]
groupby = "Source Dataset"
nonnormal = []
table1 = tableone.TableOne(
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
        "Gender": 1,
        "Ethnicity": 5,
        "Admission Type": 3,
        "Admission Urgency": 4,
        "Mortality in ICU": 1,
        "Mortality in Hospital": 1,
    },
    order={
        "Gender": ["Female", "Male"],
        "Ethnicity": [
            "Asian",
            "Black or African American",
            "Hispanic or Latino",
            "White",
            "Other",
        ],
        "Admission Type": ["Medical", "Surgical", "Other"],
        "Admission Urgency": ["Elective", "Urgent", "Emergency", "Other"],
        "Mortality in ICU": ["True", "False"],
        "Mortality in Hospital": ["True", "False"],
    },
)

# Save table1 to file
# Save Table 1 to CSV and print
table1_output_file = f"{output_dir}/icu_mortality_table1.csv"
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
with open(f"{output_dir}/icu_mortality_table1.md", "w") as f:
    f.write(table1_md)
