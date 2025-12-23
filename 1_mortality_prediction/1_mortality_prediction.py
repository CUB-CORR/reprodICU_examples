# Case study I: Development and external validation of an in-ICU mortality
# model for ICU patients
################################################################################

# Demographics:
#   Gender (1 for female, 0 for male), Admission Age (years), BMI,
#   Admission weight (kg)

# Vital Signs:
#   Heart rate (Q10/Q50/Q90), Systolic blood pressure (Q10/Q50/Q90),
#   Diastolic blood pressure (Q10/Q50/Q90), Temperature (Q10/Q50/Q90)

# Laboratory Results:
#   Albumin, Alkaline phosphatase, Aspartate aminotransferase, Bilirubin,
#   Glucose, Protein, Hemoglobin, Leukocytes, Platelets,
#   Lymphocytes/leukocytes, Monocytes/leukocytes, Neutrophils/leukocytes,
#   Calcium, Chloride, Potassium, Sodium, Oxygen, Carbon dioxide, pH,
#   Bicarbonate, Base excess

# ------------------------------------------------------------------------------

# Model: LightGBM Classifier with hyperparameter tuning via GridSearchCV
# Outcome: ICU mortality
# Covariates: demographics, vital signs, and lab results listed above

################################################################################

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import reprodICU
import shap
import tableone
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common import apply_plot_style

apply_plot_style()

# region constants
# ------------------------------------------------------------------------------
SECONDS_IN_24_HOURS = 24 * 60 * 60
N_BOOTSTRAPS = 1000  # Number of bootstrap samples
CONFIDENCE_LEVEL = 0.95  # Confidence level for CI

STAY_KEY = "Global ICU Stay ID"
TIME_KEY = "Time Relative to Admission (seconds)"

# Ensure output directory exists
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# region lazyload
# ------------------------------------------------------------------------------
# X. lazy-load the datasets
info: pl.LazyFrame = reprodICU.patient_information
labs: pl.LazyFrame = reprodICU.timeseries_labs
vitals: pl.LazyFrame = reprodICU.timeseries_vitals

# region inclusion
# ------------------------------------------------------------------------------
CASE_IDS = info.select(STAY_KEY, "Source Dataset").unique()
AGES = info.select(STAY_KEY, "Admission Age (years)")
ICU_LENGTH_OF_STAY = info.select(STAY_KEY, "ICU Length of Stay (days)")
SOURCE_DATABASE = info.select(STAY_KEY, "Source Dataset")

# 1.4.I.1 Age >= 18
INCLUSION_EXCLUSION = CASE_IDS.join(AGES, on=STAY_KEY, how="left").with_columns(
    pl.col("Admission Age (years)").ge(18).alias("Inclusion: Age > 18")
)

# 1.4.I.2 ICU Length of Stay >= 1 days
INCLUSION_EXCLUSION = INCLUSION_EXCLUSION.join(
    ICU_LENGTH_OF_STAY, on=STAY_KEY, how="left"
).with_columns(
    pl.col("ICU Length of Stay (days)")
    .ge(1)
    .alias("Inclusion: ICU LOS >= 1 days")
)

INCLUSION_CRITERIA = [
    "Inclusion: Age > 18",
    "Inclusion: ICU LOS >= 1 days",
]

INCLUDED = (
    INCLUSION_EXCLUSION.unique()
    .filter(pl.all_horizontal(*INCLUSION_CRITERIA))
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
# ------------------------------------------------------------------------------
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
        .cast(float)
        .alias("BMI")
    )
    .select(
        STAY_KEY,
        "Source Dataset",
        "Mortality in ICU",
        pl.col("Gender").eq("Female").cast(bool).alias("isfemale"),
        "Admission Age (years)",
        "BMI",
        "Admission Weight (kg)",
    )
)

# Vital Signs
vital_signs = (
    vitals.filter(pl.col(TIME_KEY).is_between(0, SECONDS_IN_24_HOURS))
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
    .group_by(STAY_KEY)
    .agg(
        pl.col(vital_sign).quantile(q).alias(f"{vital_sign} ({suffix})")
        for vital_sign in [
            "Heart rate",
            "Temperature",
            "Systolic blood pressure",
            "Diastolic blood pressure",
        ]
        for suffix, q in {"Q10": 0.1, "Q50": 0.5, "Q90": 0.9}.items()
    )
)


# Laboratory Results
# Common systems for blood-related lab tests
BLOOD_SYSTEMS = ["Blood", "Blood arterial", "Serum or Plasma", None]
LABS = [
    "Albumin",
    "Alkaline phosphatase",
    "Aspartate aminotransferase",
    "Bilirubin",
    "Glucose",
    "Protein",
    # BLOOD COUNTS
    "Hemoglobin",
    "Leukocytes",
    "Platelets",
    "Lymphocytes/leukocytes",
    "Monocytes/leukocytes",
    "Neutrophils/leukocytes",
    # ELECTROLYTES
    "Calcium",
    "Chloride",
    "Potassium",
    "Sodium",
    # BLOOD GASES
    "Oxygen",
    "Carbon dioxide",
    "pH",
    "Bicarbonate",
    "Base excess",
]

laboratory_results = (
    labs.filter(pl.col(TIME_KEY).is_between(0, SECONDS_IN_24_HOURS))
    # Select and extract 'value' from struct, filtering by 'system'
    .select(
        STAY_KEY,
        TIME_KEY,
        *[
            pl.when(
                pl.col(lab).struct.field("system").is_in(BLOOD_SYSTEMS)
                | pl.col(lab).struct.field("system").is_null()
            )
            .then(pl.col(lab).struct.field("value"))
            .otherwise(None)
            .alias(lab)
            for lab in LABS
        ],
    )
    # Group by stay ID and aggregate
    .group_by(STAY_KEY).agg(
        pl.col(lab).sort_by(TIME_KEY).first() for lab in LABS
    )
)

# Join all dataframes
icu_mortality = demographics.join(vital_signs, on=STAY_KEY, how="left").join(
    laboratory_results, on=STAY_KEY, how="left"
)

# Save the result
icu_mortality.sink_parquet(f"{output_dir}/icu_mortality_features.parquet")

################################################################################
################################################################################

# region flowchart
# ------------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STUDY FLOWCHART")
print("=" * 80)

INCLUSION_CRITERIA_FLOWCHART = [
    "Inclusion: Age > 18",
    "Inclusion: ICU LOS >= 1 days",
]

# Build inclusion/exclusion tracking
flowchart_data = (
    CASE_IDS.join(AGES, on=STAY_KEY, how="left")
    .join(ICU_LENGTH_OF_STAY, on=STAY_KEY, how="left")
    .with_columns(
        (pl.col("Admission Age (years)") >= 18).alias("Inclusion: Age > 18"),
        (pl.col("ICU Length of Stay (days)") >= 1).alias("Inclusion: ICU LOS >= 1 days"),
    )
) # fmt: skip

flowchart_data = flowchart_data.collect()
df = flowchart_data.to_pandas()

# Overall flowchart
print("\nAll databases:")
print(f"{len(df):6d} - All patients")
for criterion in INCLUSION_CRITERIA_FLOWCHART:
    idxRem = df[criterion] == False
    print(
        "{:6d} - {:6d} ({:5.2f}%) patients excluded - {}.".format(
            df[criterion].sum(),
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
        idxRem = df[criterion] == False
        print(
            "{:6d} - {:6d} ({:5.2f}%) patients excluded from {} - {}.".format(
                df[criterion].sum(),
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
# ------------------------------------------------------------------------------
table1 = info.select(
    STAY_KEY,
    "Source Dataset",
    "Mortality in ICU",
    "Gender",
    "Admission Age (years)",
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
)

columns = [
    "Admission Age (years)",
    "Gender",
    "Ethnicity",
    "Mortality in ICU",
]
categorical = [
    "Gender",
    "Ethnicity",
    "Mortality in ICU",
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
        "Mortality in ICU": 1,
        "Gender": 1,
        "Ethnicity": 5,
    },
    order={
        "Mortality in ICU": ["True", "False"],
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

################################################################################
################################################################################

# region missingness
# ------------------------------------------------------------------------------
print("Calculating missingness table...")

# Load the previously created feature table
missingness = pl.read_parquet(f"{output_dir}/icu_mortality_features.parquet")

# Define columns to check for missingness (exclude identifiers and outcome)
columns_to_check = [
    col
    for col in missingness.columns
    if col not in [STAY_KEY, "Source Dataset", "Mortality in ICU"]
]

# Calculate missing percentage for each feature, grouped by Source Dataset
missing_stats = missingness.group_by("Source Dataset").agg(
    (pl.col(c).is_null().mean() * 100).round(1).alias(c)
    for c in columns_to_check
)

# Unpivot to long format
missingness_table = (
    missing_stats.unpivot(
        index="Source Dataset", variable_name="Feature", value_name="Missing %"
    )
    # Pivot to wide format with features as rows
    .pivot(index="Feature", on="Source Dataset", values="Missing %")
)

# Sort datasets alphabetically
db_cols = sorted([col for col in missingness_table.columns if col != "Feature"])
missingness_table = missingness_table.select(["Feature"] + db_cols)

# Add summary statistics (mean and median across features for each dataset)
mean_missing = missingness_table.select(db_cols).mean()
mean_labs_missing = missingness_table.filter(pl.col("Feature").is_in(LABS)).select(db_cols).mean() # fmt: skip
median_missing = missingness_table.select(db_cols).median()

mean_row = pl.DataFrame(
    {
        "Feature": "Mean % Missing",
        **{col: mean_missing[col] for col in db_cols},
    }
)
mean_lab_row = pl.DataFrame(
    {
        "Feature": "Mean Labs % Missing",
        **{col: mean_labs_missing[col] for col in db_cols},
    }
)
median_row = pl.DataFrame(
    {
        "Feature": "Median % Missing",
        **{col: median_missing[col] for col in db_cols},
    }
)
missingness_table = pl.concat([missingness_table, mean_lab_row, mean_row, median_row]) # fmt: skip

# Save missingness table to CSV and markdown
output_file = f"{output_dir}/icu_mortality_missingness_table.csv"
missingness_table.write_csv(output_file)
missingness_table.to_pandas().to_markdown(
    f"{output_dir}/icu_mortality_missingness_table.md", index=False
)
print(f"\nMissingness table saved to {output_file}")

################################################################################
################################################################################

# region model training
# ------------------------------------------------------------------------------

# LightGBM model training and evaluation per DB
# ------------------------------------------------------------------------------
# Load the dataset
df = pl.read_parquet(f"{output_dir}/icu_mortality_features.parquet")
# Filter out rows with missing target values
df = df.filter(pl.col("Mortality in ICU").is_not_null())

# Define the features and target variable
features = [
    col
    for col in df.columns
    if col not in [STAY_KEY, "Source Dataset", "Mortality in ICU"]
]
target = "Mortality in ICU"

# Separate MIMIC-IV data for training and other data for testing
df_mimic_all = df.filter(pl.col("Source Dataset") == "MIMIC-IV")
df_external_test = df.filter(pl.col("Source Dataset") != "MIMIC-IV")

# Split MIMIC-IV data: 80% train, 20% test
indices = np.arange(df_mimic_all.height)
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=df_mimic_all.select(target).to_series().to_numpy(),
)

df_train = df_mimic_all[train_idx]
df_mimic_test = df_mimic_all[test_idx]

X_train = df_train.select(features).to_pandas().values
y_train = df_train.select(target).to_series().to_numpy()

print(f"MIMIC-IV Training set shape: {X_train.shape}, {y_train.shape}")
print(f"MIMIC-IV Test set shape: ({df_mimic_test.height}, {len(features)})") # fmt: skip
print(f"External Test set sources: {df_external_test['Source Dataset'].unique().to_list()}") # fmt: skip

# Calculate prevalence from MIMIC-IV training set for DCA threshold
prevalence_mimic_iii_train = y_train.sum() / len(y_train)
print(f"MIMIC-IV Training Set Prevalence (for DCA threshold): {prevalence_mimic_iii_train:.3f}") # fmt: skip

# Save prevalence for plotting
pl.DataFrame({"prevalence": [prevalence_mimic_iii_train]}).write_csv(
    f"{output_dir}/icu_mortality_prevalence.csv"
)

# Set up LightGBM with hyperparameter tuning
param_grid = {
    "reg_lambda": [0, 5, 10],
    "reg_alpha": [0, 0.5, 1],
    "min_split_gain": [0, 0.05, 0.1],
}

# Define the base model
base_model = LGBMClassifier(
    objective="binary",
    metric="binary_logloss",
    num_leaves=31,
    n_estimators=100,
    learning_rate=0.1,
    min_child_samples=50,
    feature_fraction=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# Set up GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=10,
)

# Train the model
print("Starting hyperparameter optimization...")
grid_search.fit(X_train, y_train, feature_name=features)

# Use the best model
model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best ROC AUC score: {grid_search.best_score_:.4f}")

# Save Model
model_filename_txt = f"{output_dir}/lgbm_model.txt"
model_filename_pkl = f"{output_dir}/lgbm_model.pkl"

with open(model_filename_pkl, "wb") as f:
    pickle.dump(model, f)
model.booster_.save_model(model_filename_txt)
print(f"Model saved as {model_filename_txt} and {model_filename_pkl}")


# region SHAP values
# ------------------------------------------------------------------------------
print("Extracting SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Save SHAP values and training data for plotting
with open(f"{output_dir}/shap_values.pkl", "wb") as f:
    pickle.dump(
        {"shap_values": shap_values, "X_train": X_train, "features": features},
        f,
    )
print(f"SHAP values saved to {output_dir}/shap_values.pkl")


# region helpers
# ------------------------------------------------------------------------------
def bootstrap_auc(y_test, y_pred_proba, n_bootstraps=N_BOOTSTRAPS, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    # Ensure y_test and y_pred_proba are numpy arrays for proper indexing
    y_test_np = np.asarray(y_test)
    y_pred_proba_np = np.asarray(y_pred_proba)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred_proba_np), len(y_pred_proba_np))
        y_test_sample = y_test_np[indices]
        y_pred_proba_sample = y_pred_proba_np[indices]
        score = roc_auc_score(y_test_sample, y_pred_proba_sample)
        bootstrapped_scores.append(score)

    if not bootstrapped_scores:  # If all bootstrap samples were skipped
        return np.nan, np.nan

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    ci_cutoff_lower = ((1 - CONFIDENCE_LEVEL) / 2) * 100
    confidence_lower = np.percentile(sorted_scores, ci_cutoff_lower)
    ci_cutoff_upper = (CONFIDENCE_LEVEL + (1 - CONFIDENCE_LEVEL) / 2) * 100
    confidence_upper = np.percentile(sorted_scores, ci_cutoff_upper)

    return confidence_lower, confidence_upper


def calculate_ici(y_true, y_prob):
    """
    Calculate the Integrated Calibration Index (ICI).

    The ICI is the weighted average absolute difference between observed and
    predicted probabilities, weighted by the density of predicted probabilities.
    """
    # LOWESS smoothing to estimate the calibration curve (xc)
    smoothed = lowess(y_true, y_prob, frac=0.3, it=0, return_sorted=False)
    # Mean absolute difference between predicted (x) and smoothed observed (xc)
    return np.mean(np.abs(y_prob - smoothed))


def create_calibration_data(y_true, y_prob, db_name, n_bins=10):
    df_cal = pl.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df_cal = df_cal.with_columns(
        (pl.col("y_prob") * n_bins).floor().clip(0, n_bins - 1).alias("bin")
    )

    return (
        df_cal.group_by("bin")
        .agg(
            pl.col("y_prob").mean().alias("predicted_prob"),
            pl.col("y_true").mean().alias("actual_prob"),
            pl.col("y_true").count().alias("n"),
        )
        .sort("bin")
        .with_columns(
            (
                pl.col("actual_prob")
                * (1 - pl.col("actual_prob"))
                / pl.when(pl.col("n") == 0).then(1).otherwise(pl.col("n"))
            )
            .sqrt()
            .alias("se")
        )
        .with_columns(
            lower_ci=(pl.col("actual_prob") - 1.96 * pl.col("se")).clip(0, 1),
            upper_ci=(pl.col("actual_prob") + 1.96 * pl.col("se")).clip(0, 1),
            db=pl.lit(db_name),
        )
        .filter(pl.col("predicted_prob").is_not_null())
    )


def calculate_net_benefit(y_true, y_prob, db_name):
    """
    Calculate the Net Benefit for Decision Curve Analysis (DCA).

    Net Benefit is the proportion of true positives minus the proportion of
    false positives, weighted by the relative harm of a false positive.
    """
    thresholds = np.linspace(0, 1, 101)
    net_benefit_data = []

    for thr in thresholds:
        if thr == 1.0:
            # Net benefit is typically 0 at threshold 1.0 as no one is treated
            net_benefit = 0.0
        else:
            y_pred_treat = (y_prob >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_treat).ravel()
            n = len(y_true)

            net_benefit = (tp / n) - (fp / n) * (thr / (1 - thr))

        net_benefit_data.append(
            {
                "threshold": round(thr, 2),
                "net_benefit": net_benefit,
                "db_name": db_name,
            }
        )

    return pl.DataFrame(net_benefit_data)


# region evaluation
# ------------------------------------------------------------------------------
test_dbs_external = df_external_test["Source Dataset"].unique().to_list()
test_dbs_for_evaluation = sorted(test_dbs_external + ["MIMIC-IV"], key=str.lower) # fmt: skip

roc_data_all = []
cal_data_all = []
dca_data_all = []
metrics_all = []

# Evaluate on Test DBs and MIMIC-IV Test Set
for db in test_dbs_for_evaluation:
    print(f"Evaluating on DB: {db}")

    X_test: np.ndarray
    y_test: np.ndarray

    if db == "MIMIC-IV":
        # Use the held-out MIMIC-IV test set for evaluation
        X_test = df_mimic_test.select(features).to_pandas().values
        y_test = df_mimic_test.select(target).to_series().to_numpy()
        print("  Using held out set for MIMIC-IV evaluation")
    else:
        # Process external test DBs
        df_db = df_external_test.filter(pl.col("Source Dataset") == db)
        X_test = df_db.select(features).to_pandas().values
        y_test = df_db.select(target).to_series().to_numpy()

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    # Bootstrap CI for AUC
    auc_ci_lower, auc_ci_upper = bootstrap_auc(y_test, y_pred_proba)

    roc_data_db = pl.DataFrame({"fpr": fpr, "tpr": tpr, "db": db})
    roc_data_all.append(roc_data_db)

    cal_data_db = create_calibration_data(y_test, y_pred_proba, db_name=db)
    cal_data_all.append(cal_data_db)

    brier_score = brier_score_loss(y_test, y_pred_proba)
    ici_score = calculate_ici(y_test, y_pred_proba)

    dca_data_db = calculate_net_benefit(y_test, y_pred_proba, db_name=db)
    dca_data_all.append(dca_data_db)

    # Find net benefit at prevalence_mimic_iii_train threshold
    # Find the index of the threshold closest to prevalence_mimic_iii_train
    closest_threshold_idx = (
        (
            dca_data_db.select("threshold").to_series()
            - prevalence_mimic_iii_train
        )
        .abs()
        .to_numpy()
        .argmin()
    )
    nb_at_prev_threshold = dca_data_db["net_benefit"][int(closest_threshold_idx)]  # fmt: skip

    metrics_all.append(
        {
            "DBName": db,
            "ROC AUC": roc_auc,
            "AUC CI Lower": auc_ci_lower,
            "AUC CI Upper": auc_ci_upper,
            "PR AUC": pr_auc,
            "Brier Score": brier_score,
            "ICI": ici_score,
            "NB@Prevalence": nb_at_prev_threshold,
        }
    )

roc_df = pl.concat(roc_data_all)
cal_df = pl.concat(cal_data_all)
dca_df = pl.concat(dca_data_all)
metrics_df = pl.DataFrame(metrics_all)

# Save evaluation data for plotting
roc_df.write_parquet(f"{output_dir}/icu_mortality_roc_data.parquet")
cal_df.write_parquet(f"{output_dir}/icu_mortality_cal_data.parquet")
dca_df.write_parquet(f"{output_dir}/icu_mortality_dca_data.parquet")

metrics_output_file = f"{output_dir}/icu_mortality_db_metrics.csv"
metrics_df.write_csv(metrics_output_file)
print("\nEvaluation data saved for plotting.")
print(f"Aggregated DB metrics saved to {metrics_output_file}")

# Export Net Benefit table
dca_output_file = f"{output_dir}/icu_mortality_net_benefit_table.csv"
(
    dca_df.pivot(index="threshold", on="db_name", values="net_benefit")
    .sort("threshold")
    .with_columns(
        pl.col(db).round_sig_figs(2).alias(db) for db in test_dbs_for_evaluation
    )
    .write_csv(dca_output_file)
)
print(f"Net Benefit table saved to {dca_output_file}")

pl.Config.set_float_precision(2)
print(metrics_df)
