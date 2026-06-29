# ICU Mortality Prediction Model Fitting and Evaluation

# Model Hierarchy:
#   1. Single-source:  train and test on each database individually
#   2. Two-source:     American–European cross-training pairs (harmonized cohort only)
#   3. LODO:           leave-one-database-out; train on N-1, test on held-out

# Databases:
#   American — eICU-CRD, MIMIC-III, MIMIC-IV, NWICU
#   European — AmsterdamUMCdb, HiRID, SICdb

# ------------------------------------------------------------------------------

# Model: LightGBM Classifier
# Outcome: ICU mortality
# Performance: AUROC, Brier score, ICI (all with 95% bootstrap CI)

################################################################################

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import reprodICU
from fastlowess import Lowess
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from _constants import (
    DATA_REGION,
    EXCLUDE_COLS,
    MORT_KEY,
    OUTPUT_DIR,
    OVERLAP_SOURCES,
    RANDOM_STATE,
    SOURCE_KEY,
    STAY_KEY,
)

# region constants
# ==============================================================================
TEST_SIZE = 0.2

N_BOOTSTRAPS     = 2000
CONFIDENCE_LEVEL = 0.95

_BOOTSTRAP_KEYS = {"auroc_bootstrap_values", "brier_bootstrap_values"}
_STAY_IDS_CACHE = {}  # Module-level cache: harmonized populates, original reads


# region utilities
# ==============================================================================
def _auroc_iter(y_test, y_pred_proba, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_test), size=len(y_test), replace=True)
    y_b = y_test[idx]
    return roc_auc_score(y_b, y_pred_proba[idx]) if len(np.unique(y_b)) > 1 else np.nan


def bootstrap_auc(y_test, y_pred_proba):
    """Calculate AUROC with bootstrap confidence intervals."""
    y_test       = np.asarray(y_test,       dtype=float).ravel()
    y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()

    point  = roc_auc_score(y_test, y_pred_proba)
    seeds  = np.random.default_rng(RANDOM_STATE).integers(0, 2**31, N_BOOTSTRAPS)
    values = np.array([v for v in Parallel(n_jobs=-1, prefer="threads")(
        delayed(_auroc_iter)(y_test, y_pred_proba, s) for s in seeds
    ) if not np.isnan(v)])

    alpha    = 1 - CONFIDENCE_LEVEL
    ci_lower = np.percentile(values, (alpha / 2) * 100)
    ci_upper = np.percentile(values, (1 - alpha / 2) * 100)
    return point, ci_lower, ci_upper


def calculate_ici(y_true, y_prob, fraction=2/3):
    """ICI via LOWESS (Austin & Steyerberg 2019). Returns (ici, e50, e90)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_t, y_p = y_true[mask], y_prob[mask]

    order    = np.argsort(y_p)
    y_p_sort = y_p[order]
    y_t_sort = y_t[order]

    result = Lowess(
        fraction=fraction, # bandwidth span
        iterations=0,      # no robust IRLS, matches Austin & Steyerberg's iter=0
    ).fit(y_p_sort, y_t_sort)
    diffs  = np.abs(np.asarray(result.y, dtype=float) - y_p_sort)
    
    ici = float(np.mean(diffs))
    e50 = float(np.percentile(diffs, 50))
    e90 = float(np.percentile(diffs, 90))

    return ici, e50, e90


def _combine_test_sets(X_test_dict: dict, y_test_dict: dict, source_names: list) -> tuple:
    """Combine per-source test arrays for the requested source names."""
    X_parts = [np.asarray(X_test_dict[s]) for s in source_names]
    y_parts = [np.asarray(y_test_dict[s]).ravel() for s in source_names]
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def _get_single_stage_combined_sources(df: pd.DataFrame, data_source: str) -> dict:
    """Return combined source groups used by the single-source stage."""
    available = sorted(df[SOURCE_KEY].unique(), key=str.lower)
    combined_sets = {"combined4": [s for s in available if s in OVERLAP_SOURCES]}
    if data_source == "harmonized":
        combined_sets["combined7"] = available
    return combined_sets


# region data loading
# ==============================================================================
def load_cohort(data_source: str) -> pl.LazyFrame:
    """Load cohort data based on source (harmonized or original SQL)."""
    paths = {"harmonized": Path("data") / "cohort.parquet",
             "original":   Path("data_source") / "cohort.parquet"}

    cohort = pl.scan_parquet(paths[data_source]).filter(pl.col(MORT_KEY).is_not_null())
    print(f"loaded {cohort.count().collect()[0, 0]} samples from {data_source}")
    return cohort


# region model training
# ==============================================================================
def train_model(X_train, y_train, feature_cols: list = None) -> LGBMClassifier:
    """Train LightGBM classifier."""
    return LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    ).fit(X_train, y_train, feature_name=feature_cols)


def _get_model_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if col not in EXCLUDE_COLS and not str(col).startswith("Mortality")
    ]


def generate_stage_combinations(df: pd.DataFrame, stage_type: str):
    """Generate train/test source combinations for a given stage."""
    available_sources = sorted(df[SOURCE_KEY].unique(), key=str.lower)

    if stage_type == "single":
        for train_source in available_sources:
            yield [train_source], [s for s in available_sources if s != train_source]

    elif stage_type == "two":
        american = [s for s in DATA_REGION["American"] if s in available_sources]
        european = [s for s in DATA_REGION["European"] if s in available_sources]
        for am_src in american:
            for eu_src in european:
                train_sources = [am_src, eu_src]
                yield train_sources, [s for s in available_sources if s not in train_sources]

    elif stage_type == "lodo":
        for test_source in available_sources:
            yield [s for s in available_sources if s != test_source], [test_source]


def train_and_evaluate_stage(
    df: pd.DataFrame,
    train_sources: list,
    test_sources: list,
    combined_sets: dict | None = None,
) -> list:
    """Train model and evaluate on all test sets with overlap tracking."""
    metrics_list = []

    print(f"train: {', '.join(train_sources)}, test: {', '.join(test_sources)}")

    X_train, X_test_dict, y_train, y_test_dict, feature_cols, overlap_info, stay_ids_dict = (
        prepare_train_test_split(df, train_sources, test_sources)
    )
    print(f"train n={len(X_train)}, test n={len(X_test_dict['combined'])}")

    model = train_model(X_train, y_train, feature_cols)

    if combined_sets:
        for label, source_names in combined_sets.items():
            X_combined, y_combined = _combine_test_sets(X_test_dict, y_test_dict, source_names)
            stay_combined = np.concatenate([stay_ids_dict[s] for s in source_names])
            m = evaluate_model(model, X_combined, y_combined, label, feature_cols, stay_combined)
            m["train_sources"] = ",".join(train_sources)
            m["test_sources"]  = ",".join(source_names)
            m["overlap"]       = "mixed"
            metrics_list.append(m)
            print(f"    {'AUROC (' + label + '):':<23} {m['auroc']:.3f} [{m['auroc_ci_lower']:.3f}-{m['auroc_ci_upper']:.3f}]")

    for source in train_sources + test_sources:
        m = evaluate_model(model, X_test_dict[source], y_test_dict[source], source, feature_cols, stay_ids_dict[source])
        m["train_sources"] = ",".join(train_sources)
        m["test_sources"]  = ",".join(test_sources)
        m["overlap"]       = overlap_info[source]
        metrics_list.append(m)
        print(f"    {'AUROC (' + source + '):':<23} {m['auroc']:.3f} [{m['auroc_ci_lower']:.3f}-{m['auroc_ci_upper']:.3f}] (n={m['n_test_samples']:6.0f})")

    return metrics_list


# region evaluation
# ==============================================================================
def evaluate_model(model, X_test, y_test, db_name: str, feature_cols: list = None, stay_ids: np.ndarray = None) -> dict:
    """Evaluate model with AUROC, Brier, and ICI metrics."""
    X_test = pd.DataFrame(np.asarray(X_test, dtype=float), columns=feature_cols)
    y_test = np.asarray(y_test, dtype=float).ravel()

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auroc, auroc_ci_lower, auroc_ci_upper = bootstrap_auc(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    ici, *_ = calculate_ici(y_test, y_pred_proba)

    return {
        "test_database":           db_name,
        "auroc":                   auroc,
        "auroc_ci_lower":          auroc_ci_lower,
        "auroc_ci_upper":          auroc_ci_upper,
        "brier":                   brier,
        "ici":                     ici,
        "stay_ids":                stay_ids,
        "y_test":                  y_test,
        "y_pred_proba":            y_pred_proba,
        "n_test_samples":          len(y_test),
        "n_test_positives":        int(y_test.sum()),
        "test_positive_rate":      float(y_test.mean()),
    }


def prepare_train_test_split(
    df: pd.DataFrame,
    train_sources: list,
    test_sources: list,
) -> tuple:
    """Stratified 80/20 split; held-out 20% from train sources in per-source test sets.

    For self-validation pairs (train_source == test_source), uses test_stay_ids from cache
    (populated by harmonized run) to ensure identical test sets across pipelines.

    Returns: X_train, X_test_dict, y_train, y_test_dict, feature_cols, overlap_info, stay_ids_dict
    """
    global _STAY_IDS_CACHE
    test_stay_ids = _STAY_IDS_CACHE
    feature_cols  = _get_model_feature_columns(df)

    X_train_list          = []
    y_train_list          = []
    X_train_heldout_list  = []
    y_train_heldout_list  = []
    id_train_heldout_list = []

    for source in train_sources:
        source_df = df[df[SOURCE_KEY] == source]

        # Load or compute unique heldout stay IDs (defines the test cohort across pipelines)
        heldout_stay_ids_unique = test_stay_ids.get(source)
        if heldout_stay_ids_unique is None:
            # Harmonized run: compute 80/20 split by row index
            X_src_train, X_src_heldout, y_src_train, y_src_heldout = train_test_split(
                source_df[feature_cols],
                source_df[MORT_KEY],
                test_size=TEST_SIZE,
                stratify=source_df[MORT_KEY],
                random_state=RANDOM_STATE,
            )
            heldout_stay_ids_unique = np.unique(y_src_heldout.index.map(lambda idx: source_df.loc[idx, STAY_KEY]).values)
        else:
            # Original run: filter by cached unique stay IDs
            heldout_stay_ids_unique = np.array(heldout_stay_ids_unique)
            heldout_mask = source_df[STAY_KEY].isin(heldout_stay_ids_unique)
            X_src_train   = source_df[~heldout_mask][feature_cols]
            y_src_train   = source_df[~heldout_mask][MORT_KEY]
            X_src_heldout = source_df[heldout_mask][feature_cols]
            y_src_heldout = source_df[heldout_mask][MORT_KEY]

        # Extract stay ID for each heldout row (for predictions and comparison)
        heldout_stay_ids = X_src_heldout.index.map(lambda idx: source_df.loc[idx, STAY_KEY]).values

        X_train_list.append(X_src_train)
        y_train_list.append(y_src_train.values)
        X_train_heldout_list.append(X_src_heldout)
        y_train_heldout_list.append(y_src_heldout.values)
        id_train_heldout_list.append(heldout_stay_ids)

    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = np.concatenate(y_train_list)

    X_test_dict    = {}
    y_test_dict    = {}
    stay_ids_dict  = {}
    overlap_info   = {}
    X_test_list    = []
    y_test_list    = []
    stay_test_list = []

    for source in test_sources:
        source_df  = df[df[SOURCE_KEY] == source]
        X_src_test = source_df[feature_cols]
        y_src_test = source_df[MORT_KEY]
        overlap_info[source]  = "no_overlap"
        X_test_dict[source]   = X_src_test.values
        y_test_dict[source]   = y_src_test.values
        stay_ids_dict[source] = source_df[STAY_KEY].values
        X_test_list.append(X_src_test)
        y_test_list.append(y_src_test.values)
        stay_test_list.append(source_df[STAY_KEY].values)

    for i, source in enumerate(train_sources):
        X_test_dict[source]   = X_train_heldout_list[i].values
        y_test_dict[source]   = y_train_heldout_list[i]
        stay_ids_dict[source] = id_train_heldout_list[i]
        overlap_info[source]  = "overlap"

    X_test_combined  = pd.concat(X_train_heldout_list + X_test_list, ignore_index=True)
    y_test_combined  = np.concatenate(y_train_heldout_list + y_test_list)
    id_test_combined = np.concatenate(id_train_heldout_list + stay_test_list)

    X_test_dict["combined"]   = X_test_combined.values
    y_test_dict["combined"]   = y_test_combined
    stay_ids_dict["combined"] = id_test_combined

    # Cache unique heldout stay IDs for self-validation pairs (original pipeline reads from cache)
    for source in train_sources:
        if source not in _STAY_IDS_CACHE:
            _STAY_IDS_CACHE[source] = np.unique(stay_ids_dict[source]).tolist()

    return X_train, X_test_dict, y_train, y_test_dict, feature_cols, overlap_info, stay_ids_dict


# region stages
# ==============================================================================
def run_single_source_models(df: pd.DataFrame, data_source: str) -> list:
    """Train on each database individually, test on all others."""
    print("\n=== STAGE 1: SINGLE-SOURCE MODELS ===")

    combined_sets = _get_single_stage_combined_sources(df, data_source)
    all_metrics   = []
    for train_sources, test_sources in generate_stage_combinations(df, "single"):
        all_metrics.extend(train_and_evaluate_stage(df, train_sources, test_sources, combined_sets=combined_sets))
    return all_metrics


def run_two_source_models(df: pd.DataFrame) -> list:
    """Train on American-European pairs, test on all others."""
    print("\n=== STAGE 2: TWO-SOURCE MODELS (American-European Pairs) ===")

    all_metrics = []
    for train_sources, test_sources in generate_stage_combinations(df, "two"):
        all_metrics.extend(train_and_evaluate_stage(df, train_sources, test_sources))
    return all_metrics


def run_lodo(df: pd.DataFrame) -> list:
    """Leave-one-database-out validation."""
    print("\n=== STAGE 3: LEAVE-ONE-DATABASE-OUT (LODO) ===")

    all_metrics = []
    for train_sources, test_sources in generate_stage_combinations(df, "lodo"):
        all_metrics.extend(train_and_evaluate_stage(df, train_sources, test_sources))
    return all_metrics


# region sensitivity analysis
# ==============================================================================
def get_mimic3_carevue_stay_ids() -> set:
    """MIMIC-III CareVue-only stay IDs (pre-2008; no temporal overlap with MIMIC-IV)."""
    carevue_ids = (
        reprodICU.patient_information
        .select(STAY_KEY, "Source Dataset Version")
        .filter(pl.col("Source Dataset Version") == "v1.4 (CareVue)")
        .collect()
        .get_column(STAY_KEY)
        .to_list()
    )
    print(f"found {len(carevue_ids):,} CareVue stays in reprodICU")
    return set(carevue_ids)


def run_sensitivity_mimic3_carevue(df: pd.DataFrame) -> dict:
    """Re-run the two cross-MIMIC LODO folds with MIMIC-III restricted to CareVue-only (pre-2008)
    to eliminate the 2008–2012 temporal overlap with MIMIC-IV."""
    print("\n=== SENSITIVITY: MIMIC-III CareVue-only ===")

    carevue_ids = get_mimic3_carevue_stay_ids()

    mimic3_mask  = df[SOURCE_KEY] == "MIMIC-III"
    carevue_mask = df[STAY_KEY].isin(carevue_ids)

    n_orig    = int(mimic3_mask.sum())
    n_carevue = int((mimic3_mask & carevue_mask).sum())
    print(f"MIMIC-III: {n_orig:,} total, {n_carevue:,} CareVue-only (dropped {n_orig - n_carevue:,})")

    # Cohort with MIMIC-III replaced by CareVue-only subset
    df_sa = df[~mimic3_mask | carevue_mask].reset_index(drop=True)
    print(f"restricted cohort: {len(df_sa):,} stays")

    available = sorted(df_sa[SOURCE_KEY].unique())
    all_metrics = []

    for hold_out in ["MIMIC-IV", "MIMIC-III"]:
        train_sources = [s for s in available if s != hold_out]
        all_metrics.extend(train_and_evaluate_stage(df_sa, train_sources, [hold_out]))

    return {"lodo_cross_mimic_carevue": all_metrics}


# region results
# ==============================================================================
def reshape_metrics_to_long_format(all_results: dict, data_source: str) -> pd.DataFrame:
    """Reshape all metrics into long format with bootstrap arrays stored as columns."""
    rows = []
    for stage_name, stage_metrics in all_results.items():
        for m in stage_metrics:
            rows.append({
                "stage":                  stage_name,
                "cohort":                 data_source,
                "train_sources":          m.get("train_sources", ""),
                "test_database":          m.get("test_database", ""),
                "test_sources":           m.get("test_sources", ""),
                "auroc":                  m.get("auroc", np.nan),
                "auroc_ci_lower":         m.get("auroc_ci_lower", np.nan),
                "auroc_ci_upper":         m.get("auroc_ci_upper", np.nan),
                "brier":                  m.get("brier", np.nan),
                "ici":                    m.get("ici", np.nan),
                "n_test_samples":         m.get("n_test_samples", 0),
                "n_test_positives":       m.get("n_test_positives", 0),
                "test_positive_rate":     m.get("test_positive_rate", np.nan),
            })
    return pd.DataFrame(rows)


def reshape_predictions_to_wide_format(all_results: dict, data_source: str) -> pd.DataFrame:
    """Build wide predictions DataFrame: index=stay_id, columns=y_test and y_pred for each (train→test)."""
    predictions_dfs = []

    for stage_name, stage_metrics in all_results.items():
        for m in stage_metrics:
            train, test  = m["train_sources"], m["test_database"]
            pair_label = f"{train}→{test}"

            stay_ids = m["stay_ids"]
            y_test   = m["y_test"]
            y_pred   = m["y_pred_proba"]

            pair_df = pd.DataFrame({
                f"y_test_{data_source}_{pair_label}": y_test,
                f"y_pred_{data_source}_{pair_label}": y_pred,
            }, index=pd.Index(stay_ids, name="stay_id"))

            predictions_dfs.append(pair_df)

    result = predictions_dfs[0]
    for df in predictions_dfs[1:]:
        result = result.join(df, on="stay_id", how="outer")

    return result


def save_results(all_results: dict, data_source: str) -> None:
    """Save metrics to CSV and parquet; predictions to separate parquet."""
    output_path = OUTPUT_DIR / data_source
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nsaving to {output_path}...")

    for stage_name, stage_metrics in all_results.items():
        csv_data = [{k: v for k, v in m.items() if k not in _BOOTSTRAP_KEYS and k not in {"stay_ids", "y_test", "y_pred_proba"}} for m in stage_metrics]
        csv_path = output_path / f"metrics_{stage_name}.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)

    long_metrics = reshape_metrics_to_long_format(all_results, data_source)
    parquet_path = output_path / "metrics_long_format.parquet"
    long_metrics.to_parquet(parquet_path)

    predictions = reshape_predictions_to_wide_format(all_results, data_source)
    if not predictions.empty:
        predictions_path = output_path / "predictions_wide_format.parquet"
        predictions.to_parquet(predictions_path)

    print(f"\nsaved to {output_path}")

################################################################################
################################################################################

# region main
# ==============================================================================
if __name__ == "__main__":
    print("\n=== MORTALITY PREDICTION ===")
    print(f"started: {datetime.now().isoformat()}")

    harmonized_df = load_cohort("harmonized").collect().to_pandas()
    original_df   = load_cohort("original").collect().to_pandas()

    for data_source, cohort_df in [("harmonized", harmonized_df), ("original", original_df)]:
        print(f"\nsource: {data_source}")
        print(f"databases: {sorted(cohort_df[SOURCE_KEY].unique(), key=str.lower)}")
        results = {}
        results["single_source"] = run_single_source_models(cohort_df, data_source)
        results["lodo"]          = run_lodo(cohort_df)
        if data_source == "harmonized":
            results["two_source"] = run_two_source_models(cohort_df)
        save_results(results, data_source)

    save_results(run_sensitivity_mimic3_carevue(harmonized_df), "sensitivity_mimic3_carevue")

    print("\ndone")
