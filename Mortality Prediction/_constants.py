from pathlib import Path

STAY_KEY        = "Global ICU Stay ID"
MORT_KEY        = "Mortality in ICU"
SOURCE_KEY      = "Source Dataset"
EXCLUDE_COLS    = {STAY_KEY, MORT_KEY, SOURCE_KEY}

OVERLAP_SOURCES   = ["MIMIC-III", "MIMIC-IV", "eICU-CRD", "AmsterdamUMCdb"]
CORRELATION_ORDER = sorted(OVERLAP_SOURCES, key=str.lower)

DATA_REGION = {
    "American": ["eICU-CRD", "MIMIC-III", "MIMIC-IV", "NWICU"],
    "European": ["AmsterdamUMCdb", "HiRID", "SICdb"],
}
_DB_ORDER = sorted((db for dbs in DATA_REGION.values() for db in dbs), key=str.lower)

RANDOM_STATE = 42

# Fixed tab10 palette — one colour per database, assigned in str.lower alphabetical order.
# Matches General Plots/1_plots.py; hardcoded so both scripts stay in sync.
DB_PALETTE = {
    "AmsterdamUMCdb": "#1f77b4",
    "eICU-CRD":       "#ff7f0e",
    "HiRID":          "#2ca02c",
    "MIMIC-III":      "#d62728",
    "MIMIC-IV":       "#9467bd",
    "NWICU":          "#8c564b",
    "SICdb":          "#e377c2",
}

METRICS          = ["AUROC", "Brier", "ICI"]
COMBINED_LABEL_4 = "combined4"
COMBINED_LABEL_7 = "combined7"

_DIR            = Path(__file__).resolve().parent
OUTPUT_DIR      = _DIR / "output"
PLOTS_DIR       = _DIR / "plots"
DATA_DIR        = _DIR / "data"
DATA_SOURCE_DIR = _DIR / "data_source"

SOFA_DATA_DIR   = _DIR.parent / "General Plots" / "data"

HARMONIZED_DIR    = OUTPUT_DIR / "harmonized"
ORIGINAL_DIR      = OUTPUT_DIR / "original"
COMPARISON_DIR    = OUTPUT_DIR / "comparison"
CORRELATION_DIR   = PLOTS_DIR / "correlations"
TABLES_DIR        = OUTPUT_DIR / "supplement_tables"
HARMONIZED_COHORT = DATA_DIR / "cohort.parquet"
ORIGINAL_COHORT   = DATA_SOURCE_DIR / "cohort.parquet"
