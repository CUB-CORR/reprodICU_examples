# reprodICU Harmonization Validation

Manuscript analyses validating reprodICU harmonization across seven ICU databases using mortality prediction models and cohort comparisons.

## Folders

### Mortality Prediction/
- **1_cohort.py** — Build harmonized cohorts (all 7 databases)
- **1_cohort_from_original_data.py** — Build same cohorts from source SQL based on public SQL queries from
    - `mimic-code`: https://github.com/MIT-LCP/mimic-code
    - `eicu-code`: https://github.com/MIT-LCP/eicu-code
    - `AmsterdamUMCdb`: https://github.com/AmsterdamUMC/AmsterdamUMCdb
- **2_models.py** — Train LightGBM models; output coefficients and performance metrics
- **3_compare_cohorts.py** — Leave-one-database-out cross-validation

### General Plots/
- **1_plots.py** — Cohort characteristics and missingness patterns
- **2_sofa_subscores.py** — SOFA component prevalence analysis