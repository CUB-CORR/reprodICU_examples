# Supplement Table Generation

# Output (output/supplement_tables/):
#   table_SA2_missingness.md                    missingness (%) per feature per database (harmonized + native SQL)
#   table_SA3_cohort_comparison.md              cohort counts and mortality (native SQL vs. reprodICU)
#   table_SA4_tier1_deltas.md                   per-fold AUROC/Brier/ICI deltas (harmonized vs. native SQL)
#   table_SA5_tost.md                           non-inferiority and TOST equivalence test (90% CI margin ±0.02 AUROC)
#   table_SA6_supplement_r2_harmonization.md    R^2 per variable and overlap source (harmonized vs. native SQL)
#   table_SB1_sensitivity_mimic_overlap.md      cross-MIMIC LODO folds: primary vs. CareVue-only
#   table_SB2_lodo_results.md                   LODO AUROC per test database vs. single-source distribution
#   table_SB3_auroc_matrix.md                   full 7x7 single-source AUROC matrix + LODO column
#   table_SB4_ici_matrix.md                     full 7x7 single-source ICI matrix + LODO column
#   table_SB5_brier_matrix.md                   full 7x7 single-source Brier matrix + LODO column
#   table_SC1_sofa_prevalence.md                SOFA subscore prevalence (score >= 1) by database

# ------------------------------------------------------------------------------

# Reads:  output/harmonized/, output/original/, output/comparison/
# Writes: output/supplement_tables/

################################################################################

import pandas as pd
import polars as pl

from _constants import (
    _DB_ORDER,
    _DIR,
    COMPARISON_DIR,
    DATA_DIR,
    DATA_SOURCE_DIR,
    MORT_KEY,
    OUTPUT_DIR,
    OVERLAP_SOURCES,
    SOFA_DATA_DIR,
    SOURCE_KEY,
    STAY_KEY,
    TABLES_DIR,
)

# region tables
# ==============================================================================

def delta_table() -> str:
    comp  = OUTPUT_DIR / "comparison"
    auroc = pl.read_csv(comp / "auroc_delta.csv")
    brier = pl.read_csv(comp / "brier_delta.csv")
    ici   = pl.read_csv(comp / "ici_delta.csv")

    df = (
        auroc.select(
            "train", "test",
            "h_auroc", "o_auroc", "delta_auroc",
            "h_auroc_ci_lower", "h_auroc_ci_upper",
            "o_auroc_ci_lower", "o_auroc_ci_upper"
        )
        .join(brier.select("train", "test", "h_brier", "o_brier", "delta_brier"), on=["train", "test"])
        .join(ici.select(  "train", "test", "h_ici",   "o_ici",   "delta_ici"),   on=["train", "test"])
        .filter(pl.col("test") != "combined3")
        .sort("train", "test")
        .to_pandas()
    )

    out = pd.DataFrame({
        "Training DB":      df["train"],
        "Test DB":          df["test"],
        "Harmonized AUROC (95% CI)":  df.apply(lambda r: f"{r.h_auroc:.3f} ({r.h_auroc_ci_lower:.3f}–{r.h_auroc_ci_upper:.3f})", axis=1),
        "Native AUROC (95% CI)":      df.apply(lambda r: f"{r.o_auroc:.3f} ({r.o_auroc_ci_lower:.3f}–{r.o_auroc_ci_upper:.3f})", axis=1),
        "ΔAUROC":           df["delta_auroc"].map(lambda x: f"{x:+.3f}"),
        "Harmonized Brier": df["h_brier"].map(lambda x: f"{x:.3f}"),
        "Native Brier":     df["o_brier"].map(lambda x: f"{x:.3f}"),
        "ΔBrier":           df["delta_brier"].map(lambda x: f"{x:+.3f}"),
        "Harmonized ICI":   df["h_ici"].map(lambda x: f"{x:.3f}"),
        "Native ICI":       df["o_ici"].map(lambda x: f"{x:.3f}"),
        "ΔICI":             df["delta_ici"].map(lambda x: f"{x:+.3f}"),
    })

    # Filter out combined4 for summary statistics
    df_no_combined = df[df["test"] != "combined4"].reset_index(drop=True)

    summary_rows = []
    for label, mask in [
        ("Internal (self-validation)", df_no_combined["train"] == df_no_combined["test"]),
        ("External (cross-validation)", df_no_combined["train"] != df_no_combined["test"]),
        ("Overall", pd.Series([True] * len(df_no_combined))),
    ]:
        subset = df_no_combined[mask]
        row = {c: "" for c in out.columns}
        row["Training DB"] = label
        row["Test DB"] = f"(n={len(subset)})"
        row["ΔAUROC"] = f"{subset['delta_auroc'].mean():+.3f}"
        row["ΔBrier"] = f"{subset['delta_brier'].mean():+.3f}"
        row["ΔICI"] = f"{subset['delta_ici'].mean():+.3f}"
        summary_rows.append(row)

    return pd.concat([out, pd.DataFrame(summary_rows)], ignore_index=True).to_markdown(index=False)


def cohort_table() -> str:
    original   = pl.read_parquet(DATA_SOURCE_DIR / "cohort.parquet")
    harmonized = pl.read_parquet(DATA_DIR / "cohort.parquet")

    rows = []
    for db in OVERLAP_SOURCES:
        for data, label in [
            (original.filter(  pl.col(SOURCE_KEY) == db), "Native SQL"),
            (harmonized.filter(pl.col(SOURCE_KEY) == db), "reprodICU"),
        ]:
            n      = data.height
            n_mort = int(data[MORT_KEY].cast(bool).sum())
            rows.append({
                "Database":      db,
                "Source":        label,
                "n":             f"{n:,}",
                "Deaths":        f"{n_mort:,}",
                "Mortality (%)": f"{100*n_mort/n:.2f}",
            })

    return pd.DataFrame(rows).to_markdown(index=False)


def r2_table() -> str:
    df = pd.read_csv(COMPARISON_DIR / "r2_by_variable.csv", index_col=0)
    df = df.sort_index(key=lambda s: s.str.lower())
    df.index.name = "Variable"
    df.columns.name = None
    return df.map(lambda x: f"{x:.4f}" if pd.notna(x) else "—").to_markdown()


def tost_table() -> str:
    tost_path = OUTPUT_DIR / "comparison" / "auroc_tost.csv"
    df = pl.read_csv(tost_path).sort("train", "test").to_pandas()

    out = pd.DataFrame({
        "Train DB":         df["train"],
        "Test DB":          df["test"],
        "Harmonized AUROC (95% CI)": df.apply(lambda r: f"{r.h_auroc:.3f} ({r.h_auroc_ci_lower:.3f}–{r.h_auroc_ci_upper:.3f})", axis=1),
        "Native AUROC (95% CI)":     df.apply(lambda r: f"{r.o_auroc:.3f} ({r.o_auroc_ci_lower:.3f}–{r.o_auroc_ci_upper:.3f})", axis=1),
        "Δ AUROC":          df["delta_auroc"].map(lambda x: f"{x:+.3f}"),
        "90 % CI for Δ":    df.apply(lambda r: f"[{r.delta_ci_lower:+.3f}, {r.delta_ci_upper:+.3f}]", axis=1),
        "p_NI":             df["p_ni"].map(lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001"),
        "Non-inferior":     df["non_inferior"].map(lambda x: "Yes" if x else "No"),
        "p_TOST":           df["p_tost"].map(lambda x: f"{x:.3f}" if x >= 0.001 else "<0.001"),
        "Equivalent":       df["equivalent"].map(lambda x: "Yes" if x else "No"),
    })

    return out.to_markdown(index=False)


def lodo_table() -> str:
    lodo   = pl.read_csv(OUTPUT_DIR / "harmonized" / "metrics_lodo.csv")
    single = pl.read_csv(OUTPUT_DIR / "harmonized" / "metrics_single_source.csv")

    single_agg = (
        single.filter(pl.col("overlap") == "no_overlap")
        .group_by("test_database")
        .agg(
            pl.col("auroc").mean().alias("siso_mean"),
            pl.col("auroc").median().alias("siso_median"),
            pl.col("auroc").quantile(0.25).alias("siso_q1"),
            pl.col("auroc").quantile(0.75).alias("siso_q3"),
        )
    )

    df = (
        lodo.filter(pl.col("overlap") == "no_overlap")
        .select("test_database", "auroc", "auroc_ci_lower", "auroc_ci_upper", "n_test_samples")
        .join(single_agg, on="test_database")
        .sort("test_database")
        .to_pandas()
    )

    out = pd.DataFrame({
        "Test Database":              df["test_database"],
        "n":                          df["n_test_samples"].map(lambda x: f"{x:,}"),
        "LODO AUROC (95% CI)":        df.apply(lambda r: f"{r.auroc:.3f} ({r.auroc_ci_lower:.3f}–{r.auroc_ci_upper:.3f})", axis=1),
        "Single-source mean":         df["siso_mean"].map(lambda x: f"{x:.3f}"),
        "Single-source median (IQR)": df.apply(lambda r: f"{r.siso_median:.3f} ({r.siso_q1:.3f}–{r.siso_q3:.3f})", axis=1),
        "LODO advantage":             (df["auroc"] - df["siso_mean"]).map(lambda x: f"{x:+.3f}"),
    })

    mean_row = {c: "" for c in out.columns}
    mean_row["Test Database"]       = "**Mean**"
    mean_row["LODO AUROC (95% CI)"] = f"**{df['auroc'].mean():.3f}**"
    mean_row["Single-source mean"]  = f"**{df['siso_mean'].mean():.3f}**"
    mean_row["LODO advantage"]      = f"**{(df['auroc'] - df['siso_mean']).mean():+.3f}**"

    return pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True).to_markdown(index=False)


def _load_matrix_data() -> tuple:
    siso = pd.read_csv(OUTPUT_DIR / "harmonized" / "metrics_single_source.csv")
    lodo = pd.read_csv(OUTPUT_DIR / "harmonized" / "metrics_lodo.csv")
    siso_ext = siso[siso["overlap"] == "no_overlap"].copy()
    lodo_ext = lodo[lodo["overlap"] == "no_overlap"].copy()
    return siso_ext, lodo_ext


def auroc_matrix() -> str:
    siso, lodo = _load_matrix_data()

    siso["cell"] = siso.apply(lambda r: f"{r.auroc:.3f} ({r.auroc_ci_lower:.3f}–{r.auroc_ci_upper:.3f})", axis=1)
    mat = siso.pivot(index="train_sources", columns="test_database", values="cell").reindex(
        index=_DB_ORDER, columns=_DB_ORDER
    ).fillna("—")

    lodo_col = (
        lodo.set_index("test_database")
        .reindex(_DB_ORDER)
        .apply(lambda r: f"**{r.auroc:.3f} ({r.auroc_ci_lower:.3f}–{r.auroc_ci_upper:.3f})**", axis=1)
    )
    mat["**LODO**"] = lodo_col.values

    siso_mean  = siso.groupby("test_database")["auroc"].mean().reindex(_DB_ORDER).map(lambda x: f"{x:.3f}")
    lodo_mean = f"**{lodo.set_index('test_database').reindex(_DB_ORDER)['auroc'].mean():.3f}**"
    mean_row  = pd.DataFrame(
        [list(siso_mean) + [lodo_mean]], columns=mat.columns, index=["**Column mean**"]
    )

    out = pd.concat([mat, mean_row])
    out.index.name = "Train \\ Test"
    return out.to_markdown()


def _scalar_metric_matrix(metric: str) -> str:
    siso, lodo = _load_matrix_data()
    mat = siso.pivot(index="train_sources", columns="test_database", values=metric).reindex(
        index=_DB_ORDER, columns=_DB_ORDER
    )
    mat_fmt = mat.map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    lodo_col = (
        lodo.set_index("test_database").reindex(_DB_ORDER)[metric]
        .map(lambda x: f"**{x:.3f}**")
    )
    mat_fmt["**LODO**"] = lodo_col.values
    siso_mean   = siso.groupby("test_database")[metric].mean().reindex(_DB_ORDER).map(lambda x: f"{x:.3f}")
    lodo_mean = f"**{lodo.set_index('test_database').reindex(_DB_ORDER)[metric].mean():.3f}**"
    mean_row  = pd.DataFrame(
        [list(siso_mean) + [lodo_mean]], columns=mat_fmt.columns, index=["**Column mean**"]
    )
    out = pd.concat([mat_fmt, mean_row])
    out.index.name = "Train \\ Test"
    return out.to_markdown()


def ici_matrix() -> str:
    return _scalar_metric_matrix("ici")


def brier_matrix() -> str:
    return _scalar_metric_matrix("brier")


def sofa_table() -> str:
    counts_path  = SOFA_DATA_DIR / "sofa_subscore_counts.csv"
    parquet_path = SOFA_DATA_DIR / "sofa_subscores_day0.parquet"
    df        = pl.read_csv(counts_path, null_values=["", "NA"])
    db_totals = pl.read_parquet(parquet_path).group_by(SOURCE_KEY).agg(pl.len().alias("total"))

    return (
        df.filter(pl.col("Points") >= 1)
        .group_by(SOURCE_KEY, "Subscore")
        .agg(pl.col("Count").sum().alias("n_gte1"))
        .join(db_totals, on=SOURCE_KEY)
        .with_columns(
            (pl.col("n_gte1") / pl.col("total") * 100)
            .round(1).cast(str).add("%").alias("cell")
        )
        .pivot(on=SOURCE_KEY, index="Subscore", values="cell")
        .sort("Subscore")
        .fill_null("No data")
        .to_pandas()
        .pipe(lambda d: d[["Subscore"] + sorted(
            [c for c in d.columns if c != "Subscore"], key=str.lower
        )])
        .to_markdown(index=False)
    )


def sensitivity_mimic_overlap_table() -> str:
    """Sensitivity analysis: cross-MIMIC LODO folds — primary vs. CareVue-only restriction."""
    primary_path = OUTPUT_DIR / "harmonized" / "metrics_lodo.csv"
    sens_path    = OUTPUT_DIR / "sensitivity_mimic3_carevue" / "metrics_lodo_cross_mimic_carevue.csv"

    primary = (
        pl.read_csv(primary_path)
        .filter(
            pl.col("overlap") == "no_overlap",
            pl.col("test_database").is_in(["MIMIC-III", "MIMIC-IV"]),
        )
        .select("test_database", "auroc", "auroc_ci_lower", "auroc_ci_upper", "brier", "ici")
    )

    sens = (
        pl.read_csv(sens_path)
        .filter(pl.col("overlap") == "no_overlap")
        .select("test_database", "auroc", "auroc_ci_lower", "auroc_ci_upper", "brier", "ici")
    )

    df = (
        primary
        .join(sens, on="test_database", suffix="_sa")
        .sort("test_database")
        .to_pandas()
    )

    auroc = pd.DataFrame({
        "Test DB": df["test_database"],
        "Primary": df.apply(lambda r: f"{r.auroc:.3f} ({r.auroc_ci_lower:.3f}–{r.auroc_ci_upper:.3f})", axis=1),
        "CareVue": df.apply(lambda r: f"{r.auroc_sa:.3f} ({r.auroc_ci_lower_sa:.3f}–{r.auroc_ci_upper_sa:.3f})", axis=1),
        "Δ":       (df["auroc_sa"].round(3) - df["auroc"].round(3)).map(lambda x: f"{x:+.3f}"),
    })

    sections = ["**AUROC (95% CI)**\n\n" + auroc.to_markdown(index=False)]
    for header, col in [("**Brier score**", "brier"), ("**ICI**", "ici")]:
        sub = pd.DataFrame({
            "Test DB": df["test_database"],
            "Primary": df[col].map(lambda x: f"{x:.3f}"),
            "CareVue": df[f"{col}_sa"].map(lambda x: f"{x:.3f}"),
            "Δ":       (df[f"{col}_sa"].round(3) - df[col].round(3)).map(lambda x: f"{x:+.3f}"),
        })
        sections.append(f"{header}\n\n" + sub.to_markdown(index=False))

    note = (
        "_Primary: full MIMIC-III (CareVue + Metavision, 2001–2012). "
        "CareVue: MIMIC-III restricted to pre-2008 admissions, "
        "eliminating the 2008–2012 temporal overlap with MIMIC-IV._"
    )
    return "\n\n".join(sections + [note])


def _missingness_md(df: pd.DataFrame, label: str) -> str:
    NON_FEATURE_COLS = [STAY_KEY, SOURCE_KEY]
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    miss = (
        df.groupby(SOURCE_KEY)[feature_cols]
        .apply(lambda g: g.isna().mean() * 100)
        .T
        .round(1)
        .reset_index()
        .rename(columns={"index": "Feature"})
    )
    miss = miss[["Feature"] + sorted([c for c in miss.columns if c != "Feature"], key=str.lower)]
    miss = miss.sort_values("Feature", key=lambda s: s.str.lower()).reset_index(drop=True)
    miss[miss.columns[1:]] = miss[miss.columns[1:]].map(lambda x: f"{x:.1f}%")

    return f"**{label}**\n\n" + miss.to_markdown(index=False)


def missingness_table() -> str:
    """Missingness tables for harmonized (reprodICU) and native SQL cohorts."""
    harmonized = pd.read_parquet(DATA_DIR / "cohort.parquet")
    original   = pd.read_parquet(DATA_SOURCE_DIR / "cohort.parquet")

    return "\n\n".join([
        _missingness_md(harmonized, "reprodICU (harmonized cohort)"),
        _missingness_md(original,   "Native SQL (original cohort; MIMIC-III, MIMIC-IV, eICU-CRD)"),
    ])


# region main
# ==============================================================================
if __name__ == "__main__":
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    tables = {
        "table_SA2_missingness.md":                 missingness_table,
        "table_SA3_cohort_comparison.md":           cohort_table,
        "table_SA4_tier1_deltas.md":                delta_table,
        "table_SA5_tost.md":                        tost_table,
        "table_SA6_supplement_r2_harmonization.md": r2_table,
        "table_SB1_sensitivity_mimic_overlap.md":   sensitivity_mimic_overlap_table,
        "table_SB2_lodo_results.md":                lodo_table,
        "table_SB3_auroc_matrix.md":                auroc_matrix,
        "table_SB4_ici_matrix.md":                  ici_matrix,
        "table_SB5_brier_matrix.md":                brier_matrix,
        "table_SC1_sofa_prevalence.md":             sofa_table,
    }
    for filename, fn in tables.items():
        path = TABLES_DIR / filename
        path.write_text(fn() + "\n", encoding="utf-8")
        print(f"generated {path.relative_to(_DIR)}")
