# Cross-Cohort Comparison: Harmonized vs. Original SQL Cohorts

# Cohorts:
#   Harmonized — 7 sources via reprodICU (AmsterdamUMCdb, eICU-CRD, HiRID, MIMIC-III, MIMIC-IV, NWICU, SICdb)
#   Original   — 4 SQL sources (AmsterdamUMCdb, eICU-CRD, MIMIC-III, MIMIC-IV)

# Analyses:
#   Delta metrics:    AUROC, Brier score, ICI (harmonized vs. original, per fold)
#   R^2 correlations: variable-level agreement between harmonized and original cohorts
#   Visualizations:   comparison plots saved to plots/

# ------------------------------------------------------------------------------

# Outcome: ICU mortality
# Model: LightGBM (loads pre-fitted results from output/harmonized/ and output/original/)

################################################################################

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from _constants import (
    COMBINED_LABEL_4,
    COMBINED_LABEL_7,
    COMPARISON_DIR,
    CORRELATION_DIR,
    CORRELATION_ORDER,
    DB_PALETTE,
    HARMONIZED_COHORT,
    HARMONIZED_DIR,
    METRICS,
    MORT_KEY,
    ORIGINAL_COHORT,
    ORIGINAL_DIR,
    OVERLAP_SOURCES,
    PLOTS_DIR,
    RANDOM_STATE,
    SOURCE_KEY,
    STAY_KEY,
)

# region constants
# ==============================================================================
EXCLUDE_CORRELATION = [STAY_KEY, SOURCE_KEY, MORT_KEY, "Gender"]


def setup() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    CORRELATION_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)


# region source normalization
# ==============================================================================
def parse_sources(value: str) -> set:
    if pd.isna(value):
        return set()
    return {t for t in str(value).split(",") if t.strip()}


def project_to_overlap(sources: set) -> set:
    return sources & set(OVERLAP_SOURCES)


# region data loading
# ==============================================================================
def load_cohort_frame(path: Path, origin: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["source"] = frame[SOURCE_KEY].astype(str)
    frame["origin"] = origin
    return frame


def load_metrics(stage_dir: Path, stage_name: str) -> pd.DataFrame:
    df = pd.read_csv(stage_dir / f"metrics_{stage_name}.csv")
    df["cohort"] = "harmonized" if "harmonized" in str(stage_dir) else "original"
    return df


def load_long_format_metrics(harmonized_dir: Path, original_dir: Path) -> pd.DataFrame:
    harm = pd.read_parquet(harmonized_dir / "metrics_long_format.parquet")
    harm["cohort"] = "harmonized"

    orig = pd.read_parquet(original_dir / "metrics_long_format.parquet")
    orig["cohort"] = "original"

    return pd.concat([harm, orig], ignore_index=True)


# region data preparation
# ==============================================================================
def prepare_correlation_pairs(
    harmonized_df: pd.DataFrame, original_df: pd.DataFrame, sources: list
) -> pd.DataFrame:
    frames = []
    for source_name in sources:
        harm_src = harmonized_df[harmonized_df["source"] == source_name].copy()
        orig_src = original_df[original_df["source"] == source_name].copy()

        common_cols = sorted(set(harm_src.columns) & set(orig_src.columns), key=str.lower)
        numeric_cols = [
            col for col in common_cols
            if col not in EXCLUDE_CORRELATION
               and pd.api.types.is_numeric_dtype(harm_src[col])
               and pd.api.types.is_numeric_dtype(orig_src[col])
        ]

        merged = pd.merge(
            harm_src[["source", STAY_KEY] + numeric_cols],
            orig_src[["source", STAY_KEY] + numeric_cols],
            on=["source", STAY_KEY],
            suffixes=("_harm", "_orig"),
        )

        for col in numeric_cols:
            param_df = merged[
                ["source", STAY_KEY, f"{col}_harm", f"{col}_orig"]
            ].rename(
                columns={
                    f"{col}_harm": "harmonized_value",
                    f"{col}_orig": "original_value",
                }
            )
            param_df["parameter"] = col
            param_df = param_df.dropna(subset=["harmonized_value", "original_value"])
            frames.append(param_df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# region metrics comparison
# ==============================================================================
def pair_metrics(
    harm_metrics: pd.DataFrame, orig_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, list]:
    """Match original rows to harmonized rows by train/test source overlap.

    Prefers test_database match; falls back to test_sources projection.
    """
    harm_parsed = [
        {
            "row":        h_row,
            "train_proj": project_to_overlap(parse_sources(h_row.get("train_sources", ""))),
            "test_proj":  project_to_overlap(parse_sources(h_row.get("test_sources", ""))),
            "test_db":    h_row.get("test_database", ""),
        }
        for _, h_row in harm_metrics.iterrows()
    ]

    paired_rows, unmapped = [], []

    for _, o_row in orig_metrics.iterrows():
        o_train_proj = project_to_overlap(parse_sources(o_row.get("train_sources", "")))
        o_test_proj  = project_to_overlap(parse_sources(o_row.get("test_sources", "")))
        o_test_db    = o_row.get("test_database", "")

        candidates = [h for h in harm_parsed if h["train_proj"] == o_train_proj]
        selected = next((h for h in candidates if h["test_db"] == o_test_db), None) if o_test_db else None
        if selected is None:
            selected = next((h for h in candidates if h["test_proj"] == o_test_proj), None)

        if selected is not None:
            h_row = selected["row"]
            row_data = {
                "train": str(o_row.get("train_sources", "")),
                "test":  str(o_row.get("test_database", o_row.get("test_sources", ""))),
            }
            for m in [m.lower() for m in METRICS]:
                row_data[f"h_{m}"]          = h_row.get(m, np.nan)
                row_data[f"h_{m}_ci_lower"] = h_row.get(f"{m}_ci_lower", np.nan)
                row_data[f"h_{m}_ci_upper"] = h_row.get(f"{m}_ci_upper", np.nan)
                row_data[f"o_{m}"]          = o_row.get(m, np.nan)
                row_data[f"o_{m}_ci_lower"] = o_row.get(f"{m}_ci_lower", np.nan)
                row_data[f"o_{m}_ci_upper"] = o_row.get(f"{m}_ci_upper", np.nan)
            paired_rows.append(row_data)
        else:
            unmapped.append((str(o_row.get("train_sources", "")), str(o_row.get("test_sources", ""))))

    return pd.DataFrame(paired_rows), unmapped


def compute_deltas(paired_df: pd.DataFrame) -> pd.DataFrame:
    delta_df = paired_df.copy()
    for m in [m.lower() for m in METRICS]:
        delta_df[f"delta_{m}"] = delta_df[f"h_{m}"].round(3) - delta_df[f"o_{m}"].round(3)
    return delta_df


# region paired-delta bootstrap
# ==============================================================================
def _paired_bootstrap_delta_iter(y_test, y_pred_h, y_pred_o, seed):
    """Single iteration of paired-delta bootstrap.

    Resamples patients (same indices for both) and returns delta = AUROC_h - AUROC_o.
    Returns NaN if resample lacks both outcome classes (unable to compute AUC).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_test), size=len(y_test), replace=True)
    y_b = y_test[idx]
    if len(np.unique(y_b)) < 2:
        return np.nan
    auroc_h = roc_auc_score(y_b, y_pred_h[idx])
    auroc_o = roc_auc_score(y_b, y_pred_o[idx])
    return auroc_h - auroc_o


def _paired_bootstrap_delta(y_test, y_pred_h, y_pred_o, n_bootstraps=2000, margin=0.02):
    """Paired-delta bootstrap with parallel resampling (Westlake 1976 TOST).

    Generates n_bootstraps delta values by jointly resampling patients.
    Returns point estimate, 90% CI (Westlake rule), p_NI, p_NS, p_TOST, and delta distribution.
    90% CI: equivalent if entirely within [−margin, +margin].
    """
    seeds = np.random.default_rng(RANDOM_STATE).integers(0, 2**31, n_bootstraps)
    deltas = [v for v in Parallel(n_jobs=-1, prefer="threads")(
        delayed(_paired_bootstrap_delta_iter)(y_test, y_pred_h, y_pred_o, s) for s in seeds
    ) if not np.isnan(v)]

    delta_dist = np.array(deltas)
    point = np.mean(delta_dist)
    ci_lower = np.percentile(delta_dist, 5)
    ci_upper = np.percentile(delta_dist, 95)
    p_ni = (delta_dist < -margin).mean()  # H₀: Δ ≤ -margin
    p_ns = (delta_dist >  margin).mean()  # H₀: Δ ≥ +margin
    p_tost = max(p_ni, p_ns)

    return point, ci_lower, ci_upper, p_ni, p_ns, p_tost, delta_dist


def compute_auroc_tost(
    long_metrics: pd.DataFrame, margin: float = 0.02
) -> pd.DataFrame:
    """Paired TOST for AUROC using paired-delta bootstrap (Westlake 1976).

    Paired observations: same patients evaluated by harmonized and original pipelines.
    Paired-delta bootstrap: resample same patient indices for both cohorts → delta distribution respects correlation.

    Equivalence test (90% CI rule):
      Equivalent iff 90% CI ⊂ [−margin, +margin]
      Equivalently: p_tost = max(p_ni, p_ns) < 0.05

    Tier 1 pairs: single-source models (train on each DB, test on overlap sources).
    """
    # Load wide-format predictions (indexed by stay_id, one column per (train→test, cohort))
    harm_pred = pl.read_parquet(HARMONIZED_DIR / "predictions_wide_format.parquet")
    orig_pred = pl.read_parquet(ORIGINAL_DIR / "predictions_wide_format.parquet")

    lm = long_metrics[long_metrics["stage"] == "single_source"].copy()
    lm["_train"] = lm["train_sources"]
    lm["_test"]  = lm["test_database"]
    lm = lm[lm["_train"].isin(OVERLAP_SOURCES) & lm["_test"].isin(OVERLAP_SOURCES)]

    rows = []
    for (train, test), grp in lm.groupby(["_train", "_test"]):
        h = grp[grp["cohort"] == "harmonized"]
        o = grp[grp["cohort"] == "original"]
        if h.empty or o.empty: continue

        pair_label   = f"{train}→{test}"
        col_y_test_h = f"y_test_harmonized_{pair_label}"
        col_y_pred_h = f"y_pred_harmonized_{pair_label}"
        col_y_test_o = f"y_test_original_{pair_label}"
        col_y_pred_o = f"y_pred_original_{pair_label}"
        print("\n3a. Processing pair:", pair_label)

        # Merge on stay_id (index) and drop rows with NaN in any prediction column
        pred_pair = (
            harm_pred.select("stay_id", col_y_test_h, col_y_pred_h)
            .join(
                orig_pred.select("stay_id", col_y_test_o, col_y_pred_o),
                on="stay_id",
                how="inner",
            )
            .drop_nulls()
            .to_pandas()
        )

        y_test   = pred_pair[col_y_test_h].values
        y_pred_h = pred_pair[col_y_pred_h].values
        y_pred_o = pred_pair[col_y_pred_o].values

        # Run paired-delta bootstrap on aligned patients
        delta_point, delta_ci_lower, delta_ci_upper, p_ni, p_ns, p_tost, _ = _paired_bootstrap_delta(
            y_test, y_pred_h, y_pred_o, margin=margin
        )

        # Point estimates from individual bootstrap (for reference)
        h_auroc = float(h.iloc[0]["auroc"])
        h_auroc_ci_lower = float(h.iloc[0]["auroc_ci_lower"])
        h_auroc_ci_upper = float(h.iloc[0]["auroc_ci_upper"])

        o_auroc = float(o.iloc[0]["auroc"])
        o_auroc_ci_lower = float(o.iloc[0]["auroc_ci_lower"])
        o_auroc_ci_upper = float(o.iloc[0]["auroc_ci_upper"])

        # Equivalence decision: Westlake rule for paired observations
        # Equivalent if 90% CI entirely within [−margin, +margin]
        equivalent_by_ci = (delta_ci_lower > -margin) and (delta_ci_upper < +margin)
        equivalent_by_pval = (p_tost < 0.05)
        # These should always agree (mathematically equivalent)
        assert equivalent_by_ci == equivalent_by_pval, \
            f"Pair {pair_label}: CI-based ({equivalent_by_ci}) and p-value-based ({equivalent_by_pval}) decisions disagree!"

        rows.append({
            "train":            train,
            "test":             test,
            "h_auroc":          h_auroc,
            "h_auroc_ci_lower": h_auroc_ci_lower,
            "h_auroc_ci_upper": h_auroc_ci_upper,
            "o_auroc":          o_auroc,
            "o_auroc_ci_lower": o_auroc_ci_lower,
            "o_auroc_ci_upper": o_auroc_ci_upper,
            "delta_auroc":      delta_point,
            "delta_ci_lower":   delta_ci_lower,
            "delta_ci_upper":   delta_ci_upper,
            "p_ni":             p_ni,
            "p_ns":             p_ns,
            "p_tost":           p_tost,
            "non_inferior":     p_ni < 0.05,
            "equivalent":       equivalent_by_pval,
            "margin":           margin,
        })

    return pd.DataFrame(rows)


# region plotting utilities
# ==============================================================================
def slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in str(name).strip()).strip("_")


def compute_axis_limits(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    lower = float(min(x.quantile(0.01), y.quantile(0.01)))
    upper = float(max(x.quantile(0.99), y.quantile(0.99)))
    if np.isclose(lower, upper):
        lower, upper = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    if np.isclose(lower, upper):
        lower, upper = lower - 0.5, upper + 0.5
    padding = (upper - lower) * 0.08
    return lower - padding, upper + padding


def compute_r2_and_fits(
    x: pd.Series, y: pd.Series, axis_min: float, axis_max: float
) -> tuple:
    mask = (x >= axis_min) & (x <= axis_max) & (y >= axis_min) & (y <= axis_max)
    x_filt, y_filt = x[mask], y[mask]
    if (
        len(x_filt) < 2
        or np.isclose(x_filt.std(ddof=0), 0)
        or np.isclose(y_filt.std(ddof=0), 0)
    ):
        return None, None, None
    r2 = float(np.corrcoef(x_filt, y_filt)[0, 1] ** 2)
    slope, intercept = np.polyfit(x_filt, y_filt, 1)
    return r2, slope, intercept


def _format_db_name(name: str) -> str:
    return "AUMCdb" if name == "AmsterdamUMCdb" else name


def get_heatmap_params(metric: str) -> tuple[str, float | None]:
    if metric == "auroc":
        return "RdYlGn", 0.5
    return "RdYlGn_r", None


def build_heatmap_matrix(
    data: pd.DataFrame,
    metric: str,
    train_col: str,
    test_col: str,
    combined_label: str,
) -> tuple[np.ndarray, list, list, dict]:
    trains = sorted(data[train_col].unique(), key=str.lower)
    tests  = sorted([t for t in data[test_col].unique() if t != combined_label], key=str.lower)
    matrix = np.full((len(trains), len(tests)), np.nan)
    combined_data = {}

    for i, train in enumerate(trains):
        for j, test in enumerate(tests):
            mask = (data[train_col] == train) & (data[test_col] == test)
            if mask.any():
                matrix[i, j] = data.loc[mask, metric].values[0]
        combined_mask = (data[train_col] == train) & (data[test_col] == combined_label)
        if combined_mask.any():
            combined_data[i] = data.loc[combined_mask, metric].values[0]

    return matrix, trains, tests, combined_data


def save_plot(filename: str, out_dir: Path) -> None:
    path = out_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"saved {path.name}")


# region plotting functions
# ==============================================================================
def compute_r2_values_from_cohorts(
    harm_cohort: pd.DataFrame, orig_cohort: pd.DataFrame
) -> pd.DataFrame:
    numeric_cols = sorted([
        col for col in harm_cohort.columns
        if col not in EXCLUDE_CORRELATION
            and pd.api.types.is_numeric_dtype(harm_cohort[col])
            and col in orig_cohort.columns
            and pd.api.types.is_numeric_dtype(orig_cohort[col])
    ], key=str.lower,)

    print(f"processing {len(numeric_cols)} params × {len(CORRELATION_ORDER)} sources...")

    harm_copy = harm_cohort.copy()
    orig_copy = orig_cohort.copy()
    harm_copy["join_id"] = harm_copy.get(STAY_KEY, harm_copy.index).astype(str)
    orig_copy["join_id"] = orig_copy.get(STAY_KEY, orig_copy.index).astype(str)

    merged = pd.merge(
        harm_copy[["source", "join_id"] + numeric_cols],
        orig_copy[["source", "join_id"] + numeric_cols],
        on=["source", "join_id"],
        suffixes=("_harm", "_orig"),
    )

    rows = {}
    for col in numeric_cols:
        x_all = merged[f"{col}_harm"].astype(float)
        y_all = merged[f"{col}_orig"].astype(float)
        axis_min, axis_max = compute_axis_limits(x_all, y_all)
        rows[col] = {}
        for source_name in CORRELATION_ORDER:
            source_data = merged[merged["source"] == source_name]
            x = source_data[f"{col}_harm"].astype(float)
            y = source_data[f"{col}_orig"].astype(float)
            r2, _, _ = compute_r2_and_fits(x, y, axis_min, axis_max)
            rows[col][source_name] = r2
            r2_str = f"{r2:.4f}" if r2 is not None else "None"
            print(f"{source_name:<15} {col:<30} R^2 = {r2_str}")

    print(f"computed {len(numeric_cols) * len(CORRELATION_ORDER)} R^2 values")
    return pd.DataFrame(rows).T.rename_axis("variable")


def plot_r2_kde_by_source(r2_df: pd.DataFrame) -> None:
    source_order = [s for s in CORRELATION_ORDER if s in r2_df.columns and r2_df[s].notna().any()]

    plot_df = pd.DataFrame([
        {"Source": s, "R^2": r2_val}
        for s in source_order
        for r2_val in r2_df[s].dropna()
    ])
    color_palette = DB_PALETTE

    fig, ax = plt.subplots(figsize=(7, 3))
    for source in source_order:
        bw = 2 if source == "eICU-CRD" else 0.5
        sns.kdeplot(
            data=plot_df[plot_df["Source"] == source],
            x="R^2",
            ax=ax,
            bw_adjust=bw,
            label=source,
            color=color_palette[source],
            linewidth=2.0,
        )

    sns.rugplot(
        data=plot_df,
        x="R^2",
        ax=ax,
        hue="Source",
        palette=color_palette,
        height=0.1,
        linewidth=1.0,
        alpha=0.3,
        legend=False,
    )

    ax.set_xlabel("R² (Pearson Correlation Squared)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(visible=False)
    ax.set_xlim(0, 1)

    summary_text = "Summary Statistics:"
    for source in source_order:
        values = r2_df[source].dropna().tolist()
        summary_text += f"\n{source}: μ={np.mean(values):.3f}, σ={np.std(values):.3f}"
    print(summary_text)

    save_plot("r2_distribution_kde.png", PLOTS_DIR)
    plt.close()


def save_r2_table(r2_df: pd.DataFrame) -> None:
    path = COMPARISON_DIR / "r2_by_variable.csv"
    r2_df.round(4).to_csv(path)
    print(f"saved {path.name} ({len(r2_df)} vars × {len(r2_df.columns)} sources)")


def plot_parameter_correlations(data: pd.DataFrame) -> None:
    print(f"\ncreating scatter plots ({len(data)} rows)...")

    color_palette = DB_PALETTE

    for param in sorted(data["parameter"].astype(str).unique(), key=str.lower):
        param_data   = data[data["parameter"] == param]
        plot_sources = [s for s in CORRELATION_ORDER if s in set(param_data["source"])]

        axis_min, axis_max = compute_axis_limits(
            param_data["harmonized_value"].astype(float),
            param_data["original_value"].astype(float),
        )
        plot_data = param_data[
            (param_data["harmonized_value"] >= axis_min) & (param_data["harmonized_value"] <= axis_max) &
            (param_data["original_value"]   >= axis_min) & (param_data["original_value"]   <= axis_max)
        ]

        g = sns.jointplot(
            data=plot_data,
            x="harmonized_value",
            y="original_value",
            hue="source",
            kind="scatter",
            height=8,
            ratio=5,
            space=0.08,
            dropna=True,
            palette=color_palette,
            marginal_ticks=False,
            joint_kws={
                "s": 5,
                "alpha": 0.1,
                "edgecolor": "white",
                "linewidth": 0.25,
            },
            marginal_kws={
                "fill": True,
                "common_norm": False,
                "warn_singular": False,
                "alpha": 0.22,
            },
        )

        r2_lines = []
        for source in plot_sources:
            source_data = plot_data[plot_data["source"] == source].dropna(
                subset=["harmonized_value", "original_value"]
            )
            x = source_data["harmonized_value"].astype(float)
            y = source_data["original_value"].astype(float)
            r2, slope, intercept = compute_r2_and_fits(x, y, axis_min, axis_max)
            sns.regplot(
                x=x,
                y=y,
                ax=g.ax_joint,
                scatter=False,
                ci=None,
                color=color_palette.get(source, "#666666"),
                line_kws={"linewidth": 2.2, "zorder": 4},
                truncate=False,
            )
            r2_lines.append(f"{source:<15} R² = {r2:.3f}  m = {slope:+.3f}  b = {intercept:+.3f}")

        g.ax_joint.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            color="#666666",
            linewidth=1.2,
            zorder=2,
        )
        g.ax_joint.set_xlim(axis_min, axis_max)
        g.ax_joint.set_ylim(axis_min, axis_max)
        g.ax_joint.set_xlabel("Harmonized", fontsize=11)
        g.ax_joint.set_ylabel("Original", fontsize=11)
        g.ax_joint.set_title(f"{param} correlation by source", fontsize=14, fontweight="bold")
        g.ax_joint.grid(alpha=0.18)

        legend = g.ax_joint.legend(loc="lower right")
        for handle in legend.legend_handles:
            handle.set_markersize(5)

        g.ax_joint.text(
            0.03,
            0.97,
            "\n".join(r2_lines),
            transform=g.ax_joint.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#808080",
                "alpha": 0.92,
            },
        )
        g.ax_marg_x.set_ylabel("Density")
        g.ax_marg_y.set_xlabel("Density")
        g.ax_marg_x.set_xlim(axis_min, axis_max)
        g.ax_marg_y.set_ylim(axis_min, axis_max)

        save_plot(f"{slugify(param)}_correlation.png", CORRELATION_DIR)
        plt.close(g.fig)


def plot_heatmaps_comparison(delta_df: pd.DataFrame) -> None:
    trains = sorted(delta_df["train"].unique(), key=str.lower)
    tests = sorted(
        [t for t in delta_df["test"].unique() if t != COMBINED_LABEL_4],
        key=str.lower,
    )
    train_labels = [_format_db_name(t) for t in trains]
    test_labels = [_format_db_name(t) for t in tests] + ["combined"]

    vmin_vmax = {"auroc": (0.5, 1.0), "ici": (0, 0.1), "brier": (0, 0.25)}

    for metric, label in zip([m.lower() for m in METRICS], METRICS):
        h_col, o_col, d_col = f"h_{metric}", f"o_{metric}", f"delta_{metric}"
        matrices = {
            k: np.full((len(trains), len(tests)), np.nan)
            for k in ("h", "o", "d")
        }

        for i, train in enumerate(trains):
            for j, test in enumerate(tests):
                mask = (delta_df["train"] == train) & (delta_df["test"] == test)
                if mask.any():
                    matrices["h"][i, j] = delta_df.loc[mask, h_col].values[0]
                    matrices["o"][i, j] = delta_df.loc[mask, o_col].values[0]
                    matrices["d"][i, j] = delta_df.loc[mask, d_col].values[0]

        combined_cols = {k: np.full((len(trains), 1), np.nan) for k in ("h", "o", "d")}
        for i, train in enumerate(trains):
            mask = (delta_df["train"] == train) & (delta_df["test"] == COMBINED_LABEL_4)
            if mask.any():
                combined_cols["h"][i, 0] = delta_df.loc[mask, h_col].values[0]
                combined_cols["o"][i, 0] = delta_df.loc[mask, o_col].values[0]
                combined_cols["d"][i, 0] = delta_df.loc[mask, d_col].values[0]
        for key in matrices:
            matrices[key] = np.hstack([matrices[key], combined_cols[key]])

        cmap, center = get_heatmap_params(metric)
        plot_vmin, plot_vmax = vmin_vmax[metric]
        delta_range = (plot_vmax - plot_vmin) * 0.2

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
        for ax, matrix, title, v_min, v_max, c_center, cbar_label in [
            (ax1, matrices["h"], f"{label} (harmonized data)",        plot_vmin,    plot_vmax,   center, label),
            (ax2, matrices["o"], f"{label} (original data)",          plot_vmin,    plot_vmax,   center, label),
            (ax3, matrices["d"], f"{label} Δ(original → harmonized)", -delta_range, delta_range, 0,      f"Δ {label}"),
        ]:
            sns.heatmap(
                matrix,
                vmin=v_min,
                vmax=v_max,
                annot=True,
                fmt=".3f",
                cmap=cmap,
                center=c_center,
                xticklabels=test_labels,
                yticklabels=train_labels,
                square=True,
                ax=ax,
                cbar_kws={"label": cbar_label},
            )
            ax.set_title(title, fontweight="bold")
            ax.set_ylabel("Train")
            ax.set_xlabel("Test")
            ax.axvline(x=len(tests), color="black", linewidth=3, zorder=5)

        save_plot(f"{metric}_heatmap_comparison.png", PLOTS_DIR)
        save_plot(f"{metric}_heatmap_comparison.pdf", PLOTS_DIR)
        plt.close()


def plot_heatmaps_per_db(metrics_df: pd.DataFrame) -> None:
    filtered_df = metrics_df[metrics_df["test_database"] != COMBINED_LABEL_4]

    vmin_vmax = {"auroc": (0.5, 1.0), "ici": (0, 0.1), "brier": (0, 0.25)}

    for metric, label in zip([m.lower() for m in METRICS], METRICS):
        matrix, trains, tests, combined_data = build_heatmap_matrix(
            filtered_df, metric, "train_sources", "test_database", COMBINED_LABEL_7
        )
        cmap, center = get_heatmap_params(metric)
        vmin, vmax = vmin_vmax[metric]

        combined_col = np.full((len(trains), 1), np.nan)
        for i, val in combined_data.items():
            combined_col[i, 0] = val
        matrix = np.hstack([matrix, combined_col])
        tests  = tests + ["combined"]
        train_labels = [_format_db_name(t) for t in trains]
        test_labels = [_format_db_name(t) for t in tests]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            matrix,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            center=center,
            xticklabels=test_labels,
            yticklabels=train_labels,
            square=True,
            ax=ax,
            cbar_kws={"label": label},
        )
        ax.axvline(x=len(tests) - 1, color="black", linewidth=3, zorder=5)
        ax.set_title(f"{label} (per Database)", fontweight="bold", fontsize=14)
        ax.set_ylabel("Train Source")
        ax.set_xlabel("Test Source")

        save_plot(f"{metric}_heatmap_per_db.png", PLOTS_DIR)
        save_plot(f"{metric}_heatmap_per_db.pdf", PLOTS_DIR)
        plt.close()


# region main
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare harmonized vs original cohorts")
    parser.add_argument("--correlations", action="store_true", help="Plot per-parameter joint correlation plots")
    parser.add_argument("--r2-dist", action="store_true", help="Compute and plot R^2 distribution")
    args = parser.parse_args()

    setup()

    print("\n=== CROSS-COHORT COMPARISON ===")

    # Load metrics
    print("\n1. Loading metrics...")
    harm_metrics = load_metrics(HARMONIZED_DIR, "single_source")
    orig_metrics = load_metrics(ORIGINAL_DIR, "single_source")
    long_metrics = load_long_format_metrics(HARMONIZED_DIR, ORIGINAL_DIR)
    print(f"harmonized: {len(harm_metrics)}, original: {len(orig_metrics)}, combined: {len(long_metrics)}")

    # Pair and compute deltas
    print("\n2. Pairing train/test...")
    paired_df, unmapped = pair_metrics(harm_metrics, orig_metrics)
    print(f"paired: {len(paired_df)}, unpaired: {len(unmapped)}")

    print("\n3. computing deltas...")
    delta_df = compute_deltas(paired_df)

    print("\n3b. AUROC TOST (margin ±0.02)...")
    tost_df = compute_auroc_tost(long_metrics, margin=0.02)
    n_ni = int(tost_df["non_inferior"].sum()) if not tost_df.empty else 0
    n_equiv = int(tost_df["equivalent"].sum()) if not tost_df.empty else 0
    print(f"{len(tost_df)} pairs: non-inf {n_ni}/{len(tost_df)}, equiv {n_equiv}/{len(tost_df)}")

    # Save results
    print("\nsaving results...")
    for metric in [m.lower() for m in METRICS]:
        cols = [
            c for c in [
                "train",
                "test",
                f"h_{metric}",          f"o_{metric}",
                f"delta_{metric}",
                f"h_{metric}_ci_lower", f"o_{metric}_ci_lower", 
                f"h_{metric}_ci_upper", f"o_{metric}_ci_upper",
            ]
            if c in delta_df.columns
        ]
        path = COMPARISON_DIR / f"{metric}_delta.csv"
        delta_df[cols].to_csv(path, index=False)
        print(f"saved {path.name}")

    if not tost_df.empty:
        tost_path = COMPARISON_DIR / "auroc_tost.csv"
        tost_df.to_csv(tost_path, index=False)
        print(f"saved {tost_path.name}")

    summary_rows = [
        {
            "Metric": label,
            "Mean": delta_df[f"delta_{m}"].mean(),
            "Std": delta_df[f"delta_{m}"].std(),
            "Min": delta_df[f"delta_{m}"].min(),
            "Max": delta_df[f"delta_{m}"].max(),
        }
        for m, label in zip([m.lower() for m in METRICS], METRICS)
    ]
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = COMPARISON_DIR / "summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{len(delta_df)} comparisons:")
    print(summary_df.to_string(index=False))
    print("\n")

    # Visualizations
    print("\n4. Creating plots...")
    plot_heatmaps_comparison(delta_df)
    plot_heatmaps_per_db(harm_metrics)

    # Load cohorts for optional visualizations
    if args.correlations or args.r2_dist:
        harm_cohort = load_cohort_frame(HARMONIZED_COHORT, "harmonized")
        orig_cohort = load_cohort_frame(ORIGINAL_COHORT,   "original")

        if args.r2_dist:
            print("\n=== R^2 Distribution (KDE + Rugplot) ===")
            print("\n4c. computing R^2 values...")
            r2_df = compute_r2_values_from_cohorts(harm_cohort, orig_cohort)
            save_r2_table(r2_df)
            
            print("\n4d. creating R^2 plot...")
            plot_r2_kde_by_source(r2_df)

        if args.correlations:
            print("\n=== Correlation Scatter Plots ===")
            plot_parameter_correlations(prepare_correlation_pairs(harm_cohort, orig_cohort, CORRELATION_ORDER))

    print("\ndone")
