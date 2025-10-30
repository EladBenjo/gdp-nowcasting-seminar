# integrated_factors.py
# Orchestrates factor extraction via DFM (unsupervised) or PLS (guided) with a common API.
# Dependencies: your local dfm.py and pls_factors.py (same folder / on PYTHONPATH).
# Author: EB's mentor

from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List

from sklearn.preprocessing import StandardScaler

# Import your provided modules
import dfm as dfm_mod
import pls_factors as pls_mod
import statsmodels.api as sm


# --------- Utilities ---------

def _train_test_split_index(idx: pd.Index, *, split_date: Optional[str] = None, test_size: float = 0.2, gap: int = 0
                            ) -> tuple[pd.Index, pd.Index]:
    """Simple, non-shuffled split by date or last fraction; returns (train_idx, test_idx)."""
    if split_date is not None:
        split_date = pd.to_datetime(split_date)
        train_idx = idx[idx <= split_date]
        test_idx = idx[idx > split_date]
    else:
        n = len(idx)
        n_test = max(1, int(np.ceil(n * test_size)))
        cut = n - n_test
        train_idx = idx[:max(0, cut - gap)]
        test_idx = idx[cut:]
    return train_idx, test_idx


def _zscore_fit_transform(X: pd.DataFrame, train_idx: pd.Index) -> tuple[pd.DataFrame, StandardScaler]:
    """Fit Z-score on train rows only; transform full X; returns (Xz_full, scaler)."""
    scaler = StandardScaler()
    X_train = X.loc[train_idx]
    Xz_train = scaler.fit_transform(X_train.values)
    Xz_full = scaler.transform(X.values)
    Xz = pd.DataFrame(Xz_full, index=X.index, columns=X.columns)
    return Xz, scaler


def _compute_r2x_from_fitted(X_z: pd.DataFrame, fitted: pd.DataFrame) -> float:
    """
    Column-wise R^2 between standardized X and its fitted values; average across columns.
    Clips to [0,1] and ignores all-NaN columns safely.
    """
    common = X_z.columns.intersection(fitted.columns)
    Xc = X_z[common]
    Fc = fitted[common].reindex(Xc.index)
    mask = ~Fc.isna()
    r2_cols = []
    for col in common:
        x = Xc[col].values
        f = Fc[col].values
        m = ~np.isnan(f)
        if m.sum() < 3:
            continue
        ss_res = ((x[m] - f[m]) ** 2).sum()
        ss_tot = (x[m] ** 2).sum() + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        r2_cols.append(np.clip(r2, 0.0, 1.0))
    return float(np.nanmean(r2_cols)) if r2_cols else np.nan


def _try_extract_dfm_loadings(res: Any, X_cols: List[str]) -> Optional[pd.DataFrame]:
    """
    Best-effort extraction of the DFM loading matrix from result params.
    Returns DataFrame [X_vars x factors] or None if structure not found.
    """
    try:
        # Many DynamicFactor models name parameters like 'loading.f1.xname'
        ps = pd.Series(res.params)
        load = ps[ps.index.str.contains("loading", case=False)]
        if load.empty:
            return None
        # Parse keys into (factor, variable)
        rows = []
        for k, v in load.items():
            # Look for patterns like 'loading.f1.x_var' or 'loading.f1.y1'
            parts = k.replace("loading.", "").split(".")
            # Fallback: split by 'f' and variable name
            if len(parts) == 1:
                parts = parts[0].split("f")
                if len(parts) == 2:
                    f_part, var_part = parts[1][0], parts[1][1:]
                    factor = f"F{f_part}"
                    varnm = var_part
                else:
                    continue
            else:
                f_part = [p for p in parts if p.startswith("f")]
                factor = f"F{f_part[0][1:]}" if f_part else "F1"
                var_candidates = [p for p in parts if p not in f_part]
                varnm = var_candidates[-1] if var_candidates else parts[-1]
            rows.append((varnm, factor, float(v)))
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["variable", "factor", "loading"])
        # Keep only variables actually present in X (rename if needed)
        df = df[df["variable"].isin(X_cols)]
        if df.empty:
            return None
        pivot = df.pivot(index="variable", columns="factor", values="loading").reindex(X_cols)
        return pivot
    except Exception:
        return None


# --------- Main Orchestrator ---------

def run_factor_extraction(
    X: pd.DataFrame,
    *,
    method: str = "dfm",                         # "dfm" or "pls"
    guide_col: str = "salaried_jobs_total_sa",   # used only for PLS
    split_date: Optional[str] = None,
    test_size: float = 0.2,
    gap: int = 0,
    # DFM options
    dfm_k_grid: List[int] = (1, 2, 3),
    dfm_p_grid: List[int] = (1, 2),
    dfm_cov_grid: List[str] = ("diagonal",),
    dfm_factors_type: str = "smoothed",          # "smoothed" (default for ex-post regression) or "filtered"
    use_init_fill: bool = False,                  # if True, apply make_init_filled() only for init
    # PLS options
    pls_n_components_grid: List[int] = tuple(range(1, 9)),
    pls_max_lag_grid: List[int] = (0, 1, 2),
    pls_cv_splits: int = 5,
    # Persistence
    save_dir: Optional[str] = None,
    return_loadings: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Unified API: run grid search + extract factors over the FULL span with the best spec.

    Returns:
        factors_df: DataFrame of factors across the full index (columns F1..FK).
        metrics: Dict of key quality metrics (aligned across methods where possible).
        grid_table: Per-spec grid results (method-specific columns).
        artifacts: Dict with extras (scalers, model objects, loadings if available).
    """
    assert method in {"dfm", "pls"}, "method must be 'dfm' or 'pls'"
    X = X.sort_index()

    # 1) Split for model selection
    train_idx, test_idx = _train_test_split_index(X.index, split_date=split_date, test_size=test_size, gap=gap)

    # 2) Optional minimal completion for DFM init (never used to estimate final model itself)
    X_for_init = dfm_mod.make_init_filled(X) if (method == "dfm" and use_init_fill) else X

    # 3) Z-score on train only, transform full panel
    Xz, scaler_X = _zscore_fit_transform(X_for_init if method == "dfm" else X, train_idx)

    artifacts: Dict[str, Any] = {"scaler_X": scaler_X}

    # ========== DFM Branch ==========
    if method == "dfm":
        # 3a) Grid on TRAIN ONLY (using z-scored inputs)
        results_table, models = dfm_mod.fit_dfm_grid(
            X_train=Xz.loc[train_idx],
            k_factors_grid=list(dfm_k_grid),
            factor_order_grid=list(dfm_p_grid),
            error_cov_grid=list(dfm_cov_grid),
            method="em",
            disp=False,
            save_dir=(os.path.join(save_dir, "dfm_grid") if save_dir else None),
        )

        # Pick best by BIC (already sorted asc by bic, aic)
        best = results_table.iloc[0]
        key = (int(best.k_factors), int(best.factor_order), best.error_cov_type)
        res_best = models[key]

        # 3b) Metrics
        metrics: Dict[str, Any] = {
            "method": "dfm",
            "k_factors": int(best.k_factors),
            "factor_order": int(best.factor_order),
            "error_cov_type": best.error_cov_type,
            "llf": float(best.llf) if pd.notna(best.llf) else np.nan,
            "aic": float(best.aic) if pd.notna(best.aic) else np.nan,
            "bic": float(best.bic) if pd.notna(best.bic) else np.nan,
            "converged": bool(best.converged),
            "iterations": None if pd.isna(best.iterations) else int(best.iterations),
        }

        # Train-fit R2X on standardized TRAIN (using fittedvalues if available)
        try:
            fitted_train = res_best.fittedvalues
            r2x_train = _compute_r2x_from_fitted(Xz.loc[fitted_train.index, :], fitted_train)
        except Exception:
            r2x_train = np.nan
        metrics["r2x_train"] = float(r2x_train)

        # Try to score out-of-sample R2X on TEST by applying the model
        r2x_test = np.nan
        try:
            res_on_test = res_best.apply(Xz.loc[test_idx])
            fitted_test = res_on_test.fittedvalues
            r2x_test = _compute_r2x_from_fitted(Xz.loc[fitted_test.index, :], fitted_test)
        except Exception:
            pass
        metrics["r2x_test"] = float(r2x_test)

        # 3c) Extract factors over the FULL span with the best model
        try:
            res_on_full = res_best.apply(Xz)  # apply to full standardized X
        except Exception:
            # Fallback: use res_best (in-sample) + reindex
            res_on_full = res_best
        F_full = dfm_mod.extract_factors(res_on_full, which=dfm_factors_type)
        # Ensure canonical names F1..FK
        F_full = F_full.copy()
        F_full.columns = [f"F{i+1}" for i in range(F_full.shape[1])]
        F_full = F_full.reindex(X.index)  # align to full timeline

        # 3d) Loadings (best-effort)
        loadings = _try_extract_dfm_loadings(res_best, list(X.columns)) if return_loadings else None
        if return_loadings:
            artifacts["loadings"] = loadings
        artifacts["model"] = res_best
        artifacts["grid_results"] = results_table

        # Optional persistence
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            results_table.to_csv(os.path.join(save_dir, "dfm_grid_results.csv"), index=False)
            with open(os.path.join(save_dir, "dfm_best_model.pkl"), "wb") as f:
                pickle.dump(res_best, f)
            # Save loadings if available
            if loadings is not None:
                loadings.to_csv(os.path.join(save_dir, "dfm_loadings.csv"))

        return F_full, metrics, results_table, artifacts

    # ========== PLS Branch ==========
    else:
        assert guide_col in X.columns, f"guide_col '{guide_col}' not found in X"
        # Use the helper grid-search that already handles scaling, CV, and factor scores
        factors_df, gridres = pls_mod.grid_search_pls_guided(
            X=X,
            guide_col=guide_col,
            train_idx=train_idx,
            test_idx=test_idx,
            n_components_grid=list(pls_n_components_grid),
            max_lag_grid=list(pls_max_lag_grid),
            n_splits_cv=int(pls_cv_splits)
        )

        # gridres contains: best_n_components, best_max_lag, cv_score, test_rmse, test_corr, r2x_train, results_table
        metrics = {
            "method": "pls",
            "guide_col": guide_col,
            "n_components": int(gridres.best_n_components),
            "max_lag": int(gridres.best_max_lag),
            "cv_rmse": float(gridres.cv_score),
            "test_rmse": None if gridres.test_rmse is None else float(gridres.test_rmse),
            "test_corr": None if gridres.test_corr is None else float(gridres.test_corr),
            "r2x_train": float(gridres.r2x_train),
        }
        grid_table = gridres.results_table.copy()

        # Refit on FULL for persistence + loadings (so you can analyze contributions)
        # We mirror PLSGuidedFactors.fit_transform but on the full index with chosen hyperparams.
        pls = pls_mod.PLSGuidedFactors(scale_X=True, scale_y=True)
        full_train_idx = X.index  # fit on all data with the chosen config
        _factors_full, _ = pls.fit_transform(
            X=X,
            guide_col=guide_col,
            train_idx=full_train_idx,
            test_idx=None,
            n_components=gridres.best_n_components,
            max_lag=gridres.best_max_lag
        )
        # Replace factors_df with factors from full fit (ensures factors over all rows possible)
        factors_df = _factors_full

        # Artifacts for interpretability
        artifacts["pls_x_loadings"] = getattr(pls.model_, "x_loadings_", None)
        artifacts["pls_y_loadings"] = getattr(pls.model_, "y_loadings_", None)
        artifacts["feature_columns"] = pls.train_cols_
        artifacts["n_components"] = pls.n_components_
        artifacts["max_lag"] = pls.max_lag_
        artifacts["scaler_y"] = pls.scaler_y_
        artifacts["pls_model"] = pls.model_
        artifacts["grid_results"] = grid_table

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            grid_table.to_csv(os.path.join(save_dir, "pls_grid_results.csv"), index=False)
            # Persist the fitted full PLS pack
            with open(os.path.join(save_dir, "pls_full_fit.pkl"), "wb") as f:
                pickle.dump({
                    "scaler_X": pls.scaler_X_,
                    "scaler_y": pls.scaler_y_,
                    "pls_model": pls.model_,
                    "feature_columns": pls.train_cols_,
                    "n_components": pls.n_components_,
                    "max_lag": pls.max_lag_,
                    "guide_col": guide_col
                }, f)
            # Save loadings for quick inspection
            if artifacts["pls_x_loadings"] is not None:
                pd.DataFrame(
                    artifacts["pls_x_loadings"],
                    index=pls.train_cols_,
                    columns=[f"F{i+1}" for i in range(pls.n_components_)]
                ).to_csv(os.path.join(save_dir, "pls_x_loadings.csv"))

        return factors_df, metrics, grid_table, artifacts
