# pls_factors.py
# Author: EB's mentor
# Purpose: Time-series split + PLS-guided factor extraction with grid search
# Comments: All comments/docstrings in English (PEP8 style).

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# -----------------------------
# Utilities
# -----------------------------

def time_series_train_test_split(
    df: pd.DataFrame,
    *,
    target_col: Optional[str] = None,
    split_date: Optional[pd.Timestamp | str] = None,
    test_size: Optional[float] = None,
    gap: int = 0
) -> Tuple[pd.Index, pd.Index]:
    """
    Create non-shuffled train/test split for time series.

    Args:
        df: Indexed by DatetimeIndex (or any sorted index).
        target_col: Optional; if provided and contains NaNs at the tail/head, the split is still done on index only.
        split_date: Absolute split by date (train <= split_date, test > split_date). If given, overrides test_size.
        test_size: Fraction in (0,1). If provided, uses last `test_size` fraction as test.
        gap: Number of observations skipped between train end and test start to reduce leakage.

    Returns:
        (train_idx, test_idx): index lists for train and test.
    """
    if split_date is not None:
        split_date = pd.to_datetime(split_date)
        train_idx = df.index[df.index <= split_date]
        test_idx = df.index[df.index > split_date]
    elif test_size is not None:
        assert 0 < test_size < 1, "test_size must be in (0,1)"
        n = len(df)
        n_test = int(np.ceil(n * test_size))
        cut = n - n_test
        train_idx = df.index[:max(0, cut - gap)]
        test_idx = df.index[cut:]
    else:
        raise ValueError("Provide either split_date or test_size.")

    if gap > 0 and split_date is not None:
        # Apply gap when using split_date
        # Find the last train loc and shift test start forward by `gap`.
        if len(train_idx) > 0:
            last_train_pos = np.where(df.index.isin([train_idx[-1]]))[0][0]
            test_start = min(len(df.index), last_train_pos + 1 + gap)
            test_idx = df.index[test_start:]
        else:
            # No train rows; leave as is
            pass

    return train_idx, test_idx


def add_lags(X: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    """
    Create lagged versions of all columns in X (1..max_lag), then concat with X (t).
    Rows with NaNs due to lagging are kept; caller should align by dropna if needed.

    Args:
        X: Features panel (time-indexed).
        max_lag: Number of lags to add.

    Returns:
        DataFrame with original columns and lagged copies: col_l1, col_l2, ...
    """
    if max_lag <= 0:
        return X.copy()
    cols = []
    for L in range(1, max_lag + 1):
        lagged = X.shift(L)
        lagged.columns = [f"{c}_l{L}" for c in X.columns]
        cols.append(lagged)
    return pd.concat([X] + cols, axis=1)


def _r2x_from_pls(X_z: np.ndarray, model: PLSRegression) -> float:
    """
    Approximate fraction of variance in X explained by the PLS components (R2X).

    Note: scikit-learn's PLS does not expose R2X directly. We reconstruct X_hat
    from scores and loadings and compute column-wise R^2 on standardized X.

    Args:
        X_z: Standardized X (mean 0, var 1), shape (n, p).
        model: Fitted PLSRegression with attributes x_scores_ and x_loadings_.

    Returns:
        R2X in [0,1].
    """
    T_scores = model.x_scores_                          # (n, n_comp)
    P_load = model.x_loadings_                          # (p, n_comp)
    X_hat = T_scores @ P_load.T                         # (n, p)
    # Column-wise R^2 on standardized scale
    ss_res = ((X_z - X_hat) ** 2).sum(axis=0)
    ss_tot = (X_z ** 2).sum(axis=0) + 1e-12
    r2_cols = 1.0 - ss_res / ss_tot
    return float(np.clip(np.nanmean(r2_cols), 0.0, 1.0))


@dataclass
class PLSGridResult:
    best_n_components: int
    best_max_lag: int
    cv_score: float
    test_rmse: Optional[float]
    test_corr: Optional[float]
    r2x_train: float
    results_table: pd.DataFrame


class PLSGuidedFactors:
    """
    Extract 'guided' factors using PLS where the guide variable y comes from
    within the same panel X. After fitting on train, transforms the entire
    sample to K factors (x_scores_).

    Typical usage:
        pls = PLSGuidedFactors(...)
        factors_df, metrics = pls.fit_transform(X, guide_col, train_idx, test_idx)
    """
    def __init__(self, scale_X: bool = True, scale_y: bool = True):
        self.scale_X = scale_X
        self.scale_y = scale_y
        self.scaler_X_: Optional[StandardScaler] = None
        self.scaler_y_: Optional[StandardScaler] = None
        self.model_: Optional[PLSRegression] = None
        self.train_cols_: Optional[List[str]] = None
        self.n_components_: Optional[int] = None
        self.max_lag_: Optional[int] = None

    def _prep_xy(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler], Optional[StandardScaler]]:
        scaler_X = StandardScaler() if self.scale_X else None
        scaler_y = StandardScaler() if self.scale_y else None
        Xz = scaler_X.fit_transform(X.values) if scaler_X else X.values
        yz = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel() if scaler_y else y.values
        return Xz, yz, scaler_X, scaler_y

    def _transform_X(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler_X_:
            return self.scaler_X_.transform(X.values)
        return X.values

    def _inverse_y(self, y_hat_z: np.ndarray) -> np.ndarray:
        if self.scaler_y_:
            return self.scaler_y_.inverse_transform(y_hat_z.reshape(-1, 1)).ravel()
        return y_hat_z

    def fit_transform(
        self,
        X: pd.DataFrame,
        guide_col: str,
        train_idx: pd.Index,
        test_idx: Optional[pd.Index] = None,
        n_components: int = 3,
        max_lag: int = 0
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Fit PLS on train and return factor scores for the entire sample.

        Args:
            X: Panel (time-indexed), includes guide_col among its columns.
            guide_col: The in-panel 'guide' variable name (supervises PLS).
            train_idx: Index of training rows.
            test_idx: Optional test rows (for metrics).
            n_components: Number of PLS components (factors).
            max_lag: Add lags 1..max_lag to X before fitting.

        Returns:
            factors_df: DataFrame of K factors over the full index (F1..FK).
            metrics: Dict with R2X (train), and if test_idx given: RMSE and corr on guide.
        """
        assert guide_col in X.columns, "guide_col must be a column in X"
        # Build feature matrix (optionally with lags), and align with y
        X_aug = add_lags(X.drop(columns=[guide_col]), max_lag=max_lag)
        # Keep also the contemporaneous guide as a regressed target
        y = X[guide_col]
        XY = pd.concat([X_aug, y], axis=1).dropna()
        # Train subset
        XY_train = XY.loc[XY.index.intersection(train_idx)]
        X_tr = XY_train.drop(columns=[guide_col])
        y_tr = XY_train[guide_col]

        # Standardize and fit
        Xz, yz, self.scaler_X_, self.scaler_y_ = self._prep_xy(X_tr, y_tr)
        pls = PLSRegression(n_components=n_components)
        pls.fit(Xz, yz)

        # Save fitted state
        self.model_ = pls
        self.train_cols_ = list(X_tr.columns)
        self.n_components_ = n_components
        self.max_lag_ = max_lag

        # Compute train R2X on standardized X
        r2x_train = _r2x_from_pls(Xz, pls)

        # Produce factor scores for the full sample (where features exist)
        X_full_aug = add_lags(X.drop(columns=[guide_col]), max_lag=max_lag)
        # Align to training feature set (columns/order)
        X_full_aug = X_full_aug.reindex(columns=self.train_cols_)
        # Factors where no NaNs
        valid_mask = ~X_full_aug.isna().any(axis=1)
        factors = pd.DataFrame(index=X_full_aug.index, columns=[f"F{i+1}" for i in range(n_components)], dtype=float)

        if valid_mask.any():
            X_full_valid = X_full_aug.loc[valid_mask]
            X_full_z = self._transform_X(X_full_valid)
            T_scores = pls.transform(X_full_z)  # x_scores_
            factors.loc[valid_mask, :] = T_scores

        metrics = {"r2x_train": float(r2x_train)}

        # If test provided, evaluate prediction of the guide variable
        if test_idx is not None:
            XY_test = XY.loc[XY.index.intersection(test_idx)]
            if len(XY_test) > 0:
                X_te = XY_test.drop(columns=[guide_col])
                y_te = XY_test[guide_col].values
                X_te_z = self._transform_X(X_te)
                y_hat_z = pls.predict(X_te_z).ravel()
                y_hat = self._inverse_y(y_hat_z)
                rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
                corr = float(np.corrcoef(y_te, y_hat)[0, 1]) if len(y_te) > 2 else np.nan
                metrics.update({"test_rmse": rmse, "test_corr": corr})

        return factors, metrics

    def transform_with_fitted(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted PLS to new X to obtain factor scores.
        """
        assert self.model_ is not None, "Model not fitted yet."
        X_aug = add_lags(X.drop(columns=[self.guide_col_]), max_lag=self.max_lag_)  # type: ignore
        X_aug = X_aug.reindex(columns=self.train_cols_)
        valid_mask = ~X_aug.isna().any(axis=1)
        factors = pd.DataFrame(index=X_aug.index, columns=[f"F{i+1}" for i in range(self.n_components_)], dtype=float)
        if valid_mask.any():
            X_z = self._transform_X(X_aug.loc[valid_mask])
            T_scores = self.model_.transform(X_z)
            factors.loc[valid_mask, :] = T_scores
        return factors


def grid_search_pls_guided(
    X: pd.DataFrame,
    guide_col: str,
    train_idx: pd.Index,
    test_idx: Optional[pd.Index],
    n_components_grid: List[int] = list(range(1, 9)),
    max_lag_grid: List[int] = [0],
    n_splits_cv: int = 5
) -> Tuple[pd.DataFrame, PLSGridResult]:
    """
    Grid search over number of factors (n_components) and optional max_lag.
    Uses expanding-window CV on the training set to score out-of-fold RMSE on guide_col.

    Args:
        X: Panel (time-indexed).
        guide_col: In-panel guide variable name.
        train_idx: Train rows.
        test_idx: Optional test rows (for final evaluation).
        n_components_grid: e.g., [1..8].
        max_lag_grid: e.g., [0, 1, 2].
        n_splits_cv: Expanding TimeSeriesSplit folds.

    Returns:
        factors_best_df: Factors (F1..FK) over full index for the best configuration.
        grid_result: PLSGridResult with best params and metrics table.
    """
    assert guide_col in X.columns, "guide_col must exist in X"

    # Assemble base matrices once
    X0 = X.drop(columns=[guide_col])
    y0 = X[guide_col]

    rows = []
    best_score = np.inf
    best_tuple = (None, None)
    best_model_pack = None

    # Prepare CV iterator over the training segment
    train_mask = X.index.isin(train_idx)
    X_train_full = X0.loc[train_mask]
    y_train_full = y0.loc[train_mask]

    # We will build lagged features inside the loop to avoid huge memory if grids are large
    for max_lag in max_lag_grid:
        X_train_aug = add_lags(X_train_full, max_lag=max_lag)
        XY_train = pd.concat([X_train_aug, y_train_full], axis=1).dropna()
        if XY_train.empty:
            continue

        X_tr_all = XY_train.drop(columns=[guide_col])
        y_tr_all = XY_train[guide_col].values

        # Standardize within CV each time to avoid leakage
        tscv = TimeSeriesSplit(n_splits=min(n_splits_cv, max(2, len(XY_train)//10)))

        for n_comp in n_components_grid:
            cv_rmses = []
            for tr_idx, val_idx in tscv.split(X_tr_all):
                X_tr, X_val = X_tr_all.iloc[tr_idx], X_tr_all.iloc[val_idx]
                y_tr, y_val = y_tr_all[tr_idx], y_tr_all[val_idx]

                scaler_X = StandardScaler()
                scaler_y = StandardScaler()

                X_tr_z = scaler_X.fit_transform(X_tr.values)
                y_tr_z = scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()

                pls = PLSRegression(n_components=n_comp)
                pls.fit(X_tr_z, y_tr_z)

                X_val_z = scaler_X.transform(X_val.values)
                y_hat_z = pls.predict(X_val_z).ravel()
                y_hat = scaler_y.inverse_transform(y_hat_z.reshape(-1, 1)).ravel()

                rmse = np.sqrt(mean_squared_error(y_val, y_hat))
                cv_rmses.append(rmse)

            cv_rmse = float(np.mean(cv_rmses))
            rows.append({
                "n_components": n_comp,
                "max_lag": max_lag,
                "cv_rmse": cv_rmse
            })

            if cv_rmse < best_score:
                best_score = cv_rmse
                best_tuple = (n_comp, max_lag)
                # Keep a fitted model on the full train for R2X estimate later:
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X_tr_z_full = scaler_X.fit_transform(X_tr_all.values)
                y_tr_z_full = scaler_y.fit_transform(y_tr_all.reshape(-1, 1)).ravel()
                pls_full = PLSRegression(n_components=n_comp).fit(X_tr_z_full, y_tr_z_full)
                r2x_train = _r2x_from_pls(X_tr_z_full, pls_full)
                best_model_pack = (scaler_X, scaler_y, pls_full, list(X_tr_all.columns), r2x_train)

    results_table = pd.DataFrame(rows).sort_values(["cv_rmse", "max_lag", "n_components"]).reset_index(drop=True)

    if best_model_pack is None or best_tuple[0] is None:
        raise RuntimeError("Grid search failed to fit any model — check data and grids.")

    best_n_components, best_max_lag = best_tuple
    scaler_X, scaler_y, pls_full, feat_cols, r2x_train = best_model_pack

    # Final test evaluation on held-out test_idx (if provided)
    test_rmse = None
    test_corr = None
    if test_idx is not None and len(test_idx) > 0:
        # Build features over the *whole* sample with best lag config
        X_aug = add_lags(X0, max_lag=best_max_lag)
        # Align feature columns exactly
        X_aug = X_aug.reindex(columns=feat_cols)
        # Joint DF with y to drop NaNs
        XY = pd.concat([X_aug, y0], axis=1).dropna()
        XY_test = XY.loc[XY.index.intersection(test_idx)]
        if len(XY_test) > 0:
            X_te = XY_test[feat_cols]
            y_te = XY_test[guide_col].values
            X_te_z = scaler_X.transform(X_te.values)
            y_hat_z = pls_full.predict(X_te_z).ravel()
            y_hat = scaler_y.inverse_transform(y_hat_z.reshape(-1, 1)).ravel()
            test_rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
            test_corr = float(np.corrcoef(y_te, y_hat)[0, 1]) if len(y_te) > 2 else np.nan

    # Produce factors for the full index using best config (scores from the fitted train model)
    # Note: We compute scores for any row where all features are available
    X_full_aug = add_lags(X0, max_lag=best_max_lag).reindex(columns=feat_cols)
    valid_mask = ~X_full_aug.isna().any(axis=1)
    factors = pd.DataFrame(index=X.index, columns=[f"F{i+1}" for i in range(best_n_components)], dtype=float)
    if valid_mask.any():
        X_full_z = scaler_X.transform(X_full_aug.loc[valid_mask].values)
        T_scores = pls_full.transform(X_full_z)  # x_scores_
        factors.loc[valid_mask, :] = T_scores

    grid_result = PLSGridResult(
        best_n_components=best_n_components,
        best_max_lag=best_max_lag,
        cv_score=float(best_score),
        test_rmse=test_rmse,
        test_corr=test_corr,
        r2x_train=float(r2x_train),
        results_table=results_table
    )
    return factors, grid_result

