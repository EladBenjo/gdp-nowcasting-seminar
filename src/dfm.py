import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Tuple, List, Any
import os
import pickle


def fit_dfm_grid(
    X_train: pd.DataFrame,
    k_factors_grid: List[int] = (1, 2, 3),
    factor_order_grid: List[int] = (1, 2),
    error_cov_grid: List[str] = ("diagonal",),
    method: str = "em",
    disp: bool = False,
    save_dir: [str] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int, str], Any]]:
    import os
    import pickle

    rows = []
    models: Dict[Tuple[int, int, str], Any] = {}

    X_train = X_train.sort_index()

    print("Starting grid search...")

    for k, p, cov in itertools.product(k_factors_grid, factor_order_grid, error_cov_grid):
        key = (k, p, cov)
        print(f"--> Fitting model: k_factors={k}, factor_order={p}, error_cov_type={cov}")
        try:
            mod = sm.tsa.DynamicFactor(
                X_train,
                k_factors=k,
                factor_order=p,
                error_cov_type=cov
            )
            res = mod.fit(method=method, maxiter=100, disp=disp)
            models[key] = res

            print(f"    ✅ Fit successful. LLF={res.llf:.2f}, BIC={res.bic:.2f}")

            # Save the fitted model as pickle
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"dfm_k{k}_p{p}_cov{cov}.pkl"
                filepath = os.path.join(save_dir, filename)
                with open(filepath, "wb") as f:
                    pickle.dump(res, f)
                print(f"    📦 Saved model to: {filepath}")

            rows.append({
                "k_factors": k,
                "factor_order": p,
                "error_cov_type": cov,
                "llf": res.llf,
                "aic": res.aic,
                "bic": res.bic,
                "nobs": res.nobs,
                "converged": getattr(res, "mle_retvals", {}).get("converged", True),
                "iterations": getattr(res, "mle_retvals", {}).get("iterations", np.nan),
            })
        except Exception as e:
            print(f"    ❌ Fit failed: {str(e)}")
            rows.append({
                "k_factors": k,
                "factor_order": p,
                "error_cov_type": cov,
                "llf": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "nobs": len(X_train),
                "converged": False,
                "iterations": np.nan,
                "error": str(e),
            })

    print("✅ Grid search completed.")
    results_table = pd.DataFrame(rows)
    results_table = results_table.sort_values(["bic", "aic"], ascending=[True, True]).reset_index(drop=True)
    return results_table, models


def extract_factors(res_obj: Any, which: str = "filtered") -> pd.DataFrame:
    """
    Extract factors from a fitted DynamicFactor results object.
    which: "filtered" (one-sided, for real-time) or "smoothed" (two-sided, for ex-post analysis).
    """
    if which == "filtered":
        return res_obj.factors.filtered
    elif which == "smoothed":
        return res_obj.factors.smoothed
    else:
        raise ValueError("which must be 'filtered' or 'smoothed'")


def max_possible_factors(X: pd.DataFrame) -> int:
    """
    Compute the theoretical maximum number of factors possible in PCA initialization.
    It is the minimum between the number of non-missing observations and the number of variables.
    
    Args:
        X (pd.DataFrame): Input dataframe (train set, standardized).
        
    Returns:
        int: Maximum number of factors allowed by PCA initialization.
    """
    # Count how many rows have no NaN at all
    n_obs_effective = X.dropna(how="any").shape[0]
    n_vars = X.shape[1]
    return min(n_obs_effective, n_vars)


# Example usage:
# max_factors = max_possible_factors(X_train_std)
# print("Max possible factors =", max_factors)

# ===== Example usage (skeleton) =====
# X_train: daily DataFrame, stationary + Z-scored on train only

# 1) Run grid
# results_table, models = fit_dfm_grid(
#     X_train=X_train_std,
#     k_factors_grid=[1, 2, 3],
#     factor_order_grid=[1, 2],
#     error_cov_grid=["diagonal"],  # start simple
#     method="em",
#     disp=False
# )

# 2) Pick a candidate (e.g., the first row = best BIC)
# best = results_table.iloc[0]
# key = (int(best.k_factors), int(best.factor_order), best.error_cov_type)
# res_best = models[key]

# 3) Extract daily factors (use 'filtered' for real-time/nowcasting)
# F_daily = extract_factors(res_best, which="filtered")  # DataFrame with columns F1, F2, ...

# 4) (Later) Apply the trained model to new data (standardized with the same train μ,σ):
# res_on_test = res_best.apply(X_test_std)
# F_daily_test = extract_factors(res_on_test, which="filtered")


def make_init_filled(df: pd.DataFrame, monthly_prefix: str = "m_") -> pd.DataFrame:
    """
    Build an 'init-only' dataframe suitable for PCA initialization:
    1) Within-month fill (ffill then bfill inside each month) – no cross-month leakage.
    2) For months that remain entirely missing in a column, build a monthly series,
       interpolate across missing months, then broadcast the monthly value back to all days in those months.
    Notes:
    - This is ONLY for initialization. Do NOT use this filled frame for final model estimation.
    - Daily series are left as-is by design; monthly series are the target for stabilization.
    """
    df = df.copy()
    # Pass 1: within-month fill (no crossing months)
    filled = (df.groupby(df.index.to_period("M"))
                .apply(lambda g: g.ffill().bfill()))
    filled.index = filled.index.droplevel(0)
    filled = filled.sort_index()

    # Pass 2: handle months that are still fully missing for a given column
    # We will only apply this to 'monthly' columns (heuristic: prefix)
    monthly_cols = [c for c in filled.columns if c.startswith(monthly_prefix)]

    if monthly_cols:
        # Helper: A PeriodIndex over months in the full date span
        months = pd.period_range(filled.index.min().to_period("M"),
                                 filled.index.max().to_period("M"), freq="M")

        # For each monthly column, create a monthly series, interpolate missing months,
        # then map back to daily rows for months that remain empty after Pass 1.
        month_of = filled.index.to_period("M")

        for col in monthly_cols:
            # Detect months that are still entirely NaN after pass 1
            # (month-by-month check)
            null_by_month = filled[col].isna().groupby(month_of).all()
            months_all_nan = null_by_month[null_by_month].index

            if len(months_all_nan) == 0:
                continue  # nothing to do

            # Build a pure monthly series: pick one value per month (e.g., last valid within the month)
            # If a month has no observation at all, it will be NaN here initially
            monthly_vals = (filled[col]
                            .groupby(month_of)
                            .last())  # could also use .mean() or .first(), depending on convention

            # Reindex to full monthly span and interpolate across months
            monthly_vals = monthly_vals.reindex(months)
            # Linear interpolation over months (no daily info used)
            monthly_vals_interped = monthly_vals.interpolate(method="linear", limit_direction="both")

            # Broadcast back to daily for months that were fully NaN
            # Create a map period->value for the interpolated monthly series
            month_to_value = monthly_vals_interped.to_dict()

            # For each month that was empty, set all its days to the monthly interpolated value
            mask_empty_months = month_of.isin(months_all_nan)
            filled.loc[mask_empty_months, col] = month_of[mask_empty_months].map(month_to_value).values

    return filled
