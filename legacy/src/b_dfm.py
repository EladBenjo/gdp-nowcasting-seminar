import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------- 1) Split to train/test (already standardized earlier) ----------
# Example: adjust the split dates to your project split
X_train = X_std.loc[:'2018-12-31'].copy()
X_test  = X_std.loc['2019-01-01':].copy()

# ---------- 2) Build an "initialization-only" copy with within-month fills ----------
def make_within_month_filled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an init-only dataframe with within-month fills:
    - For each month, forward-fill then backfill so that all days in the month have values.
    - This avoids leaking information across months and only serves PCA/initialization.
    """
    # Group by month period; inside each month, ffill then bfill
    filled = (df
              .groupby(df.index.to_period("M"))
              .apply(lambda g: g.ffill().bfill()))
    # The groupby will create a PeriodIndex on the outer level; drop it back to DatetimeIndex
    filled.index = filled.index.droplevel(0)
    filled = filled.sort_index()
    return filled

X_train_init = make_within_month_filled(X_train)

# ---------- 3) Fit a small DFM on the filled data to get start_params ----------
# Try 1–3 factors and order 1–2 in a loop if you like; here a single spec example:
k_factors = 2
factor_order = 1

mod_init = sm.tsa.DynamicFactor(
    X_train_init,
    k_factors=k_factors,
    factor_order=factor_order,
    error_cov_type="diagonal"
)

# Two-stage optimization often stabilizes initialization:
init_res_powell = mod_init.fit(method="powell", disp=False)
init_res = mod_init.fit(start_params=init_res_powell.params, disp=False)

start_params = init_res.params  # these params match the full model's dimension

# ---------- 4) Fit the full DFM on the ORIGINAL data (with NaNs) using start_params ----------
mod_full = sm.tsa.DynamicFactor(
    X_train,
    k_factors=k_factors,
    factor_order=factor_order,
    error_cov_type="diagonal"
)

# Again, two-stage can help; crucially, no PCA is triggered now since we pass start_params
res_powell = mod_full.fit(start_params=start_params, method="powell", disp=False)
res = mod_full.fit(start_params=res_powell.params, disp=False)

print(res.summary())

# ---------- 5) Extract daily factors for train (use filtered for real-time) ----------
F_train = res.factors.filtered  # DataFrame with columns like 'f1', 'f2', ...

# ---------- 6) Apply the trained model to TEST data (standardized with the same μ,σ) ----------
# This runs the Kalman filter on test using the learned parameters (no re-estimation)
res_test = res.apply(X_test)
F_test = res_test.factors.filtered

# (Later) aggregate factors to quarterly and proceed to Bridge/MIDAS
