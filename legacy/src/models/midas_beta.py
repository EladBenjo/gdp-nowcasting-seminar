import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize

class BetaMIDASRegressor:
    """
    MIDAS with normalized Beta-polynomial lag weights.
    - Supports multiple high-frequency regressors, each with its own lag-length K_i.
    - Optimizes ONLY the Beta-shape params (per regressor). Given weights -> fits OLS (closed-form).
    - Ensures positive weights that sum to 1 for each regressor (good interpretability & stability).

    X format:
      X is a list of arrays: [X1, X2, ...]
      where Xi has shape (T, K_i): T aligned low-frequency periods, K_i lags for regressor i
    y: target (T,)

    Notes:
    - Comments in English only (as requested).
    """

    def __init__(self, add_intercept=True, ridge=0.0, weight_normalization=True, random_state=0):
        """
        add_intercept: include intercept in the final OLS.
        ridge: optional L2 on beta coefficients (NOT on Beta-weights). 0.0 = OLS.
        weight_normalization: keep weights strictly positive and summing to 1.
        """
        self.add_intercept = add_intercept
        self.ridge = float(ridge)
        self.weight_normalization = bool(weight_normalization)
        self.random_state = int(random_state)

        self.theta_ = None        # shape params per regressor: a,b (>1), stored as unconstrained z, mapped via a=1+exp(z1), b=1+exp(z2)
        self.beta_ = None         # OLS/Ridge coefficients on transformed regressors (+ intercept if used)
        self.weights_ = None      # list of arrays with final lag weights per regressor
        self.K_list_ = None
        self.n_regressors_ = None

    @staticmethod
    def _beta_weights(K, a, b, normalize=True):
        """
        Compute Beta-polynomial weights for lags 1..K (from most recent=1 to oldest=K).
        a,b > 1. Uses scaled j/K in (0,1].
        """
        j = np.arange(1, K + 1, dtype=float)
        x = j / K
        w = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
        if normalize:
            s = w.sum()
            if s <= 0 or not np.isfinite(s):
                # fallback to uniform if something went wrong numerically
                w = np.ones_like(w) / K
            else:
                w = w / s
        return w

    @staticmethod
    def _ab_from_z(z):
        """
        Map unconstrained z=(z1,z2) to a,b>1 using a=1+exp(z1), b=1+exp(z2).
        """
        a = 1.0 + np.exp(z[0])
        b = 1.0 + np.exp(z[1])
        return a, b

    @staticmethod
    def _ols_closed_form(X, y, ridge=0.0):
        """
        Solve (X'X + λI)β = X'y . If ridge=0 -> OLS; else Ridge with λ on all betas (except intercept handled outside).
        """
        XtX = X.T @ X
        Xty = X.T @ y
        if ridge > 0:
            # Add ridge only to non-intercept columns (assume intercept is col 0 if present)
            R = np.eye(X.shape[1])
            R[0, 0] = 0.0  # do not penalize intercept
            beta = inv(XtX + ridge * R) @ Xty
        else:
            beta = inv(XtX) @ Xty
        return beta

    def _build_transformed_X(self, X_list, theta_vec):
        """
        Given the current theta vector -> build transformed low-frequency design:
        For each regressor i: z_i(t) = sum_{k=1..K_i} w_{i,k} * X_i(t,k)
        Returns matrix Z with columns [z_1, z_2, ..., z_m] (+ intercept if requested).
        """
        cols = []
        self.weights_ = []
        offset = 0
        for i, Xi in enumerate(X_list):
            K_i = Xi.shape[1]
            z_i = theta_vec[offset:offset + 2]          # (z1,z2) for regressor i
            a_i, b_i = self._ab_from_z(z_i)
            w_i = self._beta_weights(K_i, a_i, b_i, normalize=self.weight_normalization)
            self.weights_.append(w_i)
            cols.append((Xi * w_i).sum(axis=1))         # (T,)
            offset += 2

        Z = np.column_stack(cols)                       # (T, m)
        if self.add_intercept:
            Z = np.column_stack([np.ones(Z.shape[0]), Z])
        return Z

    def _sse_given_theta(self, theta_vec, X_list, y):
        """
        Objective for minimize: SSE of (y - Z(theta) @ beta(theta)).
        beta(theta) is found in closed-form OLS/Ridge given Z(theta).
        """
        Z = self._build_transformed_X(X_list, theta_vec)
        beta = self._ols_closed_form(Z, y, ridge=self.ridge)
        yhat = Z @ beta
        resid = y - yhat
        sse = np.dot(resid, resid)
        if not np.isfinite(sse):  # guard against NaNs/Infs
            sse = 1e12
        return sse

    def fit(self, y, X_list):
        """
        Fit the model.
        y: (T,)
        X_list: list of arrays; Xi has shape (T, K_i)
        """
        y = np.asarray(y).reshape(-1)
        self.n_regressors_ = len(X_list)
        self.K_list_ = [Xi.shape[1] for Xi in X_list]

        # Initialize theta (z params) per regressor: start near Beta(2,2)
        np.random.seed(self.random_state)
        theta0 = []
        for _ in range(self.n_regressors_):
            # a=1+exp(z1) ~ 2 -> z1 ~ ln(1) = 0 ; same for b
            theta0.extend([0.0 + 0.01*np.random.randn(), 0.0 + 0.01*np.random.randn()])
        theta0 = np.array(theta0)

        # Optimize only theta; beta is closed-form inside the objective
        res = minimize(self._sse_given_theta, theta0, args=(X_list, y), method="L-BFGS-B")
        self.theta_ = res.x

        # Final Z, beta (store)
        Z = self._build_transformed_X(X_list, self.theta_)
        self.beta_ = self._ols_closed_form(Z, y, ridge=self.ridge)
        return self

    def predict(self, X_list):
        """
        Predict y for new data. X_list must have the same shapes (T_new, K_i).
        """
        Z = self._build_transformed_X(X_list, self.theta_)
        yhat = Z @ self.beta_
        return yhat

    def get_lag_weights(self):
        """
        Return list of lag-weights per regressor [w1, w2, ...], each shape (K_i,).
        """
        return self.weights_

    def get_beta(self):
        """
        Return final linear coefficients on transformed regressors (+ intercept if used).
        """
        return self.beta_
