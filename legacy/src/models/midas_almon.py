import numpy as np
from sklearn.linear_model import LinearRegression

class AlmonMIDASRegressor:
    def __init__(self, degree=2):
        """
        Initialize Almon MIDAS regressor.

        Parameters:
        degree (int): Degree of the Almon polynomial (usually 2 or 3).
        """
        self.degree = degree
        self.theta_ = None
        self.beta_ = None
        self.model_ = None
        self.weights_ = None

    def _design_matrix(self, n_lags):
        """
        Create the Almon lag design matrix: a Vandermonde matrix of lag indices.

        Returns:
        A matrix of shape (n_lags, degree+1)
        """
        lag_indices = np.arange(1, n_lags + 1)
        return np.vander(lag_indices, N=self.degree + 1, increasing=True)

    def _compute_almon_weights(self, theta):
        """
        Compute weights from the polynomial coefficients.
        """
        Z = self._design_matrix(len(theta))
        return Z @ theta

    def fit(self, y, X):
        """
        Fit the Almon MIDAS model.

        Parameters:
        y (np.array): Low-frequency target (T,)
        X (np.array): High-frequency lags matrix (T, K)
        """
        T, K = X.shape

        # Step 1: Create Almon lag matrix (Z)
        Z = self._design_matrix(K)  # (K, degree+1)

        # Step 2: Optimize theta via least squares
        # We rewrite the model as: y = beta * (Z @ theta).T @ X.T + error

        # Construct regressor: for each time t, compute X_t @ Z @ theta
        # We'll fit it by minimizing residuals via linear regression over thetas

        # Initial guess for theta
        theta0 = np.zeros(self.degree + 1)

        # Define loss function
        def loss(theta):
            weights = Z @ theta
            filtered_X = (X * weights).sum(axis=1)  # shape (T,)
            residuals = y - filtered_X
            return np.sum(residuals**2)

        # Use numpy optimization
        from scipy.optimize import minimize
        result = minimize(loss, theta0)

        self.theta_ = result.x
        self.weights_ = Z @ self.theta_

        # Final regression: y = beta * (X * weights).sum(axis=1)
        transformed_X = (X * self.weights_).sum(axis=1).reshape(-1, 1)

        self.model_ = LinearRegression()
        self.model_.fit(transformed_X, y)
        self.beta_ = self.model_.coef_[0]

        return self

    def predict(self, X):
        """
        Predict target values for given X.

        Parameters:
        X (np.array): High-frequency lags matrix (T, K)

        Returns:
        y_pred (np.array): Predicted values
        """
        transformed_X = (X * self.weights_).sum(axis=1).reshape(-1, 1)
        return self.model_.predict(transformed_X)

    def get_lag_weights(self):
        """
        Return the full lag weights (w_j).
        """
        return self.weights_
