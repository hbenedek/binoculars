"""Module for linear regression."""

from typing import Self

from binoculars import LinearRegressionRust


class LinearRegression:
    """Least squares linear regression with Rust backend."""

    def __init__(self) -> None:
        self._rustobj = LinearRegressionRust()
        self.weights = None
        self.method = None
        self.is_bias = False

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return f"LinearModel(weights={self.weights})"

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"LinearModel(weights={self.weights})"

    def fit(self, X: list[list[float]], y: list[float]) -> Self:
        """Fit the model using X as training data and y as target values."""
        self._rustobj.fit(X, y)
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict using the linear model."""
        return self._rustobj.predict(X)

    def get_weights(self) -> list[float]:
        """Get parameters for this estimator."""
        return self._rustobj.get_weights()

    def set_weights(self, weights: list[float]) -> Self:
        """Set the parameters of this estimator."""
        self._rustobj.set_weights(weights)
        return self

    def with_bias(self, with_bias: bool) -> Self:
        """Set the parameters of this estimator."""
        self._rustobj.with_bias(with_bias)
        return self

    def with_solver(self, method: str) -> Self:
        """Set the parameters of this estimator."""
        assert method in ["normal", "ls"]
        self._rustobj.with_method(method)
        return self


class LogisticRegression:
    """Logistic regression with Rust backend."""

    def __init__(self) -> None:
        self._rustobj = LogisticRegressionRust()
        self.weights = None
        self.method = None
        self.is_bias = False

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return f"LogisticModel(weights={self.weights})"

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"LogisticModel(weights={self.weights})"

    def fit(self, X: list[list[float]], y: list[float]) -> Self:
        """Fit the model using X as training data and y as target values."""
        self._rustobj.fit(X, y)
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        """Predict using the linear model."""
        return self._rustobj.predict(X)

    def get_weights(self) -> list[float]:
        """Get parameters for this estimator."""
        return self._rustobj.get_weights()

    def set_weights(self, weights: list[float]) -> Self:
        """Set the parameters of this estimator."""
        self._rustobj.set_weights(weights)
        return self

    def with_bias(self, with_bias: bool) -> Self:
        """Set the parameters of this estimator."""
        self._rustobj.with_bias(with_bias)
        return self

    def with_solver(self, method: str) -> Self:
        """Set the parameters of this estimator."""
        assert method in ["normal", "ls"]
        self._rustobj.with_method(method)
        return self
