from typing import Self

from binoculars import _least_square, _predict


class LinearModel:
    "Least squares linear regression with Rust backend."

    def __init__(self):
        self.weights = None

    def fit(self, X: list[list[float]], y: list[float]) -> Self:
        "Fit the model using X as training data and y as target values."
        self.weights = _least_square(X, y)
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        "Predict using the linear model."
        return _predict(X, self.weights)

    def get_weights(self) -> list[float]:
        "Get parameters for this estimator."
        return self.weights

    def set_weights(self, coef_: list[float]) -> Self:
        "Set the parameters of this estimator."
        self.weights = coef_
        return self
