"""Module for linear regression."""

from typing import Self

from binoculars import KMeansRust


class KMeans:
    """KMeans clustering with Rust backend."""

    def __init__(self) -> None:
        self._rustobj = KMeansRust()
        self.n_clusters = None

    def __str__(self) -> str:
        """Return a string representation of the model."""
        return f"KMeans(n_clusters={self.n_clusters})"

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"KMeans(n_clusters={self.n_clusters})"

    def fit(self, X: list[list[float]]) -> Self:
        """Fit the model using X as training data."""
        self._rustobj.fit(X)
        return self

    def predict(self, X: list[list[float]]) -> list[int]:
        """Predict using the linear model."""
        return self._rustobj.predict(X)

    def get_centroids(self) -> list[list[float]]:
        """Get parameters for this estimator."""
        return self._rustobj.get_centroids()

    def set_centroids(self, centroids: list[list[float]]) -> Self:
        """Set the parameters of this estimator."""
        self._rustobj.set_centroids(centroids)
        return self

    def with_k(self, n_clusters: int) -> Self:
        """Set the parameters of this estimator."""
        self._rustobj.with_k(n_clusters)
        self.n_clusters = n_clusters
        return self

    def with_max_iter(self, max_iter: int) -> Self:
        """Set the parameters of this estimator."""
        self._rustobj.with_max_iter(max_iter)
        return self

    def with_random_state(self, random_state: int) -> Self:
        """Set the parameters of this estimator."""
        self._rustobj.with_random_state(random_state)
        return self

    def with_init(self, init: str) -> Self:
        """Set the parameters of this estimator."""
        assert init in ["random", "kmeans++"]
        self._rustobj.with_init(init)
        return self
