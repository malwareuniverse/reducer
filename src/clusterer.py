from abc import ABC, abstractmethod
from enum import Enum
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
import numpy as np


class ClusterMethod(str, Enum):
    """Enum for available clustering methods."""
    HDBSCAN = "hdbscan"
    # k-means added here, but can't be used. User would need to know k-cluster
    KMEANS = "kmeans"


class BaseClusterer(ABC):
    """Abstract Base Class for all clustering algorithm wrappers."""

    def __init__(self, **kwargs):
        self.model = None
        self.name = "BaseClusterer"

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fits the clustering model to the data and
        returns the cluster labels for each point.

        Args:
            X: A numpy array of shape (n_samples, n_features).

        Returns:
            A numpy array of shape (n_samples,)
            containing the integer label for each sample.
            By convention, -1 is used for noise/outliers.
        """
        pass


class HDBSCANClusterer(BaseClusterer):
    """A wrapper for the HDBSCAN clustering algorithm."""

    def __init__(self, **kwargs):
        super().__init__()

        # Make the model more performant by default
        kwargs.setdefault('core_dist_n_jobs', -1)
        self.model = HDBSCAN(**kwargs)
        self.name = "HDBSCAN"

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        # HDBSCAN can be sensitive to data type, ensure it's float
        if X.dtype != np.float64 and X.dtype != np.float32:
            X = X.astype(np.float64)

        return self.model.fit_predict(X)


class KMeansClusterer(BaseClusterer):
    """A wrapper for the scikit-learn KMeans clustering algorithm."""

    def __init__(self, **kwargs):
        super().__init__()

        kwargs.setdefault('n_clusters', 8)
        kwargs.setdefault('init', 'k-means++')
        kwargs.setdefault('n_init', 'auto')

        self.model = KMeans(**kwargs)
        self.name = "KMeans"

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fits KMeans and returns cluster labels.
        Note: KMeans does not produce noise points (-1).
        All points are assigned to a cluster.
        """
        return self.model.fit_predict(X)


class ClusterFactory:
    """Factory to create instances of different clustering algorithms."""

    @staticmethod
    def create_clusterer(method: ClusterMethod, **kwargs) -> BaseClusterer:
        """
        Creates a clusterer instance based on the specified method.

        Args:
            method: The ClusterMethod enum member.
            **kwargs: Arguments to be passed to the clusterer's constructor.

        Returns:
            An instance of a BaseClusterer subclass.
        """
        if method == ClusterMethod.HDBSCAN:
            return HDBSCANClusterer(**kwargs)
        elif method == ClusterMethod.KMEANS:
            return KMeansClusterer(**kwargs)
        else:
            raise ValueError(f"""
                            Unknown or unavailable clustering method: {method}
                            """)

    @staticmethod
    def get_available_methods() -> list[str]:
        """Returns a list of currently available clustering method names."""
        methods = []

        methods.append(ClusterMethod.HDBSCAN.value)
        methods.append(ClusterMethod.KMEANS.value)

        return methods
