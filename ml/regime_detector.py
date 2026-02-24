from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
from typing import Dict

class RegimeDetector:
    """
    Identifies the current market regime (e.g., Bull, Bear, High Volatility).
    Uses unsupervised learning on recent volatility and return features.
    """
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = GaussianMixture(n_components=n_regimes, covariance_type='full')
        self.is_fitted = False
        self.regime_labels = {
            0: "LOW_VOL_BULL",
            1: "HIGH_VOL_BEAR",
            2: "SIDEWAYS_CHOP"
        }

    def train(self, features: np.ndarray):
        """
        Trains the GMM on historical features.
        features shape: (n_samples, n_features)
        """
        self.model.fit(features)
        self.is_fitted = True

    def predict_regime(self, current_features: np.ndarray) -> str:
        """
        Returns the regime label for the current state.
        """
        if not self.is_fitted:
            return "UNKNOWN"
            
        regime_idx = self.model.predict(current_features.reshape(1, -1))[0]
        return self.regime_labels.get(regime_idx, "UNKNOWN")
