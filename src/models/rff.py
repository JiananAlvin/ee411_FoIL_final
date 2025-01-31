import numpy as np
from typing import Dict, Tuple


class RFF:
    """
    Random Fourier Features for approximating an RBF kernel.
    """

    def __init__(
        self, input_dim: int, n_features: int, n_classes: int, sigma: float = 5.0
    ):
        self.input_dim = input_dim
        self.n_features = n_features
        self.n_classes = n_classes
        self.sigma = sigma
        self.rng = np.random.default_rng()

        # Frequencies and initial offsets
        self.W = self.rng.normal(
            loc=0.0, scale=1.0 / self.sigma, size=(self.n_features, self.input_dim)
        )
        self.b = np.zeros(self.n_features, dtype=np.float32)
        self.weights = None

    @property
    def weight_norm(self) -> float:
        # Mean column-wise L2 norm of learned weights
        if self.weights is None:
            return 0.0
        norms = np.linalg.norm(self.weights, axis=0)
        return float(np.mean(norms))

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Flatten if 3D
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        Z = X @ self.W.T + self.b
        cos_part = np.cos(Z)
        sin_part = np.sin(Z)

        # Combine
        Phi = np.concatenate([cos_part, sin_part], axis=1)
        Phi *= 1.0 / np.sqrt(self.n_features)
        return Phi

    def _ensure_one_hot(self, Y: np.ndarray) -> np.ndarray:
        if Y.ndim == 1:
            n = len(Y)
            Y_1h = np.zeros((n, self.n_classes), dtype=np.float32)
            for i, lbl in enumerate(Y):
                Y_1h[i, int(lbl)] = 1.0
            return Y_1h
        return Y

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        training_parameters: Dict[str, float] = None,
    ):
        # Generate RFF features
        phi_train = self.transform(X_train)
        y_oh = self._ensure_one_hot(Y_train)
        # Solve least squares
        self.weights, _, _, _ = np.linalg.lstsq(phi_train, y_oh, rcond=None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self.transform(X)
        return feats @ self.weights

    def predict_class_and_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raw_scores = self.predict(X)
        max_rows = np.max(raw_scores, axis=1, keepdims=True)
        exps = np.exp(raw_scores - max_rows)
        sums = np.sum(exps, axis=1, keepdims=True)
        probs = exps / sums
        classes = np.argmax(raw_scores, axis=1)
        return classes, probs
