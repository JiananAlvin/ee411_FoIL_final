import numpy as np
from typing import Dict, Tuple

class RFF:
    """
    Random Fourier Features for Gaussian (RBF) kernel approximation.
    
    - Automatically flattens MNIST images inside `transform`.
    - Automatically converts integer labels to one-hot inside `train`.
    """


    def __init__(
        self,
        input_dim: int,
        n_features: int,
        n_classes: int,
        sigma: float = 5.0,
    ):
        """
        Args:
            input_dim: Dimension of the flattened input (e.g., 784 for MNIST).
            n_features: Number of random frequencies (each yields cos & sin).
            n_classes: Number of classes (e.g. 10 for MNIST).
            sigma: Bandwidth for RBF kernel approximation.
        """
        self.input_dim = input_dim    # e.g., 784 for MNIST
        self.n_features = n_features
        self.n_classes = n_classes
        self.sigma = sigma
        self.rng = np.random.default_rng()

        # frequency matrix: shape (n_features, input_dim)
        self.W = self.rng.normal(
            loc=0.0, 
            scale=1.0 / self.sigma, 
            size=(self.n_features, self.input_dim)
        )
        # optional phase offsets
        self.b = np.zeros(self.n_features, dtype=np.float32)

        # will hold learned weights after training: shape (2*n_features, n_classes)
        self.weights = None


    @property
    def weight_norm(self) -> float:
        """
        Returns the mean of the column-wise Euclidean (L2) norms of the weight matrix.
        """
        # L2 norm of each column
        column_norms = np.linalg.norm(self.weights, axis=0)
        
        # mean of column-wise norms
        return float(column_norms.mean())


    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map X into 2*n_features RFF dimension (cos & sin).
        Automatically flattens if X is 3D, e.g. MNIST shape (n,28,28).
        """
        X = X.reshape(X.shape[0], -1)  # => (n, 784)

        # Z = X * W^T + b
        Z = X @ self.W.T + self.b
        cos_part = np.cos(Z)
        sin_part = np.sin(Z)
        # concatenate => shape (n, 2*n_features)
        Phi = np.concatenate([cos_part, sin_part], axis=1)

        # ReLU
        # Phi = np.maximum(Z, 0.0)

        # scale by sqrt(1 / n_features) so RFF ~ RBF kernel
        Phi *= 1.0 / np.sqrt(self.n_features)
        return Phi


    def _ensure_one_hot(self, Y: np.ndarray) -> np.ndarray:
        """
        If Y has shape (n,) of integer labels, convert to one-hot (n, n_classes).
        If already (n, n_classes), do nothing.
        """
        # convert from integer labels to one-hot
        n = len(Y)
        Y_oh = np.zeros((n, self.n_classes), dtype=np.float32)
        for i, label in enumerate(Y):
            # just in case label is a float or out of range
            label_idx = int(label)
            Y_oh[i, label_idx] = 1.0
        return Y_oh


    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        training_parameters: Dict[str, float] = None
    ) -> None:
        """
        Fit linear coefficients via least squares in RFF space.
        Uses a tiny ridge for numeric stability if needed.
        """
        # flatten images if needed
        Phi = self.transform(X_train)  # shape (n, 2*n_features)

        # convert integer labels to one-hot if needed
        Y_train_oh = self._ensure_one_hot(Y_train)  # shape (n, n_classes)

        self.weights, residuals, rank, s = np.linalg.lstsq(Phi, Y_train_oh, rcond=None)


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns raw scores in shape (n, n_classes).
        """
        Phi = self.transform(X)
        raw_scores = Phi @ self.weights  # (n, n_classes)
        return raw_scores


    def predict_class_and_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns integer predicted labels (n,) and softmax probs (n, n_classes).
        """
        raw_scores = self.predict(X)  # shape (n, n_classes)
        # softmax
        max_per_row = np.max(raw_scores, axis=1, keepdims=True)
        exp_scores = np.exp(raw_scores - max_per_row)
        sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
        proba = exp_scores / sum_exp
        pred_labels = np.argmax(raw_scores, axis=1)

        return pred_labels, proba
