import numpy as np
import torch
from torch import nn

class RFF:

    def __init__(self, input_dim: int, n_features: int, n_classes: int):

        self.input_dim = input_dim
        self.n_features = n_features
        self.n_classes = n_classes

        # Random Fourier Features parameters
        self.W = np.random.normal(size=(self.input_dim, self.n_features))
        self.b = np.random.uniform(0, 2 * np.pi, size=self.n_features)

        # Linear model in the RFF-transformed space
        self.linear_model = nn.Linear(n_features, n_classes, bias=False)


    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the RFF transformation to the input data.
        """

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        projection = np.dot(X, self.W) + self.b
        return np.sqrt(2 / self.n_features) * np.cos(projection)


    def train(self, X_train: np.ndarray, Y_train: np.ndarray, training_parameters: dict):
        """
        Train the linear model with regularization in the RFF space.
        """

        X_train_rff = self.transform(X_train)

        X_train_rff = torch.tensor(X_train_rff, dtype=torch.float32)
        Y_train = nn.functional.one_hot(torch.tensor(Y_train, dtype=torch.long), num_classes=self.n_classes).float()

        # Solve (X^T X + reg*I)^-1 X^T Y
        reg_matrix = training_parameters.regularization * torch.eye(self.n_features)
        XTX = X_train_rff.T @ X_train_rff + reg_matrix
        XTY = X_train_rff.T @ Y_train
        self.linear_model.weight.data = torch.linalg.solve(XTX, XTY).T

        # Compute and store weight norm
        self.weight_norm = torch.norm(self.linear_model.weight.data, p=2).item()


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.
        """

        X_rff = self.transform(X)
        X_rff = torch.tensor(X_rff, dtype=torch.float32)
        outputs = self.linear_model(X_rff)
        return outputs.detach().numpy()


    def predict_class_and_proba(self, X: np.ndarray):
        """
        Predict class labels and probabilities for input data.
        """
        
        probs = self.predict(X)
        classes = np.argmax(probs, axis=1)
        return classes, probs
