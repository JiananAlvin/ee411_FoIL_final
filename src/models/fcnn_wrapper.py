from typing import Tuple
import numpy as np
import torch
from torch import nn

from src.dataloader import DataLoader
from src.fcnn import FCNN


class WrapperFCNN:
    def __init__(self,
                 in_nodes: int,
                 hidden_nodes: int,
                 out_nodes: int,
                 fcnn_parameters: dict,
                 interpolation_threshold: int,
                 smaller_model: FCNN):
        self.fcnn_params = fcnn_parameters
        self.n_classes = out_nodes
        self.interp_threshold = interpolation_threshold

        # Initialize FCNN model
        self.model = FCNN(
            in_nodes=in_nodes,
            hidden_nodes=hidden_nodes,
            out_nodes=out_nodes,
            final_activation=self.fcnn_params.final_activation,
            weight_reuse=self.fcnn_params.weight_reuse,
            weight_initialization=self.fcnn_params.weight_initialization,
            interpolation_threshold=interpolation_threshold,
            dropout=self.fcnn_params.dropout,
            smaller_model=smaller_model,
        )

    def __preprocess(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            X = X.reshape(len(X), -1)
        return X

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, training_parameters: dict):
        ep = training_parameters.n_epochs
        b_sz = training_parameters.batch_size
        step_reduce_ep = training_parameters.step_size_reduce_epochs
        step_reduce_pct = training_parameters.step_size_reduce_percent
        stop_zero = training_parameters.stop_at_zero_error

        # Choose loss
        if training_parameters.loss == 'squared_loss':
            loss_func = nn.MSELoss()
        elif training_parameters.loss == 'cross_entropy':
            loss_func = nn.CrossEntropyLoss()
        else:
            exit(f"error: loss '{training_parameters.loss}' not recognized")

        # Optimizer
        if training_parameters.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=training_parameters.learning_rate,
                momentum=0.95,
                weight_decay=training_parameters.weight_decay
            )
        else:
            exit(f"error: optimizer '{training_parameters.optimizer}' not recognized")

        # Flatten images if 2D
        X_train = self.__preprocess(X_train)
        self.model.train()

        for epoch in range(1, ep + 1):
            # Decrease LR if in under-parameterized regime
            if epoch % step_reduce_ep == 0 and self.model.n_parameters < self.interp_threshold:
                old_lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = old_lr * (1 - step_reduce_pct)

            cumulative_err = 0
            batches = DataLoader.get_train_batches(X_train, Y_train, batch_size=b_sz)

            for bx, by in batches:
                optimizer.zero_grad()
                preds = self.model(bx)
                _, pred_lbls = preds.max(dim=1)
                cumulative_err += torch.count_nonzero(by != pred_lbls)

                # For cross entropy, need one-hot for MSE or CrossEntropy
                label_1h = nn.functional.one_hot(by.long(), num_classes=self.n_classes).float()
                cur_loss = loss_func(preds, label_1h)
                cur_loss.backward()
                optimizer.step()

            if stop_zero and self.model.n_parameters < self.interp_threshold and cumulative_err == 0:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_ = self.__preprocess(X)
        tX = torch.tensor(X_, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tX)
        return outputs.numpy()

    def predict_class_and_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(X)
        classes = np.argmax(probs, axis=1)
        return classes, probs


