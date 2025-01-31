import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

import warnings

from src.utils.zero_one_loss import OneZeroLoss

warnings.filterwarnings("ignore")


class Evaluator:

    @staticmethod
    def evaluate(
        y_pred: np.ndarray, p_pred: np.ndarray, y_true: np.ndarray, n_classes: int
    ):

        y_pred = torch.from_numpy(y_pred).float()
        p_pred = torch.from_numpy(p_pred).float()
        p_true = torch.nn.functional.one_hot(
            torch.from_numpy(y_true).long(), num_classes=n_classes
        ).float()
        y_true = torch.from_numpy(y_true).float()

        # one-zero loss
        zero_one_loss = OneZeroLoss(y_true=y_true, y_pred=y_pred)

        # mean squared error
        squared_loss = torch.nn.MSELoss()(p_true, p_pred).item()

        return zero_one_loss, squared_loss
