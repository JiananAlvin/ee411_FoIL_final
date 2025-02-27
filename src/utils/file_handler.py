import os
import numpy as np
from omegaconf import OmegaConf
from src.utils.constants import Folders


class FileHandler:

    def __init__(self, config_name: str, read_mode: bool):
        self.config_name = config_name
        self.log_folder = os.path.join(Folders.LOGS, config_name)

        if not read_mode:
            if not os.path.exists(self.log_folder):
                os.makedirs(self.log_folder)
            elif os.listdir(self.log_folder):
                exit(
                    f"error: logs for {self.config_name} already exist at {self.log_folder}"
                )

    ## Config File
    def save_config(self, config):
        file = os.path.join(self.log_folder, f"{self.config_name}.yaml")
        OmegaConf.save(config, file)

    def load_config(self):
        file = os.path.join(self.log_folder, f"{self.config_name}.yaml")
        return OmegaConf.load(file)

    ## Numpy Array
    def save_numpy(self, metric_name: str, arr: np.ndarray):
        file = os.path.join(self.log_folder, metric_name + ".npy")

        with open(file, "wb") as f:
            np.save(f, arr)

    def load_numpy(self, metric_name):
        file = os.path.join(self.log_folder, metric_name + ".npy")

        with open(file, "rb") as f:
            arr = np.load(f)

        return arr

    ## Logs
    def save_logs(
        self,
        config: dict,
        zero_one_loss_train: np.ndarray,
        squared_loss_train: np.ndarray,
        zero_one_loss_test: np.ndarray,
        squared_loss_test: np.ndarray,
    ):

        self.save_config(config=config)

        # train results
        self.save_numpy("zero_one_loss_train", zero_one_loss_train)
        self.save_numpy("squared_loss_train", squared_loss_train)

        # test results
        self.save_numpy("zero_one_loss_test", zero_one_loss_test)
        self.save_numpy("squared_loss_test", squared_loss_test)

    def load_logs(self):

        # config
        config = self.load_config()

        # statistics
        statistics = {
            "train": {
                "zero_one_loss": self.load_numpy("zero_one_loss_train"),
                "squared_loss": self.load_numpy("squared_loss_train"),
            },
            "test": {
                "zero_one_loss": self.load_numpy("zero_one_loss_test"),
                "squared_loss": self.load_numpy("squared_loss_test"),
            },
        }

        return config, statistics

    def save_weight_norm(self, weight_norm: np.ndarray):

        self.save_numpy("weight_norm", weight_norm)

    def load_weight_norm(self):

        return self.load_numpy("weight_norm")
