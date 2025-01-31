from omegaconf import DictConfig
from tqdm import tqdm
from sys import stdout
import numpy as np
import torch
from wandb.sdk.wandb_run import Run
from src.data.dataloader import DataLoader
from src.eval.evaluator import Evaluator
from src.models.fcnn_wrapper import WrapperFCNN
from src.models.rff import RFF
from src.utils.file_handler import FileHandler
from src.utils.constants import Models


class Experiment:

    def __init__(
        self,
        model_type: str,
        config: DictConfig,
        logger: Run,
        file_handler: FileHandler,
    ):

        self.model_type = model_type
        self.config = config
        self.dataset_parameters = self.config.dataset_parameters
        self.training_parameters = self.config.training_parameters

        self.logger = logger
        self.file_handler = file_handler

    def run(self):
        repetitions = self.config.repetitions

        # decide parameter range based on model type
        if self.model_type == Models.FCNN:
            param_range = self.config.fcnn_parameters.hidden_nodes
        elif self.model_type == Models.RFF:
            param_range = self.config.rff_parameters.n_features
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        # initialize metric arrays
        zero_one_loss_train, squared_loss_train = [
            np.zeros(shape=(repetitions, len(param_range))) for _ in range(2)
        ]
        zero_one_loss_test, squared_loss_test = [
            np.zeros(shape=(repetitions, len(param_range))) for _ in range(2)
        ]
        if self.model_type == Models.RFF:
            weight_norm = np.zeros(shape=(repetitions, len(param_range)))

        with tqdm(
            total=repetitions * len(param_range), file=stdout, smoothing=0.1
        ) as pbar:
            for r in range(repetitions):
                torch.manual_seed(42 + r)
                np.random.seed(42 + r)
                smaller_model = None

                # load dataset
                data_loader = DataLoader(self.dataset_parameters)
                X_train, X_test, Y_train, Y_test = data_loader.load()

                for i, param in enumerate(param_range):
                    # initialize model
                    if self.model_type == Models.FCNN:
                        model = WrapperFCNN(
                            in_nodes=self.dataset_parameters.input_dim,
                            hidden_nodes=param,
                            out_nodes=self.dataset_parameters.n_classes,
                            fcnn_parameters=self.config.fcnn_parameters,
                            interpolation_threshold=data_loader.interpolation_threshold,
                            smaller_model=smaller_model,
                        )
                    elif self.model_type == Models.RFF:
                        model = RFF(
                            input_dim=self.dataset_parameters.input_dim,
                            n_features=param,
                            n_classes=self.dataset_parameters.n_classes,
                        )

                    # train model
                    model.train(
                        X_train=X_train,
                        Y_train=Y_train,
                        training_parameters=self.training_parameters,
                    )
                    if self.model_type == Models.FCNN:
                        smaller_model = model.model

                    # weight norm for RFF
                    if self.model_type == Models.RFF:
                        weight_norm[r][i] = model.weight_norm

                    # evaluate metrics for train and test sets
                    c_train, p_train = model.predict_class_and_proba(X_train)
                    c_test, p_test = model.predict_class_and_proba(X_test)

                    zero_one_loss, squared_loss = Evaluator.evaluate(
                        y_pred=c_train,
                        p_pred=p_train,
                        y_true=Y_train,
                        n_classes=data_loader.n_classes,
                    )
                    zero_one_loss_train[r][i] += zero_one_loss
                    squared_loss_train[r][i] += squared_loss

                    self.logger.log(
                        {
                            "train/zero_one_loss": zero_one_loss,
                            "train/squared_loss": squared_loss,
                        }
                    )

                    zero_one_loss, squared_loss = Evaluator.evaluate(
                        y_pred=c_test,
                        p_pred=p_test,
                        y_true=Y_test,
                        n_classes=data_loader.n_classes,
                    )
                    zero_one_loss_test[r][i] += zero_one_loss
                    squared_loss_test[r][i] += squared_loss

                    self.logger.log(
                        {
                            "test/zero_one_loss": zero_one_loss,
                            "test/squared_loss": squared_loss,
                        }
                    )

                    # update progress bar
                    pbar.update(1)

        self.file_handler.save_logs(
            config=self.config,
            zero_one_loss_train=zero_one_loss_train,
            squared_loss_train=squared_loss_train,
            zero_one_loss_test=zero_one_loss_test,
            squared_loss_test=squared_loss_test,
        )
        if self.model_type == Models.RFF:
            self.file_handler.save_weight_norm(weight_norm=weight_norm)

        # finish wandb run
        self.logger.finish()
