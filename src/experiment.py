from omegaconf import DictConfig
from tqdm import tqdm
from sys import stdout
import numpy as np
from wandb.sdk.wandb_run import Run
from src.dataloader import DataLoader
from src.evaluator import Evaluator
from src.fcnn_wrapper import WrapperFCNN
from src.rff import RFF
from src.file_handler import FileHandler
from src.utils.constants import Models


class Experiment:

    def __init__(self, model_type: str, config: DictConfig, logger: Run, file_handler: FileHandler):

        self.model_type = model_type
        self.config = config
        self.dataset_parameters = self.config.dataset_parameters
        self.training_parameters = self.config.training_parameters

        self.logger = logger
        self.file_handler = file_handler

    def run(self):
        repetitions = self.config.repetitions
        
        # Decide parameter range based on model type
        if self.model_type == Models.FCNN:
            param_range = self.config.fcnn_parameters.hidden_nodes
        elif self.model_type == Models.RFF:
            param_range = self.config.rff_parameters.n_features
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        # Initialize metric arrays
        metrics_train = {key: np.zeros((repetitions, len(param_range))) for key in
                         ['zero_one_loss', 'squared_loss', 'entropy_loss', 'precision', 'recall', 'f1']}
        metrics_test = metrics_train.copy()

        with tqdm(total=repetitions * len(param_range), file=stdout, smoothing=0.1) as pbar:
            for r in range(repetitions):
                smaller_model = None

                # Load dataset
                data_loader = DataLoader(self.dataset_parameters)
                X_train, X_test, Y_train, Y_test = data_loader.load()

                for i, param in enumerate(param_range):
                    # Initialize model
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

                    # Train model
                    model.train(X_train=X_train, Y_train=Y_train, training_parameters=self.training_parameters)
                    if self.model_type == Models.FCNN:
                        smaller_model = model.model

                    # Evaluate metrics for train and test sets
                    for split, (X, Y, metrics) in [('train', (X_train, Y_train, metrics_train)),
                                                   ('test', (X_test, Y_test, metrics_test))]:
                        y_pred, p_pred = model.predict_class_and_proba(X)
                        results = Evaluator.evaluate(
                            y_pred=y_pred,
                            p_pred=p_pred,
                            y_true=Y,
                            n_classes=data_loader.n_classes,
                        )
                        for key, value in zip(metrics.keys(), results):
                            metrics[key][r][i] += value

                            # Log each metric to wandb
                            if self.model_type == Models.FCNN:
                                self.logger.log({f"{split}/{key}": value, "hidden_nodes": param, "repetition": r})
                            elif self.model_type == Models.RFF:
                                self.logger.log({f"{split}/{key}": value, "n_features": param, "repetition": r, "weight_norm": model.weight_norm}) 

                    # Update progress bar
                    pbar.update(1)

        # Save logs
        self.file_handler.save_logs(
            config=self.config,
            zero_one_loss_train=metrics_train['zero_one_loss'],
            squared_loss_train=metrics_train['squared_loss'],
            entropy_loss_train=metrics_train['entropy_loss'],
            precision_train=metrics_train['precision'],
            recall_train=metrics_train['recall'],
            f1_train=metrics_train['f1'],
            zero_one_loss_test=metrics_test['zero_one_loss'],
            squared_loss_test=metrics_test['squared_loss'],
            entropy_loss_test=metrics_test['entropy_loss'],
            precision_test=metrics_test['precision'],
            recall_test=metrics_test['recall'],
            f1_test=metrics_test['f1'],
            weight_norm=np.empty(0) if self.model_type == Models.FCNN else np.array([model.weight_norm])
        )
        
        # Finish wandb run
        self.logger.finish()
