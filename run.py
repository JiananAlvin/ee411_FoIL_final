import os
import argparse
from omegaconf import OmegaConf

from src.experiment import Experiment
from src.file_handler import FileHandler
from src.utils.constants import Folders, Models
from src.utils.logger import init_wandb

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="fcnn_mnist_label_noise_0_10",
    help="Name of the config file (without extension) to be used for the experiment.",
)
parser.add_argument(
    "-m",
    "--model_type",
    type=str,
    required=True,
    choices=[Models.FCNN, Models.RFF],
    help='Type of model to use for the experiment: "fcnn" or "rff".',
)

args = parser.parse_args()

if __name__ == "__main__":
    config_file = os.path.join(Folders.CONFIGS, args.config + ".yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    config = OmegaConf.load(config_file)

    logger = init_wandb(args.config)

    experiment = Experiment(
        model_type=args.model_type,
        config=config,
        logger=logger,
        file_handler=FileHandler(config_name=args.config, read_mode=False),
    )
    experiment.run()
