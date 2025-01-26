import wandb
import uuid


"""
Initialize a wandb run with a unique run name based on the config file name and a UUID.
"""
def init_wandb(config_name: str) -> None:

    unique_id = str(uuid.uuid4())[:4]
    run_name = f"{config_name}_{unique_id}"

    # Initialize wandb
    wandb.init(
        project="double_descent",
        name=run_name,
    )

    return wandb
