import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from omegaconf import OmegaConf
from src.utils.constants import Folders
import os


def plot_loss(ax, x, y_train, y_test, ylabel, title, threshold, is_zero_one=False):
    ax.plot(x, y_test, label="Test", marker="d")
    ax.plot(x, y_train, label="Train", marker=".", color="orange")
    ax.axvline(
        threshold,
        linestyle="dashed",
        color="black",
        label="Interpolation threshold $N = 4 × 10^4$",
    )
    ax.set_xlabel("Number of parameters/weights (×10³)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=11)
    if is_zero_one:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}")
        )
    ax.legend(fontsize=11)


config = OmegaConf.load(os.path.join(Folders.CONFIGS, "fcnn_mnist.yaml"))
num_params = np.array(config.fcnn_parameters.hidden_nodes)

data = {
    key: np.load(f"{Folders.LOGS}fcnn_mnist_label_noise_0_10/{key}.npy").mean(axis=0)
    for key in [
        "squared_loss_test",
        "squared_loss_train",
        "zero_one_loss_test",
        "zero_one_loss_train",
    ]
}

threshold = num_params[len(num_params) // 3]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))

plot_loss(
    axes[0],
    num_params,
    data["zero_one_loss_train"],
    data["zero_one_loss_test"],
    "Zero-one loss (%)",
    "Zero-one Loss",
    threshold,
    is_zero_one=True,
)
plot_loss(
    axes[1],
    num_params,
    data["squared_loss_train"],
    data["squared_loss_test"],
    "Squared loss",
    "Squared Loss",
    threshold,
)

plt.tight_layout()
plt.savefig("figures/double_descent_fcnn_mnist.pdf", dpi=300)
plt.show()
