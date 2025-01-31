import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from omegaconf import OmegaConf
import os
from src.utils.constants import Folders
from matplotlib.lines import Line2D


def compute_min_norm(y_values, num_rff, threshold=10**4):
    return np.min(y_values[num_rff >= threshold])


def plot_vs_rff(
    num_rff,
    y_values,
    title,
    xlabel,
    ylabel,
    ax,
    label="RFF",
    marker="o",
    color=None,
    log_y=False,
    min_norm_value=None,
    add_legend=False,
    train_plot=False,
    show_title=False,
    show_xlabel=False,
    is_zero_one=False,
):
    ax.plot(num_rff, y_values, marker=marker, label=label, color=color, linestyle="-")
    if show_title:
        ax.set_title(title, fontsize=11)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=11)
    if log_y:
        ax.set_yscale("log")
    ax.axvline(x=10**4, color="black", linestyle="--", linewidth=1)
    if min_norm_value is not None and not train_plot:
        ax.hlines(
            y=min_norm_value,
            xmin=num_rff[0],
            xmax=num_rff[-1],
            colors="black",
            linestyles="-",
            linewidth=1,
        )
    ax.set_xticks([0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    ax.set_xticklabels(["0", "10", "20", "30", "40", "50", "60"])
    if is_zero_one:
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}")
        )
    if add_legend:
        legend_lines = [
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1,
                label="Interpolation threshold $N=10^4$",
            )
        ]
        if not train_plot:
            legend_lines.insert(
                0,
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="-",
                    linewidth=1,
                    label="Min-norm solution $h_{n,∞}$",
                ),
            )
        ax.legend(handles=[ax.lines[0]] + legend_lines, fontsize=11)


# Load configuration and data
config_file = os.path.join(Folders.CONFIGS, "rff_mnist_label_noise_0_10.yaml")
result_path = os.path.join(Folders.LOGS, "rff_mnist_label_noise_0_10/")
config = OmegaConf.load(config_file)
num_rff = np.array(config.rff_parameters.n_features)

data = {
    key: np.load(result_path + f"{key}.npy").mean(axis=0)
    for key in [
        "squared_loss_test",
        "squared_loss_train",
        "weight_norm",
        "zero_one_loss_test",
        "zero_one_loss_train",
    ]
}

min_norms = {
    key: compute_min_norm(data[key], num_rff)
    for key in ["zero_one_loss_test", "squared_loss_test", "weight_norm"]
}

fig, axs = plt.subplots(
    3, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [2, 1.2, 1.2]}
)

plot_vs_rff(
    num_rff,
    data["zero_one_loss_test"],
    "Zero-one Loss",
    "Number of Random Fourier Features (×10³)",
    "Test (%)",
    axs[0, 0],
    log_y=True,
    min_norm_value=min_norms["zero_one_loss_test"],
    add_legend=True,
    show_title=True,
    is_zero_one=True,
)
plot_vs_rff(
    num_rff,
    data["squared_loss_test"],
    "Squared Loss",
    "Number of Random Fourier Features (×10³)",
    "Test",
    axs[0, 1],
    log_y=True,
    min_norm_value=min_norms["squared_loss_test"],
    add_legend=True,
    show_title=True,
)
plot_vs_rff(
    num_rff,
    data["weight_norm"],
    "Norm",
    "Number of Random Fourier Features (×10³)",
    "Norm",
    axs[1, 0],
    log_y=True,
    min_norm_value=min_norms["weight_norm"],
    add_legend=True,
)
plot_vs_rff(
    num_rff,
    data["weight_norm"],
    "Norm",
    "Number of Random Fourier Features (×10³)",
    "Norm",
    axs[1, 1],
    log_y=True,
    min_norm_value=min_norms["weight_norm"],
    add_legend=True,
)
plot_vs_rff(
    num_rff,
    data["zero_one_loss_train"],
    "Zero-one Loss",
    "Number of Random Fourier Features (×10³)",
    "Train (%)",
    axs[2, 0],
    color="orange",
    add_legend=True,
    train_plot=True,
    show_xlabel=True,
    is_zero_one=True,
)
plot_vs_rff(
    num_rff,
    data["squared_loss_train"],
    "Squared Loss",
    "Number of Random Fourier Features (×10³)",
    "Train",
    axs[2, 1],
    color="orange",
    add_legend=True,
    train_plot=True,
    show_xlabel=True,
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/double_descent_rff_mnist.pdf", dpi=300)
plt.show()
