# ee411_FoIL_finalReconciling Modern Machine Learning Practice and the Bias-variance Trade-off Reproducibility Challenge

## Introduction

## Dataset
The dataset used in this experiment is the MNIST handwritten digit dataset. The dataset contains $60,000$ training images and $10,000$ test images of handwritten digits ($0-9$). The images are grayscale, with a size of $28 \times 28$ pixels.

## Requirements
* Anaconda/Miniconda
* Python 3.10
* PyTorch 2.0.1
* NumPy
* Scikit-learn
* Omegaconf
* Wandb (optional for logging)

## Dependencies
This project uses a `conda` environment to manage all dependencies. Under the root directory of the project repository, run the following command to create the environment from the `environment.yaml` file:
```
conda env create -f environment.yaml -n double_descent
```
Activate the environment:
```
conda activate double_descent
```

## Usage
1. Prepare the dataset: [Download the MNIST dataset](http://yann.lecun.com/exdb/mnist/), unzip and place the four files under:
```
data
└── MNIST
    └── raw
        ├── t10k-images.idx3-ubyte
        ├── t10k-labels.idx1-ubyte
        ├── train-images.idx3-ubyte
        └── train-labels.idx1-ubyte
```

2. Configure the model and training parameters
The configurations for the experiments are stored in YAML files under `config`. The default configuration files are `rff_mnist_label_noise_0_10.yaml` for RFF models and `fcnn_mnist_label_noise_0_10.yaml` for FCNNs. You can create a new configuration file or modify the existing one to customize the experiment parameters.

3. Run the experiment
To run the experiment, use the following command:
```
python run.py --model_type <model_type> --config <configuration_file>
```
Replace <model_type> with either `rff` or `fcnn` to select the model you want to use. Replace <configuration_file> with either `rff_mnist_label_noise_0_10` or `fcnn_mnist_label_noise_0_10` accordingly.

4. Customize and extend Model Architectures
The project includes two model architectures: FCNN and RFF. You can modify the existing models by updating the corresponding files (fcnn.py/fcnn_wrapper.py and rff.py) under `src/models/`. You can also create new ones under `src/models/`.

## Results

## Project Structure
The project has the following structure:
```
├── config
├── data
├── results
├── src
│   ├── data
│   ├── eval
│   ├── models
│   ├── utils
│   └── experiment.py
|
├── job.sh
├── plot_fcnn.py
├── plot_rff.py
├── environment.yaml
├── README.md
└── run.py
```