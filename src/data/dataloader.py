from random import randrange
import numpy as np
import torch

from ..utils.constants import Datasets
from ..utils.functions import read_mnist_image_file, read_mnist_labels


class DataLoader:
    def __init__(self, dataset_parameters: dict):
        self.dataset = dataset_parameters.dataset
        self.n_train = dataset_parameters.n_train
        self.n_classes = dataset_parameters.n_classes
        self.label_noise = dataset_parameters.label_noise
        self.interpolation_threshold = self.n_classes * self.n_train

    def load(self):
        if self.dataset == Datasets.MNIST:
            # Grab MNIST data (train/test)
            train_imgs = read_mnist_image_file('train-images-idx3-ubyte')
            test_imgs = read_mnist_image_file('t10k-images-idx3-ubyte')
            train_lbls = read_mnist_labels('train-labels-idx1-ubyte')
            test_lbls = read_mnist_labels('t10k-labels-idx1-ubyte')

            # Normalize image data
            train_imgs = train_imgs / 255.0
            test_imgs = test_imgs / 255.0
        else:
            exit(f"error: dataset '{self.dataset}' not implemented")

        # Shuffle each set
        train_imgs, train_lbls = DataLoader.shuffle_X_Y(train_imgs, train_lbls)
        test_imgs, test_lbls = DataLoader.shuffle_X_Y(test_imgs, test_lbls)

        # Truncate train set if needed
        train_imgs = train_imgs[: self.n_train]
        train_lbls = train_lbls[: self.n_train]

        # Introduce label noise
        DataLoader.apply_label_noise(train_lbls, self.label_noise, self.n_classes)

        return train_imgs, test_imgs, train_lbls, test_lbls

    @staticmethod
    def apply_label_noise(labels: np.ndarray, noise: float, n_classes: int):
        if noise > 0.0:
            def get_alternative_label(correct_label):
                new_label = correct_label
                while new_label == correct_label:
                    new_label = randrange(n_classes)
                return new_label

            n_noisy = int(len(labels) * noise)
            for i in range(n_noisy):
                labels[i] = get_alternative_label(labels[i])

        return labels

    @staticmethod
    def shuffle_X_Y(X: np.ndarray, Y: np.ndarray):
        assert len(X) == len(Y)
        perm_idx = np.random.permutation(len(X))
        return X[perm_idx], Y[perm_idx]

    @staticmethod
    def get_train_batches(X_train: np.ndarray, Y_train: np.ndarray, batch_size: int):
        X_train, Y_train = DataLoader.shuffle_X_Y(X_train, Y_train)
        all_batches = []
        total_batch_count = (len(X_train) // batch_size) + (len(X_train) % batch_size != 0)

        for i in range(total_batch_count):
            x_slice = torch.tensor(
                X_train[i * batch_size : (i + 1) * batch_size], dtype=torch.float32
            )
            y_slice = torch.tensor(
                Y_train[i * batch_size : (i + 1) * batch_size], dtype=torch.float32
            )
            all_batches.append((x_slice, y_slice))
        return all_batches
