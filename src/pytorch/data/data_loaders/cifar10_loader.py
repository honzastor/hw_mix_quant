# Author: Jan Klhufek (iklhufek@fit.vut.cz)

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Default directory for the dataset
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
CIFAR10_DATA_DIR = os.path.join(DIR_PATH, '../datasets/cifar10')


class Cifar10Loader():
    def __init__(self, dataset_path: str = CIFAR10_DATA_DIR):
        """
        Initialize the Cifar-10 data loader.

        Args:
            dataset_path (str): Path to the Cifar-10 dataset.
        """
        assert os.path.exists(dataset_path), f"'{dataset_path}'" + ' path for dataset not found!'
        self._path = dataset_path
        self._input_size = 32

    def load_training_data(self, batch_size: int, num_workers: int = 4, shuffle: bool = True, pin_memory: bool = False) -> DataLoader:
        """
        Load the training data.

        Args:
            batch_size (int): Batch size for training.
            num_workers (int): Number of workers for data loading.
            shuffle (bool): Whether to shuffle the dataset.
            pin_memory (bool): Whether to use pinned memory.

        Returns:
            DataLoader: The DataLoader for the training dataset.
        """
        traindir = os.path.join(self._path, 'train')
        assert os.path.exists(traindir), f"'{traindir}'" + ' path for training data not found!'
        print(f"Creating training DataLoader..")
        # Training data transformations
        train_transforms = transforms.Compose([
            transforms.RandomCrop(self._input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
        ])
        train_dataset = datasets.ImageFolder(root=traindir, transform=train_transforms)
        # Return an iterable object over the training dataset and iterate it in batches
        self._train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
        return self._train_loader

    def load_validation_data(self, batch_size: int, num_workers: int = 4, shuffle: bool = False, pin_memory: bool = False) -> DataLoader:
        """
        Load the validation/test data.

        Args:
            batch_size (int): Batch size for validation/testing.
            num_workers (int): Number of workers for data loading.
            shuffle (bool): Whether to shuffle the dataset.
            pin_memory (bool): Whether to use pinned memory.

        Returns:
            DataLoader: The DataLoader for the validation dataset.
        """
        valdir = os.path.join(self._path, 'val')
        assert os.path.exists(valdir), f"'{valdir}'" + ' path for validation data not found!'
        print(f"Creating validation DataLoader..")
        # Validation data transformations
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
        ])
        val_dataset = datasets.ImageFolder(root=valdir, transform=val_transforms)
        # Return an iterable object over the validation dataset and iterate it in batches
        self._val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
        return self._val_loader

    @property
    def classes(self) -> int:
        return 10

    @property
    def dataset(self) -> str:
        return "cifar-10"

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def num_train_data(self) -> int:
        if hasattr(self, '_train_loader'):
            return len(self._train_loader.dataset)
        else:
            print("Training dataset not loaded. First, load it using `load_training_data` method.")
            return 0

    @property
    def num_val_data(self) -> int:
        if hasattr(self, '_val_loader'):
            return len(self._val_loader.dataset)
        else:
            print("Validation dataset not loaded. First, load it using `load_validation_data` method.")
            return 0

    @property
    def num_train_batches(self) -> int:
        if hasattr(self, '_train_loader'):
            return len(self._train_loader)
        else:
            print("Training dataset not loaded. First, load it using `load_training_data` method.")
            return 0

    @property
    def num_val_batches(self) -> int:
        if hasattr(self, '_val_loader'):
            return len(self._val_loader)
        else:
            print("Validation dataset not loaded. First, load it using `load_validation_data` method.")
            return 0
