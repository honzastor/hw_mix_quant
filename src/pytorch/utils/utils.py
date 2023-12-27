# Author: Jan Klhufek (iklhufek@fit.vut.cz)

import sys
import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Callable, Dict, Any
from enum import Enum

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(DIR_PATH, '..'))
from data.data_loaders import ImagenetLoader, Imagenet100Loader, Cifar10Loader


class Summary(Enum):
    """Enumeration for different types of summaries."""
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name: str, fmt: str = '.3f', summary_type: Summary = Summary.AVERAGE) -> None:
        """
        Initializes the AverageMeter object.

        Args:
            name (str): Name of the metric.
            fmt (str): Format string for printing the metric values.
            summary_type (Summary): Type of summary to compute.
        """
        self._name = name
        self._fmt = fmt
        self._summary_type = summary_type
        self.reset()

    def reset(self) -> None:
        """ Resets the metric values to zero. """
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Updates the metric with a new value.

        Args:
            val (float): New value to add.
            n (int): Weight of the new value.
        """
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count if self._count > 0 else 0

    def all_reduce(self) -> None:
        """ Synchronizes the metric values across multiple processes. """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self._sum, self._count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self._sum, self._count = total.tolist()
        self._avg = self._sum / self._count if self._count > 0 else 0

    @property
    def avg(self) -> float:
        return self._avg

    def __str__(self) -> str:
        return f'{self._name} {self._val:{self._fmt}} ({self._avg:{self._fmt}})'

    def summary(self) -> str:
        """ Generates a summary string based on the summary type. """
        if self._summary_type is Summary.NONE:
            return ''
        elif self._summary_type is Summary.AVERAGE:
            return f'{self._name} {self._avg:.3f}'
        elif self._summary_type is Summary.SUM:
            return f'{self._name} {self._sum:.3f}'
        elif self._summary_type is Summary.COUNT:
            return f'{self._name} {self._count:.3f}'
        else:
            raise ValueError(f'invalid summary type {self._summary_type}')


class ProgressMeter(object):
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = "") -> None:
        """
        Initializes the ProgressMeter object.

        Args:
            num_batches (int): Total number of batches.
            meters (List[AverageMeter]): List of AverageMeter objects.
            prefix (str): Prefix string for display.
        """
        self._batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self._meters = meters
        self._prefix = prefix

    def display(self, batch: int, log_file: str = "") -> None:
        """
        Displays the progress for a given batch.

        Args:
            batch (int): Current batch number.
            log_file (str): The path to the log file where progress will be logged. Defaults to "".
        """
        entries = [self._prefix + self._batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self._meters]
        progress_info = '\t'.join(entries)
        print(progress_info)
        if log_file != "":
            with open(log_file, 'a') as f:
                f.write(progress_info + '\n')

    def display_summary(self, log_file: str = "") -> None:
        """
        Displays a summary of the progress.

        Args:
            batch (int): Current batch number.
            log_file (str): The path to the log file where progress summary will be logged. Defaults to "".
        """
        entries = [" *"]
        entries += [meter.summary() for meter in self._meters]
        progress_info = ' '.join(entries)
        print(progress_info)
        if log_file != "":
            with open(log_file, 'a') as f:
                f.write(progress_info + '\n')

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        """ Helper method to format the batch number string. """
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (torch.Tensor): The output predictions from the model.
        target (torch.Tensor): The ground truth labels.
        topk (Tuple[int, ...]): Tuple of top 'k' for which accuracy is computed.

    Returns:
        List[float]: A list of accuracies for each topk.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class Logger(object):
    def __init__(self, fpath: Optional[str], title: Optional[str] = None, resume: bool = False) -> None:
        """
        Initializes the Logger object.

        Inspired by the code from haq repository: https://github.com/mit-han-lab/haq/blob/master/lib/utils/utils.py

        Args:
            fpath (Optional[str]): File path for the log file. If None, logging is skipped.
            title (Optional[str]): Title for the log. Defaults to an empty string if None.
            resume (bool): If True, resume logging from an existing file. Otherwise, start new logging.
        """
        self._file = None
        self._resume = resume
        self._title = '' if title is None else title
        self._names = []
        self._numbers = {}

        if fpath is not None:
            if resume:
                with open(fpath, 'r') as file:
                    self._names = file.readline().rstrip().split('\t')
                    for line in file:
                        numbers = line.rstrip().split('\t')
                        for i, num in enumerate(numbers):
                            self._numbers.setdefault(self._names[i], []).append(num)

                self._file = open(fpath, 'a')
            else:
                self._file = open(fpath, 'w')

    def set_names(self, names: List[str]) -> None:
        """
        Sets the names of the metrics to log.

        Args:
            names (List[str]): A list of names of the metrics.
        """
        # initialize numbers as empty list
        self._names = names
        self._numbers = {name: [] for name in names}
        if not self._resume and self._file:
            self._file.write('    '.join(names) + '\n')
            self._file.flush()

    def append(self, values: List[Union[int, float, str]]) -> None:
        """
        Appends a new set of metric values to the log.

        Args:
            values (List[Union[int, float, str]]): A list of numbers (or formatted strings) corresponding to the metrics.
        """
        if self._file:
            for index, val in enumerate(values):
                if isinstance(val, int):
                    self._file.write(f"{val:3d}" + '    ')
                elif isinstance(val, float):
                    self._file.write(f"{val:.6f}" + '    ')
                else:
                    self._file.write(val + '    ')
                self._numbers[self._names[index]].append(val)
            self._file.write('\n')
            self._file.flush()

    def plot(self, names: Optional[List[str]] = None) -> None:
        """
        Plots the logged metrics.

        Args:
            names (Optional[List[str]]): A list of metric names to plot. If None, all metrics are plotted.
        """
        names = self._names if names is None else names
        for name in names:
            x = np.arange(len(self._numbers[name]))
            plt.plot(x, np.asarray(self._numbers[name], dtype=float))
        plt.legend([f'{self._title}({name})' for name in names])
        plt.grid(True)

    def close(self) -> None:
        """Closes the log file."""
        if self._file is not None:
            self._file.close()


def get_model_size(model: torch.nn.Module, run_id: int) -> float:
    """
    Calculates the size of the model in megabytes (MB).

    Args:
        model (torch.nn.Module): PyTorch model.
        run_id (int): Unique id to separate runs executed in parallel.

    Returns:
        float: Size of the model in MB.
    """
    torch.save(model.state_dict(), f"tmp_{run_id}.p")
    model_size = os.path.getsize(f"tmp_{run_id}.p")/1e6
    os.remove(f"tmp_{run_id}.p")
    return model_size


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Counts the total number of parameters in a model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        Tuple[int, int]: Total number of parameters, Number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_activation_function(act_name: str) -> Callable:
    """
    Retrieves the specified PyTorch activation function from a predefined list.

    Args:
        act_name (str): The name of the activation function to retrieve.
                        Supported activation functions include ReLU, LeakyReLU, PReLU, RReLU, ELU,
                        SELU, CELU, GELU, Sigmoid, Tanh, Softmax, LogSoftmax, Softplus, Softshrink,
                        Softsign, Tanhshrink, Threshold, Hardsigmoid, Hardswish, and SiLU.

    Returns:
        Callable: The PyTorch activation function corresponding to the specified name.
    """
    act_funcs = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "leaky_relu": nn.LeakyReLU,
        "prelu": nn.PReLU,
        "rrelu": nn.RReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "celu": nn.CELU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "log_softmax": nn.LogSoftmax,
        "softplus": nn.Softplus,
        "softshrink": nn.Softshrink,
        "softsign": nn.Softsign,
        "tanhshrink": nn.Tanhshrink,
        "threshold": nn.Threshold,
        "hardsigmoid": nn.Hardsigmoid,
        "hardswish": nn.Hardswish,
        "silu": nn.SiLU
    }
    assert act_name.lower() in act_funcs.keys(), f"{act_name.lower()} is not a supported activation function!"
    return act_funcs.get(act_name.lower())


def get_data_loader_class(dataset_name: str) -> Callable:
    """
    Retrieves the appropriate data loader class for a given dataset name.

    Args:
        dataset_name (str): The name of the dataset for which the data loader class is required.
                            Supported dataset names include "imagenet", "imagenet100", "cifar10", etc.

    Returns:
        Callable: The class corresponding to the specified dataset name.
    """
    loader_classes = {
        "imagenet": ImagenetLoader,
        "imagenet100": Imagenet100Loader,
        "cifar10": Cifar10Loader
        # NOTE add more...
    }
    assert dataset_name in loader_classes, f"Unsupported dataset name: {dataset_name}"
    return loader_classes[dataset_name]


def save_checkpoint(checkpoint_data: Union[Dict[str, Any], torch.nn.Module], is_best: bool, checkpoint_dir: str, filename: str = 'checkpoint.pth.tar', jit: bool = False) -> None:
    """
    Saves the model state to a file and also saves a copy if it's the best model so far.

    Args:
        checkpoint_data (Union[Dict[str, Any], torch.nn.Module]): The model or a dictionary containing the state of the model and optimizer to be saved. If it's a model, it should be a PyTorch model object. If it's a dictionary, it typically includes model's state_dict, epoch number, best accuracy, and optimizer state.
        is_best (bool): Flag indicating whether the current checkpoint is the best so far.
        checkpoint_dir (str): The directory path where the checkpoint will be saved.
        filename (str): The filename for the checkpoint file. Default is 'checkpoint.pth.tar'.
        jit (bool): Flag to indicate whether to save the model in TorchScript format. Applicable only if the model is provided.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)

    # Check if checkpoint_data is a model or a dictionary
    if isinstance(checkpoint_data, torch.nn.Module) and jit:
        torch.jit.save(torch.jit.script(checkpoint_data), filepath)
    else:
        torch.save(checkpoint_data, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def json_file_to_dict(file_path: str) -> Dict:
    """
    Reads a JSON file and returns its content as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        Dict: The content of the JSON file as a dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file `{file_path}` does not exist.")

    with open(file_path, "r") as f:
        return json.load(f)


def count_quantizable_layers(model: nn.Module) -> int:
    """
    Counts the number of quantizable layers (currently Conv2d and Linear) in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be inspected.

    Returns:
        int: The number of quantizable layers in the model.
    """
    quantizable_layers = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            quantizable_layers += 1
    return quantizable_layers
