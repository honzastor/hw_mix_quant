# Author: Jan Klhufek (iklhufek@fit.vut.cz)

import sys
import os
import time
import datetime
import random
import math
import numpy as np
import argparse
import json
import copy
from argparse import RawTextHelpFormatter
from threading import Lock
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import List, Callable, Tuple

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR_PATH)
from utils.utils import *
import models as custom_models


def parse_args() -> argparse.Namespace:
    """
    Parses and returns the command line arguments for model training.

    Returns:
        argparse.Namespace: Namespace object containing parsed arguments.
    """
    # Get list of PyTorch model names
    torch_model_names = sorted(name for name in models.__dict__
                               if name.islower() and not name.startswith("__")
                               and callable(models.__dict__[name]))

    # List of your custom model names
    customized_models_names = sorted(name for name in custom_models.__dict__
                                     if name.islower() and not name.startswith("__")
                                     and callable(custom_models.__dict__[name]))

    model_names = torch_model_names + customized_models_names

    for name in custom_models.__dict__:
        if name.islower() and not name.startswith("__") and callable(custom_models.__dict__[name]):
            models.__dict__[name] = custom_models.__dict__[name]

    # List of activation function names from torch.nn.functional
    act_functions = [
        "relu", "relu6", "leaky_relu", "prelu", "rrelu",
        "elu", "selu", "celu", "gelu", "sigmoid", "tanh",
        "softmax", "log_softmax", "softplus", "softshrink",
        "softsign", "tanhshrink", "threshold", "hardsigmoid", "hardswish", "silu"
    ]

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, prog="train",
                                     description="Script for PyTorch Model Training")
    # Model options
    parser.add_argument("-p", "--pretrained", action="store_true",
                        help="specify whether to use pretrained model\n" +
                        "(for example if provided via `pretrained_model` or torchvision pretrained models)")
    parser.add_argument("-m", "--pretrained_model", default="", type=str,
                        help="path to the saved model checkpoint to load parameters. (default: "")")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="mobilenetv1", choices=model_names,
                        help="model architecture: " + " | ".join(model_names) + " (default: mobilenetv1)")
    parser.add_argument("--act_function", metavar="ACT_FUNCTION", default="relu", choices=act_functions,
                        help="activation functions: " + " | ".join(act_functions) + " (default: relu)")
    parser.add_argument("--qat", action="store_true",
                        help="enable quantization-aware training (default is False)")
    parser.add_argument("--symmetric_quant", action="store_true",
                        help="use symmetric or asymmetric quantization (default: False, i.e. asymmetric)")
    parser.add_argument("--per_channel_quant", action="store_true",
                        help="use per-channel or per-tensor quantization (default: False, i.e. per-tensor)")
    parser.add_argument("--quant_setting", default="uniform", type=str, choices=["uniform", "non_uniform"],
                        help="quantization setting: 'uniform' for uniform quantization, 'non_uniform' for custom mixed quantization (default: uniform)")
    parser.add_argument("--uniform_width", default=8, type=int,
                        help="bit-width for uniform quantization (default: 8)")
    parser.add_argument("--non_uniform_width", default=None, type=json_file_to_dict,
                        help="non-uniform bit-width settings\n" +
                        "provide a path to a JSON file representing non-uniform bitwidths for each layer \n" +
                        "Example JSON settings: '{\"layer_1\": {\"Inputs\": 8, \"Weights\": 4, \"Outputs\": 8}, \"layer_2\": {\"Inputs\": 5, \"Weights\": 2, \"Outputs\": 3}}'\n" +
                        "NOTE: the order of keys is important, not the names! (default: None)")
    # Checkpoints
    parser.add_argument("-c", "--checkpoint_path", default="checkpoints", type=str, metavar="PATH",
                        help="path to save checkpoints (default: checkpoints)")
    parser.add_argument("--resume", action="store_true",
                        help="resume training, i.e. the pretrained model serves as the latest checkpoint")
    # Dataset
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to the training dataset")
    parser.add_argument("-n", "--dataset_name", default="imagenet", type=str,
                        help="name of the dataset to be used for training (default: imagenet)")
    parser.add_argument("-w", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers (default: 4)")
    parser.add_argument("-T", "--train_batch", default=256, type=int, metavar="N",
                        help="train batchsize (default: 256)")
    parser.add_argument("-E", "--test_batch", default=512, type=int, metavar="N",
                        help="test batchsize (default: 512)")
    # Train options
    parser.add_argument("--freeze_epochs", default=0, type=int, metavar="N",
                        help="number of initial epochs during which certain layers are frozen (default: 0, i.e. no freezing is applied)")
    parser.add_argument("--freeze_strategy", default="all_but_last", type=str, choices=["all_but_last", "batch_norm_only"],
                        help="strategy for freezing model layers (default: all_but_last)")
    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of total epochs to run (default: 100)")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N",
                        help="start epoch number (default: to 0)")
    parser.add_argument("--warmup_epoch", default=0, type=int, metavar="N",
                        help="number of warmup epochs (default: to 0)")
    parser.add_argument("--lr", "--learning-rate", default=0.05, type=float, metavar="LR",
                        help="initial learning rate (default: 0.05)")
    parser.add_argument("--lr_type", default="cos", type=str,
                        help="lr scheduler (exp/cos/step3/fixed) (default: cos)")
    parser.add_argument("--schedule", type=int, nargs="+", default=[31, 61, 91],
                        help="decrease learning rate at these epochs (default: [31, 61, 91])")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="LR is multiplied by gamma on schedule (default: 0.1)")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                        help="momentum (default: 0.9)")
    parser.add_argument("--weight_decay", "--wd", default=1e-4, type=float, metavar="W",
                        help="weight decay (default: 1e-4)")
    # Device options
    parser.add_argument("-g", "--gpu_id", default="0", type=str,
                        help="GPU ID to use (default: 0)")
    # Miscs
    parser.add_argument("-s", "--manual_seed", default=42, type=int,
                        help="seed for reproducibility. -1 means random. (default: 42)")
    parser.add_argument("-D", "--deterministic", action="store_true",
                        help="enable deterministic mode for CUDA (may impact performance)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable verbose output")
    # Logging script outputs and data to wandb
    parser.add_argument("--log", action="store_true",
                        help="enable logging of script prints")
    parser.add_argument("--wandb", action="store_true",
                        help="enable logging of model training using wandb")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Weights & Biases project name (required if using wandb logging) (default: None)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Weights & Biases entity name (required if using wandb logging) (default: None)")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="Weights & Biases run id (optional for resumed logging of wandb session) (default: None)")
    return parser.parse_args()


def log_print(message: str, args: argparse.Namespace, log_file: str, only_log: bool = False, first_write: bool = False) -> None:
    """
    Prints a message to the console and appends it to a log file if verbose/logging is enabled.

    Args:
        message (str): The message to be printed and logged.
        args (argparse.Namespace): Parsed command-line arguments containing verbose/logging configurations.
        log_file (str): The path to the log file where messages will be written.
        only_log (bool): Toggle whether to just log or allow also verbose print of the message. Defaults to False.
        first_write (bool): If True, the log file is overwritten with the new message. If False, the message is appended to the log file. Defaults to False.
    """
    if args.verbose and only_log is False:
        print(message)
    if args.log:
        if first_write is True:
            with open(log_file, "w") as log_file:
                log_file.write(message + "\n")
        else:
            with open(log_file, "a") as log_file:
                log_file.write(message + "\n")


def init_logging(args: argparse.Namespace) -> Tuple[Logger, str, str, str]:
    """
    Creates and initializes logging/checkpoint files/directories and Logger for the training script.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Tuple[Logger, str, str, str]: A tuple containing instance of the Logger class for logging metrics, path to the checkpoint directory, path with the run settings and metrics and path to the run log file.
    """
    # Create `checkpoints` directory if it doesn't exist
    os.makedirs(args.checkpoint_path, exist_ok=True)

    if args.resume:
        assert os.path.isfile(args.pretrained_model), f"Checkpoint `{args.pretrained_model}` not found!"
        checkpoint_dir = os.path.dirname(args.pretrained_model)
        log_file = os.path.join(checkpoint_dir, "log.txt") if args.log else ""
        log_print("---STARTING TRAINING SCRIPT---\n", args, log_file, first_write=True)
        settings_log = os.path.join(checkpoint_dir, "settings_log.json")
        metrics_log_file = os.path.join(checkpoint_dir, "metrics_log.txt")
        if not os.path.exists(metrics_log_file):
            msg = f"Warning: Log file `{metrics_log_file}` not found. Creating a new log file for logging metrics.\n"
            log_print(msg, args, log_file)
        append_mode = os.path.exists(metrics_log_file)
    else:
        curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        qat_opt = "qat_" if args.qat else ""
        checkpoint_dir = os.path.join(args.checkpoint_path, f"chkpt_{args.arch}_{args.dataset_name}_{qat_opt}{curr_time}_{args.gpu_id}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_file = os.path.join(checkpoint_dir, "log.txt") if args.log else ""
        log_print("---STARTING TRAINING SCRIPT---\n", args, log_file, first_write=True)
        settings_log = os.path.join(checkpoint_dir, "settings_log.json")
        metrics_log_file = os.path.join(checkpoint_dir, "metrics_log.txt")
        append_mode = False

    if args.wandb:
        # Update wandb configuration with the full checkpoint directory path
        wandb.config.update({"checkpoint save path": checkpoint_dir}, allow_val_change=True)

    # Initialize Logger
    logger = Logger(metrics_log_file, resume=append_mode)
    logger.set_names(["Epoch", "Train Loss", "Train Top-1 Acc", "Train Top-5 Acc", "Val Loss", "Val Top-1 Acc", "Val Top-5 Acc", "LR"])
    return logger, checkpoint_dir, settings_log, log_file


def setup_device_and_seed(args: argparse.Namespace) -> str:
    """
    Sets up the device for PyTorch and initializes the seed for random number generation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        str: The string `cuda` if CUDA is available and selected, otherwise `cpu`.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # Random seed setup
    if args.manual_seed == -1:
        args.manual_seed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manual_seed)

    # Set (non)deterministic CUDA operations
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return device


def init_wandb_for_train(args: argparse.Namespace, device: str) -> None:
    """
    Initializes wandb for logging training metrics.
    Args:
        args (argparse.Namespace): The parsed command-line arguments containing
                                   configuration for wandb and other training settings.
        device (str): The device (CPU or CUDA) that is being used for training.
    """
    if args.wandb:
        assert args.wandb_project is not None, "wandb project name must be specified!"
        assert args.wandb_entity is not None, "wandb entity name must be specified!"

        wandb_init_params = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "config": {
                "device": device,
                "model": args.arch,
                "dataset": args.dataset_name,
                "train batch size": args.train_batch,
                "test batch size": args.test_batch,
                "data loading workers": args.workers,
                "activation function": args.act_function,
                "freeze_epochs": args.freeze_epochs,
                "freeze_strategy": args.freeze_strategy,
                "QAT": args.qat,
                "symmetric quantization": args.symmetric_quant,
                "per-channel quantization": args.per_channel_quant,
                "quant_setting": args.quant_setting,
                "uniform_width": args.uniform_width,
                "non_uniform_width": args.non_uniform_width,
                "epochs": args.epochs,
                "start epoch": args.start_epoch,
                "warmup epoch": args.warmup_epoch,
                "learning rate": args.lr,
                "lr scheduler type": args.lr_type,
                "schedule": args.schedule,
                "gamma": args.gamma,
                "momentum": args.momentum,
                "weight decay": args.weight_decay,
                "manual_seed": (args.manual_seed if args.manual_seed is not None else "random"),
                "deterministic": args.deterministic,
                "pretrained": args.pretrained,
                "pretrained model": args.pretrained_model,
                "checkpoint save path": args.checkpoint_path,
                "resume from checkpoint": args.resume
            }
        }

        # If wandb id to resume logging is provided, id must exist, otherwise fail
        if args.wandb_id is not None:
            wandb_init_params["id"] = args.wandb_id
            wandb_init_params["resume"] = "must"

        # Initialize wandb
        wandb.init(**wandb_init_params)

        # Update run ID if resuming
        qat_opt = "qat-" if args.qat else ""
        if args.wandb_id is not None:
            wandb.run.name = f"Train-{args.arch}-{args.dataset_name}-{qat_opt}{args.wandb_id}"
        else:
            wandb.run.name = f"Train-{args.arch}-{args.dataset_name}-{qat_opt}{wandb.run.id}"


def load_model(args: argparse.Namespace, arch: Callable, pretrained_model: str, input_size: int, num_classes: int) -> nn.Module:
    """
    Loads a model from a specified file path.

    Args:
        args (argparse.Namespace): The parsed command-line arguments containing
                                   configuration for wandb and other training settings.
        arch (Callable): A callable that returns an instance of the desired model architecture.
        pretrained_model (str): The path to the file containing the model's saved state dictionary.
        input_size (int): Input image data size.
        num_classes (int): Number of classes for the classification task.

    Returns:
        nn.Module: The loaded model.
    """
    # Load and return the model
    arg_dict = vars(args)
    arg_dict["checkpoint_path"] = pretrained_model
    arg_dict["num_classes"] = num_classes
    arg_dict["input_size"] = input_size
    model = arch(**arg_dict)

    return model


def freeze_layers(model: nn.Module, freeze_strategy: str, inplace: bool = True) -> Union[nn.Module, None]:
    """
    Freezes all layers of the model except the last one.

    NOTE: Different freezing strategies for better control of transfer learning process can be added here.

    Args:
        model (nn.Module): The model whose layers are to be frozen.
        freeze_strategy (str): The strategy used for freezing model layers.
        inplace (bool): If True, modifies the model in place. If False, returns a new model with frozen layers.

    Returns:
        nn.Module or None: The modified model with frozen layers if inplace is False, otherwise None.
    """
    if not inplace:
        model = copy.deepcopy(model)

    if freeze_strategy == "all_but_last":  # Freeze all layers except the last one
        # The name of the last layer's parameters for the given model
        last_layer_name = next(reversed(list(model.named_parameters())))[0]

        for name, param in model.named_parameters():
            if name != last_layer_name:
                param.requires_grad = False
    elif freeze_strategy == "batch_norm_only":  # Specifically freeze batch normalization layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for param in module.parameters():
                    param.requires_grad = False

    return model if not inplace else None


def unfreeze_layers(model: nn.Module, inplace: bool = True) -> Union[nn.Module, None]:
    """
    Unfreezes all layers of the model.

    Args:
        model (nn.Module): The model whose layers are to be unfrozen.
        inplace (bool): If True, modifies the model in place. If False, returns a new model with unfrozen layers.

    Returns:
        nn.Module or None: The modified model with unfrozen layers if inplace is False, otherwise None.
    """
    if not inplace:
        model = copy.deepcopy(model)

    for param in model.parameters():
        param.requires_grad = True

    return model if not inplace else None


def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, scaler: GradScaler, device: str, log_file: str) -> Tuple[float, float, float]:
    """
    Trains the model for one epoch on the given training dataset.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): The DataLoader for the training dataset.
        criterion (nn.Module): The loss function to use for training.
        optimizer (optim.Optimizer): The optimization algorithm used for updating the model parameters.
        scaler (GradScaler): Gradient scaler for mixed-precision training.
        device (str): The device (CPU or CUDA) on which to perform the training.
        log_file (str): The path to the log file where progress will be logged.

    Returns:
        Tuple[float, float, float]: A tuple containing the final training loss, top-1 accuracy, and top-5 accuracy on the training dataset.
    """
    model.train()  # Set model to training mode

    batch_time = AverageMeter(name="Batch time")
    losses = AverageMeter(name="Loss")
    top1 = AverageMeter(name="Top1 acc")
    top5 = AverageMeter(name="Top5 acc")

    # Initialize ProgressMeter
    progress = ProgressMeter(num_batches=len(train_loader), meters=[batch_time, losses, top1, top5], prefix="Train: ")

    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Clears old gradients
        optimizer.zero_grad()

        # Forward pass
        if not model.qat:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # Display progress
        batch_time.update(time.time() - start)
        start = time.time()
        progress.display(batch_idx, log_file)

    return losses.avg, top1.avg, top5.avg


def test(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: str, log_file: str) -> Tuple[float, float, float]:
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        val_loader (DataLoader): The DataLoader for the evaluation dataset.
        criterion (nn.Module): The loss function to use for evaluation.
        device (str): The device (CPU or CUDA) on which to perform the evaluation.
        log_file (str): The path to the log file where progress will be logged.

    Returns:
        Tuple[float, float, float]: A tuple containing average loss, top-1 accuracy, and top-5 accuracy.
    """
    batch_time = AverageMeter(name="Batch time")
    losses = AverageMeter(name="Loss")
    top1 = AverageMeter(name="Top1 acc")
    top5 = AverageMeter(name="Top5 acc")

    # Initialize ProgressMeter
    progress = ProgressMeter(num_batches=len(val_loader), meters=[batch_time, losses, top1, top5], prefix="Test: ")

    # Evaluation loop
    with torch.no_grad():
        model.eval()
        end = time.time()

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Measure metrics
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Display progress
            progress.display(batch_idx, log_file)
        progress.display_summary(log_file)

    return losses.avg, top1.avg, top5.avg


def adjust_learning_rate(args: argparse.Namespace, optimizer: optim.Optimizer, epoch: int) -> float:
    """
    Adjusts the learning rate based on the epoch and specified schedule.

    This piece of code was inspired by the HAQ work: https://github.com/mit-han-lab/haq/blob/master/pretrain.py

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing schedule parameters.
        optimizer (optim.Optimizer): The optimizer for which the learning rate will be adjusted.
        epoch (int): The current epoch number.

    Returns:
        float: The adjusted learning rate for the current epoch.
    """
    if epoch < args.warmup_epoch:
        lr_current = args.lr * args.gamma
    elif args.lr_type == "cos":
        # cos
        lr_current = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
    elif args.lr_type == "exp":
        step = 1
        decay = args.gamma
        lr_current = args.lr * (decay ** (epoch // step))
    elif epoch in args.schedule:
        lr_current *= args.gamma
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_current
    return lr_current


def main(args: Optional[argparse.Namespace] = None) -> float:
    """
    Main function for training the model.

    This function can either be called with a set of arguments passed as an `argparse.Namespace` object or it will (by default) parse the command-line arguments. The function sets up the training environment, 
    loads the data and model, and executes the training and validation process. It supports both standard 
    training and quantization-aware training (QAT).

    Args:
        args (Optional[argparse.Namespace]): A namespace object containing training parameters. If None, 
        the function will parse command-line arguments. The namespace object should include all the required 
        arguments that would otherwise be passed through the command line.

    Returns:
        float: The best accuracy achieved during training. If QAT is enabled, it returns the best accuracy 
        of the quantized model; otherwise, it returns the best accuracy of the floating-point model.
    """
    # Parse command-line arguments and initiliaze
    if args is None:
        args = parse_args()
    args.act_function = get_activation_function(args.act_function)
    if not hasattr(args, 'qat_evaluation_lock'):  # Check if the shared lock is passed (from multigpu nsga), otherwise use a standard lock
        args.qat_evaluation_lock = Lock()
    device = setup_device_and_seed(args=args)
    init_wandb_for_train(args=args, device=device)
    logger, checkpoint_dir, settings_log, log_file = init_logging(args=args)

    # Log run's settings into JSON
    # Temporarily save and remove the lock from args (TO ENABLE THE DEEPCOPY)
    temp_lock = args.qat_evaluation_lock if 'qat_evaluation_lock' in args.__dict__ else None
    if 'qat_evaluation_lock' in args.__dict__:
        del args.__dict__['qat_evaluation_lock']
    settings = copy.deepcopy(vars(args))
    settings["act_function"] = str(settings["act_function"])
    settings["device"] = str(device)
    with open(settings_log, 'w') as f:
        json.dump(settings, f, indent=2)

    # Restore the lock in args
    if temp_lock:
        args.qat_evaluation_lock = temp_lock

    messages = (
        f"Using {device} device\n"
        f"The seed is: {args.manual_seed}\n"
        f"Deterministic CUDA: {args.deterministic}\n"
        f"Logging with W&B: {args.wandb}\n"
        f"Loading data..\n"
    )
    log_print(messages, args, log_file)

    # Prepare the dataset and data loader
    assert args.dataset_name == args.data.split('/')[-1], f"Dataset name `{args.dataset_name}` does not match with the last part of the dataset path `{args.data}` (i.e. `{args.data.split('/')[-1]}`). This is a precautious assert."
    data_loader_class = get_data_loader_class(args.dataset_name)
    data_loader = data_loader_class(dataset_path=args.data)

    # Load data
    train_loader = data_loader.load_training_data(batch_size=args.train_batch, num_workers=args.workers, pin_memory=device == "cuda")
    val_loader = data_loader.load_validation_data(batch_size=args.test_batch, num_workers=args.workers, pin_memory=device == "cuda")
    num_classes = data_loader.classes

    messages = (
        f"Dataset name: {args.dataset_name}\n"
        f"Dataset contains {data_loader.num_train_batches} train batches, {data_loader.num_val_batches} test batches and {num_classes} classes\n"
        f"Loading model..\n"
    )
    log_print(messages, args, log_file)

    # Check and set quantization settings:
    if args.qat:
        temp_model = models.__dict__[args.arch]()
        num_quant_layers = count_quantizable_layers(temp_model)

        if args.quant_setting == 'non_uniform':
            assert args.non_uniform_width is not None, "Non-uniform quantization config is required but not provided."
            # Validate the non-uniform quantization config
            assert len(args.non_uniform_width) == num_quant_layers, f"The provided non-uniform quantization config does not match the number of quantizable layers in the model. Expected {num_quant_layers} entries, got {len(args.non_uniform_width)}."
            # Renaming keys to match layer indices
            args.quant_config = {i: config for i, (_, config) in enumerate(args.non_uniform_width.items())}
        elif args.quant_setting == 'uniform':
            # Create a uniform quantization config
            args.quant_config = {i: {"Inputs": args.uniform_width, "Weights": args.uniform_width} for i in range(num_quant_layers)}

    # Load the model
    if args.resume:
        args.pretrained = True  # set it if user forgets, otherwise the model's state dict is not properly loaded in
    model = load_model(args=args, arch=models.__dict__[args.arch], pretrained_model=args.pretrained_model, input_size=data_loader.input_size, num_classes=num_classes, device=device, log_file=log_file)

    # Print model details if verbose
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model, args.gpu_id)
    messages = (
        f"Model Architecture: {args.arch}\n"
        f"Total Parameters: {total_params}\n"
        f"Trainable Parameters: {trainable_params}\n"
        f"Model Size: {model_size:.2f} MB\n"
        f"Checkpoint: {checkpoint_dir}\n\n"
    )
    log_print(messages, args, log_file)

    # Log model details to wandb
    if args.wandb:
        wandb.log({
            "Total Parameters": total_params,
            "Trainable Parameters": trainable_params,
            "Model Size MB": f"{model_size:.2f}"
        })

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = GradScaler()
    best_acc = 0.0

    # Load pretrained model's optimizer's state dict if specified and no checkpoint is provided
    if args.pretrained and args.pretrained_model != "" and not args.resume:
        assert os.path.exists(args.pretrained_model), f"Pretrained model path `{args.pretrained_model}` not found!"
        pretrain = torch.load(args.pretrained_model)
        optimizer_state = pretrain["optimizer"]
        current_param_ids = {id(p) for p in model.parameters()}
        # Remove entries from the optimizer state dict for parameters that have been replaced (i.e. if last layer has different number of outputs)
        optimizer_state["state"] = {k: v for k, v in optimizer_state["state"].items() if k in current_param_ids}
        # Load the adjusted state dict into the optimizer
        optimizer.load_state_dict(optimizer_state)

    # Load checkpoint's optimizer state if resuming from checkpoint
    if args.resume:
        assert os.path.isfile(args.pretrained_model), f"Checkpoint `{args.pretrained_model}` not found!"
        checkpoint = torch.load(args.pretrained_model)
        optimizer_state = checkpoint["optimizer"]
        current_param_ids = {id(p) for p in model.parameters()}
        # Remove entries from the optimizer state dict for parameters that have been replaced (i.e. if last layer has different number of outputs)
        optimizer_state["state"] = {k: v for k, v in optimizer_state["state"].items() if k in current_param_ids}
        # Load the adjusted state dict into the optimizer
        optimizer.load_state_dict(optimizer_state)
        args.start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]

    # Train and log
    if args.wandb:
        wandb.watch(model)
    print(f"Starting training:")
    log_print(f"Starting training:\n", args, log_file, only_log=True)
    if args.start_epoch < args.freeze_epochs:
        log_print(f"Freezing `{args.freeze_strategy}` layers\n", args, log_file)
        freeze_layers(model, args.freeze_strategy, inplace=True)

    for epoch in range(args.start_epoch, args.epochs):
        current_lr = adjust_learning_rate(args, optimizer, epoch)
        print(f"\nEpoch: {epoch+1}, learning rate: {current_lr}")
        log_print(f"\nEpoch: {epoch+1}, learning rate: {current_lr}\n", args, log_file, only_log=True)

        # Disable freezing
        if epoch+1 == args.freeze_epochs:
            unfreeze_layers(model, inplace=True)

        # Train for one epoch
        train_loss, train_top1, train_top5 = train(model, train_loader, criterion, optimizer, scaler, device, log_file)
        # Evaluate on validation set
        val_loss, val_top1, val_top5 = test(model, val_loader, criterion, device, log_file)
        # Update logger
        logger.append([f"{epoch+1:4d}", f"{train_loss:10.6f}", f"{train_top1:15.6f}", f"{train_top5:15.6f}", f"{val_loss:9.6f}", f"{val_top1:12.6f}", f"{val_top5:13.6f}", f"{current_lr:4.6f}"])

        # Check if this is the best model so far
        is_best = val_top1 > best_acc
        best_acc = max(val_top1, best_acc)

        # Save checkpoint
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }, is_best=is_best, checkpoint_dir=checkpoint_dir)

        if args.wandb:
            wandb.log({
                "Epoch": epoch+1,
                "Training Loss": train_loss,
                "Training Top-1 Accuracy": train_top1,
                "Training Top-5 Accuracy": train_top5,
                "Validation Loss": val_loss,
                "Validation Top-1 Accuracy": val_top1,
                "Validation Top-5 Accuracy": val_top5,
                "Learning Rate": current_lr,
                "Best Top-1 Accuracy": best_acc
            })

    final_stats = {}
    final_stats["FP32 Last Training Top-1 Accuracy"] = train_top1
    final_stats["FP32 Last Training Top-5 Accuracy"] = train_top5
    final_stats["FP32 Last Training Loss"] = train_loss
    final_stats["FP32 Last Validation Top-1 Accuracy"] = val_top1
    final_stats["FP32 Last Validation Top-5 Accuracy"] = val_top5
    final_stats["FP32 Last Validation Loss"] = val_loss
    final_stats["FP32 Best Top-1 Accuracy"] = best_acc
    final_stats["wandb_id"] = wandb.run.id if args.wandb else None
    if args.qat:  # Saving the best checkpoint after QAT
        best_model_path = os.path.join(checkpoint_dir, "model_best.pth.tar")
        if os.path.isfile(best_model_path):
            best_model_checkpoint = torch.load(os.path.join(checkpoint_dir, "model_best.pth.tar"))
            model.load_state_dict(best_model_checkpoint["state_dict"])

        print("\n\n------Testing accuracy (on GPU) of the floating point fake quantized model at the end of training before conversion to INT------")
        val_loss, val_top1, val_top5 = test(model, val_loader, criterion, device, log_file)

        # Convert model after QAT
        model.to("cpu")  # NOTE: Quantization operations in PyTorch are optimized for CPU backend inference (i.e. utilization of vectorization, etc.).
        model.eval()
        torch.ao.quantization.convert(model, inplace=True)

        # Save checkpoint
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }, is_best=False, checkpoint_dir=checkpoint_dir, filename="model_after_qat.pth.tar")
        save_checkpoint(checkpoint_data=model, is_best=False, checkpoint_dir=checkpoint_dir, filename="jit_model_after_qat.pth.tar", jit=True)

        # Locked cpu evaluation
        with args.qat_evaluation_lock:
            print("\n\n------Testing accuracy after converting the model into INT------")
            # Setting the batch size for eval low here to prevent possible run out of RAM
            cpu_val_loader = data_loader.load_validation_data(batch_size=16, num_workers=args.workers, pin_memory=False)
            int_loss, int_val_top1, int_val_top5 = test(model, cpu_val_loader, criterion, "cpu", log_file)

        logger.append(["\n    AFTER QAT", "      Loss", "Top-1 Acc", "Top-5 Acc"])
        logger.append([f"Converted INT model", f"{int_loss:4.6f}", f"{int_val_top1:7.6f}", f"{int_val_top5:9.6f}"])

        qat_model_size = get_model_size(model, args.gpu_id)
        messages = (
            f"Model Size After QAT: {qat_model_size:.2f} MB\n"
            f"After QAT Top-1 Accuracy: {int_val_top1}\n"
            f"After QAT Top-5 Accuracy: {int_val_top5}\n\n"
        )
        log_print(messages, args, log_file, only_log=True)
        # Log QAT model details to wandb
        if args.wandb:
            wandb.log({
                "Model Size After QAT MB": f"{qat_model_size:.2f}",
                "After QAT Top-1 Accuracy": int_val_top1,
                "After QAT Top-5 Accuracy": int_val_top5
            })

        model_after_qat_path = os.path.join(checkpoint_dir, "model_after_qat.pth.tar")
        print(f"Converted quantized model saved to {model_after_qat_path}")
        log_print(f"Converted quantized model saved to {model_after_qat_path}\n", args, log_file, only_log=True)

        print("QAT and model conversion complete. Best INT TOP-1 accuracy: {:.2f}%".format(int_val_top1))
        log_print("QAT and model conversion complete. Best INT TOP-1 accuracy: {:.2f}%\n".format(int_val_top1), args, log_file, only_log=True)

        # Log final stats into JSON
        final_stats["Int Best Validation Top-1 Accuracy"] = int_val_top1
        final_stats["Int Best Validation Top-5 Accuracy"] = int_val_top5
        final_stats["Int Validation Loss"] = int_loss
        with open(settings_log, 'r') as f:
            data = json.load(f)
        data.update(final_stats)
        with open(settings_log, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        print("Training complete. Best TOP-1 accuracy: {:.2f}%".format(best_acc))
        log_print("Training complete. Best TOP-1 accuracy: {:.2f}%\n".format(best_acc), args, log_file, only_log=True)

        # Log final stats into JSON
        with open(settings_log, 'r') as f:
            data = json.load(f)
        data.update(final_stats)
        with open(settings_log, 'w') as f:
            json.dump(data, f, indent=2)

    # Close the logger
    logger.close()
    return int_val_top1 if args.qat else best_acc


if __name__ == "__main__":
    main()
