# Author: Jan Klhufek (iklhufek@fit.vut.cz)

import os
import time
import random
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import wandb
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from typing import Callable

from utils.utils import *
import models as custom_models


def parse_args() -> argparse.Namespace:
    """
    Parses and returns the command line arguments for model evaluation.

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

    for name in custom_models.__dict__:
        if name.islower() and not name.startswith("__") and callable(custom_models.__dict__[name]):
            models.__dict__[name] = custom_models.__dict__[name]

    model_names = torch_model_names + customized_models_names

    # List of activation function names from torch.nn.functional
    act_functions = [
        "relu", "relu6", "leaky_relu", "prelu", "rrelu",
        "elu", "selu", "celu", "gelu", "sigmoid", "tanh",
        "softmax", "log_softmax", "softplus", "softshrink",
        "softsign", "tanhshrink", "threshold", "hardsigmoid", "hardswish", "silu"
    ]

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, prog="eval",
                                     description="Script for PyTorch Model Evaluation")
    # Model
    parser.add_argument("-p", "--pretrained", action="store_true",
                        help="specify whether to use pretrained model\n" +
                        "(for example if provided via `pretrained_model` or torchvision pretrained models)")
    parser.add_argument("-q", "--quantized", action="store_true",
                        help="specify whether the pretrained model is already converted into quantized form"
                        "(the model should be provided  via `pretrained_model`)")
    parser.add_argument("-m", "--pretrained_model", default="", type=str,
                        help="path to the saved model checkpoint to load parameters. (default: "")")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="mobilenetv1", choices=model_names,
                        help="model architecture: " + " | ".join(model_names) + " (default: mobilenetv1)")
    parser.add_argument("--act_function", metavar="ACT_FUNCTION", default="relu", choices=act_functions,
                        help="activation functions: " + " | ".join(act_functions) + " (default: relu)")
    parser.add_argument("--qat", action="store_true",
                        help="enable quantization-aware training (default: False)")
    parser.add_argument("--symmetric_quant", action="store_true",
                        help="use symmetric or asymmetric quantization (default: is False, i.e. asymmetric)")
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
    # Dataset
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to the evaluation dataset")
    parser.add_argument("-n", "--dataset_name", default="imagenet", type=str,
                        help="name of the dataset to be used for evaluation (default: imagenet)")
    parser.add_argument("-w", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers (default: 4)")
    parser.add_argument("-b", "--test_batch", default=512, type=int, metavar="N",
                        help="test batchsize (default: 512)")
    # Eval options
    parser.add_argument("--half_tensor", action="store_true",
                        help="use half tensor (FP16) precision for inputs and model weights (default: False)")
    parser.add_argument("--use_cpu", action="store_true",
                        help="strictly use cpu (default is False)")
    # Device options
    parser.add_argument("-g", "--gpu_id", default="0", type=str,
                        help="GPU ID to use")
    # Miscs
    parser.add_argument("-s", "--manual_seed", default=42, type=int,
                        help="manual seed for reproducibility. -1 means random. (default: 42)")
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


def init_logging(args: argparse.Namespace) -> Tuple[Logger, str]:
    """
    Creates and initializes logging files and Logger for the evaluation script.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Tuple[Logger, str]: A tuple containing instance of the Logger class for logging metrics and path to the log file.
    """
    log_file = "eval_log.txt" if args.log else ""
    log_print("---STARTING TRAINING SCRIPT---\n", args, log_file, first_write=True)
    metrics_log_file = "eval_metrics_log.txt"
    # Initialize Logger
    logger = Logger(metrics_log_file)
    logger.set_names(["Avg Valid Loss", "Avg Valid Top-1 Accuracy", "Avg Valid Top-5 Accuracy"])
    return logger, log_file


def setup_device_and_seed(args: argparse.Namespace) -> str:
    """
    Sets up the device for PyTorch and initializes the seed for random number generation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        str: The string `cuda` if CUDA is available and selected, otherwise `cpu`.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    device = "cuda" if (use_cuda and not args.use_cpu and not args.qat) else "cpu"

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


def init_wandb_for_eval(args: argparse.Namespace, device: str) -> None:
    """
    Initializes wandb for logging evaluation metrics.
    Args:
        args (argparse.Namespace): The parsed command-line arguments containing
                                   configuration for wandb and other evaluation settings.
        device (str): The device (CPU or CUDA) that is being used for evaluation.
    """
    if args.wandb:
        assert args.wandb_project is not None, "wandb project name must be specified!"
        assert args.wandb_entity is not None, "wandb entity name must be specified!"

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "device": device,
                "model": args.arch,
                "dataset": args.dataset_name,
                "test batch size": args.test_batch,
                "data loading workers": args.workers,
                "activation function": args.act_function,
                "half precision (FP16)": args.half_tensor,
                "QAT": args.qat,
                "symmetric quantization": args.symmetric_quant,
                "per-channel quantization": args.per_channel_quant,
                "quant_setting": args.quant_setting,
                "uniform_width": args.uniform_width,
                "non_uniform_width": args.non_uniform_width,
                "Use CPU": args.use_cpu,
                "manual_seed": (args.manual_seed if args.manual_seed is not None else "random"),
                "deterministic": args.deterministic,
                "pretrained": args.pretrained,
                "quantized": args.quantized,
                "pretrained model": args.pretrained_model
            }
        )
        wandb.run.name = "Eval-" + wandb.run.id


def load_model(args: argparse.Namespace, arch: Callable, pretrained_model: str, quantized: bool, input_size: int, num_classes: int, device: str, log_file: str) -> nn.Module:
    """
    Loads a model from a specified file path.

    Args:
        args (argparse.Namespace): The parsed command-line arguments containing
                                   configuration for wandb and other evaluation settings.
        arch (Callable): A callable that returns an instance of the desired model architecture.
        pretrained_model (str): The path to the file containing the model's saved state dictionary.
        quantized (bool): Specifies whether the pretrained model has already been quantized or is still in floating-point precision.
        input_size (int): Input image data size.
        num_classes (int): Number of classes for the classification task.
        device (str): The device (CPU or CUDA) that is being used for evaluation.
        log_file (str): Path to the log file.

    Returns:
        nn.Module: The loaded model.
    """
    # Load and return the model
    arg_dict = vars(args)
    arg_dict["checkpoint_path"] = pretrained_model
    arg_dict["num_classes"] = num_classes
    arg_dict["input_size"] = input_size
    model = arch(**arg_dict)

    # Check for multiple GPUs and wrap the model with DataParallel if so
    if torch.cuda.device_count() > 1:
        log_print(f"Using {torch.cuda.device_count()} GPUs\n", args, log_file)
        model = nn.DataParallel(model)
    model.to(device)
    return model


def test(model: nn.Module, val_loader: DataLoader, device: str, criterion: Optional[nn.Module] = None, half_tensor: bool = False) -> Tuple[Optional[float], float, float]:
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        val_loader (DataLoader): The DataLoader for the evaluation dataset.
        device (str): The device (CPU or CUDA) on which to perform the evaluation.
        criterion (Optional[nn.Module]): The loss function to use for evaluation, or None if not needed. Defaults to None.
        half_tensor (bool): Specify whether to convert input data to half precision (FP16). Defaults to False.

    Returns:
        Tuple[Optional[float], float, float]: A tuple containing average loss (None if criterion is None), top-1 accuracy, and top-5 accuracy.
    """
    # Assert to ensure half precision (FP16) is not used on CPU
    assert not (device == "cpu" and half_tensor), "FP16 (half tensor) is not supported on CPU."
    batch_time = AverageMeter(name="Batch time")
    losses = AverageMeter(name="Loss") if criterion else None
    top1 = AverageMeter(name="Top1 acc")
    top5 = AverageMeter(name="Top5 acc")

    # Initialize ProgressMeter
    progress = ProgressMeter(
        num_batches=len(val_loader),
        meters=[batch_time, losses, top1, top5] if criterion else [batch_time, top1, top5],
        prefix="Test: "
    )

    # Evaluation loop
    with torch.no_grad():
        model.eval()
        end = time.time()

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Convert inputs to half precision if desired
            inputs = inputs.half() if half_tensor else inputs

            outputs = model(inputs)
            # Forward pass
            if criterion:
                loss = criterion(outputs, targets)
                losses.update(loss.item(), inputs.size(0))

            # Measure metrics
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Display progress
            progress.display(batch_idx)
        progress.display_summary()

    return (losses.avg if criterion else 0), top1.avg, top5.avg


def main() -> None:
    # Parse command-line arguments and initiliaze
    args = parse_args()
    args.act_function = get_activation_function(args.act_function)
    device = setup_device_and_seed(args=args)
    # Assert to ensure half precision (FP16) is not used on CPU
    assert not (device == "cpu" and args.half_tensor), "FP16 (half tensor) is not supported on CPU."

    init_wandb_for_eval(args=args, device=device)
    logger, log_file = init_logging(args=args)

    messages = (
        f"Using {device} device\n"
        f"The seed is: {args.manual_seed}\n"
        f"Deterministic CUDA: {args.deterministic}\n"
        f"Logging with W&B: {args.wandb}\n"
        f"Loading data..\n"
    )
    log_print(messages, args, log_file)

    # Prepare the dataset and dataloader
    assert args.dataset_name == args.data.split('/')[-1], f"Dataset name `{args.dataset_name}` does not match with the last part of the dataset path `{args.data}` (i.e. `{args.data.split('/')[-1]}`). This is a precautious assert."
    data_loader_class = get_data_loader_class(args.dataset_name)
    data_loader = data_loader_class(dataset_path=args.data)

    val_loader = data_loader.load_validation_data(batch_size=args.test_batch, num_workers=args.workers, pin_memory=device == "cuda")
    num_classes = data_loader.classes

    messages = (
        f"Dataset name: {args.dataset_name}\n"
        f"Dataset contains {data_loader.num_val_batches} test batches and {num_classes} classes\n"
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
    model = load_model(args=args, arch=models.__dict__[args.arch], pretrained_model=args.pretrained_model, quantized=args.quantized, input_size=data_loader.input_size, num_classes=num_classes, device=device, log_file=log_file)
    if args.qat and not args.quantized:
        print("Converting model to quantized form")
        model.to("cpu")
        model.eval()
        model = torch.ao.quantization.convert(model, inplace=True)

    # Print model details if verbose
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model, args.gpu_id)
    messages = (
        f"Model Architecture: {args.arch}\n"
        f"Total Parameters: {total_params}\n"
        f"Trainable Parameters: {trainable_params}\n"
        f"Model Size: {model_size:.2f} MB\n\n"
    )
    log_print(messages, args, log_file)

    # Log model details to wandb
    if args.wandb:
        wandb.log({
            "Total Parameters": total_params,
            "Trainable Parameters": trainable_params,
            "Model Size MB": model_size
        })

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    avg_loss, avg_top1, avg_top5 = test(model=model, val_loader=val_loader, criterion=criterion, device=device, half_tensor=args.half_tensor)

    # Print the evaluation results
    logger.append([avg_loss, avg_top1, avg_top5])
    if args.wandb:
        wandb.log({"Avg Loss": avg_loss, "Avg Top-1 Accuracy": avg_top1, "Avg Top-5 Accuracy": avg_top5})

    print(f"Evaluation Results: Avg Loss = {avg_loss:.4f}, Avg Top-1 Accuracy = {avg_top1:.2f}%, Avg Top-5 Accuracy = {avg_top5:.2f}%")
    log_print(f"Evaluation Results: Avg Loss = {avg_loss:.4f}, Avg Top-1 Accuracy = {avg_top1:.2f}%, Avg Top-5 Accuracy = {avg_top5:.2f}%\n", args, log_file, only_log=True)

    # Close the logger
    logger.close()


if __name__ == "__main__":
    main()
