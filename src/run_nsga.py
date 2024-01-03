# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz), Jan Klhufek (iklhufek@fit.vut.cz)

import argparse
from argparse import RawTextHelpFormatter
import datetime
from torchvision import models

import pytorch.models as custom_models
from nsga.nsga_qat import QATNSGA
from nsga.nsga_qat_multigpu import MultiGPUQATNSGA


def parse_args() -> argparse.Namespace:
    """
    Parses and returns the command line arguments for NSGA guided mixed-precision QAT training.

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

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter, prog="run_nsga",
                                     description="Script for running NSGA-II for non-uniform QAT")
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
    parser.add_argument("--symmetric_quant", action="store_true",
                        help="use symmetric or asymmetric quantization (default: False, i.e. asymmetric)")
    parser.add_argument("--per_channel_quant", action="store_true",
                        help="use per-channel or per-tensor quantization (default: False, i.e. per-tensor)")
    # Dataset
    parser.add_argument("-d", "--data", type=str, required=True,
                        help="path to the training dataset")
    parser.add_argument("-n", "--dataset_name", default="imagenet100", type=str,
                        help="name of the dataset to be used for training")
    parser.add_argument("-w", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers (default: 4)")
    parser.add_argument("-T", "--train_batch", default=256, type=int, metavar="N",
                        help="train batchsize (default: 256)")
    parser.add_argument("-E", "--test_batch", default=512, type=int, metavar="N",
                        help="test batchsize (default: 512)")
    # NSGA settings
    parser.add_argument("--parent_size", type=int, default=10,
                        help="number of parents in the population for each generation of NSGA-II (default: 10)")
    parser.add_argument('--offspring_size', type=int, default=10,
                        help="number of offsprings to be generated in each generation of NSGA-II (default: 10)")
    parser.add_argument("--generations", type=int, default=25,
                        help="number of generations for NSGA-II (default: 25)")
    # Checkpoints
    parser.add_argument("--previous_run", type=str, default=None,
                        help="logs dir of previous run to continue")
    # Train options
    parser.add_argument("--qat_epochs", type=int, default=10,
                        help="number of epochs for QAT for each individual (model configuration) during NSGA-II (default: 10)")
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float,
                        help="initial learning rate (default: 0.1)")
    parser.add_argument("--lr_type", default="cos", type=str,
                        help="lr scheduler (exp/cos/step3/fixed) (default: cos)")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="lr is multiplied by gamma on schedule (default: 0.1)")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                        help="momentum (default: 0.9)")
    parser.add_argument("--weight-decay", "--wd", default=1e-5, type=float, metavar="W",
                        help="weight decay (default: 1e-5)")
    parser.add_argument("--multigpu", default=False, action="store_true",
                        help="use multiple GPUs")
    # Timeloop settings (if not HW metrics are not cached)
    parser.add_argument("--timeloop_architecture", type=str, default="eyeriss", choices=["eyeriss", "simba"],
                        help="name of the HW architecture to be used along with its associated components and constraints")
    parser.add_argument("--timeloop_heuristic", type=str, default="random", choices=["random", "exhaustive", "hybrid", "linear"],
                        help="heuristic type to use for timeloop-mapper (default: random)")
    parser.add_argument("--total_valid", type=int, default=0,
                        help="number of total valid mappings to consider across all available mapper threads; "
                        "a value of 0 means that this criteria is not used for thread termination (default: 0)")
    parser.add_argument("--primary_metric", type=str, default="edp", choices=["energy", "delay", "lla", "edp", "memsize_bytes"],
                        help="primary metric for timeloop-mapper to optimize for; NOTE: this is what NSGA-II acknowledges\n" +
                        "choose from 'energy', 'delay', 'lla', 'edp'  (default: 'edp')")

    parser.add_argument("--secondary_metric", type=str, default="", choices=["energy", "delay", "lla", ""],
                        help="secondary metric for timeloop-mapper to optimize for, optional; NOTE: this is what NSGA-II does not acknowledge and is used only for mapping search\n" +
                        "choose from 'energy', 'delay', 'lla', or '' if no secondary metric is desired\n" +
                        "NOTE: edp does not require secondary metric (default: '')")
    parser.add_argument("--cache_directory", type=str, default="timeloop_caches",
                        help="Directory to store cache files for hardware metrics estimated by Timeloop. (default: 'timeloop_caches')")
    parser.add_argument("--cache_name", type=str, default="cache",
                        help="Name of the cache file to store hardware metrics estimated by Timeloop. (default: 'cache')")
    parser.add_argument("--run_id", type=str, default="1",
                        help="ID of the current run to distinguish its own cache for writing privileges. (default: 'cache')")
    # Miscs
    parser.add_argument("-s", "--manual_seed", default=42, type=int,
                        help="manual seed for reproducibility. -1 means random. (default: 42)")
    parser.add_argument("-D", "--deterministic", action="store_true",
                        help="enable deterministic mode for CUDA (may impact performance)")
    parser.add_argument("--logs_dir", type=str, default="/tmp/run" + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S"),
                        help="logs dir (default: '/tmp/run' + datetime.datetime.now().strftime('-%Y%m%d-%H%M%S'))")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.multigpu:
        print("Initializing QAT NSGA-II MultiGPU")
        nsga = MultiGPUQATNSGA(**vars(args))
    else:
        print("Initializing QAT NSGA-II")
        nsga = QATNSGA(**vars(args))

    nsga.run()


if __name__ == "__main__":
    main()
