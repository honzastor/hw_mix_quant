# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz), Jan Klhufek (iklhufek@fit.vut.cz)

import os
import glob
import json
import gzip
import random
import numpy as np
import argparse
import time
import datetime
import copy
import torch
import torchvision.models as models
from collections import OrderedDict
from typing import Optional, Dict, List, Any, Generator

from .nsga import NSGA, NSGAAnalyzer
from pytorch import train
import pytorch.models as custom_models
from mapper_facade import MapperFacade
from mapper_facade import JSONEncoder


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


class QATNSGA(NSGA):
    """
    NSGA-II for proposed framework, it uses QAT Analyzer for evaluation of individuals.
    """

    def __init__(self, data: str, dataset_name: str, pretrained_model: str, arch: str = "mobilenetv1", pretrained: bool = False,
                 act_function: str = "relu", symmetric_quant: bool = False, per_channel_quant: bool = False, workers: int = 4, train_batch: int = 256, test_batch: int = 512, generations: int = 25, parent_size: int = 10, offspring_size: int = 10, timeloop_architecture: str = "eyeriss", timeloop_heuristic: str = "random", total_valid: int = 0, primary_metric: str = "edp", secondary_metric: Optional[str] = "", cache_directory: str = "timeloop_caches", cache_name: str = "cache", run_id: str = "1", previous_run: Optional[str] = None, qat_epochs: int = 10, lr: float = 0.1, lr_type: str = "cos", gamma: float = 0.1, momentum: float = 0.9, weight_decay: float = 1e-5, manual_seed: int = 42, deterministic: bool = False, logs_dir: str = "/tmp/run" + datetime.datetime.now().strftime("-%Y%m%d-%H%M"), verbose: bool = False, **kwargs):
        """
        Initialize the class with various parameters for QAT and NSGA-II algorithm.

        Args:
            data (str): Path to the training dataset.
            dataset_name (str): Name of the dataset.
            pretrained_model (str): Path to the saved model checkpoint.
            arch (str): Model architecture. Defaults to "mobilenetv1".
            act_function (str): Name of activation function to be used in the model. Defaults to "relu".
            pretrained (bool): Whether to use a pretrained model. Defaults to False.
            symmetric_quant (bool): Use symmetric or asymmetric quantization. Defaults to False.
            per_channel_quant (bool): Use per-channel or per-tensor quantization. Defaults to False.
            workers (int): Number of data loading workers. Defaults to 4.
            train_batch (int): Training batch size. Defaults to 256.
            test_batch (int): Testing batch size. Defaults to 512.
            generations (int): Number of generations in NSGA-II. Defaults to 25.
            parent_size (int): Number of parents in NSGA-II population. Defaults to 10.
            offspring_size (int): Number of offsprings in NSGA-II generation. Defaults to 10.
            timeloop_architecture (str): HW architecture name for timeloop. Defaults to "eyeriss".
            timeloop_heuristic (str): Heuristic type for timeloop-mapper. Defaults to "random".
            total_valid (int): Number of total valid mappings in timeloop. Defaults to 0.
            primary_metric (str): Primary metric for timeloop optimization. Defaults to "edp".
            secondary_metric (Optional[str]): Secondary metric for timeloop optimization. Defaults to "".
            cache_directory (str): Directory to store cache files for hardware metrics estimated by Timeloop. Defaults to "timeloop_caches"
            cache_name (str): Name of the cache file to store hardware metrics estimated by Timeloop. Defaults to "cache"
            run_id (str): ID of the current run to distinguish its own cache for writing privileges. Defaults to "1".
            previous_run (Optional[str]): Logs directory of the previous run. None by default.
            qat_epochs (int): Number of epochs for QAT of each individual. Defaults to 10.
            lr (float): Initial learning rate. Defaults to 0.1.
            lr_type (str): Type of learning rate scheduler. Defaults to "cos".
            gamma (float): Multiplier for learning rate adjustment. Defaults to 0.1.
            momentum (float): Momentum for optimizer. Defaults to 0.9.
            weight_decay (float): Weight decay for optimizer. Defaults to 1e-5.
            manual_seed (int): Seed for reproducibility. Defaults to 42.
            deterministic (bool): Enable deterministic mode in CUDA. Defaults to False.
            logs_dir (str): Directory for logging. Defaults to a temporary directory.
            verbose (bool): Enable verbose output. Defaults to False.
        """
        super().__init__(logs_dir=logs_dir, parent_size=parent_size, offspring_size=offspring_size, generations=generations,
                         objectives=[("accuracy", True), (f"total_{primary_metric}", False)], previous_run=previous_run)
        self._data = data
        self._dataset_name = dataset_name
        self._pretrained_model = pretrained_model
        self._arch = arch
        self._act_function = act_function
        self._pretrained = pretrained
        self._symmetric_quant = symmetric_quant
        self._per_channel_quant = per_channel_quant
        self._workers = workers
        self._train_batch = train_batch
        self._test_batch = test_batch
        self._parent_size = parent_size
        self._offspring_size = offspring_size
        self._generations = generations
        self._timeloop_architecture = timeloop_architecture
        self._timeloop_heuristic = timeloop_heuristic
        self._total_valid = total_valid
        self._primary_metric = primary_metric
        self._secondary_metric = secondary_metric
        self._cache_directory = cache_directory
        self._cache_name = cache_name
        self._run_id = run_id
        self._previous_run = previous_run
        self._qat_epochs = qat_epochs
        self._lr = lr
        self._lr_type = lr_type
        self._gamma = gamma
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._manual_seed = manual_seed
        self._deterministic = deterministic
        self._logs_dir = logs_dir
        self._verbose = verbose
        self._kwargs = kwargs
        self._quantizable_layers = self.get_analyzer().get_number_of_quantizable_layers(arch=self._arch)

    def get_configuration(self) -> Dict[str, Any]:
        """
        Retrieves the current configuration settings for the NSGA algorithm and QAT.

        Returns:
            Dict[str, Any]: A dictionary containing the configuration settings.
        """
        return {
            "data": self._data,
            "dataset_name": self._dataset_name,
            "pretrained_model": os.path.abspath(self._pretrained_model),
            "arch": self._arch,
            "act_function": self._act_function,
            "pretrained": self._pretrained,
            "symmetric_quant": self._symmetric_quant,
            "per_channel_quant": self._per_channel_quant,
            "workers": self._workers,
            "train_batch_size": self._train_batch,
            "test_batch_size": self._test_batch,
            "parent_size": self._parent_size,
            "offspring_size": self._offspring_size,
            "generations": self._generations,
            "timeloop_architecture": self._timeloop_architecture,
            "timeloop_heuristic": self._timeloop_heuristic,
            "total_valid": self._total_valid,
            "primary_metric": self._primary_metric,
            "secondary_metric": self._secondary_metric,
            "cache_directory": self._cache_directory,
            "cache_name": self._cache_name,
            "run_id": self._run_id,
            "previous_run": self._previous_run,
            "qat_epochs": self._qat_epochs,
            "lr": self._lr,
            "lr_type": self._lr_type,
            "gamma": self._gamma,
            "momentum": self._momentum,
            "weight_decay": self._weight_decay,
            "manual_seed": self._manual_seed,
            "deterministic": self._deterministic,
            "logs_dir": self._logs_dir,
            "verbose": self._verbose,
            **self._kwargs
        }

    def init_analyzer(self) -> "QATAnalyzer":
        """
        Initializes a new instance of QATAnalyzer with the specified parameters.

        Returns:
            QATAnalyzer: New instance of QATAnalyzer.
        """
        checkpoints_dir_pattern = os.path.join(self._logs_dir, "checkpoints/%s")
        return QATAnalyzer(model_name=self._arch, pretrained_model=self._pretrained_model, act_function=self._act_function,
                           data=self._data, dataset_name=self._dataset_name, train_batch=self._train_batch, test_batch=self._test_batch, workers=self._workers, cache_directory=self._cache_directory, cache_name=self._cache_name, run_id=self._run_id, qat_epochs=self._qat_epochs, lr=self._lr, lr_type=self._lr_type, gamma=self._gamma, momentum=self._momentum, weight_decay=self._weight_decay, manual_seed=self._manual_seed, deterministic=self._deterministic, symmetric_quant=self._symmetric_quant, per_channel_quant=self._per_channel_quant, checkpoints_dir_pattern=checkpoints_dir_pattern, timeloop_heuristic=self._timeloop_heuristic, timeloop_architecture=self._timeloop_architecture, primary_metric=self._primary_metric, secondary_metric=self._secondary_metric, total_valid=self._total_valid, verbose=self._verbose)

    def get_maximal(self) -> Dict[str, float]:
        """
        Determines the maximal values for the objectives, particularly accuracy and the primary metric.

        Returns:
            Dict[str, float]: A dictionary with keys as 'accuracy' and the primary metric, and their maximal values.
        """
        print("Getting maximal values of metrics...")
        max_quant_config = {i: {"Inputs": 8, "Weights": 8} for i in range(self._quantizable_layers)}
        results = list(self.get_analyzer().analyze([{"quant_conf": max_quant_config}], -1))[0]

        return {
            "accuracy": results["accuracy"],
            f"total_{self._primary_metric}": results[f"total_{self._primary_metric}"],
        }

    def get_init_parents(self) -> List[Dict[str, Dict[int, Dict[str, int]]]]:
        """
        Generates an initial set of parents for the NSGA-II algorithm.

        Returns:
            List[Dict[str, Dict[int, Dict[str, int]]]]: A list of parent configurations for the initial population.
        """
        initial_parents = []
        for bitwidth in range(2, 9):
            uniform_quant_config = {i: {"Inputs": bitwidth, "Weights": bitwidth} for i in range(self._quantizable_layers)}
            initial_parents.append({"quant_conf": uniform_quant_config})
        return initial_parents

    def crossover(self, parents: List[Dict[str, Dict[int, Dict[str, int]]]]) -> Dict[str, Dict[int, Dict[str, int]]]:
        """
        Performs crossover and mutation to create new offspring from given parents.

        Args:
            parents: A list of parent configurations.

        Returns:
            Dict[str, Dict[int, Dict[str, int]]]: A single offspring configuration resulting from crossover and mutation.
        """
        child_conf = OrderedDict()
        for li in range(self._quantizable_layers):
            if random.random() < 0.95:  # 95% probability of crossover
                child_conf[li] = random.choice(parents)["quant_conf"][li]
            else:  # 5% chance to use 8-bit quantization
                child_conf[li] = {"Inputs": 8, "Weights": 8}

        if random.random() < 0.1:  # 10% probability of mutation
            li = random.choice(range(self._quantizable_layers))
            # Decide randomly whether to mutate activations or weights
            if random.random() < 0.5:  # 50% chance to mutate activations
                act_bw = random.choice([2, 3, 4, 5, 6, 7, 8])
                child_conf[li]["Inputs"] = act_bw
            else:  # 50% chance to mutate weights
                weight_bw = random.choice([2, 3, 4, 5, 6, 7, 8])
                child_conf[li]["Weights"] = weight_bw

        return {"quant_conf": child_conf}


class QATAnalyzer(NSGAAnalyzer):
    """
    Analyzer for QATNSGA

    This analyzer analyzes individuals by running a few epochs using quantization-aware training (QAT)
    and tracks the best achieved Top-1 accuracy and the value of optimized HW metric.
    """

    # Primary metric key mappings  NOTE: ADD MORE HERE IF YOU WISH
    float_metric_key_mapping = {
        "energy": "Energy [uJ]",
        "edp": "EDP [J*cycle]"
    }

    int_metric_key_mapping = {
        "delay": "Cycles",
        "lla": "LastLevelAccesses",
        "memsize_words": "Weights model memory size [Words]"
    }

    def __init__(self, model_name: str, pretrained_model: str, act_function: str, data: str,
                 dataset_name: str, train_batch: int, test_batch: int, workers: int, cache_directory: str, cache_name: str, run_id: str, qat_epochs: int, lr: float, lr_type: str, gamma: float, momentum: float, weight_decay: float, manual_seed: int, deterministic: bool, symmetric_quant: bool, per_channel_quant: bool, checkpoints_dir_pattern: str, timeloop_heuristic: str, timeloop_architecture: str, primary_metric: str, secondary_metric: str, total_valid: int, verbose: bool):
        """
        Initializes a new QATAnalyzer instance with specific configuration settings.

        Args:
            model_name (str): Name of the model architecture.
            pretrained_model (str): Path to the pretrained model.
            act_function (Callable): Activation function used in the model.
            data (str): Path to the training dataset.
            dataset_name (str): Name of the dataset.
            train_batch (int): Training batch size.
            test_batch (int): Testing batch size.
            workers (int): Number of data loading workers.
            cache_directory (str): Directory to store cache files for hardware metrics estimated by Timeloop.
            cache_name (str): Name of the cache file to store hardware metrics estimated by Timeloop.
            run_id (str): ID of the current run to distinguish its own cache for writing privileges. Defaults to "1".
            qat_epochs (int): Number of epochs for quantization-aware training.
            lr (float): Learning rate for training.
            lr_type (str): Type of learning rate scheduler.
            gamma (float): Learning rate decay factor.
            momentum (float): Momentum factor for the optimizer.
            weight_decay (float): Weight decay factor for the optimizer.
            manual_seed (int): Seed for random number generators for reproducibility.
            deterministic (bool): Flag to enable deterministic behavior in CUDA operations.
            symmetric_quant (bool): Flag to use symmetric quantization.
            per_channel_quant (bool): Flag to use per-channel quantization.
            checkpoints_dir_pattern (Optional[str]): Pattern for the checkpoints directory.
            timeloop_heuristic (str): Heuristic for the timeloop mapper.
            timeloop_architecture (str): Architecture for the timeloop mapper.
            primary_metric (str): Primary metric for model evaluation.
            secondary_metric (Optional[str]): Secondary metric for model evaluation.
            total_valid (int): Total valid mappings to consider in timeloop mapper.
            verbose (bool): Flag to enable verbose logging.
        """
        self._model_name = model_name
        self._pretrained_model = pretrained_model
        self._act_function = act_function
        self._data = data
        self._dataset_name = dataset_name
        self._train_batch_size = train_batch
        self._test_batch_size = test_batch
        self._workers = workers
        self._cache_directory = cache_directory
        self._cache_name = cache_name
        self._run_id = run_id
        self._qat_epochs = qat_epochs
        self._lr = lr
        self._lr_type = lr_type
        self._gamma = gamma
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._manual_seed = manual_seed
        self._deterministic = deterministic
        self._symmetric_quant = symmetric_quant
        self._per_channel_quant = per_channel_quant
        self._checkpoints_dir_pattern = checkpoints_dir_pattern
        self._timeloop_heuristic = timeloop_heuristic
        self._timeloop_architecture = timeloop_architecture
        self._primary_metric = primary_metric
        self._secondary_metric = secondary_metric
        self._total_valid = total_valid
        self._verbose = verbose

        self.ensure_cache_folder()

        self._symmetric_asymmetric = "symmetric" if self._symmetric_quant else "asymmetric"
        self._pertensor_prechannel = "perchannel" if self._per_channel_quant else "pertensor"
        # Current cache file
        i = 0
        while True:
            self.cache_file = "nsga_cache/%s_%s_%s_%d_%d_%.5f_%s_%s_%s_%s_%s_%d.json.gz" % (
                self._model_name, self._act_function, self._dataset_name, self._qat_epochs, self._train_batch_size,  self._lr, self._symmetric_asymmetric, self._pertensor_prechannel, self._primary_metric, self._timeloop_architecture, self._timeloop_heuristic, i)
            if not os.path.isfile(self.cache_file):
                break
            i += 1

        print("Cache file: %s" % self.cache_file)
        self.cache = []
        self.load_cache()

    def _create_namespace_to_call_train(self) -> argparse.Namespace:
        """
        Creates an argparse.Namespace object with values from this QATAnalyzer instance to be used for calling the training script.

        Returns:
            argparse.Namespace: The namespace object with arguments for training.
        """
        # Create a Namespace object with default values
        args = argparse.Namespace()
        # Map the instance variables to the namespace arguments
        # Model options
        args.pretrained = True if self._pretrained_model != "" else False
        args.pretrained_model = self._pretrained_model
        args.arch = self._model_name
        args.act_function = self._act_function
        args.qat = True
        args.symmetric_quant = self._symmetric_quant
        args.per_channel_quant = self._per_channel_quant
        args.quant_setting = "non_uniform"
        # Dataset
        args.data = self._data
        args.dataset_name = self._dataset_name
        args.train_batch = self._train_batch_size
        args.test_batch = self._test_batch_size
        args.workers = self._workers
        # Train options
        args.epochs = self._qat_epochs
        args.lr = self._lr
        args.lr_type = self._lr_type
        args.gamma = self._gamma
        args.momentum = self._momentum
        args.weight_decay = self._weight_decay
        args.manual_seed = self._manual_seed
        args.deterministic = self._deterministic
        args.start_epoch = 0
        args.freeze_epochs = 0
        args.warmup_epoch = 0
        args.resume = False
        # Device options NOTE potential change
        args.gpu_id = "0"
        # Miscs
        args.manual_seed = self._manual_seed
        args.deterministic = self._deterministic
        args.verbose = self._verbose
        args.log = True
        args.wandb = False

        return args

    @staticmethod
    def ensure_cache_folder() -> None:
        """
        Ensures cache folder exists
        """
        os.makedirs("nsga_cache", exist_ok=True)

    def load_cache(self) -> None:
        """
        Loads all already evaluated individuals from cache files to local cache
        """
        for fn in glob.glob("nsga_cache/%s_%s_%s_%d_%d_%.5f_%s_%s_%s_%s_%s_*.json.gz" % (
                self._model_name, self._act_function, self._dataset_name, self._qat_epochs, self._train_batch_size, self._lr, self._symmetric_asymmetric, self._pertensor_prechannel, self._primary_metric, self._timeloop_architecture, self._timeloop_heuristic)):
            print("cache open", fn)

            act = json.load(gzip.open(fn))

            # Find node in cache
            for c in act:
                conf = c["quant_conf"]
                c["quant_conf"] = {int(k): v for k, v in conf.items()}

                # Try to search in cache
                if not any(filter(lambda x: np.array_equal(x["quant_conf"], conf), self.cache)):
                    self.cache.append(c)
                else:
                    cached_entry = list(filter(lambda x: np.array_equal(x["quant_conf"], conf), self.cache))[0]
                    for key in c:
                        if key not in cached_entry:
                            cached_entry[key] = c[key]

        print("Cache loaded %d" % (len(self.cache)))

    def analyze(self, quant_configuration_set: List[Dict[str, Any]], current_gen: int) -> Generator[Dict[str, Any], None, None]:
        """
        Analyze configurations.

        Args:
            quant_configuration_set (List[Dict[str, Any]]): List of configurations for evaluation.
            current_gen (int): Number of current generation being analyzed.

        Yields:
            Dict[str, Any]: The analyzed configuration with accuracy, hardware parameters, and other relevant metrics.
        """
        for node_conf in quant_configuration_set:
            quant_conf = node_conf["quant_conf"]

            start_total = time.time()
            train_time = 0
            timeloop_time = 0
            # Try to search in cache
            cache_sel = self.cache.copy()
            # Filter data
            for i in range(len(quant_conf)):
                cache_sel = filter(lambda x: x["quant_conf"][i] == quant_conf[i], cache_sel)
                cache_sel = list(cache_sel)

            # Get the accuracy
            if len(cache_sel) >= 1:  # Found in cache
                accuracy = cache_sel[0]["accuracy"]
                optimized_metric = cache_sel[0]["optimized_metric"]
                total_energy = cache_sel[0]["total_energy"]
                total_edp = cache_sel[0]["total_edp"]
                total_delay = cache_sel[0]["total_delay"]
                total_lla = cache_sel[0]["total_lla"]
                total_memsize = cache_sel[0]["total_memsize_words"]

                # Add timing information
                total_time = cache_sel[0]["total_time"]
                train_time = cache_sel[0]["train_time"]
                timeloop_time = cache_sel[0]["timeloop_time"]

                total_metric = cache_sel[0][f"total_{self._primary_metric}"]
                print(f"Cache: %s;accuracy=%s;{self._primary_metric}_{self._timeloop_architecture}=%s;" % (
                    str(quant_conf), accuracy, total_metric))
            else:  # Not found in cache
                checkpoints_dir = None
                if self._checkpoints_dir_pattern is not None:
                    # quant_conf_str = '_'.join(map(lambda x: str(x), quant_conf))
                    # checkpoints_dir = self._checkpoints_dir_pattern % (quant_conf_str + "_generation_" + str(current_gen))
                    checkpoints_dir = self._checkpoints_dir_pattern % ("generation_" + str(current_gen))

                qat_args = self._create_namespace_to_call_train()
                qat_args.checkpoint_path = checkpoints_dir
                qat_args.non_uniform_width = quant_conf

                start_train = time.time()
                accuracy = train.main(qat_args)
                train_time = time.time() - start_train

                # Retrieve HW metrics
                mapper_facade = MapperFacade(configs_rel_path="timeloop_utils/timeloop_configs", architecture=self._timeloop_architecture, run_id=self._run_id)
                # Determine num_classes and input_size based on dataset_name
                if self._dataset_name == "imagenet":
                    in_size = "224,224,3"
                    num_classes = 1000
                elif self._dataset_name == "imagenet100":
                    in_size = "224,224,3"
                    num_classes = 100
                elif self._dataset_name == "cifar10":
                    in_size = "32,32,3"
                    num_classes = 10
                else:
                    raise ValueError(f"Unknown dataset_name: {self._dataset_name}. Add support for it here.")

                start_timeloop = time.time()
                if self._pretrained_model != "":
                    hardware_params = mapper_facade.get_hw_params_parse_model(model=self._pretrained_model,
                                                                              arch=self._model_name,
                                                                              batch_size=1,  # search the space for batch size if just 1..
                                                                              bitwidths=self.transform_to_timeloop_quant_config(quant_conf),
                                                                              input_size=in_size,
                                                                              threads=8,
                                                                              heuristic=self._timeloop_heuristic,
                                                                              metrics=(self._primary_metric, self._secondary_metric),
                                                                              total_valid=self._total_valid,
                                                                              cache_dir=self._cache_directory,
                                                                              cache_name=self._cache_name,
                                                                              verbose=self._verbose
                                                                              )
                else:
                    hardware_params = mapper_facade.get_hw_params_create_model(model=self._model_name,
                                                                               num_classes=num_classes,
                                                                               batch_size=1,  # search the space for batch size if just 1..
                                                                               bitwidths=self.transform_to_timeloop_quant_config(quant_conf),
                                                                               input_size=in_size,
                                                                               threads=8,
                                                                               heuristic=self._timeloop_heuristic,
                                                                               metrics=(self._primary_metric, self._secondary_metric),
                                                                               total_valid=self._total_valid,
                                                                               cache_dir=self._cache_directory,
                                                                               cache_name=self._cache_name,
                                                                               verbose=self._verbose
                                                                               )
                timeloop_time = time.time() - start_timeloop
                optimized_metric = self._primary_metric
                # NOTE add more if needed
                total_energy = sum(map(lambda x: float(x[QATAnalyzer.float_metric_key_mapping["energy"]]), hardware_params.values()))
                total_edp = sum(map(lambda x: float(x[QATAnalyzer.float_metric_key_mapping["edp"]]), hardware_params.values()))
                total_delay = sum(map(lambda x: int(x[QATAnalyzer.int_metric_key_mapping["delay"]]), hardware_params.values()))
                total_lla = sum(map(lambda x: int(x[QATAnalyzer.int_metric_key_mapping["lla"]]), hardware_params.values()))
                total_memsize = sum(map(lambda x: int(x[QATAnalyzer.int_metric_key_mapping["memsize_words"]]), hardware_params.values()))
            total_time = time.time() - start_total
            # Create output node
            node = node_conf.copy()
            node["quant_conf"] = quant_conf
            node["accuracy"] = float(accuracy)
            node["optimized_metric"] = optimized_metric
            node["total_energy"] = total_energy
            node["total_edp"] = total_edp
            node["total_delay"] = total_delay
            node["total_lla"] = total_lla
            node["total_memsize_words"] = total_memsize
            # Add timing information
            node["total_time"] = total_time
            node["train_time"] = train_time
            node["timeloop_time"] = timeloop_time

            if len(cache_sel) == 0:  # If the data are not from the cache, cache it
                self.cache.append(node)
                json.dump(self.cache, gzip.open(self.cache_file, "wt", encoding="utf8"), indent=2, cls=JSONEncoder)

            yield node

    def __str__(self):
        return "cache (file: %s, size: %d)" % (self.cache_file, len(self.cache))

    def get_number_of_quantizable_layers(self, arch: str) -> int:
        """
        Get the number of quantizable layers in a given model architecture.

        Args:
            arch (str): The name of the model architecture.

        Returns:
            int: Number of quantizable layers in the model.
        """
        # Instantiate the model based on the provided architecture name
        assert arch in models.__dict__, f"Unknown model architecture: {arch}"
        tmp_model = models.__dict__[arch]()

        # Count quantizable layers (e.g., layers with weights)
        quantizable_layers = 0
        for tmp_module in tmp_model.modules():
            if isinstance(tmp_module, (torch.nn.Conv2d, torch.nn.Linear)):
                quantizable_layers += 1

        self._quantizable_layers = quantizable_layers
        return quantizable_layers

    @staticmethod
    def transform_to_timeloop_quant_config(quant_conf: OrderedDict) -> Dict[int, Dict[str, int]]:
        """
        Transforms the quantization configuration for each layer to include the number of output bits.

        Args:
            quant_conf (OrderedDict): An ordered dictionary with layer numbers as keys and configurations as values.

        Returns:
            Dict[int, Dict[str, int]]: A transformed configuration dictionary where each layer includes the number of bits for inputs, weights, and outputs.
        """
        transformed_config = {}
        layers = list(quant_conf.keys())

        for i, layer in enumerate(layers):
            config = quant_conf[layer]
            if i < len(layers) - 1:  # If not the last layer
                next_layer_input = quant_conf[layers[i + 1]]["Inputs"]
            else:
                next_layer_input = 8  # Default for the last layer

            transformed_config[layer] = {
                "Inputs": config["Inputs"],
                "Weights": config["Weights"],
                "Outputs": next_layer_input
            }
        return transformed_config
