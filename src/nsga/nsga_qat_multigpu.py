# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz), Jan Klhufek (iklhufek@fit.vut.cz)

import os
import json
import gzip
import argparse
import time
import datetime
import copy
import torch
import torchvision.models as models
from typing import Optional, Dict, List, Any, Tuple, Generator
import concurrent
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from threading import Lock
from queue import Queue

from .nsga import NSGAAnalyzer
from .nsga_qat import QATNSGA, QATAnalyzer
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


class MultiGPUQATNSGA(QATNSGA):
    """
    Extension of QATNSGA for multi-GPU systems.
    This class uses the MultiGPU QAT Analyzer for the evaluation of individuals across multiple GPUs.
    """

    def __init__(self, data: str, dataset_name: str, pretrained_model: str, arch: str = "mobilenetv1", pretrained: bool = False,
                 act_function: str = "relu", symmetric_quant: bool = False, per_channel_quant: bool = False, workers: int = 4, train_batch: int = 256, test_batch: int = 512, generations: int = 25, parent_size: int = 10, offspring_size: int = 10, timeloop_architecture: str = "eyeriss", timeloop_heuristic: str = "random", total_valid: int = 0, primary_metric: str = "edp", secondary_metric: Optional[str] = "", cache_directory: str = "timeloop_caches", cache_name: str = "cache", run_id: str = "1", previous_run: Optional[str] = None, qat_epochs: int = 10, lr: float = 0.1, lr_type: str = "cos", gamma: float = 0.1, momentum: float = 0.9, weight_decay: float = 1e-5, manual_seed: int = 42, deterministic: bool = False, logs_dir: str = "/tmp/run" + datetime.datetime.now().strftime("-%Y%m%d-%H%M"), verbose: bool = False, **kwargs):
        """
        Initializes the MultiGPUQATNSGA class with configurations for multi-GPU QAT.

        Inherits all parameters from QATNSGA. See QATNSGA documentation for parameter details.
        """
        super().__init__(data, dataset_name, pretrained_model, arch, pretrained, act_function, symmetric_quant,
                         per_channel_quant, workers, train_batch, test_batch, generations, parent_size, offspring_size, timeloop_architecture, timeloop_heuristic, total_valid, primary_metric, secondary_metric, cache_directory, cache_name, run_id, previous_run, qat_epochs, lr, lr_type, gamma, momentum, weight_decay, manual_seed, deterministic, logs_dir, verbose, **kwargs)

    def init_analyzer(self) -> "MultiGPUQATAnalyzer":
        """
        Initializes a new instance of MultiGPUQATAnalyzer with the specified parameters.

        Returns:
            MultiGPUQATAnalyzer: New instance of MultiGPUQATAnalyzer.
        """
        checkpoints_dir_pattern = os.path.join(self.logs_dir, "checkpoints/%s")
        return MultiGPUQATAnalyzer(model_name=self._arch, pretrained_model=self._pretrained_model,
                                   act_function=self._act_function, data=self._data, dataset_name=self._dataset_name, train_batch=self._train_batch, test_batch=self._test_batch, workers=self._workers, cache_directory=self._cache_directory, cache_name=self._cache_name, run_id=self._run_id, qat_epochs=self._qat_epochs, lr=self._lr, lr_type=self._lr_type, gamma=self._gamma, momentum=self._momentum, weight_decay=self._weight_decay, manual_seed=self._manual_seed, deterministic=self._deterministic, symmetric_quant=self._symmetric_quant, per_channel_quant=self._per_channel_quant, checkpoints_dir_pattern=checkpoints_dir_pattern, timeloop_heuristic=self._timeloop_heuristic, timeloop_architecture=self._timeloop_architecture, primary_metric=self._primary_metric, secondary_metric=self._secondary_metric, total_valid=self._total_valid, verbose=self._verbose)


class MultiGPUQATAnalyzer(QATAnalyzer):
    """
    An extension of the QATAnalyzer to support multi-GPU systems.

    The MultiGPUQATAnalyzer facilitates the simultaneous analysis and evaluation of multiple model configurations across several GPUs. This approach allows for parallel quantization-aware training (QAT) sessions, significantly speeding up the process. The analyzer computes the top-1 accuracy and hardware efficiency metrics for each configuration, leveraging the multi-GPU setup for efficient computation.
    """
    def __init__(self, model_name: str, pretrained_model: str, act_function: str, data: str,
                 dataset_name: str, train_batch: int, test_batch: int, workers: int, cache_directory: str, cache_name: str, run_id: str, qat_epochs: int, lr: float, lr_type: str, gamma: float, momentum: float, weight_decay: float, manual_seed: int, deterministic: bool, symmetric_quant: bool, per_channel_quant: bool, checkpoints_dir_pattern: str, timeloop_heuristic: str, timeloop_architecture: str, primary_metric: str, secondary_metric: str, total_valid: int, verbose: bool):
        """
        Initializes the MultiGPUQATAnalyzer class with configurations for multi-GPU QAT.

        Inherits all parameters from QATAnalyzer. See QATAnalyzer documentation for parameter details.
        """
        super().__init__(model_name, pretrained_model, act_function, data, dataset_name, train_batch, test_batch, workers,
                         cache_directory, cache_name, run_id, qat_epochs, lr, lr_type, gamma, momentum, weight_decay, manual_seed, deterministic, symmetric_quant, per_channel_quant, checkpoints_dir_pattern, timeloop_heuristic, timeloop_architecture, primary_metric, secondary_metric, total_valid, verbose)
        self._queue = None
        self._timeloop_pool = ThreadPoolExecutor(max_workers=1)  # Only 1 Timeloop can run at a time
        self._lock = Lock()
        self._qat_evaluation_lock = multiprocessing.Lock()  # Limit the number of concurrent after-qat converted model evaluations on CPU to just 1

    @property
    def queue(self) -> Queue:
        """
        A property that returns a queue of available GPU devices.

        If the queue is not initialized, it creates a new queue and populates it
        with the identifiers of the available CUDA devices.

        Returns:
            Queue: A queue with available GPU device identifiers.
        """
        if self._queue is None:
            self._queue = Queue()
            cuda_devices = torch.cuda.device_count()
            for device_id in range(cuda_devices):
                self._queue.put(device_id)
        return self._queue

    def update_cache(self, node_to_update: Dict[str, Any]) -> None:
        """
        Updates the cache in place with new information from a given node.

        If the node's configuration is not already in the cache, it is added.
        Otherwise, updates the existing cache entry with any new information from the node.

        Args:
            node_to_update (Dict[str, Any]): The node containing the configuration and evaluation results.
        """
        with self._lock:
            self.cache.clear()
            self.load_cache()

            quant_conf = node_to_update["quant_conf"]

            if not any(x["quant_conf"] == quant_conf for x in self.cache):
                self.cache.append(node_to_update)
            else:
                cached_entry = list(filter(lambda x: x["quant_conf"] == quant_conf, self.cache))[0]
                for key in node_to_update:
                    if key not in cached_entry:
                        if self._verbose:
                            print(f"Updating {key} in {cached_entry['quant_conf']} to {node_to_update[key]}")
                        cached_entry[key] = node_to_update[key]
                    else:
                        if self._verbose:
                            print(f"{key} is already in {cached_entry['quant_conf']}")

            json.dump(self.cache, gzip.open(self.cache_file, "wt", encoding="utf8"))

    def read_from_cache(self, quant_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieves a node from the cache based on a given quantization configuration.

        Args:
            quant_config (Dict[str, Any]): The quantization configuration to search for in the cache.

        Returns:
            Optional[Dict[str, Any]]: The cached node corresponding to the given quantization configuration, if found, otherwise None.
        """
        with self._lock:
            return next((x for x in self.cache if x["quant_conf"] == quant_config), None)

    def _create_namespace_to_call_train(self, gpu_id: str) -> argparse.Namespace:
        """
        Creates an argparse.Namespace object with values from this QATAnalyzer instance to be used for calling the training script.

        Args:
            gpu_id (str): CUDA ID to use.
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
        args.qat_evaluation_lock = self._qat_evaluation_lock
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
        # Device options
        args.gpu_id = gpu_id
        # Miscs
        args.manual_seed = self._manual_seed
        args.deterministic = self._deterministic
        args.verbose = self._verbose
        args.log = True
        args.wandb = False

        return args

    def analyze(self, quant_configuration_set: List[Dict[str, Any]], current_gen: int) -> Generator[Dict[str, Any], None, None]:
        """
        Analyze configurations on multiple GPUs.

        This method evaluates the configurations using all available GPUs and returns the results.

        Args:
            quant_configuration_set (List[Dict[str, Any]]): List of configurations for evaluation.
            current_gen (int): Number of current generation being analyzed.

        Yields:
            Dict[str, Any]: The analyzed configuration with accuracy, hardware parameters, and other relevant metrics.
        """
        needs_eval = []

        for node_conf in quant_configuration_set:
            quant_conf = node_conf["quant_conf"]

            entry = self.read_from_cache(quant_conf)

            if entry is None:
                entry = {"quant_conf": quant_conf}

            needs_eval.append(entry)

        # Using ThreadPoolExecutor for parallel processing
        num_gpus = torch.cuda.device_count()
        if self._verbose:
            print(f"Needs eval: {needs_eval} on {num_gpus} GPUs.")
        with ThreadPoolExecutor(max_workers=num_gpus) as pool:
            results = list(pool.map(lambda qc: self.get_eval_of_config(qc, current_gen), needs_eval))
        if self._verbose:
            print("Eval done")

        # Update cache and prepare the final result
        for i, quant_conf in enumerate(needs_eval):
            node = {
                "quant_conf": quant_conf["quant_conf"],
                "accuracy": float(results[i][0]),
                "optimized_metric": self._primary_metric,
                **results[i][1],
                **results[i][2],
            }
            self.update_cache(node)

        # Generating final output
        for node_conf in quant_configuration_set:
            quant_conf = node_conf["quant_conf"]
            cached_entry = self.read_from_cache(quant_conf)

            # Get the stats
            if cached_entry is not None:
                accuracy = cached_entry["accuracy"]
                optimized_metric = cached_entry["optimized_metric"]
                total_edp = cached_entry["total_edp"]
                total_delay = cached_entry["total_delay"]
                total_energy = cached_entry["total_energy"]
                total_lla = cached_entry["total_lla"]
                total_memsize = cached_entry["total_memsize_words"]

                total_time = cached_entry["total_time"]
                train_time = cached_entry["train_time"]
                timeloop_time = cached_entry["timeloop_time"]

                total_metric = cached_entry[f"total_{self._primary_metric}"]
                print(f"Cache: %s;accuracy=%s;{self._primary_metric}_{self._timeloop_architecture}=%s;" % (
                      str(quant_conf), accuracy, total_metric))
            else:
                raise ValueError(f"Configuration {quant_conf} not found in cache! This should never happen!")

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

            yield node

    def get_eval_of_config(self, quant_config: Dict[str, Any], current_gen: int) -> Tuple[float, Dict[str, Any], Dict[str, float]]:
        """
        Evaluate a configuration on an available GPU.

        This includes training the model and running Timeloop for hardware parameter estimation.

        Args:
            quant_config (Dict[str, Any]): Configuration to be evaluated.
            current_gen (int): Number of current generation being analyzed.

        Returns:
            Tuple[float, Dict[str, Any], Dict[str, float]]: A tuple containing the accuracy, hardware parameters, and timing metrics.
        """
        # Retrieve an available GPU from the queue
        device_id = self.queue.get()
        try:
            start_total = time.time()
            if self._verbose:
                print(f"Evaluating quant config {quant_config['quant_conf']} on GPU {device_id}")
            # Set the current device for PyTorch
            torch.cuda.set_device(device_id)

            # Check if accuracy is already in quant_config
            if "accuracy" in quant_config:
                accuracy = quant_config["accuracy"]
                train_time = quant_config["train_time"]
            else:
                checkpoints_dir = None
                if self._checkpoints_dir_pattern is not None:
                    # quant_conf_str = '_'.join(map(lambda x: str(x), quant_conf))
                    # checkpoints_dir = self._checkpoints_dir_pattern % (quant_conf_str + "_generation_" + str(current_gen))
                    checkpoints_dir = self._checkpoints_dir_pattern % ("generation_" + str(current_gen))

                # Prepare and run training on the specified GPU
                qat_args = self._create_namespace_to_call_train(gpu_id=device_id)
                qat_args.checkpoint_path = checkpoints_dir
                qat_args.non_uniform_width = quant_config['quant_conf']

                # Run training on CUDA
                start_train = time.time()
                accuracy = train.main(qat_args)
                train_time = time.time() - start_train

            # Check if hardware params are already in quant_config
            collected_params = [f"total_edp", f"total_energy", f"total_delay", f"total_lla", f"total_memsize_words"]  # NOTE add more if needed
            if all(x in quant_config for x in collected_params):
                total_hw_metrics = {param: quant_config[param] for param in collected_params}
                timeloop_time = quant_config["timeloop_time"]
            else:
                # Initialize MapperFacade for running Timeloop or reading cached metrics
                mapper_facade = MapperFacade(configs_rel_path="timeloop_utils/timeloop_configs", architecture=self._timeloop_architecture, run_id=str(self._run_id) + "_" + str(device_id))

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

                # Run Timeloop on CPU
                start_timeloop = time.time()
                timeloop_run = self.submit_timeloop_task(mapper_facade, quant_config['quant_conf'], in_size, num_classes)
                hardware_params = timeloop_run.result()  # Wait for Timeloop task to complete
                timeloop_time = time.time() - start_timeloop

                # NOTE add more if needed
                total_energy = sum(map(lambda x: float(x[QATAnalyzer.float_metric_key_mapping["energy"]]), hardware_params.values()))
                total_edp = sum(map(lambda x: float(x[QATAnalyzer.float_metric_key_mapping["edp"]]), hardware_params.values()))
                total_delay = sum(map(lambda x: int(x[QATAnalyzer.int_metric_key_mapping["delay"]]), hardware_params.values()))
                total_lla = sum(map(lambda x: int(x[QATAnalyzer.int_metric_key_mapping["lla"]]), hardware_params.values()))
                total_memsize = sum(map(lambda x: int(x[QATAnalyzer.int_metric_key_mapping["memsize_words"]]), hardware_params.values()))

                # Add calculated metrics to the dictionary
                total_hw_metrics = {
                    "total_energy": total_energy,
                    "total_edp": total_edp,
                    "total_delay": total_delay,
                    "total_lla": total_lla,
                    "total_memsize_words": total_memsize
                }

            # Check if total_time is already in quant_config (it should)
            if "total_time" in quant_config:
                total_time = quant_config["total_time"]
            else:
                total_time = time.time() - start_total

            times = {
                "total_time": total_time,
                "train_time": train_time,
                "timeloop_time": timeloop_time
            }

            return accuracy, total_hw_metrics, times
        finally:
            # Release the GPU back to the queue
            self.queue.put(device_id)

    def submit_timeloop_task(self, mapper_facade: MapperFacade, quant_config: Dict[str, Any], in_size: str, num_classes: int) -> concurrent.futures.Future:
        """
        Submits a Timeloop MapperFacade task to the ThreadPoolExecutor for asynchronous execution.

        Args:
            mapper_facade (MapperFacade): An instance of the MapperFacade class used to run Timeloop mapper and retrieve hardware metrics.
            quant_config (Dict[str, Any]): The quantization configuration for which to run Timeloop.
            in_size (str): The input size of the model.
            num_classes (int): The number of classes in the model's output.

        Returns:
            concurrent.futures.Future: A Future object representing the asynchronous execution of Timeloop.
        """
        def task():
            if self._pretrained_model != "":
                return mapper_facade.get_hw_params_parse_model(
                    model=self._pretrained_model,
                    arch=self._model_name,
                    batch_size=1,
                    bitwidths=self.transform_to_timeloop_quant_config(quant_config),
                    input_size=in_size,
                    threads=8,
                    heuristic=self._timeloop_heuristic,
                    metrics=(self._primary_metric, self._secondary_metric),
                    total_valid=self._total_valid,
                    cache_dir=self._cache_directory,
                    cache_name=self._cache_name,
                    verbose=self._verbose)
            else:
                return mapper_facade.get_hw_params_create_model(
                    model=self._model_name,
                    num_classes=num_classes,
                    batch_size=1,
                    bitwidths=self.transform_to_timeloop_quant_config(quant_config),
                    input_size=in_size,
                    threads=8,
                    heuristic=self._timeloop_heuristic,
                    metrics=(self._primary_metric, self._secondary_metric),
                    total_valid=self._total_valid,
                    cache_dir=self._cache_directory,
                    cache_name=self._cache_name,
                    verbose=self._verbose)
        return self._timeloop_pool.submit(task)
