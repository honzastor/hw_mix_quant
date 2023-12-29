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
from typing import Optional, Dict, List, Any, Tuple, Generator, OrderedDict
import concurrent
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock, Queue, Manager

from .nsga_qat import QATNSGA, QATAnalyzer, transform_to_timeloop_quant_config, create_namespace_to_call_train
from pytorch import train
import pytorch.models as custom_models
from mapper_facade import MapperFacade


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
        manager = Manager()
        cuda_devices = torch.cuda.device_count()
        self._queue = manager.Queue()
        for device_id in range(cuda_devices):
            self._queue.put(device_id)
        self._timeloop_pool = ProcessPoolExecutor(max_workers=1)  # Only 1 Timeloop can run at a time
        self._gpu_pool = ProcessPoolExecutor(max_workers=cuda_devices)
        self._lock = Lock()
        self._qat_evaluation_lock = manager.Lock()  # Limit the number of concurrent after-qat converted model evaluations on CPU to just 1

    @property
    def queue(self) -> Queue:
        """
        A property that returns a queue of available GPU devices.

        Returns:
            Queue: A queue with available GPU device identifiers.
        """
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

        # Using ProcessPoolExecutor for parallel processing (inside get_eval_of_configs())
        num_gpus = torch.cuda.device_count()
        if self._verbose:
            print(f"Needs eval: {needs_eval} on {num_gpus} GPUs.")
        results = self.get_eval_of_configs(needs_eval, current_gen)
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

    def get_eval_of_configs(self, quant_configs: List[Dict[str, Any]], current_gen: int) -> List[Tuple[float, Dict[str, Any], Dict[str, float]]]:
        """
        Evaluate configurations an available GPUs.

        This includes training the model and running Timeloop for hardware parameter estimation.

        Args:
            quant_configs (List[Dict[str, Any]]): Configurations to be evaluated.
            current_gen (int): Number of current generation being analyzed.

        Returns:
            List[Tuple[float, Dict[str, Any], Dict[str, float]]]: A list of tuples containing the accuracy, hardware parameters, and timing metrics.
        """
        collected_params = [f"total_edp", f"total_energy", f"total_delay", f"total_lla", f"total_memsize_words"]  # NOTE add more if needed
        train_futures = {}
        timeloop_futures = {}
        results = []

        # Iterate each quantization configuration and submit tasks for evaluation or retrieve precached metrics
        for quant_config in quant_configs:
            quant_conf_str = json.dumps(quant_config['quant_conf'], sort_keys=True)
            start_total = time.time()
            # If accuracy is not precomputed, submit training task
            if "accuracy" not in quant_config:
                # Submit training task on available GPU
                train_future = self.submit_train_task(quant_config, current_gen)
                train_futures[quant_conf_str] = train_future

            # If hardware params are not precomputed, submit Timeloop task
            if not all(param in quant_config for param in collected_params):
                # Initialize MapperFacade for running Timeloop or reading cached metrics
                mapper_facade = MapperFacade(configs_rel_path="timeloop_utils/timeloop_configs", architecture=self._timeloop_architecture, run_id=str(self._run_id))

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
                timeloop_future = self.submit_timeloop_task(mapper_facade, quant_config['quant_conf'], in_size, num_classes)
                timeloop_futures[quant_conf_str] = timeloop_future

        # Wait for training futures and collect results
        for quant_conf_str, future in train_futures.items():
            accuracy, train_time = future.result()
            quant_conf = json.loads(quant_conf_str)  # Deserializing back to dictionary
            quant_conf = {int(k): v for k, v in quant_conf.items()}

            # Retrieve the corresponding quant_config
            corresponding_quant_config = next((setting for setting in quant_configs if setting['quant_conf'] == quant_conf), None)
            # Update the quant_config with training results
            corresponding_quant_config["accuracy"] = accuracy
            corresponding_quant_config["train_time"] = train_time

        # Wait for Timeloop futures and collect results
        for quant_conf_str, future in timeloop_futures.items():
            hardware_params, timeloop_time = future.result()
            quant_conf = json.loads(quant_conf_str)  # Deserializing back to dictionary
            quant_conf = {int(k): v for k, v in quant_conf.items()}

            # Retrieve the corresponding quant_config
            corresponding_quant_config = next((setting for setting in quant_configs if setting['quant_conf'] == quant_conf), None)

            # Update the quant_config with the Timeloop results
            # NOTE add more if needed
            corresponding_quant_config["total_energy"] = sum(map(lambda x: float(x[QATAnalyzer.float_metric_key_mapping["energy"]]), hardware_params.values()))
            corresponding_quant_config["total_edp"] = sum(map(lambda x: float(x[QATAnalyzer.float_metric_key_mapping["edp"]]), hardware_params.values()))
            corresponding_quant_config["total_delay"] = sum(map(lambda x: int(x[QATAnalyzer.int_metric_key_mapping["delay"]]), hardware_params.values()))
            corresponding_quant_config["total_lla"] = sum(map(lambda x: int(x[QATAnalyzer.int_metric_key_mapping["lla"]]), hardware_params.values()))
            corresponding_quant_config["total_memsize_words"] = sum(map(lambda x: int(x[QATAnalyzer.int_metric_key_mapping["memsize_words"]]), hardware_params.values()))
            corresponding_quant_config["timeloop_time"] = timeloop_time

        # Combine and return all results
        for quant_config in quant_configs:
            # All data should be present in quant_config by now
            accuracy = quant_config["accuracy"]
            total_hw_metrics = {param: quant_config[param] for param in collected_params}
            train_time = quant_config["train_time"]
            timeloop_time = quant_config["timeloop_time"]
            total_time = quant_config["total_time"] if "total_time" in quant_config else time.time() - start_total
            results.append((accuracy, total_hw_metrics, {"total_time": total_time, "train_time": train_time, "timeloop_time": timeloop_time}))
        return results

    def submit_train_task(self, quant_config: Dict[str, Any], current_gen: int):
        """
        Submits a QAT training task to the ProcessPoolExecutor for asynchronous execution.

        Args:
            quant_config (Dict[str, Any]): The quantization configuration for which to run QAT.

        Returns:
            concurrent.futures.Future: A Future object representing the asynchronous execution of QAT.
        """
        return self._gpu_pool.submit(train_task, quant_config, current_gen, self._checkpoints_dir_pattern, self._pretrained_model, self._model_name, self._act_function, self._symmetric_quant, self._per_channel_quant, self._data, self._dataset_name, self._train_batch_size, self._test_batch_size, self._workers, self._qat_epochs, self._lr, self._lr_type, self._gamma, self._momentum, self._weight_decay, self._manual_seed, self._deterministic, self._qat_evaluation_lock, self._verbose, self.queue)

    def submit_timeloop_task(self, mapper_facade: MapperFacade, quant_config: Dict[str, Any], in_size: str, num_classes: int) -> concurrent.futures.Future:
        """
        Submits a Timeloop MapperFacade task to the ProcessPoolExecutor for asynchronous execution.

        Args:
            mapper_facade (MapperFacade): An instance of the MapperFacade class used to run Timeloop mapper and retrieve hardware metrics.
            quant_config (Dict[str, Any]): The quantization configuration for which to run Timeloop.
            in_size (str): The input size of the model.
            num_classes (int): The number of classes in the model's output.

        Returns:
            concurrent.futures.Future: A Future object representing the asynchronous execution of Timeloop.
        """
        return self._timeloop_pool.submit(timeloop_task, mapper_facade, self._pretrained_model, self._model_name, quant_config, in_size, num_classes, self._timeloop_heuristic, self._primary_metric, self._secondary_metric, self._total_valid, self._cache_directory, self._cache_name, self._verbose)


def train_task(quant_config: Dict[str, Any], current_gen: int, checkpoints_dir_pattern: str, pretrained_model: str, model_name: str, act_function: str, symmetric_quant: bool, per_channel_quant: bool, data: str, dataset_name: str, train_batch_size: int, test_batch_size: int, workers: int, qat_epochs: int, lr: float, lr_type: str, gamma: float, momentum: float, weight_decay: float, manual_seed: int, deterministic: bool, qat_evaluation_lock: Optional[object], verbose: bool, gpu_pool_queue: Queue) -> Tuple[float, float]:
    """
    Executes a QAT training task on a specified GPU.

    Args:
        quant_config: The quantization configuration to be used for training.
        current_gen: The current generation number in the NSGA-II process.
        checkpoints_dir_pattern: The pattern for generating the checkpoints directory.
        pretrained_model: Path to the pretrained model file.
        model_name: Name of the model architecture to be used.
        act_function: Activation function used in the model.
        symmetric_quant: Flag to indicate if symmetric quantization is used.
        per_channel_quant: Flag to indicate if per-channel quantization is used.
        data: Path to the training dataset.
        dataset_name: Name of the dataset.
        train_batch_size: Batch size for training.
        test_batch_size: Batch size for testing.
        workers: Number of worker threads for data loading.
        qat_epochs: Number of epochs for quantization-aware training.
        lr: Learning rate for training.
        lr_type: Type of learning rate scheduler.
        gamma: Factor for learning rate decay.
        momentum: Momentum factor for the optimizer.
        weight_decay: Weight decay factor for the optimizer.
        manual_seed: Seed for random number generators.
        deterministic: Flag for deterministic behavior in CUDA operations.
        qat_evaluation_lock: Lock for synchronizing QAT evaluations. Optional.
        verbose: Flag for verbose logging.
        gpu_pool_queue: Queue of available GPU devices.

    Returns:
        Tuple[float, float]: A tuple containing the accuracy and the training time.
    """
    device_id = gpu_pool_queue.get()
    try:
        if verbose:
            print(f"Evaluating quant config {quant_config['quant_conf']} on GPU {device_id}")
        # Set the current device for PyTorch
        torch.cuda.set_device(device_id)

        checkpoints_dir = None
        if checkpoints_dir_pattern is not None:
            # quant_conf_str = '_'.join(map(lambda x: str(x), quant_conf))
            # checkpoints_dir = checkpoints_dir_pattern % (quant_conf_str + "_generation_" + str(current_gen))
            checkpoints_dir = checkpoints_dir_pattern % ("generation_" + str(current_gen))

        # Prepare and run training on the specified GPU
        qat_args = create_namespace_to_call_train(gpu_id=device_id, pretrained_model=pretrained_model, model_name=model_name, act_function=act_function, symmetric_quant=symmetric_quant, per_channel_quant=per_channel_quant, data=data, dataset_name=dataset_name, train_batch_size=train_batch_size, test_batch_size=test_batch_size, workers=workers, qat_epochs=qat_epochs, lr=lr, lr_type=lr_type, gamma=gamma, momentum=momentum, weight_decay=weight_decay, manual_seed=manual_seed, deterministic=deterministic, qat_evaluation_lock=qat_evaluation_lock, verbose=verbose)
        qat_args.checkpoint_path = checkpoints_dir
        qat_args.non_uniform_width = quant_config['quant_conf']

        # Run training on CUDA
        start_train = time.time()
        accuracy = train.main(qat_args)
        train_time = time.time() - start_train
    finally:
        gpu_pool_queue.put(device_id)
    return accuracy, train_time


def timeloop_task(mapper_facade: MapperFacade, pretrained_model: str, model_name: str, quant_config: Dict[str, Any], in_size: str, num_classes: int, timeloop_heuristic: str, primary_metric: str, secondary_metric: str, total_valid: int, cache_directory: str, cache_name: str, verbose: bool) -> Tuple[Dict[str, Any], float]:
    """
    Executes a Timeloop task to estimate hardware parameters for a given quantization configuration.

    Args:
        mapper_facade: An instance of MapperFacade used to interface with the Timeloop tool.
        pretrained_model: Path to the pretrained model file, if available.
        model_name: Name of the model architecture.
        quant_config: The quantization configuration for which hardware parameters are to be estimated.
        in_size: The input size of the model.
        num_classes: The number of classes in the model's output.
        timeloop_heuristic: Heuristic for the Timeloop mapper.
        primary_metric: Primary metric for model evaluation.
        secondary_metric: Secondary metric for model evaluation.
        total_valid: Total valid mappings to consider in the Timeloop mapper.
        cache_directory: Directory for storing cache files.
        cache_name: Name of the cache file.
        verbose: Flag for verbose logging.

    Returns:
        Tuple[Dict[str, Any], float]: A tuple containing the hardware parameters and the time taken for Timeloop estimation.
    """
    start_timeloop = time.time()
    if pretrained_model != "":
        metrics = mapper_facade.get_hw_params_parse_model(
            model=pretrained_model,
            arch=model_name,
            batch_size=1,
            bitwidths=transform_to_timeloop_quant_config(quant_config),
            input_size=in_size,
            threads=8,
            heuristic=timeloop_heuristic,
            metrics=(primary_metric, secondary_metric),
            total_valid=total_valid,
            cache_dir=cache_directory,
            cache_name=cache_name,
            verbose=verbose)
    else:
        metrics = mapper_facade.get_hw_params_create_model(
            model=model_name,
            num_classes=num_classes,
            batch_size=1,
            bitwidths=transform_to_timeloop_quant_config(quant_config),
            input_size=in_size,
            threads=8,
            heuristic=timeloop_heuristic,
            metrics=(primary_metric, secondary_metric),
            total_valid=total_valid,
            cache_dir=cache_directory,
            cache_name=cache_name,
            verbose=verbose)
    timeloop_time = time.time() - start_timeloop
    return metrics, timeloop_time
