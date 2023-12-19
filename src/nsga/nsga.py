# Author: Miroslav Safar (xsafar23@stud.fit.vutbr.cz)

import abc
import datetime
import glob
import gzip
import json
import os
import random
import re
from shutil import copyfile
from typing import Dict, List, Tuple, Any

from paretoarchive.core import PyBspTreeArchive


class NSGAAnalyzer(abc.ABC):
    """
    Analyzer for NSGA-II to evaluate chromosomes before parents selection
    """

    @abc.abstractmethod
    def analyze(self, configurations: List) -> List:
        """
        This method analyzes a list of chromosomes and returns them with their evaluations.

        Args:
            configurations (List): A list of configurations (chromosomes) that need to be analyzed.

        Returns:
            List: A list of configurations with evaluations added to each one.
        """
        pass


class NSGAState:
    """
    State of the NSGA-II
    """

    def __init__(self, generation: int = None, parents: List = None, offsprings: List = None) -> None:
        """
        Initializes an NSGAState instance.

        Args:
            generation (int): The current generation number. Defaults to None.
            parents (List): The current list of parent chromosomes. Defaults to None.
            offsprings (List): The current list of offspring chromosomes. Defaults to None.
        """
        self._generation = generation
        self._parents = parents
        self._offsprings = offsprings
        self._restored = False

    def get_generation(self) -> int:
        """
        Returns the current generation number.

        Returns:
            int: The number of the current generation.
        """
        return self._generation

    def get_parents(self) -> List:
        """
        Retrieves the current list of parent chromosomes.

        Returns:
            List: The current list of parent chromosomes.
        """
        return self._parents

    def get_offsprings(self) -> List:
        """
        Retrieves the current list of offspring chromosomes.

        Returns:
            List: The current list of offspring chromosomes.
        """
        return self._offsprings

    def save_to(self, logs_dir: str) -> None:
        """
        Saves the current state to a file in the specified directory.

        Args:
            logs_dir (str): The path to the directory where the state file will be saved.
        """
        if self._restored:
            return
        json.dump({"parent": self._parents, "offspring": self._offsprings},
                  gzip.open(logs_dir + "/run.%05d.json.gz" % self._generation, "wt", encoding="utf8"))

    def set_offsprings(self, new_offsprings: List) -> None:
        """
        Updates the list of offsprings in the state.

        Args:
            new_offsprings (List): The new list of offsprings to be set.
        """
        self._offsprings = new_offsprings

    @classmethod
    def restore_from(cls, run_file: str) -> "NSGAState":
        """
        Restores and returns an NSGAState instance from a saved state file.

        Args:
            run_file (str): The path to the file containing the saved state.

        Returns:
            NSGAState: An instance of NSGAState restored from the specified file.
        """
        print("# loading %s" % run_file)
        pr = json.load(gzip.open(run_file))
        # Convert keys for parents
        parents = [
            {**parent, 'quant_conf': {int(k): v for k, v in parent['quant_conf'].items()}}
            for parent in pr["parent"]
        ]
        offsprings = [
            {**offspring, 'quant_conf': {int(k): v for k, v in offspring['quant_conf'].items()}}
            for offspring in pr["offspring"]
        ]

        generation = int(re.match(r".*run\.(\d+).json.gz", run_file).group(1))
        print(f"Restored generation {generation} with {len(parents)} parents and {len(offsprings)} offsprings")
        res_state = cls(generation=generation, parents=parents, offsprings=offsprings)
        res_state._restored = True
        return res_state


class NSGA(abc.ABC):
    """
    Abstract base class for the implementation of the NSGA algorithm.
    """

    def __init__(self, logs_dir: str, parent_size: int = 10, offspring_size: int = 10,
                 generations: int = 25, objectives: List = None, previous_run: str = None) -> None:
        """
        Initializes an NSGA instance with the specified parameters.

        Args:
            logs_dir (str): Path to the log directory.
            parent_size (int): Number of parents. Defaults to 10.
            offspring_size (int): Number of offsprings. Defaults to 10.
            generations (int): Number of generations. Defaults to 25.
            objectives (List): List of objectives for the optimization. Defaults to None.
            previous_run (str): Path to the previous run to restore from. Defaults to None.

        Raises:
            ValueError: If any of the input parameters are invalid.
        """
        if logs_dir is None:
            raise ValueError(f"Logs directory needs to be defined")

        if parent_size < 0:
            raise ValueError(f"Number of parents cannot be negative ({parent_size}<0)")

        if offspring_size < 0:
            raise ValueError(f"Number of offsprings cannot be negative ({offspring_size}<0)")

        if generations < 0:
            raise ValueError(f"Number of generations cannot be negative ({generations}<0)")

        if objectives is None:
            raise ValueError("Objectives need to be defined")

        self.logs_dir = logs_dir
        self.parent_size = parent_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.objectives = objectives

        self.analyzer = None

        self.state = None
        if previous_run is not None:
            self._restore_state(previous_run)

        self.ensure_logs_dir()

        if previous_run is None:
            self._check_if_empty()

    def _restore_state(self, previous_run: str) -> None:
        """
        Restores the state of the NSGA algorithm from a previous run.

        Args:
            previous_run (str): Path to the directory containing the logs of the previous run.
        """
        df = glob.glob(previous_run + "/run.*.gz")
        if self.logs_dir != previous_run:
            for d in df:
                copyfile(d, self.logs_dir + "/" + os.path.basename(d))
                print("# file %s copied" % d)
        df = sorted(df)
        d = df[-1]
        self.state = NSGAState.restore_from(run_file=d)

    def ensure_logs_dir(self) -> None:
        """
        Ensures logs directory exists.
        """
        try:
            os.makedirs(self.logs_dir)
        except FileExistsError:
            pass  # Folder already exists no need to create it

    def _check_if_empty(self) -> None:
        """
        Checks if the logs directory is empty. If not, exits the program.
        """
        files = os.listdir(self.logs_dir)
        if len(files) > 0:
            print("ERROR: Folder for new run is not empty")
            exit(1)

    def _generate_run_information(self) -> None:
        """
        Generates and saves the configuration information for the current run.
        """
        print("Generation configuration information to " + os.path.abspath(self.logs_dir + "/configuration.json"))
        run_info = {
            "start_time": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            "configuration": self.get_configuration()
        }
        with open(self.logs_dir + "/configuration.json", "w") as outfile:
            json.dump(run_info, outfile)

    @abc.abstractmethod
    def get_configuration(self) -> Dict:
        """
        Abstract method to retrieve the configuration of the NSGA algorithm.

        Returns:
            Dict: A dictionary containing the configuration settings.
        """
        pass

    def get_pareto_front(self, values: List) -> List[int]:
        """
        Returns the Pareto front from a set of values based on the defined objectives.

        Args:
            values (List): A list of data points to evaluate.

        Returns:
            List[int]: Indices of data points that form the Pareto front.
        """
        def map_obj_list(value) -> List[float]:
            """
            Maps a data point to a list of objective values, possibly negating some to handle maximization objectives.

            Args:
                value: A data point.

            Returns:
                List[float]: A list of transformed objective values.
            """
            return [value[obj[0]] * (-1 if obj[1] else 1) for obj in self.objectives]

        pareto_ids = PyBspTreeArchive(len(self.objectives)).filter([map_obj_list(x) for x in values], returnIds=True)
        return pareto_ids

    def get_current_state(self) -> NSGAState:
        """
        Retrieves the current state of the NSGA algorithm.

        Returns:
            NSGAState: The current state of the algorithm.
        """
        return self.state

    def run_next_generation(self) -> None:
        """
        Runs one generation of the NSGA algorithm.
        """
        current_state = self.get_current_state()
        g = self.get_current_state().get_generation()
        print("Generation %d" % g)
        print("generation:%d;cache=%s" % (g, str(self.get_analyzer())))
        # initial results from previous data:
        analyzed_offsprings = list(self.get_analyzer().analyze(current_state.get_offsprings()))
        current_state.set_offsprings(analyzed_offsprings)
        current_state.save_to(logs_dir=self.logs_dir)

        # reduce the number of elements
        filtered_results = current_state.get_parents() + current_state.get_offsprings()
        next_parents = []
        missing = self.parent_size - len(next_parents)
        while missing > 0 and len(filtered_results) > 0:
            pareto_ids = self.get_pareto_front(filtered_results)
            pareto = [filtered_results[i] for i in pareto_ids]

            if len(pareto) <= missing:
                next_parents += pareto
            else:  # distance crowding
                next_parents += self.crowding_reduce(pareto, missing)

            for i in reversed(sorted(pareto_ids)):
                filtered_results.pop(i)

            missing = self.parent_size - len(next_parents)

        # generate new candidate solutions
        offsprings = self.generate_offsprings(parents=next_parents)

        # set new state
        self.state = NSGAState(generation=g + 1, parents=next_parents, offsprings=offsprings)

    def run(self) -> None:
        """
        Runs the NSGA algorithm for the specified number of generations.
        """
        if self.state is None:
            self._generate_run_information()

            parents = self.get_init_parents()
            next_parents = list(self.get_analyzer().analyze(parents))
            self.state = NSGAState(generation=0, parents=next_parents, offsprings=[])

        while self.get_current_state().get_generation() <= self.generations:
            self.run_next_generation()

    def generate_offsprings(self, parents: List) -> List:
        """
        Generate offsprings from parents using crossover and mutation.

        Args:
            parents (List): List of parent chromosomes.

        Returns:
            List: A list of generated offspring chromosomes.
        """
        offsprings = []
        for i in range(0, self.offspring_size):
            # select two random parents
            selected_parents = random.sample(parents, k=2)
            # generate offspring from these two parents
            offsprings.append(self.crossover(selected_parents))

        return offsprings

    def crowding_distance(self, pareto_front: List) -> List[Tuple]:
        """
        Calculates crowding distance for each individual in the Pareto front.

        Args:
            pareto_front (List): Set of individuals on the Pareto front.

        Returns:
            List[Tuple]: A list of pairs (individual, crowding distance).
        """
        park = list(enumerate(pareto_front))
        distance = [0 for _ in range(len(pareto_front))]
        for obj, asc in self.objectives:
            sorted_values = sorted(park, key=lambda x: x[1][obj])
            min_val, max_val = 0, self.get_maximal()[obj]
            distance[sorted_values[0][0]] = float("inf")
            distance[sorted_values[-1][0]] = float("inf")

            for i in range(1, len(sorted_values) - 1):
                distance[sorted_values[i][0]] += abs(sorted_values[i - 1][1][obj] - sorted_values[i + 1][1][obj]) / (
                        max_val - min_val)
        return zip(pareto_front, distance)

    def crowding_reduce(self, pareto_front: List, number: int) -> List:
        """
        Reduces the Pareto front to a specified number (see `number`) of individuals based on crowding distance.

        Args:
            pareto_front (List): Set of individuals on the Pareto front.
            number (int): The desired number of individuals to retain.

        Returns:
            List: A reduced set of individuals from the Pareto front.
        """
        pareto_front = pareto_front
        while len(pareto_front) > number:
            vals = self.crowding_distance(pareto_front)
            vals = sorted(vals, key=lambda x: -x[1])  # sort by distance descending

            pareto_front = [x[0] for x in vals[:-1]]  # remove last
        return pareto_front

    def get_analyzer(self) -> NSGAAnalyzer:
        """
        Retrieves the NSGA analyzer. Initializes it if it has not been created.

        Returns:
            NSGAAnalyzer: The analyzer used for NSGA.
        """
        if self.analyzer is None:
            self.analyzer = self.init_analyzer()
        return self.analyzer

    @abc.abstractmethod
    def crossover(self, parents: List) -> Any:
        """
        Performs the crossover operation to generate a child from given parents.

        Args:
            parents (List): The parent chromosomes used for crossover.

        Returns:
            The resulting child chromosome after crossover.
        """
        pass

    @abc.abstractmethod
    def get_maximal(self) -> List:
        """
        Retrieves maximal values for the optimization objectives.

        Returns:
            List: Maximal values for each objective.
        """
        pass

    @abc.abstractmethod
    def init_analyzer(self) -> NSGAAnalyzer:
        """
        Initializes and returns the NSGA analyzer.

        Returns:
            NSGAAnalyzer: The initialized analyzer for NSGA.
        """
        pass

    @abc.abstractmethod
    def get_init_parents(self) -> List:
        """
        Generates and returns the initial set of parent chromosomes for the first population.

        Returns:
            List: The initial set of parent chromosomes.
        """
        pass
