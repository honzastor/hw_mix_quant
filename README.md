# Mixed-Precision Quantization (PyTorch) Framework for Deep Neural Networks

This repository contains the source code for a specialized framework focused on mixed-precision quantization of neural networks. The framework leverages NSGA-II for searching optimal quantization configurations and an extended version of Timeloop to model mixed-precision data types for hardware metrics retrieval.

## File Structure

- `README.md` - README description of the framework
- `requirements.yml` - requirements file for setting up the necessary environment
- `run_nsga.sh` - shell script for running the nsga experiment
- `src/` - the main directory containing all the source code
    - `run_nsga.py` - script interface to initiate and run the nsga process
    - `mapper_facade.py` - interface script for calling timeloop-mapper and caching hardware results
    - `nsga/` - contains scripts related to NSGA-II algorithm
    - `pytorch/` - includes all scripts related to PyTorch models, data management, training, evaluation, and utilities implementation
    - `timeloop_utils/` - contains Timeloop interface configurations and helper scripts

## Project Setup

### Environment Setup

#### Preferred: Conda

```shell
$ conda env create --file requirements.yml
$ conda activate qatizer
```

#### PIP

```shell
$ pip install -r requirements.yml
```

- **IMPORTANT NOTE**: `mapper_facade.py` requires Timeloop+Accelergy infrastructure to be installed. If it is not installed, you will need to use docker or load cached HW metrics – in that case, modifications to `mapper_facade.py` might be necessary. 


### Datasets

- Please download your desired datasets into the  `src/pytorch/data/datasets/` folder. See README instructions there for file structure information.

### Running NSGA-II

- `run_nsga.sh` - Script to run NSGA-II algorithm, tailored for mixed-precision quantization tasks (MODIFY AS YOU WISH).

### Training/Evaluation of Models

- `src/pytorch/run_scripts/` - For (quantization-aware) training and or evaluation of a PyTorch model, you can create scripts to run the `train.py` and `run.py` scripts located in `src/pytorch/`.
