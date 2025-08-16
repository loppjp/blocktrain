---
title: "BlockTrain"
author: Jacob Lopp
state: discussion
output:
    html_document:
        numbered_sections: true
---

# BlockTrain

Component building blocks for supervised ML training

# Quickstart

To interact with the training pipeline codebase and supporting artifacts, please use the Developer installation instructions to 

### Developer Installation

- clone the project from <URL>
- use a recent version of python (e.g. 3.12) to get started
    - `python3.12 -m venv env`
- upgrade to the latest version of the python package manager
    - `pip install -U pip`
- This project uses poetry for dependency management. Install poetry:
    - `pip install poetry`
- Install blocktrain
    - `poetry install`

### Recommended Development Environment

This project was developed with vscode running within a WSL2 Red Hat Universal Base Image development environment. 

To run jupyter notebooks a recommendation is to install the Microsoft Jupyter vscode plugin (ms-toolsai.jupyter)

### Notebook Usage

### Training Usage

#### Recommended Pre-checks for Training

It is recommended that training be run within a linux environment

To ensure the target training device is ready to support some steps can be taken:

- Test: run `nvidia smi` on the command line
- Expected Result: A text block with NVIDIA driver information
```
(env) [jacob@rhel8 blocktrain]$ nvidia-smi
Fri Aug 15 15:46:53 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 572.83         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:01:00.0  On |                  Off |
|  0%   37C    P8             23W /  450W |    1548MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

- Test: run `python -c "import torch;print(torch.cuda.is_available())"` on the command line
- Expected Result: `True` printed to stdout

### Evaluation Usage

---

# Request for Discussion

1. ### Initial Directions and statement breakdown

Given the stakeholder's request for a result within a short timeframe, some initial assumptions are considered. These initial assumptions will be refined when data analysis is completed.

Initially, given a quick read of the stakeholder's initial input, we can infer that we may be able to:
- Apply used supervised learning techniques to solve the problem

|         Data               |                             Response                      |
| -------------------------- | --------------------------------------------------------- |
| "Design an ML pipeline..." | Leverage a machine learning framework to train an ML model |
| "set of videos as target test cases" | Visual detection modality suggests we should consider computer vision models |


The stakeholder has provided the desire for a 
 > prototype model for target recognition of a drone target

This strongly implies the desire for a machine learning computer vision model capabile of target recognition that is conditioned on the provided training dataset and able to acheive some level of performance (unspecified) on the test dataset.

Some other considerations:
- Deep learning, specifically deep computer vision may be applicable for this use case due to computer vision models' ability to recognize objects when conditioned on data


1. ### Description of Training Pipeline

Machine learning can be thought of as multiple, sometimes disparate data intensive, and compute intensive processes. 

1. ### Desired Training Pipeline Features

Initially, the training pipeline should:
1) Expose a simple API:
     - with functions like `train`, `eval`, `predict` where:
        - `train`: conditions an ML on a dataset and results in the generation of a set of weights to be deployed
        - `eval`: evaluates a model and trained weights against a dataset to determine metrics
        - `predict`: performs inference on a batch of examples and produces results in some format.
2) Be Configurable:
    - such that datasets, training, parameters, and ML models can be relatively easily replaced.
3) Experiments as a first-class-citizen
    - The design should facilitate a usage pattern where a typical interaction is to specify one or more experiments to conduct with a given purpose.

1. ### "BlockTrain" Training Pipeline Component Design

1.1 Component Concepts

1.1.1 Experiment Specification

The Experiment Specification is a design construct of the pipeline that allows component level data driven experimentation. That is, a user can adjust the parameters of the training experiment to quickly acheive different results in a config-driven way. Machine learning training pipelines have canonical components that are shared across many implementations. These include datasets, models, callbacks, etc. The Experiment Specification allows these items to be changed, or for the parameters of each to be adjusted to facilitate the current experiment.

1.1.2 Torch Datasets and Dataloaders

Pytorch datasets are a common and very practical way to interact with streaming or map style datasets. Implementing datasets is typically trival, especially for small datasets that can fit on the filesystem for a single workstation. Torch Dataloaders support various batch loading paradigms and can simplify and streamling the training process. For this training pipeline we will leverage torch datasets. Future work can bring Datasets and Dataloading constructs from other training frameworks (e.g. huggingface or pytorch lighting)

1. ### Analysis of Supporting Frameworks and Tooling

1. ### Analysis of Training Datasets

1. ### Analysis of Test Dataset

1. ### Analysis of prototype sensor model

1. ### Problem Discussion

1. ### ML model discussion

1. ### Model deployment

1. ### Design goals, Ground Rules, Assumptions

1. ### Future Work

- Integration with experiment tracking tooling.

1. ### Additional Information

1. ### Citations

- [1] Svanström F, Englund C and Alonso-Fernandez F. (2020), GitHub repository, <br>
  https://github.com/DroneDetectionThesis/Drone-detection-dataset
- [2] http://dx.doi.org/10.5281/zenodo.5500576
