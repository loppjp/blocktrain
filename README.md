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

1. ### Analysis of Supporting Frameworks and Tooling

1. ### Analysis of Training Datasets

1. ### Analysis of Test Dataset

1. ### Analysis of prototype sensor model

1. ### Problem Discussion



1. ### ML model discussion

1. ### Model deployment

1. ### Design goals, Ground Rules, Assumptions

1. ### Future Work

1. ### Additional Information

1. ### Citations

- [1] Svanström F, Englund C and Alonso-Fernandez F. (2020), GitHub repository, <br>
  https://github.com/DroneDetectionThesis/Drone-detection-dataset
- [2] http://dx.doi.org/10.5281/zenodo.5500576
- [3] Richards, M., & Ford, N. (2020). Fundamentals of Software Architecture: An Engineering 
  Approach. O’Reilly Media.
