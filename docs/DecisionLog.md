---
title: "BlockTrain"
author: Jacob Lopp
output:
    html_document:
        numbered_sections: true
---

# Descision Log

1. Focus on Training Pipeline and prioritization

Per stakeholder input to "Design an ML pipeline", emphasis for this project will be to design a ML training pipeline using a given dataset and target test cases. 

### Priorities

#### High

- Design artifacts for a CV ML training pipeline.
    - At this time, CV and ML are inferred based on context
- Design artifacts supporting training pipeline development using the Request for Discussion (RFD) Format.
- Publication of source code, RFD on public facing Github.
- Model inference using the test set videos.
- Generation of model weights using one or both training datasets.
- Considerations for the expected sensor and updates to documents and code to support considerations.

#### Medium

- Documentation of code
- Training Metrics
- Exploratory data analysis notebooks

1. Poetry for dependency management

Poetry is a relatively popular, although somewhat slow, package management system for python. It trades speed for reliability. 

Due to the potential for this project to be evaluated by stakeholders, it is critical that the dependencies be loaded in a reliable way. Since this project will involve experiementation under short timeframes, dependencies may change throughout the design and build process. These changes may lead to instability and time wasted due to ensuring packages are installed in the correct order.

To the developer's knowledge, the suitable package management trade space include pip, poetry, conda, and uv. 

The instability of builds due to ordering leads us to look at options other than vanilla pip except for the initial environment bootstrap. The developer is not as familiar with conda and uv and as such, given the timeframe, would perfer to not use these.

 Without time to evaluate conda and uv, poetry has been chosen for this project to allow dependencies to be added and removed easily during the development process and ideally result in more reproducable builds.

1. Use of pytorch

There are multiple machine learning frameworks, and multiple deep learning frameworks. Given the need for speed-to-implementation we are choosing to use pytorch due to familiarity and rich ecosystem. Alternatives momentarily considered are tensorflow, jax, scikit learn, open cv.

1. Data Analysis Training Split
 
 First, shuffle split the training dataset as soon as there is a reasonable mechanism to load data from the source (disk) without doing distribution analysis.

 Begin to do data analysis on a static, stable, shuffled subset of images and data (e.g. 60%/20%/20% split). Leverage this as the training dataset.

1. Use of Torch Datasets

 This implementation will leverage the Dataset APIs from pytorch. Torch datasets are widely used, there is a rich API for combining and sampling different dataset. Given this compositional nature, and due to less familiarity with huggingface datasets and other providers, the choice is to pursue torch datasets for now.

1. 

 2 initial types of Dataset classes will be created, ones that load from the filesystem with load perations occuring at each invocation of `__getitem__` and the other pre-loading the dataset into RAM at class initialization time.

 This is done to allow flexibility in workflows depending on RAM or disk utilization. The expectation will be to use the RAM based datasets more often.