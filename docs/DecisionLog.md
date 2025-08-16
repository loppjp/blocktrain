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

2. Poetry for dependency management

Poetry is a relatively popular, although somewhat slow, package management system for python. It trades speed for reliability. 

Due to the potential for this project to be evaluated by stakeholders, it is critical that the dependencies be loaded in a reliable way. Since this project will involve experiementation under short timeframes, dependencies may change throughout the design and build process. These changes may lead to instability and time wasted due to ensuring packages are installed in the correct order.

To the developer's knowledge, the suitable package management trade space include pip, poetry, conda, and uv. 

The instability of builds due to ordering leads us to look at options other than vanilla pip except for the initial environment bootstrap. The developer is not as familiar with conda and uv and as such, given the timeframe, would perfer to not use these.

 Without time to evaluate conda and uv, poetry has been chosen for this project to allow dependencies to be added and removed easily during the development process and ideally result in more reproducable builds.