This repository contains the code required to reproduce the results in the following publication:

### Deep reinforcement learning for coordinated air-conditioner control in groups of buildings using smart meter data

- Journal: Engineering Applications of Artificial Intelligence 
- DOI: [10.1016/j.engappai.2025.113536](https://doi.org/10.1016/j.engappai.2025.113536)
- Published in Feb 2026

#### Installation and Run steps:
1. In a fresh Python (Python >=3.12) environment, run the following commands:
    - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
    - `pip install -e .`
2. There are 6 experiments contained in the project. Each one is in a dedicated subdirectory under `experiments` in the repo root. Please verify the configurations for each experiment (`<expt_dir>/config.yaml`) before running the experiment (Step 3). 
3. From the repository root, run the following commands for each experiment
    - `python scripts/run_training.py --list` - lists all available experiments
    - `python scripts/run_training.py <experiment_tag>` - run the specified experiment
4. Visualize the results and generate plots using the notebooks in the `notebooks` directory once all the experiments have been run.

#### Citation:
```
@article{ramKumarEAAI2026,
title = {Deep reinforcement learning for coordinated air-conditioner control in groups of buildings using smart meter data},
journal = {Engineering Applications of Artificial Intelligence},
volume = {166},
pages = {113536},
year = {2026},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.113536},
url = {https://www.sciencedirect.com/science/article/pii/S0952197625035675},
author = {Sharath {Ram Kumar} and Arvind Easwaran and Benoit Delinchant and Remy Rigo-Mariani}}
```
