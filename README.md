# CodeEvolve Experiments

<div align="center">

<img src="assets/codeevolve_logo.png" alt="CodeEvolve Logo" width="400">
<p align="center">
  <img src="https://img.shields.io/badge/version-v0.2.1-green" alt="v0.2.1"></a>
  <a href="https://arxiv.org/abs/2510.14150"><img src="https://img.shields.io/badge/arxiv-2510.14150-red" alt="Arxiv"></a>
  <a href="https://github.com/inter-co/science-codeevolve/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
</p>

**An open-source framework that combines large language models with evolutionary algorithms to discover and optimize high-performing code solutions.**

</div>

This is a companion repository to [science-codeevolve](https://github.com/inter-co/science-codeevolve), and contains the complete experimental setup, benchmark implementations, and reproducibility code for the CodeEvolve [paper](https://arxiv.org/abs/2510.14150).

## Overview

This repository provides:

- **Complete benchmark problems** used in the paper's evaluation
- **Experimental configurations** for reproducing all results
- **Raw experimental data** from paper runs (`.pkl`, `.py`, `.txt` files)
- **Analysis notebooks** with visualizations and statistical tests

All experiments validate CodeEvolve's performance on algorithmic discovery tasks from mathematics, demonstrating competitive or superior results compared to closed-source systems like Google DeepMind's AlphaEvolve.

## Repository Structure

```
science-codeevolve-experiments/
├── experiments/          # Raw experimental results
│   └── alphaevolve_math_problems/
│       ├── autocorrelation_problems/    # Autocorrelation inequalities
│       ├── minimizing_max_min_dist/     # Max-min distance problems
│       └── packing_problems/            # Circle and hexagon packing
├── notebooks/           # Analysis and visualization
│   ├── experiment_analysis.ipynb       # Main analysis notebook
│   └── figs/                           # Generated figures from paper
├── problems/            # Benchmark problem definitions
│   ├── alphaevolve_math_problems/
│   │   ├── autocorrelation_problems/
│   │   ├── minimizing_max_min_dist/
│   │   └── packing_problems/
└── README.md
```

### Directory Details

- **`experiments/`**: Contains results from paper experiments including:
  - Solution histories (`.py` files)
  - Checkpoints (`.pkl` files)
  - Logs and metadata (`.txt` files)
  - Multiple runs with different seeds/configurations

- **`notebooks/`**: Jupyter notebooks for analysis
  - `experiment_analysis.ipynb`: Statistical analysis and comparisons

- **`problems/`**: Problem definitions with:
  - Initial solution templates (`input/`)
  - Configuration files for different LLMs (`configs/`)
  - Evaluation scripts

## Prerequisites

### Install CodeEvolve Framework

First, install the main CodeEvolve framework:

```bash
# Clone and install CodeEvolve framework
git clone https://github.com/inter-co/science-codeevolve.git
cd science-codeevolve
conda env create -f environment.yml
conda activate codeevolve
cd ..
```

### Clone Experiments Repository

```bash
# Clone this experiments repository
git clone https://github.com/inter-co/science-codeevolve-experiments.git
cd science-codeevolve-experiments
```

### Configure LLM API Access

Set your LLM API credentials as environment variables:

```bash
export API_KEY=your_api_key_here
export API_BASE=your_api_base_url
```

## Paper Results

### Available Benchmark Problems

| Problem Category | Problem | Parameters | Description |
|-----------------|---------|------------|-------------|
| **Autocorrelation** | First Autocorr Ineq | - | First autocorrelation inequality |
| | Second Autocorr Ineq | - | Second autocorrelation inequality |
| **Heilbronn** | Triangle | - | Heilbronn triangle problem |
| | Convex | 13, 14 | Heilbronn convex hull problem |
| **Max-Min Distance** | 2D | N=16 | Place N 2D points, Minimize Max-Min distance ratio |
| | 3D | N=14 | Place N 3D points, Minimize Max-Min distance ratio |
| **Packing** | Circle in Rectangle | - | Pack N circles in rectangle |
| | Circle in Square | 26, 32 | Pack N circles in unit square |
| | Hexagon Packing | 11, 12 | Pack N hexagons in larger hexagon |

### Running a Benchmark Problem

Each problem has configuration files for different LLM providers (Gemini, Qwen, etc.). Here's how to run an experiment:

```bash
# Example: Circle packing in a square (26 circles) with Qwen
codeevolve \
  --inpt_dir=problems/alphaevolve_math_problems/packing_problems/circle_packing_square/26/input \
  --cfg_path=problems/alphaevolve_math_problems/packing_problems/circle_packing_square/26/configs/qwen_config.yaml \
  --out_dir=results/circle_packing_26_qwen \
  --terminal_logging

# Example: First autocorrelation inequality with Gemini
codeevolve \
  --inpt_dir=problems/alphaevolve_math_problems/autocorrelation_problems/first_autocorr_ineq/input \
  --cfg_path=problems/alphaevolve_math_problems/autocorrelation_problems/first_autocorr_ineq/configs/gemini_config.yaml \
  --out_dir=results/autocorr_first_gemini \
  --terminal_logging
```

### Resuming from Checkpoints

To resume an interrupted run:

```bash
codeevolve \
  --inpt_dir=problems/alphaevolve_math_problems/packing_problems/circle_packing_square/26/input \
  --out_dir=results/circle_packing_26_qwen \
  --load_ckpt=-1  # Load latest checkpoint
```

Or load a specific checkpoint epoch:

```bash
codeevolve \
  --inpt_dir=problems/alphaevolve_math_problems/packing_problems/circle_packing_square/26/input \
  --out_dir=results/circle_packing_26_qwen \
  --load_ckpt=100  # Load checkpoint from epoch 100
```

### Reproducibility

This repository supports two distinct notions of reproducibility:

#### 1) Reproducing the paper analysis (deterministic, using included artifacts)
The folder `experiments/` contains the raw artifacts used in the paper (checkpoints, histories, logs). The notebook(s) in `notebooks/` analyze those artifacts to generate the plots and comparisons. Re-running the analysis should reproduce the reported figures/tables as long as your analysis environment is compatible.

#### 2) Re-running the full search (best-effort; exact replay depends on the LLM provider)
**Exact numerical reproduction of a full evolutionary run is not guaranteed** when using hosted LLM APIs.

Why:
- Many commercial LLM providers **do not support deterministic sampling** or **do not honor `seed`**.
- Even when a provider accepts `seed`, outputs can vary due to backend nondeterminism (load balancing, infrastructure-level randomness, model version rollouts).

This is not a limitation of CodeEvolve’s evolutionary framework: CodeEvolve is **seedable for its internal stochastic decisions**, and it forwards model `seed` to OpenAI-compatible endpoints when supported. The remaining nondeterminism comes from the LLM backbone/provider.

## Citation
```bibtex
@article{assumpção2025codeevolveopensourceevolutionary,
      title={CodeEvolve: An open source evolutionary coding agent for algorithm discovery and optimization},
      author={Henrique Assumpção and Diego Ferreira and Leandro Campos and Fabricio Murai},
      year={2025},
      eprint={2510.14150},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.14150},
}
```

## Releases

Experiments are versioned to match the main repository:

- **v0.1.0**: Initial release, corresponds to v1 of CodeEvolve's [paper](https://arxiv.org/abs/2510.14150).
- **v0.2.0**: Current release, corresponds to v3 of CodeEvolve's [paper](https://arxiv.org/abs/2510.14150).

## Acknowledgements

The authors thank Bruno Grossi for his continuous support during the development of this project. We thank Fernando Augusto and Tiago Machado for useful conversations about possible applications of CodeEvolve. We also thank the [OpenEvolve](https://github.com/codelion/openevolve) community for their inspiration and discussion about evolutionary coding agents.

## License

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0.

**This is not an official Inter product.**
