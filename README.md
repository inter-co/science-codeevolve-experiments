# CodeEvolve Experiments Repository

This repository contains the complete experimental setup, benchmark implementations, and reproducibility code for the CodeEvolve research paper.

> **CodeEvolve: An open source evolutionary coding agent for algorithm discovery and optimization**  
> Henrique Assumpção, Diego Ferreira, Leandro Campos, Fabricio Murai  
> [arXiv:2510.14150](https://arxiv.org/abs/2510.14150)

## Overview

This companion repository to [science-codeevolve](https://github.com/inter-co/science-codeevolve) provides:

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

## Reproducing Paper Results

### Running a Benchmark Problem

Each problem has configuration files for different LLM providers (Gemini, Qwen, etc.). Here's how to run an experiment:

```bash
# Example: Circle packing in a square (26 circles) with Qwen
codeevolve \
  --inpt_dir=problems/alphaevolve_math_problems/packing_problems/circle_packing_square/26 \
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

### Available Benchmark Problems

| Problem Category | Problem | Dimensions | Description |
|-----------------|---------|------------|-------------|
| **Autocorrelation** | First Autocorr Ineq | - | First autocorrelation inequality |
| | Second Autocorr Ineq | - | Second autocorrelation inequality |
| **Heilbronn** | Triangle | - | Heilbronn triangle problem |
| | Convex | 13, 14 | Heilbronn convex hull problem |
| **Max-Min Distance** | Dimension 2 | 2D | Maximize minimum distance |
| | Dimension 3 | 3D | Maximize minimum distance |
| **Packing** | Circle in Rectangle | - | Pack circles in rectangle |
| | Circle in Square | 26, 32 | Pack N circles in unit square |
| | Hexagon Packing | 11, 12 | Pack N hexagons in larger hexagon |

### Resuming from Checkpoints

To resume an interrupted run:

```bash
codeevolve \
  --inpt_dir=problems/alphaevolve_math_problems/packing_problems/circle_packing_square/26 \
  --out_dir=results/circle_packing_26_qwen \
  --load_ckpt=-1  # Load latest checkpoint
```

Or load a specific checkpoint epoch:

```bash
codeevolve \
  --inpt_dir=problems/alphaevolve_math_problems/packing_problems/circle_packing_square/26 \
  --out_dir=results/circle_packing_26_qwen \
  --load_ckpt=100  # Load checkpoint from epoch 100
```

### Exact Reproducibility

Our experimental results were obtained using Qwen and Gemini models as the backbone for our LLM ensembles. Both models were accessed via an internal API system at Inter that routed requests to the respective LLM providers. Many commercial LLM providers do not guarantee deterministic outputs even when random seeds are provided. As a result, **exact numerical reproduction of our paper results is not guaranteed**, even when using the same configuration files and seeds. Despite these limitations, our ablation studies demonstrate that CodeEvolve consistently achieves **state-of-the-art results across multiple seeds and experimental runs** on all considered benchmarks. The core algorithmic contributions remain robust to LLM stochasticity.

## Analyzing Results

### Using the Analysis Notebook

```bash
# Make sure jupyter is installed
conda activate codeevolve
pip install jupyter matplotlib pandas

# Launch notebook
jupyter notebook notebooks/experiment_analysis.ipynb
```

The notebook provides:
- Solution quality over time plots
- Comparison with AlphaEvolve baselines
- Ablation study analysis

## Citation

If you use CodeEvolve or these benchmarks in your research, please cite:

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

- **v0.1.0**: Initial release, corresponds to v1 of technical report
- **v0.2.0**: Current release, corresponds to v3 of technical report

## Acknowledgements

The authors thank Bruno Grossi for his continuous support during the development of this project. We thank Fernando Augusto and Tiago Machado for useful conversations about possible applications of CodeEvolve. We also thank the [OpenEvolve](https://github.com/codelion/openevolve) community for their inspiration and discussion about evolutionary coding agents.

## License

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0.

**This is not an official Inter product.**
