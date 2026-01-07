# Experiments with CodeEvolve
This repository contains benchmark implementations, experimental configurations, and reproducibility code for the CodeEvolve paper (versions 1 and 2):

> **CodeEvolve: an open source evolutionary coding agent for algorithm discovery and optimization**  
> Henrique Assumpção, Diego Ferreira, Leandro Campos, Fabricio Murai  
> [arXiv:2510.14150](https://arxiv.org/abs/2510.14150)


## Prerequisites
Install CodeEvolve v0.1.0 and dependencies:

```bash
# Clone and install CodeEvolve framework
git clone https://github.com/inter-co/science-codeevolve.git
cd science-codeevolve
conda env create -f environment.yml
conda activate codeevolve

# Clone this experiments repository
cd ..
git clone https://github.com/inter-co/science-codeevolve-experiments.git
cd science-codeevolve-experiments

# Set your LLM API credentials
export API_KEY=your_api_key
export API_BASE=your_api_base_url
```

## Citation

If you use CodeEvolve in your research, please cite our paper:

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

## Acknowledgements

The authors thank Bruno Grossi for his continuous support during the development of this project. We thank Fernando Augusto and Tiago Machado for useful conversations about possible applications of CodeEvolve. We also thank the [OpenEvolve](https://github.com/codelion/openevolve) community for their inspiration and discussion about evolutionary coding agents.

## License and Disclaimer

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0.

**This is not an official Inter product.**