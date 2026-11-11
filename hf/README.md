---
license: apache-2.0
pretty_name: CodeEvolve Experiments
language:
- en
tags:
- code
- evolutionary-algorithms
- llm
- program-synthesis
- algorithm-discovery
- alphaevolve
size_categories:
- 100K<n<1M
configs:
- config_name: runs
  default: true
  data_files: "data/runs/*.parquet"
- config_name: programs
  data_files: "data/programs/*.parquet"
- config_name: prompts
  data_files: "data/prompts/*.parquet"
- config_name: fitness_history
  data_files: "data/fitness_history/*.parquet"
- config_name: llm_calls
  data_files: "data/llm_calls/*.parquet"
- config_name: errors
  data_files: "data/errors/*.parquet"
---

# CodeEvolve Experiments

Complete experimental results for the paper
[CodeEvolve: an open-source evolutionary framework for algorithmic discovery and optimization](https://arxiv.org/abs/2510.14150)
(Findings of EMNLP 2026).

[CodeEvolve](https://github.com/inter-co/science-codeevolve) combines LLMs with an island-based genetic
algorithm to evolve programs. This dataset contains every program, evolved prompt, LLM call and error
recorded across the paper's runs: 213 runs on 14 problems
(AlphaEvolve mathematical benchmarks and EoH combinatorial-optimization benchmarks), using Qwen, Gemini,
GPT and Claude models, including all ablations.

The original files (checkpoints, logs, configs) are in [`raw/`](#raw-files); the tables below are
derived from them with
[`scripts/export_hf.py`](https://github.com/inter-co/science-codeevolve-experiments/blob/main/scripts/export_hf.py).

## Quick start

```python
from datasets import load_dataset

runs = load_dataset("HAssumpcao/CodeEvolve-Experiments", "runs", split="train")
programs = load_dataset("HAssumpcao/CodeEvolve-Experiments", "programs", split="train")

# Best program found in each run
best = programs.filter(lambda r: r["is_run_best"])

# Fitness curves for one problem
import pandas as pd
hist = load_dataset("HAssumpcao/CodeEvolve-Experiments", "fitness_history", split="train").to_pandas()
hist[hist["problem"] == "packing_problems/circle_packing_square/26"]
```

`programs` is the largest table (~1 GB of Parquet); pass `columns=[...]` to `to_pandas`, or use
`streaming=True`, if you don't need the full LLM conversations.

## Tables

All tables share the run keys below, so they can be joined on `run_id` (and `island` / `epoch` /
`program_id` where present).

| Column | Description |
|---|---|
| `run_id` | Path of the run inside `raw/experiments/`, e.g. `eoh-problems/bp_online/qwen/reviews/qwen_full_1` |
| `domain` | `alphaevolve_math_problems` or `eoh-problems` |
| `problem` | Problem (and instance size where applicable), e.g. `packing_problems/circle_packing_square/26` |
| `model_family` | LLM family used for the run (`qwen`, `gemini`, `gpt`, `claude`) |
| `group` | Experiment group (main comparison or ablation, see [Experiment groups](#experiment-groups)) |
| `run_name` | Run name inside the group (configuration variant and replicate) |

### `runs` (213 rows)

One row per run: `num_islands`, `final_epoch`, `best_fitness`, `best_program_id`, `best_island`
(island where the best program was found), `best_code`, `best_prompt` (the system prompt that produced
it), `models` (LLMs in the run's ensembles), totals (`num_programs`, `num_prompts`, `num_llm_calls`,
`prompt_tokens`, `completion_tokens`, `num_errors`) and the full run configuration (`config_yaml`).

### `programs` (434,906 rows)

One row per program evaluated in a run.

| Column | Description |
|---|---|
| `program_id` | Program UUID (unique within a run) |
| `island_found`, `iteration_found`, `generation` | Island and epoch at which the program was created |
| `islands` | All islands holding a copy at the end of the run (more than one if the program migrated) |
| `parent_id` | Parent program (`null` for the initial program and the LLM-generated initial population) |
| `prompt_id` | System prompt (row in `prompts`) used to generate this program |
| `inspiration_ids` | Programs shown to the LLM as inspirations |
| `model_id` | Index of the LLM within the ensemble that generated the program |
| `fitness` | Fitness used for selection (higher is better) |
| `eval_metrics` | All metrics returned by the problem's evaluator (JSON string) |
| `features` | MAP-Elites behavioural features (JSON string) |
| `returncode`, `error` | Evaluator exit code and error message, if the evaluation failed |
| `is_alive` | Whether the program was still in its island's population at the end of the run |
| `is_island_best`, `is_run_best` | Best program on some island / in the whole run |
| `code` | Full program source |
| `prog_msg` | User message sent to the LLM (parent code and context) |
| `model_msg` | Raw LLM response (SEARCH/REPLACE diff) |

Programs that migrated between islands appear once, with the lineage recorded on the island where they
were created.

### `prompts` (117,701 rows)

System prompts evolved through meta-prompting. Same columns as `programs`; `code` holds the prompt text.

### `fitness_history` (459,290 rows)

One row per run, island and epoch: `best_fitness` and `avg_fitness` of the island's population, and
whether the epoch was an `exploration` step.

### `llm_calls` (586,233 rows)

One row per LLM request: `island`, `epoch`, `motive` (`generate_prog` or `meta_prompt`), `model_name`,
`prompt_tokens`, `completion_tokens`.

### `errors` (37,791 rows)

One row per failed step: `island`, `epoch`, `motive`, `error_msg`, the parent ids, and the LLM's `diff`.
Most rows are SEARCH/REPLACE diffs that failed to apply (`sr_evolve_prog`, `sr_meta_prompt`), which makes
this table a corpus of real LLM diff-application failures.

### Notes

- Non-finite values (NaN, ±inf) inside the JSON columns are stored as `null`; the numeric `fitness`
  columns keep them as-is.
- Model names are the identifiers used by our LLM gateway, e.g. `AWS_QWEN-QWEN3-CODER-30B-INSTRUCT`,
  `GOOGLE_GEMINI-2.5-FLASH`.

## Experiment groups

Run names encode the configuration variant; a trailing `_1`, `_2`, `_3` is the replicate. Unless stated,
ablations change a single parameter of the full CodeEvolve configuration and use three replicates per
variant. The exact configuration of every run is in `runs.config_yaml`.

| `group` | Model families | Runs | Description |
|---|---|---|---|
| `ablations_comp` | qwen, gemini | 115 | Main results and component ablation on the AlphaEvolve problems. `*_full`: complete CodeEvolve (meta-prompting, inspirations, depth-based exploitation, MAP-Elites, island migration). `qwen_naive`: plain evolutionary loop with all of these disabled (no meta-prompting, no inspirations, chat depth 0, no MAP-Elites). `qwen_no_evolve`: no evolution, the LLM budget is spent sampling independent programs (`init_pop` = `num_epochs`, no migration). `qwen_full_*` suffixes are variants of the full setup: `d3`/`dnull` chat depth 3/unlimited, `shallow` and `ss` depth 1, `s` depth 1–2, `grid` a grid MAP-Elites archive. `gemini_10`: Gemini with 10 islands. |
| `ablations_depth` | qwen | 12 | Maximum chat depth used for exploitation: 0, 3, 5 and unlimited (`qwen_d0`, `qwen_d3`, `qwen_d5`, `qwen_dunlim`) on `circle_packing_square/32`. |
| `ablations_elites` | qwen | 6 | Grid MAP-Elites archive (`qwen_grid`) vs. no MAP-Elites with a fixed population of 60 (`qwen_naive`) on `circle_packing_square/32`. |
| `ablations_insp` | qwen | 12 | Number of inspiration programs shown to the LLM: 0–3 (`qwen_i0` … `qwen_i3`) on `circle_packing_square/32`. |
| `ablations_size` | qwen | 6 | Six-island runs on `circle_packing_square/26` crossing the MAP-Elites archive (`cvt` vs. `grid`) with the exploration scheduler (`exp`: exponential decay, `plat`: plateau). |
| `ablations_top` | qwen | 6 | Migration topology: `complete` graph vs. `empty` (isolated islands); the default elsewhere is a ring. On `circle_packing_square/32`. |
| `gemini_depth` | gemini | 4 | Chat-depth ablation with Gemini (0, 3, 5, unlimited), one run each, on `circle_packing_square/26`. |
| `more_isl` | qwen, gemini | 9 | Same problems as `new` with more islands (Gemini 5 instead of 3, Qwen 10 instead of 8). |
| `new` | qwen, gemini | 10 | Runs on the min/max-distance and circle-packing problems with fewer islands (Gemini 3, Qwen 8); compare with `more_isl`. |
| `reviews` | qwen, gpt, claude | 21 | Additional experiments: other LLMs on `circle_packing_square/32` (Claude Sonnet 4.5, GPT-4.1, o4-mini, gpt-oss-20b; 100 epochs), and Qwen on the EoH benchmarks (`bp_online`, `fssp_gls`, `tsp_gls`). |
| `reviews2` | gpt | 3 | gpt-oss-20b on `circle_packing_square/32` with a 200-epoch budget instead of 100 (see note below). |
| `tests` | qwen | 9 | Exploratory long runs (500–1000 epochs), including the only Heilbronn (`heilbronn_convex/13`, `/14`) runs. |

The three `reviews2/gpt_oss_small_*` runs were stopped at epoch 170–180 of their 200-epoch budget, and only
their epoch-150 checkpoints are included. The tables reflect epoch 150, while `best_sol.py` and
`results.log` in `raw/` reflect the last epoch reached.

## Raw files

`raw/experiments/` mirrors the original
[science-codeevolve-experiments](https://github.com/inter-co/science-codeevolve-experiments) layout:

```
<domain>/<problem>/<model_family>/<group>/<run_name>/
├── <model>_config.yaml        # run configuration
└── <island>/
    ├── ckpt/ckpt_<epoch>.pkl  # final checkpoint (program/prompt databases, evolve state)
    ├── best_sol.py            # best program on this island
    ├── best_prompt.txt        # best prompt on this island
    └── results.log            # full run log
```

The checkpoints are Python pickles of CodeEvolve objects: loading them requires
[`codeevolve`](https://github.com/inter-co/science-codeevolve) and executes code, so only load files
from this repository. Use `scripts/export_hf.py`'s `load_ckpt`, which also handles a class renamed
since some runs were saved.

## Citation

```bibtex
@inproceedings{assumpcao-etal-2026-codeevolve,
    title = "{C}ode{E}volve: an open-source evolutionary framework for algorithmic discovery and optimization",
    author = "Assump{\c{c}}{\~a}o, Henrique  and
      Ferreira, Diego Leonardo Braga  and
      Lacerda Campos, Leandro Augusto  and
      Murai, Fabricio",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.1345/",
    pages = "28559--28581",
    ISBN = "979-8-89176-462-0"
}
```

## License

Apache 2.0.
