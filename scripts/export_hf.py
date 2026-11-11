# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# Exports the raw experiment checkpoints in `experiments/` to Parquet tables for the
# Hugging Face Hub (one config per table, one shard per problem).
#
# Usage:
#   python scripts/export_hf.py --exp_dir experiments --out_dir /path/to/hf_dataset
#
# Requires `codeevolve` (to unpickle checkpoints) and `pyarrow`.
#
# ===--------------------------------------------------------------------------------------===#

import argparse
import json
import math
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

# Classes that were renamed after some paper runs were checkpointed.
_RENAMED_CLASSES: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("codeevolve.scheduler", "ExponentialDecayScheduler"): (
        "codeevolve.scheduler",
        "ExponentialScheduler",
    ),
}


class _CompatUnpickler(pickle.Unpickler):
    """Unpickler that maps renamed codeevolve classes to their current names."""

    def find_class(self, module: str, name: str) -> Any:
        module, name = _RENAMED_CLASSES.get((module, name), (module, name))
        return super().find_class(module, name)


def load_ckpt(path: Path) -> Dict[str, Any]:
    """Loads a CodeEvolve checkpoint written by any version used in the paper."""
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_RUN_KEY_FIELDS: List[pa.Field] = [
    pa.field("run_id", pa.string()),
    pa.field("domain", pa.string()),
    pa.field("problem", pa.string()),
    pa.field("model_family", pa.string()),
    pa.field("group", pa.string()),
    pa.field("run_name", pa.string()),
]

_PROGRAM_FIELDS: List[pa.Field] = [
    pa.field("program_id", pa.string()),
    pa.field("island_found", pa.int32()),
    pa.field("islands", pa.list_(pa.int32())),
    pa.field("iteration_found", pa.int32()),
    pa.field("generation", pa.int32()),
    pa.field("parent_id", pa.string()),
    pa.field("prompt_id", pa.string()),
    pa.field("inspiration_ids", pa.list_(pa.string())),
    pa.field("model_id", pa.int32()),
    pa.field("fitness", pa.float64()),
    pa.field("returncode", pa.int32()),
    pa.field("eval_metrics", pa.string()),
    pa.field("features", pa.string()),
    pa.field("is_alive", pa.bool_()),
    pa.field("is_island_best", pa.bool_()),
    pa.field("is_run_best", pa.bool_()),
    pa.field("language", pa.string()),
    pa.field("code", pa.string()),
    pa.field("output", pa.string()),
    pa.field("warning", pa.string()),
    pa.field("error", pa.string()),
    pa.field("prog_msg", pa.string()),
    pa.field("model_msg", pa.string()),
]

SCHEMAS: Dict[str, pa.Schema] = {
    "runs": pa.schema(
        _RUN_KEY_FIELDS
        + [
            pa.field("num_islands", pa.int32()),
            pa.field("final_epoch", pa.int32()),
            pa.field("best_fitness", pa.float64()),
            pa.field("best_island", pa.int32()),
            pa.field("best_program_id", pa.string()),
            pa.field("best_code", pa.string()),
            pa.field("best_prompt", pa.string()),
            pa.field("models", pa.list_(pa.string())),
            pa.field("num_programs", pa.int32()),
            pa.field("num_prompts", pa.int32()),
            pa.field("num_llm_calls", pa.int32()),
            pa.field("prompt_tokens", pa.int64()),
            pa.field("completion_tokens", pa.int64()),
            pa.field("num_errors", pa.int32()),
            pa.field("config_file", pa.string()),
            pa.field("config_yaml", pa.string()),
        ]
    ),
    "programs": pa.schema(_RUN_KEY_FIELDS + _PROGRAM_FIELDS),
    "prompts": pa.schema(_RUN_KEY_FIELDS + _PROGRAM_FIELDS),
    "fitness_history": pa.schema(
        _RUN_KEY_FIELDS
        + [
            pa.field("island", pa.int32()),
            pa.field("epoch", pa.int32()),
            pa.field("best_fitness", pa.float64()),
            pa.field("avg_fitness", pa.float64()),
            pa.field("exploration", pa.bool_()),
        ]
    ),
    "llm_calls": pa.schema(
        _RUN_KEY_FIELDS
        + [
            pa.field("island", pa.int32()),
            pa.field("epoch", pa.int32()),
            pa.field("motive", pa.string()),
            pa.field("model_name", pa.string()),
            pa.field("prompt_tokens", pa.int64()),
            pa.field("completion_tokens", pa.int64()),
        ]
    ),
    "errors": pa.schema(
        _RUN_KEY_FIELDS
        + [
            pa.field("island", pa.int32()),
            pa.field("epoch", pa.int32()),
            pa.field("motive", pa.string()),
            pa.field("error_msg", pa.string()),
            pa.field("parent_sol_id", pa.string()),
            pa.field("parent_prompt_id", pa.string()),
            pa.field("diff", pa.string()),
        ]
    ),
}

# Rows per Parquet row group. Program rows carry ~40KB of text on average, so this
# keeps row groups well below the dataset viewer's limits.
_ROW_GROUP_SIZE: Dict[str, int] = {"programs": 1000, "prompts": 1000}
_DEFAULT_ROW_GROUP_SIZE: int = 50_000

# ---------------------------------------------------------------------------
# Value conversion
# ---------------------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Converts numpy scalars/arrays and non-finite floats into plain JSON values.

    Non-finite floats (NaN, +/-inf) become None, since they are not valid JSON.
    """
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):  # numpy arrays and scalars
        return _to_jsonable(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _json(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(_to_jsonable(value), allow_nan=False, sort_keys=True)


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def discover_runs(exp_dir: Path) -> Dict[str, List[Path]]:
    """Groups run directories by problem.

    A run directory is one whose sub-directories `<island>/ckpt/ckpt_<epoch>.pkl`
    hold the island checkpoints. The layout is
    `<domain>/<problem...>/<model_family>/<group>/<run_name>/<island>/ckpt/`.

    Returns:
        Mapping from problem path (e.g. `eoh-problems/bp_online`) to run directories.
    """
    runs: Dict[str, set] = defaultdict(set)
    for ckpt in exp_dir.rglob("ckpt_*.pkl"):
        if ckpt.parent.name != "ckpt" or not ckpt.parents[1].name.isdigit():
            continue
        run_dir: Path = ckpt.parents[2]
        parts: Tuple[str, ...] = run_dir.relative_to(exp_dir).parts
        problem: str = "/".join(parts[:-3])
        runs[problem].add(run_dir)
    return {k: sorted(v) for k, v in sorted(runs.items())}


def _run_keys(exp_dir: Path, run_dir: Path) -> Dict[str, str]:
    parts: Tuple[str, ...] = run_dir.relative_to(exp_dir).parts
    return {
        "run_id": "/".join(parts),
        "domain": parts[0],
        "problem": "/".join(parts[1:-3]),
        "model_family": parts[-3],
        "group": parts[-2],
        "run_name": parts[-1],
    }


def _island_ckpts(run_dir: Path) -> List[Tuple[int, Path]]:
    """Returns (island, final checkpoint path) pairs, one per island."""
    out: List[Tuple[int, Path]] = []
    for isl_dir in sorted(
        (d for d in run_dir.iterdir() if d.name.isdigit()), key=lambda d: int(d.name)
    ):
        ckpts = sorted(
            (isl_dir / "ckpt").glob("ckpt_*.pkl"), key=lambda p: int(p.stem.split("_")[1])
        )
        if ckpts:
            out.append((int(isl_dir.name), ckpts[-1]))
    return out


def _collect_model_names(node: Any, under_ensemble: bool = False) -> List[str]:
    names: List[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            in_ens = under_ensemble or "ENSEMBLE" in str(k).upper()
            if in_ens and k == "model_name" and isinstance(v, str):
                names.append(v)
            else:
                names.extend(_collect_model_names(v, in_ens))
    elif isinstance(node, list):
        for v in node:
            names.extend(_collect_model_names(v, under_ensemble))
    return names


def _read_config(run_dir: Path) -> Tuple[Optional[str], Optional[str], List[str]]:
    cfgs: List[Path] = sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml"))
    if not cfgs:
        return None, None, []
    text: str = cfgs[0].read_text()
    try:
        models = list(dict.fromkeys(_collect_model_names(yaml.safe_load(text))))
    except yaml.YAMLError:
        models = []
    return cfgs[0].name, text, models


# ---------------------------------------------------------------------------
# Conversion of a single run
# ---------------------------------------------------------------------------


def _program_rows(
    keys: Dict[str, str],
    dbs: List[Tuple[int, Any]],
    run_best_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Builds de-duplicated program rows from the per-island databases of one run.

    Migrated programs are present on several islands under the same id; the
    receiving island clears their `parent_id`/`prompt_id`. We keep the copy from
    the island that created the program (`island_found`) so lineage is intact,
    and record every island holding a copy in `islands`.
    """
    copies: Dict[str, List[Tuple[int, Any, Any]]] = defaultdict(list)
    for island, db in dbs:
        for pid, prog in db.programs.items():
            copies[pid].append((island, prog, db))

    rows: List[Dict[str, Any]] = []
    for pid, held in copies.items():
        origin = next((c for c in held if c[0] == c[1].island_found), held[0])
        island, prog, db = origin
        is_alive = getattr(db, "is_alive", {}) or {}
        rows.append(
            {
                **keys,
                "program_id": _str(pid),
                "island_found": _int(prog.island_found),
                "islands": sorted(c[0] for c in held),
                "iteration_found": _int(prog.iteration_found),
                "generation": _int(prog.generation),
                "parent_id": _str(prog.parent_id),
                "prompt_id": _str(prog.prompt_id),
                "inspiration_ids": [str(i) for i in (prog.inspiration_ids or [])],
                "model_id": _int(prog.model_id),
                "fitness": _float(prog.fitness),
                "returncode": _int(prog.returncode),
                "eval_metrics": _json(prog.eval_metrics),
                "features": _json(prog.features),
                "is_alive": bool(is_alive.get(pid, False)) if is_alive else None,
                "is_island_best": any(c[2].best_prog_id == pid for c in held),
                "is_run_best": pid == run_best_id,
                "language": _str(prog.language),
                "code": _str(prog.code),
                "output": _str(prog.output),
                "warning": _str(prog.warning),
                "error": _str(prog.error),
                "prog_msg": _str(prog.prog_msg),
                "model_msg": _str(prog.model_msg),
            }
        )
    rows.sort(key=lambda r: (r["island_found"] or 0, r["iteration_found"] or 0, r["program_id"]))
    return rows


def convert_run(exp_dir: Path, run_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict]:
    """Converts one run into table rows plus a validation report."""
    keys: Dict[str, str] = _run_keys(exp_dir, run_dir)
    tables: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SCHEMAS}
    report: Dict[str, Any] = {"run_id": keys["run_id"], "issues": []}

    sol_dbs: List[Tuple[int, Any]] = []
    prompt_dbs: List[Tuple[int, Any]] = []
    epochs: List[int] = []
    for island, ckpt_path in _island_ckpts(run_dir):
        data = load_ckpt(ckpt_path)
        epoch: int = int(ckpt_path.stem.split("_")[1])
        epochs.append(epoch)
        sol_db, prompt_db, state = data["sol_db"], data["prompt_db"], data["evolve_state"]
        sol_dbs.append((island, sol_db))
        prompt_dbs.append((island, prompt_db))

        best_hist, avg_hist, expl = (
            state.get("best_fit_hist", []),
            state.get("avg_fit_hist", []),
            state.get("exploration", []),
        )
        if not (len(best_hist) == len(avg_hist) == epoch):
            report["issues"].append(
                f"island {island}: history lengths {len(best_hist)}/{len(avg_hist)} != epoch {epoch}"
            )
        for i, (b, a) in enumerate(zip(best_hist, avg_hist)):
            tables["fitness_history"].append(
                {
                    **keys,
                    "island": island,
                    "epoch": i + 1,
                    "best_fitness": _float(b),
                    "avg_fitness": _float(a),
                    "exploration": bool(expl[i]) if i < len(expl) else None,
                }
            )
        for t in state.get("tok_usage", []):
            tables["llm_calls"].append(
                {
                    **keys,
                    "island": island,
                    "epoch": _int(t.get("epoch")),
                    "motive": _str(t.get("motive")),
                    "model_name": _str(t.get("model_name")),
                    "prompt_tokens": _int(t.get("prompt_tok")),
                    "completion_tokens": _int(t.get("compl_tok")),
                }
            )
        for e in state.get("errors", []):
            tables["errors"].append(
                {
                    **keys,
                    "island": island,
                    "epoch": _int(e.get("epoch")),
                    "motive": _str(e.get("motive")),
                    "error_msg": _str(e.get("error_msg")),
                    "parent_sol_id": _str(e.get("parent_sol_id")),
                    "parent_prompt_id": _str(e.get("parent_prompt_id")),
                    "diff": _str(e.get("sol_diff", e.get("prompt_diff"))),
                }
            )

        best_sol_file: Path = ckpt_path.parents[1] / "best_sol.py"
        if best_sol_file.exists():
            if best_sol_file.read_text() != sol_db.programs[sol_db.best_prog_id].code:
                report["issues"].append(f"island {island}: best_sol.py != best program code")
        else:
            report["issues"].append(f"island {island}: best_sol.py missing")

    if len(set(epochs)) > 1:
        report["issues"].append(f"islands stopped at different epochs: {sorted(set(epochs))}")

    # Run-level best across islands.
    _, best_prog = max(
        ((isl, db.programs[db.best_prog_id]) for isl, db in sol_dbs),
        key=lambda x: (_float(x[1].fitness) is not None, _float(x[1].fitness) or -math.inf),
    )

    tables["programs"] = _program_rows(keys, sol_dbs, best_prog.id)
    tables["prompts"] = _program_rows(keys, prompt_dbs, None)
    report["programs_without_origin_copy"] = sum(
        r["island_found"] not in r["islands"] for r in tables["programs"]
    )

    # Use the de-duplicated (origin-island) row: a migrated copy has its prompt_id cleared.
    best_row: Dict[str, Any] = next(r for r in tables["programs"] if r["is_run_best"])
    prompt_code: Dict[str, Optional[str]] = {r["program_id"]: r["code"] for r in tables["prompts"]}
    best_prompt: Optional[str] = prompt_code.get(best_row["prompt_id"])

    hist_best: List[float] = [
        r["best_fitness"]
        for r in tables["fitness_history"]
        if r["best_fitness"] is not None and r["epoch"] == max(epochs)
    ]
    if hist_best and _float(best_prog.fitness) != max(hist_best):
        report["issues"].append(
            f"best program fitness {best_prog.fitness} != final best_fit_hist {max(hist_best)}"
        )

    cfg_name, cfg_text, models = _read_config(run_dir)
    if cfg_text is None:
        report["issues"].append("config yaml missing")
    calls = tables["llm_calls"]
    tables["runs"].append(
        {
            **keys,
            "num_islands": len(sol_dbs),
            "final_epoch": max(epochs),
            "best_fitness": _float(best_prog.fitness),
            "best_island": best_row["island_found"],
            "best_program_id": _str(best_prog.id),
            "best_code": _str(best_prog.code),
            "best_prompt": _str(best_prompt),
            "models": models,
            "num_programs": len(tables["programs"]),
            "num_prompts": len(tables["prompts"]),
            "num_llm_calls": len(calls),
            "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
            "completion_tokens": sum(c["completion_tokens"] or 0 for c in calls),
            "num_errors": len(tables["errors"]),
            "config_file": cfg_name,
            "config_yaml": cfg_text,
        }
    )
    report["rows"] = {name: len(rows) for name, rows in tables.items()}
    report["islands"] = len(sol_dbs)
    report["program_copies"] = sum(len(db.programs) for _, db in sol_dbs)
    return tables, report


# ---------------------------------------------------------------------------
# Shard writing
# ---------------------------------------------------------------------------


def _shard_name(problem: str) -> str:
    return problem.replace("/", "__")


def export_shard(exp_dir: Path, out_dir: Path, problem: str, run_dirs: List[Path]) -> List[Dict]:
    """Converts all runs of one problem and writes one Parquet file per table."""
    shard: str = _shard_name(problem)
    writers: Dict[str, pq.ParquetWriter] = {}
    tmp_paths: Dict[str, Path] = {}
    reports: List[Dict] = []
    try:
        for name, schema in SCHEMAS.items():
            path = out_dir / "data" / name / f"{shard}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_paths[name] = path.with_suffix(".parquet.tmp")
            writers[name] = pq.ParquetWriter(
                tmp_paths[name], schema, compression="zstd", compression_level=9
            )
        for run_dir in run_dirs:
            t0 = time.time()
            tables, report = convert_run(exp_dir, run_dir)
            for name, rows in tables.items():
                if rows:
                    table = pa.Table.from_pylist(rows, schema=SCHEMAS[name])
                    writers[name].write_table(
                        table, row_group_size=_ROW_GROUP_SIZE.get(name, _DEFAULT_ROW_GROUP_SIZE)
                    )
            report["seconds"] = round(time.time() - t0, 1)
            reports.append(report)
    finally:
        for w in writers.values():
            w.close()
    for name, tmp in tmp_paths.items():
        tmp.replace(tmp.with_suffix(""))
    return reports


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_dir", type=Path, default=Path("experiments"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--problems", nargs="*", default=None, help="Only export these problem paths."
    )
    parser.add_argument(
        "--max_runs", type=int, default=None, help="Only export the first N runs per problem."
    )
    args = parser.parse_args(argv)

    exp_dir: Path = args.exp_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    runs = discover_runs(exp_dir)
    if args.problems:
        runs = {k: v for k, v in runs.items() if k in set(args.problems)}
    if args.max_runs:
        runs = {k: v[: args.max_runs] for k, v in runs.items()}
    print(f"{sum(map(len, runs.values()))} runs across {len(runs)} problems", flush=True)

    all_reports: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(export_shard, exp_dir, out_dir, problem, run_dirs): problem
            for problem, run_dirs in runs.items()
        }
        for fut in as_completed(futures):
            problem = futures[fut]
            reports = fut.result()
            all_reports.extend(reports)
            n_issues = sum(len(r["issues"]) for r in reports)
            print(f"  done {problem}: {len(reports)} runs, {n_issues} issues", flush=True)

    all_reports.sort(key=lambda r: r["run_id"])
    totals: Dict[str, int] = defaultdict(int)
    for r in all_reports:
        for name, n in r["rows"].items():
            totals[name] += n
    manifest = {
        "num_runs": len(all_reports),
        "rows": dict(totals),
        "runs_with_issues": [r for r in all_reports if r["issues"]],
        "runs": all_reports,
    }
    (out_dir / "export_report.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"num_runs": len(all_reports), "rows": dict(totals)}, indent=1))
    print(f"runs with issues: {len(manifest['runs_with_issues'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
