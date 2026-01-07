# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the circle packing problem on unit square
# for N circles (1 to 32).
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import time
import numpy as np
import json
import sys
import os
import importlib
import warnings
import concurrent.futures

BENCHMARKS = {
    1: 0.5,
    2: 0.586,
    3: 0.796,
    4: 1.007,
    5: 1.104,
    6: 1.203,
    7: 1.307,
    8: 1.424,
    9: 1.525,
    10: 1.592,
    11: 1.681,
    12: 1.766,
    13: 1.830,
    14: 1.906,
    15: 1.981,
    16: 2.054,
    17: 2.112,
    18: 2.179,
    19: 2.237,
    20: 2.302,
    21: 2.363,
    22: 2.421,
    23: 2.479,
    24: 2.531,
    25: 2.588,
    26: 2.636,
    27: 2.686,
    28: 2.738,
    29: 2.791,
    30: 2.843,
    31: 2.890,
    32: 2.938,
}

TOL = 1e-6

def validate_packing_radii(radii: np.ndarray) -> None:
    n = len(radii)
    for i in range(n):
        if radii[i] < 0:
            raise ValueError(f"Circle {i} has negative radius {radii[i]}")
        elif np.isnan(radii[i]):
            raise ValueError(f"Circle {i} has nan radius")


def validate_packing_unit_square_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if (x - r < -tol) or (x + r > 1 + tol) or (y - r < -tol) or (y + r > 1 + tol):
            raise ValueError(
                f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"
            )


def validate_packing_overlap_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((circles[i, :2] - circles[j, :2]) ** 2))
            if dist < circles[i, 2] + circles[j, 2] - tol:
                raise ValueError(
                    f"Circles {i} and {j} overlap: dist={dist}, r1+r2={circles[i,2]+circles[j,2]}"
                )

def eval_single_case(program_path: str, num_circles: int):
    """
    Worker function executed in a separate process. 
    It imports the module locally to ensure thread/process safety.
    """
    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]

    if program_dir not in sys.path:
        sys.path.insert(0, program_dir)

    try:
        program = importlib.__import__(module_name)
        importlib.reload(program)

        if not hasattr(program, 'circle_packing_square'):
             raise AttributeError(f"Module {module_name} must implement function 'circle_packing_square(num_circles: int)'")

        start_time = time.time()
        circles = program.circle_packing_square(num_circles)
        end_time = time.time()
        eval_time = end_time - start_time
        
        if not isinstance(circles, np.ndarray):
            circles = np.array(circles)
        if circles.shape != (num_circles, 3):
            raise ValueError(
                f"Invalid shapes: circles = {circles.shape}, expected {(num_circles,3)}"
            )
        validate_packing_radii(circles[:, -1])
        validate_packing_overlap_wtol(circles, TOL)
        validate_packing_unit_square_wtol(circles, TOL)
        radii_sum = np.sum(circles[:, -1])

        return {
                "benchmark_ratio": float(radii_sum / BENCHMARKS[num_circles]),
                "eval_time": float(eval_time),
               }
    except Exception as err:
        raise RuntimeError(f"Error with N = {num_circles}: {str(err)}")
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)


def evaluate(program_path: str, results_path: str = None) -> None:
    avg_benchmark_ratio = 0
    avg_eval_time = 0
    
    max_workers = os.cpu_count()

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_n = {
                executor.submit(eval_single_case, program_path, n): n 
                for n in BENCHMARKS.keys()
            }
            
            for future in concurrent.futures.as_completed(future_to_n):
                ret = future.result()
                avg_benchmark_ratio += ret["benchmark_ratio"]
                avg_eval_time += ret["eval_time"]

        avg_benchmark_ratio = avg_benchmark_ratio/len(BENCHMARKS)
        avg_eval_time = avg_eval_time/len(BENCHMARKS)
            
    except Exception as err:
        raise err

    with open(results_path, "w") as f:
        json.dump(
            {
                "avg_benchmark_ratio": float(avg_benchmark_ratio),
                "avg_eval_time": float(avg_eval_time)
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]

    evaluate(program_path, results_path)
