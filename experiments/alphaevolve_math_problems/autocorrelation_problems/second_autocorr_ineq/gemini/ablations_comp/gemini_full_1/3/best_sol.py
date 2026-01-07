# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import scipy.signal
import time
import random
import numba
from scipy.optimize import differential_evolution, minimize

# --- Configuration ---
RANDOM_SEED = 42
# A key change: systematically test different `n` values instead of fixing one.
N_CANDIDATES = [4000, 7000, 10000] # Search over a range of resolutions
# Inspired by other programs, 8 Gaussians seems a good balance for a 24-dim search space
NUM_GAUSSIANS = 8
TOTAL_TIME_LIMIT = 175 # Total time budget, slightly increased for meta-search

# --- Seed for reproducibility ---
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- Global state for parallel-safe optimization ---
# This global variable will be updated by the main loop to communicate the
# current `n` to the objective function, which is required for parallel DE.
CURRENT_OPTIM_N_STEPS = 0
# Global timer for callbacks
_callback_start_time = 0

# --- Numba-accelerated L2-norm squared calculation (unchanged) ---
@numba.jit(nopython=True, cache=True)
def _compute_l2_sq_norm_numba(g_values: np.ndarray, h: float) -> float:
    l2_sq_norm = 0.0
    for i in range(len(g_values) - 1):
        y1 = g_values[i]
        y2 = g_values[i+1]
        l2_sq_norm += (h / 3.0) * (y1**2 + y1*y2 + y2**2)
    return l2_sq_norm

# --- Core Fitness Function (unchanged) ---
def _calculate_c2(f_values_arr: np.ndarray) -> float:
    if f_values_arr.size < 2 or np.sum(f_values_arr) < 1e-12:
        return 0.0

    g_values = scipy.signal.fftconvolve(f_values_arr, f_values_arr, mode='full')
    M_g = len(g_values)
    
    if M_g <= 1: return 0.0

    l1_norm = np.sum(g_values) / (M_g + 1)
    linf_norm = np.max(g_values)
    
    if l1_norm < 1e-12 or linf_norm < 1e-12:
        return 0.0
    
    h = 1.0 / (M_g - 1)
    l2_sq_norm = _compute_l2_sq_norm_numba(g_values, h)
    
    denominator = l1_norm * linf_norm
    if denominator < 1e-12:
        return 0.0
    
    c2 = l2_sq_norm / denominator
    return c2 if np.isfinite(c2) else 0.0

# --- Parametric Function Generator (unchanged) ---
def _generate_f_from_params(params: np.ndarray, n_steps: int) -> np.ndarray:
    x_centers = np.linspace(-0.25, 0.25, n_steps, endpoint=False)
    f_values = np.zeros(n_steps, dtype=np.float64)
    for i in range(NUM_GAUSSIANS):
        A, mu, sigma = params[i*3], params[i*3+1], params[i*3+2]
        f_values += A * np.exp(-((x_centers - mu)**2) / (2 * sigma**2))
    return np.maximum(0.0, f_values)

# --- Objective Function for Optimizers (to be minimized) ---
# It now uses the global CURRENT_OPTIM_N_STEPS to be parallel-safe.
def _objective_func(params: np.ndarray) -> float:
    f_values = _generate_f_from_params(params, CURRENT_OPTIM_N_STEPS)
    c2 = _calculate_c2(f_values)
    # Return large penalty for invalid results to guide optimizer
    return -c2 if c2 > 0 else 1.0

# --- Callback for DE Time Limit and Best Solution Tracking ---
def _de_callback_factory(best_tracker: dict, time_limit: float):
    def _de_callback(xk, convergence=0.0):
        global _callback_start_time
        # Evaluate C2 for the current best individual in the population
        current_c2 = -_objective_func(xk)
        if current_c2 > best_tracker['c2_score']:
            best_tracker['c2_score'] = current_c2
            best_tracker['params'] = xk.copy()
        if time.time() - _callback_start_time > time_limit:
            raise StopIteration("Time limit for this DE run exceeded.")
    return _de_callback

def construct_function() -> list[float]:
    """
    Implements a meta-optimization strategy to find the best number of steps `n`.
    For each candidate `n`, it runs a hybrid DE (global) + L-BFGS-B (local) search.
    The best solution across all `n` values is returned.
    """
    global CURRENT_OPTIM_N_STEPS, _callback_start_time
    
    best_overall_solution = {'params': None, 'c2_score': -np.inf, 'n': 0}
    
    time_per_n = (TOTAL_TIME_LIMIT - 5) / len(N_CANDIDATES)
    de_time_per_n = time_per_n * 0.9 # Allocate 90% of time to global DE search
    
    bounds = []
    for _ in range(NUM_GAUSSIANS):
        bounds.append((0.0, 5.0))      # Amplitude
        bounds.append((-0.3, 0.3))     # Mean (slightly wider)
        bounds.append((0.01, 0.2))     # Sigma

    for n_steps in N_CANDIDATES:
        CURRENT_OPTIM_N_STEPS = n_steps
        
        # --- Phase 1: Global Search with DE for current n ---
        current_run_best = {'params': None, 'c2_score': -np.inf}
        _callback_start_time = time.time()
        try:
            de_callback = _de_callback_factory(current_run_best, de_time_per_n)
            de_result = differential_evolution(
                _objective_func,
                bounds,
                strategy='best1bin',
                maxiter=2000, # Will be stopped by time limit
                popsize=20,   # Increased population size for better exploration
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=RANDOM_SEED,
                callback=de_callback,
                disp=False,
                workers=-1
            )
            final_de_c2 = -de_result.fun
            if final_de_c2 > current_run_best['c2_score']:
                current_run_best['c2_score'] = final_de_c2
                current_run_best['params'] = de_result.x
        except StopIteration:
            pass # Expected exit via callback

        # --- Phase 2: Local Search Refinement (L-BFGS-B) ---
        time_left_for_n = time_per_n - (time.time() - _callback_start_time)
        if current_run_best['params'] is not None and time_left_for_n > 3:
            try:
                # The objective for minimize can be a lambda as it does not run in parallel
                local_obj = lambda p: -_calculate_c2(_generate_f_from_params(p, n_steps))
                
                local_search_result = minimize(
                    local_obj,
                    x0=current_run_best['params'],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-7}
                )
                local_c2 = -local_search_result.fun
                if local_c2 > current_run_best['c2_score']:
                    current_run_best['params'] = local_search_result.x
                    current_run_best['c2_score'] = local_c2
            except Exception:
                pass

        # --- Update overall best solution ---
        if current_run_best['c2_score'] > best_overall_solution['c2_score']:
            best_overall_solution.update({
                'params': current_run_best['params'],
                'c2_score': current_run_best['c2_score'],
                'n': n_steps
            })

    # --- Final Step: Generate function with the optimal n found ---
    if best_overall_solution['params'] is not None:
        best_f_values = _generate_f_from_params(best_overall_solution['params'], best_overall_solution['n'])
    else:
        # Fallback if no solution was found
        best_f_values = np.full(N_CANDIDATES[0], 0.5)

    return best_f_values.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
