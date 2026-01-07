# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
import time
from scipy.signal import fftconvolve
import numba
from scipy.optimize import differential_evolution, minimize

# --- Global Parameters (Inspired by Insp. 2 & 3) ---
RANDOM_SEED = 42
SEQUENCE_LENGTH = 30000      # High resolution for the final step function
OPTIM_SEQUENCE_LENGTH = 5000 # Lower resolution for faster evaluation during optimization
N_BASIS_FUNCTIONS = 12       # Increased number of Gaussian basis functions for more flexibility
TOTAL_TIME_LIMIT = 178       # Total time budget for the entire process
DE_TIME_LIMIT = 160          # Time allocated for the global Differential Evolution phase

# --- Seed for reproducibility ---
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --- Global variables for optimizer state ---
start_time_de_global = 0
best_solution_tracker = {'params': None, 'c2_score': -np.inf}

# --- C2 Calculation (Optimized with FFT and Numba, adapted from target) ---
@numba.jit(nopython=True, fastmath=True, cache=True)
def _compute_c2_from_g(g_values: np.ndarray, dt_g: float) -> float:
    """Numba-accelerated helper to compute C2 from the autoconvolution array g."""
    g_len = len(g_values)
    if g_len <= 1:
        return 0.0
    
    # 1. Compute ||g||₂² (L2-norm squared) via piecewise linear integration.
    y1 = g_values[:-1]
    y2 = g_values[1:]
    g_l2_sq = np.sum(y1**2 + y1*y2 + y2**2) * (dt_g / 3.0)

    # 2. Compute ||g||₁ (L1-norm) approximation. g is non-negative.
    g_l1 = np.sum(g_values) / (g_len + 1)
    
    # 3. Compute ||g||∞ (Infinity-norm).
    g_linf = np.max(g_values)

    if g_l1 < 1e-12 or g_linf < 1e-12:
        return 0.0
    
    c2 = g_l2_sq / (g_l1 * g_linf)
    return c2 if np.isfinite(c2) else 0.0

def compute_c2(f_values: np.ndarray) -> float:
    """Calculates C2 using fftconvolve and a Numba helper."""
    n_f = len(f_values)
    if n_f == 0 or np.sum(f_values) < 1e-12:
        return 0.0

    # The step width 'dx' for f on [-1/4, 1/4] is also the integration step 'dt_g' for g.
    dx = 0.5 / n_f
    g_values = fftconvolve(f_values, f_values, mode='full')
    
    return _compute_c2_from_g(g_values, dx)

# --- Parametric Function Definition (Inspired by Insp. 2) ---
def _generate_f_from_params(params: np.ndarray, n_steps: int) -> np.ndarray:
    """Generates a step function f by summing Gaussian basis functions."""
    x_grid = np.linspace(-0.25, 0.25, n_steps, endpoint=False)
    f_values = np.zeros(n_steps, dtype=np.float64)
    
    for i in range(N_BASIS_FUNCTIONS):
        amp, mu, sigma = params[i*3 : (i+1)*3]
        f_values += amp * np.exp(-((x_grid - mu)**2) / (2 * sigma**2))
    
    return np.maximum(0.0, f_values)

# --- Objective & Callback for Optimizer (Inspired by Insp. 2) ---
def _objective_de(params: np.ndarray) -> float:
    """Objective function for DE/L-BFGS-B, returns -C2 for minimization."""
    f_values = _generate_f_from_params(params, OPTIM_SEQUENCE_LENGTH)
    return -compute_c2(f_values)

def _de_callback(xk, convergence):
    """Callback to enforce time limit and track the best solution found by DE."""
    global start_time_de_global, best_solution_tracker
    
    current_c2 = -_objective_de(xk)
    if current_c2 > best_solution_tracker['c2_score']:
        best_solution_tracker['c2_score'] = current_c2
        best_solution_tracker['params'] = xk.copy()

    if time.time() - start_time_de_global > DE_TIME_LIMIT:
        raise StopIteration("DE time limit reached.")

# --- Main Optimization Driver (Hybrid DE + L-BFGS-B from Insp. 2) ---
def construct_function() -> list[float]:
    """
    Optimizes a step-function using a hybrid Differential Evolution + L-BFGS-B approach.
    """
    global start_time_de_global, best_solution_tracker

    bounds = []
    for _ in range(N_BASIS_FUNCTIONS):
        bounds.extend([
            (0.0, 5.0),      # Amplitude
            (-0.3, 0.3),     # Mean (slightly wider than [-0.25, 0.25])
            (0.005, 0.2)     # Std Dev
        ])

    start_time_de_global = time.time()
    best_solution_tracker = {'params': None, 'c2_score': -np.inf}

    # --- Phase 1: Global Search with Differential Evolution ---
    try:
        de_result = differential_evolution(
            _objective_de, bounds, strategy='best1bin', maxiter=2000,
            popsize=20, tol=1e-5, recombination=0.7, seed=RANDOM_SEED,
            callback=_de_callback, disp=False, workers=-1
        )
        final_de_c2 = -de_result.fun
        if final_de_c2 > best_solution_tracker['c2_score']:
            best_solution_tracker['c2_score'] = final_de_c2
            best_solution_tracker['params'] = de_result.x.copy()
    except StopIteration:
        pass # Expected exit via time-limit callback
    except Exception:
        pass # Catch other potential errors during optimization

    # --- Phase 2: Local Search Refinement with L-BFGS-B ---
    best_params = best_solution_tracker['params']
    time_left = TOTAL_TIME_LIMIT - (time.time() - start_time_de_global)

    if best_params is not None and time_left > 15:
        try:
            local_res = minimize(
                _objective_de, x0=best_params, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7, 'maxfun': 15000}
            )
            local_c2 = -local_res.fun
            if local_c2 > best_solution_tracker['c2_score']:
                best_params = local_res.x.copy()
        except Exception:
            pass # Local search might fail; we still have the DE result.

    # --- Final Step: Generate high-resolution f_values from the best parameters ---
    if best_params is not None:
        final_f_values = _generate_f_from_params(best_params, SEQUENCE_LENGTH)
    else:
        final_f_values = np.zeros(SEQUENCE_LENGTH)

    return final_f_values.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
