# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import scipy.signal
import time
import random
import numba
from scipy.optimize import differential_evolution # New import for Differential Evolution

# --- Global Parameters ---
RANDOM_SEED = 42
NUM_GAUSSIANS = 10    # Number of Gaussian basis functions to approximate f(x)
TARGET_N_STEPS = 4000 # Number of steps for the f_values array (high resolution)
TIME_LIMIT = 170      # Maximum execution time for optimization (slightly less than 180s)

# --- Seed for reproducibility ---
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Global variable to track the start time for the DE callback
# This needs to be declared globally because the callback itself is a nested function or needs global access.
start_time_de = time.time() 

# --- Numba-accelerated helper function for L2-norm squared ---
@numba.jit(nopython=True, cache=True) # Removed parallel=True as DE worker parallelism is higher level
def _compute_l2_sq_norm_numba(g_values: np.ndarray, dt_g: float) -> float:
    """
    Computes the L2-norm squared of g_values using piecewise linear integration,
    accelerated with Numba.
    """
    g_l2_sq = 0.0
    for i in range(len(g_values) - 1): # Changed prange to range for compatibility with DE's internal parallelism
        y1 = g_values[i]
        y2 = g_values[i+1]
        g_l2_sq += (dt_g / 3.0) * (y1**2 + y1*y2 + y2**2)
    return g_l2_sq

# --- Core Fitness Function (Calculates C2 for a high-res array) ---
def _calculate_c2(f_values_raw: np.ndarray) -> float:
    """
    Calculates the C2 constant for a given sequence of step heights f_values.
    This function is heavily optimized using techniques from the inspiration programs,
    including fftconvolve and Numba-accelerated L2 norm calculation.
    """
    # Ensure all step heights are non-negative, as per problem constraint.
    f_values = np.maximum(f_values_raw, 0.0)

    # Handle edge cases for empty or all-zero f_values.
    if f_values.size == 0 or np.sum(f_values) < 1e-12:
        return 0.0

    # 1. Autoconvolution g = f*f using FFT for O(N log N) performance.
    g_values = scipy.signal.fftconvolve(f_values, f_values, mode='full')
    g_len = len(g_values)

    # Handle edge cases for g_values (e.g., if f was empty or all zeros).
    if g_len <= 1 or np.sum(g_values) < 1e-12:
        return 0.0

    # The autoconvolution g is defined on [-1/2, 1/2], which has a total width of 1.
    # If g_values has g_len points, the spacing (dt_g) between points for piecewise linear
    # integration over a total width of 1 is 1.0 / (g_len - 1).
    dt_g = 1.0 / (g_len - 1)
    
    # 2. Compute ||g||₂² (L2-norm squared) using Numba-accelerated function.
    g_l2_sq = _compute_l2_sq_norm_numba(g_values, dt_g)
    
    # 3. Compute ||g||₁ (L1-norm), using the problem's specified approximation.
    # Since f_values are non-negative, g_values will also be non-negative.
    g_l1 = np.sum(np.abs(g_values)) / (g_len + 1)
    
    # 4. Compute ||g||∞ (Infinity-norm).
    g_linf = np.max(np.abs(g_values))

    # Handle cases where denominators might be zero or extremely small.
    if g_l1 < 1e-12 or g_linf < 1e-12:
        return 0.0
    
    # Calculate C2 constant.
    c2 = g_l2_sq / (g_l1 * g_linf)
    return c2 if np.isfinite(c2) else 0.0

# --- Helper to generate f_values from Gaussian parameters ---
def _generate_f_from_params(params: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Generates a step function (f_values) by summing Gaussian basis functions.
    `params` is a flat array: [A1, mu1, sigma1, A2, mu2, sigma2, ...]
    """
    K = NUM_GAUSSIANS
    x_start = -0.25
    x_end = 0.25
    step_width = (x_end - x_start) / n_steps
    # x_centers represent the center of each step interval for evaluating the continuous function
    x_centers = np.linspace(x_start + step_width / 2, x_end - step_width / 2, n_steps)

    f_values_generated = np.zeros(n_steps, dtype=np.float64)
    
    for i in range(K):
        A = params[i * 3 + 0]
        mu = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        
        # Add the contribution of the current Gaussian
        f_values_generated += A * np.exp(-((x_centers - mu)**2) / (2 * sigma**2))
    
    # Ensure all step heights are non-negative
    return np.maximum(0.0, f_values_generated)

# --- Objective function for Differential Evolution ---
# DE minimizes, so we return -C2
def _objective_de(params: np.ndarray) -> float:
    f_values = _generate_f_from_params(params, TARGET_N_STEPS)
    c2 = _calculate_c2(f_values)
    return -c2 # Return negative C2 for minimization

# --- Callback for Differential Evolution to handle time limit and track best result ---
def _de_callback_factory(best_tracker: dict):
    def _de_callback_with_best_tracker(xk, convergence=0.0):
        global start_time_de 
        
        # Evaluate current best xk from DE population
        current_c2 = -_objective_de(xk) # _objective_de returns -C2, so negate to get C2
        
        if current_c2 > best_tracker['c2_score']:
            best_tracker['c2_score'] = current_c2
            best_tracker['params'] = xk.copy()

        if time.time() - start_time_de > TIME_LIMIT:
            raise StopIteration("Time limit exceeded for Differential Evolution.")
    return _de_callback_with_best_tracker

def construct_function() -> list[float]:
    """
    Optimizes a step-function by finding optimal parameters for Gaussian basis functions
    using Differential Evolution to maximize the C2 constant.
    This implementation leverages the parametric representation and DE from Inspiration 2
    for better C2 values within the given time budget.
    """
    global start_time_de # Declare global for modification within this function

    # Initialize a mutable dictionary to store the best parameters found
    best_params_found_in_de = {'params': None, 'c2_score': -np.inf}
    
    # Create the callback function using the factory
    de_callback = _de_callback_factory(best_params_found_in_de)

    # Define bounds for the parameters: [A, mu, sigma] for each of NUM_GAUSSIANS
    # A (Amplitude): [0.0, 5.0] - Allowing peaks higher than 1.0 (as f can be > 1)
    # mu (Mean): [-0.25, 0.25] - Within the function's interval [-1/4, 1/4]
    # sigma (Standard Deviation): [0.005, 0.15] - Reasonable width for Gaussians
    # These bounds are crucial for effective search.
    bounds = []
    for _ in range(NUM_GAUSSIANS):
        bounds.append((0.0, 5.0))   
        bounds.append((-0.25, 0.25)) 
        bounds.append((0.003, 0.2)) # Adopted wider sigma bounds from Inspiration 3 for potentially sharper peaks

    start_time_de = time.time() # Initialize/reset start time for the DE run
    
    try:
        # Differential Evolution optimization
        result = differential_evolution(
            _objective_de,             # The objective function to minimize (-C2)
            bounds,                    # Parameter bounds
            strategy='best1bin',       # One of the recommended strategies for DE
            maxiter=2000,              # Max iterations (will be limited by TIME_LIMIT)
            popsize=15,                # Population size factor (actual pop size = popsize * len(bounds))
            tol=0.001,                 # Relative tolerance for convergence
            mutation=(0.5, 1.0),       # Differential weight factor (F)
            recombination=0.7,         # Crossover probability (Cr)
            seed=RANDOM_SEED,          # Seed for reproducibility
            callback=de_callback,      # Custom callback for time limit and best tracking
            disp=False,                # Do not print progress to console
            workers=-1                 # Use all available CPU cores for parallel evaluation
        )
        
        # If DE finishes without StopIteration, check if its final result is better
        # than what was captured by the callback (callback tracks best *individual* seen in population at each iter).
        # `result.fun` holds the minimum objective value found (-C2).
        final_c2 = -result.fun
        if final_c2 > best_params_found_in_de['c2_score']:
            best_params_found_in_de['c2_score'] = final_c2
            best_params_found_in_de['params'] = result.x.copy()

    except StopIteration:
        # print(f"Differential Evolution stopped early due to time limit ({TIME_LIMIT}s).") # Commented out for submission
        pass # Suppress print for submission
    
    # Generate the f_values from the best parameters found
    if best_params_found_in_de['params'] is not None:
        best_f_values = _generate_f_from_params(best_params_found_in_de['params'], TARGET_N_STEPS)
    else:
        # Fallback: if no valid parameters were found (e.g., due to extremely short time limit)
        # This should ideally not happen with the given time limits.
        # print("Warning: No best parameters found by Differential Evolution. Returning a default f_values.") # Commented out for submission
        best_f_values = np.full(TARGET_N_STEPS, 0.5) 

    return best_f_values.tolist()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
