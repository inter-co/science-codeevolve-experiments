# EVOLVE-BLOCK-START
import numpy as np
import random
import time
from numba import jit # Import Numba for JIT compilation
from scipy.optimize import minimize # For local refinement

# Global random seed for reproducibility to ensure consistent results
RANDOM_SEED = 42

@jit(nopython=True, fastmath=True)
def _calculate_min_max_numba(points: np.ndarray) -> (float, float):
    """
    Numba-optimized function to calculate minimum and maximum distances
    without creating a full distance matrix, significantly speeding up the calculation.
    (Adapted from Inspiration Program 3)
    """
    n = points.shape[0]
    dmin_sq = np.finfo(np.float64).max
    dmax_sq = 0.0

    if n < 2: # Handle cases with less than 2 points early
        return 0.0, 0.0

    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist_sq = dx*dx + dy*dy
            # Handle identical points or very close points to avoid numerical issues
            if dist_sq < 1e-18: 
                dist_sq = 0.0
            
            if dist_sq < dmin_sq:
                dmin_sq = dist_sq
            if dist_sq > dmax_sq:
                dmax_sq = dist_sq
    
    # If all points are identical, dmin_sq remains max_float or becomes 0 if dist_sq was 0.0
    # If dmax_sq is 0, it means all distances were 0, so dmin_sq should also be 0.
    if dmax_sq == 0.0:
        dmin_sq = 0.0

    return np.sqrt(dmin_sq), np.sqrt(dmax_sq)

def _calculate_min_max_ratio(points: np.ndarray) -> float:
    """
    Calculates the d_min / d_max ratio using a Numba-optimized helper.
    (Adapted from Inspiration Program 3)
    """
    if points.shape[0] < 2:
        return 0.0

    dmin, dmax = _calculate_min_max_numba(points)

    if dmax <= 1e-9: # Prevent division by zero or very small dmax
        return 0.0
    
    return dmin / dmax

def _objective_function_for_minimizer(coords_flat: np.ndarray, n: int, d: int) -> float:
    """
    Objective function for scipy.optimize.minimize.
    It takes a flattened array of coordinates, reshapes them into N D points,
    calculates the ratio of minimum to maximum pairwise distance, and returns
    its negative value (as scipy.optimize functions perform minimization).
    """
    points = coords_flat.reshape((n, d))
    return -_calculate_min_max_ratio(points)


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Uses a Simulated Annealing algorithm to find an arrangement of 16 points
    in a 2D unit square that maximizes the ratio of the minimum to maximum
    distance between any pair of points.

    The algorithm starts from a perturbed 4x4 grid and iteratively improves it.
    The hyperparameters (temperature, cooling rate, perturbation scale) are tuned
    to refine this strong initial configuration towards the optimum.

    Returns:
        points: np.ndarray of shape (16,2) with the optimized point coordinates.
    """
    n_points = 16
    n_dims = 2
    
    # Seed all random number generators for full reproducibility
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # --- SA Parameters (tuned for fine-tuning a grid within a time budget, adapted from Inspiration 3) ---
    TIME_LIMIT_SECONDS = 175       # Use most of the 180s budget for SA
    COOLING_RATE = 0.999998        # Very slow cooling for a long, time-budgeted run (Inspiration 3)
    INITIAL_TEMP = 0.005           # Lower temp, as we start from a high-quality grid (Inspiration 3)
    PERTURBATION_SCALE_START = 0.03 # Smaller base perturbation for refinement (Inspiration 3)

    # --- Initial Solution: A slightly perturbed 4x4 grid (a strong start, adapted from target & Inspiration 3) ---
    grid_side = int(np.sqrt(n_points))
    linear_coords = np.linspace(0, 1, grid_side)
    initial_grid = np.array([[x, y] for x in linear_coords for y in linear_coords])
    
    # Add a small random perturbation to break perfect symmetry and aid exploration
    current_points = initial_grid + (np.random.rand(n_points, n_dims) - 0.5) * 0.01
    current_points = np.clip(current_points, 0, 1)

    current_ratio = _calculate_min_max_ratio(current_points)
    best_points = current_points.copy()
    best_ratio = current_ratio
    temperature = INITIAL_TEMP
    
    start_time = time.time()

    # --- Main Simulated Annealing Loop (Time-Budgeted, adapted from Inspiration 3) ---
    # Loop while within time limit and temperature is above a threshold
    while (time.time() - start_time) < TIME_LIMIT_SECONDS and temperature > 1e-12:
        # Perturbation magnitude decreases as the system "cools"
        # The factor (temperature / INITIAL_TEMP) makes perturbation scale proportional to current temperature
        perturbation_scale = PERTURBATION_SCALE_START * (temperature / INITIAL_TEMP) + 1e-7

        point_idx = random.randint(0, n_points - 1) # Use random.randint for consistency with Inspiration 3
        new_points = current_points.copy()
        
        # Use Gaussian perturbation for finer local steps (from Inspiration 3)
        perturbation = np.random.randn(n_dims) * perturbation_scale
        new_points[point_idx] += perturbation
        new_points[point_idx] = np.clip(new_points[point_idx], 0, 1)

        new_ratio = _calculate_min_max_ratio(new_points)
        
        # SA acceptance criterion for maximization (cost is -ratio)
        # Accept if new solution is better (new_ratio > current_ratio) or with probability if worse.
        if new_ratio > current_ratio: # New solution is better, always accept
            current_points = new_points
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
        else: # Accept worse solution with a probability
            # Probability of acceptance for a worse solution (new_ratio < current_ratio)
            # is exp((new_ratio - current_ratio) / temperature)
            acceptance_prob = np.exp((new_ratio - current_ratio) / temperature)
            if random.random() < acceptance_prob:
                current_points = new_points
                current_ratio = new_ratio
        
        temperature *= COOLING_RATE
    
    # --- Local Optimization (Post-processing using L-BFGS-B from Inspiration 3) ---
    # The objective function for scipy.optimize.minimize should return a scalar to minimize.
    # We want to maximize the ratio, so we minimize its negative.
    bounds = [(0, 1)] * (n_points * n_dims)
    initial_guess_flat = best_points.flatten()

    local_result = minimize(
        _objective_function_for_minimizer, # Use the new objective function for local optimization
        initial_guess_flat,
        method='L-BFGS-B', # L-BFGS-B is a good choice for bounded problems and often performs well
        bounds=bounds,
        args=(n_points, n_dims), # Pass n_points and n_dims to the objective function
        options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10} # Generous refinement
    )

    final_points = local_result.x.reshape((n_points, n_dims))
    final_ratio = -local_result.fun # Restore to positive ratio

    # Return the best points, either from SA or after local refinement.
    # It's possible SA found a better solution than what L-BFGS-B could improve upon
    # if L-BFGS-B got stuck in a very shallow local minimum or a flat region,
    # or if the non-differentiable nature of the objective function posed issues.
    if final_ratio > best_ratio:
        return final_points
    else:
        return best_points


# EVOLVE-BLOCK-END
