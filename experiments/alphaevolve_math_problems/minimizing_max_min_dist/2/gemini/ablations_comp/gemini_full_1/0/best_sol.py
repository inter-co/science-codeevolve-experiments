# EVOLVE-BLOCK-START
import numpy as np
# from scipy.spatial.distance import pdist # Removed: no longer used
from scipy.optimize import minimize
from numba import jit # Import Numba for JIT compilation
import time # Required for time-budgeted SA
import random # Required for random.randint in SA

# --- Numba-optimized helper function from inspirations ---
@jit(nopython=True, fastmath=True)
def _calculate_min_max_sq_numba(points: np.ndarray) -> (float, float):
    """
    Numba-optimized function to calculate minimum and maximum *squared* distances
    among all pairs of points. This avoids creating a full distance matrix
    and directly computes min/max, reducing memory and speeding up.
    It calculates squared distances first to minimize sqrt calls.
    """
    n = points.shape[0]
    if n < 2:
        return 0.0, 0.0

    dmin_sq = np.finfo(np.float64).max # Initialize with a very large number for squared distance
    dmax_sq = 0.0

    for i in range(n):
        for j in range(i + 1, n): # Only check each pair once
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist_sq = dx*dx + dy*dy # Calculate squared distance

            if dist_sq < dmin_sq:
                dmin_sq = dist_sq
            if dist_sq > dmax_sq:
                dmax_sq = dist_sq
    
    # Robustness fix from inspiration: handle case where all points are identical (or effectively 1 point)
    if dmin_sq == np.finfo(np.float64).max: # If no distances were calculated (e.g., n=0 or n=1)
        dmin_sq = 0.0 # Treat as 0 distance

    return dmin_sq, dmax_sq # Return squared distances

# Helper function for calculating fitness (min/max distance ratio)
def _calculate_ratio(points: np.ndarray, squared: bool = False) -> float:
    """
    Calculates the ratio of the minimum distance to the maximum distance
    (or its square) among all pairs of points. Utilizes a Numba-optimized helper.

    Args:
        points: A numpy array of shape (N, D) where N is the number of points
                and D is the dimensionality.
        squared: If True, returns the squared ratio (dmin/dmax)^2. Otherwise, returns dmin/dmax.

    Returns:
        The d_min / d_max ratio (or its square). Returns 0.0 if unable to compute.
    """
    if points.shape[0] < 2:
        return 0.0

    dmin_sq, dmax_sq = _calculate_min_max_sq_numba(points)

    # Handle cases where dmax_sq could be very small or zero, for floating point robustness.
    # Using a small threshold instead of exact zero for robustness.
    if dmax_sq <= 1e-18: # Using a smaller threshold for squared distances
        return 0.0
    
    ratio_sq = dmin_sq / dmax_sq
    
    if squared:
        return ratio_sq
    else:
        return np.sqrt(ratio_sq) # Take sqrt only if non-squared ratio is requested

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Optimizes 16 points in 2D by combining Simulated Annealing (SA) for global
    search with L-BFGS-B for local refinement. This hybrid approach leverages the
    strengths of both methods to find a high-quality solution, with Numba acceleration.

    The points are constrained to the unit square [0,1] x [0,1].

    Returns:
        points: np.ndarray of shape (16,2) with the best point configuration found.
    """
    n_points = 16
    n_dims = 2
    fixed_seed = 42 # For reproducibility
    np.random.seed(fixed_seed)
    random.seed(fixed_seed) # Seed python's random module too

    # --- Simulated Annealing Parameters (tuned based on Inspiration Program 1 for slightly better performance) ---
    TIME_LIMIT_SECONDS = 175       # Use most of the 180s budget for SA
    COOLING_RATE = 0.999998        # Very slow cooling for a long, time-budgeted run
    INITIAL_TEMP = 0.002           # Lower initial temp to encourage focused search from the strong grid start (as in Inspiration 1)
    PERTURBATION_SCALE_START = 0.018 # Smaller base perturbation for finer initial exploration (as in Inspiration 1)

    # --- Initial Solution: A slightly perturbed 4x4 grid (a strong start) ---
    grid_side = int(np.sqrt(n_points))
    linear_coords = np.linspace(0, 1, grid_side)
    initial_grid = np.array([[x, y] for x in linear_coords for y in linear_coords])
    current_points = initial_grid + (np.random.rand(n_points, n_dims) - 0.5) * 0.01
    current_points = np.clip(current_points, 0, 1)

    current_ratio = _calculate_ratio(current_points) # Use non-squared ratio for SA
    best_points = current_points.copy()
    best_ratio = current_ratio
    temperature = INITIAL_TEMP
    
    start_time = time.time() # Start timer for time-budgeted SA

    # --- Main Simulated Annealing Loop (Time-Budgeted) ---
    while (time.time() - start_time) < TIME_LIMIT_SECONDS:
        # Perturbation magnitude decreases as the system "cools"
        # This formula ensures perturbation decreases with temperature,
        # and has a minimal value (1e-7) even at very low temperatures for fine adjustments.
        perturbation_scale = PERTURBATION_SCALE_START * (temperature / INITIAL_TEMP) + 1e-7

        point_idx = random.randint(0, n_points - 1) # Use python's random for single int
        new_points = current_points.copy()
        
        # Use Gaussian perturbation for finer local steps, as seen in inspirations
        perturbation = np.random.randn(n_dims) * perturbation_scale
        new_points[point_idx] += perturbation
        new_points[point_idx] = np.clip(new_points[point_idx], 0, 1) # Ensure points stay within [0,1]

        new_ratio = _calculate_ratio(new_points) # Use non-squared ratio for SA
        
        # SA acceptance criterion for maximization (cost is -ratio)
        delta_cost = current_ratio - new_ratio

        if delta_cost < 0: # New solution is better, always accept
            current_points = new_points
            current_ratio = new_ratio
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
        elif temperature > 1e-12: # Accept worse solution with a probability (use a very small threshold for temp)
            acceptance_prob = np.exp(-delta_cost / temperature)
            if random.random() < acceptance_prob: # Use python's random for acceptance prob
                current_points = new_points
                current_ratio = new_ratio
        
        temperature *= COOLING_RATE

        # Stop if temperature is effectively zero, preventing wasted cycles
        if temperature < 1e-12: # A very small temperature means acceptance prob of worse solutions is almost zero.
            break
    
    # --- Local Optimization (Post-processing using L-BFGS-B) ---
    # Refine the best solution found by SA with a gradient-based method.
    def local_objective(flat_points):
        # We minimize the negative of the squared ratio to maximize the actual ratio.
        # This can provide a smoother objective landscape for the gradient-based optimizer.
        return -_calculate_ratio(flat_points.reshape((n_points, n_dims)), squared=True)

    bounds = [(0, 1)] * (n_points * n_dims)
    initial_guess_flat = best_points.flatten()

    local_result = minimize(
        local_objective,
        initial_guess_flat,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-9} # More generous refinement options from inspirations
    )

    final_points = local_result.x.reshape((n_points, n_dims))
    final_ratio = np.sqrt(-local_result.fun)  # Convert minimized negative squared ratio back to actual ratio

    # Return the best of the SA result and the locally optimized result
    if final_ratio > best_ratio:
        return final_points
    else:
        return best_points


# EVOLVE-BLOCK-END
