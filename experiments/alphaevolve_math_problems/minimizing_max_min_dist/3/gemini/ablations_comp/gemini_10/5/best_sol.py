# EVOLVE-BLOCK-START
import numpy as np
from numba import jit
import time
from scipy.stats.qmc import Sobol
from scipy.optimize import minimize

# --- Core JIT-Compiled Helper Functions (from Inspiration) ---
@jit(nopython=True, cache=True)
def _get_min_max_pairs(points: np.ndarray):
    """
    Efficiently finds the minimum and maximum squared pairwise distances and the
    indices of the points that form them.
    """
    n = points.shape[0]
    min_dist_sq = np.inf
    max_dist_sq = 0.0
    min_indices = (0, 1)
    max_indices = (0, 1)

    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = np.sum((points[i] - points[j])**2)
            if dist_sq < 1e-18: continue # Ignore coincident points
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                min_indices = (i, j)
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                max_indices = (i, j)
    
    return min_dist_sq, max_dist_sq, min_indices, max_indices

@jit(nopython=True, cache=True)
def _calculate_targeted_forces(points, min_indices, max_indices, min_dist_sq, repulsion_strength, attraction_strength):
    """
    Calculates forces based on a targeted strategy:
    1. A repulsive force on the closest pair to increase d_min.
    2. An attractive (spring) force on the farthest pair to decrease d_max.
    """
    forces = np.zeros_like(points)
    
    # 1. Repulsive force on the closest pair
    i_min, j_min = min_indices
    vec_min = points[i_min] - points[j_min]
    if min_dist_sq > 1e-18:
        # Force is stronger for smaller distances, proportional to 1/d^2
        force_mag_rep = repulsion_strength / min_dist_sq
        force_rep = (vec_min / np.sqrt(min_dist_sq)) * force_mag_rep
        forces[i_min] += force_rep
        forces[j_min] -= force_rep

    # 2. Attractive force on the farthest pair
    i_max, j_max = max_indices
    vec_max = points[i_max] - points[j_max]
    force_attr_vec = attraction_strength * vec_max
    forces[i_max] -= force_attr_vec
    forces[j_max] += force_attr_vec
    
    return forces

def _objective_for_scipy(flat_points: np.ndarray, n: int, d: int) -> float:
    """
    Objective function for local optimization: returns negative dmin/dmax ratio.
    """
    points = flat_points.reshape(n, d)
    min_dist_sq, max_dist_sq, _, _ = _get_min_max_pairs(points)
    if max_dist_sq < 1e-18: return 1.0 # Penalize collapsed configurations
    return -np.sqrt(min_dist_sq / max_dist_sq)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Optimizes 14 points in a unit cube using a hybrid approach:
    1. A targeted force-directed simulation (global search).
    2. A local gradient-based optimization (L-BFGS-B) to refine the result.
    """
    n_points = 14
    dims = 3
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    # --- Phase 1: Force-Directed Simulation ---
    # Parameters
    initial_learning_rate = 0.01
    final_learning_rate = 1e-7
    n_iterations = 8_000_000 # High number, but will be stopped by time limit
    repulsion_strength = 0.01
    attraction_strength = 0.05
    force_directed_time_limit = 250 # seconds

    # Initialization using a high-quality quasi-random Sobol sequence
    sampler = Sobol(d=dims, seed=RANDOM_SEED)
    points = sampler.random(n=n_points)

    min_dist_sq, max_dist_sq, _, _ = _get_min_max_pairs(points)
    best_ratio_sq = min_dist_sq / max_dist_sq if max_dist_sq > 0 else 0.0
    best_points = points.copy()
    
    start_time = time.time()
    for iteration in range(n_iterations):
        if time.time() - start_time > force_directed_time_limit:
            break

        # Cosine annealing for learning rate
        t_normalized = iteration / n_iterations
        learning_rate = final_learning_rate + 0.5 * (initial_learning_rate - final_learning_rate) * (1 + np.cos(np.pi * t_normalized))

        min_dist_sq, max_dist_sq, min_indices, max_indices = _get_min_max_pairs(points)

        # Update best found configuration
        if max_dist_sq > 1e-18:
            current_ratio_sq = min_dist_sq / max_dist_sq
            if current_ratio_sq > best_ratio_sq:
                best_ratio_sq = current_ratio_sq
                best_points = points.copy()

        forces = _calculate_targeted_forces(points, min_indices, max_indices, min_dist_sq, repulsion_strength, attraction_strength)
        points += learning_rate * forces
        points = np.clip(points, 0, 1)

    # --- Phase 2: Local Optimization ---
    bounds = [(0, 1)] * (n_points * dims)
    x0 = best_points.flatten()

    local_opt_result = minimize(
        fun=_objective_for_scipy,
        x0=x0,
        args=(n_points, dims),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False}
    )
    
    optimized_points = local_opt_result.x.reshape(n_points, dims)

    # Final check: return the best of the two phases
    final_ratio = -_objective_for_scipy(local_opt_result.x, n_points, dims)
    if final_ratio > np.sqrt(best_ratio_sq):
        return optimized_points
    else:
        return best_points


# EVOLVE-BLOCK-END
