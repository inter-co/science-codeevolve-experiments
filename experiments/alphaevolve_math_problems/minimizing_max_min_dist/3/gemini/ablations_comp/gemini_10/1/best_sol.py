# EVOLVE-BLOCK-START
import numpy as np
from numba import njit
from scipy.optimize import minimize
import time

# --- Numba-jitted Helper Functions for Performance (from Inspirations 1 & 2) ---

@njit(cache=True)
def _get_min_max_pairs(points: np.ndarray):
    """
    Efficiently finds the minimum and maximum pairwise distances and the indices
    of the points that form them. Uses squared distances internally for speed.
    """
    n = points.shape[0]
    min_dist_sq = np.inf
    max_dist_sq = 0.0
    min_indices = (0, 1)
    max_indices = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = np.sum((points[i] - points[j])**2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                min_indices = (i, j)
            if dist_sq > max_dist_sq:
                max_dist_sq = dist_sq
                max_indices = (i, j)
    # Handle cases where all points are coincident
    if max_dist_sq <= 1e-12:
        return 0.0, 0.0, min_indices, max_indices
    return np.sqrt(min_dist_sq), np.sqrt(max_dist_sq), min_indices, max_indices

@njit(cache=True)
def _calculate_objective(points: np.ndarray):
    """Calculates the objective ratio dmin/dmax."""
    min_dist, max_dist, _, _ = _get_min_max_pairs(points)
    if max_dist == 0.0:
        return 0.0
    return min_dist / max_dist

@njit(cache=True)
def _calculate_targeted_forces(points, min_indices, max_indices, min_dist, repulsion_strength, attraction_strength, repulsion_power):
    """
    Calculates targeted forces based on the min and max distance pairs.
    - A repulsive force pushes the two closest points apart.
    - An attractive force pulls the two farthest points together.
    - Includes a special case for near-coincident points to prevent division by zero.
    """
    forces = np.zeros_like(points)
    i_min, j_min = min_indices
    vec_min = points[i_min] - points[j_min]

    # Repulsive force for the closest pair
    if min_dist < 1e-6:
        # If points are nearly on top of each other, apply a strong, fixed kick
        # to separate them, avoiding extreme forces from division by a tiny min_dist.
        fixed_kick = np.array([1e-5, 1e-5, 1e-5], dtype=points.dtype)
        forces[i_min] += fixed_kick * 100
        forces[j_min] -= fixed_kick * 100
    elif min_dist > 1e-9: # Standard inverse-square-like repulsion
        force_mag_rep = repulsion_strength / (min_dist**(repulsion_power + 1))
        force_rep = (vec_min / min_dist) * force_mag_rep
        forces[i_min] += force_rep
        forces[j_min] -= force_rep

    # Attractive force for the farthest pair
    i_max, j_max = max_indices
    vec_max = points[i_max] - points[j_max]
    # Simple linear attraction force
    force_attr_vec = attraction_strength * vec_max
    forces[i_max] -= force_attr_vec
    forces[j_max] += force_attr_vec
    
    return forces

# --- Wrapper for Scipy Optimizer ---
def _objective_for_scipy_minimize(x_flat: np.ndarray, n: int, d: int) -> float:
    """Objective function for scipy.minimize, which minimizes, so we return -ratio."""
    points = x_flat.reshape(n, d)
    return -_calculate_objective(points)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Optimizes 14 points in a unit cube [0,1]³ using a hybrid strategy inspired by top-performing solutions.
    This approach combines a global search with a local refinement for high-quality results.

    1.  **Force-Directed Simulation (FDS)**: A powerful global search method. Points are treated as
        particles. The two closest points repel, and the two farthest attract. This is combined
        with momentum and an annealed random "kick" to explore the solution space and avoid local minima.
    2.  **Local Optimization (L-BFGS-B)**: After the FDS finds a promising configuration, a gradient-based
        local optimizer (`L-BFGS-B`) is used to precisely fine-tune the point positions to the nearest optimum.
    """
    n = 14
    d = 3
    np.random.seed(42)
    # Start with a random configuration in the unit cube
    points = np.random.rand(n, d)

    # --- Hyperparameters (inspired by Inspirations 1 & 2) ---
    n_iterations = 5_000_000 # A high number, but we use a time limit
    initial_learning_rate = 0.01
    final_learning_rate = 1e-6
    repulsion_strength = 0.01
    repulsion_power = 2.0
    attraction_strength = 0.05
    damping_factor = 0.95 # Momentum term
    initial_perturbation_magnitude = 0.01 # For stochastic kicks
    perturbation_decay = 0.999997 # Annealing for kicks
    time_limit = 350 # Seconds
    fds_time_limit_ratio = 0.9 # Allocate 90% of time to global FDS search

    # --- Phase 1: Force-Directed Global Search ---
    best_points = points.copy()
    best_ratio = _calculate_objective(points)
    velocity = np.zeros_like(points)
    current_perturbation_magnitude = initial_perturbation_magnitude
    start_time = time.time()
    
    for iteration in range(n_iterations):
        # Stop if we're out of time for this phase
        if time.time() - start_time > time_limit * fds_time_limit_ratio:
            break

        # Cosine annealing schedule for the learning rate (smooth decay)
        t_normalized = iteration / n_iterations
        learning_rate = final_learning_rate + 0.5 * (initial_learning_rate - final_learning_rate) * (1 + np.cos(np.pi * t_normalized))

        # Get the current state (min/max pairs)
        min_dist, max_dist, min_indices, max_indices = _get_min_max_pairs(points)
        
        # Calculate forces based on the current state
        forces = _calculate_targeted_forces(points, min_indices, max_indices, min_dist, repulsion_strength, attraction_strength, repulsion_power)

        # Update point positions using momentum, forces, and a random kick
        velocity = damping_factor * velocity + learning_rate * forces
        random_kick = (np.random.rand(n, d) - 0.5) * 2 * current_perturbation_magnitude
        points += velocity + random_kick
        
        # Anneal the magnitude of the random kick
        current_perturbation_magnitude *= perturbation_decay
        
        # Enforce the unit cube boundary condition
        points = np.clip(points, 0, 1)

        # Check for improvement and update the best solution found so far
        if max_dist > 1e-12:
            current_ratio = min_dist / max_dist
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()

    # --- Phase 2: Local Optimization with L-BFGS-B ---
    bounds_cube = [(0, 1)] * (n * d)
    x0_local = best_points.flatten()

    try:
        # Use a reasonable number of iterations for the local search
        refined_result = minimize(
            fun=_objective_for_scipy_minimize,
            x0=x0_local,
            args=(n, d),
            method='L-BFGS-B',
            bounds=bounds_cube,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'eps': 1e-9}
        )
        optimized_points_local = np.clip(refined_result.x.reshape(n, d), 0, 1)
        local_opt_ratio = _calculate_objective(optimized_points_local)
        
        # Only accept the refined result if it's strictly better
        if local_opt_ratio > best_ratio:
            best_points = optimized_points_local
            
    except Exception:
        # If L-BFGS-B fails for any reason, we gracefully fall back
        # to the best solution found during the global FDS phase.
        pass

    return best_points


# EVOLVE-BLOCK-END
