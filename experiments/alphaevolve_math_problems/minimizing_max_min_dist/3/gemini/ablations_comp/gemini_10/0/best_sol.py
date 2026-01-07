# EVOLVE-BLOCK-START
import numpy as np
import numba
from numba import njit
import time


# EVOLVE-BLOCK-START

@njit(cache=True)
def _get_min_max_pairs(points: np.ndarray):
    """
    Efficiently finds the minimum and maximum pairwise distances and the indices
    of the points that form them. Works with squared distances internally to
    avoid unnecessary sqrt calls in the loop. Inspired by Inspiration Program 2.
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
    
    if max_dist_sq == 0:
        return 0.0, 0.0, min_indices, max_indices

    min_dist = np.sqrt(min_dist_sq)
    max_dist = np.sqrt(max_dist_sq)
    
    return min_dist, max_dist, min_indices, max_indices

@njit(cache=True)
def _calculate_objective(points: np.ndarray):
    """Calculates the objective ratio dmin/dmax using the efficient helper."""
    min_dist, max_dist, _, _ = _get_min_max_pairs(points)
    if max_dist == 0.0:
        return 0.0
    return min_dist / max_dist

@njit(cache=True)
def _calculate_targeted_forces(points, min_indices, max_indices, min_dist, repulsion_strength, attraction_strength, repulsion_power):
    """
    Calculates robust targeted forces, inspired by Insp. 1.
    Handles near-coincident points with an aggressive kick.
    """
    forces = np.zeros_like(points)
    
    # Repulsive force for the closest pair
    i_min, j_min = min_indices
    vec_min = points[i_min] - points[j_min]
    if min_dist < 1e-6:
        # Aggressive fixed perturbation for near-coincident points (from Insp. 1)
        fixed_kick = np.array([1e-5, 1e-5, 1e-5], dtype=points.dtype)
        forces[i_min] += fixed_kick * 100
        forces[j_min] -= fixed_kick * 100
    elif min_dist > 1e-9:
        force_mag_rep = repulsion_strength / (min_dist**(repulsion_power + 1))
        force_rep = (vec_min / min_dist) * force_mag_rep
        forces[i_min] += force_rep
        forces[j_min] -= force_rep

    # Attractive force for the farthest pair
    i_max, j_max = max_indices
    vec_max = points[i_max] - points[j_max]
    force_attr_vec = attraction_strength * vec_max
    forces[i_max] -= force_attr_vec
    forces[j_max] += force_attr_vec
    
    return forces

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Optimizes 14 points in a unit cube [0,1]³ using a hybrid strategy inspired by top performers:
    1. A targeted force-directed simulation (FDS) with momentum and stochastic kicks for global exploration.
    2. A local optimization (L-BFGS-B) for fine-tuning the best solution found by FDS.
    """
    from scipy.optimize import minimize # Import locally as required

    def _objective_for_scipy_minimize(x_flat: np.ndarray, n: int, d: int) -> float:
        """Objective function for scipy.minimize, returns negative ratio."""
        points = x_flat.reshape(n, d)
        return -_calculate_objective(points)

    n = 14
    d = 3
    np.random.seed(42)
    points = np.random.rand(n, d)

    # --- Hyperparameters from Inspiration 1 ---
    n_iterations = 5_000_000
    initial_learning_rate = 0.01
    final_learning_rate = 1e-6
    repulsion_strength = 0.01
    repulsion_power = 2.0
    attraction_strength = 0.05
    damping_factor = 0.95
    initial_perturbation_magnitude = 0.01
    perturbation_decay = 0.999997
    time_limit = 350
    fds_time_limit_ratio = 0.9 # Allocate 90% of time to global search

    # --- Phase 1: Force-Directed Global Search with Momentum and Annealing ---
    best_points = points.copy()
    best_ratio = _calculate_objective(points)
    velocity = np.zeros_like(points)
    current_perturbation_magnitude = initial_perturbation_magnitude
    start_time = time.time()
    
    for iteration in range(n_iterations):
        if time.time() - start_time > time_limit * fds_time_limit_ratio:
            break

        # Cosine annealing for learning rate
        t_normalized = iteration / n_iterations
        learning_rate = final_learning_rate + 0.5 * (initial_learning_rate - final_learning_rate) * (1 + np.cos(np.pi * t_normalized))

        min_dist, max_dist, min_indices, max_indices = _get_min_max_pairs(points)
        
        # Use the improved force calculation
        forces = _calculate_targeted_forces(points, min_indices, max_indices, min_dist, repulsion_strength, attraction_strength, repulsion_power)

        # Update with momentum and stochastic kick (from Insp. 1)
        velocity = damping_factor * velocity + learning_rate * forces
        random_kick = (np.random.rand(n, d) - 0.5) * 2 * current_perturbation_magnitude
        points += velocity + random_kick
        
        # Decay the random kick magnitude
        current_perturbation_magnitude *= perturbation_decay
        
        # Enforce unit cube boundary
        points = np.clip(points, 0, 1)

        # Check for improvement using pre-calculated values
        if max_dist > 1e-12:
            current_ratio = min_dist / max_dist
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()

    # --- Phase 2: Local Optimization with L-BFGS-B (from Insp. 1) ---
    bounds_cube = [(0, 1)] * (n * d)
    x0_local = best_points.flatten()

    try:
        remaining_time = time_limit - (time.time() - start_time)
        # scipy.optimize.minimize does not have a timeout, so we cap iterations
        max_local_iter = 2000 if remaining_time > 20 else 500

        refined_result = minimize(
            fun=_objective_for_scipy_minimize,
            x0=x0_local,
            args=(n, d),
            method='L-BFGS-B',
            bounds=bounds_cube,
            options={'maxiter': max_local_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'eps': 1e-9}
        )
        optimized_points_local = np.clip(refined_result.x.reshape(n, d), 0, 1)
        local_opt_ratio = _calculate_objective(optimized_points_local)
        
        if local_opt_ratio > best_ratio:
            best_points = optimized_points_local
            
    except Exception:
        # If L-BFGS-B fails, gracefully return the best result from Phase 1
        pass

    return best_points


# EVOLVE-BLOCK-END
