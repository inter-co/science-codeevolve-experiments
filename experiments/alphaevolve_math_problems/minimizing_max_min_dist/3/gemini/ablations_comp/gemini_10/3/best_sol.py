# EVOLVE-BLOCK-START
import numpy as np
from numba import njit # Use njit directly for clarity
import time # New import for time tracking

# --- JIT-COMPILED HELPER FUNCTIONS (Synthesized from Inspirations 2 & 3) ---
from scipy.optimize import minimize # New import for local optimization

# Wrapper for scipy.optimize.minimize
def _objective_for_scipy_minimize(x_flat, n_points, dim):
    """Wrapper for the objective function for scipy.optimize.minimize."""
    points = x_flat.reshape((n_points, dim))
    # minimize seeks to minimize, so we return the negative of our ratio
    return -_calculate_objective(points)

@njit(cache=True)
def _calculate_objective(points: np.ndarray) -> float:
    """Calculates the objective ratio dmin/dmax using the efficient _get_min_max_pairs helper."""
    min_dist, max_dist, _, _ = _get_min_max_pairs(points)
    if max_dist < 1e-12: # Robust check for coincident points
        return 0.0
    return min_dist / max_dist

# --- Force-Directed Helper Functions (Adopted from Inspiration Program 3) ---
@njit(cache=True)
def _get_min_max_pairs(points: np.ndarray):
    """
    Efficiently finds the minimum and maximum pairwise distances and the indices
    of the points that form them. Uses squared distances internally to avoid sqrt.
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
def _calculate_targeted_forces(points, min_indices, max_indices, min_dist, repulsion_strength, attraction_strength, repulsion_power):
    """
    Calculates forces based on a targeted strategy:
    1. A repulsive force on the closest pair to increase dmin.
    2. An attractive (spring) force on the farthest pair to decrease dmax.
    Includes a special case for nearly coincident points (from Insp. 3).
    """
    forces = np.zeros_like(points)
    
    # 1. Repulsive force on the closest pair to push them apart
    i_min, j_min = min_indices
    vec_min = points[i_min] - points[j_min]
    
    # Handle nearly-coincident points to prevent division by zero and provide a strong kick
    if min_dist < 1e-7:
        # Apply a large, fixed perturbation if points are too close
        fixed_perturbation_vec = np.array([1e-5, 1e-5, 1e-5], dtype=points.dtype)
        forces[i_min] += fixed_perturbation_vec * 100
        forces[j_min] -= fixed_perturbation_vec * 100
    elif min_dist > 1e-9: # Normal repulsion for non-coincident points
        force_mag_rep = repulsion_strength / (min_dist**(repulsion_power + 1))
        unit_vec_min = vec_min / min_dist
        force_rep = unit_vec_min * force_mag_rep
        forces[i_min] += force_rep
        forces[j_min] -= force_rep

    # 2. Attractive force on the farthest pair to pull them together
    i_max, j_max = max_indices
    vec_max = points[i_max] - points[j_max]
    force_attr_vec = attraction_strength * vec_max
    forces[i_max] -= force_attr_vec
    forces[j_max] += force_attr_vec
    
    return forces

# --- Main Optimization Function (Adopted from Inspiration Program 3) ---
def min_max_dist_dim3_14() -> np.ndarray:
    """
    Optimizes 14 points in a unit cube [0,1]³ using a hybrid two-phase strategy
    synthesizing the best features from Inspiration Program 3.
    1. A force-directed simulation with annealed forces, momentum, and perturbations.
    2. A local optimization (L-BFGS-B) for fine-tuning the result.
    """
    n = 14
    d = 3

    np.random.seed(42)
    points = np.random.rand(n, d)

    # --- Hyperparameters from the best-performing inspiration (Insp. 3) ---
    time_limit = 350
    simulation_time_limit = time_limit * 0.9 # Allocate 90% time to global search
    
    # Phase 1: Force-Directed Global Search
    n_iterations = 5_000_000
    initial_learning_rate = 0.015
    final_learning_rate = 1e-7

    # Annealing schedule for force strengths
    initial_rep_strength = 0.020
    final_rep_strength = 0.010
    initial_attr_strength = 0.050
    final_attr_strength = 0.080
    repulsion_power = 2.0
    
    # Momentum and Perturbation
    damping_factor = 0.95
    initial_perturbation_magnitude = 0.01
    perturbation_decay = 0.999997

    # --- Initialization ---
    best_points = points.copy()
    best_ratio = _calculate_objective(points)
    
    velocity = np.zeros_like(points)
    current_perturbation_magnitude = initial_perturbation_magnitude
    start_time = time.time()

    # --- Phase 1: Main Optimization Loop ---
    for iteration in range(n_iterations):
        if time.time() - start_time > simulation_time_limit:
            break

        t_normalized = iteration / n_iterations
        
        # Anneal learning rate and force strengths
        learning_rate = final_learning_rate + 0.5 * (initial_learning_rate - final_learning_rate) * (1 + np.cos(np.pi * t_normalized))
        current_repulsion_strength = initial_rep_strength * (1 - t_normalized) + final_rep_strength * t_normalized
        current_attraction_strength = initial_attr_strength * (1 - t_normalized) + final_attr_strength * t_normalized

        min_dist, max_dist, min_indices, max_indices = _get_min_max_pairs(points)
        
        # Check for improvement on every iteration (cost is negligible)
        if max_dist > 1e-9:
            current_ratio = min_dist / max_dist
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()

        forces = _calculate_targeted_forces(points, min_indices, max_indices, min_dist, current_repulsion_strength, current_attraction_strength, repulsion_power)

        # Update velocity with momentum
        velocity = damping_factor * velocity + learning_rate * forces
        # Add a decaying random kick to escape local minima
        random_kick = (np.random.rand(n, d) - 0.5) * 2 * current_perturbation_magnitude
        points += velocity + random_kick

        current_perturbation_magnitude = max(current_perturbation_magnitude * perturbation_decay, 1e-7)
        points = np.clip(points, 0, 1)

    # Check final simulation point for any last-minute improvement
    final_sim_ratio = _calculate_objective(points)
    if final_sim_ratio > best_ratio:
        best_points = points.copy()
        best_ratio = final_sim_ratio

    # --- Phase 2: Local Optimization using L-BFGS-B ---
    if time.time() - start_time < time_limit - 5: # Ensure at least 5s for local opt
        try:
            bounds = [(0, 1) for _ in range(n * d)]
            x0_local = best_points.flatten()
            
            local_opt_result = minimize(
                _objective_for_scipy_minimize,
                x0_local,
                args=(n, d),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-9, 'maxfun': 5000}
            )
            
            if local_opt_result.success:
                optimized_points = local_opt_result.x.reshape((n, d))
                optimized_points = np.clip(optimized_points, 0, 1)
                local_opt_ratio = _calculate_objective(optimized_points)

                if local_opt_ratio > best_ratio:
                    best_points = optimized_points.copy()
        except Exception:
            # If local opt fails, we still have the best result from phase 1
            pass

    return best_points


# EVOLVE-BLOCK-END
