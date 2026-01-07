# EVOLVE-BLOCK-START
import numpy as np
import numba
from numba import njit
import time
from scipy.stats.qmc import Sobol
from scipy.optimize import minimize # NEW: For local refinement

@njit(cache=True)
def _get_min_max_pairs(points: np.ndarray):
    """
    Efficiently finds the minimum and maximum pairwise distances and the indices
    of the points that form them. Works with squared distances internally to
    avoid unnecessary sqrt calls in the loop. Inspired by all inspiration programs.
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
        # If all points are coincident, return 0.0 for distances and default indices
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

# NEW: Objective function for scipy.optimize.minimize
def _objective_for_minimize(flat_points: np.ndarray, n: int, d: int) -> float:
    """
    Wrapper for _calculate_objective to be used with scipy.optimize.minimize.
    Minimizes the negative ratio to maximize the actual ratio.
    """
    points = flat_points.reshape((n, d))
    return -_calculate_objective(points)

@njit(cache=True)
def _calculate_targeted_forces(points, min_indices, max_indices, min_dist, repulsion_strength, attraction_strength, repulsion_power):
    """
    Calculates forces based on a targeted strategy (from Inspiration Program 2):
    1. A repulsive force on the closest pair to increase dmin.
    2. An attractive (spring) force on the farthest pair to decrease dmax.
    """
    forces = np.zeros_like(points)
    
    # 1. Repulsive force on the closest pair to push them apart
    i_min, j_min = min_indices
    vec_min = points[i_min] - points[j_min]
    if min_dist > 1e-9: # Avoid division by zero for unit vector and force calculation
        # Force magnitude proportional to 1/r^(repulsion_power + 1).
        # If repulsion_power=2, this is 1/r^3, similar to Coulomb for 1/r potential.
        force_mag_rep = repulsion_strength / (min_dist**(repulsion_power + 1))
        unit_vec_min = vec_min / min_dist
        force_rep = unit_vec_min * force_mag_rep
        forces[i_min] += force_rep
        forces[j_min] -= force_rep

    # 2. Attractive force on the farthest pair to pull them together
    i_max, j_max = max_indices
    vec_max = points[i_max] - points[j_max]
    force_attr_vec = attraction_strength * vec_max # Spring-like attraction
    forces[i_max] -= force_attr_vec
    forces[j_max] += force_attr_vec
    
    return forces

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Optimizes 14 points in a unit cube [0,1]³ by directly targeting the
    minimum and maximum distances, inspired by Inspiration Program 2. It applies a
    repulsive force to the closest pair and an attractive force to the farthest
    pair in each step, using a cosine annealing learning rate schedule.
    A final local optimization step using scipy.optimize.minimize is added for refinement,
    following the pattern seen in all inspiration programs.
    """
    n = 14
    d = 3

    # Use Sobol sequence for a more uniform initial distribution (from inspirations)
    sampler = Sobol(d=d, seed=42)
    points = sampler.random(n=n)

    # Parameters for the targeted force-directed simulation (tuned based on inspirations)
    initial_learning_rate = 0.01
    final_learning_rate = 1e-6
    n_iterations = 10_000_000 # Increased iterations for deeper optimization
    repulsion_strength = 0.02 # Increased repulsion strength to aggressively push d_min
    repulsion_power = 2.0     # Force ~ 1/r^(power+1) -> 1/r^3
    attraction_strength = 0.05

    best_points = points.copy()
    best_ratio = _calculate_objective(points)

    start_time = time.time()
    # Allocate most of the time budget to the simulation, reserving some for local optimization.
    # Total budget 360s. Local optimization typically takes a few seconds to tens of seconds.
    simulation_time_limit = 320 # seconds

    for iteration in range(n_iterations):
        if time.time() - start_time > simulation_time_limit:
            break

        # Cosine annealing schedule for learning rate
        # Ensure t_normalized does not exceed 1.0 even if loop breaks early.
        t_normalized = min(iteration / n_iterations, 1.0) 
        learning_rate = final_learning_rate + 0.5 * (initial_learning_rate - final_learning_rate) * (1 + np.cos(np.pi * t_normalized))

        # Get current min/max distances and the pairs that cause them
        min_dist, max_dist, min_indices, max_indices = _get_min_max_pairs(points)

        # Calculate forces targeting only the min and max pairs
        forces = _calculate_targeted_forces(points, min_indices, max_indices, min_dist, repulsion_strength, attraction_strength, repulsion_power)

        # Update point positions
        points += learning_rate * forces

        # Enforce unit cube boundary
        points = np.clip(points, 0, 1)
        
        # Check for improvement and update best_points *after* position update and clipping.
        # This ensures best_points always correspond to valid, bounded positions.
        if max_dist > 1e-9: # Only evaluate if max_dist is meaningful (not all points coincident)
            current_ratio = _calculate_objective(points) # Re-calculate for updated points
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy() # Store points that correspond to this best ratio
    
    # --- Local Refinement Phase (NEW - Inspired by all programs) ---
    # Use the best points from the simulation as the initial guess for local optimization.
    x0_local = best_points.flatten()

    # Define bounds for the unit cube [0,1]^3 for each coordinate.
    bounds = [(0.0, 1.0)] * (n * d)

    # Perform local optimization using L-BFGS-B, which is suitable for bounded problems.
    # It's a gradient-based method, often fast and effective when starting from a good point.
    # maxiter and ftol are tuned to allow for precise refinement within the time budget.
    result_local = minimize(
        fun=_objective_for_minimize, # Objective function (minimizes negative ratio)
        x0=x0_local,                 # Initial guess from simulation
        args=(n, d),                 # Additional arguments for objective function
        method='L-BFGS-B',           # Optimization method
        bounds=bounds,               # Box constraints for point coordinates
        options={'maxiter': 2500, 'ftol': 1e-9, 'disp': False} # Increased maxiter for precision
    )

    # Extract the optimized points. L-BFGS-B respects bounds.
    final_points_flat = result_local.x
    final_points = final_points_flat.reshape(n, d)
    
    # Final check: compare the ratio from local optimization with the best from simulation.
    # Return the configuration that yielded the highest ratio.
    final_ratio_optimized = _calculate_objective(final_points)
    if final_ratio_optimized > best_ratio:
        return final_points
    else:
        # If local optimization did not improve, return the best configuration from the simulation.
        return best_points


# EVOLVE-BLOCK-END
