# EVOLVE-BLOCK-START
import numpy as np
import numpy as np
from scipy.spatial.distance import pdist # Removed squareform for efficiency
import time # Added for time budgeting
from scipy.optimize import minimize # ADDED: For local refinement phase


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance
    using a Simulated Annealing (SA) metaheuristic followed by L-BFGS-B local refinement.
    This hybrid approach, inspired by high-performing solutions, combines global exploration
    with local exploitation to achieve a high-quality solution.

    Returns
        points: np.ndarray of shape (16,2) containing the optimized (x,y) coordinates.
    """

    n = 16  # Number of points
    d = 2   # Dimensions
    np.random.seed(42) # Ensure reproducibility for stochastic components

    def _get_min_max_ratio(points: np.ndarray) -> float:
        """
        Calculates the ratio of the minimum distance to the maximum distance
        among all pairs of points.
        """
        if points.shape[0] < 2:
            return 0.0
        
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0.0
        
        non_zero_distances = distances[distances > 1e-9] 
        
        if len(non_zero_distances) == 0:
            return 0.0
        
        dmin = np.min(non_zero_distances)
        dmax = np.max(distances)

        return dmin / dmax

    def _objective_func_sa(points: np.ndarray) -> float:
        """
        Objective function for Simulated Annealing: minimize the inverse of the ratio.
        """
        ratio = _get_min_max_ratio(points)
        if ratio <= 1e-9:
            return np.inf
        return 1.0 / ratio

    # --- PART 1: GLOBAL SEARCH (SIMULATED ANNEALING) ---
    # --- Initial Configuration: Perturbed 4x4 Grid (a strong starting point) ---
    grid_side = int(np.sqrt(n))
    grid_spacing = 1.0 / (grid_side + 1)
    x_coords = np.linspace(grid_spacing, 1.0 - grid_spacing, grid_side)
    y_coords = np.linspace(grid_spacing, 1.0 - grid_spacing, grid_side)
    xv, yv = np.meshgrid(x_coords, y_coords)
    current_points = np.vstack([xv.ravel(), yv.ravel()]).T
    current_points += (np.random.rand(n, d) - 0.5) * grid_spacing * 0.1
    current_points = np.clip(current_points, 0.0, 1.0)

    best_points = np.copy(current_points)
    best_objective = _objective_func_sa(current_points)
    current_objective = best_objective

    time_limit = 58
    start_time = time.time()

    # SA parameters tuned to be gentler on the initial grid structure,
    # preventing it from "melting" into a poor configuration. A lower
    # starting temperature and smaller perturbations help preserve the good
    # initial state while still allowing for exploration.
    T_start = 0.05
    T_end = 1e-7
    num_iterations = 800_000
    cooling_rate = (T_end / T_start)**(1.0 / num_iterations)
    initial_perturb_magnitude = 0.05
    current_T = T_start

    # --- Main SA Loop ---
    for i in range(num_iterations):
        if (time.time() - start_time) >= time_limit:
            break
        if current_T < T_end:
            break

        perturbation_noise = (np.random.rand(n, d) - 0.5) * 2 * initial_perturb_magnitude * (current_T / T_start)
        temp_points = np.clip(current_points + perturbation_noise, 0, 1)
        new_objective = _objective_func_sa(temp_points)

        if new_objective < current_objective:
            current_points = np.copy(temp_points)
            current_objective = new_objective
            if new_objective < best_objective:
                best_points = np.copy(temp_points)
                best_objective = new_objective
        else:
            delta_objective = new_objective - current_objective
            if current_T > 0 and delta_objective > 1e-9: 
                acceptance_probability = np.exp(-delta_objective / current_T)
                if np.random.rand() < acceptance_probability:
                    current_points = np.copy(temp_points)
                    current_objective = new_objective
        
        current_T *= cooling_rate

    # --- PART 2: TWO-STAGE LOCAL REFINEMENT (L-BFGS-B) ---
    # This approach is inspired by the high-performing solutions, which first maximize
    # the minimum distance, then refine the dmin/dmax ratio.

    # Objective for Stage 1: Maximize minimum distance (maximin problem)
    def local_objective_dmin(flat_points: np.ndarray) -> float:
        points = flat_points.reshape((n, d))
        distances = pdist(points)
        
        if distances.size == 0:
            return 0.0 # Should not happen for n=16, but for robustness.

        # Filter out near-zero distances to find the true dmin among distinct points.
        non_zero_distances = distances[distances > 1e-9]
        if len(non_zero_distances) == 0:
            # If all points are collapsed, return a large penalty to push them apart.
            return 1e12 
        
        # We want to MAXIMIZE dmin, so the optimizer must MINIMIZE -dmin.
        dmin = np.min(non_zero_distances)
        return -dmin

    # Objective for Stage 2: Maximize dmin/dmax ratio (reusing the previously fixed function)
    def local_objective_ratio(flat_points: np.ndarray) -> float:
        points = flat_points.reshape((n, d))
        ratio = _get_min_max_ratio(points)
        
        if ratio < 1e-9:
            return 1e12 # Large penalty for collapsed points.

        # We want to MAXIMIZE dmin/dmax, so the optimizer must MINIMIZE -ratio.
        return -ratio

    initial_guess = best_points.flatten()
    bounds = [(0, 1)] * (n * d)
    
    # Stage 1: Maximize minimum distance
    # This helps to spread out the points and untangle any clusters before optimizing the ratio.
    result_dmin = minimize(
        local_objective_dmin,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 1000} # Tuned options from inspirations
    )
    
    # Stage 2: Maximize the dmin/dmax ratio
    # This uses the result from Stage 1 as a much better starting point for ratio optimization.
    result_ratio = minimize(
        local_objective_ratio,
        result_dmin.x, # Use the points optimized for dmin as the starting guess
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 1000} # Tuned options from inspirations
    )
    
    final_points = result_ratio.x.reshape((n, d))
    
    return final_points


# EVOLVE-BLOCK-END
