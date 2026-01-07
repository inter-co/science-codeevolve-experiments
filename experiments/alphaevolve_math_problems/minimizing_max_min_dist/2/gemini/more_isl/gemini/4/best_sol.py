# EVOLVE-BLOCK-START
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

    def _get_dmin_dmax_and_ratio(points: np.ndarray) -> tuple[float, float, float]:
        """
        Calculates dmin, dmax, and the ratio of minimum to maximum distance
        among all pairs of points.
        """
        if points.shape[0] < 2:
            return 0.0, 0.0, 0.0 # dmin, dmax, ratio
        
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0.0, 0.0, 0.0
        
        non_zero_distances = distances[distances > 1e-9] 
        
        dmax = np.max(distances) # Max distance is always from all pairs
        
        if len(non_zero_distances) == 0:
            # This implies all points are practically identical or very close.
            # dmin is effectively 0.
            return 0.0, dmax, 0.0
        
        dmin = np.min(non_zero_distances)
        ratio = dmin / dmax if dmax > 1e-9 else 0.0
        return dmin, dmax, ratio

    def _objective_func_sa(points: np.ndarray) -> float:
        """
        Objective function for Simulated Annealing: minimize the inverse of the ratio.
        """
        _, _, ratio = _get_dmin_dmax_and_ratio(points)
        if ratio <= 1e-9: # If ratio is effectively zero, return a large cost (infinity)
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

    T_start = 0.2
    T_end = 1e-6
    num_iterations = 900_000 # Increased to better utilize time budget for global search, as per inspiration.
    cooling_rate = (T_end / T_start)**(1.0 / num_iterations)
    initial_perturb_magnitude = 0.08
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

    # --- PART 2: LOCAL REFINEMENT (L-BFGS-B) ---
    # --- PART 2: TWO-STAGE LOCAL REFINEMENT (L-BFGS-B) ---
    # Inspired by INSPIRATION 1, this uses a two-stage local optimization:
    # 1. Maximize dmin (spread points out).
    # 2. Maximize dmin/dmax ratio (fine-tune).

    def local_objective_dmin(flat_points: np.ndarray) -> float:
        """
        Objective for first local optimizer stage: minimize -dmin (maximize dmin).
        """
        points = flat_points.reshape((n, d))
        dmin, _, _ = _get_dmin_dmax_and_ratio(points)
        return -dmin # Minimize -dmin to maximize dmin

    def local_objective_ratio(flat_points: np.ndarray) -> float:
        """
        Objective for second local optimizer stage: minimize -(dmin/dmax).
        This directly optimizes the desired ratio, which is ideal for gradient-based methods.
        """
        points = flat_points.reshape((n, d))
        _, _, ratio = _get_dmin_dmax_and_ratio(points)
        if ratio <= 1e-9: # If ratio is effectively zero, return a large cost
            return np.inf
        return -ratio # Minimize -ratio to maximize ratio

    bounds = [(0, 1)] * (n * d)
    
    # Stage 1: Maximize dmin
    # Use the best solution from SA as a starting point.
    result_stage1 = minimize(
        local_objective_dmin,
        best_points.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 1000} # Maxiter reduced for each stage to fit within time budget
    )

    # Stage 2: Maximize dmin/dmax ratio, starting from the result of Stage 1
    final_points_initial_guess = result_stage1.x.reshape((n, d))
    result_stage2 = minimize(
        local_objective_ratio,
        final_points_initial_guess.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 1000} # Tighter tolerances for refinement
    )
    
    final_points = result_stage2.x.reshape((n, d))
    
    return final_points


# EVOLVE-BLOCK-END
