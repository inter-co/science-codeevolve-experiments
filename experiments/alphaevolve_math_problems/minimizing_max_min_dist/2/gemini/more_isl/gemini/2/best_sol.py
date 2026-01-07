# EVOLVE-BLOCK-START
import numpy as np
import time # Added for time budgeting
from scipy.spatial.distance import pdist # Removed squareform for efficiency


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance
    using a Simulated Annealing (SA) metaheuristic. This implementation combines the
    robust SA framework from inspiration with a strong initial configuration from the
    previous target program, aiming for a higher quality solution.

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
        
        # Calculate all pairwise Euclidean distances using pdist directly
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0.0
        
        # Filter out distances very close to zero to avoid numerical issues (dmin=0)
        non_zero_distances = distances[distances > 1e-9] 
        
        if len(non_zero_distances) == 0:
            # This implies all points are practically identical.
            return 0.0
        
        dmin = np.min(non_zero_distances)
        dmax = np.max(distances) # Max distance should be found across all pairs.
        
        return dmin / dmax

    def _objective_func_sa(points: np.ndarray) -> float:
        """
        Objective function for Simulated Annealing: we want to maximize dmin/dmax,
        so we minimize its inverse, dmax/dmin.
        """
        ratio = _get_min_max_ratio(points)
        if ratio <= 1e-9: # If ratio is effectively zero, return a large cost (infinity)
            return np.inf
        return 1.0 / ratio

    # Import for local refinement
    from scipy.optimize import minimize # Added for local refinement

    # --- Initial Configuration: Hexagonal Lattice (inspired by Inspiration 2) ---
    # 1. Generate a patch of hexagonal lattice points.
    count = 5 # Create a grid of roughly (2*count+1)^2 points. More points for better selection.
    points_list = []
    s = 1.0  # Initial spacing, will be normalized later.
    sqrt3 = np.sqrt(3.0)
    for i in range(-count, count + 1):
        for j in range(-count, count + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * (sqrt3 / 2.0) * j
            points_list.append((x, y))
    
    lattice_points = np.array(points_list)

    # 2. Select the N points closest to the geometric center of the patch.
    center = np.mean(lattice_points, axis=0)
    distances_from_center = np.linalg.norm(lattice_points - center, axis=1)
    central_indices = np.argsort(distances_from_center)
    initial_points_hex = lattice_points[central_indices[:n]]

    # 3. Normalize points to fit snugly and centered inside the [0,1]x[0,1] square.
    min_coords = np.min(initial_points_hex, axis=0)
    initial_points_hex -= min_coords
    
    max_coord_val = np.max(initial_points_hex)
    if max_coord_val > 1e-6:
        initial_points_hex /= max_coord_val
    
    # Center the resulting bounding box within the unit square.
    max_coords_after_scale = np.max(initial_points_hex, axis=0)
    offset = (1.0 - max_coords_after_scale) / 2.0
    initial_points_hex += offset
    
    # Add a small perturbation to break perfect symmetry for SA exploration
    initial_points_hex += np.random.uniform(-0.01, 0.01, initial_points_hex.shape)
    current_points = np.clip(initial_points_hex, 0, 1.0) # Ensure within bounds after perturbation
    
    # Store the best solution found so far
    best_points = np.copy(current_points)
    best_objective = _objective_func_sa(current_points) # Minimize 1/ratio
    
    current_objective = best_objective # The objective of the current configuration

    # --- Simulated Annealing Parameters (tuned for hybrid approach based on Insp1 and Insp2) ---
    T_start = 0.1 # Initial temperature for exploration (from Insp1)
    T_end = 1e-6 # Final temperature for exploitation
    num_iterations = 200_000 # Reduced iterations due to better initial guess and two-stage refinement
    
    # Calculate the geometric cooling rate required to reach T_end from T_start
    cooling_rate = (T_end / T_start)**(1.0 / num_iterations)
    
    # Initial maximum perturbation magnitude for a point in a single step.
    initial_perturb_magnitude = 0.05 # Perturbation magnitude for SA (from Insp1)

    current_T = T_start

    # --- Main SA Loop ---
    for i in range(num_iterations):
        if current_T < T_end: # Stop if temperature drops below the end threshold
            break

        # Generate a new candidate configuration by perturbing all points with random noise.
        # The magnitude of the noise is scaled by the current temperature.
        perturbation_noise = (np.random.rand(n, d) - 0.5) * 2 * initial_perturb_magnitude * (current_T / T_start)
        temp_points = current_points + perturbation_noise

        # Clip points to stay within the unit square [0,1]x[0,1] bounds.
        temp_points = np.clip(temp_points, 0, 1)

        new_objective = _objective_func_sa(temp_points)

        # Acceptance criterion based on minimizing the objective (dmax/dmin)
        if new_objective < current_objective: # Always accept solutions that improve the objective
            current_points = np.copy(temp_points)
            current_objective = new_objective
            if new_objective < best_objective: # Update the globally best configuration found so far
                best_points = np.copy(temp_points)
                best_objective = new_objective
        else:
            # If the new solution is worse, accept it with a probability
            # that decreases as temperature drops (exploration vs. exploitation).
            delta_objective = new_objective - current_objective
            # Ensure delta_objective is positive for proper probability calculation and avoid numerical issues.
            if current_T > 0 and delta_objective > 1e-9: 
                acceptance_probability = np.exp(-delta_objective / current_T)
                if np.random.rand() < acceptance_probability:
                    current_points = np.copy(temp_points)
                    current_objective = new_objective # Even if worse, accept it as current for further exploration
        
        # Anneal (decrease) the temperature for the next iteration
        current_T *= cooling_rate
        
    # --- PART 2: Two-Stage Local Refinement using L-BFGS-B (inspired by Inspiration 1) ---
    # Use the best solution from SA as a starting point for a local, gradient-based optimizer.
    # The goal is to maximize the dmin/dmax ratio, which is the problem's true objective.
    
    bounds = [(0, 1)] * (n * d) # Points must stay within [0,1]x[0,1]

    # Stage 2a: Maximize minimum distance (maximin) to spread points out.
    def local_objective_dmin(flat_points: np.ndarray) -> float:
        """Objective for local optimizer: minimize -dmin."""
        points = flat_points.reshape((n, d))
        distances = pdist(points)
        # Handle cases where distances might be empty or all zero (e.g., if n < 2)
        if len(distances) == 0 or np.all(distances < 1e-9):
            return np.inf # Penalize collapsed points heavily
        return -np.min(distances)

    result_stage1 = minimize(
        local_objective_dmin, 
        best_points.flatten(), 
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 1000}
    )
    
    # Stage 2b: Maximize the dmin/dmax ratio from the maximin result.
    def local_objective_ratio(flat_points: np.ndarray) -> float:
        """
        Objective for local optimizer: minimize -(dmin/dmax).
        This aligns the local search with the global goal.
        """
        points = flat_points.reshape((n, d))
        # Re-use the existing _get_min_max_ratio helper
        ratio = _get_min_max_ratio(points)
        # We want to MAXIMIZE dmin/dmax, so the optimizer must MINIMIZE -(dmin/dmax).
        # Use np.inf as a strong penalty for invalid (collapsed) configurations.
        return -ratio if ratio > 1e-9 else np.inf

    result_stage2 = minimize(
        local_objective_ratio, 
        result_stage1.x, # Start from the result of stage 1
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 2000} # Tighter tolerances and more iterations for refinement
    )

    final_points = result_stage2.x.reshape((n, d))
    
    return final_points


# EVOLVE-BLOCK-END
