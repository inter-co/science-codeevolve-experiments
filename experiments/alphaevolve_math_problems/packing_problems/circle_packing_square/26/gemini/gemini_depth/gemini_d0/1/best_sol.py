# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
# No need for 'time' for the final submission, as it's not a required package and doesn't affect functionality.

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# Helper function for scaling radii, adapted for the 1D params array
def _scale_radii_scipy_params(params: np.ndarray, n_circles: int) -> np.ndarray:
    """
    Scales all radii in the 1D params array up by the maximum possible common factor
    without causing overlaps or boundary violations.
    """
    current_params = params.copy() # Work on a copy
    
    # Extract positions and radii from the 1D parameter array
    positions = np.column_stack((current_params[0::3], current_params[1::3])) # Shape (n, 2)
    radii = current_params[2::3]

    # Ensure all radii are positive for calculations, preventing division by zero.
    radii_positive = np.maximum(radii, 1e-10) 

    max_scale_factor = float('inf')

    # Calculate scaling factor based on inter-circle overlaps
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            p1 = positions[i]
            p2 = positions[j]
            r1 = radii_positive[i]
            r2 = radii_positive[j]

            dist = np.linalg.norm(p2 - p1)
            
            if (r1 + r2) > 1e-10: 
                scale = dist / (r1 + r2)
                max_scale_factor = min(max_scale_factor, scale)

    # Calculate scaling factor based on boundary violations
    for i in range(n_circles):
        x, y = positions[i, 0], positions[i, 1]
        r = radii_positive[i]

        max_scale_factor = min(max_scale_factor, x / r, (1 - x) / r, y / r, (1 - y) / r)
    
    # Apply scaling only if it results in a meaningful increase (factor > 1) and is a finite number.
    if max_scale_factor > 1.0 + 1e-9 and np.isfinite(max_scale_factor):
        current_params[2::3] *= max_scale_factor
    
    # After scaling, explicitly clip radii to respect the optimization bounds [1e-6, 0.5].
    current_params[2::3] = np.clip(current_params[2::3], 1e-6, 0.5)
    
    return current_params

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This implementation uses a multi-stage, gradient-based optimization approach
    (scipy.optimize.minimize with SLSQP) coupled with multiple random initializations
    and a radius growth heuristic to escape local minima and explore the solution space effectively.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26

    # --- Objective Function ---
    # We want to maximize sum(r_i), so the objective for scipy.optimize.minimize is -sum(r_i)
    def objective(params: np.ndarray) -> float:
        radii = params[2::3]
        return -np.sum(radii)

    # --- Constraints ---
    # params = [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]

    # 1. Containment constraints: r_i <= x_i <= 1-r_i and r_i <= y_i <= 1-r_i
    # These translate to four inequality constraints for each circle:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    def containment_constraints(params: np.ndarray) -> np.ndarray:
        constraints = []
        for i in range(n):
            x, y, r = params[i*3 : (i+1)*3]
            constraints.append(x - r)       # Circle must be inside left boundary
            constraints.append(1 - x - r)   # Circle must be inside right boundary
            constraints.append(y - r)       # Circle must be inside bottom boundary
            constraints.append(1 - y - r)   # Circle must be inside top boundary
        return np.array(constraints)

    # 2. Non-overlap constraints: sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    # This is more robustly formulated by squaring both sides to avoid square roots,
    # ensuring differentiability and numerical stability:
    # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    def non_overlap_constraints(params: np.ndarray) -> np.ndarray:
        constraints = []
        for i in range(n):
            xi, yi, ri = params[i*3 : (i+1)*3]
            for j in range(i + 1, n): # Only check each unique pair once
                xj, yj, rj = params[j*3 : (j+1)*3]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                min_dist_sq = (ri + rj)**2
                constraints.append(dist_sq - min_dist_sq) # Distance squared must be greater than or equal to sum of radii squared
        return np.array(constraints)

    # --- Bounds ---
    # Define bounds for each variable (x, y, r).
    # x_i, y_i are within [0, 1].
    # r_i must be non-negative and cannot exceed 0.5 (as a single circle of radius 0.5 can fit).
    # Using a small lower bound for radius (1e-6) to avoid potential numerical issues with r=0.
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))       # x_i coordinate
        bounds.append((0.0, 1.0))       # y_i coordinate
        bounds.append((1e-6, 0.5))      # r_i radius

    # --- Assemble Constraints for SciPy's minimize function ---
    scipy_constraints = [
        {'type': 'ineq', 'fun': containment_constraints},
        {'type': 'ineq', 'fun': non_overlap_constraints}
    ]

    # --- Optimization Parameters for SLSQP ---
    # SLSQP is suitable for problems with non-linear constraints.
    # Increased maxiter and stricter ftol for better convergence.
    slsqp_options = {'maxiter': 10000, 'ftol': 1e-9, 'disp': False} # Increased maxiter and stricter ftol

    best_sum_radii = 0.0
    # Initialize best_circles_config to an array of zeros. This will be returned
    # if no successful optimization run is found, ensuring the return type is correct.
    best_circles_config = np.zeros((n, 3))

    # --- Iterative Optimization Strategy ---
    # To mitigate SLSQP's tendency to get stuck in local minima,
    # we employ multiple random initial guesses and a radius growth heuristic.
    num_initial_attempts = 20   # Increased for more extensive exploration
    refinement_stages = 5       # Increased for deeper refinement

    for attempt_idx in range(num_initial_attempts):
        # Generate a new random initial guess for each attempt
        # Fixed random seed for reproducibility, varied by attempt_idx.
        np.random.seed(42 + attempt_idx)
        current_params = np.zeros(n * 3)
        
        # Initialize positions in a more structured grid-like manner with perturbation
        num_cols = 5 # For 26 circles, a 5x6 grid works well
        num_rows = 6
        x_coords_grid = np.linspace(0.1, 0.9, num_cols)
        y_coords_grid = np.linspace(0.1, 0.9, num_rows)
        
        initial_positions_list = []
        for y_coord in y_coords_grid:
            for x_coord in x_coords_grid:
                initial_positions_list.append([x_coord, y_coord])
                if len(initial_positions_list) == n:
                    break
            if len(initial_positions_list) == n:
                break
        
        initial_positions = np.array(initial_positions_list)[:n]
        # Add small random perturbation to avoid perfect symmetry traps and aid exploration
        initial_positions += np.random.uniform(-0.02, 0.02, size=initial_positions.shape)
        # Clip to ensure initial positions are within reasonable bounds after perturbation
        initial_positions = np.clip(initial_positions, 0.05, 0.95)

        for i in range(n):
            current_params[i*3] = initial_positions[i, 0] # x
            current_params[i*3 + 1] = initial_positions[i, 1] # y
            # Small, slightly varied initial radii
            current_params[i*3 + 2] = 0.005 + np.random.uniform(0, 0.005)

        # Iterative refinement loop: optimize, then slightly increase radii and re-optimize
        for stage in range(refinement_stages):
            result = minimize(objective, current_params,
                              method='SLSQP',
                              bounds=bounds,
                              constraints=scipy_constraints,
                              options=slsqp_options)

            if result.success:
                current_params = result.x # Use the optimized parameters as the starting point for the next stage
                current_sum_radii = -result.fun # Convert minimized negative sum back to positive sum

                # Update the overall best configuration found so far
                if current_sum_radii > best_sum_radii:
                    best_sum_radii = current_sum_radii
                    best_circles_config = current_params.reshape((n, 3))

                # If not the last stage, apply the more precise radius scaling heuristic.
                if stage < refinement_stages - 1:
                    current_params = _scale_radii_scipy_params(current_params, n)
            else:
                # If an optimization stage fails, break from the refinement loop for this attempt
                # and move to the next initial attempt.
                break

    # Reshape the best found 1D parameter array into the (n, 3) circles array format
    circles_final = best_circles_config
    circles_final[:, 2] = np.maximum(circles_final[:, 2], 0.0) # Ensure radii are non-negative
    
    return circles_final


# EVOLVE-BLOCK-END
