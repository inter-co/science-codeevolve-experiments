# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import random # Import random for seeding and uniform distribution
from scipy.spatial.distance import pdist # Added for vectorized constraint calculation

# Global seed for reproducibility of the entire script.
# This ensures that all calls to np.random and random are consistent across runs.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def generate_initial_guess_for_slsqp(n_circles: int):
    """
    Generates an initial guess for circle packing parameters (x, y, r)
    based on a perturbed grid layout. This function uses global np.random.
    """
    num_cols = 5
    num_rows = 6
    
    # Estimate an initial radius based on the grid dimensions
    initial_r_estimate = min(1.0 / (2 * num_cols), 1.0 / (2 * num_rows))
    
    params_list = []
    current_circle_count = 0
    
    for r_idx in range(num_rows):
        for c_idx in range(num_cols):
            if current_circle_count >= n_circles:
                break
            
            x_center = (c_idx + 0.5) / num_cols
            y_center = (r_idx + 0.5) / num_rows
            
            # Add larger random perturbations to break perfect symmetry and explore more broadly
            x_perturb = np.random.uniform(-0.02, 0.02) # Increased perturbation range
            y_perturb = np.random.uniform(-0.02, 0.02) # Increased perturbation range
            
            # Randomize initial radius slightly around the estimate, allowing for larger initial radii
            initial_radius = initial_r_estimate * np.random.uniform(0.9, 1.1) # Wider range for initial radii
            
            params_list.extend([x_center + x_perturb, y_center + y_perturb, initial_radius])
            current_circle_count += 1
        if current_circle_count >= n_circles:
            break
            
    # Flatten the list into a single NumPy array for the optimizer
    return np.array(params_list).flatten()

def is_valid_packing(circles_data: np.ndarray, epsilon=1e-6) -> bool:
    """
    Checks if a given set of circles forms a strictly valid packing.
    Args:
        circles_data: np.array of shape (N, 3) where each row is (x, y, r).
        epsilon: Tolerance for floating point comparisons.
    Returns:
        True if the packing is valid, False otherwise.
    """
    n = circles_data.shape[0]

    # Check containment
    for i in range(n):
        x, y, r = circles_data[i]
        if r < epsilon: return False # Radii must be positive
        if not (r - epsilon <= x <= 1 - r + epsilon and r - epsilon <= y <= 1 - r + epsilon):
            return False

    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            c1 = circles_data[i]
            c2 = circles_data[j]
            r1 = c1[2]
            r2 = c2[2]
            dist_sq = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
            min_dist_sq = (r1 + r2)**2
            if dist_sq < min_dist_sq - epsilon: # Allow for small numerical precision
                return False
    return True


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This implementation uses a gradient-based optimization approach (SLSQP)
    with a multi-start strategy and explicit non-linear constraints.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26

    # --- Objective Function: Maximize sum of radii ---
    # SciPy's `minimize` performs minimization, so we minimize the negative sum of radii.
    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)

    # --- Constraint Functions: Ensure containment and non-overlap (VECTORIZED) ---
    # All constraints are formulated as g(x) >= 0.
    def constraints_func(params):
        """
        Calculates all constraint values in a vectorized manner for performance.
        Inspired by the vectorized evaluation function in the inspiration programs.
        """
        circles = params.reshape(n, 3)
        coords = circles[:, :2]
        radii = circles[:, 2]

        # 1. Vectorized Containment Constraints
        # x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
        containment_c1 = coords[:, 0] - radii
        containment_c2 = 1.0 - coords[:, 0] - radii
        containment_c3 = coords[:, 1] - radii
        containment_c4 = 1.0 - coords[:, 1] - radii
        
        # 2. Vectorized Non-overlap Constraints
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        if n > 1:
            # Squared pairwise distances between centers
            dist_sq = pdist(coords)**2
            
            # Sum of radii for each pair
            radii_sum_matrix = np.add.outer(radii, radii)
            # Extract upper triangle to match pdist order (k=1 excludes diagonal)
            radii_sum = radii_sum_matrix[np.triu_indices(n, k=1)]
            
            non_overlap_c = dist_sq - radii_sum**2
        else:
            non_overlap_c = np.array([])

        # Combine all constraints into a single flat array
        return np.concatenate([
            containment_c1,
            containment_c2,
            containment_c3,
            containment_c4,
            radii, # r_i >= 0 constraint
            non_overlap_c
        ])

    # --- Bounds for Optimization Variables ---
    lower_bounds = np.zeros(3 * n)
    upper_bounds = np.ones(3 * n)
    upper_bounds[2::3] = 0.5 # Max radius is 0.5 in unit square

    bounds = Bounds(lower_bounds, upper_bounds)

    # --- Nonlinear Constraint Object for SciPy ---
    nonlinear_constraints = NonlinearConstraint(constraints_func, 0, np.inf)

    # --- Multi-start Optimization ---
    # Increased number of starts due to faster vectorized constraint evaluation
    num_starts = 35 
    best_sum_radii = -np.inf
    best_circles = None
    last_successful_circles = None # To store at least one valid solution if no "best" is found

    for i in range(num_starts):
        # Generate a new, perturbed initial guess for each start
        initial_guess = generate_initial_guess_for_slsqp(n)

        # Perform Optimization
        result = minimize(
            fun=objective,
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=[nonlinear_constraints],
            # Use tighter tolerance inspired by inspiration program 1 for higher precision
            options={'maxiter': 5000, 'ftol': 1e-9, 'disp': False}
        )

        current_circles = result.x.reshape(n, 3)
        current_circles[:, 2] = np.maximum(0, current_circles[:, 2]) # Ensure radii are non-negative

        # Check for strict validity and update best result
        if is_valid_packing(current_circles):
            current_sum_radii = np.sum(current_circles[:, 2])
            last_successful_circles = current_circles # Keep track of the last valid solution
            if current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles = current_circles
        # else:
            # print(f"Start {i+1}: Result not strictly valid.")


    if best_circles is None:
        # If no strictly valid packing was found across all starts,
        # return the last valid one we encountered, or a default if none.
        print("Warning: No strictly valid packing found across all starts. Returning last valid attempt.")
        if last_successful_circles is not None:
            return last_successful_circles
        else:
            # Fallback: If absolutely no valid packing was found, return the result from the first run
            # after clamping radii to ensure at least a non-negative radius output.
            initial_guess = generate_initial_guess_for_slsqp(n)
            result = minimize(
                fun=objective,
                x0=initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=[nonlinear_constraints],
                options={'maxiter': 5000, 'ftol': 1e-9, 'disp': False} # Keep ftol consistent
            )
            circles = result.x.reshape(n, 3)
            circles[:, 2] = np.maximum(0, circles[:, 2])
            return circles

    return best_circles


# EVOLVE-BLOCK-END
