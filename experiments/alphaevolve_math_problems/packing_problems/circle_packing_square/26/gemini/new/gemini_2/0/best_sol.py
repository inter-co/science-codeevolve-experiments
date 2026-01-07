# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from scipy.optimize import minimize # Core optimization library
import time

# --- Configuration ---
N_CIRCLES = 26
IND_SIZE = N_CIRCLES * 3  # x, y, r for each circle
MAX_RADIUS = 0.5 # A circle can't have radius > 0.5 in a unit square
MIN_RADIUS = 1e-5 # Minimum radius to avoid numerical issues and zero-sized circles

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- Objective Function ---
def objective(params):
    """
    Objective function to be MINIMIZED.
    Returns the negative sum of radii, as we want to MAXIMIZE sum of radii.
    """
    circles = params.reshape(N_CIRCLES, 3)
    r = circles[:, 2]
    return -np.sum(r)

# --- Constraint Functions ---
def get_constraints():
    """
    Defines all non-linear inequality constraints for scipy.optimize.minimize.
    Constraints must be of the form g(x) >= 0.
    """
    constraints = []

    # 1. Containment Constraints (4 per circle)
    # r <= x <= 1-r  => x - r >= 0  AND  1 - r - x >= 0
    # r <= y <= 1-r  => y - r >= 0  AND  1 - r - y >= 0
    def containment_constraints(params):
        circles = params.reshape(N_CIRCLES, 3)
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # All these values must be >= 0
        constraints_values = np.concatenate([
            x - r,
            1 - r - x,
            y - r,
            1 - r - y
        ])
        return constraints_values
    
    constraints.append({'type': 'ineq', 'fun': containment_constraints})

    # 2. Non-overlap Constraints (N_CIRCLES * (N_CIRCLES-1) / 2 pairs)
    # (xi-xj)² + (yi-yj)² >= (ri + rj)²
    # => (xi-xj)² + (yi-yj)² - (ri + rj)² >= 0
    def non_overlap_constraints(params):
        circles = params.reshape(N_CIRCLES, 3)
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        overlap_values = []
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist_req_sq = (r[i] + r[j])**2
                overlap_values.append(dist_sq - min_dist_req_sq)
        return np.array(overlap_values)
    
    constraints.append({'type': 'ineq', 'fun': non_overlap_constraints})
    
    return constraints

# --- Bounds ---
def get_bounds():
    """
    Defines bounds for each variable (x, y, r).
    """
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0)) # x_i
        bounds.append((0.0, 1.0)) # y_i
        bounds.append((MIN_RADIUS, MAX_RADIUS)) # r_i
    return bounds

# --- Initialization for Multi-start ---
def generate_initial_guess():
    """
    Generates a random, (mostly) feasible initial configuration of circles.
    Circles are placed with small radii to minimize initial overlaps.
    """
    initial_params = []
    for _ in range(N_CIRCLES):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        
        # Start with a very small radius, ensuring containment is easy
        # and leaving room for growth.
        r = random.uniform(MIN_RADIUS, 0.05) # Max initial r of 0.05
        
        # Ensure x,y are far enough from boundaries for the small r
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        
        initial_params.extend([x, y, r])
    return np.array(initial_params)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses scipy.optimize.minimize with a multi-start strategy.
    """
    # EXPLORATION: Switching from DEAP Genetic Algorithm to SciPy's constrained optimization.
    # This provides a distinct algorithmic pathway, leveraging local search with multiple starts
    # to find a good solution, rather than a population-based evolutionary approach.
    # The physics-based repair function has been removed to reduce complexity and avoid timeouts,
    # as scipy.optimize methods are designed to handle constraints directly.
    num_starts = 20 # Number of random initial guesses to try (increased from 10 to 20 for better exploration)
    
    best_sum_radii = -np.inf
    best_circles = None

    all_constraints = get_constraints()
    all_bounds = get_bounds()

    # Use 'SLSQP' as it handles bounds and inequality constraints well.
    # 'trust-constr' is also an option, often more robust but can be slower.
    # We choose SLSQP for its balance of speed and capability for this problem size.
    optimizer_method = 'SLSQP' 

    for i in range(num_starts):
        # Generate a new random initial guess for each start
        initial_guess = generate_initial_guess()
        
        # Perform the optimization
        # `options` can be tuned, e.g., `maxiter`, `ftol`, `eps`
        result = minimize(objective, initial_guess, 
                          method=optimizer_method, 
                          bounds=all_bounds, 
                          constraints=all_constraints,
                          options={'disp': False, 'maxiter': 2000}) # Increased maxiter for deeper search
        
        # Check if the optimization was successful and improved the best result
        if result.success:
            current_sum_radii = -result.fun # Convert negative objective back to sum of radii
            
            # Additional check for feasibility, as solvers might return slightly infeasible results
            # Evaluate constraints one last time to be sure
            is_feasible = True
            for constr_dict in all_constraints:
                # Allow a small tolerance for numerical precision (e.g., 1e-6 or 1e-7)
                if np.any(constr_dict['fun'](result.x) < -1e-6): 
                    is_feasible = False
                    break
            
            if is_feasible and current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles = result.x.reshape(N_CIRCLES, 3)
                # print(f"Start {i+1}: New best sum_radii = {best_sum_radii:.6f}")
        # else:
            # print(f"Start {i+1}: Optimization failed or did not converge.")

    if best_circles is None:
        # If no successful optimization, return a default (e.g., initial guess or empty array)
        # For this problem, it's highly unlikely no start will succeed.
        print("Warning: No successful optimization found. Returning initial guess.")
        # Ensure the returned initial guess is also reshaped correctly
        return generate_initial_guess().reshape(N_CIRCLES, 3)

    return best_circles


# EVOLVE-BLOCK-END
