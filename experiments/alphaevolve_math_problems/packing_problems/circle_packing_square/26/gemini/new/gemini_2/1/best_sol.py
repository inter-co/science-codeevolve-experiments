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
        
        # Vectorized calculation for non-overlap constraints (inspired by Inspiration Program 3)
        # This is more efficient than a nested loop for repeatedly called constraint functions.
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        r_coords = circles[:, 2]

        # Calculate squared differences in x and y coordinates for all pairs using broadcasting.
        diff_x = x_coords[:, None] - x_coords[None, :]
        diff_y = y_coords[:, None] - y_coords[None, :]
        dist_sq = diff_x**2 + diff_y**2

        # Calculate squared sum of radii for all pairs using broadcasting.
        sum_radii = r_coords[:, None] + r_coords[None, :]
        sum_radii_sq = sum_radii**2

        # Extract values for unique pairs (i < j) from the upper triangle of the matrices.
        # `k=1` excludes the diagonal elements (i=j), as a circle does not overlap with itself.
        idx = np.triu_indices(N_CIRCLES, k=1) 
        overlap_constraints = dist_sq[idx] - sum_radii_sq[idx]

        return overlap_constraints
    
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
    Generates a structured, grid-like initial configuration of circles with small perturbations.
    Inspired by Inspiration Program 3 for better starting points for local optimization.
    """
    initial_params = np.zeros(N_CIRCLES * 3)
    
    # Determine grid dimensions for initial placement, aiming for a roughly square grid.
    rows = int(np.ceil(np.sqrt(N_CIRCLES))) 
    cols = int(np.ceil(N_CIRCLES / rows))   
    
    # Calculate an initial radius based on grid spacing.
    # The factor 0.45 ensures there's some initial separation between circles,
    # allowing the optimizer room to expand them without immediately violating constraints.
    r_base = min(1.0 / cols, 1.0 / rows) * 0.45 
    # Clip r_base to ensure it's within valid bounds [MIN_RADIUS, MAX_RADIUS].
    r_base = np.clip(r_base, MIN_RADIUS, MAX_RADIUS)

    # Place circles on a grid
    for i in range(N_CIRCLES):
        row = i // cols
        col = i % cols
        
        # Calculate initial center positions.
        # These are adjusted to be within [r_base, 1-r_base] to initially satisfy
        # the containment constraints as much as possible, giving the optimizer a valid start.
        center_x = r_base + (col + 0.5) * (1.0 - 2 * r_base) / cols
        center_y = r_base + (row + 0.5) * (1.0 - 2 * r_base) / rows
        
        initial_params[i*3] = center_x
        initial_params[i*3+1] = center_y
        initial_params[i*3+2] = r_base

    # Add a small random perturbation to the initial guess.
    # This helps break potential symmetries in the initial grid and allows the optimizer
    # to explore different configurations, potentially escaping poor local minima.
    # Use the global np.random state, which is seeded once.
    perturbation_scale = 0.01 # Small perturbation magnitude.
    initial_params += np.random.uniform(-perturbation_scale, perturbation_scale, size=initial_params.shape)
    
    # Clip perturbed values to ensure they remain within their general bounds.
    # This is crucial before passing the initial guess to the optimizer.
    for i in range(N_CIRCLES):
        initial_params[i*3] = np.clip(initial_params[i*3], 0.0, 1.0)       # Clip x-coordinates
        initial_params[i*3+1] = np.clip(initial_params[i*3+1], 0.0, 1.0)   # Clip y-coordinates
        initial_params[i*3+2] = np.clip(initial_params[i*3+2], MIN_RADIUS, MAX_RADIUS)  # Clip radii

    return initial_params

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
    num_starts = 30 # Increased number of random initial guesses to try for better exploration
    
    best_sum_radii = -np.inf
    best_circles = None

    all_constraints = get_constraints()
    all_bounds = get_bounds()

    # Use 'SLSQP' as it handles bounds and inequality constraints well.
    # 'trust-constr' is also an option, often more robust but can be slower.
    # We choose SLSQP for its balance of speed and capability for this problem size.
    optimizer_method = 'SLSQP' 

    # Options for the optimizer
    optimizer_options = {'disp': False, 'maxiter': 2500, 'ftol': 1e-7} # Increased maxiter, tightened ftol

    # Use a separate random number generator for initial guesses if full reproducibility
    # of the multi-start sequence is desired across runs, even if global seed changes.
    # For now, relying on the global np.random seed set once.
    
    # Store results to analyze later if needed
    # results = []

    for i in range(num_starts):
        # Generate a new initial guess for each start, using the structured approach
        initial_guess = generate_initial_guess()
        
        # Perform the optimization
        result = minimize(objective, initial_guess, 
                          method=optimizer_method, 
                          bounds=all_bounds, 
                          constraints=all_constraints,
                          options=optimizer_options)
        
        # results.append(result) # Keep track of all results

        # Check if the optimization was successful and improved the best result
        if result.success:
            current_sum_radii = -result.fun # Convert negative objective back to sum of radii
            
            # Additional check for feasibility, as solvers might return slightly infeasible results
            # Evaluate constraints one last time to be sure
            is_feasible = True
            for constr_dict in all_constraints:
                # Allow a small tolerance for numerical precision (e.g., 1e-6 or 1e-7)
                if np.any(constr_dict['fun'](result.x) < -1e-7): # Tighten tolerance slightly
                    is_feasible = False
                    # print(f"Start {i+1}: Infeasible solution detected post-optimization.")
                    break
            
            if is_feasible and current_sum_radii > best_sum_radii:
                best_sum_radii = current_sum_radii
                best_circles = result.x.reshape(N_CIRCLES, 3)
                # print(f"Start {i+1}: New best sum_radii = {best_sum_radii:.6f}")
        # else:
            # print(f"Start {i+1}: Optimization failed or did not converge: {result.message}")

    if best_circles is None:
        # If no successful optimization, return a default (e.g., initial guess or empty array)
        print("Warning: No successful optimization found. Returning a default (potentially sub-optimal) arrangement.")
        # Fallback: Generate one last initial guess and return it, ensuring it's shaped correctly
        return generate_initial_guess().reshape(N_CIRCLES, 3)

    return best_circles


# EVOLVE-BLOCK-END
