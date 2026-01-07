# EVOLVE-BLOCK-START
import numpy as np
import random
import math
from scipy.optimize import minimize # Core optimization library
import time
from joblib import Parallel, delayed # New imports for parallelization
import os # New import for system info (cpu_count)

# --- Configuration ---
N_CIRCLES = 26
IND_SIZE = N_CIRCLES * 3  # x, y, r for each circle
MAX_RADIUS = 0.5 # A circle can't have radius > 0.5 in a unit square
MIN_RADIUS = 1e-5 # Minimum radius to avoid numerical issues and zero-sized circles

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Get number of CPU cores for parallel processing
NUM_CORES = os.cpu_count() or 1
PARALLEL_JOBS = min(NUM_CORES, 8) # Cap to 8 jobs to prevent excessive resource usage, similar to Inspiration 2

# --- Objective Function ---
def objective(params):
    """
    Objective function to be MINIMIZED.
    Returns the negative sum of radii, as we want to MAXIMIZE sum of radii.
    """
    circles = params.reshape(N_CIRCLES, 3)
    r = circles[:, 2]
    return -np.sum(r)

# --- Constraint Functions (Moved to global scope for joblib compatibility) ---
def _containment_constraints(params):
    """
    Defines containment constraints: r <= x <= 1-r and r <= y <= 1-r.
    Constraints must be of the form g(x) >= 0.
    """
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

def _non_overlap_constraints(params):
    """
    Defines non-overlap constraints: (xi-xj)² + (yi-yj)² >= (ri + rj)².
    Constraints must be of the form g(x) >= 0.
    Uses vectorized numpy operations for efficiency (adapted from Inspiration 2).
    """
    circles = params.reshape(N_CIRCLES, 3)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Vectorized calculation for squared distances between centers
    centers = circles[:, :2]
    # (xi - xj)^2 + (yi - yj)^2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq_matrix = np.sum(diff**2, axis=2)

    # (ri + rj)^2
    radii_sum_matrix = r[:, np.newaxis] + r[np.newaxis, :]
    min_dist_req_sq_matrix = radii_sum_matrix**2

    # Extract upper triangle for unique pairs (i < j)
    # k=1 ensures we don't include diagonal (self-distance)
    upper_triangle_indices = np.triu_indices(N_CIRCLES, k=1)
    
    # Constraint: dist_sq_matrix - min_dist_req_sq_matrix >= 0
    return dist_sq_matrix[upper_triangle_indices] - min_dist_req_sq_matrix[upper_triangle_indices]

def get_constraints():
    """
    Bundles all constraint functions for scipy.optimize.minimize.
    """
    constraints = []
    constraints.append({'type': 'ineq', 'fun': _containment_constraints})
    constraints.append({'type': 'ineq', 'fun': _non_overlap_constraints})
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
        # and leaving room for growth. (Retaining 0.05 from target program for better local optima)
        r = random.uniform(MIN_RADIUS, 0.05)
        
        # Ensure x,y are far enough from boundaries for the small r
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        
        initial_params.extend([x, y, r])
    return np.array(initial_params)

# --- Helper function for parallel optimization (adapted from Inspiration 2) ---
def _run_single_optimization(optimizer_method, all_bounds, all_constraints):
    """
    Runs a single instance of scipy.optimize.minimize.
    Arguments are passed explicitly for clarity and joblib compatibility.
    """
    # Re-seed random generators for each job to ensure independent initial guesses
    # This uses a combination of the global seed and process ID for unique seeds per worker
    current_pid = os.getpid()
    random.seed(RANDOM_SEED + current_pid)
    np.random.seed(RANDOM_SEED + current_pid)

    initial_guess = generate_initial_guess()
    
    result = minimize(objective, initial_guess, 
                      method=optimizer_method, 
                      bounds=all_bounds, 
                      constraints=all_constraints,
                      options={'disp': False, 'maxiter': 2500, 'ftol': 1e-7}) # Increased maxiter and ftol from Inspiration 2
    
    if result.success:
        current_sum_radii = -result.fun
        
        # Stricter feasibility check after optimization
        is_feasible = True
        for constr_dict in all_constraints:
            # Using a slightly tighter tolerance for final check than default SLSQP (1e-6)
            if np.any(constr_dict['fun'](result.x) < -1e-7): 
                is_feasible = False
                break
        
        if is_feasible:
            return current_sum_radii, result.x.reshape(N_CIRCLES, 3)
    return -np.inf, None # Return very low fitness if not successful or feasible

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses scipy.optimize.minimize with a parallelized multi-start strategy.
    """
    num_starts = 100 # Increased number of random initial guesses for better exploration

    best_sum_radii = -np.inf
    best_circles = None

    all_constraints = get_constraints()
    all_bounds = get_bounds()

    optimizer_method = 'SLSQP' 

    # Run optimizations in parallel using joblib (adapted from Inspiration 2)
    # _run_single_optimization is now self-contained regarding N_CIRCLES, MIN_RADIUS, MAX_RADIUS.
    results = Parallel(n_jobs=PARALLEL_JOBS)(
        delayed(_run_single_optimization)(optimizer_method, all_bounds, all_constraints)
        for _ in range(num_starts)
    )

    for current_sum_radii, circles_data in results:
        if circles_data is not None and current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_circles = circles_data

    if best_circles is None:
        # Fallback if no successful optimization found
        print("Warning: No successful optimization found. Returning a default arrangement.")
        fallback_circles = generate_initial_guess().reshape(N_CIRCLES, 3)
        # Ensure fallback circles are at least valid in terms of containment and min_radius
        for i in range(N_CIRCLES):
            x, y, r = fallback_circles[i]
            max_r_by_pos = min(x, 1-x, y, 1-y)
            fallback_circles[i, 2] = np.clip(r, MIN_RADIUS, max_r_by_pos)
        return fallback_circles

    return best_circles


# EVOLVE-BLOCK-END
