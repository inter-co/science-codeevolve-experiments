# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint, Bounds, differential_evolution
from scipy.spatial.distance import pdist
from joblib import Parallel, delayed
import os

# Set a global random seed for reproducibility to ensure deterministic results for stochastic methods
np.random.seed(42)

# Define helper functions outside the main function, as required for parallelization.
def _unpack_variables(variables: np.ndarray, n_circles: int):
    """Unpacks the 1D optimization variable array into x, y, r, and width."""
    # Variable order: [x_0...x_n-1, y_0...y_n-1, r_0...r_n-1, width]
    x = variables[0:n_circles]
    y = variables[n_circles:2*n_circles]
    r = variables[2*n_circles:3*n_circles]
    width = variables[3*n_circles]
    height = 2.0 - width # Perimeter constraint (width + height = 2)
    
    return x, y, r, width, height

def _penalized_objective(variables: np.ndarray, n_circles: int,
                         P_BOUNDARY: float, P_OVERLAP: float, P_MIN_RADIUS: float, epsilon: float) -> float:
    """
    Objective function for global optimization with penalty terms for constraint violations.
    Minimizes -sum(r) + penalties.
    """
    x, y, r, width, height = _unpack_variables(variables, n_circles)
    
    obj = -np.sum(r) # Primary objective: maximize sum of radii
    
    penalty = 0.0

    # 1. Minimum Radius Penalty: Ensures all radii are strictly positive
    # Penalizes radii that are positive but fall below a small threshold (epsilon)
    min_radius_violation = np.maximum(0, epsilon - r)
    penalty += P_MIN_RADIUS * np.sum(min_radius_violation)

    # 2. Boundary Violations: Circles must be entirely within the rectangle
    # Penalties applied if x - r < 0, x + r > width, y - r < 0, y + r > height
    penalty += P_BOUNDARY * np.sum(np.maximum(0, r - x))            # Left boundary (x_i - r_i >= 0 => r_i - x_i <= 0)
    penalty += P_BOUNDARY * np.sum(np.maximum(0, x + r - width))     # Right boundary (width - (x_i + r_i) >= 0 => x_i + r_i - width <= 0)
    penalty += P_BOUNDARY * np.sum(np.maximum(0, r - y))            # Bottom boundary (y_i - r_i >= 0 => r_i - y_i <= 0)
    penalty += P_BOUNDARY * np.sum(np.maximum(0, y + r - height))    # Top boundary (height - (y_i + r_i) >= 0 => y_i + r_i - height <= 0)

    # 3. Overlap Violations: Circles must not overlap
    # Distance between centers (d_ij) must be greater than or equal to sum of their radii (r_i + r_j)
    # Penalize if (r_i + r_j)^2 > d_ij^2
    if n_circles > 1:
        centers = np.column_stack((x, y)) # Create (N, 2) array of circle centers
        
        # Compute pairwise squared Euclidean distances between circle centers for efficiency
        distances_sq = pdist(centers, metric='sqeuclidean')

        # Compute pairwise sum of radii (r_i + r_j) for all unique pairs
        sum_radii_matrix = r[:, None] + r[None, :] # Create a matrix where M_ij = r_i + r_j
        sum_radii_pairs = sum_radii_matrix[np.triu_indices(n_circles, k=1)] # Extract unique pairs to match pdist order

        # Calculate violation: If sum_radii_pairs^2 > distances_sq, then circles overlap
        overlap_violation = np.maximum(0, sum_radii_pairs**2 - distances_sq)
        penalty += P_OVERLAP * np.sum(overlap_violation)

    return obj + penalty

def _objective_for_trust_constr(variables: np.ndarray, n_circles: int) -> float:
    """Objective function for trust-constr: negative sum of radii."""
    _, _, r, _, _ = _unpack_variables(variables, n_circles)
    return -np.sum(r)

def _constraints_for_trust_constr(variables: np.ndarray, n_circles: int, epsilon: float) -> np.ndarray:
    """Calculates all non-linear constraint values for trust-constr.
    All constraints are formulated as g(x) >= 0.
    """
    x, y, r, width, height = _unpack_variables(variables, n_circles)
    
    constraints = []

    # 1. Circle containment constraints (4 * n_circles)
    # x_i - r_i >= 0
    constraints.extend(x - r)
    # width - (x_i + r_i) >= 0
    constraints.extend(width - x - r)
    # y_i - r_i >= 0
    constraints.extend(y - r)
    # height - (y_i + r_i) >= 0
    constraints.extend(height - y - r)
    
    # 2. Non-overlapping constraints (n_circles * (n_circles - 1) / 2)
    # d_ij^2 - (r_i + r_j)^2 >= 0
    if n_circles > 1:
        centers = np.column_stack((x, y))
        dist_sq_pairs = pdist(centers, metric='sqeuclidean')

        r_matrix = r[:, np.newaxis] + r[np.newaxis, :]
        upper_triangle_indices = np.triu_indices(n_circles, k=1)
        r_sum_pairs = r_matrix[upper_triangle_indices]
        min_dist_sq_pairs = r_sum_pairs**2
        
        constraints.extend(dist_sq_pairs - min_dist_sq_pairs)
    
    # 3. Radii must be positive (r_i >= epsilon)
    constraints.extend(r - epsilon)

    return np.array(constraints)

def _run_optimization_attempt(n_circles: int, random_seed: int,
                              de_maxiter: int, de_popsize: int, tc_maxiter: int,
                              all_bounds_list: list, penalty_coeffs: tuple, epsilon: float):
    """
    Runs a single differential evolution + trust-constr optimization attempt.
    Designed to be called in parallel.
    """
    np.random.seed(random_seed) # Set seed for this specific attempt

    # Differential Evolution Optimization (Global Search)
    de_options = {
        'maxiter': de_maxiter,
        'popsize': de_popsize,
        'tol': 0.005, # Looser tolerance for DE as it's a global search, refined by trust-constr
        'seed': random_seed,
        'disp': False,
        'workers': 1, # Set to 1 because joblib handles outer parallelization of multiple DE runs
        'polish': True, # Apply local optimization (BFGS) at the end of DE's global search
        'init': 'latinhypercube', # Use a more systematic initial population for better coverage
        'recombination': 0.7,
        'mutation': (0.5, 1.0),
        'updating': 'deferred', # Recommended for this type of parallel setup
        'strategy': 'randtobest1bin' # Good balance of exploration/exploitation
    }

    de_result = differential_evolution(
        func=_penalized_objective,
        bounds=all_bounds_list,
        args=(n_circles, *penalty_coeffs, epsilon), # P_BOUNDARY, P_OVERLAP, P_MIN_RADIUS, epsilon
        **de_options
    )
    
    optimal_variables_de = de_result.x

    # Local Refinement with trust-constr
    num_containment_constraints = 4 * n_circles
    num_overlap_constraints = n_circles * (n_circles - 1) // 2
    num_min_radius_constraints = n_circles
    num_total_constraints = num_containment_constraints + num_overlap_constraints + num_min_radius_constraints

    nlc_lb = np.full(num_total_constraints, 0.0) # All constraints are g(x) >= 0
    nlc_ub = np.full(num_total_constraints, np.inf)
    
    nonlinear_constraint = NonlinearConstraint(
        lambda vars: _constraints_for_trust_constr(vars, n_circles, epsilon),
        nlc_lb, 
        nlc_ub
    )

    bounds_trust_constr = Bounds([b[0] for b in all_bounds_list], [b[1] for b in all_bounds_list])

    refine_res = minimize(
        lambda vars: _objective_for_trust_constr(vars, n_circles),
        optimal_variables_de, # Start from the best DE result
        method='trust-constr',
        bounds=bounds_trust_constr,
        constraints=[nonlinear_constraint],
        options={'maxiter': tc_maxiter, 'verbose': 0, 'gtol': 1e-6, 'xtol': 1e-6, 'barrier_tol': 1e-6}
    )
    
    final_optimal_variables = refine_res.x if refine_res.success else optimal_variables_de

    _, _, final_r, _, _ = _unpack_variables(final_optimal_variables, n_circles)
    sum_radii = np.sum(final_r)
    
    return sum_radii, final_optimal_variables, refine_res.success, refine_res.message


def circle_packing21() -> np.ndarray:
    """
    Evolved function to place 21 non-overlapping circles inside a rectangle of perimeter 4,
    maximizing the sum of their radii.

    This implementation uses a hybrid optimization strategy inspired by state-of-the-art approaches:
    1. Global search with Differential Evolution (DE) and a penalty-based objective function.
    2. Local refinement with `trust-constr` method using explicit non-linear constraints.
    3. Multiple attempts are run in parallel using `joblib` to increase the chance of finding a
       better global optimum within the time budget.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y)
                 coordinates of the i-th circle and its radius r.
    """
    n_circles = 21
    base_random_seed = 42 # For reproducibility of the overall process

    # Decision variables order: [x_0...x_n-1, y_0...y_n-1, r_0...r_n-1, width]
    epsilon = 1e-7 # A very small positive number to ensure strict positivity
    
    bounds_x = [(epsilon, 2.0 - epsilon)] * n_circles
    bounds_y = [(epsilon, 2.0 - epsilon)] * n_circles
    bounds_r = [(epsilon, 0.18)] * n_circles # Tighter upper bound for radii, more realistic for 21 circles
    bounds_width = [(epsilon, 2.0 - epsilon)]
    
    all_bounds_list = bounds_x + bounds_y + bounds_r + bounds_width

    penalty_coeffs = (1e5, 1e7, 1e8) # (P_BOUNDARY, P_OVERLAP, P_MIN_RADIUS)

    num_restarts = max(1, os.cpu_count() or 1) 
    if num_restarts > 8: # Cap restarts to 8 to manage computational budget
        num_restarts = 8 
    
    de_maxiter_per_run = 380 # Increased for deeper global search, matching Insp2
    de_popsize_per_run = 35 # Increased for more diversity in the population, matching Insp1  
    tc_maxiter_per_run = 450 
    
    seeds = [base_random_seed + i for i in range(num_restarts)]

    results = Parallel(n_jobs=-1)(
        delayed(_run_optimization_attempt)(
            n_circles, seed, de_maxiter_per_run, de_popsize_per_run, tc_maxiter_per_run,
            all_bounds_list, penalty_coeffs, epsilon
        ) for seed in seeds
    )

    best_sum_radii = -np.inf
    best_optimal_variables = None

    for i, (sum_radii, optimal_variables, success, message) in enumerate(results):
        if sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_optimal_variables = optimal_variables

    if best_optimal_variables is None:
        return np.zeros((n_circles, 3))
    
    final_x, final_y, final_r, _, _ = _unpack_variables(best_optimal_variables, n_circles)

    final_circles = np.column_stack((final_x, final_y, final_r))
    
    final_circles[final_circles[:, 2] < epsilon, 2] = epsilon

    return final_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
