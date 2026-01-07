# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from joblib import Parallel, delayed

# --- Constants and Configuration ---
N_CIRCLES = 26
UNIT_SQUARE_SIDE = 1.0
RANDOM_SEED = 42 # Main seed for reproducibility

# --- Helper Functions for Parallel SLSQP ---

def generate_initial_guess(n_circles: int, local_rng: np.random.Generator):
    """
    Generates an initial guess for circle packing parameters (x, y, r)
    based on a perturbed grid layout. Increased perturbation for more diverse starts.
    """
    num_cols = 5
    num_rows = 6
    initial_r_estimate = min(UNIT_SQUARE_SIDE / (2 * num_cols), UNIT_SQUARE_SIDE / (2 * num_rows))
    
    params_list = []
    current_circle_count = 0
    
    for r_idx in range(num_rows):
        for c_idx in range(num_cols):
            if current_circle_count >= n_circles: break
            
            x_center = (c_idx + 0.5) * UNIT_SQUARE_SIDE / num_cols
            y_center = (r_idx + 0.5) * UNIT_SQUARE_SIDE / num_rows
            
            # Increased perturbation for more diverse initial guesses (inspired by analysis)
            x_perturb = local_rng.uniform(-0.04, 0.04)
            y_perturb = local_rng.uniform(-0.04, 0.04)
            initial_radius = initial_r_estimate * local_rng.uniform(0.9, 1.1)
            
            params_list.extend([x_center + x_perturb, y_center + y_perturb, initial_radius])
            current_circle_count += 1
        if current_circle_count >= n_circles: break
            
    return np.array(params_list).flatten()


def objective(params):
    """ Objective: Minimize the negative sum of radii to maximize the sum of radii. """
    return -np.sum(params[2::3])


def constraints_func(params):
    """
    Vectorized constraint function (g(x) >= 0).
    """
    circles_data = params.reshape(N_CIRCLES, 3)
    xs, ys, rs = circles_data[:, 0], circles_data[:, 1], circles_data[:, 2]
    
    containment_and_radii = np.concatenate([
        xs - rs, (UNIT_SQUARE_SIDE - xs) - rs,
        ys - rs, (UNIT_SQUARE_SIDE - ys) - rs,
        rs
    ])
    
    centers = circles_data[:, :2]
    dist_sq_matrix = squareform(pdist(centers, 'sqeuclidean'))
    radii_sum = rs[:, np.newaxis] + rs
    
    i_upper, j_upper = np.triu_indices(N_CIRCLES, k=1)
    non_overlap = dist_sq_matrix[i_upper, j_upper] - (radii_sum[i_upper, j_upper]**2)
    
    return np.concatenate([containment_and_radii, non_overlap])


def is_valid_packing(circles_data: np.ndarray, epsilon=1e-7) -> bool:
    """ Checks if a given set of circles forms a strictly valid packing with tolerance. """
    n = circles_data.shape[0]
    if n == 0: return True
    xs, ys, rs = circles_data[:, 0], circles_data[:, 1], circles_data[:, 2]

    if np.any(rs < epsilon): return False
    if not (np.all(rs - epsilon <= xs) and np.all(xs <= UNIT_SQUARE_SIDE - rs + epsilon) and
            np.all(rs - epsilon <= ys) and np.all(ys <= UNIT_SQUARE_SIDE - rs + epsilon)):
        return False

    if n > 1:
        centers = circles_data[:, :2]
        dist_sq_vec = pdist(centers, 'sqeuclidean')
        radii_sum = rs[:, np.newaxis] + rs
        min_dist_sq_vec = (radii_sum[np.triu_indices(n, k=1)])**2
        if np.any(dist_sq_vec < min_dist_sq_vec - epsilon):
            return False
    return True


def run_slsqp_worker(seed: int, bounds: Bounds, nonlinear_constraints: NonlinearConstraint, n_circles: int):
    """ Worker function for a single parallel SLSQP optimization run. """
    local_rng = np.random.default_rng(seed)
    initial_guess = generate_initial_guess(n_circles, local_rng)

    result = minimize(
        fun=objective, x0=initial_guess, method='SLSQP', bounds=bounds, constraints=[nonlinear_constraints],
        # Increased maxiter for more thorough convergence (inspired by analysis)
        options={'maxiter': 9000, 'ftol': 1e-9, 'disp': False}
    )

    current_circles = result.x.reshape(n_circles, 3)
    current_circles[:, 2] = np.maximum(0, current_circles[:, 2])

    if result.success and is_valid_packing(current_circles):
        return current_circles, -objective(current_circles.flatten())
    return None, -np.inf

def inflate_radii(circles: np.ndarray, max_iter: int = 25, tol: float = 1e-12) -> np.ndarray:
    """
    Post-processing step: greedily and iteratively inflates each circle's radius
    individually until convergence. This is more powerful than inflating all radii
    by a single common factor, as it allows circles in sparser regions to grow more.
    This method is a form of coordinate ascent on the radii variables.
    """
    if circles is None or circles.shape[0] == 0:
        return circles
    
    inflated_circles = circles.copy()
    n = inflated_circles.shape[0]
    
    # Use a fixed order for determinism
    indices = np.arange(n)

    for _ in range(max_iter):
        radii_before_iter = inflated_circles[:, 2].copy()
        
        for i in indices:
            xi, yi, _ = inflated_circles[i]
            
            # Constraint from walls: r <= x, r <= 1-x, etc.
            r_max_wall = min(xi, UNIT_SQUARE_SIDE - xi, yi, UNIT_SQUARE_SIDE - yi)
            
            # Constraint from other circles: r_i <= dist(i, j) - r_j
            r_max_overlap = np.inf
            if n > 1:
                # Create a mask to select all circles except the current one
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                
                # Vectorized distance calculation to all other circles
                other_centers = inflated_circles[mask, :2]
                other_radii = inflated_circles[mask, 2]
                dists = np.sqrt(np.sum((other_centers - inflated_circles[i, :2])**2, axis=1))
                
                # The maximum this radius can be is dist - r_other
                possible_radii = dists - other_radii
                
                # It's possible for a value to be negative if the input is already overlapping.
                # We only consider positive possibilities; if none, the circle is trapped.
                if np.any(possible_radii > 0):
                    r_max_overlap = np.min(possible_radii[possible_radii > 0])
                else:
                    r_max_overlap = 0.0

            # The new radius is the tightest of all constraints
            inflated_circles[i, 2] = min(r_max_wall, r_max_overlap)
            
        # Check for convergence across all radii after a full pass
        if np.sum(np.abs(inflated_circles[:, 2] - radii_before_iter)) < tol:
            break
            
    return inflated_circles


def circle_packing26() -> np.ndarray:
    """
    Finds an optimal circle packing using parallel SLSQP with post-processing inflation.
    Combines multi-start optimization for global search with a greedy inflation step for
    local refinement, aiming to surpass the benchmark.
    """
    main_rng = np.random.default_rng(RANDOM_SEED)
    
    lower_bounds = np.zeros(3 * N_CIRCLES)
    upper_bounds = np.ones(3 * N_CIRCLES)
    upper_bounds[2::3] = 0.5
    bounds = Bounds(lower_bounds, upper_bounds)

    nonlinear_constraints = NonlinearConstraint(constraints_func, 0, np.inf)

    # Increased starts to better utilize the time budget
    num_starts = 600
    seeds = main_rng.integers(0, 2**32 - 1, size=num_starts)
    
    results = Parallel(n_jobs=-1)(
        delayed(run_slsqp_worker)(seed, bounds, nonlinear_constraints, N_CIRCLES) for seed in seeds
    )
    
    best_sum_radii = -np.inf
    best_circles = None
    
    for circles, sum_radii in results:
        if circles is not None and sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_circles = circles

    if best_circles is None:
        # Robust fallback inspired by analysis
        print("Warning: No valid solution found in parallel runs. Attempting a fallback instance.")
        fallback_seed = main_rng.integers(0, 2**32 - 1)
        best_circles, _ = run_slsqp_worker(fallback_seed, bounds, nonlinear_constraints, N_CIRCLES)
        if best_circles is None:
            print("Error: Fallback also failed. Returning zero array.")
            return np.zeros((N_CIRCLES, 3))

    # Apply post-processing inflation to the best result
    final_circles = inflate_radii(best_circles)

    return final_circles


# EVOLVE-BLOCK-END
