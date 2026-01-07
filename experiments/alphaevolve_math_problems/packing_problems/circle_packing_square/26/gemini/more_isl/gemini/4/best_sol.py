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

def inflate_radii(circles: np.ndarray) -> np.ndarray:
    """
    Post-processing step: greedily inflates all circle radii by the maximum possible
    common factor until they touch a boundary or another circle.
    """
    if circles is None or circles.shape[0] == 0:
        return circles
    
    n = circles.shape[0]
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]

    non_zero_radii_mask = rs > 1e-9
    if not np.any(non_zero_radii_mask):
        return circles

    r_safe = np.where(non_zero_radii_mask, rs, 1e-9)
    
    alpha_contain = np.min(np.concatenate([
        xs[non_zero_radii_mask] / r_safe[non_zero_radii_mask],
        (UNIT_SQUARE_SIDE - xs[non_zero_radii_mask]) / r_safe[non_zero_radii_mask],
        ys[non_zero_radii_mask] / r_safe[non_zero_radii_mask],
        (UNIT_SQUARE_SIDE - ys[non_zero_radii_mask]) / r_safe[non_zero_radii_mask]
    ]))

    alpha_overlap = np.inf
    if n > 1:
        centers = circles[:, :2]
        dist_vec = pdist(centers, 'euclidean')
        
        radii_sum_matrix = rs[:, np.newaxis] + rs
        radii_sum_vec = radii_sum_matrix[np.triu_indices(n, k=1)]
        
        valid_pairs = radii_sum_vec > 1e-9
        if np.any(valid_pairs):
            alpha_overlap = np.min(dist_vec[valid_pairs] / radii_sum_vec[valid_pairs])

    max_alpha = min(alpha_contain, alpha_overlap)

    inflated_circles = circles.copy()
    if max_alpha > 1.0:
        inflated_circles[:, 2] *= max_alpha
    
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
