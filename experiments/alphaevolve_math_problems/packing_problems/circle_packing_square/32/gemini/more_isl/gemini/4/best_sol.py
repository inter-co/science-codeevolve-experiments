# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed # Added for parallel multi-start

# For reproducibility, set a fixed random seed for stochastic methods.
np.random.seed(42)

# --- Gradient & Constraint Functions (Vectorized approach from Inspiration 2) ---
def _objective(params):
    """Objective function: Minimize the negative sum of radii.
    Ensures radii are non-negative for sum calculation for robustness.
    """
    return -np.sum(np.maximum(0, params[2::3]))

def _jac_objective(params):
    """Jacobian of the objective function."""
    jac = np.zeros_like(params)
    jac[2::3] = -1.0
    return jac

def _all_containment_constraints(params, n_circles):
    """Returns an array of values for all 4*n containment constraints.
    Ensures non-negative radii for constraint evaluation.
    """
    x = params[0::3]
    y = params[1::3]
    r = np.maximum(0, params[2::3]) # Robustness: ensure non-negative radii
    return np.concatenate((x - r, 1 - x - r, y - r, 1 - y - r))

def _jac_all_containment_constraints(params, n_circles):
    """Jacobian for all containment constraints, matching the concatenated structure.
    For practical purposes with SLSQP and bounds, we assume r > 0.
    """
    n_vars = 3 * n_circles
    jac = np.zeros((4 * n_circles, n_vars))
    for i in range(n_circles):
        # Block for x_i - r_i >= 0
        jac[i, 3*i] = 1.0; jac[i, 3*i+2] = -1.0
        # Block for 1 - x_i - r_i >= 0
        jac[n_circles + i, 3*i] = -1.0; jac[n_circles + i, 3*i+2] = -1.0
        # Block for y_i - r_i >= 0
        jac[2*n_circles + i, 3*i+1] = 1.0; jac[2*n_circles + i, 3*i+2] = -1.0
        # Block for 1 - y_i - r_i >= 0
        jac[3*n_circles + i, 3*i+1] = -1.0; jac[3*n_circles + i, 3*i+2] = -1.0
    return jac

def _all_non_overlap_constraints(params, n_circles):
    """Returns an array for all n*(n-1)/2 non-overlap constraints.
    Ensures non-negative radii for constraint evaluation.
    """
    x, y, r = params[0::3], params[1::3], np.maximum(0, params[2::3]) # Robustness: ensure non-negative radii
    num_pairs = n_circles * (n_circles - 1) // 2
    constraints = np.empty(num_pairs)
    k = 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
            sum_radii_sq = (r[i] + r[j])**2
            constraints[k] = dist_sq - sum_radii_sq
            k += 1
    return constraints

def _jac_all_non_overlap_constraints(params, n_circles):
    """Jacobian for all non-overlap constraints."""
    x, y, r = params[0::3], params[1::3], np.maximum(0, params[2::3]) # Robustness: ensure non-negative radii
    n_vars = 3 * n_circles
    jac = np.zeros((n_circles * (n_circles - 1) // 2, n_vars))
    k = 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx, dy, sr = x[i] - x[j], y[i] - y[j], r[i] + r[j]
            jac[k, 3*i], jac[k, 3*i+1], jac[k, 3*i+2] = 2*dx, 2*dy, -2*sr
            jac[k, 3*j], jac[k, 3*j+1], jac[k, 3*j+2] = -2*dx, -2*dy, -2*sr
            k += 1
    return jac

def _grow_and_push_heuristic(n_circles: int, seed: int, max_main_iter: int = 300, max_inner_iter: int = 7) -> np.ndarray:
    """
    Generates an initial packing configuration using a physics-inspired grow-and-push heuristic.
    Adapted with parameters from Inspiration 1 and vectorized boundary logic from Inspiration 2.
    Accepts iteration counts to allow for different search depths.
    """
    np.random.seed(seed)

    # Heuristic parameters (tuned for N=32, inspired by Inspiration 1's performance)
    P1_max_main_iterations = max_main_iter
    P1_max_inner_iterations = max_inner_iter
    P1_growth_factor = 1.0008     # Aggressive growth
    P1_position_damping = 0.6     # Damping factor
    P1_min_displacement_threshold = 1e-7 # Stability threshold
    P1_min_overlap_threshold = 1e-8      # Overlap threshold
    P1_boundary_push_strength = 0.5 # Strength of boundary repulsion

    # Initialization on a grid with perturbation (from Inspiration 1)
    rows_grid = int(np.ceil(np.sqrt(n_circles)))
    cols_grid = int(np.ceil(n_circles / rows_grid))
    
    initial_r_base = 0.07 # Larger initial radius to start with more substantial circles
    x_coords = np.linspace(initial_r_base, 1 - initial_r_base, cols_grid)
    y_coords = np.linspace(initial_r_base, 1 - initial_r_base, rows_grid)
    initial_positions_grid = np.array([(x, y) for y in y_coords for x in x_coords])[:n_circles]

    perturbation_scale = 0.03 # Random perturbation to break symmetry
    initial_positions = initial_positions_grid + (np.random.rand(n_circles, 2) - 0.5) * perturbation_scale

    circles = np.zeros((n_circles, 3))
    circles[:, :2] = initial_positions
    circles[:, 2] = initial_r_base # Use the larger base radius

    best_circles = np.copy(circles)
    max_sum_radii = np.sum(circles[:, 2])

    for main_iter in range(P1_max_main_iterations):
        current_target_radii = circles[:, 2] * P1_growth_factor
        current_target_radii = np.clip(current_target_radii, 1e-9, 0.5)

        current_circles_pos = np.copy(circles[:, :2])
        current_circles_r = np.copy(current_target_radii)
        
        for inner_iter in range(P1_max_inner_iterations):
            displacements = np.zeros((n_circles, 2))
            max_current_overlap = 0.0
            
            # Vectorized Boundary Enforcement (from Inspiration 2)
            displacements[:, 0] += np.maximum(0, current_circles_r - current_circles_pos[:, 0]) * P1_boundary_push_strength
            displacements[:, 0] += np.minimum(0, (1 - current_circles_pos[:, 0]) - current_circles_r) * P1_boundary_push_strength
            displacements[:, 1] += np.maximum(0, current_circles_r - current_circles_pos[:, 1]) * P1_boundary_push_strength
            displacements[:, 1] += np.minimum(0, (1 - current_circles_pos[:, 1]) - current_circles_r) * P1_boundary_push_strength
            
            # Overlap Resolution
            for i in range(n_circles):
                for k in range(i + 1, n_circles):
                    pos1 = current_circles_pos[i]
                    r1 = current_circles_r[i]
                    pos2 = current_circles_pos[k]
                    r2 = current_circles_r[k]

                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    dist_sq = dx*dx + dy*dy
                    
                    sum_radii = r1 + r2
                    min_dist_required_sq = sum_radii * sum_radii

                    if dist_sq < min_dist_required_sq:
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0
                        overlap_amount = sum_radii - dist
                        max_current_overlap = max(max_current_overlap, overlap_amount)

                        direction = (pos1 - pos2) / (dist + 1e-9) if dist > 1e-9 else np.array([np.cos(np.random.rand() * 2 * np.pi), np.sin(np.random.rand() * 2 * np.pi)])

                        move_amount = overlap_amount * 0.5
                        displacements[i] += direction * move_amount
                        displacements[k] -= direction * move_amount
            
            current_circles_pos += displacements * P1_position_damping
            total_displacement_magnitude = np.sum(np.abs(displacements))

            # Vectorized radius clamping (from Inspiration 2)
            current_circles_r = np.minimum.reduce([current_target_radii, current_circles_pos[:, 0], 1 - current_circles_pos[:, 0], current_circles_pos[:, 1], 1 - current_circles_pos[:, 1]])
            current_circles_r = np.maximum(current_circles_r, 1e-9)
            
            if total_displacement_magnitude < P1_min_displacement_threshold and max_current_overlap < P1_min_overlap_threshold:
                break
        
        circles[:, :2] = current_circles_pos
        circles[:, 2] = current_circles_r

        current_sum_radii = np.sum(circles[:, 2])
        if current_sum_radii > max_sum_radii:
            max_sum_radii = current_sum_radii
            best_circles = np.copy(circles)
            
    return best_circles

def _final_cleanup_and_validate(circles: np.ndarray, n_circles: int) -> np.ndarray:
    """
    Ensures strict adherence to constraints by iteratively clamping radii and positions.
    Adapted from Inspiration 1 for robustness.
    """
    cleaned_circles = np.copy(circles)
    for _ in range(5): # A few iterations for final settling
        any_violation_found = False
        
        # 1. Enforce boundary and position constraints
        for i in range(n_circles):
            x, y, r = cleaned_circles[i]
            
            # Ensure position is within [r, 1-r]
            new_x = np.clip(x, r, 1 - r)
            new_y = np.clip(y, r, 1 - r)
            if new_x != x or new_y != y:
                cleaned_circles[i, 0] = new_x
                cleaned_circles[i, 1] = new_y
                any_violation_found = True
            
            # Re-evaluate radius based on potentially adjusted positions and boundaries
            new_r_boundary = min(r, cleaned_circles[i, 0], 1 - cleaned_circles[i, 0], cleaned_circles[i, 1], 1 - cleaned_circles[i, 1])
            if new_r_boundary < r - 1e-12: # Use a very small tolerance for change
                cleaned_circles[i, 2] = max(new_r_boundary, 1e-9) # Ensure positive radius
                any_violation_found = True
            else:
                cleaned_circles[i, 2] = max(r, 1e-9) # Just ensure positive

        # 2. Enforce non-overlap constraints (Optimized with squared distance check)
        for i in range(n_circles):
            for k in range(i + 1, n_circles):
                pos1 = cleaned_circles[i, :2]
                r1 = cleaned_circles[i, 2]
                pos2 = cleaned_circles[k, :2]
                r2 = cleaned_circles[k, 2]

                dx = pos1[0] - pos2[0]; dy = pos1[1] - pos2[1]
                dist_sq = dx*dx + dy*dy
                sum_radii = r1 + r2
                min_dist_required_sq = sum_radii * sum_radii

                if dist_sq < min_dist_required_sq - 1e-12: # Overlap detected
                    dist = np.sqrt(dist_sq)
                    shrink_factor = (dist / sum_radii) if sum_radii > 1e-9 else 0
                    cleaned_circles[i, 2] = max(r1 * shrink_factor, 1e-9)
                    cleaned_circles[k, 2] = max(r2 * shrink_factor, 1e-9)
                    any_violation_found = True
        
        if not any_violation_found:
            break
    return cleaned_circles

def _run_exploration_task(seed: int, n: int, bounds: list, constraints: list):
    """Stage 1: Runs a fast exploratory search from a random initial guess."""
    np.random.seed(seed)
    try:
        # Fast heuristic for broad exploration
        x0 = _grow_and_push_heuristic(n, seed=seed, max_main_iter=300, max_inner_iter=7).flatten()
        # Moderate optimization
        options = {'maxiter': 1500, 'ftol': 1e-9, 'disp': False, 'eps': 1e-10}
        res = minimize(fun=_objective, x0=x0, method='SLSQP', jac=_jac_objective,
                       bounds=bounds, constraints=constraints, options=options)
        
        circles = _final_cleanup_and_validate(res.x.reshape((n, 3)), n)
        return circles, np.sum(circles[:, 2])
    except Exception:
        return np.zeros((n, 3)), -np.inf

def _run_refinement_task(seed: int, n: int, bounds: list, constraints: list, base_circles: np.ndarray):
    """Stage 2: Runs a deep search starting from a perturbed version of the best known solution."""
    np.random.seed(seed)
    try:
        x0 = base_circles.flatten().copy()
        
        # Apply small random perturbations to escape the local optimum
        x_perturb = np.random.uniform(-0.005, 0.005, n)
        y_perturb = np.random.uniform(-0.005, 0.005, n)
        r_perturb = np.random.uniform(-0.002, 0.002, n)
        x0[0::3] += x_perturb
        x0[1::3] += y_perturb
        x0[2::3] += r_perturb

        # Clip perturbed values to stay within valid bounds
        for i in range(n):
            x0[3*i:3*i+2] = np.clip(x0[3*i:3*i+2], 0.0, 1.0)
            x0[3*i+2] = np.clip(x0[3*i+2], 1e-9, 0.5)

        # Deeper optimization for refinement
        options = {'maxiter': 3000, 'ftol': 1e-10, 'disp': False, 'eps': 1e-10}
        res = minimize(fun=_objective, x0=x0, method='SLSQP', jac=_jac_objective,
                       bounds=bounds, constraints=constraints, options=options)

        circles = _final_cleanup_and_validate(res.x.reshape((n, 3)), n)
        return circles, np.sum(circles[:, 2])
    except Exception:
        return np.zeros((n, 3)), -np.inf

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This implementation uses a two-stage parallel optimization process inspired by Insp. 2:
    1. Broad Exploration: Many fast parallel runs to find a good solution basin.
    2. Focused Refinement: The best solution from Stage 1 is perturbed and re-optimized
       in parallel to deeply search the most promising region.
    """
    n = 32
    N_EXPLORE_RUNS = 16
    N_REFINE_RUNS = 8
    
    constraints = [
        {'type': 'ineq', 'fun': _all_containment_constraints, 'jac': _jac_all_containment_constraints, 'args': (n,)},
        {'type': 'ineq', 'fun': _all_non_overlap_constraints, 'jac': _jac_all_non_overlap_constraints, 'args': (n,)}
    ]
    bounds = [(0, 1), (0, 1), (1e-9, 0.5)] * n

    # --- Stage 1: Broad Exploration ---
    explore_seeds = [42 + i * 101 for i in range(N_EXPLORE_RUNS)]
    explore_results = Parallel(n_jobs=-1, verbose=0)(
        delayed(_run_exploration_task)(seed, n, bounds, constraints) for seed in explore_seeds
    )
    
    best_circles_stage1 = None
    max_sum_radii_stage1 = -np.inf
    for circles, sum_radii in explore_results:
        if sum_radii > max_sum_radii_stage1:
            max_sum_radii_stage1 = sum_radii
            best_circles_stage1 = circles.copy()

    if best_circles_stage1 is None: # Handle case where all exploration runs fail
        return np.zeros((n, 3))

    # --- Stage 2: Focused Refinement ---
    refine_seeds = [777 + i * 101 for i in range(N_REFINE_RUNS)]
    refine_results = Parallel(n_jobs=-1, verbose=0)(
        delayed(_run_refinement_task)(seed, n, bounds, constraints, best_circles_stage1) for seed in refine_seeds
    )
    
    best_circles_stage2 = None
    max_sum_radii_stage2 = -np.inf
    for circles, sum_radii in refine_results:
        if sum_radii > max_sum_radii_stage2:
            max_sum_radii_stage2 = sum_radii
            best_circles_stage2 = circles.copy()

    # --- Final Selection ---
    if max_sum_radii_stage2 > max_sum_radii_stage1:
        return best_circles_stage2
    else:
        return best_circles_stage1


# EVOLVE-BLOCK-END
