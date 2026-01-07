# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed # New import for parallelization

# For reproducibility, set a fixed random seed for stochastic methods.
# This global seed is for the main process, individual runs will use their own seeds.
np.random.seed(42)

# --- Helper functions for Optimization (adapted from Inspiration Programs) ---
# These functions are defined globally to be accessible by scipy.optimize and for clarity.

def _objective(params):
    """Objective function: Minimize the negative sum of radii."""
    # Inspired by Insp. 1/2: Ensure radii are non-negative for sum calculation
    radii = np.maximum(0, params[2::3])
    return -np.sum(radii)

def _jac_objective(params):
    """Jacobian of the objective function."""
    jac = np.zeros_like(params)
    jac[2::3] = -1.0
    return jac

def _all_containment_constraints(params, n_circles):
    """Returns an array of values for all 4*n containment constraints. (from Insp. 1)"""
    x = params[0::3]
    y = params[1::3]
    # Inspired by Insp. 1/2: Ensure non-negative radii for constraint evaluation
    r = np.maximum(0, params[2::3])
    return np.concatenate((x - r, 1 - x - r, y - r, 1 - y - r))

def _jac_all_containment_constraints(params, n_circles):
    """Jacobian for all containment constraints, matching concatenated structure. (from Insp. 1)"""
    n_vars = 3 * n_circles
    jac = np.zeros((4 * n_circles, n_vars))
    # Note: `np.maximum(0, ...)` makes radius non-differentiable at 0, but this is
    # handled by the bounds and is not a practical issue for SLSQP.
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
    """Returns an array for all n*(n-1)/2 non-overlap constraints. (from Insp. 1)"""
    x, y = params[0::3], params[1::3]
    # Inspired by Insp. 1/2: Ensure non-negative radii
    r = np.maximum(0, params[2::3])
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
    """Jacobian for all non-overlap constraints. (from Insp. 1)"""
    x, y = params[0::3], params[1::3]
    # Inspired by Insp. 1/2: Ensure non-negative radii
    r = np.maximum(0, params[2::3])
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

def _grow_and_push_heuristic(n_circles: int, seed: int = 42) -> np.ndarray:
    """
    Generates an initial packing configuration using a physics-inspired grow-and-push heuristic.
    This version uses a vectorized collision detection loop for performance.
    (Adopted and refined from Inspiration Program 1)
    """
    np.random.seed(seed)
    # Heuristic parameters tuned based on analysis of inspiration programs.
    # A slower growth rate and larger initial radius may yield better starting points.
    P1_max_main_iterations = 500      # Compromise for deeper search
    P1_max_inner_iterations = 7       # Keep as is
    P1_growth_factor = 1.0008         # Slower, more stable growth (from Insp. 1)
    P1_position_damping = 0.6         # (from Insp. 1)
    P1_min_displacement_threshold = 1e-8 # Keep target's tighter threshold
    P1_min_overlap_threshold = 1e-8      # Keep as is
    P1_boundary_push_strength = 0.5      # Keep as is

    rows_grid = int(np.ceil(np.sqrt(n_circles)))
    cols_grid = int(np.ceil(n_circles / rows_grid))
    initial_r_base = 0.07             # Larger initial radius (from Insp. 1)
    x_coords = np.linspace(initial_r_base, 1 - initial_r_base, cols_grid)
    y_coords = np.linspace(initial_r_base, 1 - initial_r_base, rows_grid)
    initial_positions_grid = np.array([(x, y) for y in y_coords for x in x_coords])[:n_circles]
    perturbation_scale = 0.03         # Smaller perturbation (from Insp. 1)
    initial_positions = initial_positions_grid + (np.random.rand(n_circles, 2) - 0.5) * perturbation_scale

    circles = np.zeros((n_circles, 3))
    circles[:, :2] = initial_positions
    circles[:, 2] = initial_r_base

    best_circles = np.copy(circles)
    max_sum_radii = np.sum(circles[:, 2])

    for _ in range(P1_max_main_iterations):
        circles[:, 2] *= P1_growth_factor
        circles[:, 2] = np.clip(circles[:, 2], 1e-9, 0.5)
        current_circles_pos = np.copy(circles[:, :2])
        current_circles_r = np.copy(circles[:, 2])
        
        for __ in range(P1_max_inner_iterations):
            max_current_overlap = 0.0
            displacements = np.zeros((n_circles, 2))
            
            # Vectorized Boundary Handling (forces and radius clipping) (from Insp. 1)
            x_coords, y_coords = current_circles_pos[:, 0], current_circles_pos[:, 1]
            radii = current_circles_r

            # Apply boundary forces
            left_violations = radii - x_coords
            right_violations = x_coords + radii - 1
            bottom_violations = radii - y_coords
            top_violations = y_coords + radii - 1

            displacements[:, 0] += np.where(left_violations > 0, left_violations, 0) * P1_boundary_push_strength
            displacements[:, 0] -= np.where(right_violations > 0, right_violations, 0) * P1_boundary_push_strength
            displacements[:, 1] += np.where(bottom_violations > 0, bottom_violations, 0) * P1_boundary_push_strength
            displacements[:, 1] -= np.where(top_violations > 0, top_violations, 0) * P1_boundary_push_strength
            
            # Update radii for boundary containment
            current_circles_r = np.minimum(radii, x_coords)
            current_circles_r = np.minimum(current_circles_r, 1 - x_coords)
            current_circles_r = np.minimum(current_circles_r, y_coords)
            current_circles_r = np.minimum(current_circles_r, 1 - y_coords)
            current_circles_r = np.maximum(current_circles_r, 1e-9)

            pos = current_circles_pos
            radii = current_circles_r
            
            diffs = pos[:, np.newaxis, :] - pos
            dist_sq = np.sum(diffs**2, axis=2)
            sum_radii = radii[:, np.newaxis] + radii
            
            upper_tri_mask = np.triu(np.ones((n_circles, n_circles), dtype=bool), k=1)
            overlap_mask = (dist_sq < sum_radii**2) & upper_tri_mask
            
            i_indices, j_indices = np.where(overlap_mask)
            
            if i_indices.size > 0:
                vec = pos[i_indices] - pos[j_indices]
                d_sq = dist_sq[i_indices, j_indices]
                d = np.sqrt(d_sq)
                
                overlap_amount = sum_radii[i_indices, j_indices] - d
                max_current_overlap = np.max(overlap_amount)
                
                direction = vec / (d[:, np.newaxis] + 1e-9)
                
                zero_dist_mask = d < 1e-9
                if np.any(zero_dist_mask):
                    num_zero_dist = np.sum(zero_dist_mask)
                    rand_angles = np.random.uniform(0, 2 * np.pi, num_zero_dist)
                    direction[zero_dist_mask] = np.stack([np.cos(rand_angles), np.sin(rand_angles)], axis=1)

                move_vectors = direction * (overlap_amount[:, np.newaxis] * 0.5)
                
                np.add.at(displacements, i_indices, move_vectors)
                np.add.at(displacements, j_indices, -move_vectors)

            current_circles_pos += displacements * P1_position_damping
            total_displacement_magnitude = np.sum(np.abs(displacements))
            
            if total_displacement_magnitude < P1_min_displacement_threshold and max_current_overlap < P1_min_overlap_threshold:
                break
        
        circles[:, :2] = current_circles_pos
        circles[:, 2] = current_circles_r
        current_sum_radii = np.sum(circles[:, 2])
        if current_sum_radii > max_sum_radii:
            max_sum_radii = current_sum_radii
            best_circles = np.copy(circles)
            
    for _ in range(5):
        any_violation_found = False
        for i in range(n_circles):
            x, y, r = best_circles[i]
            new_x, new_y = np.clip(x, r, 1 - r), np.clip(y, r, 1 - r)
            if new_x != x or new_y != y:
                best_circles[i, 0], best_circles[i, 1] = new_x, new_y
                any_violation_found = True
            new_r_boundary = min(r, best_circles[i, 0], 1 - best_circles[i, 0], best_circles[i, 1], 1 - best_circles[i, 1])
            if new_r_boundary < r - 1e-10:
                best_circles[i, 2] = max(new_r_boundary, 1e-9)
                any_violation_found = True
        for i in range(n_circles):
            for k in range(i + 1, n_circles):
                pos1, r1 = best_circles[i, :2], best_circles[i, 2]
                pos2, r2 = best_circles[k, :2], best_circles[k, 2]
                dx, dy = pos1[0] - pos2[0], pos1[1] - pos2[1]
                dist_sq = dx*dx + dy*dy
                sum_radii = r1 + r2
                min_dist_required_sq = sum_radii * sum_radii
                if dist_sq < min_dist_required_sq - 1e-10:
                    dist = np.sqrt(dist_sq)
                    shrink_factor = (dist / sum_radii) if sum_radii > 1e-9 else 0
                    best_circles[i, 2] = max(r1 * shrink_factor, 1e-9)
                    best_circles[k, 2] = max(r2 * shrink_factor, 1e-9)
                    any_violation_found = True
        if not any_violation_found:
            break
    return best_circles

def _final_cleanup_and_validate(circles: np.ndarray, n_circles: int) -> np.ndarray:
    """Ensures strict adherence to constraints by iteratively clamping radii. (from Insp. 1)"""
    for _ in range(10):
        any_violation = False
        radii = circles[:, 2]
        # Clip positions to be valid for the current radii
        circles[:, 0] = np.clip(circles[:, 0], radii, 1 - radii)
        circles[:, 1] = np.clip(circles[:, 1], radii, 1 - radii)
        
        # Shrink radii if they violate boundary constraints
        new_radii = np.minimum.reduce([circles[:, 2], circles[:, 0], 1 - circles[:, 0], circles[:, 1], 1 - circles[:, 1]])
        if np.any(new_radii < circles[:, 2] - 1e-12): any_violation = True
        circles[:, 2] = np.maximum(1e-9, new_radii)

        # Shrink radii if they violate non-overlap constraints
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.linalg.norm(circles[i, :2] - circles[j, :2])
                sum_r = circles[i, 2] + circles[j, 2]
                if dist < sum_r - 1e-12:
                    any_violation = True
                    # Shrink both circles proportionally to resolve overlap
                    shrink = dist / sum_r if sum_r > 1e-9 else 0.0
                    circles[i, 2] *= shrink
                    circles[j, 2] *= shrink
        
        circles[:, 2] = np.maximum(1e-9, circles[:, 2])
        if not any_violation: break
    return circles

def _run_optimization_task(seed: int, n: int, bounds: list, constraints: list):
    """
    Runs a single full optimization chain: heuristic -> SLSQP -> cleanup.
    Designed to be executed in parallel. (Structure from Insp. 1)
    """
    np.random.seed(seed)

    try:
        # 1. Generate initial guess using the fast vectorized heuristic
        x0 = _grow_and_push_heuristic(n, seed=seed).flatten()
        
        # 2. Refine with SLSQP optimizer with tighter tolerances (from Insp. 1)
        options = {'maxiter': 3000, 'ftol': 1e-10, 'disp': False, 'eps': 1e-10}
        res = minimize(fun=_objective, x0=x0, method='SLSQP', jac=_jac_objective,
                       bounds=bounds, constraints=constraints, options=options)
        
        # 3. Apply final robust cleanup and validation (from Insp. 1)
        current_circles = _final_cleanup_and_validate(res.x.reshape((n, 3)), n)
        current_sum_radii = np.sum(current_circles[:, 2])
        
        return current_circles, current_sum_radii
    except Exception:
        # Gracefully handle any optimization failures in a parallel worker
        return np.zeros((n, 3)), -np.inf

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This implementation uses a parallel multi-start hybrid approach, combining the best
    elements from the inspiration programs:
    1. A fast, vectorized 'grow-and-push' heuristic generates diverse initial configurations.
    2. Scipy's 'SLSQP' optimizer with robust, vectorized constraints refines each configuration.
    3. The best result from all parallel runs is selected.
    """
    n = 32
    
    # We can afford many parallel runs due to the efficient heuristic and vectorized functions.
    N_PARALLEL_RUNS = 140 # Increased from 60 to fully utilize the 60-second time budget (from Insp. 1)
    seeds = [42 + i * 101 for i in range(N_PARALLEL_RUNS)]

    # Define constraints and bounds once, using the clean structure from Insp. 1
    constraints = [
        {'type': 'ineq', 'fun': _all_containment_constraints, 'jac': _jac_all_containment_constraints, 'args': (n,)},
        {'type': 'ineq', 'fun': _all_non_overlap_constraints, 'jac': _jac_all_non_overlap_constraints, 'args': (n,)}
    ]
    bounds = [(0, 1), (0, 1), (1e-9, 0.5)] * n

    # Execute optimization tasks in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_run_optimization_task)(seed, n, bounds, constraints) for seed in seeds
    )

    # Find the best result among all parallel runs
    best_circles_overall = None
    max_sum_radii_overall = -np.inf
    
    for circles, sum_radii in results:
        if sum_radii > max_sum_radii_overall:
            max_sum_radii_overall = sum_radii
            best_circles_overall = circles

    return best_circles_overall if best_circles_overall is not None else np.zeros((n, 3))


# EVOLVE-BLOCK-END
