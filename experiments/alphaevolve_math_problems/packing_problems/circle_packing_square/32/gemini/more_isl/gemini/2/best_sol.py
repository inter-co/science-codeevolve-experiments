# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed # Added for parallel multi-start

# For reproducibility, set a fixed random seed for stochastic methods.
np.random.seed(42)

# --- Gradient & Constraint Functions (Vectorized approach from Inspiration programs) ---
def _objective(params):
    """Objective function: Minimize the negative sum of radii.
    Ensure radii are non-negative for sum calculation for robustness."""
    return -np.sum(np.maximum(0, params[2::3]))

def _jac_objective(params):
    """Jacobian of the objective function."""
    jac = np.zeros_like(params)
    # The derivative of max(0, r) is 1 for r>0, 0 for r<0. Since bounds enforce r > 1e-9,
    # we can assume r > 0 and the derivative is -1.0.
    jac[2::3] = -1.0
    return jac

def _all_containment_constraints(params, n_circles):
    """Returns an array of values for all 4*n containment constraints.
    Ensure radii are non-negative for constraint evaluation for robustness."""
    x = params[0::3]
    y = params[1::3]
    r = np.maximum(0, params[2::3]) 
    return np.concatenate((x - r, 1 - x - r, y - r, 1 - y - r))

def _jac_all_containment_constraints(params, n_circles):
    """Jacobian for all containment constraints, matching the concatenated structure.
    Assumes r > 0 due to bounds and the 'maximum' safeguard in _all_containment_constraints."""
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
    Ensure radii are non-negative for constraint evaluation for robustness."""
    x, y, r = params[0::3], params[1::3], np.maximum(0, params[2::3])
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
    x, y, r = params[0::3], params[1::3], params[2::3]
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

def _grow_and_push_heuristic(n_circles: int, seed: int) -> np.ndarray:
    """Generates a good initial packing using a physics-inspired heuristic."""
    np.random.seed(seed)
    params = {'max_main_iter': 300, 'max_inner_iter': 7, 'growth': 1.0008, 'damp': 0.6,
              'disp_thresh': 1e-7, 'overlap_thresh': 1e-8, 'bound_push': 0.5}
    
    rows, cols = int(np.ceil(np.sqrt(n_circles))), int(np.ceil(n_circles / int(np.ceil(np.sqrt(n_circles)))))
    initial_r = 0.07
    x_coords = np.linspace(initial_r, 1 - initial_r, cols)
    y_coords = np.linspace(initial_r, 1 - initial_r, rows)
    
    positions = np.array([(x, y) for y in y_coords for x in x_coords])[:n_circles]
    positions += (np.random.rand(n_circles, 2) - 0.5) * 0.03

    circles = np.zeros((n_circles, 3))
    circles[:, :2], circles[:, 2] = positions, initial_r
    best_circles, max_sum_r = np.copy(circles), np.sum(circles[:, 2])

    for _ in range(params['max_main_iter']):
        target_r = np.clip(circles[:, 2] * params['growth'], 1e-9, 0.5)
        pos, r = np.copy(circles[:, :2]), np.copy(target_r)

        for _ in range(params['max_inner_iter']):
            disp = np.zeros((n_circles, 2))
            max_overlap = 0.0
            
            # Vectorized Boundary Enforcement (adapted from inspiration programs)
            # Push circles away from boundaries if they overlap
            disp[:, 0] += np.maximum(0, r - pos[:, 0]) * params['bound_push']
            disp[:, 0] -= np.maximum(0, pos[:, 0] + r - 1) * params['bound_push']
            disp[:, 1] += np.maximum(0, r - pos[:, 1]) * params['bound_push']
            disp[:, 1] -= np.maximum(0, pos[:, 1] + r - 1) * params['bound_push']
            
            # Overlap Resolution (optimized loop, with robust zero-distance handling)
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    d_vec = pos[i] - pos[j]
                    dist_sq = d_vec @ d_vec
                    sum_r = r[i] + r[j]
                    
                    if dist_sq < sum_r * sum_r: # Overlap detected
                        dist = np.sqrt(dist_sq)
                        overlap = sum_r - dist
                        max_overlap = max(max_overlap, overlap)
                        
                        # Handle zero distance robustly (adapted from Inspiration 1)
                        if dist < 1e-9: # If centers are almost identical
                            # Assign a random direction to push apart
                            angle = np.random.rand() * 2 * np.pi
                            direction = np.array([np.cos(angle), np.sin(angle)])
                        else:
                            direction = d_vec / dist
                        
                        move_amount = overlap * 0.5
                        disp[i] += direction * move_amount
                        disp[j] -= direction * move_amount
            
            pos += disp * params['damp']
            r = np.minimum.reduce([target_r, pos[:, 0], 1 - pos[:, 0], pos[:, 1], 1 - pos[:, 1]])
            r = np.maximum(r, 1e-9)

            if np.sum(np.abs(disp)) < params['disp_thresh'] and max_overlap < params['overlap_thresh']:
                break
        
        circles[:, :2], circles[:, 2] = pos, r
        if np.sum(circles[:, 2]) > max_sum_r:
            max_sum_r = np.sum(circles[:, 2])
            best_circles = np.copy(circles)
            
    return best_circles

def _final_cleanup_and_validate(circles: np.ndarray, n_circles: int) -> np.ndarray:
    """
    Ensures strict adherence to constraints by iteratively clamping radii and positions.
    Adapted for robustness and numerical stability (from Inspiration 1).
    """
    cleaned_circles = np.copy(circles)
    for _ in range(10): # Number of iterations for final settling
        any_violation_found = False
        
        # 1. Enforce boundary and position constraints
        # Clip positions first, then re-evaluate radii based on new positions
        original_radii = cleaned_circles[:, 2].copy()
        
        # Clip positions to ensure centers are at least 'r' from boundary
        # This implicitly handles the x_i >= r_i and x_i <= 1-r_i type constraints
        cleaned_circles[:, 0] = np.clip(cleaned_circles[:, 0], original_radii, 1 - original_radii)
        cleaned_circles[:, 1] = np.clip(cleaned_circles[:, 1], original_radii, 1 - original_radii)
        
        # Re-evaluate radius based on current positions and boundaries
        new_radii_from_bounds = np.minimum.reduce([
            original_radii, # Current radius (don't grow beyond it)
            cleaned_circles[:, 0], # Distance to left boundary
            1 - cleaned_circles[:, 0], # Distance to right boundary
            cleaned_circles[:, 1], # Distance to bottom boundary
            1 - cleaned_circles[:, 1] # Distance to top boundary
        ])
        
        # If any radius had to shrink due to boundary issues, mark violation
        if np.any(new_radii_from_bounds < cleaned_circles[:, 2] - 1e-12):
            any_violation_found = True
        
        cleaned_circles[:, 2] = np.maximum(new_radii_from_bounds, 1e-9) # Ensure positive radii

        # 2. Enforce non-overlap constraints
        for i in range(n_circles):
            for k in range(i + 1, n_circles):
                pos1 = cleaned_circles[i, :2]
                r1 = cleaned_circles[i, 2]
                pos2 = cleaned_circles[k, :2]
                r2 = cleaned_circles[k, 2]

                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]
                dist_sq = dx*dx + dy*dy
                sum_radii = r1 + r2
                min_dist_required_sq = sum_radii * sum_radii

                if dist_sq < min_dist_required_sq - 1e-12: # Overlap detected with tolerance
                    any_violation_found = True
                    dist = np.sqrt(dist_sq)
                    
                    # Calculate shrink factor to resolve overlap
                    shrink_factor = (dist / sum_radii) if sum_radii > 1e-9 else 0.0
                    
                    # Apply shrink factor to current radii. Ensure minimum radius.
                    cleaned_circles[i, 2] = np.maximum(r1 * shrink_factor, 1e-9)
                    cleaned_circles[k, 2] = np.maximum(r2 * shrink_factor, 1e-9)
        
        if not any_violation_found:
            break # No violations found, stop iterating
            
    return cleaned_circles

def _run_optimization_task(seed: int, n: int, bounds: list, constraints: list):
    """
    Runs a single full optimization chain: heuristic -> SLSQP -> cleanup.
    Designed to be executed in parallel.
    """
    np.random.seed(seed)
    try:
        x0 = _grow_and_push_heuristic(n, seed=seed).flatten()
        options = {'maxiter': 2500, 'ftol': 1e-10, 'disp': False, 'eps': 1e-10}
        res = minimize(fun=_objective, x0=x0, method='SLSQP', jac=_jac_objective,
                       bounds=bounds, constraints=constraints, options=options)
        current_circles = _final_cleanup_and_validate(res.x.reshape((n, 3)), n)
        current_sum_radii = np.sum(current_circles[:, 2])
        return current_circles, current_sum_radii
    except Exception:
        return np.zeros((n, 3)), -np.inf

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    This implementation uses a parallel multi-start hybrid approach:
    1. A 'grow-and-push' heuristic generates diverse initial configurations in parallel.
    2. Scipy's 'SLSQP' optimizer refines each configuration to a high-precision local optimum.
    3. The best result from all parallel runs is selected.
    This entire structure is adopted from the inspiration programs due to its superior performance.
    """
    n = 32
    N_PARALLEL_RUNS = 12
    seeds = [42 + i * 101 for i in range(N_PARALLEL_RUNS)]

    # Define constraints and bounds once using the efficient vectorized approach.
    constraints = [
        {'type': 'ineq', 'fun': _all_containment_constraints, 'jac': _jac_all_containment_constraints, 'args': (n,)},
        {'type': 'ineq', 'fun': _all_non_overlap_constraints, 'jac': _jac_all_non_overlap_constraints, 'args': (n,)}
    ]
    bounds = [(0, 1), (0, 1), (1e-9, 0.5)] * n

    # Execute optimization tasks in parallel using all available CPU cores.
    results = Parallel(n_jobs=-1)(
        delayed(_run_optimization_task)(seed, n, bounds, constraints) for seed in seeds
    )

    # Find and return the best result from all parallel runs.
    best_circles_overall = None
    max_sum_radii_overall = -np.inf
    
    for circles, sum_radii in results:
        if sum_radii > max_sum_radii_overall:
            max_sum_radii_overall = sum_radii
            best_circles_overall = circles.copy()

    return best_circles_overall if best_circles_overall is not None else np.zeros((n, 3))


# EVOLVE-BLOCK-END
