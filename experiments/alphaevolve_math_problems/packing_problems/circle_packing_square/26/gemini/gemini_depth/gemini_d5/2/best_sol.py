# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# Helper to generate a sparse gradient for a specific variable (defined globally for clarity and potential parallelization)
def _make_sparse_gradient(num_vars, indices_values):
    grad = np.zeros(num_vars)
    for idx, val in indices_values:
        grad[idx] = val
    return grad

def _optimize_slsqp_single_run(n, initial_params, objective, jac_objective, constraints, bounds, epsilon):
    """Helper function to run a single SLSQP optimization."""
    result = minimize(
        objective,
        initial_params,
        method='SLSQP',
        jac=jac_objective,
        bounds=bounds,
        constraints=constraints,
        options={'disp': False, 'maxiter': 2000, 'ftol': 1e-8}
    )

    if result.success:
        optimized_params = result.x
        x_coords = optimized_params[0:n]
        y_coords = optimized_params[n:2*n]
        radii = optimized_params[2*n:3*n]
        
        circles = np.column_stack((x_coords, y_coords, radii))
        circles[:, 2] = np.maximum(circles[:, 2], epsilon) # Ensure radii are positive
        return circles, -result.fun # -result.fun is the sum of radii
    else:
        # If optimization fails, return a very low sum_radii to indicate a poor result
        # and the initial guess as circles for consistency
        # print(f"Warning: Single optimization run failed: {result.message}")
        x_coords = initial_params[0:n]
        y_coords = initial_params[n:2*n]
        radii = initial_params[2*n:3*n]
        circles = np.column_stack((x_coords, y_coords, radii))
        circles[:, 2] = np.maximum(circles[:, 2], epsilon)
        return circles, np.sum(radii) # Return sum of initial radii

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This implementation uses a Gradient-Based Continuous Optimization approach (SLSQP)
    with analytical Jacobians for efficiency.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # --- Optimization Setup ---
    # Decision variables: [x1, ..., xN, y1, ..., yN, r1, ..., rN]
    # Total variables = 3 * N
    num_vars = 3 * n

    # Objective function: Minimize negative sum of radii
    def objective(params):
        rs = params[2*n : 3*n]
        return -np.sum(rs)

    # Jacobian of the objective function
    def jac_objective(params):
        grad = np.zeros(num_vars)
        grad[2*n : 3*n] = -1.0 # Gradient with respect to radii is -1
        return grad

    # Constraints for scipy.optimize.minimize (SLSQP can handle non-linear inequalities)
    constraints = []
    epsilon = 1e-6 # Minimum radius to prevent numerical issues and ensure positivity

    # 1. Radii bounds: r_i >= epsilon and r_i <= 0.5
    for i in range(n):
        # r_i >= epsilon
        def r_lower_bound_fun(params, i_idx=i):
            return params[2*n + i_idx] - epsilon # r_i - epsilon >= 0
        
        def r_lower_bound_jac(params, i_idx=i):
            return _make_sparse_gradient(num_vars, [(2*n + i_idx, 1.0)])
        
        constraints.append({'type': 'ineq', 'fun': r_lower_bound_fun, 'jac': r_lower_bound_jac})

        # r_i <= 0.5
        def r_upper_bound_fun(params, i_idx=i):
            return 0.5 - params[2*n + i_idx] # 0.5 - r_i >= 0
        
        def r_upper_bound_jac(params, i_idx=i):
            return _make_sparse_gradient(num_vars, [(2*n + i_idx, -1.0)])
        
        constraints.append({'type': 'ineq', 'fun': r_upper_bound_fun, 'jac': r_upper_bound_jac})

    # 2. Circle containment within [0,1] x [0,1]:
    # x_i - r_i >= 0, 1 - x_i - r_i >= 0
    # y_i - r_i >= 0, 1 - y_i - r_i >= 0
    for i in range(n):
        # x_i - r_i >= 0
        def x_lower_bound_fun(params, i_idx=i):
            return params[i_idx] - params[2*n + i_idx] # x_i - r_i >= 0
        
        def x_lower_bound_jac(params, i_idx=i):
            return _make_sparse_gradient(num_vars, [(i_idx, 1.0), (2*n + i_idx, -1.0)])
        
        constraints.append({'type': 'ineq', 'fun': x_lower_bound_fun, 'jac': x_lower_bound_jac})

        # 1 - x_i - r_i >= 0
        def x_upper_bound_fun(params, i_idx=i):
            return 1 - params[i_idx] - params[2*n + i_idx] # 1 - x_i - r_i >= 0
        
        def x_upper_bound_jac(params, i_idx=i):
            return _make_sparse_gradient(num_vars, [(i_idx, -1.0), (2*n + i_idx, -1.0)])
        
        constraints.append({'type': 'ineq', 'fun': x_upper_bound_fun, 'jac': x_upper_bound_jac})

        # y_i - r_i >= 0
        def y_lower_bound_fun(params, i_idx=i):
            return params[n + i_idx] - params[2*n + i_idx] # y_i - r_i >= 0
        
        def y_lower_bound_jac(params, i_idx=i):
            return _make_sparse_gradient(num_vars, [(n + i_idx, 1.0), (2*n + i_idx, -1.0)])
        
        constraints.append({'type': 'ineq', 'fun': y_lower_bound_fun, 'jac': y_lower_bound_jac})

        # 1 - y_i - r_i >= 0
        def y_upper_bound_fun(params, i_idx=i):
            return 1 - params[n + i_idx] - params[2*n + i_idx] # 1 - y_i - r_i >= 0
        
        def y_upper_bound_jac(params, i_idx=i):
            return _make_sparse_gradient(num_vars, [(n + i_idx, -1.0), (2*n + i_idx, -1.0)])
        
        constraints.append({'type': 'ineq', 'fun': y_upper_bound_fun, 'jac': y_upper_bound_jac})

    # 3. Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n): # Only need i < j pairs
            def non_overlap_fun(params, i_idx=i, j_idx=j):
                xi, yi, ri = params[i_idx], params[n + i_idx], params[2*n + i_idx]
                xj, yj, rj = params[j_idx], params[n + j_idx], params[2*n + j_idx]
                return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            
            def non_overlap_jac(params, i_idx=i, j_idx=j):
                grad = np.zeros(num_vars)
                xi, yi, ri = params[i_idx], params[n + i_idx], params[2*n + i_idx]
                xj, yj, rj = params[j_idx], params[n + j_idx], params[2*n + j_idx]

                # Gradients with respect to x_i, x_j, y_i, y_j, r_i, r_j
                grad[i_idx] = 2 * (xi - xj)                     # d/dx_i
                grad[n + i_idx] = 2 * (yi - yj)                 # d/dy_i
                grad[2*n + i_idx] = -2 * (ri + rj)              # d/dr_i

                grad[j_idx] = -2 * (xi - xj)                    # d/dx_j
                grad[n + j_idx] = -2 * (yi - yj)                # d/dy_j
                grad[2*n + j_idx] = -2 * (ri + rj)              # d/dr_j
                return grad

            constraints.append({'type': 'ineq', 'fun': non_overlap_fun, 'jac': non_overlap_jac})

    # General bounds for the variables (refined by non-linear constraints)
    bounds = ([(0, 1)] * n +       # x_i bounds
              [(0, 1)] * n +       # y_i bounds
              [(epsilon, 0.5)] * n) # r_i bounds

    # --- Multi-start Optimization ---
    num_starts = 200 # Number of random initial guesses
    best_sum_radii = -np.inf
    best_circles = None

    # Set a single seed for overall reproducibility of the multi-start process
    np.random.seed(42) 

    # Add the original structured initial guess as one of the starts
    # Generate initial positions based on a grid with random jitter
    grid_dim = int(np.ceil(np.sqrt(n)))
    initial_r_base_structured = 0.5 / (grid_dim + 1)
    if initial_r_base_structured < epsilon: initial_r_base_structured = epsilon

    initial_x_structured = np.zeros(n)
    initial_y_structured = np.zeros(n)
    initial_r_structured = np.full(n, initial_r_base_structured)

    for i in range(n):
        row = i // grid_dim
        col = i % grid_dim
        x_base = (col + 0.5) / grid_dim
        y_base = (row + 0.5) / grid_dim
        jitter_scale = initial_r_base_structured * 0.5
        initial_x_structured[i] = np.clip(x_base + (np.random.rand() - 0.5) * jitter_scale, 
                                initial_r_base_structured, 1 - initial_r_base_structured)
        initial_y_structured[i] = np.clip(y_base + (np.random.rand() - 0.5) * jitter_scale, 
                                initial_r_base_structured, 1 - initial_r_base_structured)
    
    initial_params_structured = np.concatenate((initial_x_structured, initial_y_structured, initial_r_structured))
    initial_params_structured[2*n:3*n] = np.clip(initial_params_structured[2*n:3*n], epsilon, 0.5)
    for i in range(n):
        ri = initial_params_structured[2*n + i]
        initial_params_structured[i] = np.clip(initial_params_structured[i], ri, 1-ri)
        initial_params_structured[n+i] = np.clip(initial_params_structured[n+i], ri, 1-ri)

    # Run the structured initial guess first
    circles, current_sum_radii = _optimize_slsqp_single_run(n, initial_params_structured, objective, jac_objective, constraints, bounds, epsilon)
    if current_sum_radii > best_sum_radii:
        best_sum_radii = current_sum_radii
        best_circles = circles
        # print(f"Structured start, sum_radii: {best_sum_radii}")


    # Now run random starts
    max_initial_r_random = 0.08 # A slightly smaller upper bound for random radii to prevent too much initial overlap

    for k in range(num_starts):
        # Generate new random initial guess
        initial_x_random = np.random.uniform(0, 1, n)
        initial_y_random = np.random.uniform(0, 1, n)
        initial_r_random = np.random.uniform(epsilon, max_initial_r_random, n)

        # Clamp initial radii and coordinates to ensure they are valid (x_i >= r_i, 1-x_i >= r_i, etc.)
        initial_r_random = np.clip(initial_r_random, epsilon, 0.5)
        for i in range(n):
            ri = initial_r_random[i]
            initial_x_random[i] = np.clip(initial_x_random[i], ri, 1-ri)
            initial_y_random[i] = np.clip(initial_y_random[i], ri, 1-ri)

        initial_params_random = np.concatenate((initial_x_random, initial_y_random, initial_r_random))

        circles, current_sum_radii = _optimize_slsqp_single_run(n, initial_params_random, objective, jac_objective, constraints, bounds, epsilon)
        
        if current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_circles = circles
            # print(f"New best found at iteration {k+1}: {best_sum_radii}")
    
    if best_circles is None:
        # Fallback if no successful optimization occurred across all runs
        print("Warning: No successful optimization run, returning initial structured guess.")
        x_coords = initial_params_structured[0:n]
        y_coords = initial_params_structured[n:2*n]
        radii = initial_params_structured[2*n:3*n]
        best_circles = np.column_stack((x_coords, y_coords, radii))
        best_circles[:, 2] = np.maximum(best_circles[:, 2], epsilon)

    return best_circles


# EVOLVE-BLOCK-END
