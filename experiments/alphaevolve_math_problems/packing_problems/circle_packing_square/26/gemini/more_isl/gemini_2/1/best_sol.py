# EVOLVE-BLOCK-START
import numpy as np
import random # For initial random positions
from scipy.optimize import minimize, Bounds # Core for gradient-based optimization
import time # To track execution time for early exit

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Helper function to check for overlaps and containment
def is_valid_packing(circles_data, square_size=1.0, epsilon=1e-7):
    n = circles_data.shape[0]
    for i in range(n):
        xi, yi, ri = circles_data[i]
        # Check containment: ri <= xi <= 1-ri and ri <= yi <= 1-ri
        # Using epsilon to allow for minor floating point deviations from strict boundaries
        if not (ri - epsilon <= xi <= square_size - ri + epsilon and \
                ri - epsilon <= yi <= square_size - ri + epsilon):
            return False
        # Check non-overlap with other circles
        for j in range(i + 1, n):
            xj, yj, rj = circles_data[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            min_dist_sq = (ri + rj)**2
            # Check if distance squared is strictly less than (ri+rj)^2 - epsilon (i.e., overlap)
            if dist_sq < min_dist_sq - epsilon:
                return False
    return True

# Helper function to calculate sum of radii
def calculate_sum_radii(circles_data):
    # Ensure radii are non-negative before summing, as optimizer might return very small negative values
    return np.sum(np.maximum(0, circles_data[:, 2]))

# Main constructor function
def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This implementation uses an iterative radius growth strategy combined with scipy.optimize.minimize
    (SLSQP method) for gradient-based optimization of positions and radii in each growth step.
    Analytic gradients are provided for objective and constraints for improved performance and accuracy.
    Multiple initial guesses and growth epochs are used to find a better global optimum.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    square_size = 1.0
    
    # --- Optimization Parameters ---
    max_total_time = 55 # seconds, leave some buffer for final processing
    start_time = time.time()

    # Parameters for the iterative growth strategy
    min_initial_r_per_circle = 0.005 # Minimum initial radius for any circle during a restart
    max_initial_r_per_circle = 0.025 # Maximum initial radius for any circle during a restart
    growth_epochs_per_restart = 40 # Reduced to balance with more restarts and higher slsqp_maxiter
    initial_growth_delta = 0.001 # How much to attempt to grow radii each step
    growth_decay_factor = 0.99 # Slower decay for growth_delta to allow more aggressive growth for longer

    # Parameters for SLSQP
    slsqp_maxiter = 350 # Increased Max iterations for each SLSQP run for more refinement
    slsqp_ftol = 1e-9 # Tolerance for termination by objective function change
    slsqp_options = {'maxiter': slsqp_maxiter, 'ftol': slsqp_ftol, 'disp': False}

    # --- 1. Define Objective Function and its Jacobian ---
    def objective(x):
        radii = x[2::3]
        return -np.sum(np.maximum(0, radii))

    def jac_objective(x):
        grad = np.zeros_like(x)
        grad[2::3] = -1.0 # Gradient with respect to each radius is -1
        return grad

    # --- 2. Define Constraints and their Jacobians for SLSQP ---
    
    # 2.1. Containment Constraints (4 * n constraints)
    def containment_constraints(x):
        constraints = np.zeros(4 * n)
        for i in range(n):
            xi, yi, ri = x[i*3], x[i*3+1], x[i*3+2]
            constraints[i*4 + 0] = xi - ri
            constraints[i*4 + 1] = square_size - ri - xi
            constraints[i*4 + 2] = yi - ri
            constraints[i*4 + 3] = square_size - ri - yi
        return constraints

    def jac_containment_constraints(x):
        jac = np.zeros((4 * n, 3 * n))
        for i in range(n):
            # Constraint: xi - ri >= 0
            jac[i*4 + 0, i*3 + 0] = 1.0 # d(xi-ri)/dxi
            jac[i*4 + 0, i*3 + 2] = -1.0 # d(xi-ri)/dri
            
            # Constraint: square_size - ri - xi >= 0
            jac[i*4 + 1, i*3 + 0] = -1.0 # d(1-ri-xi)/dxi
            jac[i*4 + 1, i*3 + 2] = -1.0 # d(1-ri-xi)/dri

            # Constraint: yi - ri >= 0
            jac[i*4 + 2, i*3 + 1] = 1.0 # d(yi-ri)/dyi
            jac[i*4 + 2, i*3 + 2] = -1.0 # d(yi-ri)/dri

            # Constraint: square_size - ri - yi >= 0
            jac[i*4 + 3, i*3 + 1] = -1.0 # d(1-ri-yi)/dyi
            jac[i*4 + 3, i*3 + 2] = -1.0 # d(1-ri-yi)/dri
        return jac

    # 2.2. Non-overlap Constraints (n * (n-1) / 2 constraints)
    def non_overlap_constraints(x):
        num_overlap_constraints = n * (n - 1) // 2
        constraints = np.zeros(num_overlap_constraints)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = x[i*3], x[i*3+1], x[i*3+2]
                xj, yj, rj = x[j*3], x[j*3+1], x[j*3+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                min_dist_sq = (ri + rj)**2
                constraints[k] = dist_sq - min_dist_sq
                k += 1
        return constraints

    def jac_non_overlap_constraints(x):
        num_overlap_constraints = n * (n - 1) // 2
        jac = np.zeros((num_overlap_constraints, 3 * n))
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = x[i*3], x[i*3+1], x[i*3+2]
                xj, yj, rj = x[j*3], x[j*3+1], x[j*3+2]

                # d(c_ij)/dx_i
                jac[k, i*3 + 0] = 2 * (xi - xj)
                # d(c_ij)/dy_i
                jac[k, i*3 + 1] = 2 * (yi - yj)
                # d(c_ij)/dr_i
                jac[k, i*3 + 2] = -2 * (ri + rj)

                # d(c_ij)/dx_j
                jac[k, j*3 + 0] = -2 * (xi - xj)
                # d(c_ij)/dy_j
                jac[k, j*3 + 1] = -2 * (yi - yj)
                # d(c_ij)/dr_j
                jac[k, j*3 + 2] = -2 * (ri + rj)
                k += 1
        return jac
    
    # 2.3. Radii must be non-negative (n constraints)
    def positive_radii_constraints(x):
        return x[2::3]

    def jac_positive_radii_constraints(x):
        jac = np.zeros((n, 3 * n))
        for i in range(n):
            jac[i, i*3 + 2] = 1.0 # d(ri)/dri
        return jac

    all_constraints_slsqp = [
        {'type': 'ineq', 'fun': containment_constraints, 'jac': jac_containment_constraints},
        {'type': 'ineq', 'fun': non_overlap_constraints, 'jac': jac_non_overlap_constraints},
        {'type': 'ineq', 'fun': positive_radii_constraints, 'jac': jac_positive_radii_constraints}
    ]
    
    # --- 3. Bounds for variables ---
    lower_bounds = np.zeros(3 * n)
    upper_bounds = np.ones(3 * n)
    for i in range(n):
        lower_bounds[i*3+2] = 0.0 # radius min
        upper_bounds[i*3+2] = square_size / 2.0 # radius max
    bounds = Bounds(lower_bounds, upper_bounds)

    # --- Global best tracking ---
    best_circles_overall = None
    best_sum_radii_overall = -np.inf

    # --- Multiple Restarts for Initial Placement ---
    num_initial_placements = 5 # Increased number of distinct initial random arrangements to try

    for restart_idx in range(num_initial_placements):
        if time.time() - start_time > max_total_time:
            # print(f"Time limit reached. Exiting after {restart_idx} initial placements.")
            break

        # Generate initial guess for positions and small radii, with varied initial radii
        current_x = np.zeros(3 * n)
        for i in range(n):
            # Place circles randomly with varied initial radii within a small range
            r_i_init = random.uniform(min_initial_r_per_circle, max_initial_r_per_circle)
            current_x[i*3] = random.uniform(r_i_init, square_size - r_i_init) # x_i
            current_x[i*3+1] = random.uniform(r_i_init, square_size - r_i_init) # y_i
            current_x[i*3+2] = r_i_init # r_i

        current_growth_delta = initial_growth_delta
        current_best_in_restart = np.copy(current_x)
        current_best_sum_radii_in_restart = calculate_sum_radii(current_x.reshape(n,3))

        # --- Iterative Radius Growth Loop ---
        for epoch in range(growth_epochs_per_restart):
            if time.time() - start_time > max_total_time:
                # print(f"Time limit reached. Exiting growth epoch {epoch} of restart {restart_idx}.")
                break

            # Propose new radii: current radii + growth_delta
            proposed_x = np.copy(current_x)
            proposed_x[2::3] += current_growth_delta
            
            # Clamp proposed radii to upper bound
            proposed_x[2::3] = np.minimum(proposed_x[2::3], square_size / 2.0)

            try:
                # Optimize positions and radii using the proposed state as initial guess
                result = minimize(
                    objective,
                    proposed_x, # Use proposed_x as initial guess
                    method='SLSQP',
                    jac=jac_objective, # Provide analytical Jacobian for objective
                    bounds=bounds,
                    constraints=all_constraints_slsqp,
                    options=slsqp_options
                )
            except Exception as e:
                # print(f"SLSQP failed in epoch {epoch}, restart {restart_idx}: {e}")
                current_growth_delta *= growth_decay_factor
                current_x = np.copy(current_best_in_restart)
                continue

            if result.success or result.status == 9:
                optimized_vars = result.x
                circles_data_candidate = optimized_vars.reshape(n, 3)
                circles_data_candidate[:, 2] = np.maximum(0, circles_data_candidate[:, 2])

                current_sum_radii = calculate_sum_radii(circles_data_candidate)

                if is_valid_packing(circles_data_candidate, square_size, epsilon=1e-7):
                    if current_sum_radii > current_best_sum_radii_in_restart:
                        current_best_sum_radii_in_restart = current_sum_radii
                        current_best_in_restart = np.copy(optimized_vars)
                        current_x = np.copy(optimized_vars) # Accept this state for next epoch
                        # No decay on growth_delta if improvement found
                    else:
                        # Valid but no improvement; accept state but decay growth_delta
                        current_x = np.copy(optimized_vars)
                        current_growth_delta *= growth_decay_factor
                else:
                    # Optimized state is invalid; revert to best valid and decay growth_delta
                    current_growth_delta *= growth_decay_factor
                    current_x = np.copy(current_best_in_restart)
            else:
                # Optimization did not converge successfully; revert and decay growth_delta
                # print(f"SLSQP did not converge successfully in epoch {epoch}, restart {restart_idx}: {result.message}")
                current_growth_delta *= growth_decay_factor
                current_x = np.copy(current_best_in_restart)

            # Ensure growth delta doesn't become too small
            if current_growth_delta < 1e-8: # Lower threshold for delta
                current_growth_delta = 1e-8
            
            # Early exit for epoch if growth is negligible or maximum radii are reached
            # Check if all radii are very close to the upper bound (square_size / 2.0)
            if np.all(current_x[2::3] >= square_size / 2.0 - 1e-8) and current_growth_delta < 1e-6:
                 break

        # After all growth epochs for this restart, update overall best
        if current_best_sum_radii_in_restart > best_sum_radii_overall:
            best_sum_radii_overall = current_best_sum_radii_in_restart
            best_circles_overall = current_best_in_restart.reshape(n, 3)
            # print(f"New overall best sum_radii: {best_sum_radii_overall:.6f} from restart {restart_idx}")

    # If no valid solution was found across all restarts (highly unlikely with proper setup),
    # return a default initial configuration.
    if best_circles_overall is None:
        default_initial_radius = 0.01
        default_circles = np.zeros((n, 3))
        for i in range(n):
            default_circles[i, 0] = random.uniform(default_initial_radius, square_size - default_initial_radius)
            default_circles[i, 1] = random.uniform(default_initial_radius, square_size - default_initial_radius)
            default_circles[i, 2] = default_initial_radius
        return default_circles

    return best_circles_overall


# EVOLVE-BLOCK-END
