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

# --- Helper functions for dynamic refinement (adapted from Inspiration 2) ---
def _dynamic_refinement(circles: np.ndarray, n_iter: int = 300, jiggle_step: float = 4e-7, epsilon: float = 1e-10) -> np.ndarray:
    """
    Dynamically refines the solution by alternating between expanding radii and "jiggling" positions.
    This allows the packing to settle into a denser local optimum by simulating physical relaxation.
    """
    n = len(circles)
    if n == 0:
        return circles

    pos = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    
    perm = np.random.permutation(n) # Initial permutation for iterating over circles

    for iter_num in range(n_iter):
        # --- Step 1: Greedy Radius Expansion ---
        had_radius_improvement = False
        # This inner loop runs until no more radius expansion is possible or max iterations reached
        for _ in range(50): # Max 50 attempts to expand radii per iter_num
            num_changes = 0
            for i in perm: # Iterate in random order
                # Max radius limited by walls
                max_r_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                
                # Max radius limited by neighbors
                max_r_neighbor = float('inf')
                if n > 1:
                    other_indices = np.arange(n) != i
                    dists = np.linalg.norm(pos[other_indices] - pos[i], axis=1)
                    available_space = dists - radii[other_indices]
                    if available_space.size > 0:
                        max_r_neighbor = np.min(available_space)

                new_r = min(max_r_wall, max_r_neighbor)

                if new_r > radii[i] + epsilon:
                    radii[i] = new_r
                    had_radius_improvement = True
                    num_changes += 1
            if num_changes == 0: # If no radius improved, converge for this inner loop
                break

        # --- Step 2: Position Jiggling based on contact forces ---
        had_position_change = False
        new_pos = pos.copy() # Apply moves to a copy to avoid immediate interaction in current iteration
        for i in perm: # Iterate in random order
            force_vector = np.zeros(2)
            # Wall forces (repulsion if too close)
            # Using slightly larger epsilon for wall contact to encourage pushing off walls
            if pos[i, 0] - radii[i] < epsilon * 10: force_vector += np.array([1.0, 0.0])
            if 1.0 - pos[i, 0] - radii[i] < epsilon * 10: force_vector += np.array([-1.0, 0.0])
            if pos[i, 1] - radii[i] < epsilon * 10: force_vector += np.array([0.0, 1.0])
            if 1.0 - pos[i, 1] - radii[i] < epsilon * 10: force_vector += np.array([0.0, -1.0])

            # Neighbor forces (repulsion if in contact or overlapping)
            if n > 1:
                other_indices = np.arange(n) != i
                diffs = pos[i] - pos[other_indices]
                dists_sq = np.sum(diffs**2, axis=1)
                r_sum_sq_approx = (radii[i] + radii[other_indices])**2
                
                # Identify circles that are in contact or slightly overlapping
                contact_indices = np.where(dists_sq - r_sum_sq_approx < epsilon * radii[i])[0] # Reverted to original epsilon * radii[i]
                
                # To prevent division by zero in case of identical positions.
                # Also, only apply force if they are truly overlapping or very close.
                for j_idx in contact_indices:
                    dist_val = np.sqrt(dists_sq[j_idx])
                    target_dist = radii[i] + radii[other_indices][j_idx]
                    
                    if dist_val < target_dist: # Actual overlap
                        if dist_val > epsilon: # Avoid division by zero for very small dist_val
                            direction = diffs[j_idx] / dist_val # Unit vector pointing from j to i
                            # Force proportional to overlap depth. Higher force for deeper overlaps.
                            overlap_depth = target_dist - dist_val
                            force_vector += direction * (overlap_depth / target_dist) * 0.1 # Scaled force
                        else: # Centers are almost identical, apply strong random repulsive force
                            force_vector += np.random.uniform(-1, 1, 2) * 1e-3 # Small random push to separate
                    elif dist_val < target_dist + epsilon: # Very close to contact, apply a gentle push
                         if dist_val > epsilon:
                            direction = diffs[j_idx] / dist_val
                            force_vector += direction * 0.01 # Small push to prevent future overlap

            norm = np.linalg.norm(force_vector)
            if norm > epsilon:
                move = (force_vector / norm) * jiggle_step
                new_pos[i] += move
                had_position_change = True
        
        pos = new_pos # Update positions globally after all jiggles
        # Enforce containment after jiggling
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
            pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])

        # If no significant change in radii or positions, assume convergence
        if not had_radius_improvement and not had_position_change and iter_num > 10: # Allow some initial jiggling
            break

    final_circles = np.column_stack([pos, radii])
    # Final polish to ensure radii are maxed out after the last jiggle
    return _refine_radii_only(final_circles)

def _refine_radii_only(circles: np.ndarray, n_iter: int = 50, epsilon: float = 1e-12) -> np.ndarray:
    """
    A simplified, fast version of the refiner that only expands radii.
    Used for a final polish after dynamic refinement or as a standalone refinement.
    """
    n = len(circles)
    if n == 0: return circles
    pos = circles[:, :2] # Read-only, no copy needed
    radii = circles[:, 2].copy()

    for _ in range(n_iter):
        had_improvement = False
        for i in np.random.permutation(n): # Re-permute each iteration
            max_r_wall = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            max_r_neighbor = float('inf')
            if n > 1:
                other_indices = np.arange(n) != i
                dists = np.linalg.norm(pos[other_indices] - pos[i], axis=1)
                available_space = dists - radii[other_indices]
                if available_space.size > 0:
                     max_r_neighbor = np.min(available_space)
            
            new_r = min(max_r_wall, max_r_neighbor)
            if new_r > radii[i] + epsilon:
                radii[i] = new_r
                had_improvement = True
        if not had_improvement:
            break
    circles[:, 2] = radii
    return circles

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
    growth_epochs_per_restart = 60 # Increased for more refinement per restart
    initial_growth_delta = 0.001 # How much to attempt to grow radii each step
    growth_decay_factor = 0.99 # Slower decay for growth_delta to allow more aggressive growth for longer

    # Parameters for SLSQP
    slsqp_maxiter = 400 # Increased Max iterations for each SLSQP run for more refinement
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
        # Removed 'positive_radii_constraints' as it's redundant with bounds
    ]
    
    # --- 3. Bounds for variables ---
    lower_bounds = np.zeros(3 * n)
    upper_bounds = np.ones(3 * n)
    for i in range(n):
        lower_bounds[i*3+2] = 1e-7 # radius min, ensuring strictly positive
        upper_bounds[i*3+2] = square_size / 2.0 # radius max
    bounds = Bounds(lower_bounds, upper_bounds)

    # --- Global best tracking ---
    best_circles_overall = None
    best_sum_radii_overall = -np.inf

    # --- Multiple Restarts for Initial Placement ---
    # Adjusted parameters to fit dynamic refinement within time budget.
    num_initial_placements = 6 # Reduced from 8 to 6
    growth_epochs_per_restart = 50 # Reduced from 60 to 50
    slsqp_maxiter = 450 # Adjusted from 400 to 450

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
            # Ensure proposed radii are also not less than the minimum bound (1e-7)
            proposed_x[2::3] = np.maximum(proposed_x[2::3], 1e-7)


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

            if result.success or result.status == 9: # status == 9 for 'Iteration limit reached'
                optimized_vars = result.x
                circles_data_candidate = optimized_vars.reshape(n, 3)
                # Ensure radii are strictly positive after optimization
                circles_data_candidate[:, 2] = np.maximum(1e-7, circles_data_candidate[:, 2])

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

    # Apply dynamic refinement to the best solution found by SLSQP
    # This step leverages the "physics-informed optimization" aspect to further improve packing.
    # Using parameters from Inspiration 2 (_dynamic_refinement defaults are n_iter=300, jiggle_step=4e-7).
    final_circles = _dynamic_refinement(best_circles_overall)
    
    return final_circles


# EVOLVE-BLOCK-END
