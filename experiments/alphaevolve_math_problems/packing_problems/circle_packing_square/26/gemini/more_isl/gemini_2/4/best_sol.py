# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import random
import time
import pymunk

# Set a fixed random seed for reproducibility to ensure deterministic results
np.random.seed(42) # Changed to use only numpy's seed for consistency with current context
random.seed(42)

N_CIRCLES = 26 # Defined as a constant for clarity

def _objective_function(params: np.ndarray) -> float:
    """Objective: Maximize sum of radii -> Minimize -sum(radii)."""
    return -np.sum(params[2::3])

def _objective_jac(params: np.ndarray) -> np.ndarray:
    """Jacobian of the objective function (from Inspiration 1)."""
    grad = np.zeros_like(params)
    grad[2::3] = -1.0 # Derivative of -r_i with respect to r_i is -1
    return grad

def _constraints_func(params: np.ndarray, n_circles: int) -> np.ndarray:
    """
    Calculates values for inequality constraints. All must be >= 0 for a valid solution.
    This function is heavily vectorized for performance.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    containment = np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])
    if n_circles > 1:
        dx = x[:, None] - x
        dy = y[:, None] - y
        dist_sq = dx**2 + dy**2
        r_sum_sq = (r[:, None] + r)**2
        upper_tri_indices = np.triu_indices(n_circles, k=1)
        overlap = dist_sq[upper_tri_indices] - r_sum_sq[upper_tri_indices]
        return np.concatenate([containment, overlap])
    else:
        return containment

def _constraints_jac(params: np.ndarray, n_circles: int) -> np.ndarray:
    """Jacobian of the constraint function (from Inspiration 1)."""
    n = n_circles
    x, y, r = params[0::3], params[1::3], params[2::3]
    num_containment, num_overlap = 4 * n, n * (n - 1) // 2
    jac = np.zeros((num_containment + num_overlap, 3 * n))

    # Containment constraints derivatives
    idx = np.arange(n)
    # x_i - r_i >= 0  => d(x-r)/dx=1, d(x-r)/dr=-1
    jac[idx, 3 * idx] = 1
    jac[idx, 3 * idx + 2] = -1
    # 1 - x_i - r_i >= 0 => d(1-x-r)/dx=-1, d(1-x-r)/dr=-1
    jac[n + idx, 3 * idx] = -1
    jac[n + idx, 3 * idx + 2] = -1
    # y_i - r_i >= 0 => d(y-r)/dy=1, d(y-r)/dr=-1
    jac[2 * n + idx, 3 * idx + 1] = 1
    jac[2 * n + idx, 3 * idx + 2] = -1
    # 1 - y_i - r_i >= 0 => d(1-y-r)/dy=-1, d(1-y-r)/dr=-1
    jac[3 * n + idx, 3 * idx + 1] = -1
    jac[3 * n + idx, 3 * idx + 2] = -1

    # Non-overlap constraints derivatives ( (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0 )
    if n > 1:
        rows, cols = np.triu_indices(n, k=1)
        
        dx_pairs = (x[rows] - x[cols])
        dy_pairs = (y[rows] - y[cols])
        r_sum_pairs = (r[rows] + r[cols])
        
        jac_row_idx_start = num_containment
        
        # d/dx_i = 2(x_i - x_j)
        jac[jac_row_idx_start + np.arange(num_overlap), 3 * rows] = 2 * dx_pairs
        # d/dy_i = 2(y_i - y_j)
        jac[jac_row_idx_start + np.arange(num_overlap), 3 * rows + 1] = 2 * dy_pairs
        # d/dr_i = -2(r_i + r_j)
        jac[jac_row_idx_start + np.arange(num_overlap), 3 * rows + 2] = -2 * r_sum_pairs

        # d/dx_j = -2(x_i - x_j)
        jac[jac_row_idx_start + np.arange(num_overlap), 3 * cols] = -2 * dx_pairs
        # d/dy_j = -2(y_i - y_j)
        jac[jac_row_idx_start + np.arange(num_overlap), 3 * cols + 1] = -2 * dy_pairs
        # d/dr_j = -2(r_i + r_j)
        jac[jac_row_idx_start + np.arange(num_overlap), 3 * cols + 2] = -2 * r_sum_pairs
    
    return jac

# --- Physics-based Initial Guess Generation (adapted from Inspiration 1) ---
def _generate_initial_guess_physics(n_circles: int, unit_square_size: float = 1.0) -> np.ndarray:
    """Generates a high-quality initial configuration using a pymunk physics simulation."""
    # Local seeds for this function to ensure its internal randomness is consistent
    np.random.seed(42)
    random.seed(42)

    space = pymunk.Space(); space.damping = 0.95
    walls = [pymunk.Segment(space.static_body, (0, 0), (unit_square_size, 0), 0),
             pymunk.Segment(space.static_body, (0, unit_square_size), (unit_square_size, unit_square_size), 0),
             pymunk.Segment(space.static_body, (0, 0), (0, unit_square_size), 0),
             pymunk.Segment(space.static_body, (unit_square_size, 0), (unit_square_size, unit_square_size), 0)]
    space.add(*walls)
    
    bodies_shapes = []
    grid_dim = int(np.ceil(np.sqrt(n_circles)))
    r_init = unit_square_size / (2.1 * grid_dim) # Slightly larger initial radius
    
    for i in range(n_circles):
        row, col = i // grid_dim, i % grid_dim
        # Add jitter to grid placement
        x = (col + 0.5) * (unit_square_size / grid_dim) + np.random.uniform(-r_init*0.1, r_init*0.1)
        y = (row + 0.5) * (unit_square_size / grid_dim) + np.random.uniform(-r_init*0.1, r_init*0.1)
        body = pymunk.Body(1, 1); body.position = x, y
        shape = pymunk.Circle(body, r_init); shape.elasticity = 0.05; shape.friction = 0.9
        space.add(body, shape)
        bodies_shapes.append((body, shape))

    # Run simulation steps to settle the circles
    for _ in range(350): # Number of steps from Inspiration 1
        space.step(1.0 / 120.0)

    guess_circles = np.zeros((n_circles, 3))
    for i, (body, shape) in enumerate(bodies_shapes):
        x, y = body.position
        r = shape.radius
        # Ensure circles are within bounds and radii are valid
        guess_circles[i, 0] = np.clip(x, r, unit_square_size - r)
        guess_circles[i, 1] = np.clip(y, r, unit_square_size - r)
        guess_circles[i, 2] = np.clip(r, 1e-7, 0.5) # Ensure min radius and max possible radius
    return guess_circles

def _dynamic_refinement(circles: np.ndarray, time_budget: float, jiggle_step_start: float = 8e-7, jiggle_step_end: float = 1e-8, epsilon: float = 1e-10) -> np.ndarray:
    """
    Dynamically refines the solution with an adaptive jiggle step.
    The jiggle step decays over the time budget, allowing for larger moves initially
    and finer adjustments later. This simulates a cooling schedule.
    """
    n = len(circles)
    if n == 0: return circles
    
    start_time = time.time()
    pos = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    
    perm = np.random.permutation(n) 

    while time.time() - start_time < time_budget:
        # --- Adaptive Jiggle Step Calculation ---
        # Linearly decay the jiggle step from start to end over the allocated time budget.
        time_fraction = min(1.0, (time.time() - start_time) / max(1e-5, time_budget))
        current_jiggle_step = jiggle_step_start * (1.0 - time_fraction) + jiggle_step_end * time_fraction

        # --- Step 1: Greedy Radius Expansion ---
        had_radius_improvement = False
        for _ in range(20): 
            num_changes = 0
            for i in perm:
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
                    had_radius_improvement = True
                    num_changes += 1
            if num_changes == 0: break

        # --- Step 2: Position Jiggling based on contact forces ---
        had_position_change = False
        new_pos = pos.copy()
        for i in perm:
            force_vector = np.zeros(2)
            if pos[i, 0] - radii[i] < epsilon: force_vector += np.array([1.0, 0.0])
            if 1.0 - pos[i, 0] - radii[i] < epsilon: force_vector += np.array([-1.0, 0.0])
            if pos[i, 1] - radii[i] < epsilon: force_vector += np.array([0.0, 1.0])
            if 1.0 - pos[i, 1] - radii[i] < epsilon: force_vector += np.array([0.0, -1.0])
            if n > 1:
                other_indices = np.arange(n) != i
                diffs = pos[i] - pos[other_indices]
                dists_sq = np.sum(diffs**2, axis=1)
                gaps_sq_approx = (radii[i] + radii[other_indices])**2 
                contact_indices = np.where(dists_sq - gaps_sq_approx < epsilon * radii[i])[0] 
                for j_idx in contact_indices:
                    dist_val = np.sqrt(dists_sq[j_idx])
                    if dist_val > 1e-9:
                        direction = diffs[j_idx] / dist_val
                        force_vector += direction
            
            norm = np.linalg.norm(force_vector)
            if norm > epsilon:
                move = (force_vector / norm) * current_jiggle_step # Use adaptive step
                new_pos[i] += move
                had_position_change = True
        
        pos = new_pos
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], radii[i], 1.0 - radii[i])
            pos[i, 1] = np.clip(pos[i, 1], radii[i], 1.0 - radii[i])

        if not had_radius_improvement and not had_position_change: 
            break
        
        perm = np.random.permutation(n)

    return np.column_stack([pos, radii])


def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Synthesizes the best features of high-performing solutions:
    1. Multi-start SLSQP with analytical Jacobians for speed and precision.
    2. A high-quality physics-based initial guess using Pymunk.
    3. A hybrid random/grid/heterogeneous guess strategy for subsequent restarts.
    4. A final time-budgeted dynamic refinement phase to polish the best solution.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = N_CIRCLES
    start_time = time.time()
    TOTAL_TIME_BUDGET = 58.0 # Total time budget for the function
    SLSQP_TIME_BUDGET = 48.0 # Time allocated for the multi-start SLSQP phase

    lower_bounds = np.zeros(3 * n); upper_bounds = np.ones(3 * n)
    lower_bounds[2::3] = 1e-7; upper_bounds[2::3] = 0.5
    bounds = list(zip(lower_bounds, upper_bounds))

    num_total_constraints = 4 * n + n * (n - 1) // 2
    nlc = NonlinearConstraint(
        fun=lambda p: _constraints_func(p, n),
        lb=np.zeros(num_total_constraints),
        ub=np.full(num_total_constraints, np.inf),
        jac=lambda p: _constraints_jac(p, n) # Use analytical Jacobian for constraints
    )

    num_restarts = 150 # Increased restarts for more thorough search within time budget
    best_sum_radii = -np.inf 
    best_circles = np.zeros((n, 3))
    
    MAX_ITER_PER_ATTEMPT = 700 # Increased iterations per SLSQP attempt
    TOLERANCE = 1e-9 # Tighter tolerance for higher precision

    for i_restart in range(num_restarts):
        if time.time() - start_time > SLSQP_TIME_BUDGET:
            break # Exit loop if SLSQP time budget is exceeded

        if i_restart == 0:
            # Strategy 0: High-quality physics simulation guess (from Inspiration 1)
            initial_circles = _generate_initial_guess_physics(n_circles=n)
            initial_params = initial_circles.flatten()
        else:
            # Strategies 1-3: Diverse random starts to explore other basins
            strategy = (i_restart - 1) % 3
            if strategy == 0: # Standard Random
                initial_r = np.random.uniform(0.001, 0.04, n) # Adjusted range for initial radii
            elif strategy == 1: # Grid-based with jitter
                initial_r = np.full(n, 0.05)
            else: # Strategy 2: Heterogeneous Radii
                initial_r = np.random.uniform(0.01, 0.04, n)
                num_large = np.random.randint(3, 6) # 3 to 5 large circles
                large_indices = np.random.choice(n, num_large, replace=False)
                initial_r[large_indices] = np.random.uniform(0.09, 0.13, len(large_indices)) # Larger radii range
            
            if strategy == 1: # Grid-based specific position generation
                grid_size = 5; spacing = 1.0 / grid_size
                coords = np.linspace(spacing / 2, 1.0 - spacing / 2, grid_size)
                initial_x, initial_y = np.zeros(n), np.zeros(n)
                count = 0
                for i_grid in range(grid_size):
                    for j_grid in range(grid_size):
                        if count < n:
                            initial_x[count] = coords[i_grid] + np.random.uniform(-0.025, 0.025) # More jitter
                            initial_y[count] = coords[j_grid] + np.random.uniform(-0.025, 0.025)
                            count += 1
                while count < n: # Fill remaining spots randomly if n > grid_size^2
                     initial_x[count] = np.random.uniform(initial_r[count], 1 - initial_r[count])
                     initial_y[count] = np.random.uniform(initial_r[count], 1 - initial_r[count])
                     count += 1
            else: # Random position generation for other strategies
                initial_x = np.random.uniform(initial_r, 1 - initial_r)
                initial_y = np.random.uniform(initial_r, 1 - initial_r)
            
            initial_params = np.ravel(np.column_stack([initial_x, initial_y, initial_r]))

        try:
            res = minimize(
                fun=_objective_function,
                x0=initial_params,
                method="SLSQP",
                jac=_objective_jac, # Use analytical Jacobian for objective
                bounds=bounds,
                constraints=[nlc],
                options={"maxiter": MAX_ITER_PER_ATTEMPT, "ftol": TOLERANCE, "disp": False, "eps": TOLERANCE}
            )

            if res.success or res.status == 0: 
                current_sum_radii = -res.fun
                if current_sum_radii > best_sum_radii:
                    best_sum_radii = current_sum_radii
                    # Ensure radii are non-negative and clip to bounds
                    best_circles = res.x.reshape((n, 3))
                    best_circles[:, 2] = np.maximum(1e-7, best_circles[:, 2])
        except Exception:
            pass # Continue to the next attempt if an error occurs

    if best_sum_radii == -np.inf: # Fallback if no successful optimization occurred
        # If no solution found, generate a physics guess as a last resort
        return _generate_initial_guess_physics(n)

    # Apply time-budgeted dynamic refinement with adaptive jiggle step to the best solution.
    time_left = TOTAL_TIME_BUDGET - (time.time() - start_time)
    if time_left > 1.0: # Ensure at least 1 second for refinement
        # The refinement function uses an adaptive jiggle step (cooling) by default.
        final_circles = _dynamic_refinement(best_circles, time_budget=time_left)
    else:
        final_circles = best_circles # No time for refinement, return best SLSQP result
    
    return final_circles


# EVOLVE-BLOCK-END
