# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import pdist

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Implements an iterative radius expansion strategy combined with scipy.optimize.minimize (SLSQP).
    This approach aims to provide a distinct pathway by guiding the optimization process with a greedy growth heuristic,
    followed by multi-restart local refinement.
    """
    n = 26
    
    # --- Objective Function ---
    def objective(params):
        # params: flattened array [x1, y1, r1, x2, y2, r2, ...]
        circles = params.reshape(n, 3)
        radii = circles[:, 2]
        
        # Penalize negative radii severely. This acts as a soft lower bound,
        # helping the optimizer (especially SLSQP) to keep radii positive.
        penalty = np.sum(np.maximum(0, -radii)) * 1e8 
        
        # We want to maximize sum of radii, so we minimize the negative sum.
        return -np.sum(radii) + penalty

    # --- Constraints Function ---
    # This function returns a vector where each element corresponds to a constraint g_i(x) >= 0.
    # For SLSQP, 'ineq' type constraints interpret this as g_i(x) >= 0.
    def constraints_func(params):
        # params: flattened array [x1, y1, r1, x2, y2, r2, ...]
        circles_arr = params.reshape(n, 3)
        x, y, r = circles_arr[:, 0], circles_arr[:, 1], circles_arr[:, 2]

        # Boundary constraints: circles must be fully contained within the unit square.
        # r <= x <= 1-r   => x - r >= 0  AND  1 - r - x >= 0
        # r <= y <= 1-r   => y - r >= 0  AND  1 - r - y >= 0
        boundary_constraints = np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])

        # Non-overlap constraints: distance between centers must be at least sum of radii.
        # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        # Squaring both sides to avoid sqrt (and ensure differentiability where possible):
        # (xi-xj)^2 + (yi-yj)^2 >= (ri + rj)^2
        # => (xi-xj)^2 + (yi-yj)^2 - (ri + rj)^2 >= 0
        if n > 1:
            coords = np.column_stack((x, y))
            dist_sq = pdist(coords, metric='sqeuclidean') # Efficiently computes squared Euclidean distances for all pairs
            
            # Compute (ri + rj)^2 for all unique pairs of radii
            radii_sum_pairs = pdist(r.reshape(-1, 1), lambda u, v: u[0] + v[0])
            radii_sum_pairs_sq = radii_sum_pairs**2
            
            overlap_constraints = dist_sq - radii_sum_pairs_sq
        else: # Case for n=1, no overlap constraints
            overlap_constraints = np.array([])
            
        # Combine all constraints into a single array. All elements must be >= 0.
        all_constraints = np.concatenate([
            boundary_constraints,
            overlap_constraints
        ])
        
        return all_constraints

    # --- Feasibility Check ---
    # A stricter check to ensure the final result is valid within a given tolerance.
    def is_feasible(circles_arr, tolerance=1e-7):
        x_val, y_val, r_val = circles_arr[:,0], circles_arr[:,1], circles_arr[:,2]
        
        # Check r >= 0
        if np.any(r_val < -tolerance):
            return False
        
        # Check boundary constraints
        if np.any(x_val - r_val < -tolerance) or \
           np.any(1 - x_val - r_val < -tolerance) or \
           np.any(y_val - r_val < -tolerance) or \
           np.any(1 - y_val - r_val < -tolerance):
            return False
        
        # Check overlap constraints
        if n > 1:
            coords = np.column_stack((x_val, y_val))
            dist_sq = pdist(coords, metric='sqeuclidean')
            radii_sum_pairs = pdist(r_val.reshape(-1, 1), lambda u, v: u[0] + v[0])
            radii_sum_pairs_sq = radii_sum_pairs**2
            if np.any(dist_sq - radii_sum_pairs_sq < -tolerance):
                return False
        return True

    # --- Optimization Setup ---
    # Bounds for x, y, r for each circle. These are hard limits for the variables.
    # x, y are within [0, 1]. r is within [0, 0.5] (a single circle cannot exceed radius 0.5 in unit square).
    lb = np.array([0.0, 0.0, 0.0] * n) # Lower bounds for x, y, r
    ub = np.array([1.0, 1.0, 0.5] * n) # Upper bounds for x, y, r
    bounds_slsqp = list(zip(lb, ub)) # Format required by SLSQP

    # Constraints for scipy.optimize.minimize (SLSQP).
    # 'type': 'ineq' means that the function 'fun' must return a value >= 0.
    slsqp_constraints = [{'type': 'ineq', 'fun': constraints_func}]

    best_overall_sum_radii = -np.inf
    best_overall_circles = np.zeros((n, 3))

    start_global_time = time.time()

    # --- Strategy 1: Iterative Radius Expansion (Greedy Growth) ---
    # This phase aims to quickly find a good, dense initial configuration by iteratively
    # growing circle radii and resolving conflicts with local optimization.
    print("Starting Iterative Radius Expansion (Greedy Growth)...")
    # np.random.seed(42) # Seed is already set at the beginning of the function

    # --- Initialization: Quasi-hexagonal grid packing (adapted from inspiration) ---
    # This aims for a denser, more organized initial placement than a simple square grid,
    # which can significantly improve the starting point for optimization.
    initial_radius_for_placement = 0.05 # Use a reasonable radius to estimate spacing
    
    # Determine grid dimensions for initial placement.
    # The inspiration uses a heuristic for hexagonal packing based on N.
    num_target_rows = int(np.round(np.sqrt(n * np.sqrt(3)/2))) 
    num_target_cols = int(np.round(n / num_target_rows)) if num_target_rows > 0 else n
    
    # Ensure num_target_rows and num_target_cols are at least 1
    num_target_rows = max(1, num_target_rows)
    num_target_cols = max(1, num_target_cols)

    # Calculate step sizes based on the target grid dimensions to fill the square
    effective_width = 1.0 - 2 * initial_radius_for_placement
    effective_height = 1.0 - 2 * initial_radius_for_placement
    
    x_step_base = (effective_width / (num_target_cols - 1)) if num_target_cols > 1 else effective_width
    y_step_base = (effective_height / (num_target_rows - 1)) if num_target_rows > 1 else effective_height

    initial_positions = []
    current_circle_idx = 0
    
    for row in range(num_target_rows):
        # Staggering for a hexagonal-like appearance: offset every other row by half a step
        x_start = initial_radius_for_placement + (row % 2) * (x_step_base / 2.0)
        
        for col in range(num_target_cols):
            if current_circle_idx >= n:
                break
            
            x = x_start + col * x_step_base
            y = initial_radius_for_placement + row * y_step_base
            
            # Clip initial position to ensure it's within bounds based on initial_radius_for_placement.
            # This is a safer approach than breaking the loop, ensuring all circles get a starting point.
            x = np.clip(x, initial_radius_for_placement, 1 - initial_radius_for_placement)
            y = np.clip(y, initial_radius_for_placement, 1 - initial_radius_for_placement)
                
            initial_positions.append([x, y])
            current_circle_idx += 1
        
        if current_circle_idx >= n:
            break
            
    # Fallback: if not all circles were placed by the grid (e.g., due to strict bounding),
    # fill remaining with random positions, still within reasonable bounds.
    while current_circle_idx < n:
        initial_positions.append([
            np.random.uniform(0.1 + initial_radius_for_placement, 0.9 - initial_radius_for_placement),
            np.random.uniform(0.1 + initial_radius_for_placement, 0.9 - initial_radius_for_placement)
        ])
        current_circle_idx += 1
    
    initial_positions = np.array(initial_positions)

    # Perturb initial positions slightly to break perfect symmetry and help convergence.
    # A smaller perturbation is used here, as the hexagonal grid already provides a good spread.
    initial_positions += (np.random.rand(n, 2) - 0.5) * 0.005 

    current_circles = np.zeros((n, 3))
    current_circles[:, :2] = initial_positions
    current_circles[:, 2] = 1e-5 # Start with very tiny radii to ensure initial feasibility

    max_growth_iterations = 60 # Increased iterations for more gradual growth
    delta_r_initial = 0.003    # Slightly smaller initial step size for radius increment
    
    previous_sum_radii = 0.0
    
    for step in range(max_growth_iterations):
        # Propose increased radii for all circles.
        # Gradually decrease delta_r to allow for finer adjustments in later steps.
        # Using a higher exponent for decay provides a slightly faster initial drop, then slower tail.
        current_delta_r = delta_r_initial * (1.0 - step / max_growth_iterations)**0.75 
        proposed_radii = current_circles[:, 2] + current_delta_r
        
        # Clamp proposed radii to ensure they remain valid (positive and not exceeding 0.5).
        proposed_radii = np.maximum(1e-7, proposed_radii)
        proposed_radii = np.minimum(0.5, proposed_radii)

        # Create the initial guess for this local optimization step, incorporating proposed radii.
        x0_growth = np.copy(current_circles)
        x0_growth[:, 2] = proposed_radii
        x0_growth_flat = x0_growth.flatten()

        # Perform local optimization using SLSQP to resolve any overlaps or boundary violations
        # caused by the radius increase. This adjusts positions and radii to maximize sum_radii.
        res_growth = minimize(
            objective,
            x0_growth_flat,
            method='SLSQP', # Using SLSQP for its speed and effectiveness with inequality constraints
            bounds=bounds_slsqp,
            constraints=slsqp_constraints,
            options={'ftol': 1e-6, 'maxiter': 120, 'disp': False} # Increased iterations per growth step for better local resolution
        )
        
        if res_growth.success:
            grown_circles = res_growth.x.reshape(n, 3)
            current_sum_radii = np.sum(grown_circles[:, 2])

            # Only accept the new configuration if it's feasible and provides a significant improvement.
            if is_feasible(grown_circles) and current_sum_radii > previous_sum_radii + 1e-5:
                current_circles = grown_circles
                previous_sum_radii = current_sum_radii
                # print(f"  Growth step {step+1}: Sum Radii = {current_sum_radii:.6f}") # Verbose output for debugging
                if current_sum_radii > best_overall_sum_radii:
                    best_overall_sum_radii = current_sum_radii
                    best_overall_circles = current_circles
            else:
                # If growth yields no significant improvement or leads to an infeasible state,
                # it suggests we've reached a local optimum for the growth phase. Stop growing.
                # print(f"  Growth step {step+1}: No significant improvement or infeasible. Stopping growth. Sum Radii = {current_sum_radii:.6f}")
                break
        else:
            # If the local optimizer fails, stop the growth phase.
            # print(f"  Growth step {step+1} failed: {res_growth.message}. Stopping growth.")
            break
            
    print(f"Iterative Radius Expansion completed. Best Sum Radii from growth: {best_overall_sum_radii:.6f}")

    # --- Strategy 2: Multi-restart Refinement from best growth point ---
    # This phase takes the best solution from the growth strategy, perturbs it slightly,
    # and runs more intensive local optimizations to escape shallow local minima and find a better solution.
    print("Starting Multi-restart Refinement...")
    num_refine_restarts = 15 # Increased number of perturbed restarts for deeper optimization
    
    # Use the best configuration found during the growth phase as the base for perturbations.
    base_circles_for_refinement = np.copy(best_overall_circles)

    for i in range(num_refine_restarts):
        # Create a new starting point by adding small random noise to positions and radii.
        x0_refine = np.copy(base_circles_for_refinement)
        
        # Slightly increased perturbation amounts to explore a wider neighborhood.
        x0_refine[:, :2] += (np.random.rand(n, 2) - 0.5) * 0.015 # Position noise (e.g., +/- 0.0075)
        x0_refine[:, 2] += (np.random.rand(n) - 0.5) * 0.003    # Radius noise (e.g., +/- 0.0015)
        
        # Ensure radii remain positive and within bounds after perturbation.
        x0_refine[:, 2] = np.maximum(1e-7, x0_refine[:, 2])
        x0_refine[:, 2] = np.minimum(0.5, x0_refine[:, 2])
        
        x0_refine_flat = x0_refine.flatten()

        # print(f"  Refinement Restart {i+1}/{num_refine_restarts}...") # Verbose output for debugging
        res_refine = minimize(
            objective,
            x0_refine_flat,
            method='SLSQP', # Continue using SLSQP for consistency and performance
            bounds=bounds_slsqp,
            constraints=slsqp_constraints,
            options={'ftol': 1e-7, 'maxiter': 700, 'disp': False} # Higher maxiter for more thorough optimization per restart
        )

        if res_refine.success:
            refined_circles = res_refine.x.reshape(n, 3)
            current_sum_radii = np.sum(refined_circles[:, 2])
            
            # Update best_overall_circles if a better, feasible solution is found.
            if is_feasible(refined_circles):
                if current_sum_radii > best_overall_sum_radii:
                    best_overall_sum_radii = current_sum_radii
                    best_overall_circles = refined_circles
                    # print(f"    New best sum_radii found: {best_overall_sum_radii:.6f}") # Verbose output
            # else:
                # print(f"    Refinement Restart {i+1} result is not feasible, sum_radii: {current_sum_radii:.6f}")
        # else:
            # print(f"    Refinement Restart {i+1} failed: {res_refine.message}")
            
    end_global_time = time.time()
    print(f"Total optimization finished in {end_global_time - start_global_time:.2f} seconds. Final best sum_radii: {best_overall_sum_radii:.6f}")

    # Final check and cleanup to ensure the returned solution is valid.
    # Explicitly ensure radii are non-negative, as numerical issues can sometimes lead to tiny negative values.
    best_overall_circles[:, 2] = np.maximum(0, best_overall_circles[:, 2])
        
    # If, despite best efforts, the final best solution is not feasible, log a warning.
    # In practice, SLSQP with `ftol` and good initial points usually converges to feasible solutions.
    if not is_feasible(best_overall_circles):
        print("Warning: The final best solution is not strictly feasible by post-check. This might indicate numerical instability or convergence issues.")
        
    return best_overall_circles


# EVOLVE-BLOCK-END
