# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import warnings
import random
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric insights with advanced optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Enhanced initial placement using Voronoi-inspired approach (from INSPIRATION 2)
    def generate_better_initial_placement(num_circles):
        # Strategy: Create a more sophisticated grid pattern with hexagonal offsets
        positions = []
        
        # Use a grid with hexagonal offset for better packing
        rows = int(np.ceil(np.sqrt(num_circles)))
        cols = int(np.ceil(num_circles / rows))
        
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= num_circles:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Offset every other row for hexagonal packing effect
                if i % 2 == 1:
                    x += spacing_x * 0.5
                
                # Apply larger random perturbations to break symmetries (from INSPIRATION 2)
                x += np.random.normal(0, spacing_x * 0.15)
                y += np.random.normal(0, spacing_y * 0.15)
                
                # Keep within bounds with safety margin
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                positions.append([x, y])
        
        # If we don't have enough positions, add strategically placed ones
        while len(positions) < num_circles:
            # Add positions near corners and edges for better boundary utilization (from INSPIRATION 2)
            corner_positions = [
                [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
                [0.05, 0.5], [0.5, 0.05], [0.95, 0.5], [0.5, 0.95]
            ]
            if len(corner_positions) > 0:
                idx = len(positions) % len(corner_positions)
                x, y = corner_positions[idx]
                # Add larger randomization (from INSPIRATION 2)
                x += np.random.normal(0, 0.05)
                y += np.random.normal(0, 0.05)
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                positions.append([x, y])
            else:
                # Fallback to random placement
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                positions.append([x, y])
        
        return np.array(positions[:num_circles])
    
    # Create bounds for parameters - stricter bounds (from INSPIRATION 2)
    bounds = []
    # Position bounds [0,1] for both x and y coordinates
    for _ in range(2*n):
        bounds.extend([(0, 1)])
    # Radius bounds [0.001, 0.499] - stricter to avoid numerical issues (from INSPIRATION 2)
    for _ in range(n):
        bounds.extend([(0.001, 0.499)])
    
    # Constraint functions with improved numerical stability and robustness (from INSPIRATION 2)
    def containment_constraints(params):
        """Ensure all circles are contained within unit square"""
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        constraints = []
        
        # Each circle must be fully contained in unit square
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Add generous safety margins to prevent numerical issues (from INSPIRATION 2)
            constraints.extend([
                x - r - 1e-6,           # x >= r + safety
                y - r - 1e-6,           # y >= r + safety
                1 - x - r - 1e-6,       # 1 - x >= r + safety
                1 - y - r - 1e-6        # 1 - y >= r + safety
            ])
        return np.array(constraints)
    
    def non_overlap_constraints(params):
        """Ensure no overlapping circles using squared distances"""
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        constraints = []
        
        # Check all pairs of circles for overlap using squared distances
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance_squared = dx*dx + dy*dy
                min_distance_squared = (radii[i] + radii[j])**2
                # We want distance >= min_distance, so we enforce constraint: distance^2 - min_distance^2 >= 0
                # Add a small epsilon to handle numerical precision issues (from INSPIRATION 2)
                constraints.append(distance_squared - min_distance_squared - 1e-10)
        return np.array(constraints)
    
    # Enhanced penalty method for better constraint handling (from INSPIRATION 2)
    def enhanced_penalized_objective(params):
        """Enhanced objective with strong penalty functions"""
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Original objective: maximize sum of radii (minimize negative sum)
        obj_val = -np.sum(radii)
        
        # Strong constraint violation penalties (from INSPIRATION 2)
        penalty = 0
        
        # Containment penalties - extremely strong penalties
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Violations with very strong penalties (from INSPIRATION 2)
            violations = [
                max(0, r - x + 1e-8),      # Left boundary violation
                max(0, r - y + 1e-8),      # Bottom boundary violation
                max(0, r + x - 1 + 1e-8),  # Right boundary violation  
                max(0, r + y - 1 + 1e-8)   # Top boundary violation
            ]
            penalty += 1000000 * sum(violations)  # Much stronger penalty than before
        
        # Overlap penalties - even stronger penalties (from INSPIRATION 2)
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (radii[i] + radii[j])**2
                if dist_sq < min_dist_sq:
                    overlap_amount = min_dist_sq - dist_sq
                    penalty += 1000000 * overlap_amount  # Extremely strong penalty
        
        return obj_val + penalty
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        radii = params[2*n:]
        return -np.sum(radii)
    
    # Multi-strategy optimization approach with increased diversity (from INSPIRATION 2)
    best_result = None
    best_sum = -np.inf
    
    # Strategy 1: Multiple SLSQP runs with diverse starting points (INCREASED ATTEMPTS)
    for attempt in range(25):  # Significantly increased attempts (from INSPIRATION 2)
        np.random.seed(attempt * 1000 + 12345)
        
        # Generate initial placement
        initial_positions = generate_better_initial_placement(n)
        
        # Initialize radii with better estimates based on theoretical packing (from INSPIRATION 2)
        total_area = 1.0  # Unit square
        circles_area = total_area * 0.75  # Allow for some empty space (higher density)
        avg_circle_area = circles_area / n
        estimated_radius = np.sqrt(avg_circle_area / np.pi) * 1.1  # Slightly generous
        
        initial_radii = np.full(n, max(0.01, estimated_radius))
        
        # Add aggressive randomness to initial conditions (from INSPIRATION 2)
        for i in range(n):
            # Wider range of scaling factors (from INSPIRATION 2)
            scale_factor = 0.8 + np.random.random() * 0.5  # Range 0.8-1.3
            initial_radii[i] *= scale_factor
            
        # Keep within bounds
        initial_radii = np.clip(initial_radii, 0.001, 0.499)
        
        current_params = np.concatenate([initial_positions.flatten(), initial_radii])
        
        try:
            result = minimize(
                objective,
                current_params,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                    {'type': 'ineq', 'fun': lambda p: non_overlap_constraints(p)}
                ],
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12},  # Very tight tolerances (from INSPIRATION 2)
                tol=1e-12
            )
            
            if result.success:
                final_radii = result.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            continue
    
    # Strategy 2: L-BFGS-B with enhanced penalty function (INCREASED PENALTY STRENGTH)
    if best_result is None or best_sum < 2.6:
        try:
            # Try with the enhanced penalty method
            np.random.seed(99999)
            initial_positions = generate_better_initial_placement(n)
            initial_radii = np.full(n, 0.15)  # Start with larger radii (from INSPIRATION 2)
            initial_params = np.concatenate([initial_positions.flatten(), initial_radii])
            
            result_lbfgs = minimize(
                enhanced_penalized_objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12}  # Very tight tolerances (from INSPIRATION 2)
            )
            
            if result_lbfgs.success:
                final_radii = result_lbfgs.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_lbfgs
        except Exception as e:
            pass
    
    # Strategy 3: Trust-constr with even more diverse initial parameters
    if best_result is None or best_sum < 2.6:
        try:
            # Try trust-constr with multiple different initialization strategies
            for seed_val in [1111, 2222, 3333, 4444]:  # Multiple seeds for variety (from INSPIRATION 2)
                np.random.seed(seed_val)
                
                # Create positions with more strategic distribution (from INSPIRATION 2)
                positions = []
                # Use a combination of grid and random placement
                for i in range(n):
                    if i < n//2:  # First half: more structured
                        row = i // 4
                        col = i % 4
                        x = 0.2 + col * 0.2
                        y = 0.2 + row * 0.2
                    else:  # Second half: more random but bounded
                        x = np.random.uniform(0.1, 0.9)
                        y = np.random.uniform(0.1, 0.9)
                    positions.append([x, y])
                
                positions = np.array(positions)
                
                # Initialize radii with different strategy (from INSPIRATION 2)
                mean_radius = 0.13
                radii = np.random.uniform(mean_radius * 0.9, mean_radius * 1.2, n)
                radii = np.clip(radii, 0.001, 0.499)
                
                params = np.concatenate([positions.flatten(), radii])
                
                result_trust = minimize(
                    objective,
                    params,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=[
                        {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                        {'type': 'ineq', 'fun': lambda p: non_overlap_constraints(p)}
                    ],
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12}  # Very tight tolerances (from INSPIRATION 2)
                )
                
                if result_trust.success:
                    final_radii = result_trust.x[2*n:]
                    current_sum = np.sum(final_radii)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result_trust
        except Exception as e:
            pass
    
    # Strategy 4: Nelder-Mead with multiple restarts and better initialization (from INSPIRATION 2)
    if best_result is None or best_sum < 2.65:
        try:
            # Try Nelder-Mead with even more aggressive restarts
            for restart in range(15):  # More restarts (from INSPIRATION 2)
                np.random.seed(55555 + restart * 1000)
                
                # Different initialization approach - more varied (from INSPIRATION 2)
                positions = []
                # Start with a more structured grid pattern
                grid_size = int(np.ceil(np.sqrt(n)))
                spacing = 1.0 / (grid_size + 1)
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        if len(positions) >= n:
                            break
                        x = (j + 1) * spacing + np.random.normal(0, spacing * 0.15)
                        y = (i + 1) * spacing + np.random.normal(0, spacing * 0.15)
                        x = max(0.05, min(0.95, x))
                        y = max(0.05, min(0.95, y))
                        positions.append([x, y])
                
                # Fill any remaining positions randomly
                while len(positions) < n:
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.uniform(0.05, 0.95)
                    positions.append([x, y])
                
                positions = np.array(positions[:n])
                
                # Initialize radii differently - more aggressive variation (from INSPIRATION 2)
                radii = np.random.uniform(0.08, 0.22, n)
                radii = np.clip(radii, 0.001, 0.499)
                
                params = np.concatenate([positions.flatten(), radii])
                
                # Use Nelder-Mead with even more iterations (from INSPIRATION 2)
                result = minimize(
                    objective,
                    params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'adaptive': True, 'disp': False}
                )
                
                if result.success:
                    final_radii = result.x[2*n:]
                    current_sum = np.sum(final_radii)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
        except Exception as e:
            pass
    
    # Strategy 5: Final refinement with trust-constr on best solution so far (from INSPIRATION 2)
    if best_result is not None and best_sum < 2.65:
        try:
            # Refine the best solution found so far
            np.random.seed(77777)
            
            # Start from the best solution and do a final optimization
            current_params = best_result.x.copy()
            
            # Add slight random perturbations to improve chances of escape (from INSPIRATION 2)
            for i in range(2*n):
                if i % 2 == 0:  # x coordinates
                    current_params[i] += np.random.normal(0, 0.01)
                else:  # y coordinates
                    current_params[i] += np.random.normal(0, 0.01)
            
            # Clip to bounds
            for i in range(2*n):
                current_params[i] = np.clip(current_params[i], 0, 1)
            
            result_refine = minimize(
                objective,
                current_params,
                method='trust-constr',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                    {'type': 'ineq', 'fun': lambda p: non_overlap_constraints(p)}
                ],
                options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result_refine.success:
                final_radii = result_refine.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_refine
        except Exception as e:
            pass
    
    # Return best result or fallback to initial
    if best_result is not None and best_result.success:
        final_positions = best_result.x[:2*n].reshape(-1, 2)
        final_radii = best_result.x[2*n:]
        circles = np.column_stack([final_positions, final_radii])
        return circles
    
    # Fallback: return a good initial configuration
    initial_positions = generate_better_initial_placement(n)
    initial_radii = np.full(n, 0.08)  # Reasonable starting radius
    circles = np.column_stack([initial_positions, initial_radii])
    return circles


# EVOLVE-BLOCK-END
