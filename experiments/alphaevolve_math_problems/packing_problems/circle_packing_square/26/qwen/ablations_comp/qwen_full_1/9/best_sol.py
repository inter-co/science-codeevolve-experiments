# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric insights with advanced optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Enhanced initial placement using hexagonal-like packing pattern
    def generate_better_initial_placement(num_circles):
        # Use a hexagonal-like packing approach for better initial distribution
        positions = []
        
        # Strategy: create a more sophisticated grid pattern with hexagonal offsets
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
                
                # Apply small random perturbations to break symmetries
                x += np.random.normal(0, spacing_x * 0.1)
                y += np.random.normal(0, spacing_y * 0.1)
                
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                positions.append([x, y])
        
        # If we don't have enough positions, add more strategically
        while len(positions) < num_circles:
            # Add positions near edges to explore boundary effects
            x = np.random.uniform(0.05, 0.95)
            y = np.random.choice([0.05, 0.95])
            positions.append([x, y])
        
        return np.array(positions[:num_circles])
    
    # Generate initial configuration
    initial_positions = generate_better_initial_placement(n)
    
    # Initialize radii with better estimates
    total_area = 1.0  # Unit square
    circles_area = total_area * 0.75  # Allow for some empty space (higher density)
    avg_circle_area = circles_area / n
    estimated_radius = np.sqrt(avg_circle_area / np.pi) * 1.1  # Slightly generous
    
    initial_radii = np.full(n, max(0.01, estimated_radius))
    
    # Combine positions and radii into a single parameter vector
    initial_params = np.concatenate([initial_positions.flatten(), initial_radii])
    
    # Define constraint functions with improved numerical stability and error handling
    def containment_constraints(params):
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        constraints = []
        
        # Each circle must be fully contained in unit square
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Add safety margins to prevent numerical issues
            constraints.extend([
                x - r - 1e-8,           # x >= r + safety
                y - r - 1e-8,           # y >= r + safety
                1 - x - r - 1e-8,       # 1 - x >= r + safety
                1 - y - r - 1e-8        # 1 - y >= r + safety
            ])
        return np.array(constraints)
    
    def non_overlap_constraints(params):
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
                # Add a small epsilon to handle numerical precision issues
                constraints.append(distance_squared - min_distance_squared - 1e-12)
        return np.array(constraints)
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        radii = params[2*n:]
        return -np.sum(radii)
    
    # Create bounds for parameters
    bounds = []
    # Position bounds [0,1] for both x and y coordinates
    for _ in range(2*n):
        bounds.extend([(0, 1)])
    # Radius bounds [0, 0.5] - reasonable upper bound
    for _ in range(n):
        bounds.extend([(0, 0.5)])
    
    # Multi-strategy optimization approach with more attempts
    best_result = None
    best_sum = -np.inf
    
    # Strategy 1: Multiple SLSQP runs with different starting points - enhanced version
    for attempt in range(15):  # Increase attempts for better exploration
        # Create different initial parameters for each attempt
        np.random.seed(attempt * 1000)
        
        # Perturb the initial solution a bit for diversity with more aggressive perturbations
        perturbed_positions = initial_positions.copy()
        perturbed_radii = initial_radii.copy()
        
        # Add more aggressive randomness to initial conditions
        for i in range(n):
            perturbed_positions[i, 0] += np.random.normal(0, 0.02)  # Larger perturbation
            perturbed_positions[i, 1] += np.random.normal(0, 0.02)
            # More varied radius scaling
            scale_factor = 0.7 + np.random.random() * 0.6  # Range 0.7-1.3
            perturbed_radii[i] *= scale_factor
            
        # Keep within bounds
        perturbed_positions[:, 0] = np.clip(perturbed_positions[:, 0], 0.05, 0.95)
        perturbed_positions[:, 1] = np.clip(perturbed_positions[:, 1], 0.05, 0.95)
        perturbed_radii = np.clip(perturbed_radii, 0.01, 0.5)
        
        current_params = np.concatenate([perturbed_positions.flatten(), perturbed_radii])
        
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
                options={'maxiter': 3500, 'ftol': 1e-10, 'gtol': 1e-10},  # Even tighter tolerances
                tol=1e-10
            )
            
            if result.success:
                final_radii = result.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            continue
    
    # Strategy 2: Try L-BFGS-B with better penalty function - enhanced version
    if best_result is None or best_sum < 2.6:  # Only if we haven't found a good solution yet
        try:
            def improved_penalized_objective(params):
                pos = params[:2*n].reshape(-1, 2)
                rad = params[2*n:]
                
                # Original objective
                obj_val = -np.sum(rad)
                
                # Stronger penalties for constraint violations with better scaling
                penalty = 0
                for i in range(n):
                    x, y = pos[i]
                    r = rad[i]
                    # Violations with more careful handling
                    violations = [
                        max(0, r - x + 1e-10),      # Left boundary
                        max(0, r - y + 1e-10),      # Bottom boundary
                        max(0, r + x - 1 + 1e-10),  # Right boundary  
                        max(0, r + y - 1 + 1e-10)   # Top boundary
                    ]
                    penalty += 50000 * sum(violations)  # Increased penalty strength
                
                # Overlap penalty with more robust calculation
                for i in range(n):
                    for j in range(i+1, n):
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        dist_sq = dx*dx + dy*dy
                        min_dist_sq = (rad[i] + rad[j])**2
                        if dist_sq < min_dist_sq:
                            # Use a quadratic penalty to encourage gradual improvements
                            violation = min_dist_sq - dist_sq
                            penalty += 50000 * violation  # Stronger penalty
                
                return obj_val + penalty
            
            # Try with better initial parameters - use the best known configuration so far
            if best_result is not None:
                # Start from the best result found so far
                start_params = best_result.x.copy()
            else:
                start_params = initial_params.copy()
                
            result_lbfgs = minimize(
                improved_penalized_objective,
                start_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-10}  # Even tighter tolerances
            )
            
            if result_lbfgs.success:
                final_radii = result_lbfgs.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_lbfgs
        except Exception as e:
            pass
    
    # Strategy 3: Trust-constr method with improved initial parameters
    if best_result is None or best_sum < 2.6:
        try:
            # Create positions that are more spread out initially
            np.random.seed(42)
            
            # Create a more evenly distributed set of initial positions
            positions = []
            # Use a more structured approach with some hexagonal-like properties
            for i in range(n):
                # Distribute circles more evenly, with some clustering near edges
                if i < n // 2:
                    # Center region
                    x = np.random.uniform(0.2, 0.8)
                    y = np.random.uniform(0.2, 0.8)
                else:
                    # Edge regions
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.choice([0.05, 0.95])
                positions.append([x, y])
            
            positions = np.array(positions)
            
            # Initialize radii with a reasonable distribution
            mean_radius = 0.12  # Reasonable for 26 circles in unit square
            radii = np.random.uniform(mean_radius * 0.8, mean_radius * 1.2, n)
            radii = np.clip(radii, 0.01, 0.45)  # Keep reasonable bounds
            
            params = np.concatenate([positions.flatten(), radii])
            
            # Use trust-constr method which often works better for constrained problems
            result_trust = minimize(
                objective,
                params,
                method='trust-constr',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                    {'type': 'ineq', 'fun': lambda p: non_overlap_constraints(p)}
                ],
                options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-10}  # Tighter tolerances
            )
            
            if result_trust.success:
                final_radii = result_trust.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_trust
        except Exception as e:
            pass
    
    # Strategy 4: Add Nelder-Mead optimization with better initialization and constraints
    if best_result is None or best_sum < 2.65:
        try:
            # Use a completely different initialization strategy for Nelder-Mead
            np.random.seed(9999)  # Fixed seed for consistency
            
            # Create a better distributed initial configuration using Voronoi-inspired approach
            # Generate points that are more evenly spaced
            positions = []
            # Use a systematic approach to get good initial distribution
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
            
            # Fill any remaining positions
            while len(positions) < n:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                positions.append([x, y])
            
            positions = np.array(positions[:n])
            
            # Initialize radii with a balanced approach
            radii = np.random.uniform(0.08, 0.18, n)
            radii = np.clip(radii, 0.01, 0.45)
            
            params = np.concatenate([positions.flatten(), radii])
            
            # Use Nelder-Mead with very tight tolerances for local refinement
            # But first, create a version that handles constraints properly
            def nm_objective_with_constraints(params):
                pos = params[:2*n].reshape(-1, 2)
                rad = params[2*n:]
                
                # Check if constraints are violated
                # This is a simplified check for constraint feasibility
                for i in range(n):
                    x, y = pos[i]
                    r = rad[i]
                    if (r > x or r > y or r > (1-x) or r > (1-y)):
                        # Large penalty for constraint violations
                        return 1e10 - np.sum(rad)  # Negative because we're minimizing
                
                # Check overlap constraints
                for i in range(n):
                    for j in range(i+1, n):
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        dist_sq = dx*dx + dy*dy
                        min_dist_sq = (rad[i] + rad[j])**2
                        if dist_sq < min_dist_sq:
                            # Large penalty for overlaps
                            return 1e10 - np.sum(rad)
                
                # If we get here, constraints are satisfied
                return -np.sum(rad)
            
            # Use Nelder-Mead for global exploration with constraint handling
            result_nm = minimize(
                nm_objective_with_constraints,
                params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'adaptive': True, 'disp': False, 'fatol': 1e-12, 'xatol': 1e-12}
            )
            
            if result_nm.success:
                final_radii = result_nm.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_nm
        except Exception as e:
            pass
    
    # Strategy 5: Physics-inspired refinement with constraint satisfaction checking
    if best_result is None or best_sum < 2.63:
        try:
            # Try a hybrid approach using both constraint-aware optimization and 
            # a physics-inspired iterative improvement
            
            # Start with the best configuration found so far
            if best_result is not None:
                current_params = best_result.x.copy()
                positions = current_params[:2*n].reshape(-1, 2)
                radii = current_params[2*n:]
            else:
                # Use initial configuration as starting point
                positions = initial_positions.copy()
                radii = initial_radii.copy()
                current_params = np.concatenate([positions.flatten(), radii])
            
            # Create a more robust penalty function that ensures feasibility
            def robust_penalized_objective(params):
                pos = params[:2*n].reshape(-1, 2)
                rad = params[2*n:]
                
                # Original objective
                obj_val = -np.sum(rad)
                
                # Penalties for constraint violations
                penalty = 0
                
                # Containment penalties with stronger enforcement
                for i in range(n):
                    x, y = pos[i]
                    r = rad[i]
                    # Ensure we don't violate boundaries too severely
                    violations = [
                        max(0, r - x + 1e-10),      # Left boundary
                        max(0, r - y + 1e-10),      # Bottom boundary
                        max(0, r + x - 1 + 1e-10),  # Right boundary  
                        max(0, r + y - 1 + 1e-10)   # Top boundary
                    ]
                    penalty += 100000 * sum(violations)
                
                # Overlap penalties with more robust calculation
                for i in range(n):
                    for j in range(i+1, n):
                        dx = pos[i][0] - pos[j][0]
                        dy = pos[i][1] - pos[j][1]
                        dist_sq = dx*dx + dy*dy
                        min_dist_sq = (rad[i] + rad[j])**2
                        if dist_sq < min_dist_sq:
                            violation = min_dist_sq - dist_sq
                            penalty += 100000 * violation
                
                return obj_val + penalty
            
            # Use L-BFGS-B with very tight tolerances but also try to ensure feasibility
            result_final = minimize(
                robust_penalized_objective,
                current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result_final.success:
                final_radii = result_final.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_final
        except Exception as e:
            pass
    
    # Return best result or fallback to initial
    if best_result is not None and best_result.success:
        final_positions = best_result.x[:2*n].reshape(-1, 2)
        final_radii = best_result.x[2*n:]
        circles = np.column_stack([final_positions, final_radii])
        return circles
    
    # Final fallback: return the initial configuration with some adjustments
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = initial_positions[i, 0]
        circles[i, 1] = initial_positions[i, 1]
        circles[i, 2] = initial_radii[i]
    return circles


# EVOLVE-BLOCK-END
