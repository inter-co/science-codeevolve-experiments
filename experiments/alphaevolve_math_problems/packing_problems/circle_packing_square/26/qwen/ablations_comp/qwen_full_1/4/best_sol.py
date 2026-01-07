# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining systematic initial placement with robust optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Better initial placement using a systematic approach
    def generate_better_initial_placement(num_circles):
        # Create a grid pattern with some randomness to avoid symmetry issues
        rows = int(np.ceil(np.sqrt(num_circles)))
        cols = int(np.ceil(num_circles / rows))
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= num_circles:
                    break
                # Create regular grid positions
                x = (j + 0.5) / cols
                y = (i + 0.5) / rows
                # Add small random perturbation to avoid perfect grid patterns
                x += np.random.uniform(-0.02, 0.02)
                y += np.random.uniform(-0.02, 0.02)
                # Keep within bounds with safety margin
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                positions.append([x, y])
        return np.array(positions[:num_circles])
    
    # Generate initial configuration
    initial_positions = generate_better_initial_placement(n)
    
    # Initialize radii with better estimates
    # Estimate based on density of circles in square
    estimated_density = n / 1.0  # 26 circles in unit square
    avg_area_per_circle = 1.0 / estimated_density
    estimated_radius = np.sqrt(avg_area_per_circle / np.pi) * 0.8  # Slightly conservative
    
    initial_radii = np.full(n, max(0.01, estimated_radius))
    
    # Combine positions and radii into a single parameter vector
    initial_params = np.concatenate([initial_radii, initial_positions.flatten()])
    
    # Define constraint functions with improved numerical stability
    def containment_constraints(params):
        radii = params[:n]
        positions = params[n:].reshape(-1, 2)
        constraints = []
        
        # Each circle must be fully contained in unit square
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Add small safety margin to prevent numerical issues
            constraints.extend([
                x - r - 1e-8,           # x >= r + safety
                y - r - 1e-8,           # y >= r + safety
                1 - x - r - 1e-8,       # 1 - x >= r + safety
                1 - y - r - 1e-8        # 1 - y >= r + safety
            ])
        return np.array(constraints)
    
    def non_overlap_constraints(params):
        radii = params[:n]
        positions = params[n:].reshape(-1, 2)
        constraints = []
        
        # Check all pairs of circles for overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance_squared = dx*dx + dy*dy
                # Use squared distance to avoid sqrt computation
                min_distance_squared = (radii[i] + radii[j])**2
                # We want distance >= min_distance, so we enforce constraint: distance^2 - min_distance^2 >= 0
                constraints.append(distance_squared - min_distance_squared)
        return np.array(constraints)
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        radii = params[:n]
        return -np.sum(radii)
    
    # Create bounds for parameters
    bounds = []
    # Radius bounds [0, 0.5] 
    for _ in range(n):
        bounds.extend([(0, 0.5)])
    # Position bounds [0,1] for both x and y coordinates
    for _ in range(2*n):
        bounds.extend([(0, 1)])
    
    # Try multiple optimization strategies with better restarts
    best_result = None
    best_sum = -np.inf
    
    # Strategy 1: SLSQP with constraints (more robust than original)
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                {'type': 'ineq', 'fun': lambda p: non_overlap_constraints(p)}
            ],
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )
        
        if result.success:
            final_radii = result.x[:n]
            current_sum = np.sum(final_radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = result
    except Exception as e:
        pass
    
    # Strategy 2: Multiple restarts with different initializations
    # This helps escape local optima and improves results significantly
    for restart in range(5):
        try:
            # Perturb initial positions slightly with better distribution
            perturbed_positions = initial_positions.copy()
            for i in range(n):
                # Add more substantial perturbations for better exploration
                perturbed_positions[i] += np.random.normal(0, 0.03, 2)
                # Keep within bounds
                perturbed_positions[i][0] = np.clip(perturbed_positions[i][0], 0.01, 0.99)
                perturbed_positions[i][1] = np.clip(perturbed_positions[i][1], 0.01, 0.99)
            
            # Recompute radii for perturbed positions with improved logic
            perturbed_radii = []
            for i in range(n):
                x, y = perturbed_positions[i]
                # Max possible radius considering boundaries and neighbors
                r = min(x, 1-x, y, 1-y)
                # Also consider nearby circles for better initial estimates
                neighbor_distances = []
                for j in range(n):
                    if i != j:
                        dx = x - perturbed_positions[j][0]
                        dy = y - perturbed_positions[j][1]
                        d = np.sqrt(dx*dx + dy*dy)
                        neighbor_distances.append(d)
                
                if neighbor_distances:
                    # Reduce radius based on proximity to neighbors
                    min_neighbor_dist = min(neighbor_distances)
                    r = min(r, min_neighbor_dist/2.5)
                
                r = max(0.005, min(0.4, r * 0.9))  # Slightly smaller to ensure feasibility
                perturbed_radii.append(r)
            
            perturbed_params = np.concatenate([np.array(perturbed_radii), perturbed_positions.flatten()])
            
            result_restart = minimize(
                objective,
                perturbed_params,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                    {'type': 'ineq', 'fun': lambda p: non_overlap_constraints(p)}
                ],
                options={'maxiter': 800, 'ftol': 1e-6, 'gtol': 1e-6},
                tol=1e-6
            )
            
            if result_restart.success:
                final_radii = result_restart.x[:n]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_restart
                    
        except Exception as e:
            continue
    
    # Strategy 3: Alternative optimization method with penalty approach for backup
    if best_result is None or best_sum < 2.0:  # If result is weak, try alternative approach
        try:
            def penalized_objective(params):
                # Add penalty for constraint violations
                radii = params[:n]
                positions = params[n:].reshape(-1, 2)
                
                # Original objective
                obj_val = -np.sum(radii)
                
                # Penalty for containment violations
                penalty = 0
                for i in range(n):
                    x, y = positions[i]
                    r = radii[i]
                    # Violations
                    violations = [
                        max(0, r - x),
                        max(0, r - y),
                        max(0, r + x - 1),
                        max(0, r + y - 1)
                    ]
                    penalty += 1000 * sum(violations)
                
                # Penalty for overlap violations
                for i in range(n):
                    for j in range(i+1, n):
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        dist_sq = dx*dx + dy*dy
                        min_dist_sq = (radii[i] + radii[j])**2
                        if dist_sq < min_dist_sq:
                            penalty += 1000 * (min_dist_sq - dist_sq)
                
                return obj_val + penalty
            
            # Try L-BFGS-B with penalty method
            result_penalty = minimize(
                penalized_objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result_penalty.success:
                final_radii = result_penalty.x[:n]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_penalty
        except Exception as e:
            pass
    
    # Return best result or fallback to initial
    if best_result is not None and best_result.success:
        final_radii = best_result.x[:n]
        final_positions = best_result.x[n:].reshape(-1, 2)
        circles = np.column_stack([final_positions, final_radii])
        return circles
    
    # Fallback: return the initial configuration with some adjustments
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = initial_positions[i, 0]
        circles[i, 1] = initial_positions[i, 1]
        circles[i, 2] = initial_radii[i]
    return circles


# EVOLVE-BLOCK-END
