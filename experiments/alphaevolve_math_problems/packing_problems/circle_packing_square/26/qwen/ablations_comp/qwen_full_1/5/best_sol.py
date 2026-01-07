# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal initial placement with advanced optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Generate better initial placement using hexagonal lattice approach (INSPIRATION 2)
    def generate_hexagonal_initial_placement(num_circles):
        """Generate initial placement using hexagonal lattice pattern"""
        # Hexagonal packing in a square grid
        rows = int(np.ceil(np.sqrt(num_circles)))
        cols = int(np.ceil(num_circles / rows))
        
        positions = []
        spacing = 1.0 / max(rows, cols)
        hex_spacing = spacing * np.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= num_circles:
                    break
                # Hexagonal offset pattern
                x = (j + 0.5 + (i % 2) * 0.5) * spacing
                y = (i + 0.5) * hex_spacing
                # Add small random perturbation to avoid perfect patterns
                x += np.random.uniform(-0.01, 0.01)
                y += np.random.uniform(-0.01, 0.01)
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                positions.append([x, y])
        return np.array(positions[:num_circles])
    
    # Generate Voronoi-inspired initial placement
    def generate_voronoi_initial_placement(num_circles):
        """Generate initial placement using Voronoi diagram concept"""
        positions = []
        for _ in range(num_circles):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            positions.append([x, y])
        return np.array(positions)
    
    # Better radius estimation based on neighbor distances (INSPIRATION 2)
    def estimate_radii(positions, n):
        radii = []
        for i in range(n):
            x, y = positions[i]
            # Maximum possible radius based on boundaries
            r_max = min(x, 1-x, y, 1-y)
            
            # Find minimum distance to any other circle to estimate safe radius
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dx = x - positions[j][0]
                    dy = y - positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
            
            # Safe radius considering neighbors and boundaries
            if min_dist < 2.0:
                r_safe = min(r_max, min_dist/2.0 - 1e-6)
            else:
                r_safe = r_max
            
            # Final radius with some safety factor
            r_final = max(0.01, min(0.4, r_safe * 0.95))
            radii.append(r_final)
        
        return np.array(radii)
    
    # Constraint functions with better numerical stability (INSPIRATION 2)
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
    
    # Objective function to maximize (negative because we minimize)
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
    
    # Try multiple strategies with enhanced restarts (INSPIRATION 2 approach)
    best_result = None
    best_sum = -np.inf
    
    # Strategy 1: Hexagonal lattice initialization with SLSQP (INSPIRATION 2)
    try:
        initial_positions = generate_hexagonal_initial_placement(n)
        initial_radii = estimate_radii(initial_positions, n)
        initial_params = np.concatenate([initial_radii, initial_positions.flatten()])
        
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                {'type': 'ineq', 'fun': lambda p: non_overlap_constraints(p)}
            ],
            options={'maxiter': 2500, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        if result.success:
            final_radii = result.x[:n]
            current_sum = np.sum(final_radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = result
                
    except Exception as e:
        pass
    
    # Strategy 2: Multiple restarts with different initializations (INSPIRATION 2)
    # This helps escape local optima and improves results significantly
    for restart in range(15):  # Increased restarts for better exploration
        try:
            # Choose different initialization strategy based on restart number
            if restart % 5 == 0:
                # Hexagonal pattern
                positions = generate_hexagonal_initial_placement(n)
            elif restart % 5 == 1:
                # Voronoi-like pattern
                positions = generate_voronoi_initial_placement(n)
            elif restart % 5 == 2:
                # Grid pattern with larger perturbations
                rows = int(np.ceil(np.sqrt(n)))
                cols = int(np.ceil(n / rows))
                positions = []
                for i in range(rows):
                    for j in range(cols):
                        if len(positions) >= n:
                            break
                        x = (j + 0.5) / cols
                        y = (i + 0.5) / rows
                        # Larger perturbations for more exploration
                        x += np.random.uniform(-0.06, 0.06)
                        y += np.random.uniform(-0.06, 0.06)
                        x = max(0.05, min(0.95, x))
                        y = max(0.05, min(0.95, y))
                        positions.append([x, y])
                positions = np.array(positions[:n])
            elif restart % 5 == 3:
                # Another hexagonal variation with even more randomness
                positions = generate_hexagonal_initial_placement(n)
                # Apply more significant perturbations
                for i in range(len(positions)):
                    positions[i] += np.random.normal(0, 0.04, 2)
                    positions[i][0] = np.clip(positions[i][0], 0.05, 0.95)
                    positions[i][1] = np.clip(positions[i][1], 0.05, 0.95)
            else:
                # Random pattern
                positions = generate_voronoi_initial_placement(n)
                # Slight perturbation to avoid degenerate cases
                for i in range(len(positions)):
                    positions[i] += np.random.uniform(-0.03, 0.03, 2)
                    positions[i][0] = np.clip(positions[i][0], 0.05, 0.95)
                    positions[i][1] = np.clip(positions[i][1], 0.05, 0.95)
            
            # Compute better initial radii
            radii = estimate_radii(positions, n)
            params = np.concatenate([radii, positions.flatten()])
            
            result_restart = minimize(
                objective,
                params,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                    {'type': 'ineq', 'fun': lambda p: non_overlap_constraints(p)}
                ],
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            if result_restart.success:
                final_radii = result_restart.x[:n]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_restart
                    
        except Exception as e:
            continue
    
    # Strategy 3: L-BFGS-B with penalty method for backup (INSPIRATION 2)
    if best_result is None or best_sum < 2.0:
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
            
            # Try L-BFGS-B with penalty method using best known solution so far
            if best_result is not None:
                # Use the best parameters we have so far
                initial_params = best_result.x
            else:
                # Fallback to hexagonal initialization
                initial_positions = generate_hexagonal_initial_placement(n)
                initial_radii = estimate_radii(initial_positions, n)
                initial_params = np.concatenate([initial_radii, initial_positions.flatten()])
            
            result_penalty = minimize(
                penalized_objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-10}
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
    
    # Fallback: use a simple hexagonal arrangement with basic optimization
    fallback_positions = generate_hexagonal_initial_placement(n)
    fallback_radii = estimate_radii(fallback_positions, n)
    circles = np.column_stack([fallback_positions, fallback_radii])
    return circles


# EVOLVE-BLOCK-END
