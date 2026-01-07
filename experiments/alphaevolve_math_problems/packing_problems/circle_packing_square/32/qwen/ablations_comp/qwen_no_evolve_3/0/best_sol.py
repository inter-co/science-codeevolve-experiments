# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
from math import sqrt

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Geometric initialization using hexagonal packing pattern
    def initialize_hexagonal():
        # Create a hexagonal grid pattern that fits in unit square
        # For 32 circles, we'll use approximately 6x6 grid with some adjustments
        rows = 6
        cols = 6
        if rows * cols < n:
            rows = 5
            cols = 7
            
        # Calculate spacing
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Adjust for hexagonal packing
        hex_spacing = spacing_x * 0.866  # sqrt(3)/2
        
        circles = []
        
        # Place circles in a hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row
                x_offset = (j * spacing_x) + (i % 2) * (spacing_x / 2)
                y_offset = i * spacing_y
                
                # Ensure we're within bounds and create initial circle
                if x_offset > 0.5 * spacing_x and y_offset > 0.5 * spacing_y:
                    x = min(x_offset, 1.0 - 0.5 * spacing_x)
                    y = min(y_offset, 1.0 - 0.5 * spacing_y)
                    
                    # Initial radius - small enough to fit
                    r = min(spacing_x, spacing_y) * 0.2
                    circles.append([x, y, r])
        
        # Fill remaining positions
        while len(circles) < n:
            # Random placement with small radius
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = random.uniform(0.01, 0.05)
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Phase 2: Constraint validation and adjustment
    def validate_circles(circles):
        """Ensure all circles are within bounds and non-overlapping"""
        valid_circles = circles.copy()
        
        # Check containment constraints
        for i in range(len(valid_circles)):
            x, y, r = valid_circles[i]
            # Adjust radius if needed to stay within bounds
            r = min(r, x, y, 1-x, 1-y)
            valid_circles[i] = [x, y, max(0.001, r)]
        
        # Resolve overlaps
        for _ in range(100):  # Limit iterations
            overlaps = False
            for i in range(len(valid_circles)):
                for j in range(i+1, len(valid_circles)):
                    x1, y1, r1 = valid_circles[i]
                    x2, y2, r2 = valid_circles[j]
                    
                    dist = sqrt((x1-x2)**2 + (y1-y2)**2)
                    if dist < (r1 + r2):
                        # Reduce both radii to resolve overlap
                        total_reduction = (r1 + r2 - dist) * 0.5
                        valid_circles[i][2] = max(0.001, valid_circles[i][2] - total_reduction * 0.3)
                        valid_circles[j][2] = max(0.001, valid_circles[j][2] - total_reduction * 0.3)
                        overlaps = True
            
            if not overlaps:
                break
                
        return valid_circles
    
    # Phase 3: Local optimization using gradient-based method
    def optimize_circles(initial_circles):
        # Convert to flat parameter vector [x1,y1,r1,x2,y2,r2,...]
        def pack_params(circles):
            params = []
            for i in range(len(circles)):
                params.extend([circles[i][0], circles[i][1], circles[i][2]])
            return np.array(params)
        
        def unpack_params(params):
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [params[3*i], params[3*i+1], params[3*i+2]]
            return circles
        
        # Objective function to maximize sum of radii (minimize negative sum)
        def objective(params):
            circles = unpack_params(params)
            return -np.sum(circles[:, 2])
        
        # Constraints for non-overlap and containment
        def constraint_nonoverlap(params):
            circles = unpack_params(params)
            # Return positive values when constraints are satisfied
            violations = []
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = sqrt((x1-x2)**2 + (y1-y2)**2)
                    # Constraint violation: should be <= 0 when circles don't overlap
                    violations.append(dist - (r1 + r2))
            return np.array(violations)
        
        def constraint_containment(params):
            circles = unpack_params(params)
            # Return positive values when constraints are satisfied
            violations = []
            for i in range(len(circles)):
                x, y, r = circles[i]
                # Each constraint: r <= x <= 1-r, r <= y <= 1-r
                violations.extend([
                    x - r,           # x >= r
                    1 - x - r,       # x <= 1-r
                    y - r,           # y >= r
                    1 - y - r        # y <= 1-r
                ])
            return np.array(violations)
        
        # Initial parameters
        initial_params = pack_params(initial_circles)
        
        # Set up bounds for parameters (x, y, r)
        bounds = []
        for i in range(n):
            # x bounds: [r, 1-r] 
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Optimization with constraints
        try:
            # Use SLSQP for constrained optimization
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: constraint_nonoverlap(p)},
                    {'type': 'ineq', 'fun': lambda p: constraint_containment(p)}
                ],
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_circles = unpack_params(result.x)
                return optimized_circles
        except:
            pass
        
        return initial_circles
    
    # Phase 4: Final refinement with simulated annealing-inspired approach
    def refine_with_sanneal(initial_circles):
        current_circles = initial_circles.copy()
        best_circles = current_circles.copy()
        best_sum = np.sum(current_circles[:, 2])
        
        # Simulated annealing parameters
        temp = 0.1
        cooling_rate = 0.999
        min_temp = 0.001
        max_iterations = 1000
        
        for iteration in range(max_iterations):
            if temp < min_temp:
                break
                
            # Make small random perturbations
            new_circles = current_circles.copy()
            
            # Select random circle to modify
            idx = random.randint(0, n-1)
            # Perturb x, y, and r slightly
            new_circles[idx][0] += random.uniform(-0.01, 0.01)
            new_circles[idx][1] += random.uniform(-0.01, 0.01)
            new_circles[idx][2] += random.uniform(-0.005, 0.005)
            
            # Keep within bounds
            new_circles[idx][0] = np.clip(new_circles[idx][0], 0.001, 0.999)
            new_circles[idx][1] = np.clip(new_circles[idx][1], 0.001, 0.999)
            new_circles[idx][2] = np.clip(new_circles[idx][2], 0.001, 0.499)
            
            # Check if new configuration is valid (non-overlapping)
            if is_valid_configuration(new_circles):
                new_sum = np.sum(new_circles[:, 2])
                delta = new_sum - best_sum
                
                # Accept or reject based on temperature
                if delta > 0 or random.random() < np.exp(delta / temp):
                    current_circles = new_circles
                    if new_sum > best_sum:
                        best_circles = new_circles.copy()
                        best_sum = new_sum
            
            temp *= cooling_rate
            
        return best_circles
    
    def is_valid_configuration(circles):
        """Check if circles don't overlap and are contained"""
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Check containment
            if x < r or x > 1-r or y < r or y > 1-r:
                return False
            # Check non-overlap with others
            for j in range(i+1, len(circles)):
                x2, y2, r2 = circles[j]
                dist = sqrt((x-x2)**2 + (y-y2)**2)
                if dist < (r + r2):
                    return False
        return True
    
    # Execute the phases
    # Phase 1: Initialize with hexagonal packing
    circles = initialize_hexagonal()
    
    # Phase 2: Validate and adjust for constraints
    circles = validate_circles(circles)
    
    # Phase 3: Local optimization
    circles = optimize_circles(circles)
    
    # Phase 4: Final refinement
    circles = refine_with_sanneal(circles)
    
    # Final validation and cleanup
    circles = validate_circles(circles)
    
    return circles


# EVOLVE-BLOCK-END
