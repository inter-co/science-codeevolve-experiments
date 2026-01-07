# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining multiple strategies for better optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    np.random.seed(42)
    n = 32
    
    # Initialize with a sophisticated starting configuration using hexagonal packing approach
    def initialize_config():
        circles = np.zeros((n, 3))
        
        # Create a hexagonal packing pattern optimized for 32 circles
        # This is inspired by known optimal arrangements for circle packing
        
        # For 32 circles, a good approximation is a 6x6 grid with 4 extra circles
        rows = 6
        cols = 6
        
        # Hexagonal packing with proper offsets
        spacing_x = 0.8 / (cols - 1) if cols > 1 else 0.5
        spacing_y = 0.8 / (rows - 1) if rows > 1 else 0.5
        offset_y = spacing_y * 0.5  # Half spacing for hexagonal pattern
        
        # Fill the grid with hexagonal packing
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                # Add hexagonal offset for odd rows
                x_offset = 0 if row % 2 == 0 else spacing_x * 0.5
                x = 0.1 + x_offset + col * spacing_x
                y = 0.1 + row * spacing_y
                
                # Add small random jitter to avoid degenerate cases
                x += np.random.normal(0, 0.003)
                y += np.random.normal(0, 0.003)
                
                # Clamp to valid range
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Initial radius guess - start with something reasonable
                r = 0.04
                
                circles[idx] = [x, y, r]
                idx += 1
                if idx >= n:
                    break
        
        # Fill remaining circles with a different strategy
        # Place additional circles in a way that avoids clustering
        for i in range(idx, n):
            if i < n:
                # Use a spiral pattern for remaining circles to distribute well
                angle = (i - idx) * 0.5
                radius = 0.3 * (1.0 - (i - idx) / (n - idx)) if (n - idx) > 0 else 0.1
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                r = 0.03
                circles[i] = [x, y, r]
        
        return circles
    
    # Validate if configuration is valid (no overlaps, all contained)
    def validate_configuration(circles):
        # Check containment
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlaps - more efficient version
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Early termination if any circle is invalid
        for i in range(len(circles)):
            if radii[i] <= 0:
                return False
        
        # Check overlaps using distance matrix
        distances = cdist(positions, positions)
        
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                if dist < min_dist * 0.999:  # Small tolerance for numerical errors
                    return False
        return True
    
    # Calculate objective function (negative because we want to maximize)
    def objective(params):
        # Reshape params into circles array
        circles = params.reshape(-1, 3)
        radii = circles[:, 2]
        
        # Sum of radii (we want to maximize this)
        return -np.sum(radii)
    
    # Constraint functions for scipy.optimize
    def containment_constraint(params):
        """Ensure all circles are contained within unit square"""
        circles = params.reshape(-1, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Circle must fit entirely in square
        # These constraints should be >= 0 for feasibility
        con1 = x - r  # x - r >= 0
        con2 = y - r  # y - r >= 0  
        con3 = 1 - (x + r)  # x + r <= 1
        con4 = 1 - (y + r)  # y + r <= 1
        
        # Stack all constraints (positive means satisfied)
        return np.concatenate([con1, con2, con3, con4])
    
    def overlap_constraints(params):
        """Ensure no overlapping circles - returns negative values when violated"""
        circles = params.reshape(-1, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Compute pairwise distances
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # For each pair of circles, ensure they don't overlap
        # Return positive values when constraint is satisfied (distance >= r[i]+r[j])
        constraints = []
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dist = distances[i, j]
                min_dist = r[i] + r[j]
                # Distance between centers must be >= sum of radii
                # Positive means satisfied, negative means violated
                constraints.append(dist - min_dist)
        
        return np.array(constraints)
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # x coordinate bounds (slightly away from edges to allow for radius)
        bounds.extend([(0.001, 0.999)])
        # y coordinate bounds  
        bounds.extend([(0.001, 0.999)])
        # radius bounds (max radius is 0.5, but we'll keep it smaller for safety)
        bounds.extend([(0.001, 0.499)])
    
    # Initialize
    initial_circles = initialize_config()
    initial_params = initial_circles.flatten()
    
    # Validate initial configuration
    if not validate_configuration(initial_circles):
        # If initial configuration is invalid, try a simpler approach
        initial_circles = np.zeros((n, 3))
        for i in range(n):
            x = 0.1 + 0.8 * (i % 6) / 5.0 + np.random.normal(0, 0.01)
            y = 0.1 + 0.8 * (i // 6) / 4.0 + np.random.normal(0, 0.01)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            r = 0.02
            initial_circles[i] = [x, y, r]
    
    # Try multiple optimization strategies for best results
    best_result = initial_params.copy()
    best_sum = -objective(initial_params)
    
    try:
        # Strategy 1: Differential Evolution (global optimization)
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False,
            polish=True,
            init='latinhypercube'
        )
        
        if de_result.success:
            de_circles = de_result.x.reshape(-1, 3)
            if validate_configuration(de_circles):
                de_sum = -objective(de_result.x)
                if de_sum > best_sum:
                    best_result = de_result.x.copy()
                    best_sum = de_sum
        
        # Strategy 2: Local optimization with trust-constr if differential evolution didn't work well
        if not de_result.success or best_sum < 2.0:  # If result seems poor, try local optimization
            # Define constraints for scipy.optimize
            cons = [
                {'type': 'ineq', 'fun': lambda x: containment_constraint(x)},
                {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
            ]
            
            # Use trust-constr optimizer which works well with constraints
            local_result = minimize(
                objective,
                initial_params,
                method='trust-constr',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 300, 'disp': False}
            )
            
            if local_result.success:
                local_circles = local_result.x.reshape(-1, 3)
                if validate_configuration(local_circles):
                    local_sum = -objective(local_result.x)
                    if local_sum > best_sum:
                        best_result = local_result.x.copy()
                        best_sum = local_sum
        
        # Strategy 3: Hybrid approach - run a few iterations of local optimization on best result
        if best_sum < 2.0:  # If still not good enough, try more optimization
            # Define constraints for scipy.optimize
            cons = [
                {'type': 'ineq', 'fun': lambda x: containment_constraint(x)},
                {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
            ]
            
            # Run more intensive local optimization
            refined_result = minimize(
                objective,
                best_result,
                method='trust-constr',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'disp': False}
            )
            
            if refined_result.success:
                refined_circles = refined_result.x.reshape(-1, 3)
                if validate_configuration(refined_circles):
                    refined_sum = -objective(refined_result.x)
                    if refined_sum > best_sum:
                        best_result = refined_result.x.copy()
                        best_sum = refined_sum
        
        final_circles = best_result.reshape(-1, 3)
        
        # Final validation
        if validate_configuration(final_circles):
            return final_circles
        else:
            # Return initial configuration if final result is invalid
            return initial_circles
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        return initial_circles


# EVOLVE-BLOCK-END
