# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
from itertools import combinations

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Better hexagonal initialization that creates a proper hexagonal pattern
    def initialize_hexagonal():
        circles = []
        
        # Create a hexagonal packing pattern
        # For 26 circles, we can use approximately 5 rows and 6 columns
        rows = 5
        cols = 6
        
        # Calculate spacing based on hexagonal packing
        # In perfect hexagonal packing, horizontal spacing = 2 * radius
        # Vertical spacing = sqrt(3) * radius
        initial_radius = 0.12  # Start with a reasonable guess
        
        # Generate hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Offset every other row for hexagonal packing
                x_offset = j + (i % 2) * 0.5
                y_offset = i * math.sqrt(3) / 2
                
                # Scale to fit nicely in [0,1]x[0,1] 
                max_x = cols + 0.5
                max_y = rows * math.sqrt(3) / 2
                
                x = x_offset / max_x * 0.8 + 0.1  # Center in the middle
                y = y_offset / max_y * 0.8 + 0.1
                
                # Make sure we're within bounds
                x = max(initial_radius, min(1 - initial_radius, x))
                y = max(initial_radius, min(1 - initial_radius, y))
                
                circles.append([x, y, initial_radius])
        
        # Fill remaining circles with strategic placement
        while len(circles) < n:
            # Add some circles near edges for better coverage
            edge_positions = [
                [0.1, 0.5, initial_radius], [0.9, 0.5, initial_radius],
                [0.5, 0.1, initial_radius], [0.5, 0.9, initial_radius],
                [0.2, 0.2, initial_radius], [0.8, 0.2, initial_radius],
                [0.2, 0.8, initial_radius], [0.8, 0.8, initial_radius]
            ]
            
            for x, y, r in edge_positions:
                if len(circles) >= n:
                    break
                x = max(initial_radius, min(1 - initial_radius, x))
                y = max(initial_radius, min(1 - initial_radius, y))
                circles.append([x, y, initial_radius])
        
        # Trim to exactly n circles
        circles = circles[:n]
        return np.array(circles)
    
    # Efficient constraint checking using vectorized operations
    def check_constraints(circles):
        """Fast constraint validation"""
        # Check containment
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check non-overlap using vectorized computation
        if len(circles) > 1:
            coords = circles[:, :2]  # x, y coordinates
            radii = circles[:, 2]    # radii
            
            # Compute pairwise distances
            distances = np.sqrt(((coords[:, None] - coords[None, :])**2).sum(axis=2))
            
            # Create mask for upper triangle (avoid duplicate pairs)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            
            # Check non-overlap condition: distance >= sum of radii
            distance_matrix = distances[mask]
            radius_sums = (radii[:, None] + radii[None, :])[mask]
            
            # If any pair violates non-overlap, return False
            if np.any(distance_matrix < radius_sums):
                return False
                
        return True
    
    # Objective function to maximize sum of radii
    def objective(params):
        # Return negative because we want to maximize
        return -sum(params[2::3])  # Sum of all radii (every third element starting from index 2)
    
    # Constraint functions for scipy optimization
    def constraint_containment(params):
        circles = params.reshape(-1, 3)
        constraints = []
        for i in range(n):
            x, y, r = circles[i]
            # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
            constraints.extend([
                x - r,           # x - r >= 0
                1 - x - r,       # 1 - x - r >= 0  
                y - r,           # y - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        return np.array(constraints)
    
    def constraint_nonoverlap(params):
        circles = params.reshape(-1, 3)
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # Distance - (r1 + r2) >= 0 (non-overlap constraint)
                constraints.append(dist - (r1 + r2))
        return np.array(constraints)
    
    # Multi-start optimization approach
    best_sum = -np.inf
    best_circles = None
    
    # Strategy 1: Multiple initializations with slight variations
    for start_iter in range(10):  # More diverse starts
        # Initialize with better hexagonal layout
        circles = initialize_hexagonal()
        
        # Slightly perturb the initial configuration for diversity
        if start_iter > 0:
            np.random.seed(start_iter * 42)  # Different seed for variety
            for i in range(len(circles)):
                # Add small random perturbations to positions
                circles[i][0] += np.random.normal(0, 0.015)  # Small perturbation
                circles[i][1] += np.random.normal(0, 0.015)
                # Keep within bounds
                circles[i][0] = np.clip(circles[i][0], 0.01, 0.99)
                circles[i][1] = np.clip(circles[i][1], 0.01, 0.99)
        
        # Flatten for optimization
        initial_params = circles.flatten()
        
        # Set up bounds for parameters (x,y in [0,1], r in [0.001, 0.499])
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Constraints for optimization
        constraints = [
            {'type': 'ineq', 'fun': constraint_containment},
            {'type': 'ineq', 'fun': constraint_nonoverlap}
        ]
        
        # Run optimization with different methods for robustness
        try:
            # Try SLSQP first
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'disp': False}
            )
            
            if result.success:
                optimized_params = result.x
                optimized_circles = optimized_params.reshape(-1, 3)
                
                # Validate the solution
                if check_constraints(optimized_circles):
                    sum_radii = sum(optimized_circles[:, 2])
                    if sum_radii > best_sum:
                        best_sum = sum_radii
                        best_circles = optimized_circles.copy()
                        
        except Exception as e:
            # Continue with next iteration if optimization fails
            continue
    
    # If no successful optimization found, return the best initialization
    if best_circles is None:
        circles = initialize_hexagonal()
        # Final validation and cleanup
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Ensure containment constraint
            r = min(r, x, 1-x, y, 1-y)
            circles[i] = [x, y, r]
        return circles
    
    # Final validation of best result
    for i in range(len(best_circles)):
        x, y, r = best_circles[i]
        # Ensure containment constraint
        r = min(r, x, 1-x, y, 1-y)
        best_circles[i] = [x, y, r]
        
    return best_circles


# EVOLVE-BLOCK-END
