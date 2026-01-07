# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, constraint satisfaction, 
    and local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Improved hexagonal grid initialization with better packing
    def initialize_hexagonal_grid():
        circles = np.zeros((n, 3))
        
        # Use a more optimal grid arrangement for 32 circles
        rows = 6
        cols = 6
        spacing_x = 0.9 / cols  # Leave 0.05 margin on each side
        spacing_y = 0.9 / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = (i % 2) * spacing_x / 2
                x = 0.05 + j * spacing_x + x_offset
                y = 0.05 + i * spacing_y
                
                # Set initial radius to be reasonable
                r = min(spacing_x, spacing_y) * 0.3
                
                # Ensure within bounds
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
                
        return circles
    
    # Better systematic grid initialization
    def initialize_systematic_grid():
        circles = np.zeros((n, 3))
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing_x = 0.9 / grid_size
        spacing_y = 0.9 / grid_size
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                y = 0.05 + (i + 0.5) * spacing_y
                r = min(spacing_x, spacing_y) * 0.3
                
                # Ensure within bounds
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
                
        return circles
    
    # Efficient constraint checking using KDTree
    def calculate_fitness(circles: np.ndarray) -> float:
        """Calculate fitness as the sum of radii with penalties for violations"""
        # Extract radii
        radii = circles[:, 2]
        
        # Calculate total radius sum
        total_radius = np.sum(radii)
        
        # Penalty for overlap violations
        penalty = 0.0
        
        # Check for containment violations
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 1000  # Large penalty for containment violation
        
        # Check for overlap violations efficiently using KDTree
        positions = circles[:, :2]
        tree = cKDTree(positions)
        
        # Query for nearby points to reduce complexity
        pairs = tree.query_pairs(0.001)  # Very small threshold to catch overlaps
        
        for i, j in pairs:
            if i < j:  # Only check each pair once
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < (r1 + r2):  # Overlap detected
                    # Penalty based on how much they overlap
                    overlap = (r1 + r2) - distance
                    penalty += overlap * 1000
        
        return total_radius - penalty
    
    # Local optimization using scipy
    def local_optimization(circles: np.ndarray) -> np.ndarray:
        """Refine solution using local optimization"""
        n = len(circles)
        
        # Flatten initial parameters [x1,y1,r1,x2,y2,r2,...]
        initial_params = circles.flatten()
        
        # Objective function to maximize sum of radii (minimize negative sum)
        def objective(params):
            total_radius = 0
            for i in range(n):
                total_radius += params[3*i + 2]  # radii are at indices 2,5,8,...
            return -total_radius  # negative because we want to maximize
        
        # Constraint functions for scipy.optimize
        def contain_constraints(params):
            """Ensure all circles stay within the unit square"""
            constraints = []
            for i in range(n):
                x, y, r = params[3*i], params[3*i+1], params[3*i+2]
                # Circle must fit within bounds: x-r >= 0, y-r >= 0, x+r <= 1, y+r <= 1
                constraints.extend([
                    x - r,      # x - r >= 0
                    y - r,      # y - r >= 0  
                    1 - x - r,  # 1 - x - r >= 0
                    1 - y - r   # 1 - y - r >= 0
                ])
            return np.array(constraints)
        
        def overlap_constraints(params):
            """Ensure no two circles overlap"""
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                    x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                    # Distance between centers should be >= sum of radii
                    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    constraints.append(distance - (r1 + r2))  # Should be >= 0
            return np.array(constraints)
        
        # Set up bounds: x,y in [0,1], r in [0.001, 0.5] to avoid numerical issues
        bounds = []
        for i in range(n):
            bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])
        
        # Set up constraints for scipy.optimize
        constraints = [
            {'type': 'ineq', 'fun': lambda p: contain_constraints(p)},
            {'type': 'ineq', 'fun': lambda p: overlap_constraints(p)}
        ]
        
        try:
            # Use scipy's minimize with SLSQP method for better constraint handling
            result = minimize(
                objective, 
                initial_params, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                # Extract refined solution
                refined_circles = np.zeros((n, 3))
                for i in range(n):
                    refined_circles[i] = [
                        result.x[3*i],     # x coordinate
                        result.x[3*i+1],   # y coordinate
                        result.x[3*i+2]    # radius
                    ]
                return refined_circles
        except:
            pass
            
        return circles
    
    # Constraint satisfaction refinement with better logic
    def refine_with_constraint_satisfaction(circles):
        """Apply constraint satisfaction refinement to improve solution"""
        improved = True
        max_iterations = 30
        
        for iteration in range(max_iterations):
            if not improved:
                break
            improved = False
            
            # Try to increase radii while maintaining constraints
            for i in range(len(circles)):
                x, y, r = circles[i]
                
                # Compute maximum possible radius
                max_r = min(x, y, 1-x, 1-y)
                
                # Check distance to all other circles
                for j in range(len(circles)):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dist = math.sqrt((x-x2)**2 + (y-y2)**2)
                        if dist < (r + r2 + 0.001):  # Could potentially overlap
                            max_r = min(max_r, dist - r2 - 0.001)
                
                # Try to increase radius if beneficial
                if max_r > r + 0.001 and max_r > 0.001:
                    new_r = min(max_r, r + 0.005)  # Smaller increment
                    # Test if this change is valid with all neighbors
                    temp_circles = circles.copy()
                    temp_circles[i] = [x, y, new_r]
                    
                    # Check if this change maintains constraints
                    valid = True
                    for j in range(len(temp_circles)):
                        if i != j:
                            x2, y2, r2 = temp_circles[j]
                            dist = math.sqrt((x-x2)**2 + (y-y2)**2)
                            if dist < (new_r + r2):
                                valid = False
                                break
                    
                    if valid:
                        circles[i] = [x, y, new_r]
                        improved = True
                        
        return circles
    
    # Try multiple initialization strategies and optimization attempts
    best_solution = None
    best_sum = 0
    
    # Multiple initialization attempts
    initial_attempts = [
        initialize_hexagonal_grid,
        initialize_systematic_grid
    ]
    
    # Try different seeds for better exploration
    seeds = [42, 123, 456, 789]
    
    for seed in seeds:
        np.random.seed(seed)
        
        for init_func in initial_attempts:
            try:
                circles = init_func()
                
                # Apply local optimization first
                optimized = local_optimization(circles)
                
                # Then apply constraint satisfaction refinement
                refined = refine_with_constraint_satisfaction(optimized)
                
                # Calculate fitness of refined solution
                fitness = calculate_fitness(refined)
                
                if fitness > best_sum:
                    best_sum = fitness
                    best_solution = refined.copy()
                    
            except Exception as e:
                continue
    
    # If no good solution found, return the best we have
    if best_solution is None:
        # Fallback to hexagonal grid with local optimization
        circles = initialize_hexagonal_grid()
        best_solution = local_optimization(circles)
    
    return best_solution


# EVOLVE-BLOCK-END
