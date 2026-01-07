# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
import random
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining good initialization and numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    # Initialize positions using a hexagonal grid pattern for good initial distribution
    def initialize_positions():
        # Create a hexagonal grid pattern with better spacing
        rows = 6
        cols = 6
        positions = []
        
        # Hexagonal grid spacing - leave some padding
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows
        offset = spacing_x * 0.5
        
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                x = 0.05 + (j + (i % 2) * 0.5) * spacing_x
                y = 0.05 + i * spacing_y
                if x <= 0.95 and y <= 0.95:
                    positions.append([x, y])
        
        # Fill remaining positions randomly within bounds
        while len(positions) < n:
            positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
            
        return np.array(positions[:n])
    
    def pack_circles(initial_positions):
        """Use direct optimization approach with improved constraints"""
        # Initialize with small radii
        initial_vars = []
        for i in range(n):
            x, y = initial_positions[i]
            # Initial radius is small but feasible
            r = min(x, y, 1-x, 1-y) * 0.2
            initial_vars.extend([x, y, r])
        
        # Optimization objective: maximize sum of radii (minimize negative sum)
        def objective(vars):
            total_radius = sum(vars[3*i+2] for i in range(n))
            return -total_radius  # Negative because we minimize
        
        # Constraints for containment and non-overlap
        constraints = []
        
        # Boundary constraints for each circle (radius and containment)
        for i in range(n):
            # r > 0
            constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: vars[3*i+2]})
            # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: vars[3*i] - vars[3*i+2]})
            # 1 - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: 1 - vars[3*i] - vars[3*i+2]})
            # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: vars[3*i+1] - vars[3*i+2]})
            # 1 - y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda vars, i=i: 1 - vars[3*i+1] - vars[3*i+2]})
        
        # Overlap constraints - use spatial indexing for efficiency
        # Only check nearby pairs to reduce computation
        positions = np.array(initial_positions)
        # Create KDTree for faster neighbor search
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        
        # Check overlaps efficiently using neighbors
        for i in range(n):
            # Find nearby points (within 2*max_radius range)
            nearby_indices = tree.query_ball_point(positions[i], 2.0)
            for j in nearby_indices:
                if i < j:  # Avoid duplicate pairs and self-comparison
                    constraints.append({
                        'type': 'ineq', 
                        'fun': lambda vars, i=i, j=j: 
                            np.sqrt((vars[3*i] - vars[3*j])**2 + (vars[3*i+1] - vars[3*j+1])**2) - (vars[3*i+2] + vars[3*j+2])
                    })
        
        # Bounds for variables
        bounds = []
        for i in range(n):
            # x bounds
            bounds.append((0.001, 0.999))
            # y bounds  
            bounds.append((0.001, 0.999))
            # r bounds
            bounds.append((0.001, 0.499))
        
        # Optimize using SLSQP which handles constraints better
        try:
            result = minimize(
                objective, 
                initial_vars, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'iprint': 0}
            )
            if result.success:
                final_vars = result.x
                circles = np.zeros((n, 3))
                for i in range(n):
                    circles[i] = [final_vars[3*i], final_vars[3*i+1], final_vars[3*i+2]]
                return circles
        except Exception as e:
            pass
        
        # Fallback: return initial positions with adjusted radii
        circles = np.zeros((n, 3))
        for i in range(n):
            x, y = initial_positions[i]
            r = min(x, y, 1-x, 1-y) * 0.3
            circles[i] = [x, y, r]
        return circles
    
    # Try several optimization attempts with different initializations
    best_circles = None
    best_sum = 0
    
    # Try with different initial configurations
    for attempt in range(5):  # Reduced from 10 to improve speed
        # Create different initial positions
        if attempt == 0:
            # Use hexagonal grid
            positions = initialize_positions()
        elif attempt == 1:
            # Random positions
            positions = np.array([[np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)] for _ in range(n)])
        else:
            # Perturb the hexagonal grid
            positions = initialize_positions()
            for i in range(n):
                positions[i] += np.random.normal(0, 0.02, 2)
                positions[i] = np.clip(positions[i], 0.05, 0.95)
        
        circles = pack_circles(positions)
        
        # Calculate sum of radii
        radii_sum = np.sum(circles[:, 2])
        if radii_sum > best_sum:
            best_sum = radii_sum
            best_circles = circles
    
    # If we still don't have a good solution, return the fallback
    if best_circles is None:
        # Final fallback: simple grid-based solution
        best_circles = np.zeros((n, 3))
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                r = min(x, y, 1-x, 1-y) * 0.4
                best_circles[idx] = [x, y, r]
                idx += 1
        # Fill remaining circles
        for i in range(idx, n):
            best_circles[i] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.05]
    
    return best_circles


# EVOLVE-BLOCK-END
