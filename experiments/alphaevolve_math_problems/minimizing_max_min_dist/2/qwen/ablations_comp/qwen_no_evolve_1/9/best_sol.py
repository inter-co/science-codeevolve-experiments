# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining structured initialization with optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    np.random.seed(42)
    
    # Strategy 1: Start with a structured hexagonal arrangement
    # This provides a good initial configuration that's already relatively well-distributed
    def generate_hexagonal_grid():
        # Create a hexagonal grid pattern that fits within [0,1] x [0,1]
        rows = 4
        cols = 4
        points = []
        
        # Hexagonal packing with spacing
        spacing_x = 1.0 / (cols - 1)
        spacing_y = 1.0 / (rows - 1)
        
        for i in range(rows):
            for j in range(cols):
                x = j * spacing_x
                y = i * spacing_y
                
                # Offset every other row for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                # Ensure points stay within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                
                points.append([x, y])
        
        # Take first 16 points (in case we have extras due to rounding)
        return np.array(points[:n])
    
    # Strategy 2: Energy-based optimization approach
    def energy_objective(points_flat):
        """Objective function to maximize the min/max distance ratio"""
        points = points_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero and penalize small ratios
        if max_dist <= 1e-10:
            return -np.inf
            
        ratio = min_dist / max_dist
        
        # Return negative because we want to maximize (minimize negative)
        return -ratio
    
    def constraint_func(points_flat):
        """Constraint to keep points within [0,1] x [0,1]"""
        points = points_flat.reshape(-1, 2)
        # Return difference from bounds (negative values indicate violations)
        return np.concatenate([
            points[:, 0],           # x coordinates
            1 - points[:, 0],       # 1 - x coordinates  
            points[:, 1],           # y coordinates
            1 - points[:, 1]        # 1 - y coordinates
        ])
    
    # Generate initial points
    initial_points = generate_hexagonal_grid()
    
    # Flatten for optimization
    initial_flat = initial_points.flatten()
    
    # Optimization bounds (points must stay in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # Apply optimization
    try:
        result = minimize(
            energy_objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            # Ensure all points are within bounds
            optimized_points = np.clip(optimized_points, 0, 1)
            return optimized_points
        else:
            # Fallback to initial points if optimization fails
            return initial_points
            
    except Exception:
        # Fallback to initial points if optimization fails
        return initial_points


# EVOLVE-BLOCK-END
