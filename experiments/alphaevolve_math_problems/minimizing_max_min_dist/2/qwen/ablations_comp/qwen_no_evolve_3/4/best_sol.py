# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses energy-based optimization with geometric constraints.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x_flat.reshape(-1, 2)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    def constraint_func(x_flat):
        """Constraint function to keep points within [0,1] x [0,1]"""
        points = x_flat.reshape(-1, 2)
        # Ensure all points are within bounds
        return np.concatenate([
            points[:, 0],           # x coordinates >= 0
            1 - points[:, 0],       # 1 - x coordinates >= 0
            points[:, 1],           # y coordinates >= 0  
            1 - points[:, 1]        # 1 - y coordinates >= 0
        ])
    
    # Multi-start optimization with different initial configurations
    best_ratio = -float('inf')
    best_points = None
    
    # Try multiple random starting configurations
    for _ in range(10):
        # Generate random initial points in [0,1] x [0,1]
        np.random.seed(random.randint(0, 10000))
        x0 = np.random.rand(32)  # 16 points * 2 coordinates
        x0 = np.clip(x0, 0, 1)
        
        # Define constraints
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                points = result.x.reshape(-1, 2)
                distances = pdist(points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points.copy()
                        
        except Exception:
            continue
    
    # If no good solution found, use a more systematic approach
    if best_points is None:
        # Use a more structured approach: hexagonal lattice with perturbations
        # Create roughly equidistant points in a grid-like pattern
        points = []
        rows, cols = 4, 4  # 4x4 grid for 16 points
        
        # Generate points in a grid pattern
        for i in range(rows):
            for j in range(cols):
                x = i / (rows - 1) if rows > 1 else 0.5
                y = j / (cols - 1) if cols > 1 else 0.5
                points.append([x, y])
        
        points = np.array(points)
        
        # Add small random perturbations to improve the distribution
        np.random.seed(42)
        points += np.random.normal(0, 0.02, points.shape)
        
        # Clip to [0,1] bounds
        points = np.clip(points, 0, 1)
        
        return points
    
    return best_points


# EVOLVE-BLOCK-END
