# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        # Avoid division by zero
        if dmax == 0:
            return float('inf')
            
        # We want to maximize dmin/dmax, so we minimize -dmin/dmax
        return -dmin / dmax
    
    def constraint_func(x_flat):
        """Constraint function to keep points within [0,1] x [0,1]"""
        points = x_flat.reshape(-1, 2)
        # Return negative values where points are outside bounds
        return np.concatenate([
            points[:, 0],                    # x coordinates >= 0
            1 - points[:, 0],               # x coordinates <= 1
            points[:, 1],                    # y coordinates >= 0
            1 - points[:, 1]                # y coordinates <= 1
        ])
    
    # Initialize with a good starting configuration (hexagonal pattern)
    # Arrange points in a roughly hexagonal grid pattern
    n = 16
    points = np.zeros((n, 2))
    
    # Create a hexagonal grid pattern
    rows = 4
    cols = 4
    
    # Generate hexagonal grid with some randomness to avoid symmetry issues
    np.random.seed(42)
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < n:
                x = j * 0.25 + (i % 2) * 0.125 + np.random.normal(0, 0.01)
                y = i * 0.25 + np.random.normal(0, 0.01)
                points[i * cols + j] = [max(0.01, min(0.99, x)), max(0.01, min(0.99, y))]
    
    # Flatten for optimization
    x0 = points.flatten()
    
    # Define bounds (points must stay within [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(32)]
    
    # Define constraints
    constraints = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Use differential evolution for global optimization
    try:
        # Try multiple restarts with different initializations
        best_result = None
        best_ratio = -float('inf')
        
        for restart in range(5):
            # Add some noise to the initial guess
            np.random.seed(42 + restart)
            x_init = x0 + np.random.normal(0, 0.05, len(x0))
            x_init = np.clip(x_init, 0, 1)
            
            # Optimize
            result = minimize(
                objective,
                x_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6}
            )
            
            # Evaluate the final result
            final_points = result.x.reshape(-1, 2)
            distances = pdist(final_points)
            dmin = np.min(distances)
            dmax = np.max(distances)
            
            if dmax > 0:
                ratio = dmin / dmax
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = result
        
        if best_result is not None:
            final_points = best_result.x.reshape(-1, 2)
        else:
            final_points = points
            
    except Exception as e:
        # Fallback to initial hexagonal pattern if optimization fails
        final_points = points
    
    return final_points


# EVOLVE-BLOCK-END
