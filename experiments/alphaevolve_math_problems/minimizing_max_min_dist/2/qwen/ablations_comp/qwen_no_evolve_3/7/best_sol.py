# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def objective(x_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        if max_dist == 0:
            return -1.0  # Avoid division by zero
        return -min_dist / max_dist
    
    def constraint_func(x_flat):
        """Constraint function to keep points within [0,1] x [0,1]"""
        points = x_flat.reshape(-1, 2)
        # Return values >= 0 for all constraints to be satisfied
        # We want 0 <= x <= 1 and 0 <= y <= 1
        return np.concatenate([
            points[:, 0],                    # x coordinates
            1 - points[:, 0],                # 1 - x coordinates  
            points[:, 1],                    # y coordinates
            1 - points[:, 1]                 # 1 - y coordinates
        ])
    
    # Initialize points using a more strategic approach than pure randomness
    # Using a hexagonal lattice pattern scaled appropriately
    n = 16
    np.random.seed(42)
    
    # Create initial configuration using a structured approach
    # Arrange points in a roughly grid-like pattern with some perturbation
    rows = 4
    cols = 4
    points = []
    
    # Generate points in a grid pattern
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            points.append([x, y])
    
    # Add small random perturbations to avoid degenerate cases
    points = np.array(points) + np.random.normal(0, 0.05, (n, 2))
    
    # Clip to ensure all points stay within bounds
    points = np.clip(points, 0, 1)
    
    # Flatten for optimization
    x0 = points.flatten()
    
    # Define constraints
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # Perform optimization
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            # Ensure points are within bounds
            optimized_points = np.clip(optimized_points, 0, 1)
            return optimized_points
        else:
            # If optimization fails, return the initial configuration
            return points
            
    except Exception:
        # If optimization fails for any reason, return initial configuration
        return points


# EVOLVE-BLOCK-END
