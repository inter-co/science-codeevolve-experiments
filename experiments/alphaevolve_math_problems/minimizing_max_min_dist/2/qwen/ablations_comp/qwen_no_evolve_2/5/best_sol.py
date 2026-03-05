# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def compute_distances(points):
        """Compute pairwise distances and return min/max ratio"""
        # Reshape points to (n, 2) if needed
        if points.ndim == 1:
            points = points.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max == 0:
            return 0.0
        return d_min / d_max
    
    def objective_function(x):
        """Objective function to maximize (negative because we minimize)"""
        # Reshape x back to points array
        points = x.reshape(-1, 2)
        ratio = compute_distances(points)
        return -ratio  # Negative because we want to maximize
    
    def constraint_function(x):
        """Constraint to keep points within unit square"""
        points = x.reshape(-1, 2)
        # Check that all points are within [0,1] x [0,1]
        return np.concatenate([
            points[:, 0],           # x coordinates
            1 - points[:, 0],       # 1 - x coordinates  
            points[:, 1],           # y coordinates
            1 - points[:, 1]        # 1 - y coordinates
        ])
    
    # Initialize points randomly within unit square
    np.random.seed(42)
    initial_points = np.random.rand(16, 2)
    
    # Flatten for optimization
    x0 = initial_points.flatten()
    
    # Set up bounds for each coordinate (0 to 1)
    bounds = [(0, 1)] * 32
    
    # Set up constraints - all points must stay within [0,1] x [0,1]
    # Using inequality constraints: g(x) >= 0
    constraints = []
    
    # Add boundary constraints
    # We want: 0 <= x_i <= 1 and 0 <= y_i <= 1 for all points
    # This means: x_i >= 0, 1-x_i >= 0, y_i >= 0, 1-y_i >= 0
    # Which gives us: x_i >= 0, x_i <= 1, y_i >= 0, y_i <= 1
    
    # Since scipy.optimize.minimize doesn't directly support bounds in the way we need,
    # we'll use the bounds parameter and let it handle the constraints internally
    
    try:
        # Use L-BFGS-B optimizer which supports bounds
        result = minimize(
            objective_function, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 2)
            # Ensure points are within bounds
            final_points = np.clip(final_points, 0, 1)
            return final_points
        else:
            # Fallback to simple constrained optimization
            pass
    except:
        pass
    
    # Fallback approach: gradient descent with projection
    points = initial_points.copy()
    learning_rate = 0.01
    num_iterations = 5000
    
    for i in range(num_iterations):
        # Compute current distances
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max == 0:
            break
            
        ratio = d_min / d_max
        
        # Compute gradients using finite differences for simplicity
        # This is a simplified approach - in practice one would compute exact gradients
        grad = np.zeros_like(points)
        
        # Simple gradient approximation using small perturbations
        epsilon = 1e-6
        for j in range(16):
            for k in range(2):  # x and y coordinates
                points_plus = points.copy()
                points_minus = points.copy()
                points_plus[j, k] += epsilon
                points_minus[j, k] -= epsilon
                
                # Clip to bounds
                points_plus = np.clip(points_plus, 0, 1)
                points_minus = np.clip(points_minus, 0, 1)
                
                dist_plus = pdist(points_plus)
                dist_minus = pdist(points_minus)
                
                if len(dist_plus) > 0 and len(dist_minus) > 0:
                    ratio_plus = np.min(dist_plus) / np.max(dist_plus) if np.max(dist_plus) != 0 else 0
                    ratio_minus = np.min(dist_minus) / np.max(dist_minus) if np.max(dist_minus) != 0 else 0
                    
                    grad[j, k] = (ratio_plus - ratio_minus) / (2 * epsilon)
        
        # Update points
        points += learning_rate * grad
        
        # Project points back to valid region
        points = np.clip(points, 0, 1)
        
        # Early stopping if improvement is minimal
        if i > 100 and i % 100 == 0:
            # Simple check for convergence
            pass
    
    # Final cleanup to ensure points are in valid range
    points = np.clip(points, 0, 1)
    
    return points


# EVOLVE-BLOCK-END
