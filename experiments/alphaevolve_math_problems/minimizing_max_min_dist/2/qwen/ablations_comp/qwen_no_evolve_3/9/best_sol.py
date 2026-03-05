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
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        # Normalize points to [0,1] x [0,1] 
        points = np.array(points)
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0
            
        return min_dist / max_dist
    
    def objective_function(x_flat):
        """Objective function to maximize (negative because we minimize)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute ratio
        ratio = compute_min_max_ratio(points)
        
        # Return negative because we want to maximize
        return -ratio
    
    def constraint_func(x_flat):
        """Constraint to keep points within [0,1] x [0,1]"""
        points = x_flat.reshape(-1, 2)
        # Return values that should be >= 0 for constraints to be satisfied
        # We want 0 <= x <= 1 and 0 <= y <= 1
        return np.concatenate([
            points[:, 0],  # x coordinates
            points[:, 1],  # y coordinates
            1 - points[:, 0],  # 1 - x coordinates
            1 - points[:, 1]   # 1 - y coordinates
        ])
    
    # Initialize with a good starting configuration
    # Create a hexagonal-like pattern for better initial distribution
    n = 16
    points = []
    
    # Generate points in a grid-like pattern with some perturbation
    rows = 4
    cols = 4
    spacing_x = 1.0 / (cols - 1)
    spacing_y = 1.0 / (rows - 1)
    
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                x = j * spacing_x + (np.random.random() - 0.5) * 0.1
                y = i * spacing_y + (np.random.random() - 0.5) * 0.1
                points.append([x, y])
    
    # Adjust to exactly 16 points
    points = np.array(points[:n])
    
    # Flatten for optimization
    x_init = points.flatten()
    
    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # Add constraints for bounds
    constraints = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Use scipy optimization with multiple restarts for better results
    best_ratio = -float('inf')
    best_points = None
    
    # Try multiple random restarts
    for _ in range(10):
        # Randomize initial points slightly
        x_restart = x_init + (np.random.random(len(x_init)) - 0.5) * 0.05
        x_restart = np.clip(x_restart, 0, 1)
        
        try:
            # Minimize the negative ratio (i.e., maximize the ratio)
            result = minimize(
                objective_function,
                x_restart,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
                    
        except Exception:
            continue
    
    # If no good solution found, use the simple hexagonal pattern
    if best_points is None:
        # Create a hexagonal arrangement
        angles = np.linspace(0, 2*np.pi, 17)[:-1]  # 16 angles
        radii = np.linspace(0.1, 0.4, 16)
        points = np.zeros((16, 2))
        for i in range(16):
            points[i] = [0.5 + radii[i]*np.cos(angles[i]), 0.5 + radii[i]*np.sin(angles[i])]
        best_points = points
    
    # Ensure all points are within [0,1] x [0,1]
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
