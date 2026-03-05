# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def objective(x_flat):
        """Minimize negative of min/max distance ratio (i.e., maximize the ratio)"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return float('inf')
            
        # Return negative ratio (since we're minimizing)
        return -min_dist / max_dist
    
    def constraint_func(x_flat):
        """Constraint: all points must be within [0,1] x [0,1]"""
        points = x_flat.reshape(-1, 2)
        # Check bounds: each coordinate should be between 0 and 1
        return np.concatenate([points.flatten() - 1, -points.flatten()])
    
    n = 16
    d = 2
    
    # Generate multiple starting configurations
    best_ratio = -float('inf')
    best_points = None
    
    # Try multiple random starting points plus structured ones
    for seed in [42, 123, 456, 789, 987]:
        np.random.seed(seed)
        
        # Initialize with a structured configuration (hexagonal-like pattern)
        # Start with a regular grid pattern and add some randomness
        initial_points = np.zeros((n, d))
        
        # Create a hexagonal-like arrangement
        rows = 4
        cols = 4
        row_spacing = 1.0 / (rows - 1)
        col_spacing = 1.0 / (cols - 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < n:
                    # Add slight perturbation to create more diverse configuration
                    y = i * row_spacing + (np.random.rand() - 0.5) * 0.1
                    x = j * col_spacing + (np.random.rand() - 0.5) * 0.1
                    # Ensure points stay within bounds
                    x = np.clip(x, 0, 1)
                    y = np.clip(y, 0, 1)
                    initial_points[idx] = [x, y]
                    idx += 1
        
        # Flatten for optimization
        initial_flat = initial_points.flatten()
        
        # Define constraints
        cons = [{'type': 'ineq', 'fun': lambda x: 1 - x[::2]},  # x <= 1
                {'type': 'ineq', 'fun': lambda x: x[::2]},      # x >= 0
                {'type': 'ineq', 'fun': lambda x: 1 - x[1::2]}, # y <= 1
                {'type': 'ineq', 'fun': lambda x: x[1::2]}]     # y >= 0
        
        # Optimize
        try:
            result = minimize(objective, initial_flat, method='SLSQP', 
                            constraints=cons, options={'maxiter': 1000, 'ftol': 1e-8})
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                distances = pdist(final_points)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
        except:
            continue
    
    # If no good solution found, fall back to a good structured arrangement
    if best_points is None:
        # Use a more systematic approach with better distribution
        np.random.seed(42)
        # Create a grid with small perturbations
        points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.333 + (np.random.rand() - 0.5) * 0.1
                y = j * 0.333 + (np.random.rand() - 0.5) * 0.1
                # Clamp to [0,1]
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
        best_points = np.array(points[:16])  # Ensure exactly 16 points
    
    return best_points


# EVOLVE-BLOCK-END
