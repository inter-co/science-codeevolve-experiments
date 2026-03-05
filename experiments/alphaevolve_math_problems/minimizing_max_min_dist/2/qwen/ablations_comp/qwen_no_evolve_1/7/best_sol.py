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
    
    def objective(x):
        # Reshape x into points
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Minimize negative of ratio (equivalent to maximizing ratio)
        # Use a smooth approximation to avoid numerical issues
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0
            
        ratio = min_dist / max_dist
        
        # Return negative because we're minimizing
        return -ratio
    
    def constraint_func(x):
        # Ensure points stay within [0,1] x [0,1]
        points = x.reshape(-1, 2)
        # Return positive values when constraints are satisfied
        return np.concatenate([
            points[:, 0],                    # x >= 0
            1 - points[:, 0],                # x <= 1
            points[:, 1],                    # y >= 0
            1 - points[:, 1]                 # y <= 1
        ])
    
    # Start with a good initial configuration
    np.random.seed(42)
    
    # Generate initial points using a more strategic approach
    # Place points in a grid pattern with some randomness
    initial_points = []
    for i in range(4):
        for j in range(4):
            # Add small random perturbation around grid points
            x = i/3.0 + (np.random.random()-0.5)*0.1
            y = j/3.0 + (np.random.random()-0.5)*0.1
            # Keep within bounds
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            initial_points.append([x, y])
    
    initial_points = np.array(initial_points).flatten()
    
    # Define constraints
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Optimize
    result = minimize(
        objective,
        initial_points,
        method='SLSQP',
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6}
    )
    
    # Extract final points
    final_points = result.x.reshape(-1, 2)
    
    return final_points


# EVOLVE-BLOCK-END
