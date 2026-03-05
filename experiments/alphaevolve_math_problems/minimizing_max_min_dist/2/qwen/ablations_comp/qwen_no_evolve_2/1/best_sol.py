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
        # Reshape flat array back to 16x2 points
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio to maximize (since we're minimizing)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    def constraint(x):
        # Ensure all points are within [0,1] x [0,1]
        points = x.reshape(-1, 2)
        # Return positive values where constraints are violated
        violations = np.concatenate([
            np.minimum(points[:, 0], 0),      # x < 0
            np.minimum(points[:, 1], 0),      # y < 0
            np.minimum(1 - points[:, 0], 0),  # x > 1
            np.minimum(1 - points[:, 1], 0)   # y > 1
        ])
        return violations
    
    # Start with a good initial configuration (hexagonal packing pattern)
    np.random.seed(42)
    
    # Create a more structured initial guess
    # Arrange points in a grid-like pattern with some perturbation
    grid_size = 4  # 4x4 grid
    points_init = np.array([[i/grid_size, j/grid_size] for i in range(grid_size) for j in range(grid_size)])
    
    # Add small random perturbations
    points_init += np.random.normal(0, 0.05, points_init.shape)
    
    # Clip to valid range [0,1]
    points_init = np.clip(points_init, 0, 1)
    
    # Flatten for optimization
    x0 = points_init.flatten()
    
    # Define bounds for each coordinate (0 to 1)
    bounds = [(0, 1) for _ in range(32)]
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': constraint},
        options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
    )
    
    # Extract final points
    final_points = result.x.reshape(-1, 2)
    
    # Ensure all points are within bounds
    final_points = np.clip(final_points, 0, 1)
    
    return final_points


# EVOLVE-BLOCK-END
