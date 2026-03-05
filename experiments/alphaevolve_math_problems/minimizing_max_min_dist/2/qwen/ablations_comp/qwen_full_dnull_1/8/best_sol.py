# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust multi-start optimization approach with geometric initialization and 
    multiple optimization strategies.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        # Ensure points are within bounds
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def objective_function(x_flat):
        """Objective function to maximize (negative because we minimize)."""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute ratio (we want to maximize this)
        ratio = compute_min_max_ratio(points)
        
        # Return negative because we're minimizing in scipy.optimize
        return -ratio
    
    def initialize_points_hexagonal():
        """Initialize points using a hexagonal lattice arrangement."""
        # Create a hexagonal pattern that fits well in [0,1] x [0,1]
        points = []
        rows, cols = 4, 4
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Hexagonal offset for alternate rows
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + offset) / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                
                # Scale to avoid boundary issues and add some randomness
                x = 0.05 + x * 0.9  # Scale to [0.05, 0.95]
                y = 0.05 + y * 0.9
                
                # Add controlled noise to break symmetry
                noise_scale = 0.015
                x += np.random.normal(0, noise_scale)
                y += np.random.normal(0, noise_scale)
                
                points.append([x, y])
        
        points_array = np.array(points[:16])  # Ensure exactly 16 points
        
        # Ensure all points are within bounds
        points_array[:, 0] = np.clip(points_array[:, 0], 0, 1)
        points_array[:, 1] = np.clip(points_array[:, 1], 0, 1)
        
        return points_array
    
    def initialize_points_grid():
        """Initialize points using a 4x4 grid pattern."""
        points = []
        spacing_x = 1.0 / 3.0
        spacing_y = 1.0 / 3.0
        
        for i in range(4):
            for j in range(4):
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Add controlled perturbation to break symmetry
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                
                points.append([x, y])
        
        points_array = np.array(points)
        
        # Ensure all points are within bounds
        points_array[:, 0] = np.clip(points_array[:, 0], 0, 1)
        points_array[:, 1] = np.clip(points_array[:, 1], 0, 1)
        
        return points_array
    
    def initialize_points_random():
        """Simple random initialization."""
        return np.random.rand(16, 2)
    
    # Try multiple initialization strategies
    initializations = [
        initialize_points_hexagonal(),
        initialize_points_grid(), 
        initialize_points_random()
    ]
    
    # Find the best initial configuration
    best_initial = initializations[0]
    best_initial_ratio = compute_min_max_ratio(best_initial)
    
    for init in initializations[1:]:
        ratio = compute_min_max_ratio(init)
        if ratio > best_initial_ratio:
            best_initial_ratio = ratio
            best_initial = init
    
    # Multi-start optimization with multiple restarts
    best_points = best_initial.copy()
    best_ratio = best_initial_ratio
    
    # Run multiple optimizations with different random seeds
    for restart in range(8):  # Increased restarts for better exploration (like INSPIRATION 2)
        # Create slightly different starting point for each restart
        np.random.seed(42 + restart)
        start_points = best_initial.copy()
        # Add moderate noise to explore the space
        noise = np.random.normal(0, 0.02, start_points.shape)
        start_points += noise
        start_points = np.clip(start_points, 0, 1)
        
        # Flatten for optimization
        x0 = start_points.flatten()
        
        # Define bounds (each coordinate must be in [0,1])
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            # Try SLSQP first (often works well for this type of problem)
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-11}  # Tighter tolerances
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            # Continue with next restart if this one fails
            continue
    
    # Final refinement with L-BFGS-B
    try:
        x0 = best_points.flatten()
        bounds = [(0, 1) for _ in range(32)]
        
        result = minimize(
            objective_function,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 2)
            refined_points = np.clip(refined_points, 0, 1)
            ratio = compute_min_max_ratio(refined_points)
            if ratio > best_ratio:
                best_points = refined_points
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
