# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Enhanced initialization using a combination of strategies
    def initialize_enhanced_placement():
        # Strategy 1: Start with a grid pattern for good coverage
        grid_size = int(np.ceil(np.sqrt(n)))
        x_coords = np.linspace(0.05, 0.95, grid_size)
        y_coords = np.linspace(0.05, 0.95, grid_size)
        X, Y = np.meshgrid(x_coords, y_coords)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        
        # Strategy 2: Use k-means to select exactly n points from the grid
        # This ensures good distribution without going too far from grid
        if len(grid_points) >= n:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=20)
            kmeans.fit(grid_points)
            initial_positions = kmeans.cluster_centers_
        else:
            # If grid is smaller than n, fill remaining positions strategically
            initial_positions = grid_points.copy()
            # Fill remaining positions with random points near existing ones
            while len(initial_positions) < n:
                # Pick a random existing point and add nearby points
                idx = np.random.randint(len(initial_positions))
                base_point = initial_positions[idx]
                new_point = base_point + np.random.normal(0, 0.05, 2)
                # Keep within bounds
                new_point[0] = np.clip(new_point[0], 0.05, 0.95)
                new_point[1] = np.clip(new_point[1], 0.05, 0.95)
                initial_positions = np.vstack([initial_positions, new_point])
            
            initial_positions = initial_positions[:n]
        
        return initial_positions
    
    # Initialize with enhanced placement
    initial_positions = initialize_enhanced_placement()
    
    # Better initial radius estimation
    # Estimate radii based on local density around each point
    initial_radii = np.full(n, 0.05)
    
    # Improve initial radii estimation using spatial proximity
    tree = cKDTree(initial_positions)
    for i in range(n):
        # Find nearest neighbors (excluding self)
        distances, indices = tree.query(initial_positions[i], k=min(6, n), p=2)
        # Use the minimum distance to set radius (but cap it)
        if len(distances) > 1:  # Exclude self-distance
            min_dist = np.min(distances[1:])  # Get the closest neighbor
            initial_radii[i] = min(min_dist/2.0, 0.25)
    
    # Flatten initial parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = np.concatenate([initial_positions.flatten(), initial_radii])
    
    # Define constraint functions with better numerical stability
    def containment_constraints(params):
        """Ensure all circles are within the unit square"""
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Each circle must be fully contained
        constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Left boundary (should be >= 0)
            constraints.append(x - r)
            # Right boundary (should be >= 0)  
            constraints.append(1 - x - r)
            # Bottom boundary (should be >= 0)
            constraints.append(y - r)
            # Top boundary (should be >= 0)
            constraints.append(1 - y - r)
            
        return np.array(constraints)
    
    def non_overlap_constraints(params):
        """Ensure no overlap between circles with numerical tolerance"""
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1 = radii[i]
                r2 = radii[j]
                
                # Distance between centers minus sum of radii should be >= 0
                # Add small epsilon to avoid numerical issues
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                dist = np.sqrt(dist_sq + 1e-12)  # Small epsilon to prevent sqrt(0)
                constraints.append(dist - (r1 + r2))
                
        return np.array(constraints)
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        radii = params[2*n:]
        return -np.sum(radii)
    
    # Create bounds for parameters
    bounds = []
    # Position bounds [0.05, 0.95] to ensure good margin
    for i in range(2*n):
        bounds.append((0.05, 0.95))
    # Radius bounds [0.01, 0.3] to allow reasonable variation
    for i in range(n):
        bounds.append((0.01, 0.3))
    
    # Define constraint dictionaries
    containment_cons = {
        'type': 'ineq',
        'fun': lambda p: containment_constraints(p)
    }
    
    non_overlap_cons = {
        'type': 'ineq', 
        'fun': lambda p: non_overlap_constraints(p)
    }
    
    # Try multiple optimization approaches with different settings
    best_result = None
    best_sum = 0
    
    # Try with different optimization methods and parameters
    optimizations = [
        ('SLSQP', {'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-6}),
        ('trust-constr', {'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6})
    ]
    
    for method, options in optimizations:
        try:
            result = minimize(
                objective,
                initial_params,
                method=method,
                bounds=bounds,
                constraints=[containment_cons, non_overlap_cons],
                options=options
            )
            
            if result.success:
                final_radii = result.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception:
            continue
    
    # If we found a good result, use it; otherwise fall back to initial
    if best_result is not None and best_result.success:
        final_positions = best_result.x[:2*n].reshape(-1, 2)
        final_radii = best_result.x[2*n:]
        
        # Create output array
        circles = np.column_stack([final_positions, final_radii])
        return circles
    else:
        # Fallback to initial configuration with improved radii
        circles = np.column_stack([initial_positions, initial_radii])
        return circles


# EVOLVE-BLOCK-END
