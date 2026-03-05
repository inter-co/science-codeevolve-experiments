# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import dual_annealing, minimize
from scipy.spatial import ConvexHull
from numba import jit
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_distances_numba(points):
    """Compute pairwise distances using numba for speed"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric insights with advanced optimization techniques.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Start with a better initial configuration based on known good distributions
    # Use a configuration inspired by the regular 16-point arrangement with slight perturbations
    np.random.seed(42)
    
    # Create a more sophisticated initial configuration
    # Start with a regular grid and add strategic perturbations
    points = []
    for i in range(4):
        for j in range(4):
            # Add more systematic jitter to create better spread
            x = (j + 0.5) * (1.0 / 4.0) + np.random.normal(0, 0.03)
            y = (i + 0.5) * (1.0 / 4.0) + np.random.normal(0, 0.03)
            points.append([x, y])
    
    # Ensure we have exactly 16 points and clip to bounds
    points = np.array(points[:16])
    points = np.clip(points, 0, 1)
    
    # Optimization objective function
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        # Use faster numba version when possible
        try:
            distances = compute_distances_numba(points)
        except:
            distances = squareform(pdist(points))
        
        # Avoid division by zero
        mask = distances > 0
        if not np.any(mask):
            return 1e10  # Very bad score if no valid distances
            
        min_dist = np.min(distances[mask])
        max_dist = np.max(distances)
        
        # Handle edge case where max_dist is very small
        if max_dist < 1e-12:
            return 1e10
            
        # Use log scaling to avoid numerical issues with very small ratios
        ratio = min_dist / max_dist
        return -ratio  # Negative because we want to maximize
    
    # Define bounds for all coordinates (0 to 1)
    bounds = [(0, 1)] * 32  # 16 points * 2 coordinates each
    
    # Use a more robust optimization approach with better parameters
    # Simplified approach: use only dual annealing with more iterations and better settings
    result = dual_annealing(
        objective, 
        bounds, 
        maxiter=2000,  # More iterations for better optimization
        initial_temp=2000,
        restart_temp_ratio=1e-6,
        visit=2.62,
        accept=-5.0,
        seed=42,
        no_local_search=False
    )
    
    # Extract the best solution found
    optimized_points = result.x.reshape(-1, 2)
    
    # Ensure all points are within bounds
    optimized_points = np.clip(optimized_points, 0, 1)
    
    # Local refinement with L-BFGS-B - more focused approach
    try:
        x0 = optimized_points.flatten()
        local_result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * len(x0),
            options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if local_result.success:
            refined_points = local_result.x.reshape(-1, 2)
            refined_points = np.clip(refined_points, 0, 1)
            # Check if this improves our result
            original_score = -objective(optimized_points.flatten())
            refined_score = -objective(refined_points.flatten())
            
            if refined_score > original_score:
                optimized_points = refined_points
                
    except Exception:
        pass
    
    # Final validation and cleanup
    final_points = optimized_points.copy()
    
    # Remove any duplicate or nearly identical points
    unique_points = []
    for i, point in enumerate(final_points):
        is_unique = True
        for existing_point in unique_points:
            if np.linalg.norm(point - existing_point) < 1e-8:
                is_unique = False
                break
        if is_unique:
            unique_points.append(point)
    
    # If we lost points, regenerate them
    if len(unique_points) < 16:
        # Reinitialize with a better spread
        unique_points = []
        # Create a more evenly distributed set
        for i in range(4):
            for j in range(4):
                x = (j + 0.5) * (1.0 / 4.0) + np.random.normal(0, 0.01)
                y = (i + 0.5) * (1.0 / 4.0) + np.random.normal(0, 0.01)
                unique_points.append([x, y])
        unique_points = np.clip(unique_points, 0, 1)[:16]
    
    # Final cleanup to ensure points are well-separated
    final_points = np.array(unique_points)
    
    # Ensure we have exactly 16 points
    if len(final_points) < 16:
        # Fill missing points with random positions
        remaining = 16 - len(final_points)
        for _ in range(remaining):
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            final_points = np.vstack([final_points, [x, y]])
    
    return final_points[:16]


# EVOLVE-BLOCK-END
