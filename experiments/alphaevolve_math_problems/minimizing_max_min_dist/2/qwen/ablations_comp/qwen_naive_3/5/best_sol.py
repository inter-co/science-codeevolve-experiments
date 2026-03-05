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
    
    # Start with a more sophisticated initial configuration
    # Use a combination of hexagonal packing and perturbations
    np.random.seed(42)
    
    # Create points in a hexagonal pattern for better initial distribution
    points = []
    rows = 4
    cols = 4
    
    # Hexagonal offset pattern
    for i in range(rows):
        for j in range(cols):
            # Offset every other row
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + offset) / cols
            y = i / rows
            # Add slight randomization
            x += np.random.normal(0, 0.02)
            y += np.random.normal(0, 0.02)
            points.append([x, y])
    
    # Ensure we have exactly 16 points
    points = points[:16]
    
    # Convert to numpy array and clip to bounds
    points = np.array(points)
    points = np.clip(points, 0, 1)
    
    # Optimization objective function with better numerical handling
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        
        # Use faster numba version when possible
        try:
            distances = compute_distances_numba(points)
        except:
            distances = squareform(pdist(points))
        
        # Avoid division by zero - find actual non-zero distances
        mask = distances > 1e-12
        if not np.any(mask):
            return -1e10  # Very bad score if no valid distances
            
        # Get all non-zero distances
        non_zero_dists = distances[mask]
        if len(non_zero_dists) == 0:
            return -1e10
            
        min_dist = np.min(non_zero_dists)
        max_dist = np.max(distances)
        
        # Handle edge case where max_dist is very small
        if max_dist < 1e-12:
            return -1e10
            
        # Use a more stable ratio calculation
        if max_dist > 0:
            ratio = min_dist / max_dist
        else:
            ratio = 0
            
        return -ratio  # Negative because we want to maximize
    
    # Define bounds for all coordinates (0 to 1)
    bounds = [(0, 1)] * 32  # 16 points * 2 coordinates each
    
    # Use a more focused optimization approach with better parameters
    # Stage 1: Global optimization with adaptive parameters
    try:
        result = dual_annealing(
            objective, 
            bounds, 
            maxiter=1500,  # Reduced iterations for faster execution
            initial_temp=2000,
            restart_temp_ratio=1e-6,
            visit=2.62,
            accept=-5.0,
            seed=42,
            no_local_search=False
        )
    except:
        # Fallback to simpler optimization if dual_annealing fails
        result = None
    
    # If dual_annealing failed, use a simpler approach
    if result is None or not result.success:
        # Start with a simple grid pattern and refine
        points = []
        for i in range(4):
            for j in range(4):
                x = (j + 0.5) * (1.0 / 4.0)
                y = (i + 0.5) * (1.0 / 4.0)
                points.append([x, y])
        points = np.array(points)
        points = np.clip(points, 0, 1)
        optimized_points = points.copy()
    else:
        # Extract the best solution found
        optimized_points = result.x.reshape(-1, 2)
        optimized_points = np.clip(optimized_points, 0, 1)
    
    # Stage 2: Local refinement with better convergence criteria
    def improved_objective(x_flat):
        points = x_flat.reshape(-1, 2)
        try:
            distances = compute_distances_numba(points)
        except:
            distances = squareform(pdist(points))
        
        # Avoid division by zero
        mask = distances > 1e-12
        if not np.any(mask):
            return 1e10
            
        non_zero_dists = distances[mask]
        if len(non_zero_dists) == 0:
            return 1e10
            
        min_dist = np.min(non_zero_dists)
        max_dist = np.max(distances)
        
        # Handle edge case where max_dist is very small
        if max_dist < 1e-12:
            return 1e10
            
        if max_dist > 0:
            ratio = min_dist / max_dist
        else:
            ratio = 0
            
        return -ratio  # Negative because we want to maximize
    
    # Try multiple local optimization approaches with different methods
    best_points = optimized_points.copy()
    best_ratio = -objective(optimized_points.flatten())
    
    # Method 1: L-BFGS-B - more aggressive settings
    try:
        x0 = optimized_points.flatten()
        local_result = minimize(
            improved_objective,
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * len(x0),
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if local_result.success:
            refined_points = local_result.x.reshape(-1, 2)
            refined_points = np.clip(refined_points, 0, 1)
            refined_ratio = -improved_objective(refined_points.flatten())
            
            if refined_ratio > best_ratio:
                best_points = refined_points
                best_ratio = refined_ratio
                
    except Exception:
        pass
    
    # Method 2: SLSQP as additional refinement
    try:
        x0 = best_points.flatten()
        local_result = minimize(
            improved_objective,
            x0,
            method='SLSQP',
            bounds=[(0, 1)] * len(x0),
            options={'maxiter': 500, 'ftol': 1e-10}
        )
        
        if local_result.success:
            refined_points = local_result.x.reshape(-1, 2)
            refined_points = np.clip(refined_points, 0, 1)
            refined_ratio = -improved_objective(refined_points.flatten())
            
            if refined_ratio > best_ratio:
                best_points = refined_points
                best_ratio = refined_ratio
                
    except Exception:
        pass
    
    # Stage 3: Additional refinement using a simple gradient-based approach
    # If we're still not satisfied, do one more optimization step
    try:
        # Try with COBYLA for robustness
        x0 = best_points.flatten()
        local_result = minimize(
            improved_objective,
            x0,
            method='COBYLA',
            bounds=[(0, 1)] * len(x0),
            options={'maxiter': 300, 'rhobeg': 0.05}
        )
        
        if local_result.success:
            refined_points = local_result.x.reshape(-1, 2)
            refined_points = np.clip(refined_points, 0, 1)
            refined_ratio = -improved_objective(refined_points.flatten())
            
            if refined_ratio > best_ratio:
                best_points = refined_points
                best_ratio = refined_ratio
                
    except Exception:
        pass
    
    # Stage 4: Final validation and cleanup
    final_points = best_points.copy()
    
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
    
    # If we lost points, regenerate them with better spread
    if len(unique_points) < 16:
        # Use a better initialization strategy
        unique_points = []
        # Create a more evenly distributed set
        for i in range(4):
            for j in range(4):
                # Better spacing with less regularity
                x = (j + 0.5 + np.random.normal(0, 0.03)) * (1.0 / 4.0)
                y = (i + 0.5 + np.random.normal(0, 0.03)) * (1.0 / 4.0)
                unique_points.append([x, y])
        unique_points = np.clip(unique_points, 0, 1)[:16]
    
    return np.array(unique_points)


# EVOLVE-BLOCK-END
