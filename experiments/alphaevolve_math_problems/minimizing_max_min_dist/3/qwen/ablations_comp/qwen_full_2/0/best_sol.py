# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses Fibonacci spiral initialization with icosahedral refinement and robust optimization.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        """Objective function: minimize negative of min/max ratio"""
        # Reshape flat array back to 3D points
        points = x.reshape(-1, 3)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio (since we want to maximize ratio)
        if max_dist == 0:
            return 0
        return -min_dist / max_dist
    
    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Compute the minimum and maximum distances between all point pairs."""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        dmin = np.min(distances)
        dmax = np.max(distances)
        return dmin / dmax if dmax > 0 else 0

    # Strategy 1: Fibonacci spiral initialization for good distribution
    np.random.seed(42)
    n = 14
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    
    for i in range(n):
        theta = np.arccos(1 - 2 * (i / (n - 1)))
        phi = ((i + 1) * golden_ratio) % n
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Scale to fit nicely in unit sphere
    max_norm = np.max(np.linalg.norm(points, axis=1))
    if max_norm > 0:
        points = points / max_norm * 0.9
    
    # Project to unit cube [0,1]^3
    points = (points + 1) / 2
    
    # Strategy 2: Enhance with icosahedral symmetry
    # Get icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    ico_vertices = ico_vertices / norms
    
    # Convert to unit cube [0,1]^3
    ico_vertices = (ico_vertices + 1) / 2
    
    # Mix with Fibonacci points for better distribution
    # Replace some points with icosahedral ones for symmetry
    points[:12] = ico_vertices
    
    # Add two strategic points
    additional_points = np.array([
        [0.5, 0.5, 0.5],  # center point
        [0.75, 0.25, 0.25]  # strategic point
    ])
    
    points[12:] = additional_points
    
    # Add small random perturbations to break perfect symmetry
    points += np.random.normal(0, 0.003, points.shape)
    
    # Ensure points are within [0,1]^3
    points = np.clip(points, 0, 1)
    
    # Strategy 3: Multiple optimization approaches for robustness
    best_points = points.copy()
    best_ratio = compute_min_max_ratio(best_points)
    
    # Try SLSQP optimization first (most robust for this type of problem)
    try:
        x0 = best_points.flatten()
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=[(0, 1) for _ in range(42)], 
            options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-14},
            tol=1e-14
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 3)
            ratio = compute_min_max_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
    except Exception:
        pass
    
    # Try L-BFGS-B as fallback
    try:
        x0 = best_points.flatten()
        result = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=[(0, 1) for _ in range(42)], 
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 3)
            ratio = compute_min_max_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
                
    except Exception:
        pass
    
    # Final boundary check
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
