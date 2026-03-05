# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining Fibonacci spiral, icosahedral symmetry, and robust optimization.

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
    
    # Initialize using Fibonacci spiral for good distribution
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
    
    # Convert to unit cube [0,1]^3
    points = (points + 1) / 2
    
    # Enhance with icosahedral symmetry for better distribution
    # Get icosahedral vertices
    phi_ico = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi_ico], [0, -1, phi_ico], [0, 1, -phi_ico], [0, -1, -phi_ico],
        [1, phi_ico, 0], [-1, phi_ico, 0], [1, -phi_ico, 0], [-1, -phi_ico, 0],
        [phi_ico, 0, 1], [phi_ico, 0, -1], [-phi_ico, 0, 1], [-phi_ico, 0, -1]
    ])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    ico_vertices = ico_vertices / norms
    
    # Convert to unit cube [0,1]^3
    ico_vertices = (ico_vertices + 1) / 2
    
    # Mix with Fibonacci points for better distribution
    points[:12] = ico_vertices
    
    # Add two strategic points for 14 total
    additional_points = np.array([
        [0.5, 0.5, 0.5],  # center point
        [0.75, 0.25, 0.25]  # strategic point
    ])
    
    points[12:] = additional_points
    
    # Add small random perturbations to break perfect symmetry
    points += np.random.normal(0, 0.003, points.shape)
    
    # Ensure points are within [0,1]^3
    points = np.clip(points, 0, 1)
    
    # Try single optimization approach for efficiency
    x0 = points.flatten()
    
    # Use SLSQP optimization with moderate precision for faster convergence
    try:
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=[(0, 1) for _ in range(42)], 
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 3)
            # Validate the improvement with reasonable checks
            distances = pdist(optimized_points)
            if len(distances) > 0:
                dmin = np.min(distances)
                dmax = np.max(distances)
                if dmax > 0 and dmin / dmax > 0.1:  # More reasonable sanity check
                    points = optimized_points
                    
    except Exception:
        pass
    
    # Final boundary check
    points = np.clip(points, 0, 1)
    
    return points


# EVOLVE-BLOCK-END
