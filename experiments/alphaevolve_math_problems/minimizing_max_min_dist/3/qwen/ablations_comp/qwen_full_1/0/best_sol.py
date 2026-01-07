# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and robust multi-start optimization.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_ratio(points):
        """Compute min/max distance ratio for given points."""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        dmin = np.min(distances)
        dmax = np.max(distances)
        return dmin / dmax if dmax > 0 else 0
    
    def objective(x_flat):
        """Minimize negative of min/max ratio (equivalent to maximizing min/max ratio)"""
        points = x_flat.reshape(-1, 3)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0  # Avoid division by zero
            
        return -min_dist / max_dist
    
    def normalize_to_sphere(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    # Strategy: Use a refined approach similar to INSPIRATION PROGRAM 2
    try:
        # Generate icosahedral points (12 vertices) with correct normalization
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = []
        
        # Properly generate icosahedral vertices - this matches INSPIRATION 2 more closely
        for i in range(12):
            if i < 4:
                x, y, z = [(-1)**i, 0, phi]
            elif i < 8:
                x, y, z = [0, (-1)**(i-4) * phi, (-1)**(i-4)]
            else:
                x, y, z = [(-1)**(i-8) * phi, (-1)**(i-8), 0]
            norm = np.sqrt(x*x + y*y + z*z)
            vertices.append([x/norm, y/norm, z/norm])
        
        ico_points = np.array(vertices)
        
        # Add 2 more points to make 14 total - north and south poles (closer to unit sphere)
        additional_points = np.array([[0, 0, 0.99], [0, 0, -0.99]])
        initial_points = np.vstack([ico_points, additional_points])
        
        # Normalize to unit sphere
        initial_points = normalize_to_sphere(initial_points)
        best_ratio = compute_ratio(initial_points)
        best_points = initial_points.copy()
        
    except Exception:
        # Fallback to fibonacci spiral - use the precise method from INSPIRATION 2
        try:
            points = []
            golden_ratio = (1 + np.sqrt(5)) / 2
            
            for i in range(14):
                y = 1 - (i / 13) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y
                theta = np.arccos(y)  # angle from z-axis (this is the key difference from my previous approach)
                phi_angle = (i * golden_ratio) % (2 * np.pi)  # azimuthal angle
                x = radius * np.cos(phi_angle)
                z = radius * np.sin(phi_angle)
                points.append([x, y, z])
            
            initial_points = np.array(points)
            initial_points = normalize_to_sphere(initial_points)
            best_ratio = compute_ratio(initial_points)
            best_points = initial_points.copy()
        except Exception:
            # Last resort: random points
            np.random.seed(42)
            points = np.random.randn(14, 3)
            points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
            return points
    
    # Multiple refinement passes with different strategies - optimized from INSPIRATION 2
    # Strategy 1: Local optimization with multiple restarts
    for seed in [123, 456, 789]:
        try:
            np.random.seed(seed)
            # Perturb with small noise
            perturbation = np.random.normal(0, 0.02, best_points.shape)
            perturbed_points = best_points + perturbation
            perturbed_points = normalize_to_sphere(perturbed_points)
            
            result = minimize(
                objective,
                perturbed_points.flatten(),
                method='SLSQP',
                options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-11},
                tol=1e-11
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                ratio = compute_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            continue
    
    # Strategy 2: Final high-precision optimization
    try:
        result = minimize(
            objective,
            best_points.flatten(),
            method='SLSQP',
            options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-12},
            tol=1e-12
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 3)
            final_points = normalize_to_sphere(final_points)
            ratio = compute_ratio(final_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = final_points.copy()
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
