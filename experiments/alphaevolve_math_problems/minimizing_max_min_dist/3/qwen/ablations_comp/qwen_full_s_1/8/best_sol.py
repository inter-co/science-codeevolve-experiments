# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining constructive geometry, symmetry exploitation, and global optimization.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        """Objective function to minimize negative of min/max ratio"""
        # Reshape flat array back to 14x3 points
        points = x.reshape(-1, 3)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return -1e10
            
        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        return -min_dist / max_dist
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio"""
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0
    
    def get_icosahedral_config():
        """Generate points based on icosahedron vertices plus two additional points"""
        # Regular icosahedron vertices (normalized to unit sphere)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # Add two more points that are symmetrically placed
        # These will be along the principal axes
        extra_points = np.array([[0, 0, 1], [0, 0, -1]])
        
        # Combine and normalize to appropriate scale
        all_points = np.vstack([vertices, extra_points])
        all_points = all_points / np.linalg.norm(all_points[0]) * 0.9
        
        # Randomly perturb slightly to break degeneracies
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, all_points.shape)
        all_points = all_points + noise
        
        return all_points
    
    def get_spherical_code_initial():
        """Generate initial configuration using spherical code approach"""
        # Use a more sophisticated spherical code approach
        n = 14
        points = np.zeros((n, 3))
        
        # Generate points using a variant of the Fibonacci spiral but with better distribution
        for i in range(n):
            # Improved spiral placement with better coverage
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            
            theta = np.arccos(y)
            phi = np.sqrt(n * np.pi) * theta  # More uniform than golden ratio
            
            points[i, 0] = radius * np.cos(phi)
            points[i, 1] = radius * np.sin(phi)
            points[i, 2] = y
        
        # Normalize to unit sphere
        points = points / np.linalg.norm(points[0]) * 0.85
        return points
    
    def get_symmetric_initial():
        """Generate symmetric initial configuration"""
        # Start with icosahedral structure
        base_points = get_icosahedral_config()
        
        # Apply random rotation to break symmetry for optimization
        np.random.seed(123)
        rotation = R.from_euler('xyz', np.random.rand(3) * 2 * np.pi).as_matrix()
        rotated_points = base_points @ rotation.T
        
        # Add some randomness to avoid local minima
        noise = np.random.normal(0, 0.02, rotated_points.shape)
        final_points = rotated_points + noise
        
        return final_points
    
    def get_fibonacci_spiral():
        """Generate points using Fibonacci spiral on sphere"""
        n = 14
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def optimize_with_multiple_methods(start_points, max_iter=500):
        """Try multiple optimization methods to find best solution"""
        best_points = start_points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        # Try different optimization methods
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        
        for method in methods:
            try:
                x0 = start_points.flatten()
                bounds = [(-1, 1) for _ in range(42)]
                
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    final_points = np.clip(final_points, -1, 1)
                    ratio = compute_min_max_ratio(final_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
            except Exception:
                continue
                
        return best_points
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Icosahedral-based configuration
    initial_configs.append(get_icosahedral_config())
    
    # 2. Spherical code configuration  
    initial_configs.append(get_spherical_code_initial())
    
    # 3. Symmetric configuration
    initial_configs.append(get_symmetric_initial())
    
    # 4. Fibonacci spiral configuration
    initial_configs.append(get_fibonacci_spiral())
    
    # 5. Random configuration for diversity
    np.random.seed(456)
    random_points = np.random.uniform(-0.9, 0.9, (14, 3))
    initial_configs.append(random_points)
    
    # 6. Another variation with different seed
    np.random.seed(789)
    random_points2 = np.random.uniform(-0.9, 0.9, (14, 3))
    initial_configs.append(random_points2)
    
    # 7. Perturbed icosahedral
    ico_points = get_icosahedral_config()
    np.random.seed(999)
    ico_points += np.random.normal(0, 0.03, ico_points.shape)
    initial_configs.append(ico_points)
    
    # Find the best initial configuration
    best_initial_ratio = -1
    best_initial_points = None
    
    for i, config in enumerate(initial_configs):
        ratio = compute_min_max_ratio(config)
        if ratio > best_initial_ratio:
            best_initial_ratio = ratio
            best_initial_points = config.copy()
    
    # Optimize the best initial configuration with multiple methods
    final_points = optimize_with_multiple_methods(best_initial_points, max_iter=1000)
    
    # Try a few more optimizations with different starting points
    for i in range(3):  # Try 3 more optimizations
        try:
            # Start from random points
            np.random.seed(1000 + i)
            random_start = np.random.uniform(-0.9, 0.9, (14, 3))
            
            # Optimize this random start
            optimized = optimize_with_multiple_methods(random_start, max_iter=500)
            ratio = compute_min_max_ratio(optimized)
            
            if ratio > compute_min_max_ratio(final_points):
                final_points = optimized.copy()
        except Exception:
            continue
    
    # Final refinement with higher precision
    try:
        final_points = optimize_with_multiple_methods(final_points, max_iter=1000)
    except Exception:
        pass
    
    return final_points


# EVOLVE-BLOCK-END
