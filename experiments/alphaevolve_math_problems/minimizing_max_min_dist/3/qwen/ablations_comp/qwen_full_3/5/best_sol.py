# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a geometric construction based on the icosahedron with optimization refinement
    and incorporates insights from mathematical optimization approaches.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all pairs"""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        if dmax <= 0:
            return 0.0
            
        return dmin / dmax
    
    # Define objective function: minimize negative of min/max ratio
    def objective(x):
        points = x.reshape(14, 3)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return float('inf')
            
        ratio = min_dist / max_dist
        # Return negative because we want to maximize
        return -ratio
    
    # Generate high-quality initial configuration based on icosahedron
    def generate_icosahedral_config():
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Icosahedron vertices (normalized)
        vertices = []
        # Add vertices of the form (±1, ±φ, 0), (±φ, 0, ±1), (0, ±1, ±φ)
        for i in [-1, 1]:
            for j in [-1, 1]:
                vertices.append([i, j*phi, 0])
                vertices.append([i*phi, 0, j])
                vertices.append([0, i, j*phi])
        
        # Convert to numpy array and normalize
        points = np.array(vertices[:12])  # First 12 vertices
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Add two more points to make 14 - place at north and south poles
        points = np.vstack([points, [0, 0, 1], [0, 0, -1]])
        
        # Perturb slightly to break symmetry and allow optimization to improve
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, (14, 3))
        points += noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Map to unit cube [0,1]^3
        points = (points + 1) / 2
        
        return points
    
    # Generate Fibonacci spiral configuration
    def generate_fibonacci_config():
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(14):
            y = 1 - (i / float(13)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = golden_angle * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        points = np.array(points)
        # Normalize to unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Perturb slightly
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, (14, 3))
        points += noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Map to unit cube [0,1]^3
        points = (points + 1) / 2
        
        return points
    
    # Generate layered configuration with better spacing
    def generate_layered_config():
        points = []
        for i in range(14):
            # Distribute along z-axis with better spacing
            z = -1 + 2 * i / 13
            # Distribute around circle at that z level with better radius
            r = np.sqrt(max(0, 1 - z*z))
            angle = 2 * np.pi * i / 14
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y, z])
        points = np.array(points)
        # Normalize to unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        # Perturb slightly
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, (14, 3))
        points += noise
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        # Map to unit cube [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Generate random configuration with good spread
    def generate_random_config():
        np.random.seed(42)
        points = np.random.rand(14, 3) * 2 - 1  # Range [-1, 1]
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        # Map to unit cube [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Multi-start optimization approach with better strategies
    best_ratio = 0.0
    best_points = None
    
    # Try different initial configurations
    initial_configs = [
        generate_icosahedral_config(),
        generate_fibonacci_config(),
        generate_layered_config(),
        generate_random_config()
    ]
    
    # Add variations with different perturbations
    for i in range(4):
        np.random.seed(100 + i * 10)
        points = np.random.rand(14, 3)
        # Ensure they're within bounds
        points = np.clip(points, 0, 1)
        initial_configs.append(points)
    
    # Optimization loop with multiple methods and better error handling
    for i, initial_points in enumerate(initial_configs):
        # Try different optimization methods
        try:
            x0 = initial_points.flatten()
            
            # Method 1: SLSQP with tighter tolerances
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                points_opt = result.x.reshape(-1, 3)
                ratio = compute_min_max_ratio(points_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points_opt.copy()
                    
        except Exception as e:
            # Continue to next configuration if this one fails
            continue
    
    # Enhanced refinement with multiple passes
    if best_points is not None:
        # Multiple refinement passes with decreasing perturbations
        for pass_num in range(5):
            # Decreasing perturbation magnitudes for fine-tuning
            perturbation_magnitude = 0.02 / (pass_num + 1)
            
            # Try multiple perturbed versions
            for i in range(5):
                np.random.seed(200 + pass_num * 10 + i)
                perturbed = best_points + np.random.normal(0, perturbation_magnitude, best_points.shape)
                perturbed = np.clip(perturbed, 0, 1)
                
                try:
                    x0 = perturbed.flatten()
                    result = minimize(
                        objective,
                        x0,
                        method='SLSQP',
                        options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10},
                        tol=1e-10
                    )
                    
                    if result.success:
                        points_opt = result.x.reshape(-1, 3)
                        ratio = compute_min_max_ratio(points_opt)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = points_opt.copy()
                            
                except Exception:
                    continue
    
    # Fallback to the best configuration if nothing worked
    if best_points is None:
        # Use the layered configuration as reliable fallback
        best_points = generate_layered_config()
    
    # Ensure final result is within bounds
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
