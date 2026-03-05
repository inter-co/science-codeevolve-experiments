# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining global optimization, geometric construction, and local refinement.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return 0.0
            
        return min_dist / max_dist
    
    # Define objective function to maximize min/max ratio
    def objective(x):
        # Reshape flat array back to points
        points = x.reshape(14, 3)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return -np.inf
            
        # Avoid division by zero
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return -np.inf
            
        # We want to maximize min/max ratio, so we minimize -min/max ratio
        return -min_dist / max_dist
    
    # Generate multiple high-quality initial configurations
    best_ratio = -np.inf
    best_solution = None
    
    # Strategy 1: Global optimization with dual_annealing (like INSPIRATION 1)
    # This approach is more likely to escape local optima and find better solutions
    np.random.seed(42)
    
    # Use dual annealing for global optimization - more effective for this problem
    try:
        # Try with both local search enabled and disabled for robustness
        for no_local_search in [True, False]:
            result = dual_annealing(
                objective, 
                bounds=[(-1, 1) for _ in range(14 * 3)],  # Bounds for [-1,1]^3
                maxiter=1000,  # More iterations for better global search
                seed=42,
                no_local_search=no_local_search
            )
            
            if result.success:
                points = result.x.reshape(14, 3)
                ratio = compute_min_max_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = points.copy()
    except:
        pass
    
    # Strategy 2: Improved geometric construction 
    if best_solution is None:
        # Create points using a combination of fibonacci spiral and icosahedral structure
        # First create points along the fibonacci spiral
        fib_points = np.zeros((14, 3))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(14):
            y = 1 - (i / (14 - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = np.arccos(y)
            phi_val = (i * golden_ratio) % (2 * np.pi)
            fib_points[i] = [radius * np.sin(theta) * np.cos(phi_val),
                            radius * np.sin(theta) * np.sin(phi_val),
                            radius * np.cos(theta)]
        
        # Try optimizing this configuration
        x0 = fib_points.flatten()
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-1, 1) for _ in range(14 * 3)],
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                points = result.x.reshape(14, 3)
                ratio = compute_min_max_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = points.copy()
        except:
            pass
    
    # Strategy 3: Another geometric approach - perturbed icosahedral structure
    if best_solution is None:
        # Icosahedral vertices scaled to [-1,1]^3 (better for optimization)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        ico_vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere then scale to [-1,1]^3
        ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0])
        ico_vertices = ico_vertices * 0.9  # Slightly reduce to give room for optimization
        
        # Add two more points - distribute them more evenly
        additional_points = np.array([[0, 0, 0.8], [0, 0, -0.8]])
        
        initial_points = np.vstack([ico_vertices, additional_points])
        
        # Try optimizing this configuration
        x0 = initial_points.flatten()
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-1, 1) for _ in range(14 * 3)],
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                points = result.x.reshape(14, 3)
                ratio = compute_min_max_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = points.copy()
        except:
            pass
    
    # Strategy 4: Pure random initialization with better optimization
    if best_solution is None:
        # Generate random points in [-1,1]^3
        np.random.seed(42)
        random_points = np.random.uniform(-1, 1, (14, 3))
        
        # Try optimizing this configuration
        x0 = random_points.flatten()
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=[(-1, 1) for _ in range(14 * 3)],
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                points = result.x.reshape(14, 3)
                ratio = compute_min_max_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_solution = points.copy()
        except:
            pass
    
    # If no optimization worked, return a reasonable configuration
    if best_solution is None:
        # Create a simple good configuration using fibonacci spiral
        fib_points = np.zeros((14, 3))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(14):
            y = 1 - (i / (14 - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = np.arccos(y)
            phi_val = (i * golden_ratio) % (2 * np.pi)
            fib_points[i] = [radius * np.sin(theta) * np.cos(phi_val),
                            radius * np.sin(theta) * np.sin(phi_val),
                            radius * np.cos(theta)]
        best_solution = fib_points
    
    return best_solution


# EVOLVE-BLOCK-END
