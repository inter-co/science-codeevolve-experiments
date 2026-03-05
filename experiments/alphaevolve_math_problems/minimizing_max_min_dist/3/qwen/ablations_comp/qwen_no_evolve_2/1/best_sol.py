# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import random
from numba import jit
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# More efficient distance matrix computation
@jit(nopython=True)
def compute_min_max_ratio_jit(points):
    """JIT compiled version for faster computation"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dz = points[i, 2] - points[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz
            dist = np.sqrt(dist_sq)
            
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    return min_dist, max_dist

# Vectorized computation for even faster evaluation
@jit(nopython=True)
def compute_min_max_ratio_vectorized(points):
    """Vectorized computation of min/max distances"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0
    
    # Compute all pairwise distances efficiently
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(
                (points[i, 0] - points[j, 0])**2 +
                (points[i, 1] - points[j, 1])**2 +
                (points[i, 2] - points[j, 2])**2
            )
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    return min_dist, max_dist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses advanced optimization techniques including multiple initialization strategies and 
    specialized algorithms for better convergence.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x_flat):
        """Objective function to minimize negative of min/max ratio"""
        points = x_flat.reshape(-1, 3)
        min_dist, max_dist = compute_min_max_ratio_vectorized(points)
        
        # Avoid division by zero
        if max_dist == 0:
            return -1.0
            
        # Return negative ratio (we want to maximize ratio, so minimize negative)
        return -min_dist / max_dist
    
    def constraint_func(x_flat):
        """Constraint to keep points within unit sphere"""
        points = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        # Return positive values where constraint is violated (norm > 1)
        return 1.0 - norms
    
    # Enhanced initialization strategies
    initial_strategies = []
    
    # Strategy 1: Fibonacci spiral on sphere (improved)
    def fibonacci_sphere(n):
        points = np.zeros((n, 3))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = np.arccos(y)  # angle from z-axis
            phi = ((i + 1) * golden_ratio) % 1 * 2 * np.pi  # angle around z-axis
            
            points[i, 0] = radius * np.cos(phi)
            points[i, 1] = radius * np.sin(phi)
            points[i, 2] = y
        return points * 0.95
    
    # Strategy 2: Regular icosahedron-based arrangement (more uniform)
    def icosahedron_arrangement():
        # Vertices of regular icosahedron scaled to unit sphere
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ])
        # Normalize to unit sphere and scale appropriately
        vertices = vertices / np.linalg.norm(vertices[0])
        # Add additional points for 14 total
        # Place 2 more points along axes
        additional = np.array([
            [0, 0, 0.9],
            [0, 0, -0.9]
        ])
        return np.vstack([vertices, additional]) * 0.8
    
    # Strategy 3: Cluster-based arrangement with repulsion
    def cluster_repulsion_arrangement():
        # Start with a basic configuration
        points = np.zeros((14, 3))
        
        # Place points in a way that tries to avoid clustering
        # Use a more systematic approach
        idx = 0
        # Place 8 points like a cube with some extra points
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    points[idx] = [(i-0.5)*1.2, (j-0.5)*1.2, (k-0.5)*1.2]
                    idx += 1
        
        # Add 6 more points along axes
        points[idx] = [0, 0, 0.8]
        idx += 1
        points[idx] = [0, 0, -0.8]
        idx += 1
        points[idx] = [0.8, 0, 0]
        idx += 1
        points[idx] = [-0.8, 0, 0]
        idx += 1
        points[idx] = [0, 0.8, 0]
        idx += 1
        points[idx] = [0, -0.8, 0]
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / np.max(norms) * 0.8
        return points
    
    # Strategy 4: Optimized arrangement from known good solutions
    def optimized_arrangement():
        # Based on research and known good configurations for 14 points
        points = np.array([
            [0.0, 0.0, 1.0],      # North pole
            [0.0, 0.0, -1.0],     # South pole
            [0.0, 0.7071, 0.7071], # Diagonal
            [0.0, 0.7071, -0.7071], # Diagonal
            [0.0, -0.7071, 0.7071], # Diagonal
            [0.0, -0.7071, -0.7071], # Diagonal
            [0.7071, 0.0, 0.7071], # Diagonal
            [0.7071, 0.0, -0.7071], # Diagonal
            [-0.7071, 0.0, 0.7071], # Diagonal
            [-0.7071, 0.0, -0.7071], # Diagonal
            [0.7071, 0.7071, 0.0], # Diagonal
            [0.7071, -0.7071, 0.0], # Diagonal
            [-0.7071, 0.7071, 0.0], # Diagonal
            [-0.7071, -0.7071, 0.0], # Diagonal
        ])
        # Normalize and scale
        points = points * 0.7
        return points
    
    # Strategy 5: Spherical code-like arrangement
    def spherical_code_arrangement():
        # Create points arranged in a more symmetric way
        points = np.zeros((14, 3))
        
        # Place points in rings
        angles = np.linspace(0, 2*np.pi, 14, endpoint=False)
        radii = np.linspace(0.2, 0.9, 14)
        
        for i in range(14):
            theta = angles[i]
            r = radii[i]
            points[i] = [r * np.cos(theta), r * np.sin(theta), 0]
            
        # Adjust to make it more 3D
        points[:, 2] = np.linspace(-0.7, 0.7, 14)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / np.max(norms) * 0.8
        return points
    
    # Strategy 6: Improved icosahedral arrangement with better symmetry
    def improved_icosahedral():
        # Start with icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [-1,  phi,  0],
            [ 1,  phi,  0],
            [-1, -phi,  0],
            [ 1, -phi,  0],
            [ 0, -1,  phi],
            [ 0,  1,  phi],
            [ 0, -1, -phi],
            [ 0,  1, -phi],
            [ phi,  0, -1],
            [ phi,  0,  1],
            [-phi,  0, -1],
            [-phi,  0,  1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # Add 2 more points along z-axis
        additional = np.array([
            [0, 0, 0.85],
            [0, 0, -0.85]
        ])
        
        points = np.vstack([vertices, additional])
        return points * 0.9
    
    # Strategy 7: Modified cube arrangement with diagonal points
    def modified_cube_arrangement():
        # Start with cube vertices
        cube_points = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ])
        
        # Add 6 more points along axes
        axis_points = np.array([
            [0, 0, 0.8],
            [0, 0, -0.8],
            [0.8, 0, 0],
            [-0.8, 0, 0],
            [0, 0.8, 0],
            [0, -0.8, 0]
        ])
        
        points = np.vstack([cube_points, axis_points])
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / np.max(norms) * 0.85
        return points
    
    # Strategy 8: Random but well-distributed points with local optimization
    def random_distributed():
        np.random.seed(42)
        points = np.random.uniform(-0.9, 0.9, (14, 3))
        # Ensure they're within unit sphere
        norms = np.linalg.norm(points, axis=1)
        mask = norms > 1
        if np.any(mask):
            points[mask] = points[mask] / norms[mask][:, np.newaxis] * 0.95
        return points
    
    # Collect initial strategies
    initial_strategies.append(fibonacci_sphere(14))
    initial_strategies.append(icosahedron_arrangement())
    initial_strategies.append(cluster_repulsion_arrangement())
    initial_strategies.append(optimized_arrangement())
    initial_strategies.append(spherical_code_arrangement())
    initial_strategies.append(improved_icosahedral())
    initial_strategies.append(modified_cube_arrangement())
    initial_strategies.append(random_distributed())
    
    # Add some random variations with better control
    np.random.seed(42)
    for _ in range(8):
        base = fibonacci_sphere(14)
        # Add controlled noise that maintains good distribution
        noise = np.random.normal(0, 0.05, (14, 3))
        perturbed = base + noise
        
        # Keep points within unit sphere by normalizing those that exceed
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 1
        if np.any(mask):
            perturbed[mask] = perturbed[mask] / norms[mask][:, np.newaxis] * 0.95
            
        initial_strategies.append(perturbed)
    
    # Define constraints
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Bounds for each coordinate (-1, 1)
    bounds = [(-1, 1) for _ in range(42)]
    
    # Multiple optimization attempts with different strategies
    best_ratio = -np.inf
    best_points = None
    
    # Try different optimization methods with enhanced settings
    methods = ['L-BFGS-B', 'TNC', 'SLSQP']
    
    # Use more aggressive optimization parameters
    opt_options = {
        'maxiter': 1000,
        'ftol': 1e-12,
        'gtol': 1e-12
    }
    
    # Track how many iterations we're doing
    total_attempts = len(initial_strategies) * len(methods)
    attempt_count = 0
    
    # Use a more intelligent approach - try with larger number of attempts
    for strategy_idx, initial_points in enumerate(initial_strategies):
        x0 = initial_points.flatten()
        
        for method in methods:
            attempt_count += 1
            try:
                # Optimize using different methods
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options=opt_options,
                    callback=lambda x: None  # Empty callback to prevent printing
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    min_dist, max_dist = compute_min_max_ratio_vectorized(final_points)
                    
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
            except Exception as e:
                continue
                
            # Early stopping if we're getting close to our target
            if best_ratio > 0.48:  # Good enough threshold
                break
                
        if best_ratio > 0.48:
            break
    
    # If no improvement found, return the best initial configuration
    if best_points is None:
        # Return the most promising initial configuration
        return optimized_arrangement()
    
    # Apply one final refinement with a more targeted approach
    if best_ratio < 0.45:
        # Try a more focused optimization on the best solution so far
        try:
            refined_points = best_points.copy()
            x0 = refined_points.flatten()
            
            # Use a more aggressive method with tighter tolerances
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 3)
                min_dist, max_dist = compute_min_max_ratio_vectorized(final_points)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_points = final_points.copy()
        except:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
