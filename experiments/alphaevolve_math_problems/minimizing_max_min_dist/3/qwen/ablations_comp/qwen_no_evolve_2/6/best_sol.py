# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')
from numba import jit

@jit(nopython=True)
def compute_distances_jit(points):
    """Compute pairwise distances efficiently using Numba"""
    n = points.shape[0]
    distances = np.zeros((n * (n - 1) // 2,))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((points[i, 0] - points[j, 0])**2 + 
                          (points[i, 1] - points[j, 1])**2 + 
                          (points[i, 2] - points[j, 2])**2)
            distances[idx] = dist
            idx += 1
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses advanced optimization techniques including differential evolution and local refinement.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    n = 14
    d = 3
    
    # Use multiple starting configurations for better optimization
    best_ratio = 0
    best_points = None
    
    # Multiple initialization strategies
    initial_configs = []
    
    # Strategy 1: Fibonacci spiral on sphere (improved)
    np.random.seed(42)
    points = np.zeros((n, d))
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        theta = golden_angle * i
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points[i] = [x, y, z]
    
    # Scale to unit sphere
    if np.linalg.norm(points[0]) > 0:
        points = points / np.linalg.norm(points[0])
    initial_configs.append(points.copy())
    
    # Strategy 2: Better spherical arrangement using icosahedron-based packing
    np.random.seed(123)
    # Generate points on sphere using icosahedral symmetry approach
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = []
    
    # Create icosahedron vertices and subdivide
    # Simple approach: use vertices of regular icosahedron scaled to unit sphere
    t = 1.0 / np.sqrt(phi**2 + 1)
    u = phi / np.sqrt(phi**2 + 1)
    
    # Icosahedron vertices
    ico_vertices = [
        [0, t, u], [0, t, -u], [0, -t, u], [0, -t, -u],
        [t, u, 0], [t, -u, 0], [-t, u, 0], [-t, -u, 0],
        [u, 0, t], [-u, 0, t], [u, 0, -t], [-u, 0, -t]
    ]
    
    ico_vertices = np.array(ico_vertices)
    # Normalize to unit sphere
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0])
    
    # Add some random points around the icosahedron
    points_icosahedron = ico_vertices.copy()
    while len(points_icosahedron) < n:
        # Add perturbed points
        random_perturb = np.random.normal(0, 0.1, (1, 3))
        random_perturb = random_perturb / np.linalg.norm(random_perturb)
        points_icosahedron = np.vstack([points_icosahedron, random_perturb])
    
    points_icosahedron = points_icosahedron[:n]
    initial_configs.append(points_icosahedron)
    
    # Strategy 3: Random points in unit sphere with rejection sampling
    np.random.seed(456)
    points_random = np.random.uniform(-1, 1, (n, d))
    norms = np.linalg.norm(points_random, axis=1)
    valid_indices = norms <= 1
    points_random = points_random[valid_indices]
    
    # If not enough points, generate more
    while len(points_random) < n:
        additional = np.random.uniform(-1, 1, (n - len(points_random), d))
        additional_norms = np.linalg.norm(additional, axis=1)
        valid_additional = additional[additional_norms <= 1]
        points_random = np.vstack([points_random, valid_additional])
    
    points_random = points_random[:n]
    initial_configs.append(points_random)
    
    # Strategy 4: Grid-based arrangement with slight perturbation
    np.random.seed(789)
    # Create a simple 3D grid-like arrangement
    grid_points = []
    side = int(np.ceil(n**(1/3)))
    count = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if count < n:
                    x = i / (side - 1) * 2 - 1
                    y = j / (side - 1) * 2 - 1  
                    z = k / (side - 1) * 2 - 1
                    grid_points.append([x, y, z])
                    count += 1
                if count >= n:
                    break
            if count >= n:
                break
        if count >= n:
            break
    
    points_grid = np.array(grid_points)
    # Perturb slightly
    points_grid += np.random.normal(0, 0.05, points_grid.shape)
    # Keep within unit sphere
    norms = np.linalg.norm(points_grid, axis=1)
    mask = norms <= 1
    points_grid = points_grid[mask]
    while len(points_grid) < n:
        additional = np.random.uniform(-1, 1, (n - len(points_grid), d))
        additional_norms = np.linalg.norm(additional, axis=1)
        valid_additional = additional[additional_norms <= 1]
        points_grid = np.vstack([points_grid, valid_additional])
    
    points_grid = points_grid[:n]
    initial_configs.append(points_grid)
    
    # Optimization function with improved error handling
    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Compute distances efficiently
        try:
            distances = pdist(points)
        except:
            # Fallback to manual computation
            distances = compute_distances_jit(points)
        
        # Avoid division by zero
        if len(distances) == 0 or np.allclose(distances, 0):
            return -1e10  # Very bad objective
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative because we want to maximize
        if max_dist > 0:
            return -min_dist / max_dist
        else:
            return -1e10
    
    # Constraint function: points must stay within unit sphere
    def constraint_func(x_flat):
        points = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return 1.0 - norms  # Should be >= 0
    
    # Try differential evolution first (global optimization)
    bounds = [(-1.0, 1.0) for _ in range(n * 3)]
    
    try:
        # Global optimization with differential evolution
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            atol=1e-6,
            rtol=1e-6
        )
        
        if de_result.success:
            optimized_points = de_result.x.reshape(-1, 3)
            # Calculate final ratio
            distances = pdist(optimized_points)
            if len(distances) > 0 and np.max(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                ratio = min_dist / max_dist
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
    except Exception as e:
        pass
    
    # Try local optimization with multiple starting points
    for i, initial_points in enumerate(initial_configs):
        # Try multiple optimization methods
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_points.flatten(),
                    method=method,
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8},
                    tol=1e-8
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 3)
                    # Calculate final ratio
                    distances = pdist(optimized_points)
                    if len(distances) > 0 and np.max(distances) > 0:
                        min_dist = np.min(distances)
                        max_dist = np.max(distances)
                        ratio = min_dist / max_dist
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
            except Exception as e:
                continue
    
    # Final refinement using local optimization on best solution
    if best_points is not None:
        try:
            result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                distances = pdist(refined_points)
                if len(distances) > 0 and np.max(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    ratio = min_dist / max_dist
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()
                        
        except Exception as e:
            pass
    
    # If no optimization worked, return the best initial configuration
    if best_points is None:
        # Use the first configuration as fallback
        return initial_configs[0]
    
    return best_points


# EVOLVE-BLOCK-END
