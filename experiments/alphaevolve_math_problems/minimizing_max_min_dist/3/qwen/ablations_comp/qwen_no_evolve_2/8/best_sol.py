# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')
from numba import jit
import time

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
    
    # Strategy 1: Better Fibonacci spiral on sphere (more uniform)
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
    
    # Strategy 2: Octahedral arrangement with perturbations
    np.random.seed(123)
    # Start with octahedron vertices
    octahedron_vertices = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    points_octahedron = octahedron_vertices.copy()
    # Add remaining points randomly distributed around octahedron
    remaining_points = n - len(octahedron_vertices)
    if remaining_points > 0:
        for i in range(remaining_points):
            # Generate random point on unit sphere
            point = np.random.randn(3)
            point = point / np.linalg.norm(point)
            points_octahedron = np.vstack([points_octahedron, point])
    
    initial_configs.append(points_octahedron[:n])
    
    # Strategy 3: Random points in unit sphere with better sampling
    np.random.seed(456)
    points_random = np.random.uniform(-1, 1, (n, d))
    norms = np.linalg.norm(points_random, axis=1)
    valid_indices = norms <= 1
    points_random = points_random[valid_indices]
    
    # If not enough points, generate more using rejection sampling
    while len(points_random) < n:
        additional = np.random.uniform(-1, 1, (n - len(points_random), d))
        additional_norms = np.linalg.norm(additional, axis=1)
        valid_additional = additional[additional_norms <= 1]
        points_random = np.vstack([points_random, valid_additional])
    
    points_random = points_random[:n]
    initial_configs.append(points_random)
    
    # Strategy 4: More sophisticated grid-based arrangement
    np.random.seed(789)
    # Create a 3D arrangement that's more evenly distributed
    # Use a face-centered cubic lattice approach
    points_fcc = []
    # Generate points in a way that avoids clustering
    for i in range(int(np.ceil(n**(1/3)))):
        for j in range(int(np.ceil(n**(1/3)))):
            for k in range(int(np.ceil(n**(1/3)))):
                if len(points_fcc) < n:
                    # Use a pattern that distributes points more uniformly
                    x = (i + 0.5 * (j + k) % 2) / (int(np.ceil(n**(1/3))) - 1) * 2 - 1
                    y = (j + 0.5 * (k + i) % 2) / (int(np.ceil(n**(1/3))) - 1) * 2 - 1
                    z = (k + 0.5 * (i + j) % 2) / (int(np.ceil(n**(1/3))) - 1) * 2 - 1
                    points_fcc.append([x, y, z])
    
    points_fcc = np.array(points_fcc)
    # Normalize to unit sphere
    norms = np.linalg.norm(points_fcc, axis=1)
    points_fcc = points_fcc / np.max(norms) * 0.9  # Keep inside sphere
    points_fcc = points_fcc[:n]
    initial_configs.append(points_fcc)
    
    # Strategy 5: Improved spherical arrangement using known good configurations
    # Generate points using a method inspired by the Thomson problem solution
    np.random.seed(999)
    points_thomson = np.zeros((n, d))
    
    # Use a more uniform distribution based on Fibonacci-like approach
    for i in range(n):
        # Better distribution using golden angle and logarithmic spiral
        phi = np.arccos(1 - 2 * (i / (n - 1)))
        theta = np.sqrt(n * np.pi) * phi
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        points_thomson[i] = [x, y, z]
    
    initial_configs.append(points_thomson)
    
    # Strategy 6: Known good configuration from literature - Dodecahedron-based arrangement
    # This often provides a good starting point for such problems
    np.random.seed(111)
    # Create points on the surface of a regular dodecahedron (approximation)
    # Vertices of a regular dodecahedron scaled to unit sphere
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    points_dodec = []
    
    # Add vertices of dodecahedron
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points_dodec.append([i, j, k])
    
    # Add edge midpoints
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            if i != j:
                for sign1 in [-1, 1]:
                    for sign2 in [-1, 1]:
                        points_dodec.append([sign1 * 1, sign2 * 1, 0])
                        points_dodec.append([sign1 * 1, 0, sign2 * 1])
                        points_dodec.append([0, sign1 * 1, sign2 * 1])
    
    # Add face centers
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            for k in [0, 1, 2]:
                if i + j + k == 2:
                    points_dodec.append([1, 1, 1])
    
    points_dodec = np.array(points_dodec)
    # Normalize to unit sphere
    norms = np.linalg.norm(points_dodec, axis=1)
    points_dodec = points_dodec / norms[:, np.newaxis]
    points_dodec = points_dodec[:n]
    initial_configs.append(points_dodec)
    
    # Strategy 7: Optimized arrangement based on known 14-point solutions
    # Using a configuration inspired by sphere packing and symmetry considerations
    np.random.seed(222)
    points_optimized = np.zeros((n, d))
    
    # Place points in a way that balances distribution and minimizes clustering
    # Use a variant of the Fibonacci spiral approach but with better spacing
    for i in range(n):
        # Improved spiral placement
        if i == 0:
            points_optimized[i] = [0, 0, 1]
        elif i == n-1:
            points_optimized[i] = [0, 0, -1]
        else:
            # Use spherical coordinates with better distribution
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            # Better angular spacing to avoid clustering
            theta = (i * 4.8) % (2 * np.pi)  # Adjusted spacing
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points_optimized[i] = [x, y, z]
    
    initial_configs.append(points_optimized)
    
    # Optimization function with improved error handling and efficiency
    def objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Compute distances efficiently - avoid recomputation when possible
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
    
    # More efficient optimization approach with early termination
    start_time = time.time()
    bounds = [(-1.0, 1.0) for _ in range(n * 3)]
    
    # Try global optimization with limited iterations due to time constraints
    try:
        # Use a faster optimization approach with fewer iterations
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=20,  # Reduced iterations to save time
            popsize=6,   # Smaller population
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
    
    # Try local optimization with multiple starting points, but limit total time
    for i, initial_points in enumerate(initial_configs):
        # Check if we're running out of time
        if time.time() - start_time > 55:  # Leave 5 seconds for final refinement
            break
            
        # Try multiple optimization methods
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        
        for method in methods:
            # Skip some methods if time is running short
            if time.time() - start_time > 58:
                break
                
            try:
                result = minimize(
                    objective,
                    initial_points.flatten(),
                    method=method,
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8},
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
    
    # Final refinement using local optimization on best solution with very limited iterations
    if best_points is not None and time.time() - start_time < 58:
        try:
            result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10},
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
