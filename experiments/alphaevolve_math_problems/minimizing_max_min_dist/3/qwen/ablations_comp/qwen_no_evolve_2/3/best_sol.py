# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import random
from numba import jit
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# More efficient distance calculation
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
        min_dist, max_dist = compute_min_max_ratio_jit(points)
        
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
    
    # Enhanced initialization strategies with focus on better starting configurations
    initial_strategies = []
    
    # Strategy 1: Improved Fibonacci spiral on sphere (more evenly distributed)
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
    
    # Strategy 2: Modified icosahedral arrangement with better spacing
    def modified_icosahedral():
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
        vertices = vertices / np.linalg.norm(vertices[0]) * 0.8
        
        # Add 2 more points along z-axis at different distances
        additional = np.array([
            [0, 0, 0.7],
            [0, 0, -0.7]
        ])
        
        return np.vstack([vertices, additional])
    
    # Strategy 3: Symmetric arrangement based on octahedral symmetry with better distribution
    def octahedral_arrangement():
        # Octahedron vertices plus some additional points
        points = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [0.5, 0.5, 0.5], [0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5], [0.5, -0.5, -0.5],
            [-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5]
        ])
        # Normalize and scale
        norms = np.linalg.norm(points, axis=1)
        points = points / np.max(norms) * 0.8
        return points
    
    # Strategy 4: Optimized arrangement from known good solutions (improved version)
    def optimized_arrangement():
        # Based on research and known good configurations for 14 points
        # Using a more balanced arrangement inspired by spherical codes
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
    
    # Strategy 5: Better random initialization with proper constraints
    def random_distributed():
        np.random.seed(42)
        points = np.random.uniform(-0.9, 0.9, (14, 3))
        # Apply simple repulsion to prevent clustering
        for _ in range(50):
            for i in range(14):
                for j in range(i+1, 14):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq < 0.02:  # Too close
                        # Repel them
                        diff_norm = np.linalg.norm(diff)
                        if diff_norm > 0:
                            move = (0.02 - dist_sq) * diff / (diff_norm * 2)
                            points[i] += move
                            points[j] -= move
        return points
    
    # Strategy 6: Spherical code inspired arrangement
    def spherical_code_arrangement():
        # Create points in a way that tries to maximize uniformity
        # Using a variant of the 14-point spherical code construction
        points = np.zeros((14, 3))
        
        # Place 8 points at vertices of a cube inscribed in unit sphere
        cube_vertices = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ])
        cube_vertices = cube_vertices / np.linalg.norm(cube_vertices[0])
        
        # Place 6 points near the poles
        points[:8] = cube_vertices * 0.8
        points[8] = [0, 0, 0.9]  # North pole
        points[9] = [0, 0, -0.9]  # South pole
        
        # Add two more points at intermediate positions
        points[10] = [0.7, 0.7, 0]
        points[11] = [-0.7, -0.7, 0]
        points[12] = [0.7, -0.7, 0]
        points[13] = [-0.7, 0.7, 0]
        
        return points
    
    # Strategy 7: Improved Fibonacci-like arrangement with better spacing
    def fibonacci_like_arrangement():
        # Create a more even distribution using a modified Fibonacci approach
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(14):
            # Distribute points along the z-axis more uniformly
            y = 1 - (i / (14 - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y * y)
            
            # Apply Fibonacci-like distribution
            theta = i * golden_angle
            
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            points.append([x, y, z])
            
        return np.array(points) * 0.9
    
    # Strategy 8: Hexagonal arrangement on sphere
    def hexagonal_arrangement():
        # Create points in a hexagonal pattern on the sphere
        points = []
        
        # Add 2 poles
        points.append([0, 0, 1])
        points.append([0, 0, -1])
        
        # Add 12 points in rings
        for i in range(12):
            angle = i * np.pi / 6
            height = np.sin(angle * 0.5)  # Non-uniform heights for better spread
            radius = np.cos(angle * 0.5)
            
            x = radius * np.cos(i * np.pi / 3)
            y = radius * np.sin(i * np.pi / 3)
            z = height
            
            points.append([x, y, z])
            
        return np.array(points) * 0.8
    
    # Strategy 9: Enhanced version of the known good solution
    def enhanced_known_solution():
        # Based on literature about optimal 14-point arrangements
        # This uses a carefully designed configuration
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
        # Scale more carefully to improve performance
        points = points * 0.8
        return points
    
    # Strategy 10: More sophisticated geometric arrangement based on symmetry
    def symmetric_arrangement():
        # Create points using a combination of tetrahedral and octahedral symmetries
        points = []
        
        # Tetrahedral vertices scaled appropriately
        tetra = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ])
        tetra = tetra / np.linalg.norm(tetra[0]) * 0.6
        
        # Add octahedral points
        octa = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [-1, 0, 0], [0, -1, 0], [0, 0, -1]
        ])
        octa = octa * 0.7
        
        # Combine and add additional points
        points.extend(tetra.tolist())
        points.extend(octa.tolist())
        
        # Add 2 more points along z-axis
        points.append([0, 0, 0.8])
        points.append([0, 0, -0.8])
        
        # Add 2 more diagonal points
        points.append([0.6, 0.6, 0])
        points.append([-0.6, -0.6, 0])
        
        return np.array(points)
    
    # Collect initial strategies
    initial_strategies.extend([
        fibonacci_sphere(14),
        modified_icosahedral(),
        octahedral_arrangement(),
        optimized_arrangement(),
        random_distributed(),
        spherical_code_arrangement(),
        fibonacci_like_arrangement(),
        hexagonal_arrangement(),
        enhanced_known_solution(),
        symmetric_arrangement()
    ])
    
    # Add some random variations with better control
    np.random.seed(42)
    for _ in range(5):  # Reduced from 10 to speed up
        base = fibonacci_sphere(14)
        # Add controlled noise that maintains good distribution
        noise = np.random.normal(0, 0.03, (14, 3))  # Reduced noise level
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
    
    # Use more aggressive optimization parameters
    opt_options = {
        'maxiter': 300,  # Further reduced iterations to save time
        'ftol': 1e-12,
        'gtol': 1e-12
    }
    
    # Try global optimization first to get close to good solution
    try:
        # Global optimization using differential evolution with fewer iterations
        de_bounds = [(-1, 1) for _ in range(42)]
        de_result = differential_evolution(
            objective, 
            de_bounds, 
            maxiter=30,  # Further reduced iterations for speed
            popsize=8,   # Reduced population size
            seed=42,
            disp=False
        )
        
        if de_result.success:
            global_points = de_result.x.reshape(-1, 3)
            min_dist, max_dist = compute_min_max_ratio_jit(global_points)
            if max_dist > 0:
                global_ratio = min_dist / max_dist
                if global_ratio > best_ratio:
                    best_ratio = global_ratio
                    best_points = global_points.copy()
    except Exception as e:
        pass
    
    # Then local optimization with multiple starting points - but limit number of attempts
    methods = ['L-BFGS-B', 'TNC', 'SLSQP']  # Added SLSQP for better convergence
    
    # Limit number of strategies tried to reduce computation time
    strategies_to_try = min(6, len(initial_strategies))  # Try fewer strategies
    
    # Use a more targeted approach - only run optimization on top performing strategies
    for strategy_idx, initial_points in enumerate(initial_strategies[:strategies_to_try]):
        x0 = initial_points.flatten()
        
        # Run optimizations with different methods
        for method in methods:
            try:
                # Optimize using different methods
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options=opt_options
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 3)
                    min_dist, max_dist = compute_min_max_ratio_jit(final_points)
                    
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
            except Exception as e:
                continue
                
            # Early stopping if we're getting close to our target
            if best_ratio > 0.48:  # Stop early if we're approaching target
                break
                
        if best_ratio > 0.48:
            break
    
    # Additional refinement step with a more focused optimization
    if best_points is not None and best_ratio < 0.485:
        try:
            # Refine with a more precise optimization
            refined_options = {
                'maxiter': 200,
                'ftol': 1e-14,
                'gtol': 1e-14
            }
            
            x0_refined = best_points.flatten()
            result_refined = minimize(
                objective,
                x0_refined,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=cons,
                options=refined_options
            )
            
            if result_refined.success:
                final_points_refined = result_refined.x.reshape(-1, 3)
                min_dist, max_dist = compute_min_max_ratio_jit(final_points_refined)
                if max_dist > 0:
                    ratio_refined = min_dist / max_dist
                    if ratio_refined > best_ratio:
                        best_ratio = ratio_refined
                        best_points = final_points_refined.copy()
        except Exception as e:
            pass
    
    # If no improvement found, return the best initial configuration
    if best_points is None:
        # Return the most promising initial configuration
        return enhanced_known_solution()
    
    return best_points


# EVOLVE-BLOCK-END
