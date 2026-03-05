# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from scipy.spatial import ConvexHull
import warnings
from numba import jit
from joblib import Parallel, delayed
import itertools

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
    Uses a hybrid approach combining geometric initialization and advanced optimization.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    
    # Enhanced initialization strategies based on mathematical insights
    # Strategy 1: Icosahedron-based configuration with refined positioning
    def get_icosahedron_initial():
        # Regular icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, -1, -phi], [0, 1, -phi],
            [1, phi, 0], [-1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, -1], [-phi, 0, 1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Scale to [0,1]^3
        points = (vertices + 1) / 2
        return points
    
    # Strategy 2: Fibonacci spiral on sphere with better distribution
    def get_fibonacci_initial():
        points = np.zeros((n, 3))
        
        # Fibonacci spiral approach for even distribution on sphere
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n)
        
        # Use a variant that gives better spacing
        theta = np.arccos(1 - 2 * indices / (n - 1))
        phi = np.mod(indices * (4 - golden_ratio), 2 * np.pi)
        
        # Convert spherical to Cartesian coordinates
        points[:, 0] = np.sin(theta) * np.cos(phi)  # x
        points[:, 1] = np.sin(theta) * np.sin(phi)  # y  
        points[:, 2] = np.cos(theta)                # z
        
        # Scale and shift to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 3: Octahedral + tetrahedral structure with better placement
    def get_octa_tetra_initial():
        # Start with octahedron vertices (scaled to unit cube)
        octahedron_points = np.array([
            [0.5, 0.5, 1.0],   # top
            [0.5, 0.5, 0.0],   # bottom
            [1.0, 0.5, 0.5],   # right
            [0.0, 0.5, 0.5],   # left
            [0.5, 1.0, 0.5],   # front
            [0.5, 0.0, 0.5],   # back
        ])
        
        # Add 8 more points arranged in a tetrahedral pattern
        tetrahedron_points = np.array([
            [0.2, 0.2, 0.8],
            [0.2, 0.8, 0.2],
            [0.8, 0.2, 0.2],
            [0.8, 0.8, 0.8],
            [0.2, 0.8, 0.8],
            [0.8, 0.2, 0.8],
            [0.8, 0.8, 0.2],
            [0.5, 0.5, 0.5]
        ])
        
        return np.vstack([octahedron_points, tetrahedron_points])
    
    # Strategy 4: Improved icosahedron with additional strategic points
    def get_improved_icosahedron_initial():
        # Start with regular icosahedron
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, -1, -phi], [0, 1, -phi],
            [1, phi, 0], [-1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, -1], [-phi, 0, 1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Scale to [0,1]^3
        points = (vertices + 1) / 2
        
        # Add 2 more strategic points to improve distribution
        # These should be near the corners of a cube inscribed in the sphere
        corner_points = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]  # This will be replaced by actual corners
        ])
        
        # Place two points along diagonals of the unit cube
        corner_points[0] = [0.5, 0.5, 0.5] + 0.5 * np.array([1, 1, 1]) / np.sqrt(3)
        corner_points[1] = [0.5, 0.5, 0.5] + 0.5 * np.array([-1, -1, -1]) / np.sqrt(3)
        
        # Clip to [0,1]^3
        corner_points = np.clip(corner_points, 0, 1)
        points = np.vstack([points, corner_points])
        
        return points
    
    # Strategy 5: Random but well-distributed configuration with better seed
    def get_random_initial():
        # Use a more systematic approach to randomness
        points = np.random.rand(n, 3)
        return points
    
    # Strategy 6: Sphere packing inspired approach with better spacing
    def get_sphere_packing_initial():
        # Generate points on a sphere using Fibonacci-like method but with better spacing
        points = np.zeros((n, 3))
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = golden_angle * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points[i] = [x, y, z]
        
        # Scale and shift to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 7: Dodecahedron-inspired approach with better vertex selection
    def get_dodecahedron_initial():
        # Dodecahedron vertices scaled to unit sphere
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Vertices of a regular dodecahedron (normalized)
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, -1, -phi], [0, 1, -phi],
            [1, phi, 0], [-1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, -1], [-phi, 0, 1],
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Select 14 points from dodecahedron vertices
        # Use first 12 vertices of dodecahedron plus 2 more strategically chosen
        selected = vertices[:12]
        
        # Add 2 points that enhance the distribution (north and south poles)
        additional = np.array([
            [0.0, 0.0, 1.0],  # North pole
            [0.0, 0.0, -1.0]  # South pole
        ])
        
        points = np.vstack([selected, additional])
        
        # Scale to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 8: Hybrid approach combining multiple structures
    def get_hybrid_initial():
        # Combine elements from different strategies
        # Start with icosahedron points
        ico_points = get_icosahedron_initial()
        
        # Add some randomly distributed points
        random_points = np.random.rand(2, 3)
        
        # Add some points near the edges of the unit cube
        edge_points = np.array([
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 1.0]
        ])
        
        # Combine all points
        points = np.vstack([ico_points, random_points, edge_points[:2]])
        
        # Ensure we have exactly 14 points
        if len(points) > 14:
            points = points[:14]
        elif len(points) < 14:
            # Fill with random points if needed
            additional = np.random.rand(14 - len(points), 3)
            points = np.vstack([points, additional])
        
        return points
    
    # Strategy 9: Better optimized geometric configuration
    def get_optimized_geometric_initial():
        # Start with icosahedron points and refine
        points = get_icosahedron_initial()
        
        # Add a central point to improve balance
        center_point = np.array([[0.5, 0.5, 0.5]])
        points = np.vstack([points, center_point])
        
        # Remove one point to get exactly 14
        points = points[:-1]
        
        # Apply slight perturbation to improve the configuration
        noise = np.random.normal(0, 0.02, points.shape)
        points = points + noise
        points = np.clip(points, 0, 1)
        
        return points
    
    # Strategy 10: Optimized cube-based approach
    def get_cube_based_initial():
        # Start with vertices of a cube and add more points
        cube_vertices = np.array([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
        ])
        
        # Add midpoints of edges
        edge_midpoints = np.array([
            [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0],
            [0, 1, 0.5], [0, 0.5, 1], [0.5, 1, 0],
            [1, 0, 0.5], [1, 0.5, 0], [0.5, 0, 1],
            [1, 1, 0.5], [1, 0.5, 1], [0.5, 1, 1]
        ])
        
        # Combine and scale to [0,1]^3
        points = np.vstack([cube_vertices, edge_midpoints])
        
        # Add more points for better distribution
        additional_points = np.random.rand(2, 3)
        points = np.vstack([points, additional_points])
        
        # Ensure we have exactly 14 points
        if len(points) > 14:
            points = points[:14]
        elif len(points) < 14:
            additional = np.random.rand(14 - len(points), 3)
            points = np.vstack([points, additional])
            
        # Normalize to [0,1]^3
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 11: Enhanced geometric configuration using known good solutions
    def get_known_good_initial():
        # Start with the vertices of a regular icosahedron and add 2 more points
        # This is a known good configuration for many point distributions
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, -1, -phi], [0, 1, -phi],
            [1, phi, 0], [-1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, -1], [-phi, 0, 1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Scale to [0,1]^3
        points = (vertices + 1) / 2
        
        # Add two more points that are well-separated
        # Place them at positions that maximize potential minimum distance
        additional = np.array([
            [0.2, 0.2, 0.8],
            [0.8, 0.8, 0.2]
        ])
        
        points = np.vstack([points, additional])
        return points
    
    # Strategy 12: More sophisticated spherical arrangement
    def get_spherical_arrangement():
        # Generate points using a method similar to spherical codes
        # This uses a more sophisticated approach to distributing points evenly on a sphere
        points = np.zeros((n, 3))
        
        # Use a variation of the Fibonacci method with better distribution properties
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n)
        
        # Improved spacing with better angular distribution
        theta = np.arccos(1 - 2 * indices / (n - 1))
        phi = np.mod(indices * golden_ratio, 2 * np.pi)
        
        # Convert to Cartesian coordinates
        points[:, 0] = np.sin(theta) * np.cos(phi)  # x
        points[:, 1] = np.sin(theta) * np.sin(phi)  # y
        points[:, 2] = np.cos(theta)                # z
        
        # Scale and shift to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 13: Better initial configuration based on known optimal solutions
    def get_better_initial():
        # Known good configuration that's close to optimal
        # Based on research into point distributions and spherical codes
        points = np.array([
            [0.5, 0.5, 0.5],  # Center point
            [0.0, 0.0, 0.0],  # Corner
            [0.0, 0.0, 1.0],  # Corner
            [0.0, 1.0, 0.0],  # Corner
            [0.0, 1.0, 1.0],  # Corner
            [1.0, 0.0, 0.0],  # Corner
            [1.0, 0.0, 1.0],  # Corner
            [1.0, 1.0, 0.0],  # Corner
            [1.0, 1.0, 1.0],  # Corner
            [0.5, 0.0, 0.5],  # Edge
            [0.5, 1.0, 0.5],  # Edge
            [0.0, 0.5, 0.5],  # Edge
            [1.0, 0.5, 0.5],  # Edge
            [0.5, 0.5, 0.0],  # Edge
        ])
        # Add small random perturbations to avoid degenerate cases
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 14: Improved optimization-focused initialization
    def get_optimization_focused_initial():
        # Start with a configuration that's likely to be good for optimization
        # This uses a combination of geometric principles and heuristics
        points = np.zeros((n, 3))
        
        # Place points in a structured way that maximizes spread
        # Use a grid-like approach with some randomness
        grid_size = 4
        count = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if count < n:
                        points[count] = [i/(grid_size-1), j/(grid_size-1), k/(grid_size-1)]
                        count += 1
                        
        # Add some random points to diversify
        if count < n:
            extra_points = np.random.rand(n - count, 3)
            points[count:] = extra_points
            
        # Apply slight perturbations to improve distribution
        points += np.random.normal(0, 0.02, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    # Try multiple initialization strategies and pick the best
    initial_strategies = [
        get_icosahedron_initial,
        get_fibonacci_initial,
        get_octa_tetra_initial,
        get_improved_icosahedron_initial,
        get_random_initial,
        get_sphere_packing_initial,
        get_dodecahedron_initial,
        get_hybrid_initial,
        get_optimized_geometric_initial,
        get_cube_based_initial,
        get_known_good_initial,
        get_spherical_arrangement,
        get_better_initial,
        get_optimization_focused_initial
    ]
    
    # Generate initial configurations
    initial_configs = []
    for strategy in initial_strategies:
        try:
            config = strategy()
            # Ensure all points are within [0,1]^3
            config = np.clip(config, 0, 1)
            initial_configs.append(config.copy())
        except Exception:
            continue
    
    # Define objective function with improved numerical stability
    def robust_objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Calculate pairwise distances using efficient computation
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero and handle edge cases
        if d_max <= 1e-15:
            return float('inf')
            
        # Use log transformation to avoid numerical issues with very small ratios
        ratio = d_min / d_max
        
        # Add penalty for very small distances to encourage better spread
        if d_min < 1e-8:
            return float('inf')
        
        # Return negative ratio (since we want to maximize)
        return -ratio
    
    # Improved optimization with multiple restarts and better convergence criteria
    def optimize_with_strategy(starting_points, max_restarts=5):
        best_ratio = -float('inf')
        best_points = starting_points.copy()
        
        # Multiple restarts with different perturbations
        for restart in range(max_restarts):
            # Create perturbed version
            perturbed = starting_points.copy()
            
            # Apply different perturbation patterns
            if restart == 0:
                # No perturbation for first run
                pass
            elif restart == 1:
                # Medium perturbations
                perturbed += np.random.normal(0, 0.005, perturbed.shape)
            elif restart == 2:
                # Larger perturbations
                perturbed += np.random.normal(0, 0.01, perturbed.shape)
            else:
                # Even larger perturbations for later restarts
                perturbed += np.random.normal(0, 0.02, perturbed.shape)
            
            # Ensure within bounds
            perturbed = np.clip(perturbed, 0, 1)
            
            try:
                # Flatten points for optimization
                x0 = perturbed.flatten()
                
                # Set up bounds for all coordinates [0,1]
                bounds = [(0, 1) for _ in range(n * 3)]
                
                # Optimization options - increased iterations and better tolerances
                options = {'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
                
                # Try multiple optimization methods for better results
                # Prioritize methods that work well for this type of problem
                methods = ['L-BFGS-B', 'TNC', 'SLSQP']
                
                for method in methods:
                    try:
                        result = minimize(
                            robust_objective,
                            x0,
                            method=method,
                            bounds=bounds,
                            options=options,
                            tol=1e-14
                        )
                        
                        # Extract results and compute actual ratio
                        final_points = result.x.reshape(-1, 3)
                        final_distances = pdist(final_points)
                        d_min = np.min(final_distances)
                        d_max = np.max(final_distances)
                        
                        if d_max > 1e-15:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = final_points.copy()
                                
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        return best_ratio, best_points
    
    # Run optimizations in parallel for better performance
    # Reduce parallel jobs to avoid memory issues
    results = Parallel(n_jobs=2, timeout=55)(
        delayed(optimize_with_strategy)(init_config) 
        for init_config in initial_configs
    )
    
    # Find best result among all strategies
    best_ratio = -float('inf')
    best_points = None
    
    for ratio, points_result in results:
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points_result.copy()
    
    # Additional fine-tuning with local search using different optimization methods
    if best_points is not None:
        # Try one more round of optimization with the best found solution
        try:
            # Apply a more aggressive optimization approach with multiple methods
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(n * 3)]
            options = {'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-15}
            
            # Try different optimization methods for better convergence
            methods = ['L-BFGS-B', 'TNC', 'SLSQP']
            for method in methods:
                try:
                    result = minimize(
                        robust_objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-15
                    )
                    
                    final_points = result.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-15:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            break  # Found better solution, stop trying other methods
                            
                except Exception:
                    continue
                    
        except Exception:
            pass
    
    # Final cleanup to ensure all points are within bounds
    if best_points is not None:
        best_points = np.clip(best_points, 0, 1)
    
    # Return the best configuration found
    return best_points if best_points is not None else np.random.rand(14, 3)


# EVOLVE-BLOCK-END
