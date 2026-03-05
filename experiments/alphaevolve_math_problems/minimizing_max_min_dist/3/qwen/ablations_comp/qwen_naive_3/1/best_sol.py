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
from scipy.spatial.transform import Rotation as R

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
    
    # Strategy 1: Improved icosahedral-based configuration
    def get_icosahedron_plus():
        # Start with icosahedron vertices (normalized to unit sphere)
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        
        # Add additional strategic points
        # Place points near edges and faces for better distribution
        edge_points = []
        # Add points along edges
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                # Check if they form an edge (distance ≈ sqrt(2))
                dist = np.linalg.norm(vertices[i] - vertices[j])
                if abs(dist - np.sqrt(2)) < 0.2:  # approximate edge length
                    # Add midpoint plus small perturbation
                    midpoint = (vertices[i] + vertices[j]) / 2
                    edge_points.append(midpoint)
        
        # Fill remaining points to reach 14
        if len(edge_points) < 2:
            # If not enough edge points, create additional ones
            additional = np.random.randn(2, 3)
            additional = additional / np.linalg.norm(additional, axis=1, keepdims=True)
            edge_points.extend(additional)
        
        # Combine and normalize to [0,1]^3
        points = np.vstack([vertices[:10], edge_points[:4]])  # Keep 10 original + 4 extra
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        points = (points + 1) / 2
        return points
    
    # Strategy 2: Fibonacci-based spiral on sphere with better uniformity
    def get_fibonacci_sphere_initial():
        points = np.zeros((n, 3))
        
        # More sophisticated Fibonacci approach with better spacing
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n)
        
        # Distribute points more uniformly
        theta = np.arccos(1 - 2 * indices / (n - 1))  # Polar angle
        phi = np.mod(indices * (4 * np.pi) / golden_ratio, 2 * np.pi)  # Azimuthal angle
        
        # Convert spherical to Cartesian coordinates
        points[:, 0] = np.sin(theta) * np.cos(phi)  # x
        points[:, 1] = np.sin(theta) * np.sin(phi)  # y  
        points[:, 2] = np.cos(theta)                # z
        
        # Scale and shift to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Strategy 3: Optimized octahedral arrangement with symmetry breaking
    def get_octahedral_symmetric():
        # Octahedron vertices + additional points
        base_points = np.array([
            [0.5, 0.5, 1.0],   # top
            [0.5, 0.5, 0.0],   # bottom
            [1.0, 0.5, 0.5],   # right
            [0.0, 0.5, 0.5],   # left
            [0.5, 1.0, 0.5],   # front
            [0.5, 0.0, 0.5],   # back
        ])
        
        # Add symmetrically placed points to increase uniformity
        # Add points along diagonals of the unit cube
        diagonal_points = np.array([
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.25, 0.25, 0.75],
            [0.25, 0.75, 0.25],
            [0.75, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ])
        
        # Combine and take first 14 points
        points = np.vstack([base_points, diagonal_points])
        points = points[:14]
        return points
    
    # Strategy 4: Random with better distribution using Latin Hypercube sampling
    def get_latin_hypercube_initial():
        # Generate points using Latin Hypercube sampling for better distribution
        points = np.zeros((n, 3))
        for i in range(3):  # for each dimension
            # Create intervals
            intervals = np.linspace(0, 1, n + 1)
            # Sample uniformly within each interval
            samples = np.random.uniform(intervals[:-1], intervals[1:])
            # Shuffle samples to avoid correlation
            np.random.shuffle(samples)
            points[:, i] = samples
        return points
    
    # Strategy 5: Hybrid of known good configurations
    def get_hybrid_initial():
        # Mix of different strategies
        points = np.zeros((n, 3))
        
        # 1st quarter: fibonacci points
        fib_points = get_fibonacci_sphere_initial()
        points[:3, :] = fib_points[:3, :]
        
        # 2nd quarter: octahedral points
        oct_points = get_octahedral_symmetric()
        points[3:7, :] = oct_points[3:7, :]
        
        # 3rd quarter: random with good distribution
        latin_points = get_latin_hypercube_initial()
        points[7:11, :] = latin_points[7:11, :]
        
        # 4th quarter: additional structured points
        additional = np.random.rand(3, 3)
        points[11:, :] = additional
        
        return points
    
    # Try multiple initialization strategies and pick the best
    initial_strategies = [
        get_icosahedron_plus,
        get_fibonacci_sphere_initial,
        get_octahedral_symmetric,
        get_latin_hypercube_initial,
        get_hybrid_initial,
    ]
    
    # Generate initial configurations
    initial_configs = []
    for strategy in initial_strategies:
        try:
            config = strategy()
            # Ensure all points are within [0,1]^3
            config = np.clip(config, 0, 1)
            initial_configs.append(config.copy())
        except Exception as e:
            continue
    
    # Define improved objective function with better numerical stability
    def improved_objective(x_flat):
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Calculate pairwise distances using more stable computation
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero and handle edge cases
        if d_max <= 1e-15 or d_min <= 1e-15:
            return float('inf')
            
        # Compute ratio
        ratio = d_min / d_max
        
        # Add penalty for very small distances to encourage better spread
        # This helps prevent points from clustering together
        penalty = 0.0
        if d_min < 1e-8:
            return float('inf')
        elif d_min < 0.01:
            penalty = 1e15 * (1.0 / (d_min + 1e-12))
        
        # Return negative ratio (since we want to maximize) 
        # with penalty for very small distances
        return -ratio + penalty
    
    # Better optimization with adaptive restarts and improved convergence
    def optimize_with_strategy(starting_points, max_restarts=3):
        best_ratio = -float('inf')
        best_points = starting_points.copy()
        
        # Multiple restarts with different perturbation levels
        for restart in range(max_restarts):
            # Create perturbed version with adaptive perturbation
            perturbed = starting_points.copy()
            
            # Vary perturbation intensity based on restart number
            if restart == 0:
                # No perturbation for first run (baseline)
                pass
            elif restart == 1:
                # Small perturbations
                perturbed += np.random.normal(0, 0.005, perturbed.shape)
            else:
                # Larger perturbations for later restarts
                perturbed += np.random.normal(0, 0.01, perturbed.shape)
            
            # Ensure within bounds
            perturbed = np.clip(perturbed, 0, 1)
            
            try:
                # Flatten points for optimization
                x0 = perturbed.flatten()
                
                # Set up bounds for all coordinates [0,1]
                bounds = [(0, 1) for _ in range(n * 3)]
                
                # Optimization options
                options = {'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
                
                # Try different optimization methods
                # Method 1: L-BFGS-B (good for smooth functions)
                try:
                    result1 = minimize(
                        improved_objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options=options,
                        tol=1e-12
                    )
                    
                    # Extract results and compute actual ratio
                    final_points = result1.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-15 and d_min > 1e-15:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
                except Exception:
                    pass
                    
                # Method 2: SLSQP for constrained optimization
                try:
                    result2 = minimize(
                        improved_objective,
                        x0,
                        method='SLSQP',
                        bounds=bounds,
                        options=options,
                        tol=1e-12
                    )
                    
                    # Extract results and compute actual ratio
                    final_points = result2.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-15 and d_min > 1e-15:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            
                except Exception:
                    pass
                        
            except Exception:
                continue
        
        return best_ratio, best_points
    
    # Run optimizations in parallel for better performance
    results = Parallel(n_jobs=2)(
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
    
    # Additional fine-tuning with local search
    if best_points is not None:
        # Try one more round of optimization with the best found solution
        try:
            # Apply a more aggressive optimization approach
            x0 = best_points.flatten()
            bounds = [(0, 1) for _ in range(n * 3)]
            options = {'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            
            # Use multiple methods for better chance of finding good solution
            methods = ['L-BFGS-B', 'SLSQP']
            for method in methods:
                try:
                    result = minimize(
                        improved_objective,
                        x0,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-14
                    )
                    
                    final_points = result.x.reshape(-1, 3)
                    final_distances = pdist(final_points)
                    d_min = np.min(final_distances)
                    d_max = np.max(final_distances)
                    
                    if d_max > 1e-15 and d_min > 1e-15:
                        ratio = d_min / d_max
                        if ratio > best_ratio:
                            best_points = final_points.copy()
                            best_ratio = ratio
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
