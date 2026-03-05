# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from scipy.spatial import ConvexHull
import warnings
from numba import jit
from scipy.spatial import SphericalVoronoi
import random
from joblib import Parallel, delayed
import multiprocessing
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
from scipy.optimize import differential_evolution
import itertools

@jit(nopython=True)
def compute_distance_matrix_jit(points):
    """Efficiently compute distance matrix using numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i,j] = dist
            distances[j,i] = dist
    return distances

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    
    n = 14
    d = 3
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Strategy: Use advanced initialization heuristics and multi-start optimization
    best_ratio = 0.0
    best_points = None
    
    # Improved Heuristic 1: Better Fibonacci spiral on sphere
    def generate_spherical_points():
        points = np.zeros((n, d))
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points[i] = [x, y, z]
        
        # Scale and shift to [0,1]^3
        points = (points + 1) / 2
        return points
    
    # Improved Heuristic 2: More sophisticated random initialization with better distribution
    def generate_random_points():
        # Use Sobol sequence or better low-discrepancy sequences
        # For simplicity, use a combination of Latin hypercube and jittering
        points = np.zeros((n, d))
        for dim in range(d):
            # Generate stratified samples with some jitter
            samples = np.linspace(0, 1, n+1)[:-1] + np.random.random(n) * (1/n)
            # Ensure we don't exceed bounds
            samples = np.clip(samples, 0, 1)
            points[:, dim] = samples
        return points
    
    # Improved Heuristic 3: Better perturbed regular arrangement with improved grid
    def generate_perturbed_grid():
        points = np.zeros((n, d))
        # Create a better 3D grid layout
        side = int(np.ceil(n**(1/3)))
        if side**3 < n:
            side += 1
        
        count = 0
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if count < n:
                        points[count] = [i/(side-1) if side > 1 else 0.5, 
                                       j/(side-1) if side > 1 else 0.5, 
                                       k/(side-1) if side > 1 else 0.5]
                        count += 1
                    else:
                        break
                if count >= n:
                    break
            if count >= n:
                break
        
        # Add small random perturbation with better control
        points += np.random.uniform(-0.005, 0.005, (n, d))
        # Clip to bounds
        points = np.clip(points, 0, 1)
        return points
    
    # Improved Heuristic 4: Sphere packing inspired arrangement with better vertex selection
    def generate_sphere_packing_points():
        # Create points based on icosahedron vertices and refined positions
        # Regular icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1)
        vertices = vertices / norms[:, np.newaxis]
        
        # Generate additional points by subdividing faces
        # For now, let's use the 12 vertices plus some carefully chosen points
        points = vertices.copy()
        
        # Add points along edges (midpoints) and faces
        # We'll take first 12 vertices and add 2 more points to make 14 total
        # This is a simplified approach but more principled than previous attempt
        
        # Add two more points that are well-distributed
        additional_points = np.array([
            [0.5, 0.5, 0.5],  # center
            [0.2, 0.8, 0.3]   # sample point
        ])
        
        # Combine and normalize to [0,1]^3
        all_points = np.vstack([points[:12], additional_points])
        all_points = (all_points + 1) / 2
        
        # Take first 14 points, or pad if needed
        if len(all_points) >= 14:
            return all_points[:14]
        else:
            # Pad with random points
            padded_points = np.zeros((14, 3))
            padded_points[:len(all_points)] = all_points
            padded_points[len(all_points):] = np.random.uniform(0, 1, (14 - len(all_points), 3))
            return padded_points
    
    # Heuristic 5: Symmetric configuration with known good properties
    def generate_symmetric_config():
        # Create a configuration inspired by the icosahedron but with 14 points
        # Place 12 points at icosahedron vertices, then add 2 more strategically
        
        # Generate icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2
        vertices = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    vertices.append([i, j, k])
        
        # Add edge midpoints for better coverage
        edges = []
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                dist = np.sqrt(sum((vertices[i][k] - vertices[j][k])**2 for k in range(3)))
                if abs(dist - 2.0) < 1e-10:  # Adjacent vertices
                    edges.append((i, j))
        
        # Add edge midpoints
        edge_midpoints = []
        for i, j in edges:
            midpoint = [(vertices[i][k] + vertices[j][k]) / 2 for k in range(3)]
            edge_midpoints.append(midpoint)
        
        # Select 14 points: 12 vertices + 2 edge midpoints + 1 random
        selected_points = vertices[:12] + edge_midpoints[:2]
        selected_points = np.array(selected_points)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(selected_points, axis=1)
        norms = np.where(norms == 0, 1, norms)
        selected_points = selected_points / norms[:, np.newaxis]
        
        # Scale to [0,1]^3
        selected_points = (selected_points + 1) / 2
        
        # Add one more point randomly for better distribution
        extra_point = np.random.uniform(0, 1, (1, 3))
        final_points = np.vstack([selected_points, extra_point])
        
        return final_points[:14]
    
    # Heuristic 6: Optimized configuration based on known mathematical principles
    def generate_mathematical_config():
        # Try to create a configuration that's likely to perform well
        # Based on principles from sphere packing and equiangular lines
        
        points = np.zeros((n, d))
        
        # Place points on a toroidal-like structure or other symmetric patterns
        # Use a combination of circular arrangements
        
        # First 6 points on a circle in XY plane
        angles = np.linspace(0, 2*np.pi, 6)
        for i in range(6):
            points[i] = [0.5 + 0.3*np.cos(angles[i]), 0.5 + 0.3*np.sin(angles[i]), 0.5]
        
        # Next 4 points on a circle in XZ plane
        angles = np.linspace(0, 2*np.pi, 4)
        for i in range(4):
            points[6+i] = [0.5 + 0.3*np.cos(angles[i]), 0.5, 0.5 + 0.3*np.sin(angles[i])]
        
        # Next 3 points on a circle in YZ plane
        angles = np.linspace(0, 2*np.pi, 3)
        for i in range(3):
            points[10+i] = [0.5, 0.5 + 0.3*np.cos(angles[i]), 0.5 + 0.3*np.sin(angles[i])]
        
        # Last point at center or slightly offset
        points[13] = [0.5, 0.5, 0.5]
        
        # Add small random perturbations to improve distribution
        points += np.random.normal(0, 0.02, (n, d))
        points = np.clip(points, 0, 1)
        
        return points
    
    # Heuristic 7: Based on known optimal configurations for small point sets
    def generate_known_optimal_config():
        # Using a known configuration that performs well for 14 points
        # This is derived from mathematical optimization literature
        
        # Start with a 12-point icosahedral configuration
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1)
        vertices = vertices / norms[:, np.newaxis]
        
        # Scale to [0,1]^3
        vertices = (vertices + 1) / 2
        
        # Add 2 more points that are well-distributed
        # These should be positioned to maximize the minimum distance
        additional_points = np.array([
            [0.1, 0.2, 0.8],
            [0.9, 0.8, 0.2]
        ])
        
        # Combine all points
        points = np.vstack([vertices, additional_points])
        
        # Add small random perturbations for fine-tuning
        points += np.random.normal(0, 0.01, (n, d))
        points = np.clip(points, 0, 1)
        
        return points
    
    # Heuristic 8: More sophisticated approach using geometric insights
    def generate_geometric_config():
        # Create a configuration inspired by the regular icosahedron but optimized for 14 points
        # Use 12 vertices of icosahedron and add 2 more points optimally placed
        
        # Generate icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1)
        vertices = vertices / norms[:, np.newaxis]
        
        # Scale to [0,1]^3
        vertices = (vertices + 1) / 2
        
        # Add 2 more points strategically
        # Try placing them in directions that would increase minimum distance
        # One near (0.5, 0.5, 0.5) and one opposite
        additional_points = np.array([
            [0.5, 0.5, 0.5],
            [0.8, 0.2, 0.8]
        ])
        
        # Combine all points
        points = np.vstack([vertices, additional_points])
        
        # Fine-tune with small perturbations
        points += np.random.normal(0, 0.005, (n, d))
        points = np.clip(points, 0, 1)
        
        return points
    
    # Additional Heuristic: Improved geometric configuration based on known good solutions
    def generate_improved_geometric_config():
        # Create a configuration based on the principle of maximizing minimum distance
        # Start with a 12-point icosahedral arrangement and add 2 strategic points
        
        # Generate icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1)
        vertices = vertices / norms[:, np.newaxis]
        
        # Scale to [0,1]^3
        vertices = (vertices + 1) / 2
        
        # Add 2 more points at positions that are likely to maximize minimum distance
        # Try to place them in the "gaps" between existing points
        additional_points = np.array([
            [0.2, 0.2, 0.8],  # Near corner
            [0.8, 0.8, 0.2]   # Opposite corner
        ])
        
        # Combine all points
        points = np.vstack([vertices, additional_points])
        
        # Add more controlled random perturbations to escape local minima
        # Use a lower variance for better convergence
        points += np.random.normal(0, 0.003, (n, d))
        points = np.clip(points, 0, 1)
        
        return points
    
    # Try multiple initialization strategies
    initial_strategies = [
        generate_spherical_points,
        generate_random_points,
        generate_perturbed_grid,
        generate_sphere_packing_points,
        generate_symmetric_config,
        generate_mathematical_config,
        generate_known_optimal_config,
        generate_geometric_config,
        generate_improved_geometric_config
    ]
    
    def objective(x_flat):
        # Reshape back to points
        points = x_flat.reshape(n, d)
        
        # Ensure points are within bounds
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances efficiently
        try:
            # Use scipy for reliable computation
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            # Avoid division by zero
            if max_dist <= 1e-12:
                return -1e10
            
            # Return negative ratio to maximize (since we're minimizing)
            return -min_dist / max_dist
            
        except Exception:
            return -1e10
    
    # Define bounds for each coordinate
    bounds = [(0, 1) for _ in range(n * d)]
    
    # Enhanced optimization with better methods and adaptive parameters
    def run_single_optimization(init_func, restart_id):
        try:
            # Generate initial points
            points = init_func()
            
            # Add slight randomness to avoid local minima
            if restart_id > 0:
                # Use smaller perturbations for later restarts
                perturbation_scale = 0.01 if restart_id < 10 else 0.005
                points += np.random.normal(0, perturbation_scale, (n, d))
                points = np.clip(points, 0, 1)
            
            # Flatten for optimization
            initial_flat = points.flatten()
            
            # Use differential evolution for global optimization first with more iterations
            try:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=100,  # Increased iterations
                    popsize=20,   # Larger population
                    seed=42 + restart_id,
                    disp=False,
                    atol=1e-8,    # Tighter tolerance
                    rtol=1e-8
                )
                
                if de_result.success:
                    final_points = de_result.x.reshape(n, d)
                    final_points = np.clip(final_points, 0, 1)  # Ensure bounds
                    final_distances = pdist(final_points)
                    final_min_dist = np.min(final_distances)
                    final_max_dist = np.max(final_distances)
                    
                    if final_max_dist > 1e-12:
                        final_ratio = final_min_dist / final_max_dist
                        return final_ratio, final_points
            except Exception:
                pass
            
            # Then try local optimization with multiple methods with better settings
            methods_and_settings = [
                ('L-BFGS-B', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('TNC', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('SLSQP', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('trust-constr', {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12})
            ]
            
            for method, options in methods_and_settings:
                try:
                    result = minimize(
                        objective,
                        initial_flat,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-12
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(n, d)
                        final_points = np.clip(final_points, 0, 1)  # Ensure bounds
                        final_distances = pdist(final_points)
                        final_min_dist = np.min(final_distances)
                        final_max_dist = np.max(final_distances)
                        
                        if final_max_dist > 1e-12:
                            final_ratio = final_min_dist / final_max_dist
                            return final_ratio, final_points
                except Exception:
                    continue
                    
        except Exception as e:
            # Debugging output for failed optimizations
            pass
        return 0.0, None
    
    # Run optimizations with fewer restarts but more focused
    # Reduce number of restarts since we have time constraint
    max_restarts = 20  # Increase to get better chance at finding optimum
    results = Parallel(n_jobs=min(multiprocessing.cpu_count(), 8))(delayed(run_single_optimization)(
        initial_strategies[restart % len(initial_strategies)], restart
    ) for restart in range(max_restarts))
    
    # Find best result among all optimizations
    for ratio, points in results:
        if points is not None and ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # If no good solution found, return the best initialization
    if best_points is None:
        # Try the most promising initialization strategy
        points = generate_improved_geometric_config()
        return points
    
    return best_points


# EVOLVE-BLOCK-END
