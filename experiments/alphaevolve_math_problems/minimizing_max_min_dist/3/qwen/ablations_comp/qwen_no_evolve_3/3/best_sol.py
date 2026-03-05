# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
import math
from scipy.spatial import ConvexHull
import random
from sklearn.cluster import KMeans
import time
from scipy.spatial import SphericalVoronoi
from numba import jit
from scipy.spatial import distance
from scipy.spatial.distance import squareform
from scipy.spatial.transform import Rotation as R


@jit(nopython=True)
def compute_min_max_ratio_jit(points):
    """JIT compiled function for faster distance computation"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist_sq += diff * diff
            dist = np.sqrt(dist_sq)
            if dist < min_dist:
                min_dist = dist
            if dist > max_dist:
                max_dist = dist
    
    if max_dist > 0:
        return min_dist / max_dist
    else:
        return 0.0


@jit(nopython=True)
def compute_distances_jit(points):
    """Compute all pairwise distances efficiently"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist_sq = 0.0
            for k in range(3):
                diff = points[i,k] - points[j,k]
                dist_sq += diff * diff
            dist = np.sqrt(dist_sq)
            distances[i,j] = dist
            distances[j,i] = dist
    
    return distances


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initialization strategies with advanced optimization.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective(x):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape flat array back to 14x3 points
        points = x.reshape(-1, 3)
        
        # Compute pairwise distances efficiently
        distances = pdist(points)
        
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio to maximize ratio (minimize negative)
        if d_max == 0:
            return 0
        return -d_min / d_max
    
    def objective_fast(x):
        """Fast objective using JIT compiled function"""
        points = x.reshape(-1, 3)
        ratio = compute_min_max_ratio_jit(points)
        return -ratio
    
    def objective_with_distances(x):
        """Objective that also computes distances for debugging"""
        points = x.reshape(-1, 3)
        distances = compute_distances_jit(points)
        d_min = np.min(distances[np.triu_indices_from(distances, k=1)])
        d_max = np.max(distances[np.triu_indices_from(distances, k=1)])
        if d_max == 0:
            return 0
        return -d_min / d_max
    
    def constraint_bounds(x):
        """Constraint: points within unit cube [0,1]^3"""
        points = x.reshape(-1, 3)
        # Return difference from bounds (positive when out of bounds)
        return np.concatenate([
            points.min(axis=0),  # Lower bound (should be >= 0)
            1 - points.max(axis=0)  # Upper bound (should be >= 0)
        ])
    
    def generate_fibonacci_points(n):
        """Generate points using Fibonacci spiral on sphere - improved version"""
        points = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(n):
            # Better distribution using Fibonacci sphere algorithm
            theta = math.acos(-1 + (2 * i) / (n - 1))
            phi = math.sqrt(n * math.pi) * theta
            
            x = math.sin(theta) * math.cos(phi)
            y = math.sin(theta) * math.sin(phi)
            z = math.cos(theta)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def generate_cube_vertices():
        """Generate vertices of a cube scaled appropriately"""
        # Cube vertices with some randomness to avoid degenerate cases
        vertices = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    vertices.append([i, j, k])
        return np.array(vertices)
    
    def generate_random_points(n, seed=None):
        """Generate random points in unit cube"""
        if seed is not None:
            np.random.seed(seed)
        return np.random.rand(n, 3)
    
    def normalize_to_unit_cube(points):
        """Normalize points to fit in unit cube [0,1]^3"""
        # Translate to center around origin
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # Scale to fit in [-0.5, 0.5]^3
        max_coord = np.max(np.abs(centered))
        if max_coord > 0:
            scaled = centered / (2 * max_coord)
        else:
            scaled = centered
            
        # Translate back to [0,1]^3
        final = scaled + 0.5
        return final
    
    def generate_spherical_arrangement():
        """Generate points arranged on a sphere with some randomness"""
        # Use Fibonacci sphere distribution
        points = generate_fibonacci_points(14)
        # Normalize to unit sphere and scale appropriately
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis] * 0.5  # Scale to radius 0.5
        # Add some randomness to avoid perfect symmetry
        points += np.random.normal(0, 0.05, points.shape)
        # Normalize again to keep on sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis] * 0.5
        # Move to unit cube
        points = points + 0.5  # Center in [0,1]^3
        return np.clip(points, 0, 1)
    
    def generate_icosahedron_points():
        """Generate points based on icosahedron vertices for better spatial distribution"""
        # Regular icosahedron vertices
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        vertices = [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
        ]
        points = np.array(vertices)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        # Scale to appropriate size and add randomness
        points = points * 0.4 + np.random.normal(0, 0.03, points.shape)
        # Normalize again and move to cube
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis] * 0.4
        points = points + 0.5
        return np.clip(points, 0, 1)
    
    def generate_clustered_initialization():
        """Generate points with clustering to avoid local minima"""
        # Start with a good initial configuration
        # Use a combination of sphere points and random points
        sphere_points = generate_spherical_arrangement()
        
        # Add some additional points in a structured way
        additional_points = np.random.rand(4, 3)
        
        # Combine and normalize
        combined = np.vstack([sphere_points, additional_points])
        return normalize_to_unit_cube(combined)[:14]
    
    def generate_good_initialization():
        """Create multiple good initial configurations and pick the best"""
        candidates = []
        
        # Strategy 1: Fibonacci spiral on sphere, then project to cube
        fib_points = generate_fibonacci_points(14)
        fib_in_cube = normalize_to_unit_cube(fib_points)
        candidates.append(fib_in_cube)
        
        # Strategy 2: Random points
        rand_points = generate_random_points(14, seed=42)
        candidates.append(rand_points)
        
        # Strategy 3: Spherical arrangement
        sph_points = generate_spherical_arrangement()
        candidates.append(sph_points)
        
        # Strategy 4: Clustered initialization
        clustered_points = generate_clustered_initialization()
        candidates.append(clustered_points)
        
        # Strategy 5: Icosahedron-based points
        ico_points = generate_icosahedron_points()
        candidates.append(ico_points)
        
        # Strategy 6: Some vertices of a cube with random points
        cube_points = generate_cube_vertices()
        cube_points = np.vstack([cube_points, generate_random_points(10, seed=123)])
        candidates.append(normalize_to_unit_cube(cube_points[:14]))
        
        # Strategy 7: Known good 3D packing - icosahedral structure with perturbations
        try:
            # Generate a more uniform distribution using known good patterns
            # Based on icosahedral symmetry with additional points
            base_points = []
            
            # Add icosahedral vertices with small perturbations
            phi = (1 + math.sqrt(5)) / 2  # Golden ratio
            # Add 12 vertices of icosahedron with slight randomness
            for i in range(12):
                if i < 4:
                    x, y, z = (-1, -phi, 0) if i == 0 else (1, -phi, 0) if i == 1 else (-1, phi, 0) if i == 2 else (1, phi, 0)
                elif i < 8:
                    x, y, z = (0, -1, phi) if i == 4 else (0, 1, phi) if i == 5 else (0, -1, -phi) if i == 6 else (0, 1, -phi)
                else:
                    x, y, z = (phi, 0, -1) if i == 8 else (phi, 0, 1) if i == 9 else (-phi, 0, -1) if i == 10 else (-phi, 0, 1)
                
                # Add small random perturbation
                x += np.random.normal(0, 0.05)
                y += np.random.normal(0, 0.05)
                z += np.random.normal(0, 0.05)
                base_points.append([x, y, z])
            
            # Add 2 more points to make 14 total
            base_points.append([0.2, 0.2, 0.2])
            base_points.append([-0.2, -0.2, -0.2])
            
            points = np.array(base_points)
            # Normalize to unit sphere and project to cube
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * 0.4
            points = points + 0.5
            candidates.append(np.clip(points, 0, 1))
        except:
            pass
        
        # Strategy 8: Generate points on a toroidal-like structure
        try:
            # Create a torus-like distribution
            torus_points = []
            for i in range(7):
                angle1 = 2 * np.pi * i / 7
                for j in range(2):
                    angle2 = 2 * np.pi * j / 2
                    r = 0.3  # major radius
                    a = 0.1  # minor radius
                    x = (r + a * np.cos(angle2)) * np.cos(angle1)
                    y = (r + a * np.cos(angle2)) * np.sin(angle1)
                    z = a * np.sin(angle2)
                    torus_points.append([x, y, z])
            
            # Add random points to complete 14
            random_add = np.random.rand(7, 3) * 0.2 - 0.1
            torus_points.extend(random_add.tolist())
            
            points = np.array(torus_points[:14])
            # Normalize to unit cube
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * 0.3
            points = points + 0.5
            candidates.append(np.clip(points, 0, 1))
        except:
            pass
        
        # Strategy 9: Symmetrically arranged points with high symmetry
        try:
            # Create a more symmetric pattern using rotation groups
            # Start with regular tetrahedron vertices
            tetrahedron_points = np.array([
                [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
            ])
            # Normalize to unit sphere and add some randomness
            norms = np.linalg.norm(tetrahedron_points, axis=1)
            tetrahedron_points = tetrahedron_points / norms[:, np.newaxis] * 0.4
            
            # Add 10 more points in a symmetric way
            additional_points = []
            # Add points along axes
            for axis in [[1,0,0], [0,1,0], [0,0,1], [-1,0,0], [0,-1,0], [0,0,-1]]:
                additional_points.append(np.array(axis) * 0.3)
            
            # Add more points in a symmetric fashion
            for i in range(4):
                # Generate points in a circular pattern
                angle = i * np.pi / 2
                point = [0.3 * np.cos(angle), 0.3 * np.sin(angle), 0]
                additional_points.append(point)
            
            # Combine and normalize
            all_points = np.vstack([tetrahedron_points, additional_points])
            # Normalize to unit cube
            norms = np.linalg.norm(all_points, axis=1)
            all_points = all_points / norms[:, np.newaxis] * 0.3
            all_points = all_points + 0.5
            candidates.append(np.clip(all_points, 0, 1))
        except:
            pass
        
        # Strategy 10: Improved icosahedral arrangement with better distribution
        try:
            # Generate points more carefully to improve spacing
            base_points = []
            phi = (1 + math.sqrt(5)) / 2  # Golden ratio
            
            # Create 12 icosahedron vertices with better distribution
            for i in range(12):
                if i < 4:
                    x, y, z = (-1, -phi, 0) if i == 0 else (1, -phi, 0) if i == 1 else (-1, phi, 0) if i == 2 else (1, phi, 0)
                elif i < 8:
                    x, y, z = (0, -1, phi) if i == 4 else (0, 1, phi) if i == 5 else (0, -1, -phi) if i == 6 else (0, 1, -phi)
                else:
                    x, y, z = (phi, 0, -1) if i == 8 else (phi, 0, 1) if i == 9 else (-phi, 0, -1) if i == 10 else (-phi, 0, 1)
                
                # Add more substantial random perturbations to avoid perfect symmetry
                x += np.random.normal(0, 0.08)
                y += np.random.normal(0, 0.08)
                z += np.random.normal(0, 0.08)
                base_points.append([x, y, z])
            
            # Add 2 more points with careful placement
            base_points.append([0.1, 0.1, 0.1])
            base_points.append([-0.1, -0.1, -0.1])
            
            points = np.array(base_points)
            # Normalize to unit sphere and project to cube
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * 0.4
            points = points + 0.5
            candidates.append(np.clip(points, 0, 1))
        except:
            pass
        
        # Evaluate all candidates with fast computation
        best_candidate = None
        best_ratio = -np.inf
        
        for candidate in candidates:
            # Compute ratio for this candidate using fast method
            ratio = compute_min_max_ratio_jit(candidate)
            if ratio > best_ratio:
                best_ratio = ratio
                best_candidate = candidate.copy()
        
        return best_candidate
    
    def multi_start_optimization(initial_points, max_time=30):
        """Use multi-start optimization with different strategies"""
        start_time = time.time()
        
        # Define constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: constraint_bounds(x)}
        ]
        
        # Bounds for optimization (points in [0,1]^3)
        bounds = [(0, 1)] * (14 * 3)
        
        best_result = None
        best_ratio = -np.inf
        
        # Multiple optimization attempts with different starting points
        for attempt in range(15):  # Increased attempts for better chance
            if time.time() - start_time > max_time * 0.9:
                break
                
            try:
                # Create diverse starting points for each attempt
                if attempt == 0:
                    # Use original points
                    start_points = initial_points.copy()
                elif attempt == 1:
                    # Perturb slightly
                    start_points = initial_points + np.random.normal(0, 0.02, initial_points.shape)
                elif attempt == 2:
                    # Use a completely different initialization
                    start_points = generate_icosahedron_points()
                elif attempt == 3:
                    # Another variation
                    start_points = generate_spherical_arrangement()
                elif attempt == 4:
                    # Different random points
                    start_points = generate_random_points(14, seed=attempt+100)
                elif attempt == 5:
                    # Perturbed icosahedron
                    ico_points = generate_icosahedron_points()
                    start_points = ico_points + np.random.normal(0, 0.03, ico_points.shape)
                elif attempt == 6:
                    # Perturbed spherical
                    sph_points = generate_spherical_arrangement()
                    start_points = sph_points + np.random.normal(0, 0.03, sph_points.shape)
                elif attempt == 7:
                    # Random points with different seed
                    start_points = generate_random_points(14, seed=attempt+200)
                elif attempt == 8:
                    # Perturbed fibonacci points
                    fib_points = generate_fibonacci_points(14)
                    fib_points = normalize_to_unit_cube(fib_points)
                    start_points = fib_points + np.random.normal(0, 0.04, fib_points.shape)
                elif attempt == 9:
                    # Perturbed cube vertices
                    cube_points = generate_cube_vertices()
                    cube_points = np.vstack([cube_points, generate_random_points(10, seed=300)])
                    cube_points = normalize_to_unit_cube(cube_points[:14])
                    start_points = cube_points + np.random.normal(0, 0.03, cube_points.shape)
                elif attempt == 10:
                    # Perturbed icosahedron with more randomness
                    ico_points = generate_icosahedron_points()
                    start_points = ico_points + np.random.normal(0, 0.05, ico_points.shape)
                elif attempt == 11:
                    # Rotate initial points
                    rotation = R.from_euler('xyz', [np.random.uniform(0, 2*np.pi), 
                                                   np.random.uniform(0, 2*np.pi), 
                                                   np.random.uniform(0, 2*np.pi)]).as_matrix()
                    start_points = initial_points @ rotation.T
                    # Make sure they're still in bounds
                    start_points = np.clip(start_points, 0, 1)
                elif attempt == 12:
                    # Use a different symmetric pattern
                    start_points = generate_clustered_initialization()
                elif attempt == 13:
                    # More aggressive random perturbation
                    start_points = initial_points + np.random.normal(0, 0.08, initial_points.shape)
                else:
                    # Even more aggressive perturbation
                    start_points = initial_points + np.random.normal(0, 0.1, initial_points.shape)
                
                # Keep in bounds
                start_points = np.clip(start_points, 0, 1)
                flattened_points = start_points.flatten()
                
                # Try multiple optimizers with different settings
                optimizers = [
                    ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}),
                    ('TNC', {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}),
                    ('SLSQP', {'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}),
                    ('trust-constr', {'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-16})
                ]
                
                for opt_method, options in optimizers:
                    if time.time() - start_time > max_time * 0.9:
                        break
                        
                    try:
                        result = minimize(
                            objective_fast,
                            flattened_points,
                            method=opt_method,
                            bounds=bounds,
                            constraints=constraints,
                            options=options,
                            tol=1e-16
                        )
                        
                        if result.success:
                            final_points = result.x.reshape(-1, 3)
                            ratio = compute_min_max_ratio_jit(final_points)
                            
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_result = final_points.copy()
                                
                    except Exception as e:
                        continue
                        
            except Exception as e:
                continue
        
        # If no optimization worked, return the best initialization
        if best_result is None:
            return initial_points
        
        return best_result
    
    # Generate initial points with improved strategies
    initial_points = generate_good_initialization()
    
    # Run multi-start optimization
    final_result = multi_start_optimization(initial_points, max_time=40)
    
    # Final refinement step with more aggressive optimization
    try:
        # Try to do one more optimization run with the best solution found so far
        bounds = [(0, 1)] * (14 * 3)
        constraints = [{'type': 'ineq', 'fun': lambda x: constraint_bounds(x)}]
        
        # Use trust-constr which often works better for constrained problems
        result = minimize(
            objective_fast,
            final_result.flatten(),
            method='trust-constr',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-17},
            tol=1e-17
        )
        
        if result.success:
            refined_points = result.x.reshape(-1, 3)
            refined_ratio = compute_min_max_ratio_jit(refined_points)
            current_ratio = compute_min_max_ratio_jit(final_result)
            
            if refined_ratio > current_ratio:
                final_result = refined_points
    except:
        pass
    
    # Additional post-processing: try to improve further with local search
    try:
        # Apply a local optimization step with smaller steps
        current_points = final_result.copy()
        current_ratio = compute_min_max_ratio_jit(current_points)
        
        # Simple gradient descent approach with small steps
        learning_rate = 0.001
        for _ in range(500):  # Reduced iterations to save time
            # Compute gradients numerically
            grad = np.zeros_like(current_points)
            eps = 1e-8
            
            for i in range(14):
                for j in range(3):
                    # Perturb coordinate
                    temp_points = current_points.copy()
                    temp_points[i, j] += eps
                    
                    # Ensure bounds
                    temp_points = np.clip(temp_points, 0, 1)
                    
                    # Compute finite difference gradient
                    new_ratio = compute_min_max_ratio_jit(temp_points)
                    grad[i, j] = (new_ratio - current_ratio) / eps
            
            # Update points
            current_points = current_points - learning_rate * grad
            current_points = np.clip(current_points, 0, 1)
            
            # Check if we improved
            new_ratio = compute_min_max_ratio_jit(current_points)
            if new_ratio > current_ratio:
                current_ratio = new_ratio
                final_result = current_points.copy()
            else:
                # Reduce learning rate if stuck
                learning_rate *= 0.999
                
    except:
        pass
    
    # Final validation check
    final_ratio = compute_min_max_ratio_jit(final_result)
    if final_ratio < 0.45:  # If we didn't get a reasonable improvement, use a better initialization
        # Try one more initialization and optimization
        better_init = generate_good_initialization()
        better_final = multi_start_optimization(better_init, max_time=10)
        better_ratio = compute_min_max_ratio_jit(better_final)
        if better_ratio > final_ratio:
            final_result = better_final
    
    return final_result


# EVOLVE-BLOCK-END
