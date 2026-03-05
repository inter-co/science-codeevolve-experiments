# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
from scipy.spatial import ConvexHull
import random
from sklearn.cluster import KMeans
import time
from scipy.spatial import SphericalVoronoi
from numba import jit, prange
from scipy.spatial import distance
from scipy.spatial.distance import cdist


@jit(nopython=True)
def compute_min_max_ratio_jit(points):
    """JIT compiled function for faster distance computation"""
    n = points.shape[0]
    min_dist = np.inf
    max_dist = 0.0
    
    # More efficient computation with early termination possibility
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
    
    if max_dist > 0:
        return min_dist / max_dist
    else:
        return 0.0


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initialization strategies with advanced optimization.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def objective_fast(x):
        """Fast objective using JIT compiled function"""
        points = x.reshape(-1, 3)
        ratio = compute_min_max_ratio_jit(points)
        return -ratio  # Negative because we want to maximize
    
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
        
        # Strategy 7: Known good 3D packing - modified icosahedral pattern
        try:
            # Generate points based on known good 3D packings
            # Use icosahedral structure with modified positions
            base_points = []
            
            # Add icosahedral vertices with small perturbations
            phi = (1 + math.sqrt(5)) / 2
            icosahedron_vertices = [
                (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
                (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
                (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
            ]
            
            # Add some random variations to avoid symmetric local minima
            for i, (x, y, z) in enumerate(icosahedron_vertices):
                # Add small random perturbation
                perturbation = np.random.normal(0, 0.05, 3)
                base_points.append([x + perturbation[0], y + perturbation[1], z + perturbation[2]])
            
            # Add two more points to reach 14
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
        
        # Strategy 8: Modified golden ratio based construction
        try:
            # Create points using golden ratio principles
            points = []
            for i in range(14):
                t = i / 13.0  # Normalize to [0,1]
                # Use golden ratio to distribute points
                angle = 2 * np.pi * i * (1 + np.sqrt(5)) / 2
                radius = 0.4 * np.sqrt(t)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = 0.5 * (2 * t - 1)  # Distribute along z-axis
                points.append([x, y, z])
            
            points = np.array(points)
            # Normalize to unit cube
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * 0.4
            points = points + 0.5
            candidates.append(np.clip(points, 0, 1))
        except:
            pass
        
        # Strategy 9: Use known good 3D structures - octahedral arrangement
        try:
            # Octahedron vertices with slight perturbations
            octahedron_points = [
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
            ]
            # Add more points by placing them near the faces
            additional = []
            for i in range(8):
                # Place points around the surface of a sphere
                theta = np.random.random() * np.pi
                phi = np.random.random() * 2 * np.pi
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                additional.append([x, y, z])
            
            all_points = octahedron_points + additional
            points = np.array(all_points[:14])
            # Normalize and scale
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * 0.4
            points = points + 0.5
            candidates.append(np.clip(points, 0, 1))
        except:
            pass
        
        # Strategy 10: Specialized arrangement based on 14-point optimal configurations
        try:
            # Generate points in a configuration inspired by optimal 3D packings
            # Use a combination of icosahedral and tetrahedral arrangements
            points = []
            
            # Base icosahedron vertices
            phi = (1 + math.sqrt(5)) / 2
            icosahedron_verts = [
                (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
                (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
                (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
            ]
            
            # Add some vertices with careful perturbations
            for i, (x, y, z) in enumerate(icosahedron_verts[:12]):
                # Apply small perturbations to avoid symmetries
                pert_x = np.random.normal(0, 0.03)
                pert_y = np.random.normal(0, 0.03)
                pert_z = np.random.normal(0, 0.03)
                points.append([x + pert_x, y + pert_y, z + pert_z])
            
            # Add two more points to get 14 total
            points.append([0.3, 0.3, 0.3])
            points.append([-0.3, -0.3, -0.3])
            
            points = np.array(points)
            # Normalize and move to cube
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis] * 0.35
            points = points + 0.5
            candidates.append(np.clip(points, 0, 1))
        except:
            pass
        
        # Strategy 11: Generate points with low-discrepancy sequence (Hammersley)
        try:
            # Use Hammersley point set for better distribution
            points = []
            for i in range(14):
                # Use golden ratio for radial component
                phi = (1 + math.sqrt(5)) / 2
                t = i / 13.0
                r = 0.4 * math.sqrt(t)
                theta = 2 * math.pi * i * phi
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                z = 0.5 * (2 * t - 1)
                points.append([x, y, z])
            
            points = np.array(points)
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
        # Use fewer attempts but more sophisticated strategies
        for attempt in range(20):  # Increased number of attempts
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
                    # Random perturbation with larger variance
                    start_points = initial_points + np.random.normal(0, 0.05, initial_points.shape)
                elif attempt == 8:
                    # Alternate initialization method
                    start_points = generate_good_initialization()
                elif attempt == 9:
                    # More aggressive perturbation
                    start_points = initial_points + np.random.normal(0, 0.1, initial_points.shape)
                elif attempt == 10:
                    # Use different random seed
                    start_points = generate_random_points(14, seed=attempt*1000)
                elif attempt == 11:
                    # Even more aggressive perturbation
                    start_points = initial_points + np.random.normal(0, 0.07, initial_points.shape)
                elif attempt == 12:
                    # Perturbed octahedron
                    try:
                        oct_points = generate_good_initialization()
                        start_points = oct_points + np.random.normal(0, 0.04, oct_points.shape)
                    except:
                        start_points = generate_random_points(14, seed=attempt*1000)
                elif attempt == 13:
                    # Random points with high variance
                    start_points = generate_random_points(14, seed=attempt*500)
                elif attempt == 14:
                    # Very aggressive perturbation
                    start_points = initial_points + np.random.normal(0, 0.15, initial_points.shape)
                elif attempt == 15:
                    # Use specialized configuration
                    start_points = generate_good_initialization()
                elif attempt == 16:
                    # Another specialized configuration
                    start_points = generate_icosahedron_points()
                elif attempt == 17:
                    # Different low-discrepancy approach
                    start_points = generate_spherical_arrangement()
                elif attempt == 18:
                    # Mix of strategies
                    start_points = initial_points + np.random.normal(0, 0.08, initial_points.shape)
                else:
                    # Very diverse random initialization
                    start_points = generate_random_points(14, seed=attempt*10000)
                
                # Keep in bounds
                start_points = np.clip(start_points, 0, 1)
                flattened_points = start_points.flatten()
                
                # Try multiple optimizers with different settings
                optimizers = [
                    ('L-BFGS-B', {'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-16}),
                    ('TNC', {'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-16}),
                    ('SLSQP', {'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-16}),
                    ('Powell', {'maxiter': 1500})
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
                            options=options
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
    
    # Enhanced optimization with better search strategies
    def enhanced_optimization(initial_points, max_time=45):
        """Enhanced optimization with adaptive strategies"""
        start_time = time.time()
        
        # First, try a coarse optimization to find promising region
        bounds = [(0, 1)] * (14 * 3)
        constraints = [{'type': 'ineq', 'fun': lambda x: constraint_bounds(x)}]
        
        # Coarse optimization first
        try:
            result_coarse = minimize(
                objective_fast,
                initial_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result_coarse.success:
                initial_points = result_coarse.x.reshape(-1, 3)
        except:
            pass
        
        # Then perform multi-start optimization
        final_result = multi_start_optimization(initial_points, max_time)
        
        # Final refinement step with higher precision
        try:
            # Try to do one more optimization run with the best solution found so far
            result = minimize(
                objective_fast,
                final_result.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 3)
                refined_ratio = compute_min_max_ratio_jit(refined_points)
                current_ratio = compute_min_max_ratio_jit(final_result)
                
                if refined_ratio > current_ratio:
                    final_result = refined_points
        except:
            pass
        
        return final_result
    
    # Generate initial points
    initial_points = generate_good_initialization()
    
    # Run enhanced optimization
    final_result = enhanced_optimization(initial_points)
    
    return final_result


# EVOLVE-BLOCK-END
