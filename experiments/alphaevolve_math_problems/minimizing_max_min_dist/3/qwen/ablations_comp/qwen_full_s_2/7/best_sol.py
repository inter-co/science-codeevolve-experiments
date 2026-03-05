# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple geometric constructions, energy minimization,
    and advanced optimization strategies inspired by successful approaches.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points"""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    # Generate initial points using Fibonacci spiral on sphere for good distribution
    def fibonacci_spiral_on_sphere(n):
        points = []
        phi = math.pi * (3 - math.sqrt(5))  # golden angle
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            
            points.append((x, y, z))
        
        return np.array(points)
    
    # Construct from icosahedron plus poles for better symmetry
    def construct_icosahedral_plus_poles():
        """Construct points using icosahedral symmetry with additional pole points"""
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2
        
        # Vertices of regular icosahedron (normalized)
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]
        
        # Normalize vertices to unit sphere
        normalized_vertices = []
        for vertex in vertices:
            norm = math.sqrt(sum(v**2 for v in vertex))
            if norm > 0:
                normalized_vertices.append([v/norm for v in vertex])
            else:
                normalized_vertices.append([0, 0, 0])
        
        # Start with 12 icosahedral vertices
        points = normalized_vertices[:]
        
        # Add 2 more points at the poles for 14 total (slightly perturbed)
        points.extend([[0, 0, 0.98], [0, 0, -0.98]])
        
        return np.array(points)
    
    # Energy minimization approach with better parameters
    def energy_function(points, p=12):
        """
        Computes total electrostatic energy with inverse power law potential.
        Minimizing this energy tends to distribute points uniformly.
        """
        n = len(points)
        total_energy = 0.0
        
        # For each pair of points
        for i in range(n):
            for j in range(i+1, n):
                # Calculate squared distance
                dist_sq = np.sum((points[i] - points[j]) ** 2)
                
                # Avoid division by zero
                if dist_sq < 1e-12:
                    continue
                    
                # Inverse power law potential (repulsive force)
                total_energy += 1.0 / (dist_sq ** (p/2))
        
        return total_energy
    
    # Objective function for optimization
    def objective(x_flat):
        """
        Objective function to maximize the min/max distance ratio.
        Reshapes input and computes the ratio, returns negative for minimization.
        """
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Ensure points remain on unit sphere by normalizing
        for i in range(len(points)):
            norm = np.linalg.norm(points[i])
            if norm > 0:
                points[i] = points[i] / norm
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        if len(distances) == 0:
            return float('inf')
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio since we want to maximize
        if max_dist == 0:
            return float('inf')
        return -min_dist / max_dist
    
    # Enhanced optimization with multiple strategies
    def enhanced_optimization(initial_points, max_iter=500):
        """Enhanced optimization with multiple restarts and strategies"""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        # Strategy 1: Energy minimization first
        try:
            x0 = best_points.flatten()
            
            def energy_objective(x_flat):
                points = x_flat.reshape(-1, 3)
                for i in range(len(points)):
                    norm = np.linalg.norm(points[i])
                    if norm > 0:
                        points[i] = points[i] / norm
                return energy_function(points, p=12)
            
            result_energy = minimize(
                energy_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9}
            )
            
            if result_energy.success:
                energy_points = result_energy.x.reshape(-1, 3)
                energy_ratio = compute_min_max_ratio(energy_points)
                if energy_ratio > best_ratio:
                    best_ratio = energy_ratio
                    best_points = energy_points.copy()
        except Exception:
            pass
        
        # Strategy 2: Multiple restarts with different optimization methods
        for restart in range(10):
            try:
                # Perturb the current best points
                np.random.seed(1000 + restart)
                perturbed = best_points + np.random.normal(0, 0.03, best_points.shape)
                
                # Normalize
                norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
                safe_norms = np.where(norms == 0, 1, norms)
                perturbed = perturbed / safe_norms
                
                # Try different optimization methods
                methods_to_try = ['L-BFGS-B', 'SLSQP']
                for method in methods_to_try:
                    try:
                        x0 = perturbed.flatten()
                        result = minimize(
                            objective,
                            x0,
                            method=method,
                            options={'maxiter': 300, 'ftol': 1e-11, 'gtol': 1e-11}
                        )
                        
                        if result.success:
                            restart_points = result.x.reshape(-1, 3)
                            # Normalize points
                            norms = np.linalg.norm(restart_points, axis=1, keepdims=True)
                            safe_norms = np.where(norms == 0, 1, norms)
                            restart_points = restart_points / safe_norms
                            restart_ratio = compute_min_max_ratio(restart_points)
                            
                            if restart_ratio > best_ratio:
                                best_ratio = restart_ratio
                                best_points = restart_points.copy()
                                
                    except Exception:
                        continue
                        
            except Exception:
                continue
        
        # Strategy 3: Nelder-Mead as fallback for robustness
        try:
            x0 = best_points.flatten()
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'adaptive': True}
            )
            
            if result.success:
                nelder_points = result.x.reshape(-1, 3)
                # Normalize points
                norms = np.linalg.norm(nelder_points, axis=1, keepdims=True)
                safe_norms = np.where(norms == 0, 1, norms)
                nelder_points = nelder_points / safe_norms
                nelder_ratio = compute_min_max_ratio(nelder_points)
                
                if nelder_ratio > best_ratio:
                    best_ratio = nelder_ratio
                    best_points = nelder_points.copy()
        except Exception:
            pass
        
        return best_points
    
    # Generate multiple initial configurations
    candidates = []
    
    # Try several known good constructions
    try:
        ico_points = construct_icosahedral_plus_poles()
        candidates.append(("icosahedral", ico_points))
    except Exception:
        pass
    
    try:
        fib_points = fibonacci_spiral_on_sphere(14)
        candidates.append(("fibonacci", fib_points))
    except Exception:
        pass
    
    # Also try a more structured configuration from known mathematical solutions
    try:
        # Create a configuration inspired by the icosahedral group plus additional points
        phi = (1 + math.sqrt(5)) / 2
        points = []
        
        # Icosahedron vertices
        for i, (x, y, z) in enumerate([
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]):
            norm = math.sqrt(x*x + y*y + z*z)
            points.append([x/norm, y/norm, z/norm])
        
        # Add two more points at poles
        points.extend([[0, 0, 0.99], [0, 0, -0.99]])
        structured_points = np.array(points[:14])
        candidates.append(("structured", structured_points))
    except Exception:
        pass
    
    # Evaluate all candidates
    best_ratio = -1.0
    best_points = None
    
    for name, points in candidates:
        ratio = compute_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # If nothing worked, use Fibonacci spiral as fallback
    if best_points is None:
        best_points = fibonacci_spiral_on_sphere(14)
    
    # Apply enhanced optimization to the best initial configuration
    optimized_points = enhanced_optimization(best_points, max_iter=500)
    
    # Final validation and normalization
    final_ratio = compute_min_max_ratio(optimized_points)
    
    # If optimization didn't improve much, try one final refinement with high precision
    if final_ratio <= best_ratio * 1.001:  # Only if significantly worse
        # Try even more aggressive optimization
        try:
            x0 = optimized_points.flatten()
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 3)
                # Normalize points
                norms = np.linalg.norm(final_points, axis=1, keepdims=True)
                safe_norms = np.where(norms == 0, 1, norms)
                final_points = final_points / safe_norms
                final_ratio = compute_min_max_ratio(final_points)
                
                if final_ratio > best_ratio:
                    optimized_points = final_points
        except Exception:
            pass
    
    # Final normalization to unit sphere
    norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1, norms)
    final_points = optimized_points / safe_norms
    
    return final_points


# EVOLVE-BLOCK-END
