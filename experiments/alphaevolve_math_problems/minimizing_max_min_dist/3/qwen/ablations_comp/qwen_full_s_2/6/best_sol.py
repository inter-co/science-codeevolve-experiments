# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import math
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric constructions, simulated annealing, 
    and advanced numerical optimization techniques.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    n = 14
    
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
        phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle
        
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
        points.extend([[0, 0, 0.99], [0, 0, -0.99]])
        
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
    
    # Simulated annealing implementation for global optimization
    def simulated_annealing(initial_points, max_iter=10000):
        """Optimize using simulated annealing with adaptive cooling schedule."""
        current_points = initial_points.copy()
        current_score = compute_min_max_ratio(current_points)
        
        # Annealing parameters
        temperature = 1.0
        min_temperature = 1e-8
        cooling_rate = 0.9995
        max_iterations = max_iter
        
        # Track best solution
        best_points = current_points.copy()
        best_score = current_score
        
        # Generate neighbor function
        def neighbor_move(points, step_size=0.05):
            new_points = points.copy()
            # Choose a random point to perturb
            idx = np.random.randint(len(points))
            # Perturb that point in random direction
            delta = np.random.normal(0, step_size, 3)
            new_points[idx] += delta
            # Project back onto unit sphere
            norm = np.linalg.norm(new_points[idx])
            if norm > 0:
                new_points[idx] = new_points[idx] / norm
            return new_points
        
        # Simulated annealing loop
        for iteration in range(max_iterations):
            # Generate neighbor
            new_points = neighbor_move(current_points)
            new_score = compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_score > current_score:
                # Always accept better solutions
                current_points = new_points
                current_score = new_score
            else:
                # Accept worse solutions with probability based on temperature
                delta = new_score - current_score
                acceptance_prob = np.exp(delta / temperature)
                if np.random.random() < acceptance_prob:
                    current_points = new_points
                    current_score = new_score
            
            # Update best solution
            if current_score > best_score:
                best_points = current_points.copy()
                best_score = current_score
            
            # Cool down temperature
            temperature *= cooling_rate
            
            # Stop if temperature gets too low
            if temperature < min_temperature:
                break
                
        return best_points, best_score
    
    # Objective function for optimization - maximizes min/max ratio
    def objective(x_flat):
        """
        Objective function to maximize the min/max distance ratio.
        Reshapes input and computes the ratio, returns negative for minimization.
        """
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 3)
        
        # Ensure points remain on unit sphere by normalizing
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0, 1, norms)
        points = points / safe_norms
        
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
    
    # Constraint function to keep points on unit sphere
    def sphere_constraint(x_flat):
        points = x_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0  # Should equal zero for unit sphere
    
    # Local refinement with gradient-based optimization
    def local_refinement(points, max_iter=200):
        """Apply local refinement to improve the solution."""
        current_points = points.copy()
        current_score = compute_min_max_ratio(current_points)
        
        for _ in range(max_iter):
            improved = False
            # Try small perturbations to all points
            for i in range(len(current_points)):
                # Try a small perturbation
                test_points = current_points.copy()
                delta = np.random.normal(0, 0.002, 3)
                test_points[i] += delta
                
                # Project back to unit sphere
                norm = np.linalg.norm(test_points[i])
                if norm > 0:
                    test_points[i] = test_points[i] / norm
                
                new_score = compute_min_max_ratio(test_points)
                
                # Accept if improvement
                if new_score > current_score:
                    current_points = test_points
                    current_score = new_score
                    improved = True
            
            # If no improvement was made, reduce step size
            if not improved:
                break
                
        return current_points
    
    # Enhanced optimization with multiple strategies and better constraint handling
    def enhanced_optimization(initial_points, max_iter=500):
        """Enhanced optimization with proper constraints and multiple restarts"""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        # Strategy 1: Energy minimization first to get a good starting configuration
        try:
            x0 = best_points.flatten()
            
            def energy_objective(x_flat):
                points = x_flat.reshape(-1, 3)
                # Normalize points before computing energy
                norms = np.linalg.norm(points, axis=1, keepdims=True)
                safe_norms = np.where(norms == 0, 1, norms)
                points = points / safe_norms
                return energy_function(points, p=12)
            
            result_energy = minimize(
                energy_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result_energy.success:
                energy_points = result_energy.x.reshape(-1, 3)
                # Normalize after energy minimization
                norms = np.linalg.norm(energy_points, axis=1, keepdims=True)
                safe_norms = np.where(norms == 0, 1, norms)
                energy_points = energy_points / safe_norms
                energy_ratio = compute_min_max_ratio(energy_points)
                if energy_ratio > best_ratio:
                    best_ratio = energy_ratio
                    best_points = energy_points.copy()
        except Exception as e:
            pass
        
        # Strategy 2: Simulated annealing for global exploration
        try:
            sa_points, sa_score = simulated_annealing(best_points, max_iter=5000)
            if sa_score > best_ratio:
                best_ratio = sa_score
                best_points = sa_points.copy()
        except Exception:
            pass
        
        # Strategy 3: Multiple restarts with different optimization methods and constraints
        for restart in range(5):  # Reduced restarts to stay within time budget
            try:
                # Perturb the current best points with different noise levels
                np.random.seed(1000 + restart)
                noise_level = 0.01 + restart * 0.005  # Gradually increase noise
                perturbed = best_points + np.random.normal(0, noise_level, best_points.shape)
                
                # Normalize the perturbed points
                norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
                safe_norms = np.where(norms == 0, 1, norms)
                perturbed = perturbed / safe_norms
                
                # Try different optimization methods with constraints
                methods_to_try = ['SLSQP', 'L-BFGS-B']
                for method in methods_to_try:
                    try:
                        x0 = perturbed.flatten()
                        constraints = {'type': 'eq', 'fun': sphere_constraint}
                        
                        result = minimize(
                            objective,
                            x0,
                            method=method,
                            constraints=constraints,
                            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
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
        
        # Strategy 4: Final high precision refinement with strict constraints
        try:
            x0 = best_points.flatten()
            constraints = {'type': 'eq', 'fun': sphere_constraint}
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 3)
                # Ensure normalization
                norms = np.linalg.norm(final_points, axis=1, keepdims=True)
                safe_norms = np.where(norms == 0, 1, norms)
                final_points = final_points / safe_norms
                final_ratio = compute_min_max_ratio(final_points)
                
                if final_ratio > best_ratio:
                    best_points = final_points
        except Exception:
            pass
        
        # Strategy 5: Local refinement as final step
        try:
            refined_points = local_refinement(best_points, max_iter=100)
            refined_ratio = compute_min_max_ratio(refined_points)
            if refined_ratio > best_ratio:
                best_points = refined_points
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
    
    # Also try a configuration inspired by known mathematical solutions
    try:
        # Use a configuration that's known to work well for small numbers of points
        # Based on the principle of distributing points as uniformly as possible
        points = []
        # Create a configuration that balances symmetry and uniformity
        phi = (1 + math.sqrt(5)) / 2
        
        # Icosahedral vertices
        for x, y, z in [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]:
            norm = math.sqrt(x*x + y*y + z*z)
            points.append([x/norm, y/norm, z/norm])
        
        # Add two more points to make 14
        points.extend([[0, 0, 0.98], [0, 0, -0.98]])
        structured_points = np.array(points[:14])
        candidates.append(("structured", structured_points))
    except Exception:
        pass
    
    # Add a few more random configurations for diversity
    for i in range(3):
        try:
            np.random.seed(2000 + i)
            random_points = np.random.randn(14, 3)
            norms = np.linalg.norm(random_points, axis=1, keepdims=True)
            random_points = random_points / np.maximum(norms, 1e-10)
            candidates.append((f"random_{i}", random_points))
        except Exception:
            continue
    
    # Evaluate all candidates and find the best one
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
    
    # Final validation
    final_ratio = compute_min_max_ratio(optimized_points)
    
    return optimized_points


# EVOLVE-BLOCK-END
