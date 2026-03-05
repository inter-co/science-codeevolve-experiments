# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction, multi-start optimization, 
    and advanced refinement strategies inspired by evolutionary algorithms.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def calculate_ratio(points):
        """Calculate the min/max distance ratio for given points"""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def create_spherical_code_points(n_points):
        """Create points based on known spherical codes for better initial distribution"""
        if n_points == 14:
            # Use a configuration inspired by known good arrangements
            # This is a carefully chosen configuration based on mathematical principles
            phi = (1 + np.sqrt(5)) / 2  # golden ratio
            
            # Create vertices of icosahedron and add strategic points
            ico_vertices = np.array([
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
            ])
            
            # Normalize to unit sphere
            ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0])
            
            # Add 2 more points strategically
            points = np.vstack([
                ico_vertices,
                [[0, 0, 0.9]],
                [[0, 0, -0.9]]
            ])
            
            # Ensure exactly 14 points
            points = points[:14]
            
            # Add slight randomization to break symmetry
            points += np.random.normal(0, 0.01, points.shape)
            
            # Normalize again to ensure unit sphere
            for i in range(len(points)):
                norm = np.linalg.norm(points[i])
                if norm > 0:
                    points[i] = points[i] / norm
                    
            return points
        else:
            # Fallback to Fibonacci spiral for other numbers
            points = []
            phi = np.pi * (3 - np.sqrt(5))  # Golden angle
            
            for i in range(n_points):
                y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y
                
                theta = phi * i  # golden angle increment
                
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                
                points.append([x, y, z])
                
            return np.array(points)
    
    def create_fibonacci_points(n_points):
        """Create points distributed on a sphere using Fibonacci spiral method"""
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        
        for i in range(n_points):
            y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
            
        return np.array(points)
    
    def create_initial_population(num_points=14, pop_size=30):
        """Create diverse initial population for optimization"""
        population = []
        
        # 1. Spherical code configurations (about 1/3)
        for i in range(pop_size // 3):
            points = create_spherical_code_points(num_points)
            points += np.random.normal(0, 0.02, (num_points, 3))
            # Normalize to unit sphere
            for j in range(num_points):
                norm = np.linalg.norm(points[j])
                if norm > 0:
                    points[j] = points[j] / norm
            population.append(points)
        
        # 2. Fibonacci spiral configurations (about 1/3)
        for i in range(pop_size // 3, 2 * pop_size // 3):
            points = create_fibonacci_points(num_points)
            points += np.random.normal(0, 0.02, (num_points, 3))
            # Normalize to unit sphere
            for j in range(num_points):
                norm = np.linalg.norm(points[j])
                if norm > 0:
                    points[j] = points[j] / norm
            population.append(points)
        
        # 3. Random configurations (about 1/3) - but more controlled
        for i in range(2 * pop_size // 3, pop_size):
            points = []
            # Generate points uniformly distributed inside the unit sphere
            while len(points) < num_points:
                point = np.random.uniform(-1, 1, 3)
                if np.linalg.norm(point) <= 1:
                    points.append(point)
            population.append(np.array(points))
            
        return population
    
    def optimize_with_multiple_methods(points, maxiter=800):
        """Optimize points using multiple methods with different strategies"""
        def objective(points_flat):
            points = points_flat.reshape(-1, 3)
            distances = pdist(points)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist == 0:
                return 0
            return -min_dist / max_dist
        
        # Try multiple optimization approaches with different settings
        best_points = points.copy()
        best_ratio = calculate_ratio(best_points)
        
        # Method 1: L-BFGS-B with high precision
        try:
            bounds = [(0, 1) for _ in range(len(points) * 3)]
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                ratio = calculate_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
        
        # Method 2: SLSQP with stricter tolerances
        try:
            bounds = [(0, 1) for _ in range(len(points) * 3)]
            result = minimize(
                objective,
                points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                ratio = calculate_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
        
        # Method 3: Nelder-Mead as fallback with more iterations
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter//2, 'adaptive': True}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                ratio = calculate_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
        
        return best_points
    
    def local_refinement(points, iterations=50):
        """Apply local refinement to improve convergence"""
        current_points = points.copy()
        current_ratio = calculate_ratio(current_points)
        
        # Simple gradient-like approach with adaptive steps
        for iteration in range(iterations):
            # Try small perturbations to improve the solution
            improved = False
            for i in range(len(current_points)):
                # Try small displacements in each direction
                for dim in range(3):
                    test_points = current_points.copy()
                    step = 0.001 * (1 - iteration/iterations)  # Decreasing step size
                    test_points[i, dim] += step
                    
                    # Keep within bounds
                    test_points[i, dim] = np.clip(test_points[i, dim], 0, 1)
                    
                    new_ratio = calculate_ratio(test_points)
                    if new_ratio > current_ratio:
                        current_points = test_points
                        current_ratio = new_ratio
                        improved = True
                        
            # If no improvement, reduce step size and try again
            if not improved:
                break
                
        return current_points
    
    # Start timing
    start_time = time.time()
    
    # Create diverse initial population
    initial_population = create_initial_population(pop_size=30)
    
    # Evaluate initial population and find best
    best_ratio = 0.0
    best_points = None
    
    # Evaluate all initial points
    for points in initial_population:
        ratio = calculate_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    
    # Multi-start optimization with different initial configurations
    # Use a more systematic approach with better diversity
    for start_iter in range(15):  # Return to 15 for better exploration
        # Select a random initial point from the population
        np.random.seed(start_iter * 1000 + 42)
        current_points = initial_population[np.random.randint(0, len(initial_population))].copy()
        
        # Add more randomization for better exploration
        current_points += np.random.normal(0, 0.015, current_points.shape)
        
        # Optimize this starting point with multiple methods
        optimized_points = optimize_with_multiple_methods(current_points, maxiter=300)
        ratio = calculate_ratio(optimized_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = optimized_points.copy()
            
        # Early termination if we're running out of time
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break
    
    # Final refinement with enhanced optimization
    if best_points is not None:
        # Try several rounds of optimization
        refined_points = best_points.copy()
        
        # First, do a high-quality optimization
        refined_points = optimize_with_multiple_methods(refined_points, maxiter=600)
        
        # Then apply local refinement
        refined_points = local_refinement(refined_points, iterations=80)
        
        final_ratio = calculate_ratio(refined_points)
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_points = refined_points
    
    # If no good solution found, fall back to a good geometric configuration
    if best_points is None:
        best_points = create_spherical_code_points(14)
    
    # Ensure points are within [0,1]^3
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
