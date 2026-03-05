# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction, multi-start optimization, 
    and advanced refinement strategies inspired by evolutionary algorithms and energy models.
    
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
            
            # Add 2 more points strategically - place them near poles but slightly offset
            points = np.vstack([
                ico_vertices,
                [[0, 0, 0.95]],
                [[0, 0, -0.95]]
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
    
    def create_initial_population(num_points=14, pop_size=40):
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
        
        # 3. Random configurations (about 1/3)
        for i in range(2 * pop_size // 3, pop_size):
            points = []
            while len(points) < num_points:
                point = np.random.uniform(-1, 1, 3)
                if np.linalg.norm(point) <= 1:
                    points.append(point)
            population.append(np.array(points))
            
        return population
    
    def energy_based_optimization(points, maxiter=500):
        """Optimize using energy-based approach inspired by physics (repulsive forces)"""
        def compute_energy_and_gradients(points):
            """Compute total energy and gradients for repulsive force model"""
            n = len(points)
            if n < 2:
                return 0.0, np.zeros_like(points)
            
            # Compute distance matrix
            distances = pdist(points)
            dist_matrix = squareform(distances)
            
            # Add small epsilon to avoid division by zero
            eps = 1e-12
            dist_matrix = np.maximum(dist_matrix, eps)
            
            # Compute energies (inverse of distances) - this creates repulsive forces
            energy = 0.0
            gradients = np.zeros_like(points)
            
            for i in range(n):
                for j in range(i+1, n):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq > eps:
                        inv_dist = 1.0 / np.sqrt(dist_sq)
                        energy += inv_dist
                        
                        # Gradient contribution
                        grad_ij = diff / (dist_sq**(3/2))
                        gradients[i] -= grad_ij
                        gradients[j] += grad_ij
            
            return energy, gradients
        
        def objective_and_gradient(points_flat):
            """Objective function and gradient for scipy optimization"""
            points = points_flat.reshape(-1, 3)
            energy, gradients = compute_energy_and_gradients(points)
            # Return negative energy (we want to maximize energy, which means minimize -energy)
            return -energy, -gradients.flatten()
        
        # Try optimization with different methods
        try:
            # Start with L-BFGS-B for smooth optimization
            result = minimize(
                lambda x: objective_and_gradient(x)[0],
                points.flatten(),
                method='L-BFGS-B',
                jac=lambda x: objective_and_gradient(x)[1],
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                return optimized_points
        except Exception:
            pass
            
        return points
    
    def optimize_with_multiple_methods(points, maxiter=1000):
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
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12}
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
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                ratio = calculate_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
        
        # Method 3: COBYLA as additional robust method
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='COBYLA',
                options={'maxiter': maxiter//2, 'rhobeg': 0.1}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                ratio = calculate_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
        
        # Method 4: Nelder-Mead as fallback
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
        
        # Method 5: Energy-based optimization (inspired by physics)
        try:
            energy_optimized = energy_based_optimization(points, maxiter=maxiter//2)
            ratio = calculate_ratio(energy_optimized)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = energy_optimized.copy()
        except Exception:
            pass
        
        return best_points
    
    def enhanced_local_refinement(points, iterations=150):
        """Enhanced local refinement with more sophisticated approach"""
        current_points = points.copy()
        current_ratio = calculate_ratio(current_points)
        
        # Enhanced local refinement with better step size control and multiple strategies
        for iteration in range(iterations):
            improved = False
            
            # Strategy 1: Try perturbations in all directions for all points
            for i in range(len(current_points)):
                for dim in range(3):
                    # Try both positive and negative small steps
                    for step_size in [0.0005, -0.0005, 0.001, -0.001]:
                        test_points = current_points.copy()
                        test_points[i, dim] += step_size
                        
                        # Keep within bounds
                        test_points[i, dim] = np.clip(test_points[i, dim], 0, 1)
                        
                        new_ratio = calculate_ratio(test_points)
                        if new_ratio > current_ratio:
                            current_points = test_points
                            current_ratio = new_ratio
                            improved = True
                            
            # Strategy 2: Try simultaneous small adjustments to multiple points
            if not improved and iteration % 5 == 0 and iteration > 0:
                # Perturb two random points at once
                idx1, idx2 = np.random.choice(len(current_points), 2, replace=False)
                for dim1 in range(3):
                    for dim2 in range(3):
                        test_points = current_points.copy()
                        step1 = 0.0005 * (1 - iteration/iterations)
                        step2 = 0.0005 * (1 - iteration/iterations)
                        test_points[idx1, dim1] += step1
                        test_points[idx2, dim2] += step2
                        
                        # Keep within bounds
                        test_points[idx1, dim1] = np.clip(test_points[idx1, dim1], 0, 1)
                        test_points[idx2, dim2] = np.clip(test_points[idx2, dim2], 0, 1)
                        
                        new_ratio = calculate_ratio(test_points)
                        if new_ratio > current_ratio:
                            current_points = test_points
                            current_ratio = new_ratio
                            improved = True
            
            # Strategy 3: Global adjustment based on ratio improvement
            if not improved and iteration % 10 == 0 and iteration > 0:
                # Try global adjustment to improve the overall configuration
                test_points = current_points.copy()
                # Apply small random perturbations to all points
                perturbations = np.random.normal(0, 0.0005, test_points.shape)
                test_points += perturbations
                
                # Keep within bounds
                test_points = np.clip(test_points, 0, 1)
                
                new_ratio = calculate_ratio(test_points)
                if new_ratio > current_ratio:
                    current_points = test_points
                    current_ratio = new_ratio
                    improved = True
            
            # Reduce step size if no improvement
            if not improved:
                break
                
        return current_points
    
    # Create diverse initial population
    initial_population = create_initial_population(pop_size=40)
    
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
    for start_iter in range(25):  # Increased from 20 to 25 for better exploration
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
    
    # Final refinement with enhanced optimization
    if best_points is not None:
        # Try several rounds of optimization
        refined_points = best_points.copy()
        
        # First, do a high-quality optimization with more iterations
        refined_points = optimize_with_multiple_methods(refined_points, maxiter=1000)
        
        # Then apply enhanced local refinement with more iterations
        refined_points = enhanced_local_refinement(refined_points, iterations=150)
        
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
