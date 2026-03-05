# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import random


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a combination of geometric initialization and simulated annealing optimization.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        distances = pdist(points)
        if len(distances) == 0 or np.max(distances) == 0:
            return 0
        return np.min(distances) / np.max(distances)
    
    def project_to_sphere(points):
        """Project points onto unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def generate_icosahedral_initialization():
        """Generate initial points using icosahedral symmetry."""
        # Vertices of regular icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ]
        
        # Normalize to unit sphere
        normalized_vertices = []
        for x, y, z in vertices:
            norm = np.sqrt(x*x + y*y + z*z)
            normalized_vertices.append([x/norm, y/norm, z/norm])
        
        # Return first 12 vertices plus 2 more points near poles
        points = normalized_vertices[:12]
        points.append([0, 0, 0.95])  # Near north pole
        points.append([0, 0, -0.95])  # Near south pole
        
        return np.array(points)
    
    def generate_fibonacci_sphere_initialization(n=14):
        """Generate points using Fibonacci spiral on sphere."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle
        
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def initialize_points():
        """Initialize points using a geometric approach inspired by icosahedron."""
        # Try multiple initialization strategies and pick the best
        strategies = [
            generate_icosahedral_initialization,
            lambda: generate_fibonacci_sphere_initialization(14),
            lambda: np.random.randn(14, 3)
        ]
        
        best_points = None
        best_ratio = -np.inf
        
        for strategy in strategies:
            points = strategy()
            points = project_to_sphere(points)
            
            # Add some random noise to break symmetry for random strategy
            if strategy == strategies[-1]:
                np.random.seed(42)
                points += np.random.normal(0, 0.05, points.shape)
                points = project_to_sphere(points)
            
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
        
        return best_points
    
    def simulated_annealing():
        """Run simulated annealing optimization with enhanced parameters."""
        best_points = None
        best_ratio = -np.inf
        
        # Multiple random starts with diverse initializations
        for start_iter in range(25):  # Reduced from 30 to stay within time budget
            # Initialize with different strategies
            if start_iter % 3 == 0:
                points = generate_icosahedral_initialization()
            elif start_iter % 3 == 1:
                points = generate_fibonacci_sphere_initialization(14)
            else:
                # Random initialization as fallback
                points = np.random.randn(14, 3)
                points = project_to_sphere(points)
            
            # Add some random noise to break symmetry
            np.random.seed(start_iter)
            points += np.random.normal(0, 0.05, points.shape)
            points = project_to_sphere(points)
            
            # Convert to flat array for optimization
            points_flat = points.flatten()
            
            # Optimization parameters - tuned for faster convergence
            temp = 1.0
            cooling_rate = 0.995
            min_temp = 1e-6
            iterations_per_temp = 150  # Reduced iterations to save time
            
            current_points_flat = points_flat.copy()
            current_ratio = compute_min_max_ratio(current_points_flat.reshape(-1, 3))
            
            # Store best solution found
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = current_points_flat.copy()
            
            # Simulated annealing loop
            iteration_count = 0
            while temp > min_temp and iteration_count < 3000:  # Reduced iterations
                for _ in range(iterations_per_temp):
                    # Create neighbor by perturbing one point
                    new_points_flat = current_points_flat.copy()
                    point_idx = random.randint(0, 13)
                    dim_idx = random.randint(0, 2)
                    
                    # Perturb the point with adaptive step size
                    step_size = 0.03 if temp > 0.1 else 0.008
                    new_points_flat[point_idx * 3 + dim_idx] += np.random.normal(0, step_size)
                    
                    # Project back to sphere
                    new_points = new_points_flat.reshape(-1, 3)
                    new_points = project_to_sphere(new_points)
                    new_points_flat = new_points.flatten()
                    
                    # Evaluate new solution
                    new_ratio = compute_min_max_ratio(new_points)
                    
                    # Accept or reject based on Metropolis criterion
                    if new_ratio > current_ratio or np.random.rand() < np.exp((new_ratio - current_ratio) / temp):
                        current_points_flat = new_points_flat.copy()
                        current_ratio = new_ratio
                        
                        if current_ratio > best_ratio:
                            best_ratio = current_ratio
                            best_points = current_points_flat.copy()
                
                temp *= cooling_rate
                iteration_count += 1
            
            # Early termination if we're getting very good results
            if best_ratio > 0.45:
                break
        
        return best_points.reshape(-1, 3) if best_points is not None else initialize_points()
    
    # Run optimization using the proven simulated annealing approach
    optimized_points = simulated_annealing()
    
    # Final refinement with scipy optimizer for high precision
    points_flat = optimized_points.flatten()
    
    # Use L-BFGS-B for fine-tuning with tighter tolerances
    def objective_function(x_flat):
        points = x_flat.reshape(-1, 3)
        points = project_to_sphere(points)
        ratio = compute_min_max_ratio(points)
        return -ratio if ratio > 0 else 1e10  # Large penalty for invalid cases
    
    try:
        result = minimize(
            objective_function,
            points_flat,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15}  # Reduced iterations for speed
        )
        
        if result.success:
            final_points = result.x.reshape(-1, 3)
            final_points = project_to_sphere(final_points)
        else:
            final_points = optimized_points
    except Exception:
        # If optimization fails, use the best we have
        final_points = optimized_points
    
    return final_points


# EVOLVE-BLOCK-END
