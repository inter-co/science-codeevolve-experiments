# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust hybrid approach combining multiple initialization strategies, 
    force-based optimization, and smart restart mechanisms.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        # Avoid division by zero
        if dmax <= 0:
            return 0.0
            
        return dmin / dmax
    
    def generate_fibonacci_spiral(n):
        """Generate points using Fibonacci spiral on sphere"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(n):
            # y goes from 1 to -1
            y = 1 - (i / float(n - 1)) * 2
            # radius at y
            radius = np.sqrt(1 - y * y)
            # golden angle increment
            theta = i * 2.399963229728653  # approximately 4π/(φ+1) 
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def generate_icosahedral_points():
        """Generate points based on icosahedral symmetry with better distribution"""
        # Vertices of regular icosahedron
        phi = (1 + np.sqrt(5)) / 2
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, -1, -phi), (0, 1, -phi),
            (1, phi, 0), (-1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (phi, 0, 1), (-phi, 0, 1), (-phi, 0, -1), (phi, 0, -1)
        ]
        
        points = np.array(vertices)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Add two more points for 14 total (north and south poles)
        points = np.vstack([points, [[0, 0, 1], [0, 0, -1]]])
        
        return points
    
    def generate_cube_plus_axes():
        """Generate points using cube vertices plus axis points"""
        # Start with vertices of a cube (8 points) 
        points = []
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    points.append([i, j, k])
        
        # Add 6 more points along axes for better distribution
        points.extend([
            [2, 0, 0], [-2, 0, 0],  # x-axis
            [0, 2, 0], [0, -2, 0],  # y-axis  
            [0, 0, 2], [0, 0, -2]   # z-axis
        ])
        
        # Keep only first 14 points and normalize
        points = np.array(points[:14], dtype=float)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        return points
    
    def generate_random_on_sphere(n):
        """Generate random points uniformly distributed on sphere"""
        points = []
        for _ in range(n):
            point = np.random.rand(3) * 2 - 1  # [-1, 1]^3
            norm = np.linalg.norm(point)
            if norm > 0:
                point = point / norm  # Project to unit sphere
            points.append(point)
        return np.array(points)
    
    def force_based_optimization(initial_points, max_iter=400):
        """Force-based optimization with improved convergence and early stopping"""
        points = initial_points.copy()
        best_points = points.copy()
        best_ratio = 0
        
        # Track improvements for early stopping
        improvements = []
        
        for iteration in range(max_iter):
            # Compute current ratio
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
                improvements = []  # Reset when we improve
            else:
                improvements.append(ratio)
                
                # Early stopping if no improvement in last 50 iterations
                if len(improvements) > 50:
                    improvements.pop(0)
                    if len(improvements) >= 2:
                        if abs(improvements[-1] - improvements[0]) < 1e-12:
                            break
            
            # Compute forces between all pairs
            forces = np.zeros_like(points)
            
            # Calculate pairwise forces (repulsive with damping)
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    
                    if dist_sq > 1e-12:  # Avoid division by zero
                        # Repulsive force with damping factor
                        force_magnitude = 1.0 / (dist_sq**2.0 + 1e-15)
                        force_vector = force_magnitude * diff
                        
                        forces[i] += force_vector
                        forces[j] -= force_vector
            
            # Update positions with adaptive learning rate
            learning_rate = 0.01 * (1.0 - iteration/max_iter * 0.5)
            points += learning_rate * forces
            
            # Project back to unit sphere
            norms = np.linalg.norm(points, axis=1)
            norms = np.maximum(norms, 1e-12)
            points = points / norms[:, np.newaxis]
            
        return best_points
    
    def objective_function(x):
        """Objective function to minimize (negative of ratio)"""
        points = x.reshape(-1, 3)
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero or invalid cases
        if d_max <= 1e-12:
            return 1e10
            
        ratio = d_min / d_max
        return -ratio  # Negative because we want to maximize ratio
    
    def sphere_constraint(x):
        """Constraint function for unit sphere"""
        points = x.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1)
        return norms - 1.0
    
    # Generate multiple diverse initial configurations
    initial_configs = []
    
    # Strategy 1: Fibonacci spiral (good uniform distribution)
    fib_points = generate_fibonacci_spiral(14)
    initial_configs.append(('fibonacci', fib_points))
    
    # Strategy 2: Icosahedral points (highly symmetric)
    ico_points = generate_icosahedral_points()
    initial_configs.append(('icosahedral', ico_points))
    
    # Strategy 3: Cube + axes (structured and balanced)
    cube_points = generate_cube_plus_axes()
    initial_configs.append(('cube_axes', cube_points))
    
    # Strategy 4: Random points on sphere (diversity)
    random_points = generate_random_on_sphere(14)
    initial_configs.append(('random', random_points))
    
    best_points = None
    best_ratio = -np.inf
    
    # Multi-start optimization with intelligent restarts
    total_restarts = 0
    max_total_restarts = 20  # To stay within time budget
    
    for strategy_name, init_config in initial_configs:
        if total_restarts >= max_total_restarts:
            break
            
        # Try multiple restarts for each strategy with varying perturbations
        for restart in range(6):  # Increased from 5 to 6 for better exploration
            if total_restarts >= max_total_restarts:
                break
                
            total_restarts += 1
            
            # Add small random perturbation to initialization
            if restart > 0:
                # Vary perturbation scales for better exploration
                if restart <= 2:
                    perturbation_scale = 0.08
                elif restart <= 4:
                    perturbation_scale = 0.04
                else:
                    perturbation_scale = 0.02
                
                perturbed = init_config + np.random.normal(0, perturbation_scale, init_config.shape)
                # Normalize again to keep on unit sphere
                norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
                perturbed = perturbed / norms
            else:
                perturbed = init_config.copy()
            
            # Flatten for optimization
            x0 = perturbed.flatten()
            
            # Set optimization parameters for better balance of speed and quality
            if restart < 2:
                max_iter = 800
                method = 'L-BFGS-B'
            elif restart < 4:
                max_iter = 600
                method = 'SLSQP'
            else:
                max_iter = 400
                method = 'L-BFGS-B'
            
            try:
                result = minimize(
                    objective_function,
                    x0,
                    method=method,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )
                
                if result.success:
                    optimized_points = result.x.reshape(-1, 3)
                    # Evaluate the result
                    ratio = compute_min_max_ratio(optimized_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        # Early exit if we're close to the benchmark
                        if ratio > 0.45:
                            return best_points
                                
            except Exception:
                continue
    
    # Final force-based refinement if we have a solution
    if best_points is not None:
        try:
            # Apply force-based optimization for final improvement
            final_refinement = force_based_optimization(best_points, max_iter=300)
            final_ratio = compute_min_max_ratio(final_refinement)
            
            if final_ratio > best_ratio:
                best_points = final_refinement
        except Exception:
            pass
    
    # If no good solution found, return the best initial configuration
    if best_points is None:
        # Return the fibonacci spiral as the fallback since it's generally good
        return generate_fibonacci_spiral(14)
    
    return best_points


# EVOLVE-BLOCK-END
