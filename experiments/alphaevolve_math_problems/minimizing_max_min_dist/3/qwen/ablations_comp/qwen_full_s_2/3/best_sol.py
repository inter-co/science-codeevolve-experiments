# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Implements a hybrid approach combining stochastic optimization with deterministic 
    optimization methods for superior results.

    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Calculate the min/max distance ratio for a given configuration."""
        # Ensure points are on unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0:
            return 0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return ratio (avoiding division by zero)
        if max_dist <= 0:
            return 0
        return min_dist / max_dist
    
    def fibonacci_sphere(samples=14):
        """Generate points using Fibonacci spiral on sphere."""
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def icosahedral_config():
        """Create configuration based on icosahedral symmetry."""
        # Icosahedron vertices (normalized)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / np.maximum(norms, 1e-10)
        
        # Select 12 points and add 2 more for 14 total
        extra_points = np.array([
            [0, 0, 0.95],  # Near north pole
            [0, 0, -0.95]   # Near south pole
        ])
        
        return np.vstack([vertices[:12], extra_points])
    
    def neighbor_move(points, step_size=0.05):
        """Generate a neighboring point configuration via random perturbations."""
        # Make a copy to avoid modifying original
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
    
    def simulated_annealing(initial_points):
        """Optimize using simulated annealing with adaptive cooling schedule."""
        
        current_points = initial_points.copy()
        current_score = compute_min_max_ratio(current_points)
        
        # Annealing parameters
        temperature = 1.0
        min_temperature = 1e-6
        cooling_rate = 0.999
        max_iterations = 15000
        
        # Track best solution
        best_points = current_points.copy()
        best_score = current_score
        
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
    
    def local_refinement(points, max_iter=1000):
        """Apply local refinement to improve the solution."""
        current_points = points.copy()
        current_score = compute_min_max_ratio(current_points)
        
        for _ in range(max_iter):
            improved = False
            # Try small perturbations to all points
            for i in range(len(current_points)):
                # Try a small perturbation
                test_points = current_points.copy()
                delta = np.random.normal(0, 0.005, 3)
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
    
    def constrained_optimization(initial_points):
        """Use constrained optimization as a refinement step."""
        def objective_function(x):
            """Objective function to maximize (negative because we minimize)"""
            points = x.reshape(-1, 3)
            ratio = compute_min_max_ratio(points)
            return -ratio  # Negative because we minimize
        
        def constraint_sphere(x):
            """Constraint that all points lie on unit sphere"""
            points = x.reshape(-1, 3)
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0
        
        # Flatten the initial points
        x0 = initial_points.flatten()
        cons = {'type': 'eq', 'fun': constraint_sphere}
        
        try:
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
                tol=1e-12
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                # Ensure points are on unit sphere after optimization
                norms = np.linalg.norm(optimized_points, axis=1, keepdims=True)
                optimized_points = optimized_points / np.maximum(norms, 1e-10)
                return optimized_points
        except Exception:
            pass
            
        return initial_points
    
    # Main optimization loop
    best_points = None
    best_score = float('-inf')
    
    # Strategy 1: Multiple Fibonacci restarts with varying perturbations
    for restart in range(5):
        if restart == 0:
            initial_points = fibonacci_sphere(14)
        else:
            # Fibonacci with random perturbations
            initial_points = fibonacci_sphere(14) + np.random.normal(0, 0.02, (14, 3))
        
        # Ensure points are on unit sphere
        norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
        initial_points = initial_points / np.maximum(norms, 1e-10)
        
        # Run simulated annealing
        sa_points, sa_score = simulated_annealing(initial_points)
        
        # Refine with local optimization
        refined_points = local_refinement(sa_points)
        refined_score = compute_min_max_ratio(refined_points)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
    
    # Strategy 2: Icosahedral-based initialization
    try:
        initial_points = icosahedral_config()
        # Ensure points are on unit sphere
        norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
        initial_points = initial_points / np.maximum(norms, 1e-10)
        
        # Run simulated annealing
        sa_points, sa_score = simulated_annealing(initial_points)
        
        # Refine with local optimization and constrained optimization
        refined_points = local_refinement(sa_points)
        refined_score = compute_min_max_ratio(refined_points)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
            
        # Additional refinement with constrained optimization
        constrained_points = constrained_optimization(refined_points)
        constrained_score = compute_min_max_ratio(constrained_points)
        
        if constrained_score > best_score:
            best_score = constrained_score
            best_points = constrained_points.copy()
            
    except Exception:
        pass
    
    # Strategy 3: Random restarts for global exploration
    for restart in range(3):
        np.random.seed(42 + restart)  # Different seed for each restart
        initial_points = np.random.randn(14, 3)
        norms = np.linalg.norm(initial_points, axis=1, keepdims=True)
        initial_points = initial_points / np.maximum(norms, 1e-10)
        
        # Run simulated annealing
        sa_points, sa_score = simulated_annealing(initial_points)
        
        # Refine with local optimization
        refined_points = local_refinement(sa_points)
        refined_score = compute_min_max_ratio(refined_points)
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()
    
    # Final refinement step with constrained optimization
    if best_points is not None:
        final_points = constrained_optimization(best_points)
        final_score = compute_min_max_ratio(final_points)
        
        if final_score > best_score:
            best_points = final_points
    
    # If no good solution found, return Fibonacci spiral as fallback
    if best_points is None:
        best_points = fibonacci_sphere(14)
    
    return best_points


# EVOLVE-BLOCK-END
