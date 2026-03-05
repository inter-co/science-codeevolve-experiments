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
    and advanced refinement strategies with energy-based methods inspired by physics.
    
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
    
    def create_initial_configuration():
        """Create a high-quality initial configuration using known geometric principles"""
        # Start with icosahedral configuration as suggested by mathematical literature
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        # Vertices of regular icosahedron (12 vertices)
        ico_vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0])
        
        # Add 2 strategic points to get 14 points
        # Place them along the z-axis but slightly away from poles to avoid degeneracy
        points = np.vstack([
            ico_vertices,
            [[0, 0, 0.98]],  # Slightly off north pole
            [[0, 0, -0.98]]  # Slightly off south pole
        ])
        
        # Ensure exactly 14 points
        points = points[:14]
        
        # Add some noise to break symmetry and improve optimization
        points += np.random.normal(0, 0.005, points.shape)
        
        # Normalize to unit sphere
        for i in range(len(points)):
            norm = np.linalg.norm(points[i])
            if norm > 0:
                points[i] = points[i] / norm
                
        return points
    
    def energy_based_optimization(points, maxiter=800):
        """Optimize using energy-based approach inspired by physics (repulsive forces)"""
        def compute_energy_and_gradients(points):
            """Compute total energy and gradients for repulsive force model"""
            n = len(points)
            if n < 2:
                return 0.0, np.zeros_like(points)
            
            # Compute distance matrix efficiently
            distances = pdist(points)
            dist_matrix = squareform(distances)
            
            # Add small epsilon to avoid division by zero
            eps = 1e-12
            dist_matrix = np.maximum(dist_matrix, eps)
            
            # Compute energies (inverse of distances) - this creates repulsive forces
            # Sum of inverse distances is our energy function
            energy = 0.0
            gradients = np.zeros_like(points)
            
            # Vectorized computation for efficiency
            for i in range(n):
                for j in range(i+1, n):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq > eps:
                        inv_dist = 1.0 / np.sqrt(dist_sq)
                        energy += inv_dist
                        
                        # Gradient contribution (negative because we're minimizing energy)
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
        
        # Try optimization with L-BFGS-B for smooth optimization
        try:
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
    
    def local_refinement(points, iterations=100):
        """Apply local refinement to improve convergence"""
        current_points = points.copy()
        current_ratio = calculate_ratio(current_points)
        
        # More sophisticated local search approach
        for iteration in range(iterations):
            improved = False
            
            # Try multiple small perturbations for each point
            for i in range(len(current_points)):
                for dim in range(3):
                    # Try several small steps in both directions
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
                            
            # If no improvement, reduce step size and try again
            if not improved:
                break
                
        return current_points
    
    # Create initial configuration
    points = create_initial_configuration()
    
    # Multiple optimization attempts to find the best configuration
    best_ratio = 0.0
    best_points = None
    
    # Run several optimization attempts with different starting points
    for attempt in range(15):
        # Create a new starting point with some randomness
        current_points = points.copy()
        
        # Add noise to break symmetry
        np.random.seed(attempt * 100 + 42)
        current_points += np.random.normal(0, 0.01, current_points.shape)
        
        # Normalize to unit sphere
        for i in range(len(current_points)):
            norm = np.linalg.norm(current_points[i])
            if norm > 0:
                current_points[i] = current_points[i] / norm
        
        # Apply multi-method optimization
        optimized_points = optimize_with_multiple_methods(current_points, maxiter=600)
        
        # Also try energy-based optimization
        energy_optimized = energy_based_optimization(optimized_points, maxiter=400)
        
        # Evaluate final result
        ratio = calculate_ratio(energy_optimized)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = energy_optimized.copy()
    
    # Final refinement
    if best_points is not None:
        # Apply final local refinement
        best_points = local_refinement(best_points, iterations=80)
        
        # Double-check the final ratio
        final_ratio = calculate_ratio(best_points)
        if final_ratio > best_ratio:
            best_ratio = final_ratio
    
    # If no good solution found, fall back to a good geometric configuration
    if best_points is None:
        best_points = create_initial_configuration()
    
    # Ensure points are within [0,1]^3
    best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
