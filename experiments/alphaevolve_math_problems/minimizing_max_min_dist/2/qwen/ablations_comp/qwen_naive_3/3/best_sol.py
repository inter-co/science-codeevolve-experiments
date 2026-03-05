# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
import time
from scipy.spatial import distance_matrix
import math
from itertools import combinations


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid optimization approach combining multiple strategies.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # More sophisticated initialization using circle packing principles
    def initialize_points():
        # Start with a regular 4x4 grid pattern
        points = []
        spacing = 1.0 / 3.0  # Spacing to fit in [0,1] x [0,1]
        
        # Create grid points
        for i in range(4):
            for j in range(4):
                x = i * spacing
                y = j * spacing
                points.append([x, y])
        
        # Add small random perturbations to avoid degeneracy
        points = np.array(points)
        noise = np.random.normal(0, 0.02, (len(points), 2))
        points += noise
        
        # Clip to bounds
        points = np.clip(points, 0, 1)
        
        # Additional refinement: move points away from boundaries to improve ratios
        for i in range(len(points)):
            # Move points slightly away from edges  
            points[i][0] = 0.05 + 0.9 * points[i][0]  # Scale to [0.05, 0.95]
            points[i][1] = 0.05 + 0.9 * points[i][1]
        
        return points
    
    # Improved objective function that focuses on the key ratio
    def objective_ratio(points_flat):
        points = points_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return 0.0
            
        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return -np.inf
            
        # Return negative ratio (since we're minimizing to maximize ratio)
        return -min_dist / max_dist
    
    # Enhanced force model approach
    def force_model(points_flat, k_repel=1.0, k_attr=0.05):
        """Compute forces between points based on distance with improved physics"""
        points = points_flat.reshape(-1, 2)
        n_points = len(points)
        
        # Compute distance matrix
        dist_matrix = distance_matrix(points, points)
        
        # Initialize forces
        forces = np.zeros_like(points)
        
        # Repulsive forces (inverse square law for small distances, but with better scaling)
        for i in range(n_points):
            for j in range(i+1, n_points):
                if i != j:
                    dist = dist_matrix[i, j]
                    if dist > 0:
                        # Repulsive force with better scaling
                        force_magnitude = k_repel / (dist * dist + 1e-10)
                        direction = points[j] - points[i]
                        direction = direction / (np.linalg.norm(direction) + 1e-10)
                        forces[i] += force_magnitude * direction
                        forces[j] -= force_magnitude * direction
        
        # Attractive forces (to keep points within bounds)
        for i in range(n_points):
            # Pull towards center with some damping
            center_pull = -k_attr * points[i]
            forces[i] += center_pull
            
        return forces.flatten()
    
    # Gradient-based optimization with adaptive learning rate
    def optimize_with_gradient_descent(initial_points, max_iter=1000):
        points = initial_points.copy().flatten()
        velocity = np.zeros_like(points)
        learning_rate = 0.01
        momentum = 0.9
        decay_rate = 0.999
        
        best_points = points.copy()
        best_ratio = -np.inf
        
        prev_gradients = np.zeros_like(points)
        
        for iteration in range(max_iter):
            # Compute current ratio
            current_ratio = -objective_ratio(points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()
            
            # Compute gradients numerically using finite differences
            eps = 1e-6
            gradients = np.zeros_like(points)
            
            for i in range(len(points)):
                points_plus = points.copy()
                points_minus = points.copy()
                points_plus[i] += eps
                points_minus[i] -= eps
                
                grad = (objective_ratio(points_plus) - objective_ratio(points_minus)) / (2 * eps)
                gradients[i] = grad
            
            # Adaptive learning rate based on gradient changes
            if iteration > 0:
                grad_change = np.mean(np.abs(gradients - prev_gradients))
                if grad_change < 1e-8:
                    learning_rate *= 0.9  # Reduce learning rate if stuck
            
            # Update velocity and positions with momentum
            velocity = momentum * velocity - learning_rate * gradients
            points += velocity
            
            # Apply boundary constraints
            points = np.clip(points, 0, 1)
            
            # Decay learning rate
            learning_rate *= decay_rate
            
            # Early stopping if improvement is minimal
            if iteration > 100 and abs(gradients).mean() < 1e-8:
                break
                
            prev_gradients = gradients.copy()
                
        return best_points.reshape(-1, 2), best_ratio
    
    # Global optimization approach with multiple restarts
    def optimize_global(initial_points):
        best_points = initial_points.copy()
        best_ratio = -np.inf
        
        # Multiple restarts with different initializations
        for restart in range(5):
            # Perturb the initial points slightly
            perturbed = initial_points + np.random.normal(0, 0.01, initial_points.shape)
            perturbed = np.clip(perturbed, 0, 1)
            
            # Try different optimization methods
            try:
                # Method 1: L-BFGS-B
                bounds = [(0, 1) for _ in range(2 * n)]
                result = minimize(
                    objective_ratio,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    test_points = result.x.reshape(-1, 2)
                    test_ratio = -objective_ratio(result.x)
                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_points = test_points.copy()
                        
            except Exception:
                pass
                
            # Method 2: Nelder-Mead as fallback
            try:
                result = minimize(
                    objective_ratio,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'adaptive': True, 'xtol': 1e-10, 'ftol': 1e-10}
                )
                
                if result.success:
                    test_points = result.x.reshape(-1, 2)
                    test_ratio = -objective_ratio(result.x)
                    if test_ratio > best_ratio:
                        best_ratio = test_ratio
                        best_points = test_points.copy()
                        
            except Exception:
                pass
        
        return best_points, best_ratio
    
    # Main optimization flow
    # Step 1: Initialize points with better strategy
    initial_points = initialize_points()
    
    # Step 2: Try multiple optimization approaches
    # Approach 1: Gradient descent
    optimized_points_gd, ratio_gd = optimize_with_gradient_descent(initial_points, max_iter=500)
    
    # Approach 2: Global optimization with restarts
    optimized_points_global, ratio_global = optimize_global(initial_points)
    
    # Choose the best result
    if ratio_global > ratio_gd:
        final_points = optimized_points_global
        final_ratio = ratio_global
    else:
        final_points = optimized_points_gd
        final_ratio = ratio_gd
    
    # Step 3: Final refinement with more aggressive optimization
    try:
        bounds = [(0, 1) for _ in range(2 * n)]
        
        # Try with different optimization methods
        methods = ['L-BFGS-B', 'TNC', 'SLSQP']
        for method in methods:
            try:
                result = minimize(
                    objective_ratio,
                    final_points.flatten(),
                    method=method,
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if result.success:
                    test_points = result.x.reshape(-1, 2)
                    test_ratio = -objective_ratio(result.x)
                    if test_ratio > final_ratio:
                        final_points = test_points
                        final_ratio = test_ratio
            except Exception:
                continue
                
    except Exception:
        pass
    
    # Final verification
    distances = pdist(final_points)
    if len(distances) > 0:
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist > 0:
            actual_ratio = min_dist / max_dist
            if actual_ratio > 0.2786:  # Beat the benchmark
                pass  # Good result
            else:
                # If still not good enough, try a different approach
                # Generate a better initial configuration
                better_initial = generate_better_initial()
                try:
                    # Try again with better initial points
                    bounds = [(0, 1) for _ in range(2 * n)]
                    result = minimize(
                        objective_ratio,
                        better_initial.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                    )
                    
                    if result.success:
                        test_points = result.x.reshape(-1, 2)
                        test_distances = pdist(test_points)
                        if len(test_distances) > 0:
                            test_min = np.min(test_distances)
                            test_max = np.max(test_distances)
                            if test_max > 0:
                                test_ratio = test_min / test_max
                                if test_ratio > final_ratio:
                                    final_points = test_points
                except Exception:
                    pass
    
    return final_points


def generate_better_initial():
    """Generate an even better initial configuration"""
    # Try a hexagonal lattice arrangement which often works well for point distributions
    points = []
    
    # Create a pattern that resembles a hexagonal close packing
    rows = 4
    cols = 4
    
    for i in range(rows):
        for j in range(cols):
            # Offset every other row
            offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + offset) * (1.0 / 3.0)
            y = i * (1.0 / 3.0)
            points.append([x, y])
    
    # Add small random perturbations
    points = np.array(points)
    noise = np.random.normal(0, 0.015, (len(points), 2))
    points += noise
    points = np.clip(points, 0, 1)
    
    # Adjust to avoid edge clustering
    for i in range(len(points)):
        points[i][0] = 0.05 + 0.9 * points[i][0]
        points[i][1] = 0.05 + 0.9 * points[i][1]
    
    return points


# EVOLVE-BLOCK-END
