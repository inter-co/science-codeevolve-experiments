# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import math
import random
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
from scipy.spatial import distance_matrix

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a physics-inspired particle system with repulsive forces combined with Fibonacci spiral construction.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Approach 1: Physics-inspired particle system with repulsive forces
    def particle_system_approach():
        """Use a physics-based approach where points repel each other with inverse-distance forces"""
        np.random.seed(42)  # For reproducibility
        
        # Initialize points randomly within unit square
        points = np.random.rand(16, 2)
        
        # Parameters for simulation
        num_iterations = 5000
        learning_rate = 0.01
        repulsion_strength = 1.0
        boundary_repulsion = 10.0
        
        for iteration in range(num_iterations):
            # Calculate distance matrix
            dist_matrix = distance_matrix(points, points)
            
            # Avoid division by zero by setting diagonal to large value
            np.fill_diagonal(dist_matrix, 1e10)
            
            # Calculate forces between all pairs of points
            # Force is inversely proportional to distance squared (repulsive)
            forces = np.zeros_like(points)
            
            for i in range(len(points)):
                for j in range(len(points)):
                    if i != j:
                        # Vector from point j to point i
                        diff = points[i] - points[j]
                        distance = np.linalg.norm(diff)
                        
                        # Repulsive force (inverse square law)
                        if distance > 1e-10:
                            force_magnitude = repulsion_strength / (distance ** 2)
                            force_direction = diff / distance
                            forces[i] += force_magnitude * force_direction
            
            # Add boundary repulsion to keep points within [0,1] x [0,1]
            for i in range(len(points)):
                boundary_force = np.zeros(2)
                # Repel from boundaries
                boundary_force[0] += boundary_repulsion * max(0, 1 - points[i, 0]) if points[i, 0] > 0.99 else 0
                boundary_force[0] -= boundary_repulsion * max(0, points[i, 0]) if points[i, 0] < 0.01 else 0
                boundary_force[1] += boundary_repulsion * max(0, 1 - points[i, 1]) if points[i, 1] > 0.99 else 0
                boundary_force[1] -= boundary_repulsion * max(0, points[i, 1]) if points[i, 1] < 0.01 else 0
                
                forces[i] += boundary_force
            
            # Update positions
            points += learning_rate * forces
            
            # Clamp points to [0,1] x [0,1]
            points = np.clip(points, 0, 1)
            
            # Early stopping based on convergence
            if iteration > 1000 and iteration % 100 == 0:
                # Check if forces are small enough
                max_force = np.max(np.linalg.norm(forces, axis=1))
                if max_force < 1e-6:
                    break
        
        return points
    
    # Approach 2: Fibonacci spiral-based arrangement
    def fibonacci_spiral_approach():
        """Create points using a Fibonacci spiral pattern with careful spacing"""
        # Generate points on a Fibonacci spiral
        points = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        # Create points in a way that approximates good dispersion
        for i in range(16):
            # Use Fibonacci-like distribution
            theta = i * 2 * math.pi / golden_ratio
            radius = math.sqrt(i / 15.0)  # Normalize to [0,1] range
            
            x = 0.5 + radius * math.cos(theta) * 0.4  # Center at (0.5, 0.5) with radius 0.4
            y = 0.5 + radius * math.sin(theta) * 0.4
            
            points.append([x, y])
        
        points = np.array(points)
        
        # Normalize to [0,1] x [0,1] if needed
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        if x_max > x_min and y_max > y_min:
            scale_x = 1.0 / (x_max - x_min)
            scale_y = 1.0 / (y_max - y_min)
            
            points[:, 0] = (points[:, 0] - x_min) * scale_x
            points[:, 1] = (points[:, 1] - y_min) * scale_y
            
            # Further adjust to avoid edge effects
            points[:, 0] = 0.1 + 0.8 * points[:, 0]
            points[:, 1] = 0.1 + 0.8 * points[:, 1]
        
        return points
    
    # Approach 3: Enhanced local optimization with simulated annealing
    def simulated_annealing_approach():
        """Use simulated annealing to refine point placement"""
        # Start with Fibonacci spiral
        points = fibonacci_spiral_approach()
        
        def calculate_ratio(points):
            distances = pdist(points)
            if len(distances) == 0:
                return 0
            d_min = np.min(distances)
            d_max = np.max(distances)
            if d_max == 0:
                return 0
            return d_min / d_max
        
        current_points = points.copy()
        current_ratio = calculate_ratio(current_points)
        
        # Simulated annealing parameters
        temperature = 1.0
        cooling_rate = 0.999
        min_temperature = 1e-6
        iterations_per_temp = 100
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        while temperature > min_temperature:
            for _ in range(iterations_per_temp):
                # Make a small random perturbation to one point
                point_idx = random.randint(0, len(current_points) - 1)
                new_points = current_points.copy()
                
                # Perturb this point
                delta = np.random.normal(0, 0.005, 2)
                new_points[point_idx] = current_points[point_idx] + delta
                
                # Keep within bounds
                new_points[point_idx] = np.clip(new_points[point_idx], 0, 1)
                
                # Calculate new ratio
                new_ratio = calculate_ratio(new_points)
                
                # Accept or reject based on Metropolis criterion
                if new_ratio > current_ratio or random.random() < math.exp((new_ratio - current_ratio) / temperature):
                    current_points = new_points
                    current_ratio = new_ratio
                    
                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_points = current_points.copy()
            
            temperature *= cooling_rate
        
        return best_points
    
    # Try multiple approaches and return the best
    approaches = [
        ("particle_system", particle_system_approach),
        ("fibonacci_spiral", fibonacci_spiral_approach),
        ("simulated_annealing", simulated_annealing_approach)
    ]
    
    best_points = None
    best_ratio = 0
    
    for name, approach_func in approaches:
        try:
            points = approach_func()
            ratio = calculate_ratio(points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = points.copy()
        except Exception as e:
            continue
    
    # Final refinement using coordinate descent
    if best_points is not None:
        def coordinate_descent_refinement(initial_points):
            def calculate_ratio(points):
                distances = pdist(points)
                if len(distances) == 0:
                    return 0
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max == 0:
                    return 0
                return d_min / d_max
            
            points = initial_points.copy()
            current_ratio = calculate_ratio(points)
            
            # Coordinate descent with adaptive steps
            for iteration in range(2000):
                improved = False
                for i in range(len(points)):
                    original_point = points[i].copy()
                    
                    # Try several perturbations in each dimension
                    for dim in range(2):
                        best_perturbed = original_point.copy()
                        best_ratio = current_ratio
                        
                        for delta in [-0.002, -0.001, 0.001, 0.002]:
                            new_point = original_point.copy()
                            new_point[dim] += delta
                            new_point = np.clip(new_point, 0, 1)
                            
                            points[i] = new_point
                            ratio = calculate_ratio(points)
                            
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_perturbed = new_point.copy()
                            
                            points[i] = original_point  # Restore
                        
                        # Apply best perturbation
                        points[i] = best_perturbed
                        current_ratio = best_ratio
                        improved = True
                
                # Stop early if no improvement
                if not improved:
                    break
                    
            return points
        
        best_points = coordinate_descent_refinement(best_points)
    
    # Ensure we have a valid result
    if best_points is None:
        # Fallback to Fibonacci spiral if all approaches failed
        best_points = fibonacci_spiral_approach()
    
    return best_points

def calculate_ratio(points):
    """Calculate the ratio of minimum to maximum distance"""
    distances = pdist(points)
    if len(distances) == 0:
        return 0
    d_min = np.min(distances)
    d_max = np.max(distances)
    if d_max == 0:
        return 0
    return d_min / d_max


# EVOLVE-BLOCK-END
