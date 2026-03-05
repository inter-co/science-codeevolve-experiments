# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a physics-inspired Monte Carlo approach with simulated annealing and energy minimization.
    
    This approach models points as charged particles that repel each other, finding equilibrium
    through a simulated annealing process that gradually reduces temperature to converge to 
    a stable configuration that maximizes the minimum-to-maximum distance ratio.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Physics-inspired approach: simulate repulsive forces between points
    def compute_energy_and_forces(points):
        """Compute total potential energy and forces acting on each point"""
        n = len(points)
        # Compute distance matrix
        dist_matrix = squareform(pdist(points))
        
        # Avoid division by zero
        np.fill_diagonal(dist_matrix, 1e-10)
        
        # Compute forces (repulsion proportional to 1/r^2)
        # But we'll also consider the ratio maximization objective
        energies = []
        forces = np.zeros_like(points)
        
        # Compute energy based on inverse distance (repulsive force)
        # We're trying to maximize min/max ratio, so we'll use a proxy
        # that encourages both large minimum distances and small maximum distances
        total_energy = 0
        
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                dist_sq = dx*dx + dy*dy
                
                # Avoid division by zero
                if dist_sq < 1e-15:
                    continue
                    
                # Energy is inversely proportional to distance squared (repulsive)
                energy = 1.0 / dist_sq
                total_energy += energy
                
                # Force magnitude (inverse square law)
                force_magnitude = energy / dist_sq
                
                # Force vector
                fx = force_magnitude * dx
                fy = force_magnitude * dy
                
                # Apply forces (opposite directions)
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
        
        return total_energy, forces
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio"""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 1e-10:
            return 0.0
            
        return min_dist / max_dist
    
    def simulated_annealing_optimization():
        """Main optimization loop using simulated annealing"""
        # Initialize points randomly in [0,1] x [0,1]
        points = np.random.rand(16, 2)
        
        # Initial temperature and cooling schedule
        temp = 0.1
        cooling_rate = 0.995
        min_temp = 1e-6
        steps_per_temp = 50
        
        # Track best solution
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        # Annealing loop
        step = 0
        max_steps = 10000  # Limit iterations
        
        while temp > min_temp and step < max_steps:
            # Take several steps at current temperature
            for _ in range(steps_per_temp):
                # Make small random perturbations to points
                new_points = points.copy()
                
                # Pick a random point to perturb
                idx = random.randint(0, 15)
                # Small random displacement
                new_points[idx, 0] += random.uniform(-0.01, 0.01)
                new_points[idx, 1] += random.uniform(-0.01, 0.01)
                
                # Keep points within bounds
                new_points[:, 0] = np.clip(new_points[:, 0], 0, 1)
                new_points[:, 1] = np.clip(new_points[:, 1], 0, 1)
                
                # Compute ratios
                old_ratio = compute_min_max_ratio(points)
                new_ratio = compute_min_max_ratio(new_points)
                
                # Accept or reject based on Metropolis criterion
                if new_ratio > old_ratio:
                    points = new_points
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_points = points.copy()
                else:
                    # Sometimes accept worse moves to escape local minima
                    delta = new_ratio - old_ratio
                    acceptance_prob = math.exp(delta / temp)
                    if random.random() < acceptance_prob:
                        points = new_points
                        
            # Cool down
            temp *= cooling_rate
            step += 1
            
        return best_points
    
    # Alternative approach: gradient-based optimization with proper initialization
    def gradient_based_optimization():
        """Use gradient-based optimization with good initialization"""
        # Start with a hexagonal arrangement (known good starting point)
        points = []
        rows = 4
        cols = 4
        spacing = 0.3
        
        for i in range(rows):
            for j in range(cols):
                if len(points) < 16:
                    offset = spacing * 0.5 if i % 2 == 1 else 0
                    points.append([j * spacing + offset, i * spacing * math.sqrt(3)/2])
        
        # Normalize to fit in [0,1] x [0,1]
        if points:
            points = np.array(points)
            max_val = np.max(points)
            if max_val > 0:
                points = points / max_val
            points = np.clip(points, 0, 1)
        
        # Add some randomness to avoid local minima
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        # Simple gradient descent approach
        learning_rate = 0.01
        max_iterations = 5000
        tolerance = 1e-8
        
        for iteration in range(max_iterations):
            # Compute current ratio
            current_ratio = compute_min_max_ratio(points)
            
            # Compute gradients numerically (simple finite differences)
            gradients = np.zeros_like(points)
            epsilon = 1e-6
            
            for i in range(16):
                for j in range(2):  # x and y coordinates
                    # Perturb point slightly
                    points_plus = points.copy()
                    points_minus = points.copy()
                    points_plus[i, j] += epsilon
                    points_minus[i, j] -= epsilon
                    
                    # Clip to bounds
                    points_plus[:, 0] = np.clip(points_plus[:, 0], 0, 1)
                    points_plus[:, 1] = np.clip(points_plus[:, 1], 0, 1)
                    points_minus[:, 0] = np.clip(points_minus[:, 0], 0, 1)
                    points_minus[:, 1] = np.clip(points_minus[:, 1], 0, 1)
                    
                    # Compute finite difference approximation
                    ratio_plus = compute_min_max_ratio(points_plus)
                    ratio_minus = compute_min_max_ratio(points_minus)
                    gradients[i, j] = (ratio_plus - ratio_minus) / (2 * epsilon)
            
            # Update points
            points += learning_rate * gradients
            points[:, 0] = np.clip(points[:, 0], 0, 1)
            points[:, 1] = np.clip(points[:, 1], 0, 1)
            
            # Check convergence
            if np.linalg.norm(gradients) < tolerance:
                break
                
        return points
    
    # Run both approaches and return the better one
    try:
        # Try the physics-inspired simulated annealing approach
        sa_result = simulated_annealing_optimization()
        sa_ratio = compute_min_max_ratio(sa_result)
        
        # Try the gradient-based approach  
        grad_result = gradient_based_optimization()
        grad_ratio = compute_min_max_ratio(grad_result)
        
        # Return the better result
        if sa_ratio >= grad_ratio:
            return sa_result
        else:
            return grad_result
            
    except Exception as e:
        # Fallback to a simple hexagonal arrangement
        points = []
        rows = 4
        cols = 4
        spacing = 0.3
        
        for i in range(rows):
            for j in range(cols):
                if len(points) < 16:
                    offset = spacing * 0.5 if i % 2 == 1 else 0
                    points.append([j * spacing + offset, i * spacing * math.sqrt(3)/2])
        
        if points:
            points = np.array(points)
            max_val = np.max(points)
            if max_val > 0:
                points = points / max_val
            points = np.clip(points, 0, 1)
            return points
        else:
            # Last resort: random points
            return np.random.rand(16, 2)


# EVOLVE-BLOCK-END
