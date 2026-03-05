# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import random
import math
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    Uses energy-based optimization with geometric initialization and simulated annealing for global search.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        distances = pdist(points)
        if len(distances) == 0 or np.max(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist != 0 else 0
    
    # Energy-based optimization approach inspired by INSPIRATION 1
    def energy_model(points, alpha=2.0):
        """Calculate total repulsive energy between points"""
        distances = pdist(points)
        # Avoid division by zero and penalize very small distances
        distances = np.maximum(distances, 1e-10)
        # Energy is sum of inverse distance powers (repulsive force)
        energy = np.sum(1.0 / (distances ** alpha))
        return energy
    
    def gradient_energy(points, alpha=2.0):
        """Compute gradient of energy with respect to point positions"""
        n = len(points)
        grad = np.zeros_like(points)
        
        # For each point, compute contribution to gradient from all other points
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    if dist_sq > 1e-12:  # Avoid numerical issues
                        dist = np.sqrt(dist_sq)
                        # Gradient of 1/dist^alpha w.r.t. point i
                        factor = alpha * (1.0 / (dist**(alpha+2))) 
                        grad[i] += factor * diff
        
        return grad
    
    # Generate initial configuration using improved geometric construction
    def generate_initial_configuration():
        """Create initial point configuration based on mathematical principles"""
        # Use a more sophisticated approach inspired by discrete spherical codes
        # and the golden ratio for better distribution
        
        # Start with a more structured approach
        points = []
        
        # Create points in a pattern that mimics good spherical codes
        # Use a combination of grid and perturbed positions
        
        # Grid-based construction with perturbations
        for i in range(4):
            for j in range(4):
                # Create a 4x4 grid with slight perturbations
                x = j / 3.0  # Normalized to [0,1]
                y = i / 3.0  # Normalized to [0,1]
                
                # Add small perturbations to break symmetry
                # Use golden ratio related perturbations for better distribution
                perturbation_x = 0.02 * np.sin(i * np.pi/2) * np.cos(j * np.pi/2)
                perturbation_y = 0.02 * np.cos(i * np.pi/2) * np.sin(j * np.pi/2)
                
                x += perturbation_x
                y += perturbation_y
                
                # Ensure within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                
                points.append([x, y])
        
        points = np.array(points[:16])
        return points
    
    # Enhanced optimization using gradient descent with adaptive step size
    def optimize_with_gradient_descent(points, max_iter=1000, tolerance=1e-6):
        """Optimize point configuration using gradient descent with momentum"""
        current_points = points.copy()
        velocity = np.zeros_like(current_points)
        momentum = 0.9
        learning_rate = 0.01
        
        prev_energy = float('inf')
        
        for iteration in range(max_iter):
            # Compute energy and gradient
            current_energy = energy_model(current_points)
            
            # Check for convergence
            if abs(prev_energy - current_energy) < tolerance:
                break
                
            prev_energy = current_energy
            
            # Compute gradient
            grad = gradient_energy(current_points)
            
            # Apply momentum and update
            velocity = momentum * velocity - learning_rate * grad
            current_points += velocity
            
            # Project back to valid bounds [0,1]
            current_points = np.clip(current_points, 0, 1)
            
            # Occasionally reproject to avoid drift
            if iteration % 50 == 0:
                current_points = np.clip(current_points, 0, 1)
                
        return current_points
    
    # Enhanced simulated annealing approach
    def simulated_annealing_optimization(initial_points):
        """Use simulated annealing for robust global optimization"""
        current_points = initial_points.copy()
        current_ratio = compute_min_max_ratio(current_points)
        
        # Simulated annealing parameters - optimized for better performance
        temperature = 1.0
        cooling_rate = 0.995
        min_temperature = 1e-6
        iterations_per_temp = 100
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Annealing loop
        while temperature > min_temperature:
            for _ in range(iterations_per_temp):
                # Make small random perturbations
                new_points = current_points.copy()
                idx = np.random.randint(0, 16)
                # Use smaller perturbations for fine-tuning
                new_points[idx] += np.random.normal(0, 0.005, 2)
                new_points = np.clip(new_points, 0, 1)
                
                # Accept or reject based on energy change
                new_ratio = compute_min_max_ratio(new_points)
                
                if new_ratio > current_ratio:
                    current_points = new_points
                    current_ratio = new_ratio
                    
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_points = new_points.copy()
                else:
                    # Accept with probability based on temperature
                    delta = new_ratio - current_ratio
                    if np.random.random() < np.exp(delta / temperature):
                        current_points = new_points
                        current_ratio = new_ratio
            
            temperature *= cooling_rate
        
        return best_points
    
    # Multi-start optimization with better strategy
    def multi_start_optimization():
        """Run multiple optimizations from different starting points"""
        best_points = None
        best_ratio = 0
        
        # Try multiple random initializations to find global optimum
        for attempt in range(30):  # Increase attempts for better exploration
            np.random.seed(attempt * 42 + 12345)
            
            # Generate initial configuration
            initial_points = generate_initial_configuration()
            
            # Optimize using gradient descent
            optimized_points = optimize_with_gradient_descent(
                initial_points, 
                max_iter=400,  # Slightly more iterations for better convergence
                tolerance=1e-5
            )
            
            # Evaluate quality
            ratio = compute_min_max_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
        
        return best_points if best_points is not None else generate_initial_configuration()
    
    # Main optimization pipeline
    try:
        # Start with improved mathematical construction
        initial_points = generate_initial_configuration()
        
        # Optimize with gradient descent
        gd_points = optimize_with_gradient_descent(initial_points, max_iter=500)
        
        # Refine with simulated annealing for global optimization
        sa_points = simulated_annealing_optimization(gd_points)
        
        # Final refinement with gradient descent
        final_points = optimize_with_gradient_descent(sa_points, max_iter=300)
        
        # Compare results and return best
        ratios = [
            compute_min_max_ratio(initial_points),
            compute_min_max_ratio(gd_points),
            compute_min_max_ratio(sa_points),
            compute_min_max_ratio(final_points)
        ]
        
        best_idx = np.argmax(ratios)
        return [initial_points, gd_points, sa_points, final_points][best_idx]
        
    except Exception as e:
        # Fallback to basic configuration
        points = generate_initial_configuration()
        return points


# EVOLVE-BLOCK-END
