# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import dual_annealing, minimize
import math
from scipy.spatial import ConvexHull
import time
from numba import jit
from sklearn.cluster import KMeans
from scipy.spatial import distance

@jit(nopython=True)
def compute_distances_numba(points):
    """Compute pairwise distances efficiently using numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric insights with advanced optimization techniques.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Initialize with a more sophisticated starting configuration
    points = initialize_better_config()
    
    # Define the objective function for optimization
    def objective(x_flat):
        points = x_flat.reshape(-1, 2)
        # Use faster numba version for distance computation
        distances = compute_distances_numba(points)
        
        # Avoid division by zero
        mask = distances > 0
        if not np.any(mask):
            return -1e10  # Very bad score if no valid distances
            
        min_dist = np.min(distances[mask])
        max_dist = np.max(distances)
        
        # Handle edge case where max_dist is very small
        if max_dist < 1e-12:
            return -1e10
            
        ratio = min_dist / max_dist
        return -ratio  # Negative because we want to maximize
    
    # Define bounds for all coordinates (0 to 1)
    bounds = [(0, 1)] * 32  # 16 points * 2 coordinates each
    
    # First phase: Global optimization with dual annealing - more aggressive settings
    start_time = time.time()
    
    # Use fewer iterations but higher quality sampling with better parameters
    result = dual_annealing(
        objective, 
        bounds, 
        maxiter=2000,  # More iterations for better exploration
        initial_temp=2000,  # Higher initial temperature
        restart_temp_ratio=1e-5,
        visit=2.62,
        accept=-5.0,
        seed=42,
        no_local_search=False
    )
    
    optimized_points = result.x.reshape(-1, 2)
    
    # Ensure all points are within bounds
    optimized_points = np.clip(optimized_points, 0, 1)
    
    # Second phase: Local refinement with multiple strategies
    best_points = optimized_points.copy()
    best_ratio = -objective(optimized_points.flatten())
    
    # Strategy 1: L-BFGS-B optimization with better convergence criteria
    try:
        local_result = minimize(
            objective,
            optimized_points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1)] * 32,
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
        )
        
        if local_result.success:
            refined_points = local_result.x.reshape(-1, 2)
            refined_points = np.clip(refined_points, 0, 1)
            refined_ratio = -objective(refined_points.flatten())
            
            if refined_ratio > best_ratio:
                best_points = refined_points
                best_ratio = refined_ratio
                
    except Exception:
        pass
    
    # Strategy 2: Improved gradient-based refinement using analytical gradients
    try:
        # More robust gradient computation with better numerical stability
        current_points = optimized_points.copy()
        current_ratio = -objective(current_points.flatten())
        
        # Use a simpler but more reliable gradient-based approach
        step_size = 0.001  # Even smaller step size for stability
        max_iterations = 500
        patience = 15
        patience_counter = 0
        
        for iteration in range(max_iterations):
            # Simple gradient approximation using central differences
            epsilon = 1e-6
            gradients = np.zeros_like(current_points)
            
            # Compute gradient for each point
            for i in range(16):
                for j in range(2):  # x and y coordinates
                    # Perturb coordinate in both directions
                    perturbed_plus = current_points.copy()
                    perturbed_minus = current_points.copy()
                    perturbed_plus[i, j] += epsilon
                    perturbed_minus[i, j] -= epsilon
                    
                    # Clamp to bounds
                    perturbed_plus = np.clip(perturbed_plus, 0, 1)
                    perturbed_minus = np.clip(perturbed_minus, 0, 1)
                    
                    # Compute central difference
                    plus_ratio = -objective(perturbed_plus.flatten())
                    minus_ratio = -objective(perturbed_minus.flatten())
                    gradients[i, j] = (plus_ratio - minus_ratio) / (2 * epsilon)
            
            # Update with gradient descent
            new_points = current_points - step_size * gradients
            
            # Clamp to bounds
            new_points = np.clip(new_points, 0, 1)
            
            # Accept improvement
            new_ratio = -objective(new_points.flatten())
            if new_ratio > current_ratio:
                current_points = new_points
                current_ratio = new_ratio
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break  # Early termination if no improvement
            
            # Early termination if time is running out
            if time.time() - start_time > 55:  # Leave some buffer
                break
        
        if current_ratio > best_ratio:
            best_points = current_points
            best_ratio = current_ratio
            
    except Exception:
        pass
    
    # Third phase: Enhanced simulated annealing for final polishing
    try:
        # Use a more effective annealing schedule
        temp = 0.02  # Lower initial temperature for more focused search
        cooling_rate = 0.995
        max_iter = 2000  # More iterations for better exploration
        
        current_points = best_points.copy()
        current_ratio = best_ratio
        
        for iteration in range(max_iter):
            # Make a small random perturbation
            candidate_points = current_points.copy()
            idx = np.random.randint(0, 16)
            dim = np.random.randint(0, 2)
            
            # Small random move with adaptive magnitude
            move_magnitude = 0.0005 * (temp / 0.02)  # Smaller moves when temp is low
            candidate_points[idx, dim] += np.random.normal(0, move_magnitude)
            
            # Clamp to bounds
            candidate_points = np.clip(candidate_points, 0, 1)
            
            candidate_ratio = -objective(candidate_points.flatten())
            
            # Accept with Metropolis criterion
            if candidate_ratio > current_ratio or np.random.rand() < np.exp((candidate_ratio - current_ratio) / temp):
                current_points = candidate_points
                current_ratio = candidate_ratio
                
            temp *= cooling_rate
            
            # Early termination if time is running out
            if time.time() - start_time > 55:
                break
        
        if current_ratio > best_ratio:
            best_points = current_points
            best_ratio = current_ratio
            
    except Exception:
        pass
    
    # Fourth phase: Local search with coordinate descent for fine-tuning
    try:
        # Coordinate descent for further improvement
        current_points = best_points.copy()
        current_ratio = best_ratio
        
        for _ in range(500):  # More iterations for better tuning
            improved = False
            # Try moving each point in both directions
            for i in range(16):
                for j in range(2):  # x and y coordinates
                    # Try small moves in positive and negative direction
                    for delta in [-0.0005, 0.0005]:  # Smaller steps for precision
                        candidate_points = current_points.copy()
                        candidate_points[i, j] += delta
                        candidate_points = np.clip(candidate_points, 0, 1)
                        
                        candidate_ratio = -objective(candidate_points.flatten())
                        
                        if candidate_ratio > current_ratio:
                            current_points = candidate_points
                            current_ratio = candidate_ratio
                            improved = True
                            
            if not improved:
                break  # No more improvements
                
            if time.time() - start_time > 55:
                break
                
        if current_ratio > best_ratio:
            best_points = current_points
            best_ratio = current_ratio
            
    except Exception:
        pass
    
    # Fifth phase: Try a few more local searches with different strategies
    try:
        # Try a simple hill climbing approach from the best solution
        current_points = best_points.copy()
        current_ratio = best_ratio
        
        # Run a few rounds of coordinate descent
        for round_num in range(3):
            for _ in range(100):  # Fewer iterations per round
                improved = False
                # Try moving each point in both directions
                for i in range(16):
                    for j in range(2):  # x and y coordinates
                        # Try small moves in positive and negative direction
                        for delta in [-0.0002, 0.0002]:
                            candidate_points = current_points.copy()
                            candidate_points[i, j] += delta
                            candidate_points = np.clip(candidate_points, 0, 1)
                            
                            candidate_ratio = -objective(candidate_points.flatten())
                            
                            if candidate_ratio > current_ratio:
                                current_points = candidate_points
                                current_ratio = candidate_ratio
                                improved = True
                                
                if not improved:
                    break  # No more improvements
                    
                if time.time() - start_time > 55:
                    break
                    
        if current_ratio > best_ratio:
            best_points = current_points
            best_ratio = current_ratio
            
    except Exception:
        pass
    
    # Final check: if we have a better configuration, return it
    return best_points


def initialize_better_config():
    """Initialize points using a more sophisticated approach based on known good configurations."""
    # Start with a known good configuration - a 4x4 grid with perturbations
    points = []
    
    # Generate a 4x4 grid with some perturbations
    for i in range(4):
        for j in range(4):
            # Create a grid with proper spacing
            x = 0.1 + j * 0.225
            y = 0.1 + i * 0.225
            
            # Add small random perturbations to avoid regular patterns
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            
            points.append([x, y])
    
    # Ensure we have exactly 16 points
    points = points[:16]
    points_array = np.array(points)
    
    # Apply a more systematic approach for better spreading
    # Use a combination of grid and strategic placement
    # Place points in a way that maximizes minimum distance
    
    # Create a more sophisticated initial layout
    # Start with a regular grid and slightly perturb for better dispersion
    base_grid = []
    for i in range(4):
        for j in range(4):
            x = 0.1 + j * 0.225
            y = 0.1 + i * 0.225
            base_grid.append([x, y])
    
    # Add some strategic corner points
    corner_points = [
        [0.05, 0.05], [0.95, 0.95], [0.05, 0.95], [0.95, 0.05],
        [0.5, 0.5], [0.25, 0.25], [0.75, 0.75], [0.25, 0.75],
        [0.75, 0.25], [0.1, 0.8], [0.8, 0.1], [0.1, 0.1],
        [0.8, 0.8], [0.3, 0.6], [0.6, 0.3], [0.4, 0.4]
    ]
    
    # Mix the two approaches with weighted selection
    points_array = np.array(base_grid)
    
    # Add some corner points strategically
    for i in range(min(16, len(corner_points))):
        points_array[i] = corner_points[i]
    
    # Add slight jitter to all points to avoid degenerate cases
    for i in range(len(points_array)):
        points_array[i][0] += np.random.normal(0, 0.005)
        points_array[i][1] += np.random.normal(0, 0.005)
    
    # Clip to bounds
    points_array = np.clip(points_array, 0, 1)
    
    return points_array


# EVOLVE-BLOCK-END
