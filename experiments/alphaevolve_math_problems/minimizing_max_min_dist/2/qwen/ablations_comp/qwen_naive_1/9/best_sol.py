# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random
import time
from scipy.spatial import ConvexHull

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining smart initialization, local optimization, and advanced global search
    to beat the benchmark.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Problem constants
    n_points = 16
    n_dimensions = 2
    
    def calculate_ratio(points):
        """Calculate the min/max distance ratio for given points"""
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist > 0:
            return min_dist / max_dist
        else:
            return 0
    
    # Better initialization using a more structured approach
    def initialize_points():
        # Create a more sophisticated initial configuration
        points = []
        
        # Strategy: place points in a combination of regular grid and perturbed positions
        # Start with a regular 4x4 grid pattern
        grid_positions = []
        for i in range(4):
            for j in range(4):
                grid_positions.append([i * 0.25, j * 0.25])
        
        # Add some perturbation to avoid degenerate cases
        for pos in grid_positions:
            x, y = pos
            # Add small random perturbations
            x += random.uniform(-0.02, 0.02)
            y += random.uniform(-0.02, 0.02)
            # Keep within bounds
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            points.append([x, y])
        
        return np.array(points)
    
    # Enhanced local optimization using multiple methods
    def local_optimization(initial_points):
        def objective(params):
            points_flat = params.reshape(-1, 2)
            distances = pdist(points_flat)
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                return -min_dist / max_dist  # Negative because we want to maximize
            else:
                return 0
        
        # Flatten the points for optimization
        start_params = initial_points.flatten()
        
        # Try multiple optimization approaches
        best_points = initial_points.copy()
        best_ratio = calculate_ratio(best_points)
        
        # Method 1: L-BFGS-B
        try:
            result = minimize(
                objective,
                start_params,
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(len(start_params))],
                options={'maxiter': 300, 'ftol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points[:, 0] = np.clip(optimized_points[:, 0], 0, 1)
                optimized_points[:, 1] = np.clip(optimized_points[:, 1], 0, 1)
                ratio = calculate_ratio(optimized_points)
                if ratio > best_ratio:
                    best_points = optimized_points
                    best_ratio = ratio
        except:
            pass
        
        # Method 2: Nelder-Mead (more robust fallback)
        try:
            result = minimize(
                objective,
                start_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'fatol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points[:, 0] = np.clip(optimized_points[:, 0], 0, 1)
                optimized_points[:, 1] = np.clip(optimized_points[:, 1], 0, 1)
                ratio = calculate_ratio(optimized_points)
                if ratio > best_ratio:
                    best_points = optimized_points
                    best_ratio = ratio
        except:
            pass
            
        return best_points
    
    # Improved Simulated Annealing with better cooling schedule
    def improved_simulated_annealing():
        current_points = initialize_points()
        current_ratio = calculate_ratio(current_points)
        
        # Parameters for improved SA
        temp = 0.05  # Lower initial temperature
        cooling_rate = 0.999  # Faster cooling
        min_temp = 1e-8
        max_iterations = 15000
        
        for iteration in range(max_iterations):
            # Generate neighbor solution - perturb multiple points
            new_points = current_points.copy()
            
            # Perturb 2-3 points at random
            num_perturbations = random.randint(2, 3)
            for _ in range(num_perturbations):
                idx = random.randint(0, n_points - 1)
                new_points[idx, 0] += random.uniform(-0.015, 0.015)
                new_points[idx, 1] += random.uniform(-0.015, 0.015)
                
                # Keep within bounds
                new_points[idx, 0] = max(0, min(1, new_points[idx, 0]))
                new_points[idx, 1] = max(0, min(1, new_points[idx, 1]))
            
            # Calculate new ratio
            new_ratio = calculate_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or random.random() < np.exp((new_ratio - current_ratio) / temp):
                current_points = new_points
                current_ratio = new_ratio
            
            # Cool down temperature
            temp *= cooling_rate
            
            if temp < min_temp:
                break
                
        return current_points
    
    # Advanced initialization using known good configurations
    def advanced_initialization():
        # Try different starting configurations
        configs = []
        
        # Configuration 1: Regular grid with small perturbations
        points1 = []
        for i in range(4):
            for j in range(4):
                x = i * 0.25 + random.uniform(-0.015, 0.015)
                y = j * 0.25 + random.uniform(-0.015, 0.015)
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                points1.append([x, y])
        configs.append(np.array(points1))
        
        # Configuration 2: Hexagonal-like arrangement
        points2 = []
        # Place points in a pattern that's more evenly distributed
        for i in range(4):
            for j in range(4):
                # Offset every other row
                offset = 0.125 if i % 2 == 1 else 0
                x = i * 0.25 + offset + random.uniform(-0.01, 0.01)
                y = j * 0.25 + random.uniform(-0.01, 0.01)
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                points2.append([x, y])
        configs.append(np.array(points2))
        
        # Configuration 3: Spiral-like arrangement
        points3 = []
        angles = np.linspace(0, 2*np.pi, 16)
        radii = np.linspace(0.1, 0.45, 16)
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            x = 0.5 + radius * np.cos(angle) + random.uniform(-0.01, 0.01)
            y = 0.5 + radius * np.sin(angle) + random.uniform(-0.01, 0.01)
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            points3.append([x, y])
        configs.append(np.array(points3))
        
        # Find the best among these initializations
        best_config = configs[0]
        best_ratio = calculate_ratio(configs[0])
        
        for config in configs[1:]:
            ratio = calculate_ratio(config)
            if ratio > best_ratio:
                best_ratio = ratio
                best_config = config
                
        return best_config
    
    # Try multiple strategies and pick the best
    best_points = None
    best_ratio = 0
    
    # Strategy 1: Advanced initialization + local optimization
    points1 = advanced_initialization()
    points1_optimized = local_optimization(points1)
    ratio1 = calculate_ratio(points1_optimized)
    
    if ratio1 > best_ratio:
        best_ratio = ratio1
        best_points = points1_optimized
    
    # Strategy 2: Improved Simulated Annealing
    points2 = improved_simulated_annealing()
    ratio2 = calculate_ratio(points2)
    
    if ratio2 > best_ratio:
        best_ratio = ratio2
        best_points = points2
    
    # Strategy 3: Multiple local optimizations from different starting points
    for _ in range(8):  # More attempts
        points3 = advanced_initialization()
        points3_optimized = local_optimization(points3)
        ratio3 = calculate_ratio(points3_optimized)
        
        if ratio3 > best_ratio:
            best_ratio = ratio3
            best_points = points3_optimized
    
    # Strategy 4: Hybrid approach - SA followed by local optimization
    if best_points is not None:
        # Try to improve further with local optimization
        points4 = improved_simulated_annealing()
        points4_optimized = local_optimization(points4)
        ratio4 = calculate_ratio(points4_optimized)
        
        if ratio4 > best_ratio:
            best_ratio = ratio4
            best_points = points4_optimized
    
    # Final refinement with global search
    if best_points is not None:
        # Run one more round of local optimization with the best found so far
        final_points = local_optimization(best_points)
        final_ratio = calculate_ratio(final_points)
        
        if final_ratio > best_ratio:
            return final_points
        else:
            return best_points
    else:
        # Fallback to the best initialization
        return advanced_initialization()


# EVOLVE-BLOCK-END
