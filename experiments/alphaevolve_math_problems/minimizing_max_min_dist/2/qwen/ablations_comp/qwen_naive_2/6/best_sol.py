# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
from typing import Tuple
from numba import jit
import math

@jit(nopython=True)
def compute_distances_fast(points):
    """Fast computation of pairwise distances using Numba"""
    n = points.shape[0]
    distances = np.zeros(n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            distances[idx] = np.sqrt(dx * dx + dy * dy)
            idx += 1
    return distances

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a global optimization approach with simulated annealing and geometric insights.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    
    def compute_distances(points):
        """Compute all pairwise distances efficiently"""
        if len(points) < 2:
            return np.array([])
        return compute_distances_fast(points)
    
    def objective_function(points):
        """Objective function to maximize min/max ratio"""
        distances = compute_distances(points)
        
        if len(distances) == 0 or len(distances) < 2:
            return -np.inf
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 1e-12:
            return -np.inf
            
        # We want to maximize min/max ratio
        return min_dist / max_dist
    
    def generate_initial_configurations():
        """Generate multiple diverse initial configurations"""
        configs = []
        
        # Configuration 1: Regular grid with perturbations
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = 0.1 + 0.8 * i / 3 + np.random.normal(0, 0.02)
                y = 0.1 + 0.8 * j / 3 + np.random.normal(0, 0.02)
                grid_points.append([x, y])
        configs.append(np.clip(np.array(grid_points), 0, 1))
        
        # Configuration 2: Circle-based arrangement with random perturbations
        circle_points = []
        angles = np.linspace(0, 2*np.pi, 16)
        for i, angle in enumerate(angles):
            r = 0.4 + 0.3 * np.random.random()  # Random radius
            x = 0.5 + r * np.cos(angle) * 0.4
            y = 0.5 + r * np.sin(angle) * 0.4
            circle_points.append([x, y])
        configs.append(np.clip(np.array(circle_points), 0, 1))
        
        # Configuration 3: Spiral arrangement
        spiral_points = []
        for i in range(16):
            angle = 2 * np.pi * i / 16
            radius = 0.4 * (i / 16)
            x = 0.5 + radius * np.cos(angle) * 0.8
            y = 0.5 + radius * np.sin(angle) * 0.8
            spiral_points.append([x, y])
        configs.append(np.clip(np.array(spiral_points), 0, 1))
        
        # Configuration 4: Hexagonal packing approximation
        hex_points = []
        rows = 4
        cols = 4
        for i in range(rows):
            for j in range(cols):
                if len(hex_points) < 16:
                    x = 0.1 + 0.8 * j / 3 + (i % 2) * 0.2
                    y = 0.1 + 0.8 * i / 3
                    # Add small random perturbation
                    x += np.random.normal(0, 0.015)
                    y += np.random.normal(0, 0.015)
                    hex_points.append([x, y])
        configs.append(np.clip(np.array(hex_points), 0, 1))
        
        return configs
    
    def simulated_annealing(initial_points, max_iter=50000, temp_start=1.0, cooling_rate=0.9995):
        """Simulated annealing optimization for better global exploration"""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        
        current_ratio = objective_function(current_points)
        best_ratio = current_ratio
        
        temp = temp_start
        
        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing one point
            neighbor_points = current_points.copy()
            point_idx = np.random.randint(0, n)
            
            # Perturb the selected point
            neighbor_points[point_idx, 0] += np.random.normal(0, 0.005)
            neighbor_points[point_idx, 1] += np.random.normal(0, 0.005)
            
            # Keep within bounds
            neighbor_points[point_idx, 0] = np.clip(neighbor_points[point_idx, 0], 0, 1)
            neighbor_points[point_idx, 1] = np.clip(neighbor_points[point_idx, 1], 0, 1)
            
            # Calculate new ratio
            new_ratio = objective_function(neighbor_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or np.random.random() < np.exp((new_ratio - current_ratio) / temp):
                current_points = neighbor_points
                current_ratio = new_ratio
                
                # Update best solution
                if new_ratio > best_ratio:
                    best_points = neighbor_points.copy()
                    best_ratio = new_ratio
            
            # Cool down temperature
            temp *= cooling_rate
            
            # Occasionally restart with a better configuration
            if iteration % 1000 == 0 and iteration > 0:
                if np.random.random() < 0.1:  # 10% chance to restart
                    restart_configs = generate_initial_configurations()
                    restart_config = restart_configs[np.random.randint(len(restart_configs))]
                    current_points = restart_config.copy()
                    current_ratio = objective_function(current_points)
        
        return best_points, best_ratio
    
    def optimize_with_global_search() -> Tuple[np.ndarray, float]:
        """Use global optimization with multiple restarts"""
        # Generate several diverse initial configurations
        initial_configs = generate_initial_configurations()
        
        best_points = None
        best_ratio = -np.inf
        
        # Try multiple starting points with simulated annealing
        for i, config in enumerate(initial_configs):
            try:
                # Run simulated annealing from this initial configuration
                points, ratio = simulated_annealing(config, max_iter=20000)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
                    
                # Also try with different parameters for variety
                points2, ratio2 = simulated_annealing(config, max_iter=30000, temp_start=0.5)
                if ratio2 > best_ratio:
                    best_ratio = ratio2
                    best_points = points2.copy()
                    
            except Exception as e:
                continue
        
        # If no improvement found, return the best initial configuration
        if best_points is None:
            best_points = initial_configs[0]
            best_ratio = objective_function(best_points)
        
        return best_points, best_ratio
    
    # Execute global optimization
    final_points, ratio = optimize_with_global_search()
    
    # Final verification and cleanup
    if final_points is not None:
        final_points = np.clip(final_points, 0, 1)
        # Recalculate ratio to ensure correctness
        distances = compute_distances(final_points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 1e-12:
                ratio = min_dist / max_dist
    
    return final_points


# EVOLVE-BLOCK-END
