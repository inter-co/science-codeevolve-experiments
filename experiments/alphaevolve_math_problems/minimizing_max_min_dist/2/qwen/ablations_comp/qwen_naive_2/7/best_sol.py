# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
from typing import Tuple
from numba import jit
import math
from scipy.optimize import differential_evolution
import warnings

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
    Uses a targeted optimization approach with focus on mathematical principles.

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
        """Objective function to maximize min/max ratio (return negative for minimization)"""
        distances = compute_distances(points)
        
        if len(distances) == 0 or len(distances) < 2:
            return np.inf  # Return large value for invalid cases
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 1e-12:
            return np.inf
            
        # We want to maximize min/max ratio, so we minimize its negative
        return -min_dist / max_dist
    
    def generate_initial_configurations():
        """Generate high-quality initial configurations based on known good patterns"""
        configs = []
        
        # Configuration 1: Square grid with slight perturbations
        grid_points = []
        for i in range(4):
            for j in range(4):
                x = 0.1 + 0.8 * i / 3 + np.random.normal(0, 0.01)
                y = 0.1 + 0.8 * j / 3 + np.random.normal(0, 0.01)
                grid_points.append([x, y])
        configs.append(np.clip(np.array(grid_points), 0, 1))
        
        # Configuration 2: Circle-based arrangement with radial variation
        circle_points = []
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        # Use a more structured radial pattern
        radii = 0.3 + 0.3 * np.sin(2 * angles) * 0.5  # Radial variation to avoid uniformity
        for i, (angle, r) in enumerate(zip(angles, radii)):
            x = 0.5 + r * np.cos(angle) * 0.4
            y = 0.5 + r * np.sin(angle) * 0.4
            circle_points.append([x, y])
        configs.append(np.clip(np.array(circle_points), 0, 1))
        
        # Configuration 3: Hexagonal-like arrangement (more uniform than grid)
        hex_points = []
        for i in range(4):
            for j in range(4):
                if len(hex_points) < 16:
                    # Create hexagonal pattern with offset rows
                    x = 0.1 + 0.8 * j / 3 + (i % 2) * 0.12
                    y = 0.1 + 0.8 * i / 3
                    # Add small random perturbation to avoid perfect symmetry
                    x += np.random.normal(0, 0.015)
                    y += np.random.normal(0, 0.015)
                    hex_points.append([x, y])
        configs.append(np.clip(np.array(hex_points), 0, 1))
        
        # Configuration 4: Golden ratio-based arrangement for better spread
        golden_points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(16):
            angle = 2 * np.pi * i / 16
            # Use logarithmic spiral for even distribution
            radius = 0.4 * np.exp(i * 0.1) / np.exp(15 * 0.1)  # Scale appropriately
            x = 0.5 + radius * np.cos(angle) * 0.4
            y = 0.5 + radius * np.sin(angle) * 0.4
            golden_points.append([x, y])
        configs.append(np.clip(np.array(golden_points), 0, 1))
        
        # Configuration 5: Concentric rings with good angular distribution
        ring_points = []
        # Distribute points in two rings
        for ring_idx, radius in enumerate([0.3, 0.6]):
            n_points_in_ring = 8 if ring_idx == 0 else 8
            angles = np.linspace(0, 2*np.pi, n_points_in_ring, endpoint=False)
            for i, angle in enumerate(angles):
                x = 0.5 + radius * np.cos(angle) * 0.4
                y = 0.5 + radius * np.sin(angle) * 0.4
                ring_points.append([x, y])
                if len(ring_points) >= 16:
                    break
            if len(ring_points) >= 16:
                break
        # Fill remaining points with random distribution
        while len(ring_points) < 16:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            ring_points.append([x, y])
        configs.append(np.clip(np.array(ring_points), 0, 1))
        
        return configs
    
    def fast_simulated_annealing(initial_points, max_iter=25000, temp_start=1.0, cooling_rate=0.9995):
        """Fast simulated annealing optimized for this specific problem"""
        current_points = initial_points.copy()
        best_points = current_points.copy()
        
        current_ratio = -objective_function(current_points)  # Convert back to maximization
        best_ratio = current_ratio
        
        temp = temp_start
        
        for iteration in range(max_iter):
            # Generate neighbor solution by perturbing one point
            neighbor_points = current_points.copy()
            point_idx = np.random.randint(0, n)
            
            # Adaptive perturbation size that decreases over time
            adaptive_pert = 0.01 * (1.0 - iteration / max_iter) + 0.002
            neighbor_points[point_idx, 0] += np.random.normal(0, adaptive_pert)
            neighbor_points[point_idx, 1] += np.random.normal(0, adaptive_pert)
            
            # Keep within bounds
            neighbor_points[point_idx, 0] = np.clip(neighbor_points[point_idx, 0], 0, 1)
            neighbor_points[point_idx, 1] = np.clip(neighbor_points[point_idx, 1], 0, 1)
            
            # Calculate new ratio
            new_ratio = -objective_function(neighbor_points)  # Convert back to maximization
            
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
            if iteration % 3000 == 0 and iteration > 0:
                if np.random.random() < 0.2:  # Restart with 20% probability
                    restart_configs = generate_initial_configurations()
                    restart_config = restart_configs[np.random.randint(len(restart_configs))]
                    current_points = restart_config.copy()
                    current_ratio = -objective_function(current_points)
        
        return best_points, best_ratio
    
    def direct_optimization_approach():
        """Direct optimization approach with multiple restarts"""
        best_points = None
        best_ratio = -np.inf
        
        # Generate diverse initial configurations
        initial_configs = generate_initial_configurations()
        
        # Try multiple starting points with fast simulated annealing
        for i, config in enumerate(initial_configs):
            try:
                # Run fast simulated annealing from this initial configuration
                points, ratio = fast_simulated_annealing(config, max_iter=20000)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
                    
                # Also try with different parameters for variety
                points2, ratio2 = fast_simulated_annealing(config, max_iter=25000, temp_start=0.5)
                if ratio2 > best_ratio:
                    best_ratio = ratio2
                    best_points = points2.copy()
                    
            except Exception as e:
                continue
        
        # If no improvement found, return the best initial configuration
        if best_points is None:
            best_points = initial_configs[0]
            best_ratio = -objective_function(best_points)
        
        return best_points, best_ratio
    
    # Execute optimization with more efficient approach
    final_points, ratio = direct_optimization_approach()
    
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
