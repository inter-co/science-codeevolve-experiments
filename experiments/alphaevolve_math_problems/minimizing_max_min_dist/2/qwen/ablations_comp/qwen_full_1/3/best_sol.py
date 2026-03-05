# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist
import math
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical construction and stochastic optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(x_flat):
        """Objective function to minimize (negative of the ratio we want to maximize)"""
        points = x_flat.reshape(-1, 2)
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return -np.inf
        return -min_dist / max_dist
    
    # Mathematical construction approach: use vertices of a regular 16-gon inscribed in unit circle
    # This provides a good starting configuration with high symmetry
    def construct_regular_polygon():
        n = 16
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        # Scale to fit in unit square [0,1] x [0,1] 
        # Center at origin, then shift and scale
        points = points * 0.4 + 0.5  # Scale and center
        return points
    
    # Alternative: Use a construction based on the 16-cell (tesseract's dual)
    # Projected to 2D, this gives a symmetric configuration
    def construct_16_cell_projection():
        # Vertices of 16-cell in 4D: all permutations of (±1,0,0,0) with even number of minus signs
        # We'll project to 2D using specific orthogonal projection
        vertices_4d = []
        for a in [1, -1]:
            for b in [1, -1]:
                for c in [1, -1]:
                    for d in [1, -1]:
                        if (a+b+c+d) % 2 == 0:  # Even number of minus signs
                            vertices_4d.append([a, b, c, d])
        
        # Project to 2D using first two coordinates scaled appropriately
        points_2d = np.array(vertices_4d)[:16, :2] * 0.3 + 0.5
        return points_2d
    
    # Stochastic optimization approach using simulated annealing
    def stochastic_optimization():
        # Start with a good mathematical configuration
        initial_points = construct_regular_polygon()
        
        # Simulated Annealing parameters
        max_iter = 5000
        T = 1.0
        T_min = 1e-6
        alpha = 0.99
        best_points = initial_points.copy()
        best_ratio = -objective(initial_points.flatten())
        
        current_points = initial_points.copy()
        current_ratio = best_ratio
        
        # Track history for convergence
        history = []
        
        for i in range(max_iter):
            # Generate neighbor by perturbing one point
            idx = random.randint(0, 15)
            new_points = current_points.copy()
            
            # Small random perturbation
            delta_x = np.random.normal(0, 0.01)
            delta_y = np.random.normal(0, 0.01)
            
            # Ensure we stay within bounds
            new_x = np.clip(new_points[idx, 0] + delta_x, 0, 1)
            new_y = np.clip(new_points[idx, 1] + delta_y, 0, 1)
            
            new_points[idx] = [new_x, new_y]
            
            # Calculate new ratio
            new_ratio = -objective(new_points.flatten())
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio or random.random() < math.exp((new_ratio - current_ratio) / T):
                current_points = new_points
                current_ratio = new_ratio
                
                if current_ratio > best_ratio:
                    best_points = current_points.copy()
                    best_ratio = current_ratio
            
            # Cool down temperature
            T = max(T * alpha, T_min)
            
            # Track progress
            if i % 100 == 0:
                history.append(best_ratio)
        
        return best_points
    
    # Try multiple approaches and return the best result
    best_points = None
    best_ratio = -np.inf
    
    # Approach 1: Mathematical construction with local refinement
    try:
        # Start with regular polygon
        math_points = construct_regular_polygon()
        math_ratio = -objective(math_points.flatten())
        
        if math_ratio > best_ratio:
            best_ratio = math_ratio
            best_points = math_points
        
        # Refine with stochastic optimization
        refined_points = stochastic_optimization()
        refined_ratio = -objective(refined_points.flatten())
        
        if refined_ratio > best_ratio:
            best_ratio = refined_ratio
            best_points = refined_points
            
    except Exception as e:
        pass
    
    # If nothing worked, fall back to simple approach
    if best_points is None:
        # Simple random initialization with bounds checking
        points = np.random.rand(16, 2)
        # Ensure they're within bounds
        points = np.clip(points, 0, 1)
        return points
    
    return best_points


# EVOLVE-BLOCK-END
