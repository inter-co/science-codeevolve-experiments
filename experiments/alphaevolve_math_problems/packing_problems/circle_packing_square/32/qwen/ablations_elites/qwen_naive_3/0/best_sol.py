# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization,
    enhanced with iterative refinement inspired by physics-based methods.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a refined grid pattern inspired by INSPIRATION 1
    def initialize_refined_grid():
        circles = np.zeros((n, 3))
        
        # Create a more refined grid pattern with better coverage
        grid_size = 6
        
        # Generate points in a grid pattern with slight randomness
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= n:
                    break
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                # Add small random jitter for better distribution
                x += random.uniform(-0.03, 0.03)
                y += random.uniform(-0.03, 0.03)
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                positions.append([x, y])
        
        # Fill remaining spots with random placements near boundaries
        while len(positions) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            positions.append([x, y])
        
        # Set initial positions and small initial radii
        for i in range(n):
            x, y = positions[i]
            circles[i] = [x, y, 0.02]  # Small initial radius
        
        return circles
    
    # Create initial configuration
    circles = initialize_refined_grid()
    
    # Helper functions for constraint handling
    def enforce_boundaries(circles_array):
        """Ensure all circles respect boundary constraints"""
        circles_array = circles_array.copy()
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            # Enforce boundary constraints
            r = min(r, x, y, 1-x, 1-y)
            # Clip to valid range
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            circles_array[i] = [x, y, r]
        return circles_array

    def check_overlap(circles_array):
        """Check if any circles overlap"""
        n = len(circles_array)
        for i in range(n):
            for j in range(i+1, n):
                dx = circles_array[i, 0] - circles_array[j, 0]
                dy = circles_array[i, 1] - circles_array[j, 1]
                dist_sq = dx*dx + dy*dy
                radii_sum = circles_array[i, 2] + circles_array[j, 2]
                if dist_sq < radii_sum * radii_sum:
                    return True
        return False

    def evaluate_fitness(circles_array):
        """Evaluate fitness as sum of radii"""
        return np.sum(circles_array[:, 2])

    # Flatten array for optimization
    def flatten_circles(circles_array):
        flat = []
        for circle in circles_array:
            flat.extend(circle)
        return np.array(flat)
    
    def unflatten_circles(flat_array):
        circles = []
        for i in range(0, len(flat_array), 3):
            circles.append([flat_array[i], flat_array[i+1], flat_array[i+2]])
        return np.array(circles)
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(flat_circles):
        total_radius = 0
        for i in range(0, len(flat_circles), 3):
            total_radius += flat_circles[i+2]
        return -total_radius
    
    # Constraints
    def boundary_constraint(flat_circles):
        # Each circle's radius must be such that it fits within the square
        constraints = []
        for i in range(0, len(flat_circles), 3):
            x, y, r = flat_circles[i], flat_circles[i+1], flat_circles[i+2]
            # r <= x, r <= 1-x, r <= y, r <= 1-y
            constraints.append(x - r)      # x >= r
            constraints.append(1 - x - r)  # x <= 1-r
            constraints.append(y - r)      # y >= r
            constraints.append(1 - y - r)  # y <= 1-r
        return np.array(constraints)
    
    def overlap_constraint(flat_circles):
        # Non-overlapping constraints
        constraints = []
        n_circles = len(flat_circles) // 3
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                x1, y1, r1 = flat_circles[3*i], flat_circles[3*i+1], flat_circles[3*i+2]
                x2, y2, r2 = flat_circles[3*j], flat_circles[3*j+1], flat_circles[3*j+2]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # Distance >= r1 + r2 (non-overlapping)
                constraints.append(distance - (r1 + r2))
        return np.array(constraints)
    
    # Initial guess
    initial_flat = flatten_circles(circles)
    
    # Set up bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: boundary_constraint(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
    ]
    
    # Run primary optimization
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-6}
        )
        
        if result.success:
            final_circles = unflatten_circles(result.x)
            final_circles = enforce_boundaries(final_circles)
            
            # Post-processing: iterative refinement if needed
            if check_overlap(final_circles):
                # Apply a simple refinement step
                final_circles = refine_solution(final_circles)
            
            return final_circles
    except Exception as e:
        pass
    
    # Fallback to the refined grid if optimization fails
    circles = enforce_boundaries(circles)
    circles = refine_solution(circles)
    return circles

def refine_solution(circles):
    """Apply iterative refinement to improve solution quality"""
    # Simple iterative improvement approach
    for iteration in range(20):
        improved = False
        
        # Try to increase each radius
        for i in range(len(circles)):
            x, y, old_radius = circles[i]
            
            # Maximum radius constrained by boundaries
            max_radius = min(x, y, 1-x, 1-y)
            
            # Check overlap with all other circles
            for j in range(len(circles)):
                if i != j:
                    dist = np.sqrt(
                        (x - circles[j, 0])**2 + 
                        (y - circles[j, 1])**2
                    )
                    max_radius = min(max_radius, dist - circles[j, 2])
            
            # Increase radius if beneficial and valid
            if max_radius > old_radius + 1e-6:
                circles[i, 2] = max_radius
                improved = True
                
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
