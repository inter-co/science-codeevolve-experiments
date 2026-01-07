# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining systematic initialization, scipy optimization, and local search.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Improved initial placement using hexagonal grid for better coverage
    def create_hexagonal_initial_placement(n_circles):
        circles = []
        
        # Create hexagonal grid pattern
        rows = 6
        cols = 6
        if rows * cols < n_circles:
            rows = 7
            cols = 5
            
        # Hexagonal packing parameters
        spacing_x = 0.15
        spacing_y = 0.15 * np.sqrt(3)/2
        
        # Generate points in hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n_circles:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, 0.05])  # Initial radius
        
        # Fill remaining positions with random placements near edges
        while len(circles) < n_circles:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        return np.array(circles)
    
    # Initialize with improved placement
    circles = create_hexagonal_initial_placement(n)
    
    # Compute initial radii based on available space
    def compute_initial_radii(circles):
        # For each circle, compute maximum possible radius
        for i in range(len(circles)):
            x, y = circles[i, 0], circles[i, 1]
            # Boundary constraints
            boundary_radius = min(x, 1-x, y, 1-y)
            
            # Find minimum distance to other circles
            min_distance = float('inf')
            for j in range(len(circles)):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    min_distance = min(min_distance, distance)
            
            # Set radius based on boundary and overlap constraints
            if min_distance < float('inf') and min_distance > 0:
                # Allow up to 1/2 of min distance to neighbors, but capped by boundary
                max_radius = min(boundary_radius, min_distance / 2.0)
            else:
                max_radius = boundary_radius
                
            # Ensure reasonable radius
            circles[i, 2] = max(0.001, min(0.1, max_radius))
    
    compute_initial_radii(circles)
    
    # Constraint functions for scipy optimization
    def get_constraints():
        cons = []
        
        # Boundary constraints: radius <= x <= 1-radius and radius <= y <= 1-radius
        for i in range(n):
            # x >= r
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: x[3*i] - x[3*i+2] - 1e-6
            })
            # y >= r  
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: x[3*i+1] - x[3*i+2] - 1e-6
            })
            # 1-x >= r
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2] - 1e-6
            })
            # 1-y >= r
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2] - 1e-6
            })
        
        # Non-overlap constraints: sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (
                        np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) 
                        - x[3*i+2] - x[3*j+2] - 1e-6
                    )
                })
        
        return cons
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(x):
        return -np.sum(x[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Flatten circles array for optimization
    x0 = circles.flatten()
    
    # Apply optimization with multiple attempts
    best_result = None
    best_sum = -float('inf')
    
    # Try multiple optimization runs with different methods
    methods = ['SLSQP', 'trust-constr']
    for method in methods:
        try:
            result = minimize(
                objective, 
                x0,
                method=method,
                constraints=get_constraints(),
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                current_sum = -result.fun  # Convert back to sum of radii
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception as e:
            continue
    
    # If optimization succeeded, return optimized result
    if best_result is not None and best_result.success:
        optimized_circles = best_result.x.reshape(-1, 3)
        
        # Validate constraints
        if check_constraints(optimized_circles):
            return optimized_circles
    
    # If optimization failed or didn't produce valid result, do local search refinement
    return local_search_refinement(circles)

def check_constraints(circles: np.ndarray) -> bool:
    """Check if all circles satisfy containment and non-overlap constraints."""
    n = len(circles)
    
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False
    return True

def local_search_refinement(initial_circles: np.ndarray) -> np.ndarray:
    """Refine configuration using local search approach."""
    circles = initial_circles.copy()
    best_sum = np.sum(circles[:, 2])
    max_iter = 2000
    
    # Simple gradient descent approach
    for iteration in range(max_iter):
        improved = False
        # Try improving each circle individually
        for i in range(len(circles)):
            original_circle = circles[i].copy()
            
            # Try to increase radius first
            x, y, r = original_circle
            # Calculate maximum possible radius at this position
            boundary_radius = min(x, 1-x, y, 1-y)
            
            # Find minimum distance to other circles
            min_distance = float('inf')
            for j in range(len(circles)):
                if i != j:
                    dx = x - circles[j, 0]
                    dy = y - circles[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    min_distance = min(min_distance, distance)
            
            # Maximum radius considering both boundary and overlap constraints
            max_radius = boundary_radius
            if min_distance < float('inf') and min_distance > 0:
                max_radius = min(max_radius, min_distance - 0.001)  # Small buffer
            
            if max_radius > r + 1e-6:
                circles[i] = [x, y, max_radius]
                improved = True
                continue
            
            # Try small position adjustments to increase radius
            for _ in range(10):
                dx = np.random.uniform(-0.01, 0.01)
                dy = np.random.uniform(-0.01, 0.01)
                test_x = max(0.01, min(0.99, x + dx))
                test_y = max(0.01, min(0.99, y + dy))
                
                # Calculate maximum radius for this new position
                test_boundary_radius = min(test_x, 1-test_x, test_y, 1-test_y)
                
                test_min_distance = float('inf')
                for j in range(len(circles)):
                    if i != j:
                        dx = test_x - circles[j, 0]
                        dy = test_y - circles[j, 1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        test_min_distance = min(test_min_distance, distance)
                
                test_max_radius = test_boundary_radius
                if test_min_distance < float('inf') and test_min_distance > 0:
                    test_max_radius = min(test_max_radius, test_min_distance - 0.001)
                
                if test_max_radius > r + 1e-6:
                    circles[i] = [test_x, test_y, test_max_radius]
                    improved = True
                    break
        
        # Check if we made progress
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum + 1e-8:
            best_sum = current_sum
        elif not improved:
            # No improvements, stop early
            break
    
    return circles


# EVOLVE-BLOCK-END
