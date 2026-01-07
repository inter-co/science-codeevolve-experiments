# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math
import time

# Global constants
N_CIRCLES = 32
MAX_TIME = 55.0  # Leave some buffer for cleanup

def initialize_circles_hexagonal() -> np.ndarray:
    """Initialize circles using a hexagonal packing pattern as starting point"""
    circles = np.zeros((N_CIRCLES, 3))
    
    # Try to arrange in a roughly hexagonal pattern
    rows = 6
    cols = 6
    
    # Adjust grid size to fit 32 circles
    if rows * cols < N_CIRCLES:
        rows = 7
        cols = 5
    
    # Create initial positions in a grid pattern
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= N_CIRCLES:
                break
            # Offset every other row for hexagonal packing
            x_offset = 0.5 if i % 2 == 1 else 0.0
            x = (j + x_offset) / cols
            y = i / rows
            
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            positions.append([x, y])
    
    # Fill remaining circles with random positions
    while len(positions) < N_CIRCLES:
        positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
    
    # Initialize with small radii
    for i in range(N_CIRCLES):
        circles[i] = [positions[i][0], positions[i][1], 0.02]
    
    return circles

def get_circle_radius(circle, circles, tree=None):
    """Calculate maximum possible radius for a circle at given position without overlap"""
    x, y = circle[0], circle[1]
    
    # Check boundary constraints
    max_radius = min(x, y, 1-x, 1-y)
    
    # Check overlap constraints with existing circles
    if tree is not None:
        # Find nearby circles using spatial indexing
        nearby = tree.query_ball_point([x, y], 2*max_radius)
        for idx in nearby:
            if idx >= len(circles):
                continue
            cx, cy, cr = circles[idx]
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < cr:
                # This shouldn't happen since we're checking against existing circles
                continue
            # Calculate maximum radius to avoid overlapping this circle
            new_radius = distance - cr
            max_radius = min(max_radius, new_radius)
    else:
        # Brute force check against all circles
        for i in range(len(circles)):
            cx, cy, cr = circles[i]
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            if distance > 0:  # Avoid self-intersection
                new_radius = distance - cr
                max_radius = min(max_radius, new_radius)
    
    return max_radius

def compute_total_radius(circles):
    """Compute total sum of radii"""
    return sum(circles[:, 2])

def validate_circles(circles):
    """Validate that all circles are within bounds and non-overlapping"""
    # Check bounds
    for i in range(len(circles)):
        x, y, r = circles[i]
        if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
            return False
    
    # Check overlaps using spatial indexing for efficiency
    try:
        tree = cKDTree(circles[:, :2])
        pairs = tree.query_pairs(0.0001)  # Very small threshold to catch overlaps
        for i, j in pairs:
            if i >= len(circles) or j >= len(circles):
                continue
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False
    except:
        # Fallback to brute force if spatial indexing fails
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return False
    
    return True

def objective_function(circles_flat):
    """Objective function to maximize (negative because scipy minimizes)"""
    # Reshape flat array back to circles
    circles = circles_flat.reshape(-1, 3)
    
    # Compute negative sum of radii (since we want to maximize)
    return -compute_total_radius(circles)

def constraint_bounds(circles_flat):
    """Constraint function ensuring circles stay within bounds"""
    circles = circles_flat.reshape(-1, 3)
    # Return positive values for valid constraints (violated constraints are negative)
    violations = []
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Radius constraint
        violations.append(r)
        # Boundary constraints
        violations.append(x - r)  # x >= r
        violations.append(y - r)  # y >= r
        violations.append(1 - x - r)  # 1 - x >= r
        violations.append(1 - y - r)  # 1 - y >= r
    return np.array(violations)

def constraint_overlap(circles_flat):
    """Constraint function ensuring no overlaps"""
    circles = circles_flat.reshape(-1, 3)
    violations = []
    
    # Check all pairs of circles for overlap
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # Violation is negative when overlap occurs
            violations.append(distance - (r1 + r2))
    
    return np.array(violations)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Initialize with hexagonal pattern
    circles = initialize_circles_hexagonal()
    
    # Set up optimization problem
    # Flatten circles array for optimization
    circles_flat = circles.flatten()
    
    # Initial optimization using scipy's minimize
    try:
        # First phase: optimize just positions with fixed radii
        def optimize_positions():
            # Start with current circles
            current_circles = circles.copy()
            
            # Use simple iterative approach with local optimization
            for iteration in range(100):
                if time.time() - start_time > MAX_TIME:
                    break
                    
                # For each circle, find optimal position with current radii
                for i in range(N_CIRCLES):
                    if time.time() - start_time > MAX_TIME:
                        break
                        
                    # Create a temporary copy
                    temp_circles = current_circles.copy()
                    
                    # Get current position and radius
                    x, y, r = temp_circles[i]
                    
                    # Define optimization bounds
                    bounds = [(r, 1-r), (r, 1-r)]
                    
                    # Simple gradient descent step
                    # We'll do a simple grid search around current position
                    best_x, best_y = x, y
                    best_radius = r
                    best_sum = 0
                    
                    # Grid search around current position
                    steps = 20
                    for dx in range(-steps, steps+1):
                        for dy in range(-steps, steps+1):
                            if time.time() - start_time > MAX_TIME:
                                break
                                
                            test_x = max(r, min(1-r, x + dx * 0.01))
                            test_y = max(r, min(1-r, y + dy * 0.01))
                            
                            # Test if this position works
                            temp_circles[i] = [test_x, test_y, r]
                            
                            # Check if valid
                            valid = True
                            for j in range(N_CIRCLES):
                                if i != j:
                                    x1, y1, r1 = temp_circles[j]
                                    x2, y2, r2 = temp_circles[i]
                                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                                    if dist < r1 + r2:
                                        valid = False
                                        break
                            
                            if valid:
                                # Calculate what radius we could have at this position
                                max_r = get_circle_radius([test_x, test_y], temp_circles)
                                if max_r > best_radius:
                                    best_radius = max_r
                                    best_x, best_y = test_x, test_y
                            
                            if time.time() - start_time > MAX_TIME:
                                break
                        if time.time() - start_time > MAX_TIME:
                            break
                    
                    # Update circle with best found position/radius
                    current_circles[i] = [best_x, best_y, best_radius]
                
                # Rebuild spatial index for better performance
                if iteration % 10 == 0:
                    try:
                        tree = cKDTree(current_circles[:, :2])
                    except:
                        pass
                
                # Early stopping if improvement is minimal
                if iteration > 10:
                    old_sum = compute_total_radius(current_circles)
                    # Run a few more iterations to see if we can improve
                    if iteration > 20:
                        # Check if we've plateaued
                        pass
            
            return current_circles
        
        # Run optimization
        optimized_circles = optimize_positions()
        
        # Final refinement using a more systematic approach
        final_circles = optimized_circles.copy()
        
        # Try to improve by adjusting radii
        for _ in range(50):
            if time.time() - start_time > MAX_TIME:
                break
                
            improved = False
            # Try to increase radii while maintaining constraints
            for i in range(N_CIRCLES):
                if time.time() - start_time > MAX_TIME:
                    break
                    
                x, y, r = final_circles[i]
                
                # Calculate maximum possible radius at this location
                max_r = get_circle_radius([x, y], final_circles)
                
                # Increase radius if beneficial and feasible
                if max_r > r * 1.01:  # Allow 1% improvement threshold
                    # Check if increasing radius helps without violating constraints
                    new_r = min(max_r, r * 1.1)  # Increase by up to 10%
                    test_circles = final_circles.copy()
                    test_circles[i, 2] = new_r
                    
                    # Validate new configuration
                    valid = True
                    for j in range(N_CIRCLES):
                        if i != j:
                            x1, y1, r1 = test_circles[j]
                            x2, y2, r2 = test_circles[i]
                            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                            if dist < r1 + r2:
                                valid = False
                                break
                    
                    if valid:
                        final_circles[i, 2] = new_r
                        improved = True
            
            if not improved:
                break
        
        # Final validation
        if validate_circles(final_circles):
            return final_circles
        else:
            # If validation failed, return the best valid configuration found
            return optimized_circles
            
    except Exception as e:
        # If optimization fails, return initial configuration
        return circles


# EVOLVE-BLOCK-END
