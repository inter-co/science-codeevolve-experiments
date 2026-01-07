# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a robust multi-start optimization approach with careful constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Multi-start approach: try several different initializations
    best_circles = None
    best_sum = -np.inf
    
    # Try multiple random initializations
    for attempt in range(5):
        # Phase 1: Initialize with a better geometric pattern
        circles = initialize_better_packing(n)
        
        # Phase 2: Direct optimization with scipy
        circles = optimize_directly(circles)
        
        # Phase 3: Fine-grained local optimization
        circles = fine_tune_solution(circles)
        
        # Evaluate solution
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
    
    return best_circles if best_circles is not None else initialize_better_packing(n)

def initialize_better_packing(n):
    """Initialize circle positions using a refined geometric approach."""
    # Use a grid-based approach with better spacing
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    circles = np.zeros((n, 3))
    idx = 0
    
    # Create a grid pattern with slight randomness for better distribution
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            
            # Add randomness to avoid regular patterns
            x += np.random.uniform(-spacing_x * 0.1, spacing_x * 0.1)
            y += np.random.uniform(-spacing_y * 0.1, spacing_y * 0.1)
            
            # Ensure we're within bounds
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            
            # Initial radius - larger to give more room for optimization
            radius = min(spacing_x, spacing_y) * 0.35
            circles[idx] = [x, y, radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining positions with random valid placements
    for i in range(idx, n):
        attempts = 0
        while attempts < 100:  # Prevent infinite loop
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            radius = np.random.uniform(0.02, 0.15)
            
            # Check if this placement is valid (not too close to existing circles)
            valid = True
            for j in range(i):
                dx = x - circles[j, 0]
                dy = y - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < (radius + circles[j, 2]):
                    valid = False
                    break
            
            if valid:
                circles[i] = [x, y, radius]
                break
            attempts += 1
    
    return circles

def optimize_directly(circles):
    """Direct optimization using scipy with efficient constraint handling."""
    n = len(circles)
    
    # Define objective function: minimize negative sum of radii
    def objective(x_flat):
        circles_temp = x_flat.reshape(-1, 3)
        return -np.sum(circles_temp[:, 2])
    
    # Define bounds for variables
    bounds = []
    for i in range(n):
        # x, y positions: [0.01, 0.99]
        # r radius: [0.01, 0.5]
        bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.01, 0.5)])
    
    # Flatten initial circles
    x0 = circles.flatten()
    
    # Define constraints more efficiently
    def boundary_constraint(x_flat):
        circles_temp = x_flat.reshape(-1, 3)
        result = []
        for i in range(len(circles_temp)):
            x, y, r = circles_temp[i]
            # x >= r, y >= r, 1-x >= r, 1-y >= r
            result.extend([x - r, y - r, 1 - x - r, 1 - y - r])
        return np.array(result)
    
    def overlap_constraint(x_flat):
        circles_temp = x_flat.reshape(-1, 3)
        result = []
        
        # Use spatial indexing for efficiency
        positions = circles_temp[:, :2]
        tree = cKDTree(positions)
        
        # Query pairs within reasonable distance
        pairs = tree.query_pairs(0.5, output_type='ndarray')
        for i, j in pairs:
            if i < j:  # Only process each pair once
                x1, y1, r1 = circles_temp[i]
                x2, y2, r2 = circles_temp[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                result.append(dist - (r1 + r2))
        return np.array(result)
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': boundary_constraint},
        {'type': 'ineq', 'fun': overlap_constraint}
    ]
    
    # Perform optimization with more iterations and better settings
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 800, 'ftol': 1e-7, 'disp': False}
        )
        
        if res.success:
            circles = res.x.reshape(-1, 3)
    except Exception:
        # If optimization fails, continue with current configuration
        pass
    
    return circles

def fine_tune_solution(circles):
    """Apply fine-tuning to improve the solution quality."""
    n = len(circles)
    
    # Strategy 1: Try to increase radii globally with careful validation
    for iter_num in range(200):  # More iterations
        improved = False
        for i in range(n):
            # Try to increase radius of circle i
            old_radius = circles[i, 2]
            # Increase radius by slightly larger amount
            new_radius = min(old_radius * 1.02, 0.45)  # Larger increase
            
            # Check if new radius is valid with current configuration
            if is_valid_placement(circles, i, new_radius):
                circles[i, 2] = new_radius
                improved = True
        
        # Stop if no improvement
        if not improved:
            break
    
    # Strategy 2: Local position adjustments with better bounds checking
    for iter_num in range(150):  # More iterations
        improved = False
        for i in range(n):
            # Try small position adjustments
            dx = np.random.uniform(-0.0015, 0.0015)
            dy = np.random.uniform(-0.0015, 0.0015)
            
            old_x, old_y, old_r = circles[i]
            new_x = old_x + dx
            new_y = old_y + dy
            
            # Clamp to valid range ensuring radius constraint
            new_x = np.clip(new_x, old_r, 1 - old_r)
            new_y = np.clip(new_y, old_r, 1 - old_r)
            
            # Check if this adjustment improves the configuration
            temp_circles = circles.copy()
            temp_circles[i] = [new_x, new_y, old_r]
            
            # Check if the change maintains validity
            if is_valid_configuration(temp_circles):
                circles[i] = [new_x, new_y, old_r]
                improved = True
        
        if not improved:
            break
    
    return circles

def is_valid_placement(circles, target_idx, new_radius):
    """Check if placing a circle with new_radius at target_idx is valid."""
    x, y, _ = circles[target_idx]
    
    # Check boundary constraints
    if x - new_radius < 0 or x + new_radius > 1 or y - new_radius < 0 or y + new_radius > 1:
        return False
    
    # Check overlap with other circles
    for i in range(len(circles)):
        if i != target_idx:
            x2, y2, r2 = circles[i]
            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
            if distance < new_radius + r2:
                return False
    
    return True

def is_valid_configuration(circles):
    """Check if entire configuration is valid."""
    n = len(circles)
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False
    
    return True


# EVOLVE-BLOCK-END
