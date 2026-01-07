# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
                 the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = 32
    
    # Phase 1: Geometric initialization using hexagonal packing pattern
    # Create a hexagonal grid pattern that fits well in the unit square
    circles = np.zeros((n, 3))
    
    # Determine grid dimensions for roughly hexagonal packing
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    
    # Adjust to fit exactly 32 circles
    while rows * cols < n:
        rows += 1
        cols = int(np.ceil(n / rows))
    
    # Create hexagonal grid with proper spacing
    hex_radius = 0.15  # Initial guess for hexagonal packing
    spacing_x = 2 * hex_radius * 0.9  # Slightly reduced for better packing
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    # Place circles in hexagonal pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = 0.1 + j * spacing_x
            y = 0.1 + i * spacing_y
            
            # Offset every other row for hexagonal packing
            if i % 2 == 1:
                x += spacing_x / 2
            
            # Ensure within bounds
            if x <= 1 - hex_radius and y <= 1 - hex_radius:
                circles[idx] = [x, y, hex_radius]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with random placements ensuring no overlaps
    for i in range(idx, n):
        attempts = 0
        while attempts < 1000:
            x = np.random.uniform(hex_radius, 1 - hex_radius)
            y = np.random.uniform(hex_radius, 1 - hex_radius)
            radius = hex_radius
            
            # Check overlap with existing circles
            overlap = False
            for j in range(i):
                dx = x - circles[j, 0]
                dy = y - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radius + circles[j, 2]:
                    overlap = True
                    break
            
            if not overlap:
                circles[i] = [x, y, radius]
                break
            attempts += 1
    
    # Phase 2: Optimization using constrained nonlinear programming
    # Define optimization variables: [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
    def objective(vars):
        # Return negative sum of radii (since we want to maximize)
        return -np.sum(vars[2::3])  # Every third element starting from index 2
    
    def constraint_func(vars):
        # Check all constraints: non-overlap and containment
        positions = vars.reshape(-1, 3)[:, :2]  # Extract x,y coordinates
        radii = vars[2::3]  # Extract radii
        
        # Containment constraints (each circle must be fully inside unit square)
        containment_constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Circle must be within bounds
            containment_constraints.extend([
                x - r,  # x >= r
                y - r,  # y >= r
                1 - x - r,  # 1 - x >= r
                1 - y - r   # 1 - y >= r
            ])
        
        # Non-overlap constraints
        non_overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Distance between centers >= sum of radii (negative for constraint)
                non_overlap_constraints.append(dist - radii[i] - radii[j])
        
        return np.array(containment_constraints + non_overlap_constraints)
    
    # Initial guess
    initial_vars = circles.flatten()
    
    # Set up bounds for optimization
    bounds = []
    # Bounds for positions: [min_x, max_x, min_y, max_y, min_r, max_r] * 32
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.49)])  # x, y, r
    
    # Add bounds for the constraints
    # We'll use a simplified approach: let's optimize with a more structured method
    
    # Phase 3: Improved optimization using sequential quadratic programming
    # First, create a better initial configuration using a greedy approach
    def generate_initial_config():
        # Start with a dense hexagonal packing approximation
        config = np.zeros((n, 3))
        
        # Grid size for initial placement
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 0.9 / (grid_size + 1)
        radius = spacing / 2.5  # Initial radius
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                x = spacing * (j + 1) + 0.05
                y = spacing * (i + 1) + 0.05
                config[idx] = [x, y, radius]
                idx += 1
            if idx >= n:
                break
        
        # Adjust for any remaining circles
        while idx < n:
            config[idx] = [
                np.random.uniform(0.05, 0.95),
                np.random.uniform(0.05, 0.95),
                0.03
            ]
            idx += 1
            
        return config
    
    # Generate better initial configuration
    circles = generate_initial_config()
    
    # Refinement using a more sophisticated approach
    # Use iterative improvement with local search
    best_sum = 0
    best_circles = circles.copy()
    
    # Try several random restarts
    for restart in range(5):
        # Start with current configuration
        current_circles = circles.copy()
        
        # Local search: try to improve individual circles
        for _ in range(1000):
            improved = False
            
            # Try to increase radius of each circle
            for i in range(n):
                # Try to increase radius
                old_radius = current_circles[i, 2]
                test_radius = min(old_radius + 0.005, 0.45)
                
                # Check if this is valid
                valid = True
                for j in range(n):
                    if i != j:
                        dx = current_circles[i, 0] - current_circles[j, 0]
                        dy = current_circles[i, 1] - current_circles[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < test_radius + current_circles[j, 2]:
                            valid = False
                            break
                
                if valid:
                    current_circles[i, 2] = test_radius
                    improved = True
            
            if not improved:
                break
        
        # Calculate sum of radii
        current_sum = np.sum(current_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = current_circles.copy()
    
    # Final optimization using scipy minimize with constraints
    def optimize_final():
        # Flatten for optimization
        x0 = best_circles.flatten()
        
        # Define constraints
        def contain_bounds(x_flat):
            # Each circle must be within square with padding for radius
            constraints = []
            for i in range(n):
                x, y, r = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
                constraints.extend([
                    x - r,      # x >= r
                    y - r,      # y >= r
                    1 - x - r,  # 1 - x >= r
                    1 - y - r   # 1 - y >= r
                ])
            return np.array(constraints)
        
        def overlap_constraints(x_flat):
            # Non-overlap constraints
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
                    x2, y2, r2 = x_flat[3*j], x_flat[3*j+1], x_flat[3*j+2]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = np.sqrt(dx*dx + dy*dy)
                    # Distance >= sum of radii (negative for constraint)
                    constraints.append(dist - r1 - r2)
            return np.array(constraints)
        
        # Create bounds
        bounds = [(0.001, 0.999) for _ in range(3*n)]
        
        # Minimize negative sum of radii (equivalent to maximizing sum)
        result = minimize(
            lambda x: -np.sum(x[2::3]),  # Objective: negative sum of radii
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': contain_bounds},
                {'type': 'ineq', 'fun': overlap_constraints}
            ],
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            return result.x.reshape(-1, 3)
        else:
            # If optimization fails, return the best configuration found so far
            return best_circles
    
    final_result = optimize_final()
    return final_result


# EVOLVE-BLOCK-END
