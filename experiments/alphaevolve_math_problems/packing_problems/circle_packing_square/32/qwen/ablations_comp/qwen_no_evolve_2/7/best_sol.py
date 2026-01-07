# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial placement followed by optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initial placement strategy: arrange in a hexagonal grid pattern
    # This gives a good starting configuration that's likely to be feasible
    def initial_placement():
        circles = np.zeros((n, 3))
        
        # Arrange in roughly a 6x6 grid with some spacing
        rows = 6
        cols = 6
        if n < rows * cols:
            rows = math.ceil(n / cols)
            
        # Calculate spacing based on number of circles
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for hexagonal packing
                x_offset = 0.0 if i % 2 == 0 else spacing_x / 2
                x = (j + 1) * spacing_x + x_offset
                y = (i + 1) * spacing_y
                
                # Set initial radius to be small but feasible
                r = min(spacing_x, spacing_y) / 3
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
        
        # Fill remaining slots with circles at random valid positions
        for i in range(idx, n):
            # Random valid position
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = 0.02  # Small initial radius
            circles[i] = [x, y, r]
            
        return circles
    
    # Constraint functions
    def radius_constraint(circles_flat):
        """Ensure all radii are positive"""
        radii = circles_flat[2::3]
        return radii
    
    def containment_constraint(circles_flat):
        """Ensure all circles are within the unit square"""
        positions = circles_flat.reshape(-1, 3)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        radii = positions[:, 2]
        
        # Min distance from boundaries should be >= radius
        left_dist = x_coords - radii
        right_dist = 1 - x_coords - radii
        bottom_dist = y_coords - radii
        top_dist = 1 - y_coords - radii
        
        return np.minimum.reduce([left_dist, right_dist, bottom_dist, top_dist])
    
    def overlap_constraint(circles_flat):
        """Ensure no overlapping circles"""
        positions = circles_flat.reshape(-1, 3)
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        radii = positions[:, 2]
        
        # Compute pairwise distances
        coords = np.column_stack([x_coords, y_coords])
        distances = cdist(coords, coords)
        
        # Create constraint: distance >= sum of radii for non-identical circles
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                min_distance = radii[i] + radii[j]
                actual_distance = distances[i, j]
                constraints.append(actual_distance - min_distance)
        
        return np.array(constraints)
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])
    
    # Initialize
    circles = initial_placement()
    
    # Flatten for optimization
    circles_flat = circles.flatten()
    
    # Optimization bounds (x,y in [0,1], radius > 0)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])  # x, y, r bounds
    
    # Constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: radius_constraint(x)},
        {'type': 'ineq', 'fun': lambda x: containment_constraint(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
    ]
    
    # Run optimization
    try:
        result = minimize(objective, circles_flat, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            circles = result.x.reshape(-1, 3)
        else:
            # If optimization fails, return the initial placement
            pass
    except Exception:
        # If optimization fails due to any error, return initial placement
        pass
    
    # Ensure final constraints are satisfied
    # Adjust positions to satisfy boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        # Adjust for boundary constraints
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]
    
    # Final adjustment to ensure no overlaps
    # Simple greedy approach to reduce overlaps if any remain
    for _ in range(100):  # Limited iterations to prevent infinite loop
        overlap_found = False
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < r1 + r2:
                    # Move circles apart
                    overlap = (r1 + r2) - dist
                    dx = (x2 - x1) / dist if dist > 0 else 0
                    dy = (y2 - y1) / dist if dist > 0 else 0
                    
                    # Reduce overlap by moving circles apart
                    move_amount = overlap * 0.5
                    circles[i][0] -= dx * move_amount
                    circles[i][1] -= dy * move_amount
                    circles[j][0] += dx * move_amount
                    circles[j][1] += dy * move_amount
                    
                    # Clamp to boundaries
                    circles[i][0] = np.clip(circles[i][0], circles[i][2], 1-circles[i][2])
                    circles[i][1] = np.clip(circles[i][1], circles[i][2], 1-circles[i][2])
                    circles[j][0] = np.clip(circles[j][0], circles[j][2], 1-circles[j][2])
                    circles[j][1] = np.clip(circles[j][1], circles[j][2], 1-circles[j][2])
                    
                    overlap_found = True
        if not overlap_found:
            break
    
    return circles


# EVOLVE-BLOCK-END
