# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining grid initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize using a grid-based approach for good starting configuration
    def initialize_grid():
        circles = np.zeros((n, 3))
        # Create a roughly hexagonal grid pattern
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Adjust grid to fit within unit square with margin
        margin = 0.05
        cell_size_x = (1 - 2*margin) / cols
        cell_size_y = (1 - 2*margin) / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = margin + (j + 0.5) * cell_size_x
                y = margin + (i + 0.5) * cell_size_y
                # Initial radius - small enough to fit in cell
                r = min(cell_size_x, cell_size_y) * 0.3
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= n:
                break
        return circles
    
    # Check if configuration is valid
    def is_valid(circles):
        # Check containment
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        # Check overlaps
        positions = circles[:, :2]
        radii = circles[:, 2]
        distances = cdist(positions, positions)
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                if distances[i,j] < radii[i] + radii[j]:
                    return False
        return True
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])
    
    # Constraints
    def containment_constraint(params):
        circles = params.reshape(-1, 3)
        constraints = []
        for x, y, r in circles:
            # r <= x <= 1-r
            constraints.append(x - r)  # Should be >= 0
            constraints.append(1 - x - r)  # Should be >= 0
            # r <= y <= 1-r
            constraints.append(y - r)  # Should be >= 0
            constraints.append(1 - y - r)  # Should be >= 0
        return np.array(constraints)
    
    def overlap_constraint(params):
        circles = params.reshape(-1, 3)
        constraints = []
        positions = circles[:, :2]
        radii = circles[:, 2]
        distances = cdist(positions, positions)
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                # Distance should be >= radii[i] + radii[j]
                dist = distances[i,j]
                constraints.append(dist - radii[i] - radii[j])  # Should be >= 0
        return np.array(constraints)
    
    # Generate initial configuration
    circles = initialize_grid()
    
    # Refine using optimization
    # Flatten parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = circles.flatten()
    
    # Set bounds for optimization
    bounds = []
    for i in range(n):
        # x bounds: [r, 1-r]
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Create constraint dictionaries
    constraints = [
        {'type': 'ineq', 'fun': lambda x: containment_constraint(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
    ]
    
    # Optimization with bounds
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            refined_circles = result.x.reshape(-1, 3)
            # Ensure final validation
            if is_valid(refined_circles):
                return refined_circles
    except Exception:
        pass
    
    # Fallback to grid-based solution if optimization fails
    return circles


# EVOLVE-BLOCK-END
