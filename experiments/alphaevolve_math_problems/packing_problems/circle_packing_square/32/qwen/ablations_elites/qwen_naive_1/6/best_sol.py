# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Enhanced initialization using hexagonal packing (inspired by INSPIRATION 1)
    def initialize_hexagonal():
        # Create a hexagonal packing pattern that fits well in the unit square
        rows = 6
        cols = 6
        
        # Hexagonal spacing - optimized for unit square
        spacing_x = 1.0 / (cols + 0.5)  # Leave some margin
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                # Offset every other row for hexagonal packing
                x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                if 0 <= x <= 1 and 0 <= y <= 1:
                    positions.append([x, y])
        
        # Trim to exact number needed
        positions = positions[:n]
        
        # Set initial radii - start with a reasonable value based on spacing
        radii = [spacing_x * 0.3] * n
        
        return np.array(positions), radii
    
    # Generate initial configuration with better hexagonal layout
    positions, radii = initialize_hexagonal()
    initial_circles = np.column_stack([positions, radii])
    
    # Define objective function (negative because we want to maximize)
    def objective(params):
        # Reshape params into (x, y, r) for each circle
        circles = params.reshape(-1, 3)
        radii_sum = np.sum(circles[:, 2])
        return -radii_sum  # Negative because we're minimizing
    
    # Improved constraint handling with better numerical stability
    def containment_constraint(params):
        circles = params.reshape(-1, 3)
        # Each circle must be fully contained in the unit square
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]
        
        # Check containment constraints (positive when satisfied)
        cons = []
        for i in range(len(circles)):
            # Left boundary (x - radius >= 0)  
            cons.append(x_coords[i] - radii[i])
            # Right boundary (1 - x - radius >= 0) 
            cons.append(1 - x_coords[i] - radii[i])
            # Bottom boundary (y - radius >= 0)
            cons.append(y_coords[i] - radii[i])  
            # Top boundary (1 - y - radius >= 0)
            cons.append(1 - y_coords[i] - radii[i]) 
            
        return np.array(cons)
    
    def overlap_constraint(params):
        circles = params.reshape(-1, 3)
        # Distance between centers must be >= sum of radii
        distances = cdist(circles[:, :2], circles[:, :2])
        radii = circles[:, 2]
        cons = []
        
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                # We want dist >= min_dist, so constraint is dist - min_dist >= 0
                # Add small epsilon to handle numerical precision issues
                cons.append(dist - min_dist)
                
        return np.array(cons)
    
    # Flatten initial values: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = initial_circles.flatten()
    
    # Set up bounds for optimization - ensure radii are reasonable
    bounds = []
    for i in range(n):
        # x coordinates: [radius, 1-radius] - ensure we don't go too close to edges
        bounds.append((0.001, 0.999))
        # y coordinates: [radius, 1-radius] 
        bounds.append((0.001, 0.999))
        # radii: [0.001, 0.49] (smaller upper bound to prevent extreme values)
        bounds.append((0.001, 0.49))
    
    # Define constraints for scipy optimization
    containment_cons = {
        'type': 'ineq',
        'fun': lambda x: containment_constraint(x)
    }
    
    overlap_cons = {
        'type': 'ineq', 
        'fun': lambda x: overlap_constraint(x)
    }
    
    # Try multiple optimization approaches for better results
    best_result = None
    best_sum = 0
    
    # Try SLSQP with tight tolerances
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=[containment_cons, overlap_cons],
            options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-8, 'iprint': -1}
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            current_sum = np.sum(final_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = final_circles
    except Exception as e:
        pass
    
    # If SLSQP failed or didn't produce a good result, try L-BFGS-B with different settings
    if best_result is None:
        try:
            # More aggressive optimization with better options
            result2 = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2500, 'ftol': 1e-8, 'gtol': 1e-8, 'iprint': -1}
            )
            
            if result2.success:
                final_circles = result2.x.reshape(-1, 3)
                current_sum = np.sum(final_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = final_circles
        except Exception as e:
            pass
    
    # If no optimization worked, return the initial hexagonal configuration
    if best_result is None:
        return initial_circles
    
    return best_result


# EVOLVE-BLOCK-END
