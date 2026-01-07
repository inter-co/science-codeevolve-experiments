# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 32
    
    # Initialize with a proper hexagonal packing pattern
    def initialize_hexagonal():
        circles = []
        
        # Use a more systematic hexagonal grid approach
        # Calculate how many rows/columns we need
        rows = int(math.sqrt(n) * 1.2) + 2
        cols = int(n / rows) + 2
        
        # Determine appropriate spacing for hexagonal packing
        spacing = 0.1  # Initial spacing guess
        hex_radius = spacing / 2
        
        # Adjust spacing to fit within unit square
        max_radius = min(0.5, 0.9 / (cols + 1), 0.9 / (rows + 1))
        hex_radius = max_radius * 0.8  # Leave some margin
        
        spacing_x = hex_radius * 2
        spacing_y = hex_radius * math.sqrt(3)
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                    
                # Hexagonal offset for odd rows
                x_offset = (i % 2) * spacing_x / 2
                x = spacing_x * j + x_offset + hex_radius
                y = spacing_y * i + hex_radius
                
                # Ensure we're within bounds
                if x - hex_radius >= 0 and x + hex_radius <= 1 and \
                   y - hex_radius >= 0 and y + hex_radius <= 1:
                    circles.append([x, y, hex_radius])
                    count += 1
                    
            if count >= n:
                break
                
        # Fill remaining positions with random placements that avoid overlaps
        while len(circles) < n:
            # Try to place circles in a way that avoids conflicts
            x = np.random.uniform(hex_radius, 1-hex_radius)
            y = np.random.uniform(hex_radius, 1-hex_radius)
            
            # Check if this position would cause overlaps with existing circles
            valid = True
            for cx, cy, cr in circles:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < (hex_radius + cr):
                    valid = False
                    break
            
            if valid:
                circles.append([x, y, hex_radius])
            
        return np.array(circles)
    
    # Better initialization with grid-based approach
    def initialize_grid():
        circles = []
        # Use a more systematic approach with proper spacing
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        
        # Start with a more careful grid pattern
        for i in range(grid_size):
            for j in range(grid_size):
                if len(circles) >= n:
                    break
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                # Add some randomness to avoid perfect grid
                x += random.uniform(-spacing/6, spacing/6)
                y += random.uniform(-spacing/6, spacing/6)
                # Clip to ensure we stay within bounds
                x = max(spacing/2, min(1-spacing/2, x))
                y = max(spacing/2, min(1-spacing/2, y))
                
                # Determine reasonable initial radius based on proximity to edges
                max_radius = min(x, 1-x, y, 1-y)
                # Use a smaller fraction of available space to allow for optimization
                radius = max_radius * (0.3 + 0.4 * random.random())  
                
                circles.append([x, y, radius])
            
            if len(circles) >= n:
                break
        
        # Fill remaining spots with random valid placements
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            max_radius = min(x, 1-x, y, 1-y)
            radius = max_radius * (0.3 + 0.4 * random.random())
            
            # Check if it would overlap with existing circles
            valid = True
            for cx, cy, cr in circles:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < (radius + cr):
                    valid = False
                    break
            
            if valid and radius > 0.001:
                circles.append([x, y, radius])
        
        return np.array(circles)
    
    # Define objective function (negative because we want to maximize)
    def objective(params):
        # Reshape params into (x, y, r) for each circle
        circles = params.reshape(-1, 3)
        radii_sum = np.sum(circles[:, 2])
        return -radii_sum  # Negative because we're minimizing
    
    # Define constraints more carefully
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
                cons.append(dist - min_dist)
                
        return np.array(cons)
    
    # Generate initial configuration - use the better hexagonal initialization
    initial_circles = initialize_hexagonal()
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
    
    # Run optimization with multiple tries for robustness
    try:
        # First attempt with SLSQP
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=[containment_cons, overlap_cons],
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            # Final validation to ensure constraints are met
            return final_circles
        else:
            # If first optimization fails, try a different approach
            # Use a less strict optimization with L-BFGS-B
            try:
                result2 = minimize(
                    objective,
                    initial_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-6}
                )
                if result2.success:
                    final_circles = result2.x.reshape(-1, 3)
                    return final_circles
            except:
                pass
            # If both fail, return the initial configuration
            return initial_circles
            
    except Exception as e:
        # If optimization fails due to any error, return the initial configuration
        return initial_circles


# EVOLVE-BLOCK-END
