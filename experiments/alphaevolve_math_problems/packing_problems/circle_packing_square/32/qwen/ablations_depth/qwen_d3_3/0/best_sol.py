# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a mathematical optimization approach with proper constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using a hexagonal lattice pattern
    def initialize_hexagonal():
        circles = []
        
        # Arrange in a hexagonal pattern with approximately 6 rows
        rows = 6
        cols_per_row = [5, 6, 5, 6, 5, 6]
        
        sqrt3 = math.sqrt(3)
        row_spacing = 1.0
        col_spacing = 1.0
        
        # Calculate total dimensions needed
        total_width = max(cols_per_row) * col_spacing
        total_height = rows * row_spacing * sqrt3 / 2
        
        # Scale to fit in unit square
        scale_x = 1.0 / total_width if total_width > 0 else 1.0
        scale_y = 1.0 / total_height if total_height > 0 else 1.0
        scale = min(scale_x, scale_y) * 0.8  # Leave margin
        
        # Generate positions
        y_offset = 0.5 * scale * row_spacing * sqrt3 / 2
        for i in range(rows):
            row_cols = cols_per_row[i]
            x_offset = 0.5 * scale * col_spacing
            if i % 2 == 1:  # Offset every other row
                x_offset += 0.5 * scale * col_spacing
            
            for j in range(row_cols):
                x = x_offset + j * scale * col_spacing
                y = y_offset + i * scale * row_spacing * sqrt3 / 2
                
                # Ensure within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append([x, y, 0.05 * scale])  # Initial radius
                    
        # Pad to exactly 32 circles if needed
        while len(circles) < n:
            circles.append([0.5, 0.5, 0.02])
            
        return np.array(circles[:n])
    
    # Initialize
    initial_circles = initialize_hexagonal()
    
    # Define optimization variables: [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
    initial_vars = initial_circles.flatten()
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(vars):
        radii = vars[2::3]  # Every third element starting from index 2
        return -np.sum(radii)  # Negative because we're minimizing
    
    # Constraint functions - optimized to reduce redundancy
    def constraint_non_overlap(vars):
        """Non-overlap constraints"""
        circles = vars.reshape(-1, 3)
        constraints = []
        
        # Non-overlap constraints - only compute upper triangle to avoid duplicates
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist_sq = (xi-xj)**2 + (yi-yj)**2
                dist = math.sqrt(dist_sq)
                constraints.append(dist - (ri + rj))  # Must be >= 0
        
        return np.array(constraints)
    
    def constraint_boundaries(vars):
        """Boundary constraints - ensure circles stay inside unit square"""
        circles = vars.reshape(-1, 3)
        constraints = []
        
        # Each circle must be fully contained in [0,1]x[0,1]
        for i in range(n):
            x, y, r = circles[i]
            # Distance to each boundary must be >= r
            constraints.append(x - r)      # Left boundary
            constraints.append(y - r)      # Bottom boundary
            constraints.append(1 - x - r)  # Right boundary
            constraints.append(1 - y - r)  # Top boundary
            
        return np.array(constraints)
    
    # Set up constraints
    cons = []
    
    # Add non-overlap constraints
    cons.append({'type': 'ineq', 'fun': constraint_non_overlap})
    
    # Add boundary constraints
    cons.append({'type': 'ineq', 'fun': constraint_boundaries})
    
    # Bounds for variables: x,y in [0,1], r in [0,0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Use scipy's minimize with SLSQP method
    try:
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2000, 'ftol': 1e-7, 'eps': 1e-7, 'iprint': -1}
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            # Ensure final positions are valid
            for i in range(n):
                x, y, r = final_circles[i]
                # Clamp positions to valid range
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                final_circles[i] = [x, y, r]
            return final_circles
        else:
            # Return initial configuration if optimization fails
            return initial_circles
    except Exception as e:
        # If optimization fails, return initial configuration
        return initial_circles


# EVOLVE-BLOCK-END
