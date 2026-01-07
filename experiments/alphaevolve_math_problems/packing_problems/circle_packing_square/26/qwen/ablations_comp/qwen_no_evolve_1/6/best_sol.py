# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining spatial partitioning and gradient-based optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a simple heuristic placement
    initial_circles = _initialize_placement(n)
    
    # Convert to optimization variables (x, y, r for each circle)
    initial_vars = []
    for i in range(n):
        initial_vars.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define constraint functions
    def constraint_func(vars):
        # Extract positions and radii
        circles = []
        for i in range(n):
            x = vars[3*i]
            y = vars[3*i+1]
            r = vars[3*i+2]
            circles.append([x, y, r])
        
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < r1 + r2:
                    return False
        return True
    
    # Define objective function (negative because we want to maximize)
    def objective(vars):
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i+2]  # radius is at index 3*i+2
        return -total_radius
    
    # Define constraint dictionaries for scipy.optimize
    constraints = []
    
    # Add containment constraints
    def containment_constraint(vars, i):
        x = vars[3*i]
        y = vars[3*i+1]
        r = vars[3*i+2]
        return min(x-r, 1-x-r, y-r, 1-y-r)  # Return minimum distance to boundary
    
    # Add overlap constraints
    def overlap_constraint(vars, i, j):
        x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
        x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        return distance - (r1 + r2)  # Should be >= 0
    
    # Create constraints for bounds (containment)
    bounds = []
    for i in range(n):
        # x, y, r bounds
        bounds.append((0.001, 0.999))  # x coordinate
        bounds.append((0.001, 0.999))  # y coordinate  
        bounds.append((0.001, 0.5))    # radius (max radius constrained by square size)
    
    # Create inequality constraints for overlaps
    for i in range(n):
        for j in range(i+1, n):
            def make_overlap_constraint(i, j):
                def constraint(vars):
                    return overlap_constraint(vars, i, j)
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
    
    # Create inequality constraints for containment
    for i in range(n):
        def make_containment_constraint(i):
            def constraint(vars):
                return containment_constraint(vars, i)
            return constraint
        constraints.append({'type': 'ineq', 'fun': make_containment_constraint(i)})
    
    # Optimize using SLSQP method
    try:
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_vars = result.x
        else:
            # Fallback to initial placement if optimization fails
            optimized_vars = initial_vars
    except Exception as e:
        # If optimization fails completely, use initial placement
        optimized_vars = initial_vars
    
    # Convert back to circles array
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i][0] = optimized_vars[3*i]      # x
        circles[i][1] = optimized_vars[3*i+1]    # y  
        circles[i][2] = optimized_vars[3*i+2]    # r
    
    return circles

def _initialize_placement(n):
    """Initialize circle positions using a hexagonal packing approach"""
    circles = []
    
    # Start with a simple grid-like arrangement
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Adjust dimensions to fit in unit square
    cell_width = 1.0 / cols
    cell_height = 1.0 / rows
    
    # Place circles with some padding
    padding = 0.05
    max_radius = min(cell_width, cell_height) * (1 - 2*padding) / 2
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
                
            # Position circle at center of cell with padding
            x = (j + 0.5) * cell_width
            y = (i + 0.5) * cell_height
            
            # Ensure we're within bounds
            x = max(max_radius, min(1-max_radius, x))
            y = max(max_radius, min(1-max_radius, y))
            
            circles.append([x, y, max_radius])
            count += 1
            
        if count >= n:
            break
    
    # Fill remaining slots with small radii
    while len(circles) < n:
        circles.append([0.5, 0.5, 0.01])
    
    return circles


# EVOLVE-BLOCK-END
