# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial hexagonal grid placement with scipy optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Create initial configuration using hexagonal grid pattern
    def create_hexagonal_initial():
        # For 26 circles, we can arrange them in approximately 5 rows with varying column counts
        # Try to distribute them in a hexagonal pattern
        circles = []
        
        # Hexagonal packing parameters
        # In hexagonal packing, the horizontal spacing is sqrt(3)/2 times the vertical spacing
        # For circles of equal radius r, we want to fit them optimally
        
        # Try different arrangements - let's start with 5 rows
        rows = 5
        cols_per_row = [5, 4, 5, 4, 5]  # alternating pattern
        
        # Calculate appropriate radius for initial placement
        max_radius = 0.1  # Start with small radius
        
        # Place circles in hexagonal pattern
        y_offset = max_radius
        row_height = max_radius * 2 * 0.866  # sqrt(3)/2 factor for hexagonal packing
        
        for i in range(rows):
            row_cols = cols_per_row[i]
            x_offset = max_radius if i % 2 == 0 else max_radius * 1.5  # Offset every other row
            
            for j in range(row_cols):
                x_pos = x_offset + j * max_radius * 2
                y_pos = y_offset + i * row_height
                
                # Check if circle fits in unit square
                if x_pos - max_radius >= 0 and x_pos + max_radius <= 1 and \
                   y_pos - max_radius >= 0 and y_pos + max_radius <= 1:
                    circles.append([x_pos, y_pos, max_radius])
        
        # If we didn't get enough circles, add more in a systematic way
        if len(circles) < n:
            # Fill remaining positions systematically
            for i in range(len(circles), n):
                # Simple grid placement for remaining circles
                row = i // 5
                col = i % 5
                x = 0.1 + col * 0.18
                y = 0.1 + row * 0.18
                r = 0.05
                circles.append([x, y, r])
                
        return np.array(circles[:n])
    
    # Initialize with hexagonal pattern
    circles = create_hexagonal_initial()
    
    # Define constraint functions
    def get_constraints():
        """Generate constraint functions for optimization"""
        constraints = []
        
        # Boundary constraints: each circle must be within the unit square
        def boundary_constraint(i):
            def constraint(x):
                # x[3*i:3*i+2] are (x,y) coordinates, x[3*i+2] is radius
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                # Circle must be within bounds
                return min(xi - ri, 1 - xi - ri, yi - ri, 1 - yi - ri)
            return constraint
        
        # Non-overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(i, j):
            def constraint(x):
                # x[3*i:3*i+2] are (x,y) coordinates for circle i
                # x[3*j:3*j+2] are (x,y) coordinates for circle j
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                # Distance between centers minus sum of radii should be >= 0
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                return dist - ri - rj
            return constraint
        
        # Add boundary constraints
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Add non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        return constraints
    
    # Objective function to maximize sum of radii
    def objective(x):
        # We want to maximize sum of radii, so we minimize negative sum
        total_radius = sum(x[3*i+2] for i in range(n))
        return -total_radius
    
    # Flatten initial configuration for optimization
    x0 = np.array([circles[i][j] for i in range(n) for j in range(3)])
    
    # Get constraints
    constraints = get_constraints()
    
    # Bounds for variables: x, y in [0,1], radius in [0,0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # (x_min, x_max), (y_min, y_max), (r_min, r_max)
    
    try:
        # Perform optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            # Extract final configuration
            final_circles = []
            for i in range(n):
                x = result.x[3*i]
                y = result.x[3*i+1]
                r = result.x[3*i+2]
                final_circles.append([x, y, r])
            
            return np.array(final_circles)
        else:
            # If optimization fails, return the initial configuration
            return circles
            
    except Exception as e:
        # Return initial configuration if optimization fails
        return circles


# EVOLVE-BLOCK-END
