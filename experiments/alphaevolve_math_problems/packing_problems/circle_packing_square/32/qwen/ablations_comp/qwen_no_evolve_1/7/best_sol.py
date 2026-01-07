# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Generate initial configuration using hexagonal packing approximation
    def generate_initial_config():
        # Try to place circles in a hexagonal pattern first
        circles = []
        
        # Start with a simple grid-based approach for initial placement
        rows = int(math.sqrt(n)) + 1
        cols = int(math.ceil(n / rows))
        
        # Create a more structured initial layout
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        for i in range(min(n, rows * cols)):
            row = i // cols
            col = i % cols
            x = (col + 1) * spacing_x
            y = (row + 1) * spacing_y
            # Initial radius - small enough to fit in square
            r = min(spacing_x, spacing_y) * 0.3
            circles.append([x, y, r])
            
        # Fill remaining circles with smaller radii in corners
        for i in range(len(circles), n):
            x = 0.1 + (i % 3) * 0.3
            y = 0.1 + (i // 3) * 0.3
            r = 0.02
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # Phase 2: Define constraint functions
    def get_constraints(circles):
        """Generate constraints for optimization"""
        constraints = []
        
        # Boundary constraints: each circle must fit completely in unit square
        for i in range(n):
            x, y, r = circles[i]
            # Circle must be within bounds
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1 - x - r >= 0
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})  # 1 - y - r >= 0
            
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                def overlap_constraint(c, i=i, j=j):
                    x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                    x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    return dist_sq - (r1 + r2)**2  # Should be >= 0
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
                
        return constraints
    
    # Phase 3: Optimization objective and constraints
    def objective(circles_flat):
        # Sum of all radii
        return -np.sum(circles_flat[2::3])  # Negative because we minimize
    
    def boundary_constraints(circles_flat):
        # Ensure all circles stay within unit square
        constraints = []
        for i in range(n):
            x, y, r = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
            constraints.extend([
                x - r,           # x >= r
                y - r,           # y >= r
                1 - x - r,       # 1 - x >= r
                1 - y - r        # 1 - y >= r
            ])
        return np.array(constraints)
    
    def overlap_constraints(circles_flat):
        # Ensure no overlaps between circles
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_flat[3*i], circles_flat[3*i+1], circles_flat[3*i+2]
                x2, y2, r2 = circles_flat[3*j], circles_flat[3*j+1], circles_flat[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # Distance squared should be >= (r1 + r2)^2
                constraints.append(dist_sq - (r1 + r2)**2)
        return np.array(constraints)
    
    # Phase 4: Run optimization
    try:
        # Generate initial configuration
        initial_circles = generate_initial_config()
        initial_flat = initial_circles.flatten()
        
        # Set up bounds for optimization (x, y, r for each circle)
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r
        
        # Optimization options
        options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        
        # Run optimization
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': lambda x: boundary_constraints(x)},
                {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
            ],
            options=options,
            tol=1e-6
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure all circles are valid
            final_circles = []
            for i in range(n):
                x, y, r = optimized_circles[i]
                # Clamp values to valid ranges
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                r = max(0.001, min(0.499, r))
                final_circles.append([x, y, r])
            return np.array(final_circles)
        else:
            # Fallback to initial configuration if optimization fails
            return initial_circles
            
    except Exception as e:
        # Fallback to initial configuration if anything goes wrong
        return generate_initial_config()


# EVOLVE-BLOCK-END
