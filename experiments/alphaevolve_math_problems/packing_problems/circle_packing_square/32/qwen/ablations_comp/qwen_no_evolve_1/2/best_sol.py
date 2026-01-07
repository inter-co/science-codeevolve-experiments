# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a hexagonal packing pattern as starting point
    def initialize_hexagonal():
        circles = []
        # Create a hexagonal lattice pattern
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                # Adjust positions to avoid boundary issues
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                # Initial radius - small enough to fit in square
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles.append([x, y, r])
        
        # Fill remaining circles with random positions
        np.random.seed(42)
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Create initial configuration
    circles = initialize_hexagonal()
    
    # Define constraint functions
    def get_constraints():
        """Generate constraints for the optimization"""
        cons = []
        
        # Boundary constraints: each circle must fit completely in the unit square
        def boundary_constraint(i):
            def constraint(vars):
                x, y, r = vars[3*i:3*i+3]
                return min(r, x-r, 1-x-r, y-r, 1-y-r)
            return constraint
        
        # Non-overlap constraints: distance between centers >= sum of radii
        def overlap_constraint(i, j):
            def constraint(vars):
                x1, y1, r1 = vars[3*i:3*i+3]
                x2, y2, r2 = vars[3*j:3*j+3]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                return dist - (r1 + r2)
            return constraint
        
        # Add boundary constraints for all circles
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
        
        # Add non-overlap constraints for all pairs
        for i in range(n):
            for j in range(i+1, n):
                cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
        
        return cons
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(vars):
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i+2]  # radius is third component
        return -total_radius
    
    # Bounds for variables: [x, y, r] for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r
    
    # Get constraints
    constraints = get_constraints()
    
    # Flatten initial guess
    x0 = circles.flatten()
    
    # Optimization parameters
    options = {'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Run optimization
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=1e-6
        )
        
        if result.success:
            # Extract optimized values
            optimized_circles = result.x.reshape(-1, 3)
            circles = optimized_circles.copy()
        else:
            # If optimization fails, return the initial configuration
            pass
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        pass
    
    # Final refinement: ensure all constraints are met and adjust radii if needed
    def validate_and_refine():
        # Ensure all circles are valid
        refined_circles = circles.copy()
        
        # Check and fix boundary violations
        for i in range(n):
            x, y, r = refined_circles[i]
            # Make sure circle fits in unit square
            max_r = min(x, 1-x, y, 1-y)
            if r > max_r:
                refined_circles[i, 2] = max_r * 0.99  # Slightly smaller to ensure validity
        
        # Apply iterative improvement for overlap removal
        improved = True
        iterations = 0
        while improved and iterations < 20:
            improved = False
            for i in range(n):
                x1, y1, r1 = refined_circles[i]
                # Try to increase radius if possible
                max_r = min(x1, 1-x1, y1, 1-y1)
                new_r = min(max_r, r1 * 1.01)  # Slightly increase
                
                # Check if we can increase radius without violating other constraints
                valid = True
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = refined_circles[j]
                        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if dist < (new_r + r2):
                            valid = False
                            break
                
                if valid and new_r > r1:
                    refined_circles[i, 2] = new_r
                    improved = True
                    
            iterations += 1
            
        return refined_circles
    
    final_circles = validate_and_refine()
    
    return final_circles


# EVOLVE-BLOCK-END
