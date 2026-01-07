# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach: initial geometric placement followed by constrained optimization.
    """
    n = 32
    
    # Initialize with a good heuristic placement
    circles = _initial_placement()
    
    # Convert to optimization variables: [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
    initial_vars = _circles_to_vars(circles)
    
    # Define bounds for optimization (x,y in [0,1], r in [0,0.5])
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Define constraints
    constraints = []
    
    # Add containment constraints
    for i in range(n):
        def contain_constraint(vars, idx=i):
            x, y, r = vars[3*idx], vars[3*idx+1], vars[3*idx+2]
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        constraints.append({'type': 'ineq', 'fun': contain_constraint})
    
    # Add non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(vars, idx1=i, idx2=j):
                x1, y1, r1 = vars[3*idx1], vars[3*idx1+1], vars[3*idx1+2]
                x2, y2, r2 = vars[3*idx2], vars[3*idx2+1], vars[3*idx2+2]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                return distance - (r1 + r2)
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    # Optimization objective: minimize negative sum of radii (equivalent to maximizing sum)
    def objective(vars):
        total_radius = sum(vars[3*i+2] for i in range(n))
        return -total_radius
    
    # Run optimization
    try:
        result = minimize(objective, initial_vars, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            circles = _vars_to_circles(result.x)
            return circles
        else:
            # If optimization fails, return the initial placement
            return circles
    except Exception:
        # If anything goes wrong, return initial placement
        return circles

def _initial_placement():
    """Create an initial good placement using a hexagonal packing approach"""
    n = 32
    circles = np.zeros((n, 3))
    
    # Create a grid-like pattern with some randomness to avoid regular patterns
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Distribute points more evenly using a modified hexagonal packing
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            # Add slight jitter to avoid perfect grid
            x = (j + 1) * spacing_x + (np.random.random() - 0.5) * spacing_x * 0.3
            y = (i + 1) * spacing_y + (np.random.random() - 0.5) * spacing_y * 0.3
            
            # Ensure we're inside the unit square
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Set initial radius to a reasonable value based on spacing
            r = min(0.05, 0.5 * spacing_x, 0.5 * spacing_y)
            circles[count] = [x, y, r]
            count += 1
            
            if count >= n:
                break
    
    # Adjust for better packing density
    _refine_initial_placement(circles)
    
    return circles

def _refine_initial_placement(circles):
    """Refine the initial placement to improve density"""
    n = len(circles)
    
    # Try to increase radii while maintaining constraints
    for attempt in range(100):
        improved = False
        for i in range(n):
            # Try to increase radius
            old_r = circles[i, 2]
            max_r = min(
                circles[i, 0], 1 - circles[i, 0],
                circles[i, 1], 1 - circles[i, 1]
            )
            
            # Check if we can increase radius without violating constraints
            new_r = min(max_r, old_r + 0.005)
            
            # Verify non-overlap constraint with neighbors
            valid = True
            for j in range(n):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < (new_r + circles[j, 2]):
                        valid = False
                        break
            
            if valid and new_r > old_r:
                circles[i, 2] = new_r
                improved = True
                
        if not improved:
            break

def _circles_to_vars(circles):
    """Convert circles array to optimization variable vector"""
    vars = []
    for i in range(len(circles)):
        vars.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
    return np.array(vars)

def _vars_to_circles(vars):
    """Convert optimization variable vector back to circles array"""
    n = len(vars) // 3
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [vars[3*i], vars[3*i+1], vars[3*i+2]]
    return circles


# EVOLVE-BLOCK-END
