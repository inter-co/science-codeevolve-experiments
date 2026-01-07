# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

# Global constants
CIRCLE_COUNT = 32
UNIT_SQUARE_SIZE = 1.0

def initialize_better_hexagonal_placement() -> np.ndarray:
    """Initialize circles using a better hexagonal grid pattern similar to inspiration 2"""
    circles = np.zeros((CIRCLE_COUNT, 3))
    
    # Create a more systematic hexagonal grid pattern
    rows = 6
    cols = 6
    
    # Adjust spacing to fit within unit square with proper margins
    spacing_x = 0.85 / cols
    spacing_y = 0.85 / rows
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= CIRCLE_COUNT:
                break
            x = 0.075 + (j + 1) * spacing_x
            y = 0.075 + (i + 1) * spacing_y
            
            # Slightly stagger odd rows for hexagonal pattern
            if i % 2 == 1:
                x += spacing_x * 0.5
                
            # Initial radius - start with smaller radius to allow for optimization
            max_radius = min(x, 1-x, y, 1-y) * 0.35
            circles[idx] = [x, y, max_radius]
            idx += 1
        if idx >= CIRCLE_COUNT:
            break
    
    # Fill remaining positions with carefully placed circles
    np.random.seed(42)
    for i in range(idx, CIRCLE_COUNT):
        # Try to place near existing circles or in free areas
        if i < 20:  # Use more systematic placement for first few
            x = 0.1 + (i % 5) * 0.18
            y = 0.1 + (i // 5) * 0.18
            max_radius = min(x, 1-x, y, 1-y) * 0.3
        else:  # Random placement with good bounds
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, 0.9)
            max_radius = min(x, 1-x, y, 1-y) * 0.25
        
        circles[i] = [x, y, max_radius]
    
    return circles

def get_constraints():
    """Create constraint functions for scipy optimization like inspiration 2"""
    constraints = []
    
    # Boundary constraints for each circle
    for i in range(CIRCLE_COUNT):
        # x >= r and x <= 1-r
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})  # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})  # 1 - x - r >= 0
        # y >= r and y <= 1-r
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})  # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})  # 1 - y - r >= 0
    
    # Non-overlap constraints (more efficient implementation)
    def overlap_constraint(i, j):
        def constraint(x):
            x1, y1, r1 = x[3*i], x[3*i+1], x[3*i+2]
            x2, y2, r2 = x[3*j], x[3*j+1], x[3*j+2]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            dist = np.sqrt(dist_sq)
            # Return positive value when constraint is satisfied (no overlap)
            return dist - (r1 + r2)
        return constraint
    
    # Add all non-overlap constraints
    for i in range(CIRCLE_COUNT):
        for j in range(i+1, CIRCLE_COUNT):
            constraints.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
    
    return constraints

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Multi-start optimization to avoid local minima
    best_circles = None
    best_sum = -np.inf
    
    # Try multiple optimization runs with different initializations
    for run in range(3):
        # Initialize with better hexagonal pattern
        circles = initialize_better_hexagonal_placement()
        
        # Slight perturbation of the initial configuration
        np.random.seed(42 + run)
        for i in range(CIRCLE_COUNT):
            circles[i, 0] += np.random.uniform(-0.01, 0.01)
            circles[i, 1] += np.random.uniform(-0.01, 0.01)
            circles[i, 2] += np.random.uniform(-0.005, 0.005)
        
        # Ensure they're still valid
        for i in range(CIRCLE_COUNT):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
            circles[i, 2] = max(0.001, min(0.499, circles[i, 2]))
        
        # Prepare optimization parameters
        initial_params = circles.flatten()
        
        # Define bounds for each parameter (x, y, r) for each circle
        bounds = []
        for i in range(CIRCLE_COUNT):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
        
        # Get constraint functions
        constraints = get_constraints()
        
        # Objective function (negative because we want to maximize sum of radii)
        def objective(params):
            # Extract radii
            radii = params[2::3]  # Every third element starting from index 2
            # Return negative sum (since minimize minimizes)
            return -np.sum(radii)
        
        # Optimization with SLSQP method which handles constraints well
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6},
                tol=1e-6
            )
            
            if result.success:
                final_params = result.x
                optimized_circles = final_params.reshape((CIRCLE_COUNT, 3))
                
                # Check if this is better
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized_circles
                    
        except Exception:
            # If optimization fails, continue with next run
            continue
    
    # If no optimization succeeded, use the initial configuration
    if best_circles is None:
        circles = initialize_better_hexagonal_placement()
        return circles
    
    # Final validation and cleanup
    for i in range(CIRCLE_COUNT):
        # Ensure radius is positive
        best_circles[i, 2] = max(0.001, best_circles[i, 2])
        
        # Ensure circle is within bounds
        best_circles[i, 0] = np.clip(best_circles[i, 0], best_circles[i, 2], 1 - best_circles[i, 2])
        best_circles[i, 1] = np.clip(best_circles[i, 1], best_circles[i, 2], 1 - best_circles[i, 2])
        
        # Ensure radius doesn't exceed boundary constraints
        boundary_radius = min(best_circles[i, 0], 1 - best_circles[i, 0], 
                             best_circles[i, 1], 1 - best_circles[i, 1])
        best_circles[i, 2] = min(best_circles[i, 2], boundary_radius)
    
    return best_circles


# EVOLVE-BLOCK-END
