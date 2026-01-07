# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a mathematical programming approach with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Generate initial configuration using a more systematic approach
    # Start with a regular grid pattern, then optimize
    
    # Create a more sophisticated initial placement based on circle packing theory
    # For 32 circles in a unit square, we can use a pattern inspired by hexagonal packing
    # But ensure no overlaps initially
    
    # Determine grid layout that works well for 32 circles
    rows = 6
    cols = 6
    # Adjust for 32 circles (we'll have some empty spaces)
    
    # Create initial positions with proper spacing
    positions = []
    radii = []
    
    # Use a grid pattern with some randomness to avoid symmetry issues
    np.random.seed(42)  # For reproducibility
    
    # Generate points in a structured way
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    # Place circles in a grid pattern with slight randomness
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Add some jitter to avoid perfect symmetry
            x = spacing_x * (j + 1) + np.random.uniform(-spacing_x*0.1, spacing_x*0.1)
            y = spacing_y * (i + 1) + np.random.uniform(-spacing_y*0.1, spacing_y*0.1)
            
            # Ensure within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            positions.append([x, y])
            # Start with small but reasonable radii
            radii.append(0.05)
    
    # Trim to exactly 32 circles if needed
    positions = np.array(positions[:n])
    radii = np.array(radii[:n])
    
    # Convert to flattened parameter vector: [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
    params = np.concatenate([positions.flatten(), radii])
    
    def objective(params):
        # Extract positions and radii
        positions = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        
        # Return negative because we want to maximize sum of radii
        return -np.sum(radii)
    
    def constraint_func(params):
        """Constraint function returning positive values when satisfied"""
        positions = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        
        # Check containment constraints (each circle must be fully inside unit square)
        containment = np.ones(n)
        for i in range(n):
            # Circle i must be within bounds
            containment[i] = min(
                positions[i, 0] - radii[i],      # left boundary
                1 - positions[i, 0] - radii[i],  # right boundary
                positions[i, 1] - radii[i],      # bottom boundary
                1 - positions[i, 1] - radii[i]   # top boundary
            )
        
        # Check overlap constraints
        overlap = np.ones(n*(n-1)//2)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Constraint is that distance >= radii[i] + radii[j]
                # So we want: dist - radii[i] - radii[j] >= 0
                overlap[idx] = dist - radii[i] - radii[j]
                idx += 1
        
        # Combine all constraints (positive means satisfied)
        return np.concatenate([containment, overlap])
    
    # Define bounds for variables
    # Positions: [0,1] for x and y
    # Radii: [0.001, 0.5] to prevent degenerate cases
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1)])  # x, y bounds
    for i in range(n):
        bounds.extend([(0.001, 0.5)])    # radius bounds
    
    # Create constraint dictionary
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Optimize using SLSQP method which handles constraints well
    try:
        # Use a simple optimization approach with warm start from our initial guess
        result = minimize(objective, params, method='SLSQP', bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6})
        
        if result.success:
            # Extract optimized parameters
            optimized_positions = result.x[:2*n].reshape(n, 2)
            optimized_radii = result.x[2*n:]
            
            # Validate the result
            final_positions = optimized_positions
            final_radii = optimized_radii
        else:
            # Fall back to initial configuration if optimization fails
            final_positions = positions
            final_radii = radii
            
    except Exception as e:
        # If optimization fails, return initial configuration
        final_positions = positions
        final_radii = radii
    
    # Construct final result
    circles = np.column_stack([final_positions, final_radii])
    
    return circles


# EVOLVE-BLOCK-END
