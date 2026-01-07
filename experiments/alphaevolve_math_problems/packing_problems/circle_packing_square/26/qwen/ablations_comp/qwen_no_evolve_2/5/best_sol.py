# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and gradient-based optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize positions and radii
    # Start with a hexagonal-like pattern for good initial distribution
    circles = np.zeros((n, 3))
    
    # Create a grid-based initial configuration
    rows = int(math.sqrt(n)) + 1
    cols = int(math.ceil(n / rows))
    
    # Distribute points in a grid pattern with some randomness
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Position in grid with padding
        x = 0.1 + (col + 0.5) * (0.8 / cols)
        y = 0.1 + (row + 0.5) * (0.8 / rows)
        
        # Initial radius - start small and let optimization increase
        r = 0.05
        
        circles[i] = [x, y, r]
    
    # Define objective function
    def objective(params):
        # Reshape parameters back into circles array
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Calculate negative sum of radii (since we want to maximize)
        return -np.sum(radii)
    
    # Define constraints
    def containment_constraint(params):
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Each circle must be fully contained in the unit square
        constraints = []
        
        # Lower bounds (x >= r, y >= r)
        constraints.extend(positions[:, 0] - radii)  # x - r >= 0
        constraints.extend(positions[:, 1] - radii)  # y - r >= 0
        
        # Upper bounds (x <= 1-r, y <= 1-r)
        constraints.extend(1 - radii - positions[:, 0])  # 1 - r - x >= 0
        constraints.extend(1 - radii - positions[:, 1])  # 1 - r - y >= 0
        
        return np.array(constraints)
    
    def non_overlap_constraint(params):
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Calculate distances between all pairs of circles
        distances = cdist(positions, positions)
        constraints = []
        
        # For each pair of circles, ensure they don't overlap
        for i in range(n):
            for j in range(i+1, n):
                # Distance between centers minus sum of radii should be >= 0
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                constraints.append(dist - min_dist)
        
        return np.array(constraints)
    
    # Create initial parameter vector
    initial_params = np.concatenate([
        circles[:, :2].flatten(),  # positions
        circles[:, 2]              # radii
    ])
    
    # Set up constraints
    # Containment constraints (all >= 0)
    containment_lb = np.zeros(4*n)  # 2*n for x, 2*n for y bounds
    containment_ub = np.full(4*n, np.inf)
    
    # Non-overlap constraints (all >= 0)
    non_overlap_lb = np.zeros(n*(n-1)//2)
    non_overlap_ub = np.full(n*(n-1)//2, np.inf)
    
    # Combine constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda p: containment_constraint(p)},
        {'type': 'ineq', 'fun': lambda p: non_overlap_constraint(p)}
    ]
    
    # Bounds for parameters (positions: [0,1], radii: [0,0.5])
    bounds = []
    for i in range(2*n):  # positions
        bounds.append((0, 1))
    for i in range(n):   # radii
        bounds.append((0, 0.5))
    
    # Optimize using SLSQP method
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            final_positions = result.x[:2*n].reshape(-1, 2)
            final_radii = result.x[2*n:]
            
            # Update circles with optimized values
            for i in range(n):
                circles[i] = [final_positions[i, 0], final_positions[i, 1], final_radii[i]]
        else:
            # Fallback to initial configuration if optimization fails
            pass
            
    except Exception as e:
        # If optimization fails, return initial configuration
        pass
    
    # Final validation to ensure constraints are met
    # Check containment
    valid = True
    for i in range(n):
        x, y, r = circles[i]
        if not (r <= x <= 1-r and r <= y <= 1-r):
            valid = False
            break
    
    # Check non-overlap
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < r1 + r2:
                    valid = False
                    break
            if not valid:
                break
    
    # If constraints violated, use a more conservative approach
    if not valid:
        # Reinitialize with a better placement strategy
        circles = np.zeros((n, 3))
        # Place circles in a more conservative pattern
        for i in range(n):
            row = i // 5
            col = i % 5
            x = 0.1 + col * 0.18
            y = 0.1 + row * 0.18
            r = 0.05
            circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
