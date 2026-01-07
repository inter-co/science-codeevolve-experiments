# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with mathematical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 32
    
    # Step 1: Geometric initialization using hexagonal packing pattern
    def initialize_hexagonal():
        # Try to place circles in a hexagonal lattice pattern
        # For 32 circles, we can arrange in roughly 6x6 grid with some adjustments
        rows = 6
        cols = 6
        positions = []
        
        # Generate hexagonal pattern
        spacing_x = 0.15
        spacing_y = 0.15 * np.sqrt(3)/2
        
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                positions.append([x, y])
        
        # If we don't have enough positions, fill with random ones
        while len(positions) < n:
            positions.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
            
        return np.array(positions[:n])
    
    # Step 2: Mathematical optimization approach
    def objective_and_constraints(x_and_r):
        # x_and_r contains [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
        positions = x_and_r.reshape(-1, 3)[:, :2]
        radii = x_and_r.reshape(-1, 3)[:, 2]
        
        # Objective: negative sum of radii (we minimize)
        obj = -np.sum(radii)
        
        # Constraints: 
        # 1. Boundary constraints (circle within unit square)
        # 2. Non-overlap constraints
        
        constraints = []
        
        # Boundary constraints: radius must be within bounds
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+2] - 0.001})  # r_i >= 0.001
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+2]})     # r_i <= 1
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*i+2]})   # x_i >= r_i
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*i+2]}) # y_i >= r_i
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i] - x[2*i+2]}) # x_i <= 1 - r_i
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[2*i+1] - x[2*i+2]}) # y_i <= 1 - r_i
            
        # Non-overlap constraints
        for i, j in combinations(range(n), 2):
            def overlap_constraint(x, i=i, j=j):
                pos_i = x[2*i:2*i+2]
                pos_j = x[2*j:2*j+2]
                r_i = x[2*i+2]
                r_j = x[2*j+2]
                dist = np.linalg.norm(pos_i - pos_j)
                return dist - (r_i + r_j)
            
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return obj, constraints
    
    # Step 3: Improved initialization with better geometric layout
    initial_positions = initialize_hexagonal()
    
    # Set initial radii based on available space
    initial_radii = np.full(n, 0.05)
    
    # Better initialization: start with larger radii in dense areas
    # Place circles more densely in center, less in corners
    center_positions = initial_positions.copy()
    for i in range(n):
        # Adjust based on proximity to center
        dist_to_center = np.linalg.norm(center_positions[i] - [0.5, 0.5])
        # Start with smaller radii near edges, larger in center
        initial_radii[i] = min(0.15, 0.1 * (1 + 0.5 * np.exp(-dist_to_center * 2)))
    
    # Flatten initial guess
    initial_guess = np.concatenate([center_positions.flatten(), initial_radii])
    
    # Step 4: Optimization using scipy minimize with SLSQP method
    def optimized_objective(x):
        positions = x[:-n].reshape(-1, 2)
        radii = x[-n:]
        
        # Objective: negative sum of radii
        return -np.sum(radii)
    
    def constraint_func(x):
        positions = x[:-n].reshape(-1, 2)
        radii = x[-n:]
        
        # Non-overlap constraints
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            # Circle must be within bounds
            constraints.append(positions[i][0] - radii[i])  # x - r >= 0
            constraints.append(positions[i][1] - radii[i])  # y - r >= 0
            constraints.append(1 - positions[i][0] - radii[i])  # 1 - x - r >= 0
            constraints.append(1 - positions[i][1] - radii[i])  # 1 - y - r >= 0
            constraints.append(radii[i])  # r >= 0
            
        # Non-overlap constraints
        for i, j in combinations(range(n), 2):
            dist = np.linalg.norm(positions[i] - positions[j])
            constraints.append(dist - (radii[i] + radii[j]))
            
        return np.array(constraints)
    
    # Create bounds for variables
    bounds = []
    # Position bounds [0,1] for both coordinates
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999)])  # x, y bounds
    # Radius bounds [0.001, 0.5] (max possible radius is 0.5)
    for i in range(n):
        bounds.extend([(0.001, 0.5)])  # r bounds
    
    # Use a two-stage optimization approach
    # Stage 1: Coarse optimization with fewer iterations
    options1 = {'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
    
    # Try to find a good solution with constraints
    try:
        # First attempt with bounds
        result = minimize(
            optimized_objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options=options1,
            tol=1e-6
        )
        
        if result.success:
            final_positions = result.x[:-n].reshape(-1, 2)
            final_radii = result.x[-n:]
        else:
            # Fallback to the hexagonal initialization if optimization fails
            final_positions = center_positions
            final_radii = initial_radii
            
    except Exception as e:
        # If optimization fails, fall back to geometric approach
        final_positions = center_positions
        final_radii = initial_radii
    
    # Final validation and adjustment
    # Make sure all constraints are satisfied
    valid_positions = final_positions.copy()
    valid_radii = final_radii.copy()
    
    # Ensure no overlaps and boundaries are respected
    max_iterations = 100
    for _ in range(max_iterations):
        # Check overlaps
        overlaps = False
        distances = cdist(valid_positions, valid_positions)
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < valid_radii[i] + valid_radii[j]:
                    overlaps = True
                    # Adjust positions to resolve overlap
                    diff = valid_positions[i] - valid_positions[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-8:
                        correction = (valid_radii[i] + valid_radii[j] - dist) * diff / dist / 2
                        valid_positions[i] += correction
                        valid_positions[j] -= correction
        if not overlaps:
            break
    
    # Ensure boundaries
    for i in range(n):
        # Keep within bounds
        valid_positions[i][0] = np.clip(valid_positions[i][0], valid_radii[i], 1 - valid_radii[i])
        valid_positions[i][1] = np.clip(valid_positions[i][1], valid_radii[i], 1 - valid_radii[i])
    
    # Create final result array
    circles = np.column_stack([valid_positions, valid_radii])
    
    return circles


# EVOLVE-BLOCK-END
