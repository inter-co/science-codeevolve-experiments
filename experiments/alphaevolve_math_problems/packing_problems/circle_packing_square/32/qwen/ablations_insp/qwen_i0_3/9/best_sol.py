# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Initialize using hexagonal packing pattern
    def initialize_hexagonal():
        # Create a hexagonal lattice pattern
        rows = 6
        cols = 6
        circles = []
        
        # Hexagonal spacing
        spacing_x = 0.15
        spacing_y = 0.15 * np.sqrt(3)/2
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Adjust for odd rows
                if i % 2 == 1:
                    x += spacing_x / 2
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, 0.05])
        return np.array(circles)
    
    # Phase 2: Optimization using scipy minimize
    def objective(radii_and_positions):
        # Extract positions and radii
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Calculate negative sum of radii (we want to maximize)
        return -np.sum(radii)
    
    def constraint_func(radii_and_positions):
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Position constraints (circle must fit in unit square)
        pos_constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Circle must stay inside unit square
            pos_constraints.extend([
                x - r,  # x - r >= 0
                1 - x - r,  # 1 - x - r >= 0
                y - r,  # y - r >= 0
                1 - y - r  # 1 - y - r >= 0
            ])
        
        # Non-overlap constraints
        dist_matrix = cdist(positions, positions)
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = dist_matrix[i, j]
                r_i = radii[i]
                r_j = radii[j]
                # Distance between centers must be >= sum of radii
                overlap_constraints.append(dist - r_i - r_j)
        
        return np.array(pos_constraints + overlap_constraints)
    
    # Initialize
    circles = initialize_hexagonal()
    
    # Fill remaining circles with small radii
    while len(circles) < n:
        circles = np.vstack([circles, [0.5, 0.5, 0.01]])
    
    # Flatten initial configuration
    initial_guess = np.concatenate([
        circles[:, :2].flatten(),  # positions
        circles[:, 2]              # radii
    ])
    
    # Set up constraints
    bounds = []
    # Position bounds: [0+r, 1-r] for both x and y
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999)])  # x, y bounds
    # Radius bounds: [0.001, 0.499] 
    for i in range(n):
        bounds.extend([(0.001, 0.499)])
    
    # Create constraint dictionary
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Optimize
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        # Extract final solution
        final_positions = result.x[:2*n].reshape(-1, 2)
        final_radii = result.x[2*n:]
        
        # Create final circles array
        circles = np.column_stack([final_positions, final_radii])
        
        # Ensure we have exactly 32 circles
        if len(circles) < n:
            # Add more circles with very small radii if needed
            additional = n - len(circles)
            for _ in range(additional):
                circles = np.vstack([circles, [0.5, 0.5, 0.001]])
                
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        print(f"Optimization failed: {e}")
    
    # Ensure we have exactly 32 circles
    if len(circles) < n:
        # Fill with default values
        while len(circles) < n:
            circles = np.vstack([circles, [0.5, 0.5, 0.01]])
    
    return circles[:n]


# EVOLVE-BLOCK-END
