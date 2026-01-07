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
    
    # Phase 1: Generate initial configuration using hexagonal packing pattern
    def generate_initial_config():
        # Create a hexagonal grid pattern that fits within the unit square
        # We'll place circles in a grid-like pattern with some randomness
        circles = []
        
        # Try different arrangements - start with a grid-based approach
        rows = int(math.sqrt(n)) + 1
        cols = int(math.ceil(n / rows))
        
        # Adjust to fit exactly 32 circles
        while rows * cols < n:
            rows += 1
            
        # Calculate spacing
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Place circles in a grid with small perturbations
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Add small random perturbation to create better distribution
                x = (j + 0.5 + np.random.uniform(-0.1, 0.1)) * spacing_x
                y = (i + 0.5 + np.random.uniform(-0.1, 0.1)) * spacing_y
                
                # Ensure we're still within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                circles.append([x, y, 0.0])
                count += 1
            if count >= n:
                break
        
        # Set initial radii to be small but feasible
        for i in range(len(circles)):
            circles[i][2] = min(0.05, 
                               min(circles[i][0], 1 - circles[i][0]),
                               min(circles[i][1], 1 - circles[i][1]))
        
        return np.array(circles)
    
    # Phase 2: Optimization using scipy minimize
    def objective(radii_and_positions):
        # Extract positions and radii from flattened array
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Return negative sum of radii (since we want to maximize)
        return -np.sum(radii)
    
    def constraint_containment(radii_and_positions):
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Check containment constraints
        result = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # r <= x <= 1-r and r <= y <= 1-r
            result.extend([
                x - r,           # x - r >= 0
                1 - x - r,       # 1 - x - r >= 0
                y - r,           # y - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        return np.array(result)
    
    def constraint_nonoverlap(radii_and_positions):
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Check non-overlap constraints
        result = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1 = radii[i]
                r2 = radii[j]
                
                # Distance between centers >= sum of radii
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # We want: dist_sq >= min_dist_sq
                # So: dist_sq - min_dist_sq >= 0
                result.append(dist_sq - min_dist_sq)
        return np.array(result)
    
    # Generate initial configuration
    initial_circles = generate_initial_config()
    
    # Flatten for optimization
    initial_flat = np.concatenate([
        initial_circles[:, :2].flatten(),  # positions
        initial_circles[:, 2]              # radii
    ])
    
    # Create bounds for variables (positions and radii)
    bounds = []
    # Position bounds: [0,1] for both x and y
    for _ in range(2*n):
        bounds.extend([(0, 1)])
    # Radius bounds: [0, 0.5] (maximum possible for any single circle)
    for _ in range(n):
        bounds.extend([(0, 0.5)])
    
    # Create constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
    ]
    
    # Optimize
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            final_positions = result.x[:2*n].reshape(-1, 2)
            final_radii = result.x[2*n:]
            
            # Create final result
            circles = np.column_stack([final_positions, final_radii])
            return circles
    except Exception as e:
        pass
    
    # Fallback to initial configuration if optimization fails
    return initial_circles


# EVOLVE-BLOCK-END
