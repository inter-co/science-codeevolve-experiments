# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize using hexagonal lattice arrangement
    def initialize_hexagonal():
        # Try to place circles in a hexagonal pattern
        circles = []
        
        # Parameters for hexagonal packing
        # We'll try to fit circles in a roughly 5x5 grid (since 5*5=25)
        rows = 5
        cols = 5
        
        # Calculate spacing based on circle diameter
        # Start with a reasonable initial radius
        initial_radius = 0.1
        spacing_x = 2 * initial_radius
        spacing_y = initial_radius * math.sqrt(3)
        
        # Adjust spacing to fit in unit square
        max_width = 1.0
        max_height = 1.0
        
        # Center the pattern in the unit square
        offset_x = (max_width - (cols - 1) * spacing_x) / 2
        offset_y = (max_height - (rows - 1) * spacing_y) / 2
        
        # Generate hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                # Offset every other row
                x_offset = (i % 2) * spacing_x / 2
                x = offset_x + j * spacing_x + x_offset
                y = offset_y + i * spacing_y
                
                # Add some randomness to avoid perfect patterns
                x += np.random.uniform(-spacing_x/10, spacing_x/10)
                y += np.random.uniform(-spacing_y/10, spacing_y/10)
                
                # Ensure we're within bounds
                x = max(initial_radius, min(1 - initial_radius, x))
                y = max(initial_radius, min(1 - initial_radius, y))
                
                circles.append([x, y, initial_radius])
        
        # Trim to exactly 26 circles
        if len(circles) > n:
            circles = circles[:n]
        elif len(circles) < n:
            # Fill with additional circles at random positions
            for i in range(n - len(circles)):
                x = np.random.uniform(initial_radius, 1 - initial_radius)
                y = np.random.uniform(initial_radius, 1 - initial_radius)
                circles.append([x, y, initial_radius])
        
        return np.array(circles)
    
    # Initialize circles
    circles = initialize_hexagonal()
    
    # Optimization objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        # Reshape flat array back to circles
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Constraint functions
    def constraint_containment(circles_flat):
        """Ensure all circles are contained in the unit square"""
        circles = circles_flat.reshape(-1, 3)
        # For each circle, check containment constraints
        constraints = []
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Left boundary: x >= r
            constraints.append(x - r)  # Should be >= 0
            # Right boundary: x <= 1-r  
            constraints.append(1 - x - r)  # Should be >= 0
            # Bottom boundary: y >= r
            constraints.append(y - r)  # Should be >= 0
            # Top boundary: y <= 1-r
            constraints.append(1 - y - r)  # Should be >= 0
        return np.array(constraints)
    
    def constraint_nonoverlap(circles_flat):
        """Ensure no overlaps between circles"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                # Distance constraint: distance >= r1 + r2
                # Using squared distance to avoid sqrt and improve numerical stability
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                # We want dist_sq >= min_dist_sq, so we return dist_sq - min_dist_sq
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)
    
    # Create bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x coordinate bounds: [r, 1-r]
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])
    
    # Set up constraints for scipy optimizer
    constraints = [
        {'type': 'ineq', 'fun': constraint_containment},
        {'type': 'ineq', 'fun': constraint_nonoverlap}
    ]
    
    # Multiple restarts for better optimization
    best_result = None
    best_sum = -float('inf')
    
    # Try 10 different random restarts (increased from 5 for better exploration)
    for restart in range(10):
        # Set seed for reproducibility
        np.random.seed(restart * 1000 + 42)
        
        # Create slightly perturbed initial guess for each restart
        if restart == 0:
            initial_flat = circles.flatten()
        else:
            # Add small random perturbations
            initial_flat = circles.flatten().copy()
            for i in range(0, len(initial_flat), 3):
                # Perturb x, y, r with small random values
                initial_flat[i] += np.random.normal(0, 0.01)  # x
                initial_flat[i+1] += np.random.normal(0, 0.01)  # y
                initial_flat[i+2] += np.random.normal(0, 0.005)  # r
            # Clip to valid ranges
            initial_flat = np.clip(initial_flat, [1e-6]*len(initial_flat), [0.999]*len(initial_flat))
        
        # Run optimization with higher iteration limits and tighter tolerances
        try:
            result = minimize(
                objective,
                initial_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-10}  # Tighter tolerances
            )
            
            if result.success:
                current_sum = -result.fun  # Convert back to positive sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception:
            continue  # Skip this restart if it fails
    
    # Return the best result found
    if best_result is not None and best_result.success:
        final_circles = best_result.x.reshape(-1, 3)
        # Ensure all circles are properly contained
        for i in range(len(final_circles)):
            x, y, r = final_circles[i]
            final_circles[i] = [
                max(r, min(1-r, x)), 
                max(r, min(1-r, y)), 
                max(1e-6, min(0.5, r))
            ]
        return final_circles
    else:
        # If all optimizations fail, return the initial configuration
        return circles


# EVOLVE-BLOCK-END
