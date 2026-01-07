# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a sophisticated initialization with a multi-stage optimization approach.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 32
    circles = np.zeros((n, 3))
    
    # Step 1: Multi-stage initialization for better starting configuration
    # Stage 1: Place circles in a structured pattern
    points = []
    radii = []
    
    # Place circles in corners with large initial radii
    corner_positions = [
        (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),
        (0.05, 0.05), (0.05, 0.95), (0.95, 0.05), (0.95, 0.95)
    ]
    
    # Start with corner placements
    for i, (x, y) in enumerate(corner_positions[:min(8, n)]):
        points.append([x, y])
        # Initial large radius near corners
        max_radius = min(0.1, x, 1-x, y, 1-y)
        radii.append(max_radius * 0.9)
    
    # Stage 2: Fill with a more sophisticated grid
    remaining = n - len(points)
    if remaining > 0:
        # Use a more efficient packing pattern - a combination of grid and spiral
        # Create a rectangular grid pattern
        rows = max(2, int(np.ceil(np.sqrt(remaining * 1.2))))
        cols = max(2, int(np.ceil(remaining / rows)))
        
        # Adjust spacing to leave room for better packing
        spacing_x = 0.9 / cols if cols > 0 else 0.5
        spacing_y = 0.9 / rows if rows > 0 else 0.5
        
        # Generate grid points
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                y = 0.05 + (i + 0.5) * spacing_y
                
                # Apply hexagonal offset for odd rows
                if i % 2 == 1:
                    x += spacing_x / 2
                    
                # Ensure within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                points.append([x, y])
                # Set initial radius based on distance to nearest edge
                min_dist = min(x, 1-x, y, 1-y)
                radii.append(min(0.08, min_dist * 0.6))
    
    # Pad if we don't have enough points
    while len(points) < n:
        points.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
        radii.append(0.05)
    
    points = points[:n]
    radii = radii[:n]
    
    # Step 2: First optimization pass with relaxed constraints to improve initial configuration
    # Create initial guess
    x0 = np.array(points).flatten()
    x0 = np.concatenate([x0, radii])
    
    # Step 3: Define constraints and objective
    def objective(vars):
        # Extract positions and radii
        positions = vars[:2*n].reshape(-1, 2)
        radii = vars[2*n:]
        
        # Minimize negative sum of radii (maximize sum)
        return -np.sum(radii)
    
    def constraint_positions(vars):
        # Ensure all circles are within unit square
        positions = vars[:2*n].reshape(-1, 2)
        radii = vars[2*n:]
        
        # Check containment constraints
        constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            
            # Circle must be fully inside the unit square
            constraints.append(x - r)  # x - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        return np.array(constraints)
    
    def constraint_overlaps(vars):
        # Ensure no overlaps between circles
        positions = vars[:2*n].reshape(-1, 2)
        radii = vars[2*n:]
        
        constraints = []
        # More efficient pairwise comparison with early termination
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1 = radii[i]
                r2 = radii[j]
                
                # Distance between centers minus sum of radii must be >= 0
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                dist = np.sqrt(dist_sq)
                constraints.append(dist - (r1 + r2))
                
        return np.array(constraints)
    
    # Step 4: Run optimization with multiple strategies
    # Create bounds for variables
    bounds = []
    
    # Position bounds: [0, 1] for both x and y coordinates
    for _ in range(2*n):
        bounds.extend([(0, 1)])
    
    # Radius bounds: [0, 0.5] (reasonable upper bound)
    for _ in range(n):
        bounds.extend([(0, 0.5)])
    
    # Define constraint dictionaries
    pos_constraints = {
        'type': 'ineq',
        'fun': constraint_positions
    }
    
    overlap_constraints = {
        'type': 'ineq', 
        'fun': constraint_overlaps
    }
    
    # Optimization options
    options = {'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
    
    # Try different optimization methods with different strategies
    best_result = None
    best_sum = -np.inf
    
    # Strategy 1: Trust-constr with better tolerances
    try:
        result1 = minimize(
            objective,
            x0.copy(),
            method='trust-constr',
            bounds=bounds,
            constraints=[pos_constraints, overlap_constraints],
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
            tol=1e-8
        )
        
        if result1.success:
            final_radii = result1.x[2*n:]
            current_sum = np.sum(final_radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = result1
                
    except Exception as e:
        pass
    
    # Strategy 2: SLSQP with better initialization
    if best_result is None:
        try:
            result2 = minimize(
                objective,
                x0.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=[pos_constraints, overlap_constraints],
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8},
                tol=1e-8
            )
            
            if result2.success:
                final_radii = result2.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result2
                    
        except Exception as e:
            pass
    
    # Strategy 3: If both fail, try a simpler approach with fewer iterations but better convergence
    if best_result is None:
        # Try a very simple but effective approach: just optimize radii while keeping positions fixed
        # This is a quick fix to get something reasonable
        try:
            # First, try to increase radii only while keeping positions fixed
            fixed_positions = x0[:2*n].reshape(-1, 2)
            fixed_radii = x0[2*n:].copy()
            
            # Simple iterative approach to increase radii
            for _ in range(50):
                # Calculate current overlaps
                overlaps = False
                for i in range(n):
                    for j in range(i+1, n):
                        x1, y1 = fixed_positions[i]
                        x2, y2 = fixed_positions[j]
                        r1 = fixed_radii[i]
                        r2 = fixed_radii[j]
                        
                        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                        dist = np.sqrt(dist_sq)
                        
                        if dist < (r1 + r2):
                            overlaps = True
                            # Reduce both radii slightly
                            reduction_factor = 0.99
                            fixed_radii[i] *= reduction_factor
                            fixed_radii[j] *= reduction_factor
                            break
                    if overlaps:
                        break
                
                if not overlaps:
                    # Try to increase radii a bit
                    for i in range(n):
                        # Increase radius if possible without overlap
                        min_dist_to_edge = min(fixed_positions[i][0], 1-fixed_positions[i][0], 
                                             fixed_positions[i][1], 1-fixed_positions[i][1])
                        max_possible_radius = min(0.5, min_dist_to_edge)
                        
                        # Allow some increase if there's no overlap
                        if fixed_radii[i] < max_possible_radius:
                            fixed_radii[i] = min(max_possible_radius, fixed_radii[i] * 1.05)
            
            # Final check for feasibility
            valid_radii = fixed_radii.copy()
            for _ in range(10):
                overlaps = False
                for i in range(n):
                    for j in range(i+1, n):
                        x1, y1 = fixed_positions[i]
                        x2, y2 = fixed_positions[j]
                        r1 = valid_radii[i]
                        r2 = valid_radii[j]
                        
                        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                        dist = np.sqrt(dist_sq)
                        
                        if dist < (r1 + r2):
                            overlaps = True
                            # Reduce both radii
                            reduction_factor = 0.95
                            valid_radii[i] *= reduction_factor
                            valid_radii[j] *= reduction_factor
                            break
                    if overlaps:
                        break
                        
            # Create result using this improved configuration
            final_positions = fixed_positions
            final_radii = valid_radii
            
            # Build a fake result object for consistency
            class FakeResult:
                def __init__(self, positions, radii):
                    self.success = True
                    self.x = np.concatenate([positions.flatten(), radii])
            
            best_result = FakeResult(final_positions, final_radii)
            best_sum = np.sum(final_radii)
            
        except Exception as e:
            pass
    
    # If no optimization succeeded, use the best initialization
    if best_result is None or not best_result.success:
        print("Optimization failed, using improved initialization")
        # Use the initial configuration directly
        for i in range(n):
            circles[i] = [points[i][0], points[i][1], radii[i]]
        return circles
    
    # Extract final solution from best result
    final_positions = best_result.x[:2*n].reshape(-1, 2)
    final_radii = best_result.x[2*n:]
    
    # Construct circles array
    for i in range(n):
        circles[i] = [final_positions[i][0], final_positions[i][1], final_radii[i]]
    
    return circles


# EVOLVE-BLOCK-END
