# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time
from typing import Tuple, List

# Fixed seed for reproducibility
np.random.seed(42)

def initialize_circles_hexagonal(n: int) -> np.ndarray:
    """Initialize circles in a hexagonal pattern to get a good starting configuration."""
    circles = np.zeros((n, 3))
    
    # Use a hexagonal packing approach
    # For 26 circles, we can try a pattern that approximates hexagonal packing
    # This creates a more structured starting point than random placement
    
    # Try to place in a hexagonal grid pattern
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Calculate spacing based on maximum possible radius
    # For better packing, we'll calculate based on density
    max_radius = 0.1  # Starting estimate
    
    # Place circles in a hexagonal grid pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
                
            # Hexagonal offset
            x_offset = j + (i % 2) * 0.5
            y_offset = i * np.sqrt(3) / 2
            
            # Scale to fit in unit square with reasonable spacing
            spacing = 2 * max_radius
            x = (x_offset * spacing) + max_radius
            y = (y_offset * spacing) + max_radius
            
            # Ensure we're still within bounds
            if x <= 1 - max_radius and y <= 1 - max_radius:
                circles[idx] = [x, y, max_radius]
                idx += 1
    
    # If we didn't fill all positions, fill with random valid positions
    if idx < n:
        # Use a more systematic approach for remaining circles
        for i in range(idx, n):
            # Try to find a valid position using a grid approach
            attempts = 0
            while attempts < 1000:
                # Random position with some bias towards center
                x = np.random.uniform(max_radius, 1 - max_radius)
                y = np.random.uniform(max_radius, 1 - max_radius)
                
                # Try different radii values
                r = np.random.uniform(0.001, min(0.1, 0.5 * min(x, 1-x, y, 1-y)))
                
                # Check if this position is valid with existing circles
                valid = True
                for k in range(i):
                    dx = circles[k, 0] - x
                    dy = circles[k, 1] - y
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < circles[k, 2] + r:
                        valid = False
                        break
                
                if valid:
                    circles[i] = [x, y, r]
                    break
                attempts += 1
            
            # If still no valid position, fall back to uniform distribution
            if attempts >= 1000:
                circles[i] = [
                    np.random.uniform(max_radius, 1 - max_radius),
                    np.random.uniform(max_radius, 1 - max_radius),
                    np.random.uniform(0.001, 0.1)
                ]
    
    return circles

def build_constraint_functions(circles: np.ndarray) -> Tuple[List, List]:
    """
    Build constraint functions for scipy optimization.
    Returns lists of equality and inequality constraints.
    """
    n = len(circles)
    
    # Inequality constraints (non-overlap and containment)
    def get_nonoverlap_constraints():
        constraints = []
        
        # For each pair of circles, add constraint that distance >= sum of radii
        for i in range(n):
            for j in range(i+1, n):
                def constraint_func(params):
                    # Extract positions and radii from flattened parameters
                    positions_and_radii = params.reshape(-1, 3)
                    x1, y1, r1 = positions_and_radii[i]
                    x2, y2, r2 = positions_and_radii[j]
                    
                    # Distance between centers
                    dx = x1 - x2
                    dy = y1 - y2
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Non-overlap constraint: distance >= r1 + r2
                    # We want to maximize sum of radii, so we want to avoid overlap
                    # So we add constraint: r1 + r2 - distance <= 0
                    return r1 + r2 - distance
                
                constraints.append({'type': 'ineq', 'fun': constraint_func})
        
        return constraints
    
    # Containment constraints: each circle must be fully contained
    def get_containment_constraints():
        constraints = []
        
        for i in range(n):
            def constraint_func(params):
                positions_and_radii = params.reshape(-1, 3)
                x, y, r = positions_and_radii[i]
                
                # Each constraint returns positive value when satisfied
                # r <= x means x - r >= 0
                # r <= y means y - r >= 0
                # r <= 1-x means 1-x-r >= 0
                # r <= 1-y means 1-y-r >= 0
                return np.array([
                    x - r,      # x >= r
                    y - r,      # y >= r
                    1 - x - r,  # 1-x >= r
                    1 - y - r   # 1-y >= r
                ])
            
            constraints.append({'type': 'ineq', 'fun': constraint_func})
        
        return constraints
    
    # Combine all constraints
    all_constraints = []
    all_constraints.extend(get_nonoverlap_constraints())
    all_constraints.extend(get_containment_constraints())
    
    return [], all_constraints

def evaluate_constraints(circles: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate constraint violations for debugging.
    Returns tuple of (max_overlap_violation, min_containment_violation)
    """
    n = len(circles)
    max_overlap = 0.0
    min_containment = 1.0
    
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        containment_violations = [r - x, r - y, r - (1 - x), r - (1 - y)]
        min_containment = min(min_containment, min(containment_violations))
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dx = x1 - x2
            dy = y1 - y2
            distance = np.sqrt(dx*dx + dy*dy)
            overlap = distance - (r1 + r2)
            max_overlap = max(max_overlap, overlap)
    
    return max_overlap, min_containment

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses Sequential Quadratic Programming (SQP) with efficient constraint handling.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Initialize with hexagonal pattern
    circles = initialize_circles_hexagonal(26)
    
    print(f"Initial configuration sum of radii: {np.sum(circles[:, 2]):.6f}")
    
    # Flatten the circles array for optimization
    initial_params = circles.flatten()
    
    # Define objective function (negative because we want to maximize)
    def objective(params):
        circles_flat = params.reshape(-1, 3)
        return -np.sum(circles_flat[:, 2])  # Negative because minimize
    
    # Define constraint functions
    def non_overlap_constraint(params):
        circles_flat = params.reshape(-1, 3)
        n = len(circles_flat)
        constraints = []
        
        # For each pair of circles, add constraint that distance >= sum of radii
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_flat[i]
                x2, y2, r2 = circles_flat[j]
                dx = x1 - x2
                dy = y1 - y2
                distance = np.sqrt(dx*dx + dy*dy)
                # Non-overlap constraint: distance >= r1 + r2
                # This should be positive when satisfied: distance - (r1 + r2) >= 0
                constraints.append(distance - (r1 + r2))
        
        return np.array(constraints)
    
    def containment_constraint(params):
        circles_flat = params.reshape(-1, 3)
        constraints = []
        
        # For each circle, add four containment constraints
        for i in range(len(circles_flat)):
            x, y, r = circles_flat[i]
            # x >= r, y >= r, 1-x >= r, 1-y >= r
            constraints.extend([x - r, y - r, 1 - x - r, 1 - y - r])
        
        return np.array(constraints)
    
    # Create constraint dictionaries
    constraints = [
        {'type': 'ineq', 'fun': non_overlap_constraint},
        {'type': 'ineq', 'fun': containment_constraint}
    ]
    
    # Bounds: each parameter has bounds
    bounds = []
    for i in range(26):
        # For x coordinate: r <= x <= 1-r
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Optimize using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6},
            callback=lambda x: print(f"Current objective: {-objective(x):.6f}")
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            print(f"Optimization successful! Final sum of radii: {-result.fun:.6f}")
        else:
            print(f"Optimization failed: {result.message}")
            optimized_circles = circles
            
    except Exception as e:
        print(f"Optimization error: {e}")
        optimized_circles = circles
    
    # Validate final solution
    final_circles = optimized_circles.copy()
    
    # Ensure all circles are valid (containment constraints)
    for i in range(len(final_circles)):
        x, y, r = final_circles[i]
        # Ensure containment
        r = min(r, x, y, 1-x, 1-y)
        final_circles[i] = [x, y, r]
    
    # Final constraint check
    max_overlap, min_containment = evaluate_constraints(final_circles)
    print(f"Final constraint violations - Max overlap violation: {max_overlap:.6f}, Min containment violation: {min_containment:.6f}")
    
    # Ensure we don't have negative values
    final_circles[:, 2] = np.maximum(final_circles[:, 2], 0.001)
    
    print(f"Final solution found with sum of radii: {np.sum(final_circles[:, 2]):.6f}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
    
    return final_circles


# EVOLVE-BLOCK-END
