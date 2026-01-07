# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from sklearn.cluster import KMeans
import warnings
from scipy.spatial import cKDTree

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: initial placement with hexagonal packing heuristic, followed by 
    advanced constrained optimization with proper constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Better initial placement using hexagonal packing heuristic
    circles = initialize_hexagonal_placement(n)
    
    # Phase 2: Refine using advanced optimization with proper constraint handling
    circles = optimize_circles_advanced(circles)
    
    return circles

def initialize_hexagonal_placement(n: int) -> np.ndarray:
    """Initialize circle positions using a hexagonal packing approach for better density"""
    # For 32 circles, try to create a more efficient hexagonal-like arrangement
    # We'll use a combination of hexagonal packing and refinement
    
    # Estimate how many rows/columns we need
    # For hexagonal packing, area efficiency is ~0.9069
    target_area = n * 1.0  # We want to fill the unit square
    estimated_rows_cols = int(np.ceil(np.sqrt(n)))
    
    # Create a hexagonal pattern
    circles = []
    
    # Hexagonal packing parameters
    sqrt3 = np.sqrt(3)
    row_spacing = 2 * 0.1  # Initial spacing
    col_spacing = row_spacing * sqrt3 / 2
    
    # Place circles in a hexagonal pattern
    row_count = 0
    col_count = 0
    
    # Start from center and expand outward
    center_x, center_y = 0.5, 0.5
    
    # Generate points in hexagonal pattern around center
    max_radius = 0.15  # Reasonable starting radius
    
    # Try to place circles in a hexagonal pattern first
    placed_count = 0
    
    # Create a grid of potential positions in hexagonal pattern
    positions = []
    max_dist = 0.8  # Maximum distance from center
    
    # Generate hexagonal grid points
    for i in range(-5, 6):
        for j in range(-5, 6):
            if placed_count >= n:
                break
                
            # Hexagonal coordinates
            x = i * col_spacing + (j % 2) * col_spacing / 2
            y = j * row_spacing
            
            # Center at (0.5, 0.5) and scale appropriately
            x = 0.5 + x * 0.8
            y = 0.5 + y * 0.8
            
            # Check bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Adjust radius based on proximity to edges
                min_edge_dist = min(x, 1-x, y, 1-y)
                radius = min(max_radius, min_edge_dist * 0.8)
                if radius > 0.001:
                    positions.append([x, y, radius])
                    placed_count += 1
        
        if placed_count >= n:
            break
    
    # Fill remaining positions with random placements near good locations
    while len(positions) < n:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        radius = np.random.uniform(0.01, 0.15)
        positions.append([x, y, radius])
    
    return np.array(positions[:n])

def optimize_circles_advanced(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii with advanced constraint handling"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_flat = []
    for i in range(n):
        initial_flat.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    def objective(x_flat):
        # Extract positions and radii
        total_radius = 0
        for i in range(n):
            total_radius += x_flat[3*i + 2]
        return -total_radius  # Negative because we want to maximize
    
    def constraint_func(x_flat):
        # More robust constraint handling with proper bounds checking
        constraints = []
        
        # Circle containment constraints (more precise)
        for i in range(n):
            x, y, r = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
            
            # Radius should be positive
            constraints.append(r)
            
            # Boundary constraints with safety margins
            constraints.append(1 - r - x)  # Right boundary
            constraints.append(1 - r - y)  # Top boundary
            constraints.append(x - r)      # Left boundary
            constraints.append(y - r)      # Bottom boundary
        
        # Non-overlap constraints with early termination for performance
        # Use spatial indexing to reduce comparisons
        points = [(x_flat[3*i], x_flat[3*i+1]) for i in range(n)]
        tree = cKDTree(points)
        
        # Find nearby pairs using spatial indexing
        pairs = tree.query_pairs(0.01, output_type='ndarray')  # Only check close pairs
        
        for i, j in pairs:
            if i < j:  # Avoid duplicate checks
                x1, y1, r1 = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
                x2, y2, r2 = x_flat[3*j], x_flat[3*j+1], x_flat[3*j+2]
                
                # Distance constraint: d >= r1 + r2
                dx = x1 - x2
                dy = y1 - y2
                distance_sq = dx*dx + dy*dy
                # To avoid sqrt computation, check if distance^2 >= (r1+r2)^2
                radius_sum = r1 + r2
                constraints.append(distance_sq - radius_sum * radius_sum)
        
        # Also add direct comparisons for all pairs to ensure no missed constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
                x2, y2, r2 = x_flat[3*j], x_flat[3*j+1], x_flat[3*j+2]
                
                # Distance constraint: d >= r1 + r2
                dx = x1 - x2
                dy = y1 - y2
                distance_sq = dx*dx + dy*dy
                # To avoid sqrt computation, check if distance^2 >= (r1+r2)^2
                radius_sum = r1 + r2
                constraints.append(distance_sq - radius_sum * radius_sum)
        
        return np.array(constraints)
    
    # Set up bounds: x, y in [r, 1-r], r in [0.001, 0.45] (reasonable upper bound)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.45)])  # x, y, r bounds
    
    # Try multiple optimization approaches with better error handling
    best_result = None
    best_value = float('-inf')
    
    # First attempt: Trust Region Constrained with tighter tolerances
    try:
        result = minimize(
            objective,
            initial_flat,
            method='trust-constr',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'gtol': 1e-6, 'xtol': 1e-6, 'barrier_tol': 1e-8}
        )
        
        if result.success:
            # Check if this solution is better
            current_total = -objective(result.x)  # Convert back to maximization
            if current_total > best_value:
                best_value = current_total
                best_result = result
    except Exception as e:
        warnings.warn(f"Trust-Constr failed: {e}")
    
    # Second attempt: SLSQP with reduced tolerance if trust-constr fails
    if best_result is None:
        try:
            result = minimize(
                objective,
                initial_flat,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 500, 'ftol': 1e-5, 'eps': 1e-5}
            )
            
            if result.success:
                current_total = -objective(result.x)
                if current_total > best_value:
                    best_value = current_total
                    best_result = result
        except Exception as e:
            warnings.warn(f"SLSQP failed: {e}")
    
    # Third attempt: Multiple restarts with different strategies
    if best_result is None:
        # Try with a different approach - just optimize with bounds and no constraints for initial phase
        try:
            result = minimize(
                objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-5}
            )
            
            if result.success:
                current_total = -objective(result.x)
                if current_total > best_value:
                    best_value = current_total
                    best_result = result
        except Exception as e:
            warnings.warn(f"L-BFGS-B failed: {e}")
    
    # If we have a valid result, return optimized circles
    if best_result is not None and best_result.success:
        optimized = best_result.x
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [optimized[3*i], optimized[3*i+1], optimized[3*i+2]]
        return circles
    
    # If optimization fails, return initial configuration but with some refinement
    # Apply a simple local optimization to improve the initial configuration
    refined = initial_circles.copy()
    # Simple local improvement: adjust positions to avoid overlaps
    for _ in range(10):  # Run a few iterations
        for i in range(n):
            # Try small adjustments to position
            best_pos = refined[i][:2].copy()
            best_radius = refined[i][2]
            best_score = -objective(np.concatenate([best_pos, [best_radius]]))
            
            # Try small perturbations
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    test_x = max(0.001, min(0.999, refined[i][0] + dx))
                    test_y = max(0.001, min(0.999, refined[i][1] + dy))
                    
                    # Test constraint violations
                    valid = True
                    for j in range(n):
                        if i != j:
                            dist_sq = (test_x - refined[j][0])**2 + (test_y - refined[j][1])**2
                            min_dist_sq = (refined[i][2] + refined[j][2])**2
                            if dist_sq < min_dist_sq:
                                valid = False
                                break
                    
                    if valid:
                        test_radius = refined[i][2]  # Keep same radius for simplicity
                        score = -objective(np.concatenate([[test_x, test_y], [test_radius]]))
                        if score > best_score:
                            best_score = score
                            best_pos = [test_x, test_y]
            
            refined[i][:2] = best_pos
    
    return refined


# EVOLVE-BLOCK-END
