# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal grid initialization, local optimization, and 
    multi-start mathematical programming for final refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Multi-start approach to improve chances of finding better solution
    best_circles = None
    best_sum = 0
    
    # Try multiple random starts to avoid local optima
    for start in range(5):
        # Phase 1: Sophisticated initialization using hexagonal grid with refinement
        def initialize_hexagonal_placement(n):
            """Initialize circle positions using a refined hexagonal grid pattern"""
            # Create a hexagonal grid pattern with better spacing and distribution
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            
            # Ensure we have enough space
            while rows * cols < n:
                rows += 1
            
            # Generate positions in a hexagonal pattern
            positions = []
            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= n:
                        break
                    # Hexagonal offset for better packing
                    x_offset = j + 0.5 * (i % 2)
                    y_offset = i * np.sqrt(3) / 2
                    
                    # Add different amount of randomness for each start to improve diversity
                    random_factor = 0.1 + 0.05 * start  # Increasing randomness with start number
                    x = x_offset + (random.random() - 0.5) * random_factor
                    y = y_offset + (random.random() - 0.5) * random_factor
                    
                    positions.append([x, y])
            
            # Normalize positions to fit in unit square
            positions = np.array(positions[:n])
            
            if len(positions) > 0:
                # Normalize coordinates to fit within [0.1, 0.9] range to allow room for radii
                max_x = np.max(positions[:, 0]) if np.max(positions[:, 0]) > 0 else 1
                max_y = np.max(positions[:, 1]) if np.max(positions[:, 1]) > 0 else 1
                
                if max_x > 0:
                    positions[:, 0] = 0.8 * positions[:, 0] / max_x + 0.1
                if max_y > 0:
                    positions[:, 1] = 0.8 * positions[:, 1] / max_y + 0.1
            
            # Assign initial radii based on available space
            radii = np.full(n, 0.02)
            
            # Adjust radii to respect bounds and avoid overlaps
            for i in range(n):
                # Check minimum distance to edges
                min_edge_dist = min(positions[i, 0], positions[i, 1], 1 - positions[i, 0], 1 - positions[i, 1])
                radii[i] = min(radii[i], min_edge_dist * 0.8)
            
            # Combine into circle array
            circles = np.column_stack([positions, radii])
            
            return circles

        # Phase 2: Local optimization to expand radii while maintaining constraints
        def local_radius_expansion(circles):
            """Perform local optimization to expand radii while maintaining constraints"""
            n = len(circles)
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            improved = True
            max_iterations = 100
            
            for iteration in range(max_iterations):
                if not improved:
                    break
                improved = False
                new_radii = radii.copy()
                
                # Try to expand each radius
                for i in range(n):
                    # Start with a small expansion
                    expansion_factor = 1.05
                    test_radius = new_radii[i] * expansion_factor
                    
                    # Check if this radius is valid
                    valid = True
                    
                    # Check containment
                    if (test_radius > positions[i, 0] or 
                        test_radius > positions[i, 1] or
                        test_radius > 1 - positions[i, 0] or
                        test_radius > 1 - positions[i, 1]):
                        valid = False
                    
                    # Check overlaps with all other circles
                    if valid:
                        for j in range(n):
                            if i != j:
                                dx = positions[i, 0] - positions[j, 0]
                                dy = positions[i, 1] - positions[j, 1]
                                dist_sq = dx*dx + dy*dy
                                min_dist_sq = (test_radius + new_radii[j])**2
                                
                                if dist_sq < min_dist_sq:
                                    valid = False
                                    break
                    
                    # If valid, accept the expansion
                    if valid and test_radius > new_radii[i]:
                        new_radii[i] = test_radius
                        improved = True
                
                if improved:
                    radii = new_radii
            
            # Update circles with new radii
            circles[:, 2] = radii
            return circles

        # Phase 3: Mathematical optimization for final refinement
        def optimize_with_scipy(circles):
            """Use scipy optimization for final refinement"""
            n = len(circles)
            
            # Flatten initial data for optimization
            x0 = np.zeros(3*n)
            for i in range(n):
                x0[3*i] = circles[i, 0]  # x coordinate
                x0[3*i + 1] = circles[i, 1]  # y coordinate
                x0[3*i + 2] = circles[i, 2]  # radius
            
            # Define objective function to maximize sum of radii
            def objective(x):
                # x contains [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
                total_radius = 0
                for i in range(n):
                    total_radius += x[3*i + 2]  # Extract radius for circle i
                return -total_radius  # Negative because we want to maximize
            
            # Define constraint functions
            def containment_constraints(x):
                # Ensure all circles are within the unit square
                constraints = []
                for i in range(n):
                    xi = x[3*i]
                    yi = x[3*i + 1]
                    ri = x[3*i + 2]
                    
                    # Circle must be within bounds (with margin for numerical stability)
                    constraints.append(xi - ri - 1e-6)  # x - r >= 0
                    constraints.append(yi - ri - 1e-6)  # y - r >= 0
                    constraints.append(1 - xi - ri - 1e-6)  # 1 - x - r >= 0
                    constraints.append(1 - yi - ri - 1e-6)  # 1 - y - r >= 0
                    
                return np.array(constraints)
            
            def overlap_constraints(x):
                # Ensure no overlaps between circles (with tolerance for numerical stability)
                constraints = []
                for i in range(n):
                    for j in range(i+1, n):
                        xi = x[3*i]
                        yi = x[3*i + 1]
                        ri = x[3*i + 2]
                        xj = x[3*j]
                        yj = x[3*j + 1]
                        rj = x[3*j + 2]
                        
                        # Distance between centers minus sum of radii must be >= 0
                        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                        # Use small tolerance to prevent tight overlaps
                        constraints.append(dist - ri - rj - 1e-6)  # d - r1 - r2 >= 0
                        
                return np.array(constraints)
            
            # Set bounds for variables (x, y, r) with tighter constraints
            bounds = []
            for i in range(n):
                bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.4)])  # x, y in [1e-6, 1-1e-6], r in [1e-6, 0.4]
            
            # Set up constraints
            cons = []
            
            # Add containment constraints
            cons.append({'type': 'ineq', 'fun': lambda x: containment_constraints(x)})
            
            # Add overlap constraints
            cons.append({'type': 'ineq', 'fun': lambda x: overlap_constraints(x)})
            
            # Optimize using SLSQP with better parameters
            try:
                result = minimize(
                    objective, 
                    x0, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=cons, 
                    options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
                )
                
                if result.success:
                    # Convert back to circle array
                    optimized_circles = np.zeros((n, 3))
                    for i in range(n):
                        optimized_circles[i, 0] = result.x[3*i]      # x
                        optimized_circles[i, 1] = result.x[3*i + 1]  # y
                        optimized_circles[i, 2] = result.x[3*i + 2]  # r
                    return optimized_circles
            except Exception as e:
                # If optimization fails, return original circles
                pass
            
            return circles

        # Execute the full pipeline for this start
        circles = initialize_hexagonal_placement(n)
        circles = local_radius_expansion(circles)
        circles = optimize_with_scipy(circles)
        
        # Track best solution found so far
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
    
    return best_circles


# EVOLVE-BLOCK-END
