# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining mathematical optimization with local refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 32
    
    def _initialize_circles_hexagonal(n: int) -> np.ndarray:
        """Initialize circles using a hexagonal packing approach for better density"""
        # Hexagonal lattice approach - more efficient packing
        sqrt_n = int(np.ceil(np.sqrt(n)))
        grid_size = sqrt_n
        
        # Create hexagonal grid with offset rows
        x_coords = []
        y_coords = []
        
        # Generate hexagonal pattern
        spacing = 0.9 / grid_size  # Leave some margin
        radius_estimate = spacing * 0.45  # Approximate radius
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Offset odd rows
                x_offset = (i % 2) * spacing * 0.5
                x = 0.05 + j * spacing + x_offset
                y = 0.05 + i * spacing * np.sqrt(3)/2
                
                # Only add if within bounds
                if x <= 0.95 and y <= 0.95:
                    x_coords.append(x)
                    y_coords.append(y)
        
        # If we have too many points, sample them
        if len(x_coords) > n:
            indices = np.random.choice(len(x_coords), size=n, replace=False)
            x_coords = [x_coords[i] for i in indices]
            y_coords = [y_coords[i] for i in indices]
        elif len(x_coords) < n:
            # If we don't have enough, pad with random points
            extra_points = n - len(x_coords)
            random_x = np.random.rand(extra_points) * 0.9 + 0.05
            random_y = np.random.rand(extra_points) * 0.9 + 0.05
            x_coords.extend(random_x)
            y_coords.extend(random_y)
        
        # Truncate to exactly n points
        x_coords = x_coords[:n]
        y_coords = y_coords[:n]
        
        # Initialize with reasonable radii
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i, 0] = x_coords[i]  # x coordinate
            circles[i, 1] = y_coords[i]  # y coordinate
            circles[i, 2] = 0.05  # Initial medium radius
        
        return circles

    def _objective_function(x: np.ndarray) -> float:
        """Objective function to maximize sum of radii (minimize negative sum)"""
        # Sum of all radii (every 3rd element starting from index 2)
        total_radius = np.sum(x[2::3])  
        return -total_radius  # Negative because we minimize

    def _constraint_functions(x: np.ndarray, n: int) -> list:
        """Generate constraint functions for scipy optimization"""
        constraints = []
        
        # Boundary constraints: each circle must stay within the unit square
        for i in range(n):
            def bound_x_constraint(x_vec, idx=i):
                pos_idx = 3 * idx
                r = x_vec[pos_idx + 2]
                # Both x - r >= 0 AND 1 - x - r >= 0
                return min(x_vec[pos_idx] - r, 1 - x_vec[pos_idx] - r)
            
            def bound_y_constraint(x_vec, idx=i):
                pos_idx = 3 * idx
                r = x_vec[pos_idx + 2]
                # Both y - r >= 0 AND 1 - y - r >= 0
                return min(x_vec[pos_idx + 1] - r, 1 - x_vec[pos_idx + 1] - r)
            
            constraints.append({'type': 'ineq', 'fun': bound_x_constraint})
            constraints.append({'type': 'ineq', 'fun': bound_y_constraint})
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                def overlap_constraint(x_vec, idx1=i, idx2=j):
                    pos_idx1 = 3 * idx1
                    pos_idx2 = 3 * idx2
                    x1, y1, r1 = x_vec[pos_idx1], x_vec[pos_idx1 + 1], x_vec[pos_idx1 + 2]
                    x2, y2, r2 = x_vec[pos_idx2], x_vec[pos_idx2 + 1], x_vec[pos_idx2 + 2]
                    
                    # Distance between centers
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    # Constraint: dist >= r1 + r2 (non-overlapping)
                    # So we want: dist - r1 - r2 >= 0
                    return dist - r1 - r2
                
                constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return constraints

    def _optimize_circles_mathematical(circles: np.ndarray, max_iter: int = 5000) -> np.ndarray:
        """Optimize using mathematical programming approach with scipy"""
        # Flatten the circles array for optimization (x1, y1, r1, x2, y2, r2, ...)
        x0 = circles.flatten()
        
        # Bounds for variables: x, y in [r, 1-r], r in [0.001, 0.499] 
        bounds = []
        for i in range(n):
            # x coordinate bounds (avoid exact boundaries)
            bounds.append((0.001, 0.999))  
            # y coordinate bounds  
            bounds.append((0.001, 0.999))
            # radius bounds (positive, reasonable max)
            bounds.append((0.001, 0.499))
        
        # Build constraints
        constraints = _constraint_functions(x0, n)
        
        # Use SLSQP method which handles both constraints and bounds well
        try:
            result = minimize(
                fun=_objective_function,
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'eps': 1e-8},
                callback=None
            )
            
            if result.success:
                optimized_array = result.x.reshape(-1, 3)
                return optimized_array
            else:
                print(f"Optimization failed: {result.message}")
                return circles
                
        except Exception as e:
            print(f"Mathematical optimization error: {e}")
            return circles

    def _refine_with_local_search(circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
        """Refine solution with local search to improve quality"""
        # Create a copy to work with
        refined_circles = circles.copy()
        
        # Try to improve by adjusting individual circles
        for iteration in range(max_iter):
            improved = False
            
            # Try moving each circle slightly to improve the total sum
            for i in range(n):
                original_pos = refined_circles[i, :2].copy()
                original_rad = refined_circles[i, 2]
                
                # Store the current total
                current_sum = np.sum(refined_circles[:, 2])
                
                # Try small adjustments
                for _ in range(3):  # Even fewer tries to save time but still explore
                    # Small random perturbation to position
                    delta_x = random.uniform(-0.003, 0.003)
                    delta_y = random.uniform(-0.003, 0.003)
                    new_x = max(0.01, min(0.99, original_pos[0] + delta_x))
                    new_y = max(0.01, min(0.99, original_pos[1] + delta_y))
                    
                    # Try to increase radius while maintaining feasibility
                    test_radius = min(0.45, original_rad + random.uniform(0, 0.005))
                    
                    # Temporarily update this circle
                    old_circle = refined_circles[i].copy()
                    refined_circles[i] = [new_x, new_y, test_radius]
                    
                    # Check if this violates any constraints
                    valid = True
                    for j in range(n):
                        if i != j:
                            dist = np.sqrt((new_x - refined_circles[j, 0])**2 + 
                                         (new_y - refined_circles[j, 1])**2)
                            if dist < (test_radius + refined_circles[j, 2]):
                                valid = False
                                break
                    
                    # Check boundary constraints
                    if valid and (new_x - test_radius < 0 or new_x + test_radius > 1 or 
                                  new_y - test_radius < 0 or new_y + test_radius > 1):
                        valid = False
                    
                    if valid:
                        # Check if this actually improves the total sum
                        new_sum = np.sum(refined_circles[:, 2])
                        if new_sum > current_sum:
                            improved = True
                            current_sum = new_sum
                        else:
                            # Revert if not beneficial
                            refined_circles[i] = old_circle
                    else:
                        # Revert if invalid
                        refined_circles[i] = old_circle
            
            if not improved:
                break
        
        return refined_circles

    # Main optimization process - try multiple initializations for better results
    best_sum = 0.0
    best_circles = None
    
    # Try several different initializations
    for attempt in range(3):
        # Use different seed for variety
        np.random.seed(42 + attempt * 100)
        random.seed(42 + attempt * 100)
        
        # Use hexagonal initialization which typically gives better starting points
        circles = _initialize_circles_hexagonal(n)
        
        # First optimization using mathematical programming approach
        optimized_circles = _optimize_circles_mathematical(circles, max_iter=3000)
        
        # Refinement with local search
        final_config = _refine_with_local_search(optimized_circles, max_iter=300)
        
        # Check if this is better
        current_sum = np.sum(final_config[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = final_config.copy()
    
    # If we didn't get a good result, fall back to single run with better settings
    if best_circles is None:
        print("Using single optimization run...")
        circles = _initialize_circles_hexagonal(n)
        optimized_circles = _optimize_circles_mathematical(circles, max_iter=4000)
        best_circles = _refine_with_local_search(optimized_circles, max_iter=500)
    
    return best_circles


# EVOLVE-BLOCK-END
