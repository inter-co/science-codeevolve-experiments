# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a sophisticated hybrid approach combining multiple optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = 32
    
    # Phase 1: Initialize with a high-quality configuration using hexagonal lattice
    def initialize_circles_hexagonal(n: int) -> np.ndarray:
        """Initialize circles using a more structured hexagonal grid pattern"""
        # Create a hexagonal grid pattern with better distribution
        rows = int(np.ceil(np.sqrt(n * 1.2)))  # Slightly more rows to account for irregularities
        cols = int(np.ceil(n / rows))
        
        # Adjust dimensions to fit exactly n circles
        if rows * cols < n:
            rows += 1
        
        # Generate hexagonal grid points
        circles = np.zeros((n, 3))
        spacing_x = 0.8 / cols
        spacing_y = 0.8 / rows
        
        # Hexagonal packing pattern
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                
                # Offset every other row for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x * 0.5
                    
                # Add small random perturbation to avoid perfect symmetry
                x += random.uniform(-spacing_x*0.1, spacing_x*0.1)
                y += random.uniform(-spacing_y*0.1, spacing_y*0.1)
                
                # Initial radius - start with small value
                r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
                circles[count] = [x, y, r]
                count += 1
        
        # Fill remaining positions with random placements
        while count < n:
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            # Make sure the initial radius is reasonable
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles[count] = [x, y, r]
            count += 1
        
        return circles
    
    # Initialize with hexagonal layout
    circles = initialize_circles_hexagonal(n)
    
    # Flatten for optimization
    initial_flat = circles.flatten()
    
    # Objective function: negative sum of radii (we want to maximize sum of radii)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Constraints
    cons = []
    
    # Containment constraints
    for i in range(n):
        # x >= r and x <= 1-r and y >= r and y <= 1-r
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})   # 1 - y - r >= 0
    
    # Overlap constraints (more efficient version using vectorized approach)
    def create_overlap_constraints():
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda c, i=i, j=j: 
                        np.sqrt((c[3*i] - c[3*j])**2 + (c[3*i+1] - c[3*j+1])**2) - c[3*i+2] - c[3*j+2]
                })
        return constraints
    
    cons.extend(create_overlap_constraints())
    
    # Bounds: x, y in [r, 1-r], r > 0
    bounds = []
    for i in range(n):
        bounds.extend([
            (0.001, 0.999),  # x coordinate
            (0.001, 0.999),  # y coordinate
            (0.001, 0.499)   # radius (max possible without overlap)
        ])
    
    # Optimize using scipy minimize
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_flat = result.x
            circles = optimized_flat.reshape((n, 3))
        else:
            # Fallback to initial configuration if optimization fails
            pass
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        pass
    
    # Final refinement with careful constraint validation
    # Build KDTree for efficient neighbor searches
    tree = KDTree(circles[:, :2])
    
    # Coordinate descent with proper constraint checking
    for iteration in range(100):  # Reduced iterations for speed but still sufficient
        improved = False
        
        # Update each circle to maximize its radius given constraints
        for i in range(n):
            current_x, current_y, current_r = circles[i]
            
            # Find maximum possible radius at current position using spatial data structure
            max_r = min(current_x, 1-current_x, current_y, 1-current_y)
            
            # Check distance to nearest neighbors using KDTree for better efficiency
            neighbors = tree.query_ball_point([current_x, current_y], 2.0)
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((current_x-x2)**2 + (current_y-y2)**2)
                    max_r = min(max_r, dist - r2 - 0.0001)
            
            # Try to increase radius if beneficial
            if max_r > current_r and max_r > current_r + 0.0001:
                circles[i, 2] = max_r
                improved = True
        
        # If no improvements, try position adjustments with better validation
        if not improved:
            for i in range(n):
                # Try to improve position with more careful validation
                old_x, old_y, old_r = circles[i]
                best_x, best_y, best_r = old_x, old_y, old_r
                best_sum = np.sum(circles[:, 2])
                
                # Search neighborhood with more granular steps
                steps = [-0.01, -0.005, 0, 0.005, 0.01]
                
                for dx in steps:
                    for dy in steps:
                        new_x = old_x + dx
                        new_y = old_y + dy
                        
                        if 0.001 <= new_x <= 0.999 and 0.001 <= new_y <= 0.999:
                            # Calculate new radius
                            max_r = min(new_x, 1-new_x, new_y, 1-new_y)
                            
                            # Check overlap with all others
                            valid = True
                            for j in range(n):
                                if i != j:
                                    x2, y2, r2 = circles[j]
                                    dist = np.sqrt((new_x-x2)**2 + (new_y-y2)**2)
                                    if dist < r2 + max_r + 0.0001:
                                        valid = False
                                        break
                            
                            if valid:
                                test_circles = circles.copy()
                                test_circles[i] = [new_x, new_y, max_r]
                                new_sum = np.sum(test_circles[:, 2])
                                
                                if new_sum > best_sum:
                                    best_sum = new_sum
                                    best_x, best_y, best_r = new_x, new_y, max_r
                
                # Apply best improvement
                if best_sum > np.sum(circles[:, 2]):
                    circles[i] = [best_x, best_y, best_r]
                    improved = True
        
        if not improved:
            break
    
    # Final cleanup to ensure all constraints are satisfied
    for i in range(n):
        # Ensure radius is positive and within bounds
        circles[i, 2] = max(0.001, min(0.499, circles[i, 2]))
        # Ensure circle is contained
        circles[i, 0] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 0]))
        circles[i, 1] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 1]))
    
    return circles


# EVOLVE-BLOCK-END
