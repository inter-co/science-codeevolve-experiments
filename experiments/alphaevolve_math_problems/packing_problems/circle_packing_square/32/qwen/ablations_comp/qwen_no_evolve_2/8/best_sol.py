# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using hexagonal packing pattern for good starting configuration
    def initialize_hexagonal():
        # Create a hexagonal grid pattern
        rows = 6
        cols = 6
        circles = []
        
        # Hexagonal packing parameters
        spacing_x = 0.2
        spacing_y = 0.1732  # sqrt(3)/2 * spacing_x
        
        # Generate points in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Offset odd rows
                if i % 2 == 1:
                    x += spacing_x / 2
                
                # Only add if within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append([x, y, 0.05])  # Initial radius guess
                    
        # Fill remaining positions with random points if needed
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        return np.array(circles[:n])
    
    # Constraint functions
    def containment_constraints(circles_flat):
        """Ensure all circles are fully contained in unit square"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Each circle's center must be at least radius away from edges
        for i in range(len(circles)):
            x, y, r = circles[i]
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1-x >= r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})  # 1-y >= r
            
        return constraints
    
    def non_overlap_constraints(circles_flat):
        """Ensure no two circles overlap"""
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # For each pair of circles, ensure distance >= sum of radii
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                def constraint_func(c, i=i, j=j):
                    x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                    x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                    distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_distance_sq = (r1 + r2)**2
                    return distance_sq - min_distance_sq
                
                constraints.append({'type': 'ineq', 'fun': constraint_func})
                
        return constraints
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Initialize
    initial_circles = initialize_hexagonal()
    initial_flat = initial_circles.flatten()
    
    # Set up constraints
    cons = []
    cons.extend(containment_constraints(initial_flat))
    cons.extend(non_overlap_constraints(initial_flat))
    
    # Bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Optimization using SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
        else:
            # Fallback to initial configuration if optimization fails
            final_circles = initial_circles
            
    except Exception as e:
        # If optimization fails, return initial configuration
        final_circles = initial_circles
    
    # Ensure final configuration respects all constraints
    # Re-adjust radii to prevent overlaps
    final_circles = adjust_for_overlaps(final_circles)
    
    return final_circles


def adjust_for_overlaps(circles):
    """Adjust circles to ensure no overlaps while maintaining good packing"""
    # Simple iterative adjustment
    max_iter = 100
    for _ in range(max_iter):
        changed = False
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            for j in range(i+1, len(circles)):
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                min_dist = r1 + r2
                
                if distance < min_dist:
                    # Adjust radii to prevent overlap
                    total_radius = r1 + r2
                    new_total_radius = max(0.001, distance - 0.001)
                    
                    # Reduce both radii proportionally
                    ratio = new_total_radius / total_radius
                    circles[i][2] *= ratio
                    circles[j][2] *= ratio
                    changed = True
                    
        if not changed:
            break
            
    # Ensure containment
    for i in range(len(circles)):
        x, y, r = circles[i]
        circles[i][0] = np.clip(x, r, 1-r)
        circles[i][1] = np.clip(y, r, 1-r)
        
    return circles


# EVOLVE-BLOCK-END
