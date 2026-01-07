# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Better initialization using a more systematic approach
    def initialize_better():
        # Try different arrangements and pick the best starting point
        best_config = None
        best_sum = 0
        
        # Strategy 1: Grid-based initialization with adaptive spacing
        for rows in [5, 6, 7]:
            cols = math.ceil(n / rows)
            if rows * cols >= n:
                spacing_x = 1.0 / cols
                spacing_y = 1.0 / rows
                
                circles = []
                count = 0
                
                for i in range(rows):
                    y = spacing_y * (i + 0.5)
                    for j in range(cols):
                        x = spacing_x * (j + 0.5)
                        
                        # Ensure we don't exceed boundaries
                        if x >= 0 and x <= 1 and y >= 0 and y <= 1:
                            # Initial radius: based on spacing
                            max_r = min(spacing_x, spacing_y) * 0.4
                            circles.append([x, y, max_r])
                            count += 1
                            if count >= n:
                                break
                    if count >= n:
                        break
                
                # Fill remaining circles if needed
                while len(circles) < n:
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.uniform(0.05, 0.95)
                    r = 0.05
                    circles.append([x, y, r])
                
                # Calculate sum of radii for this configuration
                sum_radii = sum(circle[2] for circle in circles)
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_config = circles
        
        # Strategy 2: Alternative - place circles near corners and center
        if best_config is None:
            circles = []
            # Place some circles near corners
            corner_positions = [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]
            for i, (x, y) in enumerate(corner_positions):
                if i < n:
                    circles.append([x, y, 0.05])
            
            # Fill remaining positions with center distribution
            remaining = n - len(circles)
            for i in range(remaining):
                x = np.random.uniform(0.2, 0.8)
                y = np.random.uniform(0.2, 0.8)
                r = 0.05
                circles.append([x, y, r])
            
            best_config = circles
            
        return np.array(best_config)
    
    # Phase 2: Optimization using scipy minimize with improved constraints
    def objective(x):
        # x contains [x1,y1,r1,x2,y2,r2,...,x32,y32,r32]
        # We want to maximize sum of radii
        total_radius = 0
        for i in range(0, len(x), 3):
            total_radius += x[i+2]  # radius component
        return -total_radius  # negative because we minimize
    
    def constraint_containment(x):
        # Ensure all circles are within bounds
        constraints = []
        for i in range(0, len(x), 3):
            x_pos, y_pos, r = x[i], x[i+1], x[i+2]
            # Circle must be fully inside unit square
            constraints.append(x_pos - r)  # x - r >= 0
            constraints.append(y_pos - r)  # y - r >= 0
            constraints.append(1 - x_pos - r)  # 1 - x - r >= 0
            constraints.append(1 - y_pos - r)  # 1 - y - r >= 0
        return np.array(constraints)
    
    def constraint_nonoverlap(x):
        # Ensure no overlap between circles
        constraints = []
        # Create positions and radii arrays
        positions = []
        radii = []
        for i in range(0, len(x), 3):
            positions.append([x[i], x[i+1]])
            radii.append(x[i+2])
        
        positions = np.array(positions)
        radii = np.array(radii)
        
        # More efficient constraint checking using vectorized operations
        # Only check relevant pairs to reduce computation
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                # Vectorized distance calculation
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                sum_radii = radii[i] + radii[j]
                # Constraint: dist >= sum_radii => dist - sum_radii >= 0
                constraints.append(dist - sum_radii)
        
        return np.array(constraints)
    
    # Initialize
    initial_circles = initialize_better()
    x0 = initial_circles.flatten()
    
    # Set up bounds: [x_min, x_max, y_min, y_max, r_min, r_max] for each circle
    bounds = []
    for i in range(n):
        # More generous bounds for better optimization
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
    ]
    
    # Run optimization with better parameters
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # Fallback to initial configuration if optimization fails
            return initial_circles
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        return initial_circles


# EVOLVE-BLOCK-END
