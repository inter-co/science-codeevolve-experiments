# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial hexagonal lattice placement with optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a good heuristic placement
    # Try hexagonal lattice pattern first
    circles = initialize_hexagonal_placement(n)
    
    # Refine using optimization
    circles = optimize_circles(circles)
    
    return circles

def initialize_hexagonal_placement(n):
    """Initialize circle positions using hexagonal lattice pattern"""
    # For 26 circles, we can arrange in roughly a 5x5 grid with hexagonal offset
    # But let's be more systematic
    
    # Create a better initial configuration
    circles = np.zeros((n, 3))
    
    # Use a systematic approach with different ring patterns
    # First try to place in a way that maximizes initial spacing
    radius_estimate = 0.1  # Starting guess
    
    # Place circles in a structured pattern
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Ensure we don't exceed the unit square bounds
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Adjust for hexagonal pattern in odd rows
            if i % 2 == 1:
                x += spacing_x * 0.5
            circles[idx] = [x, y, radius_estimate]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with small radii
    for i in range(idx, n):
        circles[i] = [0.5, 0.5, 0.01]
    
    return circles

def optimize_circles(initial_circles):
    """Optimize circle positions and radii using constrained optimization"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_flat = []
    for i in range(n):
        initial_flat.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    def objective(radii_flat):
        """Minimize negative sum of radii (maximize sum of radii)"""
        total_radius = sum(radii_flat[2::3])  # Every third element starting from index 2
        return -total_radius
    
    def constraint_containment(radii_flat):
        """Ensure all circles stay within unit square"""
        constraints = []
        for i in range(n):
            x = radii_flat[i*3]
            y = radii_flat[i*3 + 1]
            r = radii_flat[i*3 + 2]
            
            # Circle must fit within bounds
            constraints.append(x - r)  # x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        return np.array(constraints)
    
    def constraint_overlap(radii_flat):
        """Ensure no overlapping circles"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = radii_flat[i*3], radii_flat[i*3 + 1], radii_flat[i*3 + 2]
                x2, y2, r2 = radii_flat[j*3], radii_flat[j*3 + 1], radii_flat[j*3 + 2]
                
                # Distance between centers must be at least r1 + r2
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # Constraint: dist_sq >= min_dist_sq (equivalent to dist_sq - min_dist_sq >= 0)
                constraints.append(dist_sq - min_dist_sq)
        
        return np.array(constraints)
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # x, y, r bounds
        bounds.append((0.001, 0.999))  # x coordinate
        bounds.append((0.001, 0.999))  # y coordinate  
        bounds.append((0.001, 0.499))  # radius (limited to prevent overlap issues)
    
    # Create constraints dictionary
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_overlap(x)}
    ]
    
    # Perform optimization
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
            optimized_flat = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [optimized_flat[i*3], optimized_flat[i*3 + 1], optimized_flat[i*3 + 2]]
            return circles
    except Exception as e:
        pass
    
    # If optimization fails, return initial configuration
    return initial_circles


# EVOLVE-BLOCK-END
