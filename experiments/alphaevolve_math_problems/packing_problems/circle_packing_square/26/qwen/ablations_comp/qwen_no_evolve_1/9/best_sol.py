# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining Voronoi initialization with sequential quadratic programming.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize using Voronoi-based distribution
    initial_circles = _voronoi_initialization(n)
    
    # Refine using SQP optimization
    optimized_circles = _optimize_circles(initial_circles)
    
    # Final refinement with local search
    final_circles = _local_refinement(optimized_circles)
    
    return final_circles

def _voronoi_initialization(n: int) -> np.ndarray:
    """Initialize circle positions using a Voronoi-like distribution"""
    # Generate points using a quasi-random sequence for good coverage
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    points = []
    
    # Generate points in a way that avoids clustering
    for i in range(n):
        x = (i + 0.5) / n
        y = (i * phi) % 1
        points.append([x, y])
    
    points = np.array(points)
    
    # Distribute circles with initial radii based on density
    circles = np.zeros((n, 3))
    
    # Simple heuristic: place circles with radii inversely proportional to distance to edges
    for i in range(n):
        x, y = points[i]
        # Ensure circles don't touch boundaries
        min_dist_to_edge = min(x, 1-x, y, 1-y)
        # Initial radius guess based on proximity to center and spacing
        radius = min(0.1, min_dist_to_edge * 0.8)
        circles[i] = [x, y, radius]
    
    return circles

def _distance_constraint(circles: np.ndarray, i: int, j: int) -> float:
    """Calculate minimum required distance between circles i and j"""
    x1, y1, r1 = circles[i]
    x2, y2, r2 = circles[j]
    return np.sqrt((x1-x2)**2 + (y1-y2)**2) - (r1 + r2)

def _constraint_violation(circles: np.ndarray) -> float:
    """Calculate total constraint violation (negative values indicate violations)"""
    n = len(circles)
    violation = 0.0
    
    # Check all pairs for overlap
    for i in range(n):
        for j in range(i+1, n):
            dist = _distance_constraint(circles, i, j)
            if dist < 0:
                violation += dist
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x-r < 0 or x+r > 1 or y-r < 0 or y+r > 1:
            violation -= 1.0  # Large penalty for boundary violations
    
    return violation

def _objective_function(circles: np.ndarray) -> float:
    """Objective: maximize sum of radii"""
    return -np.sum(circles[:, 2])  # Negative because we minimize

def _boundary_constraints(circles: np.ndarray) -> np.ndarray:
    """Generate boundary constraints for optimization"""
    n = len(circles)
    constraints = []
    
    # Boundary constraints: each circle must stay within [0,1]x[0,1]
    for i in range(n):
        x, y, r = circles[i]
        # r <= x <= 1-r and r <= y <= 1-r
        constraints.extend([
            {'type': 'ineq', 'fun': lambda c, i=i: c[i, 0] - c[i, 2]},  # x >= r
            {'type': 'ineq', 'fun': lambda c, i=i: 1 - c[i, 0] - c[i, 2]},  # 1-x >= r
            {'type': 'ineq', 'fun': lambda c, i=i: c[i, 1] - c[i, 2]},  # y >= r
            {'type': 'ineq', 'fun': lambda c, i=i: 1 - c[i, 1] - c[i, 2]}   # 1-y >= r
        ])
    
    return constraints

def _overlap_constraints(circles: np.ndarray) -> list:
    """Generate overlap constraints for optimization"""
    n = len(circles)
    constraints = []
    
    # Overlap constraints: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(c, i=i, j=j):
                x1, y1, r1 = c[i]
                x2, y2, r2 = c[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                return dist - (r1 + r2)
            
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    return constraints

def _optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Refine circle positions using sequential quadratic programming"""
    n = len(initial_circles)
    
    # Flatten initial circles for optimization
    initial_flat = initial_circles.flatten()
    
    # Define bounds for optimization variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x, y, r bounds
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # r capped at 0.5
    
    # Create constraint dictionaries
    constraints = []
    
    # Add boundary constraints
    for i in range(n):
        def bound_constraint_x(c, i=i):
            return c[3*i] - c[3*i+2]  # x >= r
        def bound_constraint_x_inv(c, i=i):
            return 1 - c[3*i] - c[3*i+2]  # 1-x >= r
        def bound_constraint_y(c, i=i):
            return c[3*i+1] - c[3*i+2]  # y >= r
        def bound_constraint_y_inv(c, i=i):
            return 1 - c[3*i+1] - c[3*i+2]  # 1-y >= r
            
        constraints.extend([
            {'type': 'ineq', 'fun': bound_constraint_x},
            {'type': 'ineq', 'fun': bound_constraint_x_inv},
            {'type': 'ineq', 'fun': bound_constraint_y},
            {'type': 'ineq', 'fun': bound_constraint_y_inv}
        ])
    
    # Add overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(c, i=i, j=j):
                x1, y1, r1 = c[3*i], c[3*i+1], c[3*i+2]
                x2, y2, r2 = c[3*j], c[3*j+1], c[3*j+2]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                return dist - (r1 + r2)
            
            constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    # Optimization parameters
    options = {'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
    
    try:
        # Use SLSQP optimizer
        result = minimize(
            fun=lambda c: _objective_function(c.reshape(-1, 3)),
            x0=initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=1e-6
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception:
        pass
    
    # If optimization fails, return initial circles
    return initial_circles

def _local_refinement(circles: np.ndarray) -> np.ndarray:
    """Perform local refinement using greedy improvement steps"""
    n = len(circles)
    current_circles = circles.copy()
    
    # Try to improve by adjusting individual circles
    improved = True
    iterations = 0
    
    while improved and iterations < 100:
        improved = False
        iterations += 1
        
        # Try to increase radii of each circle
        for i in range(n):
            original_radius = current_circles[i, 2]
            
            # Find maximum possible radius for this circle
            max_radius = _compute_max_radius(current_circles, i)
            
            if max_radius > original_radius:
                current_circles[i, 2] = max_radius
                improved = True
                
                # Check if we still have valid configuration
                if not _is_valid_configuration(current_circles):
                    current_circles[i, 2] = original_radius
                    improved = False
    
    return current_circles

def _compute_max_radius(circles: np.ndarray, idx: int) -> float:
    """Compute maximum possible radius for circle at index idx"""
    x, y, r = circles[idx]
    
    # Distance to boundaries
    min_dist_to_edge = min(x, 1-x, y, 1-y)
    
    # Distance to other circles
    min_dist_to_others = float('inf')
    
    for i in range(len(circles)):
        if i != idx:
            x2, y2, r2 = circles[i]
            dist = np.sqrt((x-x2)**2 + (y-y2)**2)
            min_dist_to_others = min(min_dist_to_others, dist)
    
    # Maximum radius is limited by both boundaries and other circles
    if min_dist_to_others < float('inf'):
        max_radius = min(min_dist_to_edge, min_dist_to_others/2)
    else:
        max_radius = min_dist_to_edge
    
    return max(0, max_radius)

def _is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if current configuration is valid"""
    n = len(circles)
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x-r < 0 or x+r > 1 or y-r < 0 or y+r > 1:
            return False
    
    # Check overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < r1 + r2:
                return False
    
    return True


# EVOLVE-BLOCK-END
