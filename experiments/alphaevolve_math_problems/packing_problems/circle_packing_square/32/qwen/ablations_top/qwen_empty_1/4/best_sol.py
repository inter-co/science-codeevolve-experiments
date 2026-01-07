# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a Voronoi-based initialization followed by gradient optimization with constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Step 1: Generate initial configuration using Voronoi diagram
    # Create a grid of points and use Voronoi to generate initial positions
    initial_points = _generate_voronoi_initialization(n)
    
    # Step 2: Initialize radii based on Voronoi cell areas
    initial_radii = _initialize_radii(initial_points)
    
    # Step 3: Combine positions and radii into circles array
    circles = np.column_stack([initial_points, initial_radii])
    
    # Step 4: Optimize using constrained optimization
    optimized_circles = _optimize_circles(circles)
    
    return optimized_circles

def _generate_voronoi_initialization(n: int) -> np.ndarray:
    """Generate initial point distribution using Voronoi-like spacing"""
    # Create a more uniform distribution using a grid with perturbations
    sqrt_n = int(np.ceil(np.sqrt(n)))
    x = np.linspace(0.05, 0.95, sqrt_n)
    y = np.linspace(0.05, 0.95, sqrt_n)
    
    # Create grid points
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Trim to exact number needed
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        # Add extra points with random perturbations
        extra_points = np.random.rand(n - len(points), 2) * 0.9 + 0.05
        points = np.vstack([points, extra_points])
    
    return points

def _initialize_radii(points: np.ndarray) -> np.ndarray:
    """Initialize radii based on Voronoi cell areas"""
    # For simplicity, initialize with equal small radii
    # Then adjust based on local density
    n = len(points)
    radii = np.full(n, 0.02)
    
    # Adjust radii to fit within square bounds
    for i in range(n):
        min_dist_to_boundary = min(
            points[i, 0], 1 - points[i, 0],
            points[i, 1], 1 - points[i, 1]
        )
        radii[i] = min(radii[i], min_dist_to_boundary * 0.9)
    
    return radii

def _compute_constraints(circles: np.ndarray) -> tuple:
    """Compute constraint violations"""
    n = len(circles)
    # Check boundary constraints
    boundary_violations = []
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            boundary_violations.append(i)
    
    # Check overlap constraints
    overlap_violations = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < r1 + r2:
                overlap_violations.append((i, j))
    
    return boundary_violations, overlap_violations

def _objective_function(circles_flat: np.ndarray) -> float:
    """Objective function to maximize sum of radii"""
    # Reshape flat array back to circles
    n = len(circles_flat) // 3
    circles = circles_flat.reshape(n, 3)
    
    # Sum of radii (negative because we minimize)
    return -np.sum(circles[:, 2])

def _constraint_functions(circles_flat: np.ndarray) -> list:
    """Generate constraint functions for optimization"""
    n = len(circles_flat) // 3
    circles = circles_flat.reshape(n, 3)
    
    constraints = []
    
    # Boundary constraints (each circle must fit in unit square)
    for i in range(n):
        def bound_constraint(x_flat, idx=i):
            circles = x_flat.reshape(n, 3)
            x, y, r = circles[idx]
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        
        constraints.append({
            'type': 'ineq',
            'fun': bound_constraint
        })
    
    # Overlap constraints (distance between centers >= sum of radii)
    for i in range(n):
        for j in range(i+1, n):
            def overlap_constraint(x_flat, idx1=i, idx2=j):
                circles = x_flat.reshape(n, 3)
                x1, y1, r1 = circles[idx1]
                x2, y2, r2 = circles[idx2]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                return dist - (r1 + r2)
            
            constraints.append({
                'type': 'ineq',
                'fun': overlap_constraint
            })
    
    return constraints

def _optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circles using scipy's minimize with constraints"""
    n = len(initial_circles)
    
    # Flatten the initial configuration
    initial_flat = initial_circles.flatten()
    
    # Define bounds for each variable (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x bounds: r <= x <= 1-r
        bounds.append((initial_circles[i, 2], 1 - initial_circles[i, 2]))
        # y bounds: r <= y <= 1-r  
        bounds.append((initial_circles[i, 2], 1 - initial_circles[i, 2]))
        # r bounds: 0 < r <= min(x, 1-x, y, 1-y)
        max_radius = min(
            initial_circles[i, 0], 1 - initial_circles[i, 0],
            initial_circles[i, 1], 1 - initial_circles[i, 1]
        ) * 0.99
        bounds.append((1e-6, max_radius))
    
    # Get constraint functions
    constraints = _constraint_functions(initial_flat)
    
    # Optimize
    try:
        result = minimize(
            _objective_function,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(n, 3)
            return optimized_circles
        else:
            # Return initial if optimization fails
            return initial_circles
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        return initial_circles


# EVOLVE-BLOCK-END
