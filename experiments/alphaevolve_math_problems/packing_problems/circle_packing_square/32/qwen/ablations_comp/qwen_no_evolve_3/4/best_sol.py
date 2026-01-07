# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import math
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal lattice initialization and gradient-based optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using hexagonal lattice pattern for good initial distribution
    circles = initialize_hexagonal_lattice(n)
    
    # Optimize using scipy minimize with constraints
    optimized_circles = optimize_circles(circles)
    
    return optimized_circles

def initialize_hexagonal_lattice(n: int) -> np.ndarray:
    """Initialize circle positions using a hexagonal lattice pattern."""
    # Create a hexagonal grid pattern
    rows = int(math.sqrt(n))
    cols = int(math.ceil(n / rows))
    
    # Adjust grid size to fit n circles
    while rows * cols < n:
        rows += 1
    
    # Calculate spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Create hexagonal pattern
    circles = np.zeros((n, 3))
    idx = 0
    
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row
            x_offset = (i % 2) * spacing_x / 2
            x = (j + 0.5) * spacing_x + x_offset
            y = (i + 0.5) * spacing_y
            
            # Ensure we're within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Initial radius - small enough to fit in the cell
            radius = min(spacing_x, spacing_y) / 4
            
            circles[idx] = [x, y, radius]
            idx += 1
    
    # Fill remaining circles with random positions near edges
    for i in range(idx, n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = np.random.uniform(0.01, 0.1)
        circles[i] = [x, y, radius]
    
    return circles

def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using constrained optimization."""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define constraint functions
    def contain_constraints(params):
        """Ensure all circles are contained within the unit square"""
        constraints = []
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            # Circle must be contained in unit square
            constraints.append(x - r)  # x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
        return np.array(constraints)
    
    def overlap_constraints(params):
        """Ensure no two circles overlap"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                
                # Distance between centers minus sum of radii should be >= 0
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                constraints.append(dist - (r1 + r2))
        return np.array(constraints)
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(params):
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i+2]  # radius is at index 3*i+2
        return -total_radius  # Negative because we want to maximize
    
    # Set up constraints for scipy.optimize
    cons = []
    
    # Add containment constraints
    cons.append({'type': 'ineq', 'fun': lambda p: contain_constraints(p)})
    
    # Add overlap constraints
    cons.append({'type': 'ineq', 'fun': lambda p: overlap_constraints(p)})
    
    # Set bounds for parameters (x, y, r)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r
    
    # Multi-start optimization to improve results
    best_result = None
    best_sum = float('-inf')
    
    # Try several random starts
    for _ in range(5):
        try:
            # Random perturbation of initial solution
            perturbed_params = initial_params.copy()
            for i in range(len(perturbed_params)):
                if i % 3 == 2:  # radius parameter
                    # Small random change to radius
                    perturbed_params[i] += np.random.normal(0, 0.01)
                    perturbed_params[i] = max(0.001, min(0.499, perturbed_params[i]))
                else:
                    # Small random change to position
                    perturbed_params[i] += np.random.normal(0, 0.02)
                    perturbed_params[i] = max(0.001, min(0.999, perturbed_params[i]))
            
            # Optimize with this start
            result = minimize(
                objective,
                perturbed_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if result.success:
                # Calculate sum of radii for this solution
                current_sum = -result.fun  # Convert back to positive
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result.x
        except Exception:
            continue
    
    # If we have a better solution, use it; otherwise use initial
    if best_result is not None:
        final_params = best_result
    else:
        final_params = initial_params
    
    # Convert back to circle array format
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
    
    return circles


# EVOLVE-BLOCK-END
