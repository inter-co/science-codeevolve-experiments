# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a good starting configuration
    initial_config = _initialize_circles()
    
    # Extract initial parameters (x, y, r for all circles)
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_config[i][0], initial_config[i][1], initial_config[i][2]])
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x coordinate: [r, 1-r] 
        bounds.append((0.001, 0.999))  # Small epsilon to avoid boundary issues
        # y coordinate: [r, 1-r]
        bounds.append((0.001, 0.999))
        # r coordinate: [0.001, min(x, 1-x, y, 1-y)]
        bounds.append((0.001, 0.5))
    
    # Define constraint function for non-overlapping circles
    def non_overlap_constraint(params):
        """Returns positive values when constraints are violated"""
        positions = []
        radii = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            positions.append([x, y])
            radii.append(r)
        
        positions = np.array(positions)
        distances = cdist(positions, positions)
        violations = []
        
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i,j]
                min_dist = radii[i] + radii[j]
                # Violation: negative when constraint is satisfied
                violations.append(min_dist - dist)
        
        return np.array(violations)
    
    # Define constraint for containment
    def containment_constraint(params):
        """Returns positive values when constraints are violated"""
        violations = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            
            # Check containment constraints
            violations.append(r - x)      # x >= r
            violations.append(r - y)      # y >= r
            violations.append(x + r - 1)  # x + r <= 1
            violations.append(y + r - 1)  # y + r <= 1
            
        return np.array(violations)
    
    # Optimization constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda p: -non_overlap_constraint(p)},
        {'type': 'ineq', 'fun': lambda p: -containment_constraint(p)}
    ]
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(params):
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i+2]  # Sum of all radii
        return -total_radius  # Negative because minimize
    
    # Run optimization
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            final_params = result.x
        else:
            # If optimization fails, return the initial configuration
            final_params = initial_params
    except Exception:
        # If optimization fails due to any reason, return initial configuration
        final_params = initial_params
    
    # Construct final result
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i][0] = final_params[3*i]     # x coordinate
        circles[i][1] = final_params[3*i+1]   # y coordinate
        circles[i][2] = final_params[3*i+2]   # radius
    
    return circles

def _initialize_circles() -> list:
    """
    Initialize circles using a hexagonal packing approach for better starting configuration.
    """
    n = 32
    circles = []
    
    # Try different packing arrangements and pick the best one
    best_config = None
    best_sum = 0
    
    # Try several random initializations
    for trial in range(10):
        # Create a hexagonal grid pattern with some randomness
        temp_circles = []
        
        # Grid dimensions
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Hexagonal packing parameters
        spacing_x = 0.15
        spacing_y = 0.15 * np.sqrt(3)/2
        
        # Generate circles in a grid pattern with slight perturbations
        radius = 0.05  # Initial radius guess
        
        for i in range(rows):
            for j in range(cols):
                if len(temp_circles) >= n:
                    break
                    
                # Position with hexagonal offset
                x = 0.1 + j * spacing_x + (i % 2) * spacing_x/2
                y = 0.1 + i * spacing_y
                
                # Keep within bounds
                if x + radius > 0.9 or y + radius > 0.9:
                    continue
                    
                temp_circles.append([x, y, radius])
        
        # Add more circles if needed
        while len(temp_circles) < n:
            x = 0.1 + np.random.random() * 0.8
            y = 0.1 + np.random.random() * 0.8
            radius = 0.05 + np.random.random() * 0.05
            temp_circles.append([x, y, radius])
            
        temp_circles = temp_circles[:n]
        
        # Calculate sum of radii for this configuration
        sum_radii = sum(circle[2] for circle in temp_circles)
        
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_config = temp_circles[:]
    
    return best_config if best_config is not None else [[0.5, 0.5, 0.1]] * 32


# EVOLVE-BLOCK-END
