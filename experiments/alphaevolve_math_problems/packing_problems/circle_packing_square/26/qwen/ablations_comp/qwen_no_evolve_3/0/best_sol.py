# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def _hexagonal_grid_placement(n: int) -> np.ndarray:
    """Create initial placement using hexagonal grid pattern."""
    # For 26 circles, we'll use approximately 5x5 grid with hexagonal packing
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    
    # Create hexagonal grid points
    positions = []
    radius = 0.1  # Initial estimate
    
    # Hexagonal packing with spacing
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Offset every other row
            x_offset = (j if i % 2 == 0 else j + 0.5)
            y = i * math.sqrt(3) / 2
            x = x_offset
            
            # Scale to fit in unit square
            positions.append([x, y])
    
    # Normalize to fit in [0,1] x [0,1]
    if positions:
        positions = np.array(positions[:n])
        # Scale and shift to fit in unit square
        min_x, max_x = positions[:, 0].min(), positions[:, 0].max()
        min_y, max_y = positions[:, 1].min(), positions[:, 1].max()
        
        if max_x > min_x and max_y > min_y:
            scale_x = 0.8 / (max_x - min_x)
            scale_y = 0.8 / (max_y - min_y)
            positions[:, 0] = (positions[:, 0] - min_x) * scale_x + 0.1
            positions[:, 1] = (positions[:, 1] - min_y) * scale_y + 0.1
    
    # Fill with zeros if not enough positions
    result = np.zeros((n, 2))
    if len(positions) > 0:
        result[:len(positions)] = positions
    return result

def _compute_radius_constraints(positions: np.ndarray, radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute constraints for containment and non-overlap."""
    n = len(positions)
    
    # Containment constraints: radius <= x, y and 1-radius >= x, y
    contain_constraints = []
    for i in range(n):
        contain_constraints.extend([
            positions[i][0] - radii[i],  # x >= r
            positions[i][1] - radii[i],  # y >= r
            1 - positions[i][0] - radii[i],  # 1-x >= r
            1 - positions[i][1] - radii[i]   # 1-y >= r
        ])
    
    # Non-overlap constraints: distance >= r_i + r_j
    overlap_constraints = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                          (positions[i][1] - positions[j][1])**2)
            overlap_constraints.append(dist - radii[i] - radii[j])
    
    return np.array(contain_constraints), np.array(overlap_constraints)

def _objective_function(params: np.ndarray, n: int) -> float:
    """Objective function to maximize sum of radii."""
    # Extract positions and radii
    positions = params[:2*n].reshape((n, 2))
    radii = params[2*n:]
    
    # Return negative because we're minimizing
    return -np.sum(radii)

def _constraint_function(params: np.ndarray, n: int) -> np.ndarray:
    """Constraint function for optimization."""
    positions = params[:2*n].reshape((n, 2))
    radii = params[2*n:]
    
    # Compute constraints
    contain, overlap = _compute_radius_constraints(positions, radii)
    
    # Return positive values when constraints are satisfied
    return np.concatenate([contain, overlap])

def _initial_guess(n: int) -> np.ndarray:
    """Create initial guess for optimization."""
    # Start with hexagonal placement
    positions = _hexagonal_grid_placement(n)
    
    # Initialize radii to small values
    radii = np.full(n, 0.05)
    
    # Adjust based on proximity
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                              (positions[i][1] - positions[j][1])**2)
                min_dist = min(min_dist, dist)
        
        # Set radius to be smaller than minimum distance between centers
        if min_dist > 0:
            radii[i] = min(0.1, min_dist / 3.0)
    
    # Ensure all radii are positive and within bounds
    radii = np.maximum(radii, 0.001)
    
    # Combine into single parameter vector
    initial_params = np.concatenate([positions.flatten(), radii])
    return initial_params

def _refine_with_simulated_annealing(circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Refine solution using simulated annealing approach."""
    current_circles = circles.copy()
    n = len(current_circles)
    
    # Temperature schedule
    temp = 0.1
    cooling_rate = 0.995
    
    def get_distance_matrix(circles_arr):
        positions = circles_arr[:, :2]
        radii = circles_arr[:, 2]
        distances = cdist(positions, positions)
        return distances, radii
    
    def compute_fitness(circles_arr):
        # Minimize negative sum of radii (maximize sum of radii)
        return -np.sum(circles_arr[:, 2])
    
    def is_valid(circles_arr):
        # Check containment and overlap constraints
        positions = circles_arr[:, :2]
        radii = circles_arr[:, 2]
        
        # Check containment
        for i in range(len(circles_arr)):
            if (radii[i] > positions[i][0] or 
                radii[i] > positions[i][1] or
                radii[i] > 1 - positions[i][0] or
                radii[i] > 1 - positions[i][1]):
                return False
        
        # Check overlap
        distances, _ = get_distance_matrix(circles_arr)
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < radii[i] + radii[j]:
                    return False
        return True
    
    current_fitness = compute_fitness(current_circles)
    
    for iteration in range(max_iter):
        # Generate neighbor solution
        neighbor_circles = current_circles.copy()
        
        # Perturb one circle
        idx = np.random.randint(0, n)
        # Randomly perturb position and radius
        neighbor_circles[idx, 0] += np.random.normal(0, 0.01)
        neighbor_circles[idx, 1] += np.random.normal(0, 0.01)
        neighbor_circles[idx, 2] += np.random.normal(0, 0.005)
        
        # Keep within bounds
        neighbor_circles[idx, 0] = np.clip(neighbor_circles[idx, 0], 0.001, 0.999)
        neighbor_circles[idx, 1] = np.clip(neighbor_circles[idx, 1], 0.001, 0.999)
        neighbor_circles[idx, 2] = np.clip(neighbor_circles[idx, 2], 0.001, 0.499)
        
        # Accept if better or with probability based on temperature
        new_fitness = compute_fitness(neighbor_circles)
        
        if new_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness) / temp):
            current_circles = neighbor_circles
            current_fitness = new_fitness
        
        # Cool down
        temp *= cooling_rate
    
    return current_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    circles = np.zeros((n, 3))
    
    # Stage 1: Initial placement using hexagonal grid
    positions = _hexagonal_grid_placement(n)
    
    # Stage 2: Initialize radii
    radii = np.full(n, 0.05)
    
    # Adjust radii based on proximity
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                              (positions[i][1] - positions[j][1])**2)
                min_dist = min(min_dist, dist)
        
        # Set radius to be smaller than minimum distance between centers
        if min_dist > 0:
            radii[i] = min(0.1, min_dist / 3.0)
    
    # Ensure all radii are positive
    radii = np.maximum(radii, 0.001)
    
    # Combine into initial configuration
    initial_circles = np.column_stack([positions, radii])
    
    # Stage 3: Refinement with local search
    refined_circles = _refine_with_simulated_annealing(initial_circles)
    
    # Stage 4: Final optimization using numerical method
    try:
        # Create initial guess
        initial_params = _initial_guess(n)
        
        # Define constraints
        def constraint_func(params):
            return _constraint_function(params, n)
        
        # Define bounds for parameters (positions [0,1], radii [0,0.5])
        bounds = []
        for i in range(2*n):  # positions
            bounds.extend([(0.001, 0.999)])
        for i in range(n):  # radii
            bounds.extend([(0.001, 0.499)])
        
        # Perform optimization
        result = minimize(
            fun=_objective_function,
            x0=initial_params,
            args=(n,),
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if result.success:
            # Extract optimized solution
            optimized_positions = result.x[:2*n].reshape((n, 2))
            optimized_radii = result.x[2*n:]
            
            # Ensure valid ranges
            optimized_radii = np.maximum(optimized_radii, 0.001)
            optimized_radii = np.minimum(optimized_radii, 0.499)
            
            circles = np.column_stack([optimized_positions, optimized_radii])
        else:
            # Fall back to refined solution if optimization fails
            circles = refined_circles
            
    except Exception as e:
        # If optimization fails, use the refined solution
        circles = refined_circles
    
    # Final validation and refinement
    # Make sure all circles are valid
    for i in range(n):
        # Ensure containment
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
    
    return circles


# EVOLVE-BLOCK-END
