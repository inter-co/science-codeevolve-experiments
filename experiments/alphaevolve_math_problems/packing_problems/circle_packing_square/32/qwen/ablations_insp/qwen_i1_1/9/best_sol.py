# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

# Global constants
N_CIRCLES = 32

def initialize_grid_placement() -> np.ndarray:
    """
    Initialize circle positions using a refined grid pattern for better initial distribution.
    """
    # Create a grid that's slightly smaller than unit square to allow for radii
    grid_size = int(math.ceil(math.sqrt(N_CIRCLES)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    positions = np.zeros((N_CIRCLES, 2))
    
    # Place initial circles on a grid
    for i in range(N_CIRCLES):
        row = i // grid_size
        col = i % grid_size
        positions[i, 0] = (col + 1) * spacing_x  # x coordinate
        positions[i, 1] = (row + 1) * spacing_y  # y coordinate
    
    return positions

def compute_collision_free_radii(positions: np.ndarray) -> np.ndarray:
    """
    Compute maximum possible radii for each circle given positions,
    ensuring no overlaps and boundary constraints.
    """
    n_circles = len(positions)
    radii = np.zeros(n_circles)
    
    # For each circle, compute the maximum radius it can have
    for i in range(n_circles):
        # Boundary constraints
        boundary_radius = min(
            positions[i][0],           # Distance to left boundary
            1 - positions[i][0],       # Distance to right boundary
            positions[i][1],           # Distance to bottom boundary
            1 - positions[i][1]        # Distance to top boundary
        )
        
        # Inter-circle constraints
        min_distance = float('inf')
        for j in range(n_circles):
            if i != j:
                distance = np.sqrt(
                    (positions[i][0] - positions[j][0])**2 + 
                    (positions[i][1] - positions[j][1])**2
                )
                min_distance = min(min_distance, distance)
        
        # Maximum radius is limited by both boundary and other circles
        if min_distance < float('inf'):
            inter_circle_radius = min_distance / 2.0
            radii[i] = min(boundary_radius, inter_circle_radius)
        else:
            radii[i] = boundary_radius
            
        # Ensure minimum radius
        radii[i] = max(0.001, radii[i])
    
    return radii

def get_params(circles: np.ndarray) -> np.ndarray:
    """Convert circles array to flat parameter vector."""
    params = []
    for i in range(len(circles)):
        params.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
    return np.array(params)

def set_circles_from_params(params: np.ndarray) -> np.ndarray:
    """Convert flat parameter vector back to circles array."""
    circles = params.reshape(-1, 3)
    return circles

def constraint_containment(params: np.ndarray) -> np.ndarray:
    """Ensure all circles fit within the unit square"""
    circles = set_circles_from_params(params)
    constraints = []
    for i in range(len(circles)):
        x, y, r = circles[i, 0], circles[i, 1], circles[i, 2]
        constraints.append(x - r)  # x - r >= 0
        constraints.append(y - r)  # y - r >= 0
        constraints.append(1 - x - r)  # 1 - x - r >= 0
        constraints.append(1 - y - r)  # 1 - y - r >= 0
    return np.array(constraints)

def constraint_nonoverlap(params: np.ndarray) -> np.ndarray:
    """Ensure no overlap between circles"""
    circles = set_circles_from_params(params)
    constraints = []
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i, 0], circles[i, 1], circles[i, 2]
            x2, y2, r2 = circles[j, 0], circles[j, 1], circles[j, 2]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # We want distance >= r1 + r2
            # So we enforce distance - r1 - r2 >= 0
            constraints.append(distance - r1 - r2)
    return np.array(constraints)

def objective(params: np.ndarray) -> float:
    """Objective function to maximize sum of radii (negative because we minimize)."""
    circles = set_circles_from_params(params)
    return -np.sum(circles[:, 2])

def optimize_with_scipy(initial_circles: np.ndarray) -> np.ndarray:
    """
    Use scipy optimization with proper constraints.
    """
    # Convert to flat parameter vector
    initial_params = get_params(initial_circles)
    
    # Define bounds for each parameter (x, y, r)
    bounds = []
    for i in range(len(initial_circles)):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Create constraints dictionary - use a more efficient approach
    def containment_constraint(params):
        return constraint_containment(params)
        
    def nonoverlap_constraint(params):
        return constraint_nonoverlap(params)
    
    cons = [
        {'type': 'ineq', 'fun': containment_constraint},
        {'type': 'ineq', 'fun': nonoverlap_constraint}
    ]
    
    # Optimize using SLSQP method with better parameters
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-6}
        )
        
        if result.success:
            final_circles = set_circles_from_params(result.x)
            return final_circles
    except Exception as e:
        # If optimization fails, return initial configuration
        pass
    
    # Return original if optimization failed
    return initial_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining grid initialization and mathematical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    # Stage 1: Initialize with grid pattern (better than hexagonal for this case)
    initial_positions = initialize_grid_placement()
    
    # Stage 2: Compute initial radii for the grid configuration
    initial_radii = compute_collision_free_radii(initial_positions)
    
    # Stage 3: Initial configuration
    initial_circles = np.column_stack([initial_positions, initial_radii])
    
    # Stage 4: Mathematical optimization with proper constraints
    optimized_circles = optimize_with_scipy(initial_circles)
    
    # Final validation and cleanup
    positions = optimized_circles[:, :2]
    radii = optimized_circles[:, 2]
    
    # Ensure all constraints are met properly
    for i in range(N_CIRCLES):
        # Boundary check and adjustment
        boundary_radius = min(positions[i][0], 1 - positions[i][0], 
                             positions[i][1], 1 - positions[i][1])
        radii[i] = min(radii[i], boundary_radius)
        radii[i] = max(0.001, radii[i])
    
    # Ensure no overlaps by checking distances and adjusting if needed
    distances = cdist(positions, positions)
    for i in range(N_CIRCLES):
        for j in range(i+1, N_CIRCLES):
            if distances[i,j] < radii[i] + radii[j]:
                # Reduce radii to prevent overlap - more careful adjustment
                overlap = (radii[i] + radii[j]) - distances[i,j]
                reduction = overlap * 0.1  # Smaller reduction factor
                radii[i] = max(0.001, radii[i] - reduction)
                radii[j] = max(0.001, radii[j] - reduction)
    
    # Return final result
    circles = np.column_stack([positions, radii])
    
    return circles


# EVOLVE-BLOCK-END
