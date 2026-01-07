# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time

# Global constants for optimization
N_CIRCLES = 26
TIME_LIMIT = 170  # seconds

def initialize_circles_hexagonal(n: int) -> np.ndarray:
    """Initialize circles using a hexagonal packing approximation for better density"""
    circles = np.zeros((n, 3))
    
    # Try to place circles in a hexagonal pattern
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Adjust for better packing
    spacing_x = 0.9 / (cols + 1)
    spacing_y = 0.9 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            offset = 0.5 if i % 2 == 1 else 0.0
            x = 0.05 + (j + 1) * spacing_x + offset * spacing_x * 0.5
            y = 0.05 + (i + 1) * spacing_y
            
            # Initial radius - start with reasonable value
            r = min(spacing_x, spacing_y) * 0.4
            
            # Ensure within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining slots with random positions
    for i in range(idx, n):
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        r = random.uniform(0.01, 0.1)
        circles[i] = [x, y, r]
    
    # Refine initial placement to maximize radii while respecting constraints
    refine_initial_placement(circles)
    
    return circles

def refine_initial_placement(circles: np.ndarray) -> None:
    """Improve initial placement by maximizing radii while respecting constraints"""
    n = len(circles)
    max_attempts = 1000
    
    for attempt in range(max_attempts):
        improved = False
        for i in range(n):
            # Try to increase radius of circle i
            current_radius = circles[i][2]
            # Calculate maximum possible radius based on boundaries
            max_radius = min(
                circles[i][0], 1 - circles[i][0],
                circles[i][1], 1 - circles[i][1]
            )
            
            # Check overlap with other circles
            for j in range(n):
                if i != j:
                    dist = np.sqrt((circles[i][0] - circles[j][0])**2 + 
                                 (circles[i][1] - circles[j][1])**2)
                    max_radius = min(max_radius, dist - circles[j][2])
            
            # Increase radius if possible
            if max_radius > current_radius + 1e-6:
                circles[i][2] = max_radius
                improved = True
                
        if not improved:
            break

def check_constraints(circles: np.ndarray) -> bool:
    """Check if all circles satisfy containment and non-overlap constraints"""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check non-overlap constraints efficiently using distance matrix
    if n > 1:
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        radii = circles[:, 2]
        
        # Check all pairs for overlap
        for i in range(n):
            for j in range(i+1, n):
                if distances[i,j] < (radii[i] + radii[j]) - 1e-8:
                    return False
    
    return True

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate the sum of all radii"""
    return np.sum(circles[:, 2])

def create_optimization_bounds(n: int) -> List[Tuple[float, float]]:
    """Create bounds for scipy optimization"""
    bounds = []
    for i in range(n):
        # Bounds: [x_min, x_max], [y_min, y_max], [r_min, r_max]
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    return bounds

def create_optimization_constraints(n: int) -> List[dict]:
    """Create constraint functions for scipy optimization"""
    constraints = []
    
    # Boundary constraints for each circle - more precise handling
    def boundary_constraint(x_flat):
        # Check that all circles are within bounds
        constraints_list = []
        for i in range(n):
            x, y, r = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
            # Ensure circle stays within unit square with radius r
            constraints_list.extend([
                x - r,      # x - r >= 0 (left boundary)  
                1 - x - r,  # 1 - x - r >= 0 (right boundary)
                y - r,      # y - r >= 0 (bottom boundary)
                1 - y - r   # 1 - y - r >= 0 (top boundary)
            ])
        return np.array(constraints_list)
    
    # Non-overlap constraints for all pairs - using a more robust approach
    def overlap_constraint(x_flat):
        # Check that no circles overlap
        constraints_list = []
        for i in range(n):
            for j in range(i+1, n):
                x_i, y_i, r_i = x_flat[3*i], x_flat[3*i+1], x_flat[3*i+2]
                x_j, y_j, r_j = x_flat[3*j], x_flat[3*j+1], x_flat[3*j+2]
                distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                # We want distance >= r_i + r_j, so we enforce: distance - (r_i + r_j) >= 0
                # Adding a small epsilon to prevent numerical issues
                constraints_list.append(distance - (r_i + r_j) + 1e-8)
        return np.array(constraints_list)
    
    # Create constraint dictionaries
    constraints.append({'type': 'ineq', 'fun': boundary_constraint})
    constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    return constraints

def optimize_with_scipy(initial_circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Optimize using scipy's minimize with constraints"""
    n = len(initial_circles)
    
    # Flatten the initial circles array for optimization
    initial_flat = []
    for i in range(n):
        initial_flat.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define objective function (minimize negative sum of radii)
    def objective(x_flat):
        # Extract radii
        radii = x_flat[2::3]  # Every third element starting from index 2
        return -np.sum(radii)  # Negative because we want to maximize
    
    # Create constraints
    constraints = create_optimization_constraints(n)
    bounds = create_optimization_bounds(n)
    
    # Run optimization with multiple attempts for better results
    best_result = None
    best_value = -float('inf')
    
    # Try different optimization approaches with different parameters
    solver_configs = [
        {'method': 'SLSQP', 'options': {'maxiter': max_iter, 'ftol': 1e-8, 'eps': 1e-6}},
        {'method': 'trust-constr', 'options': {'maxiter': max_iter, 'ftol': 1e-8}},
        {'method': 'L-BFGS-B', 'options': {'maxiter': max_iter, 'ftol': 1e-8}}
    ]
    
    # Try multiple random restarts for better exploration
    for restart in range(3):
        try:
            # Add small random perturbations to initial guess for diversity
            perturbed_initial = initial_flat.copy()
            for i in range(len(perturbed_initial)):
                if i % 3 < 2:  # x and y coordinates
                    perturbed_initial[i] += np.random.normal(0, 0.001)
                else:  # radius
                    perturbed_initial[i] += np.random.normal(0, 0.0005)
            
            for solver_config in solver_configs:
                result = minimize(
                    objective,
                    perturbed_initial,
                    **solver_config,
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    # Convert back to circle format
                    optimized_circles = np.zeros((n, 3))
                    for i in range(n):
                        optimized_circles[i] = [
                            result.x[3*i],     # x coordinate
                            result.x[3*i+1],   # y coordinate
                            result.x[3*i+2]    # radius
                        ]
                    
                    # Check if this is better
                    current_value = calculate_radius_sum(optimized_circles)
                    if current_value > best_value:
                        best_value = current_value
                        best_result = optimized_circles.copy()
                        
        except Exception:
            continue
    
    # Return best result or fallback to initial
    if best_result is not None:
        return best_result
    else:
        return initial_circles

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    # Try multiple initialization strategies and pick the best
    best_circles = None
    best_sum = -float('inf')
    
    # Strategy 1: Hexagonal initialization with refinement
    circles1 = initialize_circles_hexagonal(26)
    circles1 = optimize_with_scipy(circles1, 500)
    
    # Strategy 2: Another hexagonal initialization with different randomness
    circles2 = initialize_circles_hexagonal(26)
    circles2 = optimize_with_scipy(circles2, 500)
    
    # Strategy 3: Fresh start with better optimization
    circles3 = initialize_circles_hexagonal(26)
    circles3 = optimize_with_scipy(circles3, 1000)
    
    # Pick the better of the three strategies
    sum1 = calculate_radius_sum(circles1)
    sum2 = calculate_radius_sum(circles2)
    sum3 = calculate_radius_sum(circles3)
    
    if sum1 >= sum2 and sum1 >= sum3:
        circles = circles1
    elif sum2 >= sum3:
        circles = circles2
    else:
        circles = circles3
    
    # Final optimization pass with high iteration count
    circles = optimize_with_scipy(circles, 1000)
    
    # Final validation and constraint check
    if not check_constraints(circles):
        # If constraints violated, try a fresh start with better initialization
        circles = initialize_circles_hexagonal(26)
        circles = optimize_with_scipy(circles, 1000)
    
    return circles


# EVOLVE-BLOCK-END
