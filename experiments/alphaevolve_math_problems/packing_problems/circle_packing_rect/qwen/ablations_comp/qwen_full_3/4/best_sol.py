# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining hexagonal initialization with advanced mathematical optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Rectangle dimensions: width + height = 2 (perimeter = 4)
    # Try different aspect ratios to find optimal configuration
    best_sum = 0
    best_circles = None
    
    # Test several rectangle aspect ratios - including more extreme ratios for better exploration
    ratios = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0]
    
    for ratio in ratios:
        width = 1.0 * ratio
        height = 2.0 - width  # Ensures perimeter = 4
        
        # Skip invalid dimensions
        if width <= 0 or height <= 0:
            continue
            
        # Try multiple initializations for better exploration
        for attempt in range(10):  # Increased attempts for better exploration
            # Generate initial configuration using hexagonal pattern for better starting point
            circles = generate_hexagonal_initialization(width, height, 21)
            
            # Apply local optimization to refine the solution
            optimized_circles = optimize_with_mathematical_programming(circles, width, height)
            
            current_sum = np.sum(optimized_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized_circles.copy()
    
    # If no good solution found, fallback to a good baseline
    if best_circles is None:
        width, height = 1.0, 1.0
        best_circles = generate_hexagonal_initialization(width, height, 21)
        best_circles = optimize_with_mathematical_programming(best_circles, width, height)
    
    return best_circles

def generate_hexagonal_initialization(width: float, height: float, n: int) -> np.ndarray:
    """
    Generate initial configuration using hexagonal packing pattern for better starting point.
    """
    circles = np.zeros((n, 3))
    
    # Calculate grid parameters for hexagonal packing
    total_area = width * height
    avg_radius = math.sqrt(total_area / (math.pi * n))
    
    # Hexagonal packing parameters
    hex_radius = avg_radius * 0.9  # Slightly smaller for better packing
    hex_width = hex_radius * 2
    hex_height = hex_radius * math.sqrt(3)
    
    # Create hexagonal grid
    rows = int(height / hex_height) + 2
    cols = int(width / hex_width) + 2
    
    # Fill the grid
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row
            x_offset = (i % 2) * (hex_width / 2)
            x = x_offset + j * hex_width + hex_radius
            y = i * hex_height + hex_radius
            
            # Ensure within bounds
            if x >= hex_radius and x <= width - hex_radius and \
               y >= hex_radius and y <= height - hex_radius:
                circles[idx] = [x, y, hex_radius]
                idx += 1
                
        if idx >= n:
            break
    
    # Fill remaining circles with better distributed positions
    for i in range(idx, n):
        # Use more systematic random placement with better distribution
        x = np.random.uniform(hex_radius, width - hex_radius)
        y = np.random.uniform(hex_radius, height - hex_radius)
        # Use slightly smaller initial radii for more flexibility
        circles[i] = [x, y, hex_radius * 0.7] 
        
    return circles

def optimize_with_mathematical_programming(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    Use mathematical optimization with better constraint handling and penalty functions.
    """
    # Create optimization variables: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    n_circles = len(initial_circles)
    initial_vars = []
    for i in range(n_circles):
        initial_vars.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define the optimization problem with proper penalty functions
    def objective(vars):
        # Maximize sum of radii (minimize negative sum)
        radii = vars[2::3]  # Every third element starting from index 2
        return -sum(radii)
    
    def constraint_func(vars):
        # Constraint function for all pairwise non-overlap conditions
        # Also handle boundary constraints with penalties
        constraints = []
        
        # Non-overlap constraints
        for i in range(n_circles):
            x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
            
            # Boundary constraints (penalized for violations)
            constraints.append(x1 - r1)  # x >= r1 (left boundary)
            constraints.append(width - x1 - r1)  # width - x >= r1 (right boundary)
            constraints.append(y1 - r1)  # y >= r1 (bottom boundary)
            constraints.append(height - y1 - r1)  # height - y >= r1 (top boundary)
            
            # Non-overlap with all other circles
            for j in range(i+1, n_circles):
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Constraint: distance >= r1 + r2 (non-overlapping)
                constraints.append(distance - r1 - r2)
        
        return np.array(constraints)
    
    # Create bounds for variables with tighter constraints
    bounds = []
    for i in range(n_circles):
        # x bounds
        bounds.append((0.001, width - 0.001))  # Need to leave some room for radius
        # y bounds  
        bounds.append((0.001, height - 0.001))
        # r bounds - more conservative to avoid numerical issues
        bounds.append((0.001, min(width, height) * 0.45))
    
    # Use scipy's minimize with constraints
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
            options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-8}
        )
        
        if result.success:
            # Extract optimized values
            optimized_vars = result.x
            optimized_circles = []
            for i in range(n_circles):
                x = optimized_vars[3*i]
                y = optimized_vars[3*i+1]
                r = optimized_vars[3*i+2]
                optimized_circles.append([x, y, r])
            return np.array(optimized_circles)
    except Exception:
        pass
    
    # If optimization fails, return the initial configuration
    return initial_circles.copy()


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
