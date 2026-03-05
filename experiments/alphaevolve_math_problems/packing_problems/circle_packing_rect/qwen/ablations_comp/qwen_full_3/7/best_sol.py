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
    Uses a hybrid approach combining systematic initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Rectangle dimensions: width + height = 2 (perimeter = 4)
    # Try different aspect ratios to find optimal configuration
    best_sum = 0
    best_circles = None
    
    # Test several rectangle aspect ratios that have shown good results
    # Including more extreme ratios from successful approaches
    ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
    
    for ratio in ratios:
        width = 1.0 * ratio
        height = 2.0 - width  # Ensures perimeter = 4
        
        # Skip invalid dimensions
        if width <= 0 or height <= 0:
            continue
            
        # Try multiple initializations for better exploration
        for attempt in range(10):  # More attempts for better exploration
            # Generate initial configuration using systematic approach
            circles = generate_systematic_config(width, height, 21)
            
            # Optimize using constrained optimization
            optimized_circles = optimize_with_constraints(circles, width, height)
            
            current_sum = np.sum(optimized_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized_circles.copy()
    
    # If no good solution found, fallback to a good baseline
    if best_circles is None:
        width, height = 1.0, 1.0
        best_circles = generate_systematic_config(width, height, 21)
        best_circles = optimize_with_constraints(best_circles, width, height)
    
    return best_circles

def generate_systematic_config(width: float, height: float, n: int) -> np.ndarray:
    """
    Generate initial configuration using a systematic approach based on packing theory.
    """
    # Use a grid-like approach with strategic spacing
    circles = []
    
    # Try to place circles in a structured pattern
    rows = max(3, int(np.sqrt(n)))
    cols = max(3, int(np.ceil(n / rows)))
    
    # Calculate spacing
    cell_width = width / cols
    cell_height = height / rows
    
    # Place circles in a structured grid pattern with hexagonal offset for better packing
    placed_count = 0
    for i in range(rows):
        for j in range(cols):
            if placed_count >= n:
                break
                
            # Position in center of cell
            x = (j + 0.5) * cell_width
            y = (i + 0.5) * cell_height
            
            # Apply hexagonal offset for better packing
            if i % 2 == 1:
                x += cell_width * 0.5
                
            # Adjust for better packing near edges with stronger edge constraints
            if i == 0 or i == rows - 1:
                y = max(y, cell_height * 0.2)
                y = min(y, height - cell_height * 0.2)
            if j == 0 or j == cols - 1:
                x = max(x, cell_width * 0.2)
                x = min(x, width - cell_width * 0.2)
            
            # Ensure we stay within bounds
            if x >= 0 and x <= width and y >= 0 and y <= height:
                # Initial radius based on proximity to boundaries
                min_dist_to_boundary = min(x, width - x, y, height - y)
                # Use a fraction of the minimum distance to boundary for radius
                radius = min(min_dist_to_boundary * 0.3, min(cell_width, cell_height) * 0.4)
                
                # Ensure reasonable minimum radius
                radius = max(0.01, min(radius, 0.5))
                
                circles.append([x, y, radius])
                placed_count += 1
                
        if placed_count >= n:
            break
    
    # Fill any remaining positions with strategic placement
    while len(circles) < n:
        # Try to place in a way that maximizes the chance of fitting
        # Start with a random position near the center but with bias towards less crowded areas
        x = random.uniform(width * 0.1, width * 0.9)
        y = random.uniform(height * 0.1, height * 0.9)
        
        # Find the closest existing circle to determine appropriate radius
        min_distance = float('inf')
        for cx, cy, _ in circles:
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            min_distance = min(min_distance, distance)
        
        # Radius should be less than half the minimum distance to nearest circle
        # and also respect boundaries
        max_radius = min(
            min(x, width - x, y, height - y),
            min_distance * 0.3 if min_distance > 0 else width * 0.2
        )
        
        radius = max(0.01, min(max_radius, 0.3))
        
        # Check if this would cause overlap with existing circles
        valid = True
        for cx, cy, cr in circles:
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < (radius + cr) * 0.9:  # Add safety margin
                valid = False
                break
                
        if valid:
            circles.append([x, y, radius])
    
    return np.array(circles)

def optimize_with_constraints(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    Use mathematical optimization with constraints to improve the initial configuration.
    """
    # Create optimization variables: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    n_circles = len(initial_circles)
    initial_vars = []
    for i in range(n_circles):
        initial_vars.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define the optimization problem
    def objective(vars):
        # Maximize sum of radii (minimize negative sum)
        radii = vars[2::3]  # Every third element starting from index 2
        return -sum(radii)
    
    def constraint_func(vars):
        # Constraint function for all pairwise non-overlap conditions
        # Also handle boundary constraints
        constraints = []
        
        # Non-overlap constraints and boundary constraints
        for i in range(n_circles):
            x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
            
            # Boundary constraints (ensure circle is within bounds with a margin)
            constraints.append(x1 - r1)  # x >= r1 (left boundary)
            constraints.append(width - x1 - r1)  # width - x >= r1 (right boundary)
            constraints.append(y1 - r1)  # y >= r1 (bottom boundary)
            constraints.append(height - y1 - r1)  # height - y >= r1 (top boundary)
            
            # Non-overlap with all other circles
            for j in range(i+1, n_circles):
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Constraint: distance >= r1 + r2 (non-overlapping)
                constraints.append(distance - (r1 + r2))
        
        return np.array(constraints)
    
    # Create bounds for variables
    bounds = []
    for i in range(n_circles):
        # x bounds - keep away from boundaries for radius
        bounds.append((0.001, width - 0.001))  
        # y bounds  
        bounds.append((0.001, height - 0.001))
        # r bounds - small but reasonable minimum
        bounds.append((0.001, min(width, height) * 0.4))
    
    # Use scipy's minimize with constraints - SLSQP is more robust for this type of problem
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
            options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-8}
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
