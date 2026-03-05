# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import math
from scipy.optimize import minimize


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric initialization and mathematical programming.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Try the most successful ratios from inspiration programs
    best_sum = 0
    best_circles = None
    
    # Test a broader range of ratios that showed success in inspirations
    ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for ratio in ratios:
        width = 2 * ratio / (1 + ratio)
        height = 2 / (1 + ratio)
        
        # Try two different initialization strategies to maximize chances of finding good solution
        # Strategy 1: Systematic placement
        circles1 = initialize_systematic_placement(width, height, 21)
        optimized_circles1 = optimize_mathematical_programming(circles1, width, height)
        current_sum1 = np.sum(optimized_circles1[:, 2])
        
        # Strategy 2: Hexagonal pattern with slight randomness
        circles2 = initialize_hexagonal_pattern(width, height, 21)
        optimized_circles2 = optimize_mathematical_programming(circles2, width, height)
        current_sum2 = np.sum(optimized_circles2[:, 2])
        
        # Select the better of the two strategies
        if current_sum1 > current_sum2:
            current_sum = current_sum1
            optimized_circles = optimized_circles1
        else:
            current_sum = current_sum2
            optimized_circles = optimized_circles2
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
    
    return best_circles


def initialize_systematic_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circle positions using a systematic approach based on geometric packing principles."""
    circles = np.zeros((n, 3))
    
    # For 21 circles, arrange in approximately 4x5 grid with some spacing
    rows = 4
    cols = 5
    
    # Calculate grid spacing
    cell_width = width / cols
    cell_height = height / rows
    
    # Place circles in grid with slight offset to allow for optimization
    for i in range(rows):
        for j in range(cols):
            if len(circles[circles[:, 2] > 0]) >= n:
                break
            # Position in cell
            x = (j + 0.5) * cell_width
            y = (i + 0.5) * cell_height
            
            # Add small random offset to prevent perfect alignment
            x += random.uniform(-cell_width * 0.1, cell_width * 0.1)
            y += random.uniform(-cell_height * 0.1, cell_height * 0.1)
            
            # Keep within bounds
            x = max(cell_width * 0.2, min(width - cell_width * 0.2, x))
            y = max(cell_height * 0.2, min(height - cell_height * 0.2, y))
            
            # Set initial radius based on available space
            max_radius = min(x, width - x, y, height - y) * 0.4
            radius = max(0.01, max_radius * random.uniform(0.6, 0.9))
            
            # Find next empty slot
            idx = len(circles[circles[:, 2] > 0])
            if idx < n:
                circles[idx] = [x, y, radius]
    
    # Fill remaining slots with random placements
    for i in range(len(circles[circles[:, 2] > 0]), n):
        x = random.uniform(0.1, width - 0.1)
        y = random.uniform(0.1, height - 0.1)
        radius = random.uniform(0.01, min(width, height) * 0.1)
        circles[i] = [x, y, radius]
    
    return circles


def initialize_hexagonal_pattern(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circle positions using a hexagonal lattice pattern."""
    circles = np.zeros((n, 3))
    
    # For 21 circles, arrange in roughly 5 rows x 4 columns with offset rows
    rows = 5
    cols = 4
    
    # Calculate spacing based on available space
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)
    
    # Adjust for hexagonal packing efficiency
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    # Place circles in hexagonal pattern
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
                
            # Offset every other row for hexagonal packing
            offset = spacing_x * 0.5 if row % 2 == 1 else 0.0
            
            x = (col + 1) * spacing_x + offset
            y = (row + 1) * spacing_y
            
            # Ensure within bounds with safety margin
            safe_margin = spacing_x * 0.2
            x = max(safe_margin, min(width - safe_margin, x))
            y = max(safe_margin, min(height - safe_margin, y))
            
            # Reasonable initial radius
            r = min(spacing_x, spacing_y) * 0.4
            
            circles[idx] = [x, y, r]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining circles strategically
    np.random.seed(42)
    for i in range(idx, n):
        # Better random positioning
        x = random.uniform(spacing_x * 0.5, width - spacing_x * 0.5)
        y = random.uniform(spacing_y * 0.5, height - spacing_y * 0.5)
        r = random.uniform(0.01, min(spacing_x, spacing_y) * 0.3)
        circles[i] = [x, y, r]
    
    return circles


def optimize_mathematical_programming(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    Optimize using a mathematical programming approach with explicit constraints.
    """
    # Create a structured optimization problem
    n = len(initial_circles)
    
    # Define parameter bounds: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((initial_circles[i, 2], width - initial_circles[i, 2]))
        # y bounds  
        bounds.append((initial_circles[i, 2], height - initial_circles[i, 2]))
        # r bounds
        bounds.append((0.001, min(width, height) * 0.5))
    
    # Flatten initial solution
    initial_solution = initial_circles.flatten()
    
    # Define the optimization objective
    def objective(params):
        # Reshape to circles array
        circles = params.reshape(n, 3)
        # Maximize sum of radii (minimize negative sum)
        return -np.sum(circles[:, 2])
    
    # Define constraints
    def constraint_function(params):
        circles = params.reshape(n, 3)
        
        constraints = []
        
        # Boundary constraints: each circle must be fully inside rectangle
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([
                x - r,           # Left boundary
                width - x - r,   # Right boundary  
                y - r,           # Bottom boundary
                height - y - r   # Top boundary
            ])
        
        # Non-overlap constraints: distance >= sum of radii
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # Constraint: distance >= r1 + r2 (or equivalently: distance - r1 - r2 >= 0)
                constraints.append(distance - r1 - r2)
        
        return np.array(constraints)
    
    # Create constraint dictionary
    constraints = {'type': 'ineq', 'fun': constraint_function}
    
    # Run optimization with more robust settings
    try:
        # Try multiple optimization methods for better results
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_objective = float('inf')
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_solution,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                
                if result.success and result.fun < best_objective:
                    best_objective = result.fun
                    best_result = result
            except:
                continue
        
        if best_result is not None and best_result.success:
            optimized_circles = best_result.x.reshape(n, 3)
            # Ensure all radii are positive
            optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], 0.001)
            return optimized_circles
    except Exception as e:
        # Fallback to original if optimization fails
        pass
    
    return initial_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
