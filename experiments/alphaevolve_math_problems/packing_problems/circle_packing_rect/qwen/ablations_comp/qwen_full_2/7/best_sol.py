# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach: multiple rectangle aspect ratios + constrained optimization with multiple restarts.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Try several aspect ratios to find optimal configuration
    # Focus on narrow aspect ratios that typically perform better for circle packing
    aspect_ratios = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    
    best_sum = 0
    best_circles = None
    
    for ratio in aspect_ratios:
        width = 2 * ratio / (1 + ratio)  # Ensuring perimeter = 4
        height = 2 * 1 / (1 + ratio)
        
        # Initialize using hexagonal grid pattern
        def initialize_hexagonal():
            circles = []
            # Estimate initial radius based on area
            total_area = width * height
            circle_area = total_area / 21 * 0.8  # Leave margin for packing
            radius_estimate = math.sqrt(circle_area / math.pi)
            
            # Create hexagonal grid pattern
            rows = max(3, int(math.sqrt(21)) + 1)
            cols = max(3, int(21 / rows) + 1)
            
            # Hexagonal spacing
            spacing_x = radius_estimate * 2.0
            spacing_y = radius_estimate * math.sqrt(3)
            
            # Fill grid with circles
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= 21:
                        break
                        
                    # Offset odd rows for hexagonal packing
                    x_offset = (i % 2) * spacing_x / 2
                    x = spacing_x * j + x_offset + radius_estimate
                    y = spacing_y * i + radius_estimate
                    
                    # Check bounds
                    if (radius_estimate <= x <= width - radius_estimate and 
                        radius_estimate <= y <= height - radius_estimate):
                        circles.append([x, y, radius_estimate])
                
                if len(circles) >= 21:
                    break
            
            # Fill remaining positions with random placement
            while len(circles) < 21:
                x = random.uniform(radius_estimate, width - radius_estimate)
                y = random.uniform(radius_estimate, height - radius_estimate)
                circles.append([x, y, radius_estimate])
                
            return np.array(circles)
        
        # Objective function: maximize sum of radii (minimize negative sum)
        def objective(params):
            # Reshape params to circles array [x1, y1, r1, x2, y2, r2, ...]
            reshaped = params.reshape(-1, 3)
            return -np.sum(reshaped[:, 2])  # Negative because we minimize
        
        # Constraint function for scipy optimization
        def constraint_func(params):
            # params: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
            circles = params.reshape(-1, 3)
            
            constraints = []
            
            # Boundary constraints: each circle must stay within rectangle
            for i in range(21):
                x, y, r = circles[i]
                # Circle must be fully within bounds
                constraints.extend([
                    x - r,                    # left boundary
                    width - x - r,           # right boundary  
                    y - r,                   # bottom boundary
                    height - y - r           # top boundary
                ])
            
            # Overlap constraints: distance between centers >= sum of radii
            for i in range(21):
                for j in range(i+1, 21):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    # Constraint: distance >= r1 + r2 (we negate for inequality constraint)
                    constraints.append(distance - (r1 + r2))
            
            return np.array(constraints)
        
        # Bounds for optimization: (lower_bound, upper_bound) for each parameter
        bounds = []
        for i in range(21):
            # x bounds
            bounds.append((0.001, width - 0.001))
            # y bounds  
            bounds.append((0.001, height - 0.001))
            # r bounds (not too large)
            bounds.append((0.001, min(width, height)/2))
        
        # Try multiple optimization attempts with different initializations
        for attempt in range(5):  # Increase attempts to 5 for better chance
            # Initialize
            circles = initialize_hexagonal()
            initial_params = circles.flatten()
            
            # Add small random perturbations to initial positions for variety
            if attempt > 0:
                for i in range(0, len(initial_params), 3):
                    # Reduce perturbation magnitude for stability
                    initial_params[i] += random.uniform(-0.02, 0.02)  # x
                    initial_params[i+1] += random.uniform(-0.02, 0.02)  # y
                    # Keep radius positive
                    initial_params[i+2] = max(0.001, initial_params[i+2] + random.uniform(-0.005, 0.005))
            
            try:
                # Use SLSQP method which works well with constraints
                result = minimize(
                    objective,
                    initial_params,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 1200, 'ftol': 1e-6, 'eps': 1e-6}
                )
                
                if result.success:
                    # Extract optimized circles
                    optimized_circles = result.x.reshape(-1, 3)
                    current_sum = np.sum(optimized_circles[:, 2])
                    
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_circles = optimized_circles
                        
            except Exception as e:
                continue
    
    # If no good solution found, return a reasonable configuration
    if best_circles is None:
        # Fallback to basic hexagonal packing with some refinement
        width, height = 1.5, 0.5  # Standard aspect ratio
        circles = []
        radius_estimate = 0.15  # Rough estimate
        
        # Simple hexagonal arrangement
        rows = 5
        cols = 5
        spacing_x = radius_estimate * 2.0
        spacing_y = radius_estimate * math.sqrt(3)
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= 21:
                    break
                x_offset = (i % 2) * spacing_x / 2
                x = spacing_x * j + x_offset + radius_estimate
                y = spacing_y * i + radius_estimate
                if (radius_estimate <= x <= width - radius_estimate and 
                    radius_estimate <= y <= height - radius_estimate):
                    circles.append([x, y, radius_estimate])
        
        # Fill remaining positions
        while len(circles) < 21:
            x = random.uniform(radius_estimate, width - radius_estimate)
            y = random.uniform(radius_estimate, height - radius_estimate)
            circles.append([x, y, radius_estimate])
            
        best_circles = np.array(circles[:21])
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
