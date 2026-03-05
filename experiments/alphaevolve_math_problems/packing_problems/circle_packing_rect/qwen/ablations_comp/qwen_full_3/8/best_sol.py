# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a sophisticated approach combining mathematical programming with intelligent initialization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Rectangle dimensions: width + height = 2 (perimeter = 4)
    # Use a more systematic approach to explore aspect ratios
    best_sum = 0
    best_circles = None
    
    # Test multiple aspect ratios systematically with more focus on proven good ratios
    ratios = [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]
    
    for ratio in ratios:
        width = 1.0 * ratio
        height = 2.0 - width  # Ensures perimeter = 4
        
        if width <= 0 or height <= 0:
            continue
            
        # Try multiple initializations for robustness
        for attempt in range(3):
            # Generate initial configuration using a hybrid approach
            circles = generate_initial_configuration(width, height, 21)
            
            # Optimize using a more robust optimization approach
            optimized_circles = optimize_with_improved_constraints(circles, width, height)
            
            current_sum = np.sum(optimized_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized_circles.copy()
    
    # If no good solution found, fallback to a proven configuration
    if best_circles is None:
        width, height = 1.0, 1.0
        best_circles = generate_initial_configuration(width, height, 21)
        best_circles = optimize_with_improved_constraints(best_circles, width, height)
    
    return best_circles

def generate_initial_configuration(width: float, height: float, n: int) -> np.ndarray:
    """
    Generate initial configuration using a combination of hexagonal and grid-based approaches.
    """
    # Start with a hexagonal packing pattern as suggested by mathematical theory
    circles = []
    
    # Estimate reasonable average radius based on area
    total_area = width * height
    # Hexagonal packing density is ~0.907
    hex_density = 0.907
    estimated_area = total_area * hex_density
    avg_radius_squared = estimated_area / (math.pi * n)
    avg_radius = math.sqrt(avg_radius_squared) * 0.8
    
    # Try to create a structured hexagonal pattern
    # Determine rows and columns based on aspect ratio
    if width >= height:
        cols = int(width / (avg_radius * 1.5)) + 1
        rows = int(height / (avg_radius * 1.732)) + 1
    else:
        rows = int(height / (avg_radius * 1.732)) + 1
        cols = int(width / (avg_radius * 1.5)) + 1
    
    # Ensure we don't go too far over
    cols = max(1, min(cols, 10))
    rows = max(1, min(rows, 10))
    
    # Create hexagonal pattern
    hex_row_spacing = avg_radius * 1.732  # sqrt(3)
    hex_col_spacing = avg_radius * 1.5
    
    placed_count = 0
    for i in range(rows):
        for j in range(cols):
            if placed_count >= n:
                break
                
            # Hexagonal offset for even/odd rows
            x_offset = 0 if i % 2 == 0 else hex_col_spacing * 0.5
            x = (j + 0.5) * hex_col_spacing + x_offset
            y = (i + 0.5) * hex_row_spacing
            
            # Apply boundary checks
            if (x >= avg_radius and x <= width - avg_radius and 
                y >= avg_radius and y <= height - avg_radius):
                
                # Add slight randomness to avoid perfect symmetry
                x += np.random.uniform(-hex_col_spacing*0.1, hex_col_spacing*0.1)
                y += np.random.uniform(-hex_row_spacing*0.1, hex_row_spacing*0.1)
                
                # Clamp to bounds
                x = max(avg_radius, min(width - avg_radius, x))
                y = max(avg_radius, min(height - avg_radius, y))
                
                # Set radius with some variation to help optimization
                radius = avg_radius * (0.7 + np.random.random() * 0.6)
                radius = min(radius, avg_radius * 1.5)
                
                circles.append([x, y, radius])
                placed_count += 1
                
        if placed_count >= n:
            break
    
    # Fill remaining positions with strategic random placements
    while len(circles) < n:
        # Random placement near center with boundary awareness
        x = np.random.uniform(avg_radius, width - avg_radius)
        y = np.random.uniform(avg_radius, height - avg_radius)
        
        # Find closest existing circle to determine appropriate radius
        min_distance = float('inf')
        for cx, cy, _ in circles:
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            min_distance = min(min_distance, distance)
        
        # Radius based on proximity to boundaries and neighbors
        max_radius = min(
            x, width - x, y, height - y,
            min_distance * 0.4 if min_distance > 0 else width * 0.2
        )
        
        radius = max(0.01, min(max_radius, avg_radius * 1.2))
        
        # Check overlap with existing circles
        valid = True
        for cx, cy, cr in circles:
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < (radius + cr) * 0.95:
                valid = False
                break
                
        if valid:
            circles.append([x, y, radius])
    
    return np.array(circles)

def optimize_with_improved_constraints(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    Use improved mathematical optimization with better constraint handling.
    """
    n_circles = len(initial_circles)
    initial_vars = []
    for i in range(n_circles):
        initial_vars.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    def objective(vars):
        # Maximize sum of radii (minimize negative sum)
        radii = vars[2::3]  # Every third element starting from index 2
        return -sum(radii)
    
    def constraint_func(vars):
        # Improved constraint function that handles all constraints robustly
        constraints = []
        
        # Non-overlap constraints and boundary constraints
        for i in range(n_circles):
            x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
            
            # Boundary constraints (ensure circles stay within rectangle)
            constraints.append(x1 - r1)  # x >= r1 (left boundary)
            constraints.append(width - x1 - r1)  # width - x >= r1 (right boundary)
            constraints.append(y1 - r1)  # y >= r1 (bottom boundary)
            constraints.append(height - y1 - r1)  # height - y >= r1 (top boundary)
            
            # Non-overlap constraints with all other circles
            for j in range(i+1, n_circles):
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                # Use a safe distance calculation to prevent numerical issues
                dx = x2 - x1
                dy = y2 - y1
                distance = math.sqrt(dx*dx + dy*dy)
                # Constraint: distance >= r1 + r2 (non-overlapping)
                # Use a small epsilon to prevent numerical instability
                constraints.append(distance - r1 - r2 - 1e-8)
        
        return np.array(constraints)
    
    # Create bounds for variables
    bounds = []
    for i in range(n_circles):
        # x bounds (leave some margin for radius)
        bounds.append((0.001, width - 0.001))
        # y bounds  
        bounds.append((0.001, height - 0.001))
        # r bounds (ensure positive and reasonable)
        bounds.append((0.001, min(width, height) * 0.49))
    
    # Use multiple optimization approaches for robustness
    try:
        # First try trust-constr which is often more robust for complex constraints
        result = minimize(
            objective,
            initial_vars,
            method='trust-constr',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'verbose': 0}
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
    
    # Fallback to SLSQP if trust-constr fails
    try:
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
            options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6}
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
    
    # If all optimization methods fail, return initial configuration with validation
    return validate_and_fix_initial(initial_circles, width, height)

def validate_and_fix_initial(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    Validate and fix the initial configuration to ensure all constraints are met.
    """
    validated = []
    
    for x, y, r in circles:
        # Ensure within bounds
        max_radius = min(x, width - x, y, height - y)
        r = min(r, max_radius * 0.99)
        r = max(r, 0.001)
        validated.append([x, y, r])
    
    # Apply local refinement to maximize radii
    improved = True
    iterations = 0
    max_iterations = 50
    
    while improved and iterations < max_iterations:
        improved = False
        for i in range(len(validated)):
            x, y, r = validated[i]
            
            # Calculate maximum possible radius at this position
            max_radius = min(x, width - x, y, height - y)
            
            # Try to increase radius slightly with overlap checking
            if r < max_radius:
                test_radius = min(r + 0.005, max_radius)
                
                # Check overlap with all other circles
                valid = True
                for j in range(len(validated)):
                    if i != j:
                        x2, y2, r2 = validated[j]
                        dist = math.sqrt((x - x2)**2 + (y - y2)**2)
                        if dist < test_radius + r2:
                            valid = False
                            break
                
                if valid and test_radius > r:
                    validated[i] = [x, y, test_radius]
                    improved = True
                    
        iterations += 1
    
    return np.array(validated)


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
