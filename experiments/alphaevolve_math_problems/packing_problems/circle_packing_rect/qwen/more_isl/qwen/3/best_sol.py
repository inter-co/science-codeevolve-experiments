# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
import time
from typing import Tuple


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach with smart initialization, multiple optimization strategies, and aspect ratio optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 21
    
    # Try multiple aspect ratios and initialization strategies
    best_sum = 0
    best_circles = None
    best_ratio = 1.0
    
    # Focus on promising aspect ratios based on previous experience
    ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
    
    # Track time to respect 60-second limit
    start_time = time.time()
    
    for ratio in ratios:
        if time.time() - start_time > 55:  # Leave buffer for final processing
            break
            
        width = 2 / (1 + ratio)  # width + height = 2, width/height = ratio
        height = 2 / (1 + 1/ratio)
        
        # Try multiple initialization strategies
        init_strategies = [
            lambda w, h, num: initialize_hexagonal(w, h, num),
            lambda w, h, num: initialize_grid(w, h, num),
            lambda w, h, num: initialize_random(w, h, num)
        ]
        
        for init_strategy in init_strategies:
            if time.time() - start_time > 55:  # Leave buffer
                break
                
            # Try multiple random starts for same initialization
            for attempt in range(5):  # More attempts for better exploration
                try:
                    circles = init_strategy(width, height, n)
                    
                    # Try multiple optimization approaches with more iterations
                    result = optimize_with_multiple_methods(circles, width, height)
                    
                    if result is not None:
                        total_radius = np.sum(result[:, 2])
                        if total_radius > best_sum:
                            best_sum = total_radius
                            best_circles = result.copy()
                            best_ratio = ratio
                            
                except Exception as e:
                    continue  # Skip failed attempts
    
    # Final refinement with best found configuration
    if best_circles is not None and best_sum > 0:
        width = 2 / (1 + best_ratio)
        height = 2 / (1 + 1/best_ratio)
        final_result = refine_final_configuration(best_circles, width, height)
        if final_result is not None:
            return final_result
    
    # Fallback to a robust initialization
    width = 1.2
    height = 0.8
    circles = initialize_hexagonal(width, height, n)
    return optimize_with_multiple_methods(circles, width, height)


def initialize_hexagonal(width, height, n):
    """Initialize circles using a hexagonal packing pattern - improved version"""
    circles = np.zeros((n, 3))
    
    # Better estimate of radius based on area
    area_per_circle = (width * height) / n
    estimated_radius = np.sqrt(area_per_circle / np.pi) * 0.9  # Slightly smaller for better packing
    
    # Determine grid layout for hexagonal packing
    cols = max(1, int(np.sqrt(n * 1.2)))  # Slightly adjusted
    rows = int(np.ceil(n / cols))
    
    # Adjust for better packing
    if cols * rows < n:
        cols += 1
    
    # Hexagonal packing parameters
    row_spacing = estimated_radius * 2 * 0.866  # sqrt(3)/2
    col_spacing = estimated_radius * 1.5
    
    # Generate hexagonal pattern
    circle_idx = 0
    for i in range(rows):
        y_offset = i * row_spacing
        for j in range(cols):
            if circle_idx >= n:
                break
            x_offset = j * col_spacing + (i % 2) * (col_spacing / 2)
            
            # Center the pattern within the rectangle
            x = x_offset + (width - col_spacing * (cols - 1)) / 2
            y = y_offset + (height - row_spacing * (rows - 1)) / 2
            
            # Ensure we're within bounds
            if (x >= estimated_radius and x <= width - estimated_radius and 
                y >= estimated_radius and y <= height - estimated_radius):
                circles[circle_idx] = [x, y, estimated_radius]
                circle_idx += 1
        if circle_idx >= n:
            break
    
    # Fill remaining circles with small radii if needed
    for i in range(circle_idx, n):
        circles[i] = [
            width/2 + np.random.uniform(-0.15, 0.15),
            height/2 + np.random.uniform(-0.15, 0.15),
            estimated_radius * 0.25
        ]
    
    return circles


def initialize_grid(width, height, n):
    """Initialize circles using a regular grid pattern"""
    circles = np.zeros((n, 3))
    
    # Create a grid layout - optimized for 21 circles
    rows = int(np.ceil(np.sqrt(n * 1.1)))  # Slightly adjusted
    cols = int(np.ceil(n / rows))
    
    # Adjust to fit exactly n circles
    actual_rows = min(rows, int(np.ceil(n / cols)))
    actual_cols = int(np.ceil(n / actual_rows))
    
    # Calculate spacing
    cell_width = width / actual_cols
    cell_height = height / actual_rows
    cell_size = min(cell_width, cell_height) * 0.9  # Slightly smaller for better packing
    
    # Place circles in grid
    idx = 0
    for i in range(actual_rows):
        for j in range(actual_cols):
            if idx >= n:
                break
            x = (j + 0.5) * cell_width
            y = (i + 0.5) * cell_height
            r = cell_size * 0.4
            
            # Ensure within bounds
            x = max(r, min(width - r, x))
            y = max(r, min(height - r, y))
            
            circles[idx] = [x, y, r]
            idx += 1
    
    return circles


def initialize_random(width, height, n):
    """Initialize circles with random placement"""
    circles = np.zeros((n, 3))
    max_radius = min(width, height) * 0.18  # Slightly larger max radius
    
    for i in range(n):
        # Random position with minimum distance from edges
        x = np.random.uniform(max_radius, width - max_radius)
        y = np.random.uniform(max_radius, height - max_radius)
        r = np.random.uniform(max_radius * 0.25, max_radius * 0.75)
        circles[i] = [x, y, r]
    
    return circles


def optimize_with_multiple_methods(initial_circles, width, height):
    """Try multiple optimization methods with enhanced parameters"""
    best_result = None
    best_sum = 0
    
    # Method 1: Trust-Constr optimization (often better for constrained problems)
    try:
        tc_result = optimize_trust_constr(initial_circles, width, height)
        if tc_result is not None:
            tc_sum = np.sum(tc_result[:, 2])
            if tc_sum > best_sum:
                best_sum = tc_sum
                best_result = tc_result
    except:
        pass
    
    # Method 2: SLSQP optimization with enhanced settings
    try:
        sqp_result = optimize_sqp_enhanced(initial_circles, width, height)
        if sqp_result is not None:
            sqp_sum = np.sum(sqp_result[:, 2])
            if sqp_sum > best_sum:
                best_sum = sqp_sum
                best_result = sqp_result
    except:
        pass
    
    # If no methods worked, return the initial configuration
    if best_result is None:
        return initial_circles
    
    return best_result


def optimize_trust_constr(initial_circles, width, height):
    """Use trust-constr optimization for better constrained problems"""
    n = len(initial_circles)
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        total_radius = 0
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
            total_radius += r
        
        # Return negative because we want to maximize
        return -total_radius
    
    # Constraint function: ensure no overlaps and all circles stay within bounds
    def constraint_func(params):
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
        
        positions = np.array(positions)
        radii = np.array(radii)
        
        constraints = []
        
        # Non-overlap constraints: distance >= sum of radii
        distances = cdist(positions, positions)
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                constraints.append(dist - min_dist)  # Should be >= 0
        
        # Boundary constraints: all circles must be within rectangle
        for i in range(n):
            x, y, r = positions[i][0], positions[i][1], radii[i]
            # Circle center must be at least radius away from edges
            constraints.append(x - r)  # left bound
            constraints.append(width - x - r)  # right bound
            constraints.append(y - r)  # bottom bound
            constraints.append(height - y - r)  # top bound
        
        return np.array(constraints)
    
    # Set up bounds: [x_min, x_max, y_min, y_max, r_min, r_max] for each circle
    bounds = []
    for i in range(n):
        # x bounds: [radius, width-radius]
        bounds.extend([(1e-6, width - 1e-6)])
        # y bounds: [radius, height-radius] 
        bounds.extend([(1e-6, height - 1e-6)])
        # r bounds: [1e-6, min(width, height)/2]
        bounds.extend([(1e-6, min(width, height)/2)])
    
    # Initial guess from our configuration
    initial_guess = []
    for i in range(n):
        x, y, r = initial_circles[i]
        initial_guess.extend([x, y, r])
    
    # Apply optimization with bounds and constraints
    try:
        # Use trust-constr method which is often better for constrained problems
        result = minimize(
            objective,
            initial_guess,
            method='trust-constr',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result.success:
            # Extract final solution
            final_positions = []
            final_radii = []
            for i in range(n):
                idx = i * 3
                x, y, r = result.x[idx], result.x[idx+1], result.x[idx+2]
                final_positions.append([x, y])
                final_radii.append(r)
            
            # Update circles array
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_positions[i][0], final_positions[i][1], final_radii[i]]
            return circles
        else:
            return initial_circles
            
    except Exception as e:
        # If optimization fails for any reason, return the initial arrangement
        return initial_circles


def optimize_sqp_enhanced(initial_circles, width, height):
    """Use SLSQP for local optimization with enhanced settings"""
    n = len(initial_circles)
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        total_radius = 0
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
            total_radius += r
        
        # Return negative because we want to maximize
        return -total_radius
    
    # Constraint function: ensure no overlaps and all circles stay within bounds
    def constraint_func(params):
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
        
        positions = np.array(positions)
        radii = np.array(radii)
        
        constraints = []
        
        # Non-overlap constraints: distance >= sum of radii
        distances = cdist(positions, positions)
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                constraints.append(dist - min_dist)  # Should be >= 0
        
        # Boundary constraints: all circles must be within rectangle
        for i in range(n):
            x, y, r = positions[i][0], positions[i][1], radii[i]
            # Circle center must be at least radius away from edges
            constraints.append(x - r)  # left bound
            constraints.append(width - x - r)  # right bound
            constraints.append(y - r)  # bottom bound
            constraints.append(height - y - r)  # top bound
        
        return np.array(constraints)
    
    # Set up bounds: [x_min, x_max, y_min, y_max, r_min, r_max] for each circle
    bounds = []
    for i in range(n):
        # x bounds: [radius, width-radius]
        bounds.extend([(1e-6, width - 1e-6)])
        # y bounds: [radius, height-radius] 
        bounds.extend([(1e-6, height - 1e-6)])
        # r bounds: [1e-6, min(width, height)/2]
        bounds.extend([(1e-6, min(width, height)/2)])
    
    # Initial guess from our configuration
    initial_guess = []
    for i in range(n):
        x, y, r = initial_circles[i]
        initial_guess.extend([x, y, r])
    
    # Apply optimization with bounds and constraints
    try:
        # Use SLSQP method with more iterations and tighter tolerance
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1500, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result.success:
            # Extract final solution
            final_positions = []
            final_radii = []
            for i in range(n):
                idx = i * 3
                x, y, r = result.x[idx], result.x[idx+1], result.x[idx+2]
                final_positions.append([x, y])
                final_radii.append(r)
            
            # Update circles array
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_positions[i][0], final_positions[i][1], final_radii[i]]
            return circles
        else:
            return initial_circles
            
    except Exception as e:
        # If optimization fails for any reason, return the initial arrangement
        return initial_circles


def refine_final_configuration(initial_circles, width, height):
    """Apply additional refinement to the best configuration"""
    # Try a second round of optimization with even more iterations
    try:
        result = optimize_sqp_enhanced(initial_circles, width, height)
        if result is not None:
            # Check if it improved significantly
            original_sum = np.sum(initial_circles[:, 2])
            refined_sum = np.sum(result[:, 2])
            if refined_sum > original_sum * 1.02:  # At least 2% improvement
                return result
    except:
        pass
    
    return initial_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
