# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
import random
from typing import Tuple, List

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a mathematically informed approach combining hexagonal lattice construction with geometric optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Mathematical approach: use known optimal hexagonal lattice configurations
    # For 21 circles in a rectangle, we'll try to construct the best hexagonal lattice
    # and then optimize the final arrangement
    
    # Try a few key aspect ratios that typically work well
    # Based on mathematical analysis and known circle packing results
    aspect_ratios = [
        (1.3, 0.7),      # Golden ratio related - very promising
        (1.2, 0.8),      # 3:2 ratio - commonly optimal  
        (1.4, 0.6),      # Another common ratio
        (1.5, 0.5),      # 3:1 ratio - often good for dense packing
        (1.1, 0.9),      # Nearly square
        (0.8, 1.2),      # 2:3 ratio
        (1.0, 1.0),      # Square
        (1.6, 0.4),      # Extreme aspect ratio
    ]
    
    best_sum = 0
    best_circles = None
    
    for width_ratio, height_ratio in aspect_ratios:
        # Normalize to make perimeter = 4
        width = 2 * width_ratio / (width_ratio + height_ratio)
        height = 2 * height_ratio / (width_ratio + height_ratio)
        
        # Use mathematical approach for initialization - construct hexagonal lattice
        circles = construct_hexagonal_lattice(width, height, 21)
        
        # Refine with optimization
        optimized_circles = optimize_with_mathematical_constraints(circles, width, height)
        
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
    
    # If we didn't find a good solution, fallback to a robust construction
    if best_circles is None:
        width, height = 1.3, 0.7  # Golden ratio
        best_circles = construct_hexagonal_lattice(width, height, 21)
        best_circles = optimize_with_mathematical_constraints(best_circles, width, height)
    
    return best_circles


def construct_hexagonal_lattice(width: float, height: float, n: int) -> np.ndarray:
    """
    Construct an initial configuration using hexagonal lattice pattern based on mathematical analysis.
    This provides a much better starting point than random initialization.
    """
    circles = np.zeros((n, 3))
    
    # Calculate area available for circles
    total_area = width * height
    
    # Estimate radius based on packing density considerations
    # For 21 circles in a rectangle, we can estimate a good average radius
    # Using packing density of hexagonal packing (~0.9069) as upper bound
    max_area_for_circles = total_area * 0.85  # Leave some margin
    avg_area_per_circle = max_area_for_circles / n
    estimated_radius = np.sqrt(avg_area_per_circle / np.pi)
    
    # Use hexagonal lattice spacing
    hex_radius = estimated_radius * 0.9  # Slightly smaller for practical packing
    hex_width = hex_radius * 2
    hex_height = hex_radius * np.sqrt(3)
    
    # Calculate grid dimensions
    cols = max(1, int(width / hex_width) + 2)
    rows = max(1, int(height / hex_height) + 2)
    
    # Create hexagonal grid
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal pattern
            x_offset = (i % 2) * (hex_width / 2)
            x = x_offset + j * hex_width + hex_radius
            y = i * hex_height + hex_radius
            
            # Check if circle fits within bounds
            if (x >= hex_radius and x <= width - hex_radius and 
                y >= hex_radius and y <= height - hex_radius):
                circles[idx] = [x, y, hex_radius]
                idx += 1
                
        if idx >= n:
            break
    
    # Fill remaining positions with better distributed placements
    for i in range(idx, n):
        # Use more strategic placement based on mathematical bounds
        x = np.random.uniform(hex_radius, width - hex_radius)
        y = np.random.uniform(hex_radius, height - hex_radius)
        # Use radius that's closer to what we expect for optimal packing
        radius = hex_radius * (0.8 + np.random.random() * 0.4)
        circles[i] = [x, y, radius]
    
    return circles


def optimize_with_mathematical_constraints(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    Apply mathematical optimization with better constraint handling and analytical insights.
    """
    n = len(initial_circles)
    # Flatten initial configuration
    initial_flat = []
    for i in range(n):
        initial_flat.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    def objective(params):
        # Minimize negative sum of radii (maximize sum)
        radii = params[2::3]  # Every third element starting from index 2
        return -sum(radii)
    
    def constraint_func(params):
        # Mathematical constraint function with better handling
        circles = []
        for i in range(n):
            x = params[i*3]
            y = params[i*3 + 1]
            r = params[i*3 + 2]
            circles.append([x, y, r])
        
        constraints = []
        
        # Boundary constraints - ensure circles are within rectangle
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([
                x - r,              # x >= r (left boundary)
                width - x - r,      # width - x >= r (right boundary)
                y - r,              # y >= r (bottom boundary) 
                height - y - r      # height - y >= r (top boundary)
            ])
        
        # Non-overlap constraints with improved numerical stability
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)
                # Safety margin to prevent numerical issues
                constraints.append(distance - (r1 + r2 + 1e-12))
        
        return np.array(constraints)
    
    # Set up bounds with mathematical justification
    bounds = []
    for i in range(n):
        # x bounds - leave safe margins
        bounds.append((0.001, width - 0.001))
        # y bounds
        bounds.append((0.001, height - 0.001))
        # r bounds - reasonable limits based on container size
        max_r = min(width, height) * 0.49
        bounds.append((0.001, max_r))
    
    # Try different optimization approaches with better settings
    methods_and_options = [
        ('SLSQP', {'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}),
        ('trust-constr', {'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-12}),
    ]
    
    for method, options in methods_and_options:
        try:
            result = minimize(
                objective,
                initial_flat,
                method=method,
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options=options
            )
            
            if result.success:
                # Extract optimized values
                optimized_vars = result.x
                optimized_circles = []
                for i in range(n):
                    x = optimized_vars[3*i]
                    y = optimized_vars[3*i+1]
                    r = optimized_vars[3*i+2]
                    optimized_circles.append([x, y, r])
                return np.array(optimized_circles)
        except Exception:
            continue
    
    # Fallback to initial configuration if optimization fails
    return initial_circles.copy()


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
