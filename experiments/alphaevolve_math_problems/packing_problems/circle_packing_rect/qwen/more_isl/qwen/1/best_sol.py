# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
from typing import Tuple
from scipy.spatial import Voronoi
import warnings
warnings.filterwarnings('ignore')


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining physics-based initialization with advanced optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    best_sum = 0
    best_circles = None
    
    # Test multiple aspect ratios more systematically around the optimal region
    # Focus on ratios where circle packing typically performs well (like INSPIRATION 1)
    ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5, 4.0]
    
    # Multi-start approach with different initialization strategies (like INSPIRATION 1)
    num_starts = 15
    
    for start_idx in range(num_starts):
        # Try different aspect ratios
        ratio_idx = start_idx % len(ratios)
        ratio = ratios[ratio_idx]
        
        # Determine dimensions based on ratio
        width = ratio * 2 / (1 + ratio)
        height = 2 / (1 + ratio)
        
        # Different initialization strategies (like INSPIRATION 1)
        if start_idx < 5:
            # Strategy 1: Hexagonal packing
            circles = initialize_hexagonal_pack(width, height, n)
        elif start_idx < 10:
            # Strategy 2: Grid-based with perturbation
            circles = initialize_grid_pack(width, height, n, start_idx)
        else:
            # Strategy 3: Random placement with overlap checking
            circles = initialize_random_pack(width, height, n)
        
        # Refine using enhanced optimization (like INSPIRATION 1)
        refined_circles = optimize_circles_enhanced(circles, width, height)
        
        current_sum = np.sum(refined_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = refined_circles.copy()
    
    # Final refinement with the best configuration found
    if best_circles is not None:
        # Run additional optimization with improved settings
        final_circles = optimize_circles_final(best_circles, width, height)
        final_sum = np.sum(final_circles[:, 2])
        if final_sum > best_sum:
            best_circles = final_circles
    
    return best_circles if best_circles is not None else initialize_hexagonal_pack(1.0, 1.0, n)


def initialize_hexagonal_pack(width, height, n):
    """Initialize circles in a hexagonal lattice pattern"""
    # Determine grid dimensions for hexagonal packing
    rows = int(np.sqrt(n) * 0.8) + 2
    cols = int(n / rows) + 2
    
    # Adjust to fit within bounds
    if rows * cols < n:
        rows += 1
        cols = int(n / rows) + 2
    
    # Calculate spacing for hexagonal packing
    max_radius = min(width, height) / 8.0
    spacing_x = 2 * max_radius * 0.866  # sqrt(3)/2 for hexagonal
    spacing_y = 2 * max_radius * 0.75   # 3/4 for hexagonal
    
    # Ensure spacing fits within rectangle
    if spacing_x > width or spacing_y > height:
        spacing_x = width / 5.0
        spacing_y = height / 5.0
        max_radius = spacing_x / 2.0
    
    circles = np.zeros((n, 3))
    idx = 0
    
    # Create hexagonal grid
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = (i % 2) * spacing_x / 2
            x = x_offset + (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            
            # Keep within bounds
            if x > width or y > height:
                continue
                
            # Set radius based on available space
            min_dist = min(x, width - x, y, height - y)
            radius = min(min_dist / 2.0, max_radius)
            
            circles[idx] = [x, y, radius]
            idx += 1
            if idx >= n:
                break
    
    # Fill remaining slots with random positions if needed
    for i in range(idx, n):
        x = random.uniform(0.1, width - 0.1)
        y = random.uniform(0.1, height - 0.1)
        # Radius based on distance to nearest edges
        min_dist = min(x, width - x, y, height - y)
        radius = min(min_dist / 3.0, max_radius)
        circles[i] = [x, y, radius]
    
    return circles[:n]


def initialize_grid_pack(width, height, n, seed):
    """Initialize circles in a grid pattern with some perturbation"""
    random.seed(seed)
    
    rows = int(np.sqrt(n)) + 1
    cols = int(n / rows) + 1
    
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)
    
    circles = np.zeros((n, 3))
    idx = 0
    
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x + random.uniform(-spacing_x/6, spacing_x/6)
            y = (i + 1) * spacing_y + random.uniform(-spacing_y/6, spacing_y/6)
            
            # Set radius based on available space
            min_dist = min(x, width - x, y, height - y)
            radius = min(min_dist / 3.0, min(width, height) / 8.0)
            
            circles[idx] = [x, y, radius]
            idx += 1
            if idx >= n:
                break
    
    # Fill remaining slots if needed
    for i in range(idx, n):
        x = random.uniform(0.1, width - 0.1)
        y = random.uniform(0.1, height - 0.1)
        min_dist = min(x, width - x, y, height - y)
        radius = min(min_dist / 3.0, min(width, height) / 8.0)
        circles[i] = [x, y, radius]
    
    return circles


def initialize_random_pack(width, height, n):
    """Initialize circles with random placement avoiding overlaps"""
    max_radius = min(width, height) / 6.0
    circles = np.zeros((n, 3))
    
    # Place circles randomly with collision avoidance
    for i in range(n):
        placed = False
        attempts = 0
        while not placed and attempts < 1000:
            x = random.uniform(0.1, width - 0.1)
            y = random.uniform(0.1, height - 0.1)
            radius = random.uniform(0.01, max_radius)
            
            # Check if it's valid (not overlapping with others)
            valid = True
            for j in range(i):
                other_x, other_y, other_r = circles[j]
                distance = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                if distance < (radius + other_r):
                    valid = False
                    break
            
            if valid:
                circles[i] = [x, y, radius]
                placed = True
            attempts += 1
        
        # If we couldn't place after many attempts, just place somewhere
        if not placed:
            x = random.uniform(0.1, width - 0.1)
            y = random.uniform(0.1, height - 0.1)
            radius = min(0.1, max_radius)
            circles[i] = [x, y, radius]
    
    return circles


def optimize_circles_enhanced(initial_circles, width, height):
    """Enhanced optimization with better parameters and multiple fallback strategies"""
    n = len(initial_circles)
    
    # Flatten initial parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define bounds for optimization
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, width - 0.001), (0.001, height - 0.001), (0.001, min(width, height)/2 - 0.001)])
    
    def objective(params):
        # Sum of radii (negative because we're minimizing)
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i + 2]  # radius is at index 3*i + 2
        return -total_radius
    
    def constraint_bounds(params):
        """Ensure circles are within bounds"""
        constraints = []
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            constraints.append(x - r)  # left boundary
            constraints.append(y - r)  # bottom boundary
            constraints.append(width - x - r)  # right boundary
            constraints.append(height - y - r)  # top boundary
        return np.array(constraints)
    
    def constraint_overlaps(params):
        """Ensure no overlaps between circles"""
        constraints = []
        for i in range(n):
            x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
            for j in range(i+1, n):
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                # Constraint: distance^2 >= (r1 + r2)^2 (positive means feasible)
                constraints.append(distance_sq - min_distance_sq)
        return np.array(constraints)
    
    # First, try differential evolution with high precision and population
    try:
        # Define constraints for differential evolution
        constraints = [
            {'type': 'ineq', 'fun': constraint_bounds},
            {'type': 'ineq', 'fun': constraint_overlaps}
        ]
        
        result = differential_evolution(
            objective,
            bounds,
            constraints=constraints,
            seed=42,
            maxiter=1000,
            popsize=60,
            mutation=(0.5, 1),
            recombination=0.7,
            atol=1e-10,
            rtol=1e-10,
            disp=False
        )
        
        if result.success:
            final_params = result.x
            optimized_circles = np.zeros((n, 3))
            for i in range(n):
                optimized_circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
            return optimized_circles
    except Exception as e:
        pass
    
    # If that fails, try local optimization with better tolerance
    try:
        # Set up constraints
        cons = [
            {'type': 'ineq', 'fun': constraint_bounds},
            {'type': 'ineq', 'fun': constraint_overlaps}
        ]
        
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-10, 'eps': 1e-10}
        )
        
        if result.success:
            final_params = result.x
            optimized_circles = np.zeros((n, 3))
            for i in range(n):
                optimized_circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
            return optimized_circles
    except Exception as e:
        pass
    
    # If all optimization fails, return initial configuration
    return initial_circles

def optimize_circles_final(initial_circles, width, height):
    """Final optimization with even more precise parameters"""
    n = len(initial_circles)
    
    # Flatten initial parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define bounds for optimization
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, width - 0.001), (0.001, height - 0.001), (0.001, min(width, height)/2 - 0.001)])
    
    def objective(params):
        # Sum of radii (negative because we're minimizing)
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i + 2]  # radius is at index 3*i + 2
        return -total_radius
    
    def constraint_bounds(params):
        """Ensure circles are within bounds"""
        constraints = []
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            constraints.append(x - r)  # left boundary
            constraints.append(y - r)  # bottom boundary
            constraints.append(width - x - r)  # right boundary
            constraints.append(height - y - r)  # top boundary
        return np.array(constraints)
    
    def constraint_overlaps(params):
        """Ensure no overlaps between circles"""
        constraints = []
        for i in range(n):
            x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
            for j in range(i+1, n):
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                # Constraint: distance^2 >= (r1 + r2)^2 (positive means feasible)
                constraints.append(distance_sq - min_distance_sq)
        return np.array(constraints)
    
    # Try with even more iterations and tighter tolerances
    try:
        # Set up constraints
        cons = [
            {'type': 'ineq', 'fun': constraint_bounds},
            {'type': 'ineq', 'fun': constraint_overlaps}
        ]
        
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1500, 'ftol': 1e-12, 'eps': 1e-12}
        )
        
        if result.success:
            final_params = result.x
            optimized_circles = np.zeros((n, 3))
            for i in range(n):
                optimized_circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
            return optimized_circles
    except Exception as e:
        pass
    
    return initial_circles





# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
