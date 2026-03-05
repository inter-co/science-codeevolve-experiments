# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import random
import time
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import warnings

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining evolutionary optimization with local refinement.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    # Rectangle dimensions: width + height = 2 (perimeter = 4)
    # Try different width/height ratios to find optimal configuration
    width = 1.0
    height = 1.0
    
    # Improved initialization using a more sophisticated approach
    def initialize_better_packing():
        circles = []
        
        # Use a more efficient initialization strategy based on known good patterns
        # Try a hexagonal-like packing pattern that works well for this number of circles
        
        # First, try a more structured approach
        # For 21 circles, we can arrange in approximately 5 rows and 5 columns with some offsetting
        rows = 5
        cols = 5
        
        # Calculate spacing that allows for good packing
        cell_width = width / cols
        cell_height = height / rows
        
        # Use hexagonal packing approach for better results
        max_radius = min(width, height) * 0.12  # Slightly smaller than previous attempt
        
        # Place circles in a grid with alternating rows offset
        placed = 0
        for i in range(rows):
            for j in range(cols):
                if placed >= n:
                    break
                    
                # Calculate position
                x = (j + 0.5) * cell_width
                y = (i + 0.5) * cell_height
                
                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += cell_width / 2
                
                # Adjust for boundaries
                x = max(max_radius, min(width - max_radius, x))
                y = max(max_radius, min(height - max_radius, y))
                
                # Only add if within bounds
                if (x - max_radius >= 0 and x + max_radius <= width and
                    y - max_radius >= 0 and y + max_radius <= height):
                    circles.append([x, y, max_radius])
                    placed += 1
                    
        # Fill remaining circles with strategic placement
        while len(circles) < n:
            # Try to place in less crowded areas with better checking
            attempts = 0
            while attempts < 100:
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                
                # Check overlap with existing circles efficiently
                valid = True
                for cx, cy, r in circles:
                    dist_sq = (x - cx)**2 + (y - cy)**2
                    min_dist_sq = (r + max_radius)**2
                    if dist_sq < min_dist_sq:
                        valid = False
                        break
                
                if valid:
                    circles.append([x, y, max_radius * 0.9])
                    break
                attempts += 1
            
            # If still couldn't place, just add at random with smaller radius
            if len(circles) < n:
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                circles.append([x, y, max_radius * 0.8])
            
        return np.array(circles)
    
    # Even better initialization using a more scientific approach
    def initialize_optimized():
        circles = []
        
        # Create a better initial configuration using a combination of approaches
        # Start with a grid pattern, then refine
        
        # Calculate approximate area per circle
        total_area = width * height
        avg_circle_area = total_area / n
        avg_radius = math.sqrt(avg_circle_area / math.pi)
        
        # Try different grid layouts
        best_layout = None
        best_coverage = 0
        
        # Try various grid configurations
        configs = [
            (4, 6),  # 4 rows, 6 cols
            (5, 5),  # 5 rows, 5 cols  
            (6, 4),  # 6 rows, 4 cols
        ]
        
        for rows, cols in configs:
            if rows * cols >= n:
                # Try to place circles in a structured way
                spacing_x = width / (cols + 1)
                spacing_y = height / (rows + 1)
                
                temp_circles = []
                placed = 0
                
                for i in range(rows):
                    for j in range(cols):
                        if placed >= n:
                            break
                            
                        x = spacing_x * (j + 1)
                        y = spacing_y * (i + 1)
                        
                        # Add slight randomness to avoid perfect grid artifacts
                        x += random.uniform(-spacing_x/6, spacing_x/6)
                        y += random.uniform(-spacing_y/6, spacing_y/6)
                        
                        # Keep within bounds
                        x = max(avg_radius, min(width - avg_radius, x))
                        y = max(avg_radius, min(height - avg_radius, y))
                        
                        temp_circles.append([x, y, avg_radius * 0.9])
                        placed += 1
                        
                if placed >= n:
                    # Check how well this layout covers the area
                    coverage = calculate_coverage(temp_circles)
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_layout = temp_circles[:]
        
        # If we found a good layout, use it; otherwise fall back to simpler approach
        if best_layout is not None:
            circles = best_layout
        else:
            # Fall back to basic approach
            circles = initialize_better_packing()
            
        # Ensure we have exactly n circles
        while len(circles) < n:
            x = random.uniform(avg_radius, width - avg_radius)
            y = random.uniform(avg_radius, height - avg_radius)
            circles.append([x, y, avg_radius * 0.8])
            
        return np.array(circles[:n])
    
    # Coverage calculation helper
    def calculate_coverage(circles):
        # Simple heuristic: measure how much of the area is covered by circles
        covered_area = sum(math.pi * r**2 for _, _, r in circles)
        total_area = width * height
        return covered_area / total_area if total_area > 0 else 0
    
    # Initialize with better configuration
    circles = initialize_optimized()
    
    # Optimization using a more robust approach
    def objective(params):
        """Minimize negative sum of radii (maximize sum of radii)"""
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        # We want to maximize sum of radii, so minimize negative sum
        return -np.sum(radii)
    
    def constraint_func(params):
        """Constraint function returning positive values when satisfied"""
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Non-overlap constraints - more efficient pairwise checking
        constraints = []
        
        # Use efficient distance checking with early termination
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance_sq = dx*dx + dy*dy
                min_distance_sq = (radii[i] + radii[j])**2
                
                # Constraint should be positive when satisfied (distance >= min_distance)
                constraints.append(distance_sq - min_distance_sq)
        
        # Boundary constraints (positive when satisfied)
        for i in range(n):
            # Left boundary
            constraints.append(positions[i][0] - radii[i])
            # Right boundary  
            constraints.append(width - positions[i][0] - radii[i])
            # Bottom boundary
            constraints.append(positions[i][1] - radii[i])
            # Top boundary
            constraints.append(height - positions[i][1] - radii[i])
            
        return np.array(constraints)
    
    # Create initial parameter vector: [x1, y1, x2, y2, ..., xn, yn, r1, r2, ..., rn]
    initial_params = np.concatenate([
        circles[:, :2].flatten(),  # Positions
        circles[:, 2]              # Radii
    ])
    
    # Set bounds for positions and radii
    bounds = [(0, width) for _ in range(2*n)] + [(1e-6, width/2) for _ in range(n)]
    
    # Define constraints
    cons = {
        'type': 'ineq',  # Inequality constraints (g(x) >= 0)
        'fun': constraint_func
    }
    
    # Multi-start optimization approach
    best_result = None
    best_sum = -float('inf')
    
    # Try multiple optimization strategies with different starting points
    strategies = []
    
    # Strategy 1: Differential Evolution (global search)
    try:
        bounds_de = [(0, width) for _ in range(2*n)] + [(1e-6, width/2) for _ in range(n)]
        
        de_result = differential_evolution(
            objective,
            bounds_de,
            args=(),
            maxiter=50,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if de_result.success:
            strategies.append(('DE', de_result))
    except Exception as e:
        pass
    
    # Strategy 2: Multiple local optimizations with different starting points
    try:
        seeds = [42, 123, 456, 789, 101]
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            
            # Perturb the initial solution slightly
            perturbed_params = initial_params.copy()
            # Add small random noise to positions
            for i in range(2*n):
                if i % 2 == 0:  # x coordinates
                    perturbed_params[i] += random.uniform(-0.05, 0.05) * width
                else:  # y coordinates
                    perturbed_params[i] += random.uniform(-0.05, 0.05) * height
            # Keep radii unchanged or slightly perturb
            for i in range(n):
                perturbed_params[2*n + i] *= random.uniform(0.95, 1.05)
            
            # Clip to bounds
            for i in range(2*n):
                perturbed_params[i] = np.clip(perturbed_params[i], 0, width if i % 2 == 0 else height)
            for i in range(n):
                perturbed_params[2*n + i] = np.clip(perturbed_params[2*n + i], 1e-6, width/2)
            
            result = minimize(
                objective,
                perturbed_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                strategies.append(('Local', result))
    except Exception as e:
        pass
    
    # Strategy 3: Direct optimization from initial solution
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result.success:
            strategies.append(('Direct', result))
    except Exception as e:
        pass
    
    # Evaluate all strategies
    for name, result in strategies:
        try:
            final_positions = result.x[:-n].reshape(-1, 2)
            final_radii = result.x[-n:]
            final_sum = np.sum(final_radii)
            
            if final_sum > best_sum:
                best_sum = final_sum
                best_result = result
        except Exception:
            continue
    
    # If we didn't get a good result, fall back to just using the initial solution
    if best_result is None:
        # Just return the initial solution with some small optimization
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                final_positions = result.x[:-n].reshape(-1, 2)
                final_radii = result.x[-n:]
                circles[:, 0] = final_positions[:, 0]
                circles[:, 1] = final_positions[:, 1]
                circles[:, 2] = final_radii
            else:
                # Return the initial solution if optimization fails
                pass
        except Exception:
            pass
    else:
        # Use the best result
        final_positions = best_result.x[:-n].reshape(-1, 2)
        final_radii = best_result.x[-n:]
        circles[:, 0] = final_positions[:, 0]
        circles[:, 1] = final_positions[:, 1]
        circles[:, 2] = final_radii
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
