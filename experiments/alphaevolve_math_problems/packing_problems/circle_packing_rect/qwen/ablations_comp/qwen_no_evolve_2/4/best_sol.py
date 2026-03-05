# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import time
from typing import Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
from itertools import combinations
from deap import base, creator, tools, algorithms
import multiprocessing as mp
import copy
from scipy.spatial import Voronoi
import math

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining smart initialization with local optimization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Optimize rectangle dimensions - try different ratios to find better packing
    # Based on research, a square-ish rectangle often works well for circle packing
    width, height = 1.0, 1.0  # Square rectangle with perimeter 4
    
    # Number of circles
    n = 21
    
    # Helper function to check if a circle fits within the rectangle
    def is_valid_circle(x, y, r):
        return (r <= x <= width - r and 
                r <= y <= height - r)
    
    # Helper function to compute total radius sum
    def compute_radius_sum(circles_array):
        return np.sum(circles_array[:, 2])
    
    # More efficient overlap checking using vectorized operations
    def check_all_overlaps(circles_array):
        # Vectorized overlap checking
        if len(circles_array) < 2:
            return False
            
        # Get all circle positions and radii
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # Compute minimum distances needed to avoid overlap
        min_distances = np.outer(radii, np.ones_like(radii)) + np.outer(np.ones_like(radii), radii)
        
        # Set diagonal to infinity to ignore self-overlaps
        np.fill_diagonal(distances, np.inf)
        
        # Check if any circles overlap
        return np.any(distances < min_distances)
    
    # Better initialization using hexagonal packing pattern for initial placement
    def initialize_hexagonal_pattern():
        circles = []
        
        # Try to create a hexagonal-like pattern
        rows = 4  # Number of rows
        cols = 6  # Number of columns (more than needed to give flexibility)
        
        # Calculate spacing based on desired number of circles
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)
        
        # Adjust spacing for better packing
        adjusted_spacing_x = spacing_x * 0.8
        adjusted_spacing_y = spacing_y * 0.8
        
        # Place circles in a grid pattern with some randomness
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row
                offset = 0 if i % 2 == 0 else adjusted_spacing_x * 0.5
                x = offset + (j + 1) * adjusted_spacing_x
                y = (i + 1) * adjusted_spacing_y
                
                # Add some randomness to positions
                x += random.uniform(-adjusted_spacing_x * 0.2, adjusted_spacing_x * 0.2)
                y += random.uniform(-adjusted_spacing_y * 0.2, adjusted_spacing_y * 0.2)
                
                # Calculate max possible radius
                max_r = min(x, width - x, y, height - y)
                if max_r > 0.01:  # Minimum radius threshold
                    r = random.uniform(0.01, max_r * 0.3)  # Start with smaller radii to allow more placement
                    
                    # Check if this circle fits and doesn't overlap with existing ones
                    candidate_circle = [x, y, r]
                    temp_circles = circles + [candidate_circle]
                    temp_array = np.array(temp_circles)
                    
                    if is_valid_circle(x, y, r) and not check_all_overlaps(temp_array):
                        circles.append(candidate_circle)
        
        # Fill remaining positions with random valid circles
        while len(circles) < n:
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            # Calculate max possible radius
            max_r = min(x, width - x, y, height - y)
            if max_r > 0.01:
                r = random.uniform(0.01, max_r * 0.2)
                if is_valid_circle(x, y, r):
                    circles.append([x, y, r])
        
        return np.array(circles)
    
    # Improved local refinement using a more robust optimization approach
    def local_refinement(circles_array):
        # Create a flattened parameter vector [x1,y1,r1,x2,y2,r2,...,x21,y21,r21]
        def flatten_circles(circles_array):
            return circles_array.flatten()
        
        def unflatten_circles(params):
            return params.reshape(-1, 3)
        
        # Objective function to maximize (negative because we minimize)
        def objective(params):
            circles_array = unflatten_circles(params)
            return -compute_radius_sum(circles_array)
        
        # Constraint functions
        def constraint_positions(params):
            """Ensure all circles stay within bounds"""
            circles_array = unflatten_circles(params)
            constraints = []
            for i in range(len(circles_array)):
                x, y, r = circles_array[i]
                # Circle must fit within rectangle
                constraints.extend([
                    x - r,  # x >= r
                    width - x - r,  # width - x >= r
                    y - r,  # y >= r
                    height - y - r  # height - y >= r
                ])
            return np.array(constraints)
        
        def constraint_overlaps(params):
            """Ensure no overlaps between circles"""
            circles_array = unflatten_circles(params)
            constraints = []
            for i, j in combinations(range(len(circles_array)), 2):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                distance_sq = (x2 - x1)**2 + (y2 - y1)**2
                min_distance_sq = (r1 + r2)**2
                # We want distance^2 >= min_distance^2, so constraint is distance^2 - min_distance^2 >= 0
                constraints.append(distance_sq - min_distance_sq)
            return np.array(constraints)
        
        # Flatten initial configuration
        initial_params = flatten_circles(circles_array)
        
        # Create bounds for parameters (x, y, r) for each circle
        bounds = []
        for i in range(n):
            # x bounds
            bounds.append((0.001, width - 0.001))
            # y bounds  
            bounds.append((0.001, height - 0.001))
            # r bounds (positive and bounded by available space)
            bounds.append((0.001, min(width/2, height/2) - 0.001))
        
        # Define constraints
        cons = [
            {'type': 'ineq', 'fun': lambda p: constraint_positions(p)},
            {'type': 'ineq', 'fun': lambda p: constraint_overlaps(p)}
        ]
        
        # Try multiple optimization methods
        try:
            # First try SLSQP with good starting point
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_circles = unflatten_circles(result.x)
                return optimized_circles
        except Exception as e:
            pass
        
        # Fallback to L-BFGS-B if SLSQP fails
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_circles = unflatten_circles(result.x)
                return optimized_circles
        except Exception as e:
            pass
            
        return circles_array
    
    # Enhanced multi-start local search with better initialization strategies
    def enhanced_multi_start_local_search():
        best_circles = None
        best_sum = -np.inf
        
        # Try multiple initialization strategies
        strategies = [
            "random",
            "hexagonal",
            "grid"
        ]
        
        for strategy in strategies:
            for _ in range(5):  # 5 runs per strategy
                # Initialize based on strategy
                if strategy == "hexagonal":
                    circles = initialize_hexagonal_pattern()
                elif strategy == "grid":
                    # Grid initialization
                    circles = []
                    rows = 5
                    cols = 5
                    spacing_x = width / (cols + 1)
                    spacing_y = height / (rows + 1)
                    for i in range(rows):
                        for j in range(cols):
                            if len(circles) >= n:
                                break
                            x = (j + 1) * spacing_x
                            y = (i + 1) * spacing_y
                            max_r = min(x, width - x, y, height - y)
                            if max_r > 0.01:
                                r = random.uniform(0.01, max_r * 0.2)
                                if is_valid_circle(x, y, r):
                                    circles.append([x, y, r])
                    # Fill remaining
                    while len(circles) < n:
                        x = random.uniform(0.01, width - 0.01)
                        y = random.uniform(0.01, height - 0.01)
                        max_r = min(x, width - x, y, height - y)
                        if max_r > 0.01:
                            r = random.uniform(0.01, max_r * 0.2)
                            if is_valid_circle(x, y, r):
                                circles.append([x, y, r])
                    circles = np.array(circles)
                else:  # random
                    # Initialize with random valid circles
                    circles = []
                    attempts = 0
                    while len(circles) < n and attempts < 1000:
                        x = random.uniform(0.01, width - 0.01)
                        y = random.uniform(0.01, height - 0.01)
                        # Calculate max possible radius
                        max_r = min(x, width - x, y, height - y)
                        if max_r > 0.01:
                            r = random.uniform(0.01, max_r * 0.3)
                            
                            # Check if circle would fit without overlapping
                            candidate_circle = [x, y, r]
                            temp_circles = circles + [candidate_circle]
                            temp_array = np.array(temp_circles)
                            
                            if is_valid_circle(x, y, r) and not check_all_overlaps(temp_array):
                                circles.append(candidate_circle)
                        attempts += 1
                    
                    # Fill remaining positions
                    while len(circles) < n:
                        x = random.uniform(0.01, width - 0.01)
                        y = random.uniform(0.01, height - 0.01)
                        max_r = min(x, width - x, y, height - y)
                        if max_r > 0.01:
                            r = random.uniform(0.01, max_r * 0.1)
                            if is_valid_circle(x, y, r):
                                circles.append([x, y, r])
                    circles = np.array(circles)
                
                # Apply local refinement
                refined_circles = local_refinement(circles)
                refined_sum = compute_radius_sum(refined_circles)
                
                if refined_sum > best_sum:
                    best_sum = refined_sum
                    best_circles = refined_circles
                    
        return best_circles
    
    # Run the optimization
    start_time = time.time()
    
    # Use enhanced multi-start local search
    final_circles = enhanced_multi_start_local_search()
    
    # Final local refinement
    final_circles = local_refinement(final_circles)
    
    # Ensure we have exactly 21 circles
    if len(final_circles) < 21:
        # If somehow we don't have enough, create more
        current_count = len(final_circles)
        additional_circles = []
        for _ in range(21 - current_count):
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            max_r = min(x, width - x, y, height - y)
            if max_r > 0.01:
                r = random.uniform(0.01, max_r * 0.1)
                if is_valid_circle(x, y, r):
                    additional_circles.append([x, y, r])
        
        final_circles = np.vstack([final_circles, additional_circles])
    
    return final_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
