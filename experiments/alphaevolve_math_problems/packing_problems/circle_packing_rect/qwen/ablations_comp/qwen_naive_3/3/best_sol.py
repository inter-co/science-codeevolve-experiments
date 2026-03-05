# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import random
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from functools import partial
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid evolutionary and optimization approach to beat the benchmark.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Try different aspect ratios to find optimal
    width, height = 1.0, 1.0
    
    # Number of circles
    n = 21
    
    # More effective initialization using better hexagonal packing
    def generate_better_initialization():
        # Test different width/height ratios
        best_ratio = 1.0
        best_sum = 0
        best_config = None
        
        # Try several aspect ratios
        ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
        
        for ratio in ratios:
            w = 2.0 / (1 + ratio)  # width + height = 2
            h = 2.0 / (1 + 1/ratio)
            
            # Calculate area per circle
            total_area = w * h
            circle_area = total_area / n * 0.8  # Leave some margin
            avg_radius = np.sqrt(circle_area / np.pi)
            
            # Hexagonal packing parameters
            spacing = 2 * avg_radius
            hex_radius = spacing * np.sqrt(3) / 2
            
            # Generate hexagonal grid
            rows = int(np.ceil(h / hex_radius)) + 1
            cols = int(np.ceil(w / spacing)) + 1
            
            centers = []
            for i in range(rows):
                for j in range(cols):
                    x = 0.1 + j * spacing + (i % 2) * spacing / 2
                    y = 0.1 + i * hex_radius
                    if x <= w - 0.1 and y <= h - 0.1:
                        centers.append([x, y])
            
            # Take first n centers, or pad if needed
            if len(centers) >= n:
                selected_centers = np.array(centers[:n])
            else:
                # Add random points for remaining circles
                selected_centers = np.array(centers)
                remaining = n - len(centers)
                for _ in range(remaining):
                    x = random.uniform(0.1, w - 0.1)
                    y = random.uniform(0.1, h - 0.1)
                    selected_centers = np.vstack([selected_centers, [x, y]])
            
            # Test this configuration
            test_radii = np.full(n, avg_radius * 0.8)  # Start with reasonable radii
            test_sum = np.sum(test_radii)
            
            if test_sum > best_sum:
                best_sum = test_sum
                best_config = (selected_centers, w, h, ratio)
        
        if best_config is not None:
            return best_config[0], best_config[1], best_config[2], best_config[3]
        else:
            # Fallback to default
            centers = []
            for i in range(n):
                x = random.uniform(0.1, 1.9)
                y = random.uniform(0.1, 1.9)
                centers.append([x, y])
            return np.array(centers), 1.0, 1.0, 1.0
    
    # Generate initial configuration
    initial_centers, width, height, best_ratio = generate_better_initialization()
    
    # Set initial radii based on available space
    initial_radii = np.full(n, 0.05)
    
    # Combine into one array for optimization
    initial_params = np.column_stack([initial_centers, initial_radii])
    
    # Define constraint checking functions with vectorized operations for efficiency
    def check_containment_vectorized(xs, ys, rs, w, h):
        """Vectorized check if circles fit within bounds"""
        return (xs - rs >= 0) & (ys - rs >= 0) & (xs + rs <= w) & (ys + rs <= h)
    
    def check_non_overlap_vectorized(xs, ys, rs):
        """Vectorized check for non-overlapping constraints"""
        n = len(xs)
        # Create distance matrix
        dist_matrix = cdist(np.column_stack([xs, ys]), np.column_stack([xs, ys]))
        # Create sum of radii matrix
        r_matrix = rs[:, np.newaxis] + rs[np.newaxis, :]
        # Non-overlap condition: distance >= sum of radii
        # We need to avoid self-comparison (diagonal elements)
        np.fill_diagonal(dist_matrix, np.inf)
        return np.all(dist_matrix >= r_matrix)
    
    def evaluate_individual_vectorized(individual, w, h):
        """Vectorized evaluation of fitness"""
        # Reshape individual to (n, 3) format
        circles = individual.reshape((n, 3))
        
        # Extract parameters
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Check constraints
        valid_containment = check_containment_vectorized(xs, ys, rs, w, h)
        if not np.all(valid_containment):
            return 1000000  # Large penalty for containment violation
        
        # Check non-overlap constraints
        if not check_non_overlap_vectorized(xs, ys, rs):
            return 1000000  # Large penalty for overlap violation
        
        # Return negative sum of radii (since we want to maximize)
        return -np.sum(rs)
    
    # Optimized local optimization approach
    def optimize_with_local_search(initial_circles, w, h, max_iter=100):
        """Improved local search with better constraint handling"""
        circles = initial_circles.copy()
        n = len(circles)
        
        # Create bounds
        bounds = [(0.01, w - 0.01), (0.01, h - 0.01), (0.001, min(w, h)/2 - 0.01)]
        
        for iteration in range(max_iter):
            improved = False
            
            # Try to improve each circle
            for i in range(n):
                current_x, current_y, current_r = circles[i]
                
                # Calculate maximum possible radius increase
                max_radius_increase = current_r
                
                # Boundary constraints
                max_radius_increase = min(max_radius_increase, current_x - 0.01)
                max_radius_increase = min(max_radius_increase, current_y - 0.01)
                max_radius_increase = min(max_radius_increase, w - current_x - 0.01)
                max_radius_increase = min(max_radius_increase, h - current_y - 0.01)
                
                # Overlap constraints with all other circles
                for j in range(n):
                    if i != j:
                        other_x, other_y, other_r = circles[j]
                        dist = np.sqrt((current_x - other_x)**2 + (current_y - other_y)**2)
                        max_radius_increase = min(max_radius_increase, dist - other_r - 0.001)
                
                # If we can increase radius, do it
                if max_radius_increase > 0.001:
                    # Increase by a small fraction of the maximum
                    new_r = min(current_r + max_radius_increase * 0.3, current_r * 1.2)
                    
                    # Validate this change
                    valid = True
                    for j in range(n):
                        if i != j:
                            other_x, other_y, other_r = circles[j]
                            dist = np.sqrt((current_x - other_x)**2 + (current_y - other_y)**2)
                            if dist < (new_r + other_r):
                                valid = False
                                break
                    
                    if valid:
                        circles[i, 2] = new_r
                        improved = True
            
            if not improved:
                break
                
        return circles
    
    # Enhanced optimization using scipy with better bounds
    def optimize_scipy_approach(initial_params, w, h):
        # Define bounds for x, y, r
        bounds = []
        for i in range(n):
            bounds.extend([
                (0.01, w - 0.01),   # x bounds
                (0.01, h - 0.01),   # y bounds
                (0.001, min(w, h)/2 - 0.01)  # r bounds
            ])
        
        # Objective function
        def objective(params):
            # Extract radii
            radii = params[2::3]
            # Return negative sum of radii (we want to maximize)
            return -np.sum(radii)
        
        # Constraints for scipy
        def containment_constraint(params):
            results = []
            for i in range(n):
                x, y, r = params[3*i], params[3*i+1], params[3*i+2]
                results.extend([x - r, y - r, w - x - r, h - y - r])
            return np.array(results)
        
        def non_overlap_constraint(params):
            results = []
            for i in range(n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                for j in range(i+1, n):
                    x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    results.append(dist - (r1 + r2))
            return np.array(results)
        
        constraints = [
            {'type': 'ineq', 'fun': containment_constraint},
            {'type': 'ineq', 'fun': non_overlap_constraint}
        ]
        
        # Multiple attempts with different starting points
        best_result = None
        best_sum = float('-inf')
        
        for attempt in range(3):
            # Perturb initial parameters slightly
            perturbed_params = initial_params.copy()
            if attempt > 0:
                for i in range(n):
                    perturbed_params[i, 0] += random.uniform(-0.1, 0.1)
                    perturbed_params[i, 1] += random.uniform(-0.1, 0.1)
                    # Keep within bounds
                    perturbed_params[i, 0] = np.clip(perturbed_params[i, 0], 0.01, w - 0.01)
                    perturbed_params[i, 1] = np.clip(perturbed_params[i, 1], 0.01, h - 0.01)
            
            try:
                result = minimize(
                    objective,
                    perturbed_params.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4}
                )
                
                if result.success:
                    final_params = result.x
                    test_radii = final_params[2::3]
                    test_sum = np.sum(test_radii)
                    if test_sum > best_sum:
                        best_sum = test_sum
                        best_result = result
                        
            except Exception as e:
                continue
        
        if best_result is not None:
            final_params = best_result.x
            return np.reshape(final_params, (n, 3))
        else:
            return None
    
    # Main optimization flow
    # Try direct optimization first
    optimized_circles = optimize_scipy_approach(initial_params, width, height)
    
    if optimized_circles is not None:
        # Apply local search refinement
        refined_circles = optimize_with_local_search(optimized_circles, width, height)
        return refined_circles
    else:
        # Fall back to better hexagonal packing
        circles = np.zeros((n, 3))
        # Generate hexagonal pattern with optimized spacing
        spacing = 0.25  # Adjusted spacing
        hex_radius = spacing * np.sqrt(3) / 2
        row_count = int(np.ceil(np.sqrt(n)))
        col_count = int(np.ceil(n / row_count))
        
        idx = 0
        for i in range(row_count):
            for j in range(col_count):
                if idx >= n:
                    break
                x = 0.1 + j * spacing + (i % 2) * spacing / 2
                y = 0.1 + i * hex_radius
                # Make sure we stay within bounds
                if x <= width - 0.1 and y <= height - 0.1:
                    circles[idx] = [x, y, 0.05]
                    idx += 1
                else:
                    circles[idx] = [width/2, height/2, 0.05]
                    idx += 1
                if idx >= n:
                    break
        
        # Refine with local search
        circles = optimize_with_local_search(circles, width, height)
        return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
