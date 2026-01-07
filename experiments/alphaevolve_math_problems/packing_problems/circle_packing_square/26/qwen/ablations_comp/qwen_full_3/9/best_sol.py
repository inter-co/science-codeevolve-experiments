# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, multiple optimization attempts,
    and local refinement to achieve high-quality results within time constraints.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with hexagonal grid pattern for good distribution
    def initialize_hexagonal():
        circles = np.zeros((n, 3))
        
        # Create a hexagonal grid pattern - 5 rows, 6 columns for 30 positions
        rows = 5
        cols = 6
        spacing_x = 0.8 / cols
        spacing_y = 0.8 / rows
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Offset odd rows for hexagonal packing
                if i % 2 == 1:
                    x += spacing_x / 2
                circles[idx] = [x, y, 0.02]  # Start with small radius
                idx += 1
                if idx >= n:
                    break
        return circles
    
    # Constraint checking function with vectorization for efficiency
    def check_constraints(circles):
        """Check if all circles satisfy containment and non-overlap constraints"""
        # Check containment using vectorized operations
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Check containment: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
        contain_check = (
            (positions[:, 0] - radii >= 0) &
            (positions[:, 0] + radii <= 1) &
            (positions[:, 1] - radii >= 0) &
            (positions[:, 1] + radii <= 1)
        )
        
        if not np.all(contain_check):
            return False
        
        # Check non-overlap using vectorized distance computation
        dist_matrix = cdist(positions, positions)
        # Create upper triangular mask to avoid double counting
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        # Check if any pairs violate the non-overlap constraint
        overlap_check = dist_matrix[upper_mask] < (radii[:, None] + radii)[upper_mask]
        
        return not np.any(overlap_check)
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        total_radius = np.sum(circles_flat[2::3])  # Extract all radii
        return -total_radius  # Negative because we minimize
    
    # Vectorized constraint functions for better performance
    def containment_constraint(circles_flat):
        # Ensure all circles stay within bounds
        constraints = []
        for i in range(n):
            x, y, r = circles_flat[i*3], circles_flat[i*3+1], circles_flat[i*3+2]
            # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
            constraints.extend([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.array(constraints)
    
    def overlap_constraint(circles_flat):
        # Ensure no overlaps between circles
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_flat[i*3], circles_flat[i*3+1], circles_flat[i*3+2]
                x2, y2, r2 = circles_flat[j*3], circles_flat[j*3+1], circles_flat[j*3+2]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                min_dist_sq = (r1+r2)**2
                # We want dist >= r1 + r2, so we want dist_sq >= min_dist_sq
                # For optimization, we'll use dist_sq - min_dist_sq >= 0
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)
    
    # Enhanced local refinement with better radius adjustment (inspired by inspiration programs)
    def refine_solution(circles):
        """Apply enhanced local refinement to improve solution quality"""
        refined = circles.copy()
        improved = True
        iteration = 0
        max_iterations = 1000  # Increased for better refinement
        
        # Track consecutive non-improvements for early stopping
        no_improvement_count = 0
        max_no_improvement = 100  # Stop early if no improvement for many iterations
        
        while improved and iteration < max_iterations and no_improvement_count < max_no_improvement:
            improved = False
            iteration += 1
            
            # Try to increase radii while maintaining constraints
            for i in range(n):
                # Try to increase radius with more aggressive approach
                old_r = refined[i, 2]
                # Use more aggressive increment based on available space
                max_radius = min(
                    refined[i, 0], 1 - refined[i, 0],
                    refined[i, 1], 1 - refined[i, 1]
                )
                
                # Check overlap constraints with all other circles
                for j in range(n):
                    if i != j:
                        x1, y1, r1 = refined[i]
                        x2, y2, r2 = refined[j]
                        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        max_allowed = distance - r2
                        max_radius = min(max_radius, max_allowed)
                
                # Use more aggressive increment but still safe
                increment = min(0.005, (max_radius - old_r) * 0.8)  # More aggressive but conservative
                test_r = min(old_r + increment, max_radius)
                
                # Check if we can increase radius
                valid = True
                if test_r > old_r:
                    # Verify that the new radius doesn't cause overlap
                    positions = refined[:, :2]
                    radii = refined[:, 2]
                    
                    # Vectorized overlap check
                    for j in range(n):
                        if i != j:
                            dist_sq = (refined[i, 0] - refined[j, 0])**2 + \
                                      (refined[i, 1] - refined[j, 1])**2
                            min_dist_sq = (test_r + radii[j])**2
                            # Add tolerance to handle floating point precision
                            if dist_sq < min_dist_sq * 0.99999:
                                valid = False
                                break
                    
                    # Check containment
                    x, y = refined[i, :2]
                    if x - test_r < 0 or x + test_r > 1 or y - test_r < 0 or y + test_r > 1:
                        valid = False
                
                if valid and test_r > old_r:
                    refined[i, 2] = test_r
                    improved = True
                    no_improvement_count = 0  # Reset counter when improvement happens
                elif not valid:
                    no_improvement_count += 1  # Count failed attempts
            
            # Apply more sophisticated position adjustments to improve spacing
            for i in range(n):
                # Try to slightly adjust positions to resolve conflicts
                x, y, r = refined[i]
                
                # Move away from crowded areas with stronger force
                move_x, move_y = 0.0, 0.0
                positions = refined[:, :2]
                radii = refined[:, 2]
                
                for j in range(n):
                    if i != j:
                        x1, y1, r1 = refined[i]
                        x2, y2, r2 = refined[j]
                        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if distance < r1 + r2 + 1e-6:
                            # Move away from overlapping neighbor with stronger force
                            dx = x1 - x2
                            dy = y1 - y2
                            if distance > 0.001:  # Avoid division by zero
                                scale = (r1 + r2 - distance) / distance
                                move_x += dx * scale * 0.15  # Slightly stronger adjustment
                                move_y += dy * scale * 0.15  # Slightly stronger adjustment
                
                # Apply moves with bounds checking
                new_x = np.clip(x + move_x, r, 1-r)
                new_y = np.clip(y + move_y, r, 1-r)
                
                # Only update if there's meaningful movement
                if abs(new_x - x) > 1e-6 or abs(new_y - y) > 1e-6:
                    refined[i, 0] = new_x
                    refined[i, 1] = new_y
                    improved = True
                    no_improvement_count = 0  # Reset counter when improvement happens
                else:
                    no_improvement_count += 1  # Count when no meaningful movement
        
        return refined
    
    # Try multiple initialization strategies like in inspiration programs
    strategies = []
    
    # Strategy 1: Hexagonal grid
    strategies.append(initialize_hexagonal())
    
    # Strategy 2: Grid initialization  
    def initialize_grid():
        circles = np.zeros((n, 3))
        rows = 5
        cols = 6
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                y = 0.05 + (i + 0.5) * spacing_y
                circles[idx] = [x, y, min(spacing_x, spacing_y) * 0.25]
                idx += 1
                if idx >= n:
                    break
        return circles
    
    strategies.append(initialize_grid())
    
    # Strategy 3: Random initialization with better distribution
    def initialize_random():
        circles = np.zeros((n, 3))
        for i in range(n):
            # Better distributed random points
            row = i // 5
            col = i % 5
            x = 0.1 + col * 0.18 + random.uniform(-0.02, 0.02)
            y = 0.1 + row * 0.18 + random.uniform(-0.02, 0.02)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            r = 0.07
            circles[i] = [x, y, r]
        return circles
    
    strategies.append(initialize_random())
    
    # Find the best initialization
    best_initial = None
    best_initial_sum = 0
    for strategy in strategies:
        sum_radii = np.sum(strategy[:, 2])
        if sum_radii > best_initial_sum:
            best_initial_sum = sum_radii
            best_initial = strategy.copy()
    
    # Now run multiple optimization attempts from the best initial
    best_result = None
    best_sum = -np.inf
    
    # Run multiple optimizations from different starting points
    for attempt in range(20):  # Use 20 attempts like inspiration programs
        # Create a slightly randomized version of our best initial circles
        current_circles = best_initial.copy()
        
        # Add more significant random perturbations for better exploration
        random.seed(attempt)  # Fixed seed for reproducibility
        for i in range(n):
            current_circles[i, 0] += random.uniform(-0.02, 0.02)
            current_circles[i, 1] += random.uniform(-0.02, 0.02)
            current_circles[i, 2] += random.uniform(-0.01, 0.01)
            # Keep within bounds
            current_circles[i, 0] = np.clip(current_circles[i, 0], 0.001, 0.999)
            current_circles[i, 1] = np.clip(current_circles[i, 1], 0.001, 0.999)
            current_circles[i, 2] = np.clip(current_circles[i, 2], 0.001, 0.5)
        
        # Flatten for optimization
        flat_start = current_circles.flatten()
        
        # Set up bounds for optimization (x, y, r) for each circle
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])  # Slightly wider radius bound
        
        # Optimize using SLSQP method with very high precision like inspiration programs
        try:
            result = minimize(
                objective,
                flat_start,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: containment_constraint(x)},
                    {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
                ],
                options={'maxiter': 5000, 'ftol': 1e-10, 'eps': 1e-8},  # Much tighter tolerances
                callback=lambda x: None  # No callback needed
            )
            
            if result.success:
                final_circles = result.x.reshape((n, 3))
                sum_radii = np.sum(final_circles[:, 2])
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = final_circles
                    
        except Exception:
            continue
    
    # Apply final refinement to the best result
    if best_result is not None:
        # Apply enhanced refinement
        final_circles = refine_solution(best_result)
        
        # If the refined version is better, use it
        refined_sum = np.sum(final_circles[:, 2])
        if refined_sum > best_sum:
            best_sum = refined_sum
            best_result = final_circles
        
        # Final validation
        if check_constraints(best_result):
            return best_result
    
    # Return the best result found or fallback to initial
    if best_result is not None:
        return best_result
    else:
        # Fallback to a robust initialization
        circles = initialize_hexagonal()
        return refine_solution(circles)


# EVOLVE-BLOCK-END
