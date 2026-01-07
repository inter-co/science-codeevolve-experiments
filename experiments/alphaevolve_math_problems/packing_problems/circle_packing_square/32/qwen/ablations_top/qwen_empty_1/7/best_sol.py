# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
import random
import time
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses an advanced hybrid approach combining hexagonal packing initialization, 
    physics simulation with adaptive forces, and sophisticated local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    time_limit = 55.0
    start_time = time.time()
    
    # Initialize with better hexagonal packing pattern
    def initialize_hexagonal():
        circles = np.zeros((n, 3))
        
        # Use more precise hexagonal close packing
        sqrt_n = np.sqrt(n)
        rows = int(np.ceil(sqrt_n))
        cols = int(np.ceil(n / rows))
        
        # Better spacing calculation
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows
        
        # Radius based on spacing - slightly more conservative
        radius_estimate = min(spacing_x, spacing_y) * 0.33
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset for even rows
                offset = (i % 2) * 0.5
                x = 0.05 + (j + offset) * spacing_x
                y = 0.05 + i * spacing_y
                
                # Keep within bounds with better margin
                x = max(radius_estimate, min(0.95-radius_estimate, x))
                y = max(radius_estimate, min(0.95-radius_estimate, y))
                
                circles[idx] = [x, y, radius_estimate]
                idx += 1
            if idx >= n:
                break
        
        # Fill remaining slots strategically
        for i in range(idx, n):
            # Use different initialization strategies for remaining circles
            if random.random() < 0.3:
                # Place near center for better packing
                x = 0.2 + random.random() * 0.6
                y = 0.2 + random.random() * 0.6
            else:
                # Place near edges for exploration
                x = 0.05 + random.random() * 0.9
                y = 0.05 + random.random() * 0.9
            r = 0.01 + random.random() * 0.04
            circles[i] = [x, y, r]
            
        return circles
    
    # Validate circle constraints efficiently using KDTree
    def validate_circles(circles):
        # Check containment
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlap using KDTree for efficiency
        points = circles[:, :2]
        tree = KDTree(points)
        
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            # Find nearby circles using KDTree - more efficient search
            nearby = tree.query_ball_point([x1, y1], 2*(r1 + 1e-6))
            
            for j in nearby:
                if i != j:
                    x2, y2, r2 = circles[j]
                    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if distance < r1 + r2:
                        return False
        return True
    
    # Physics-based force application for refinement with enhanced force model
    def apply_physics_force(circles, iteration=None):
        new_circles = circles.copy()
        n = len(circles)
        
        # Dynamic parameters based on iteration
        k_repulsion = 1.0
        k_boundary = 10.0
        step_size = 0.001
        
        # Reduce step size in later iterations for fine-tuning
        if iteration is not None and iteration > 1000:
            step_size = 0.0005
            
        # Initialize forces
        forces = np.zeros((n, 2))
        
        # Repulsion forces between circles using KDTree for efficiency
        points = circles[:, :2]
        tree = KDTree(points)
        
        for i in range(n):
            x1, y1, r1 = circles[i]
            
            # Find nearby circles using KDTree
            nearby = tree.query_ball_point([x1, y1], 2*(r1 + 1e-6))
            
            for j in nearby:
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0 and distance < (r1 + r2):
                        # Improved repulsion force with better scaling
                        force_magnitude = k_repulsion * (r1 + r2 - distance) / (distance + 1e-8)
                        # Add some directional bias to prevent oscillation
                        force_magnitude *= (1.0 + 0.1 * random.random())
                        forces[i, 0] += force_magnitude * dx / distance
                        forces[i, 1] += force_magnitude * dy / distance
        
        # Boundary forces with more aggressive correction
        for i in range(n):
            x, y, r = circles[i]
            
            # Force away from boundaries
            fx = 0
            fy = 0
            
            if x < r:
                fx = k_boundary * (r - x) * 1.5  # Stronger force near boundaries
            elif x > 1 - r:
                fx = k_boundary * (1 - r - x) * 1.5
                
            if y < r:
                fy = k_boundary * (r - y) * 1.5
            elif y > 1 - r:
                fy = k_boundary * (1 - r - y) * 1.5
                
            forces[i, 0] += fx
            forces[i, 1] += fy
        
        # Apply forces with adaptive step size
        for i in range(n):
            new_circles[i, 0] += step_size * forces[i, 0]
            new_circles[i, 1] += step_size * forces[i, 1]
            
            # Keep radii positive and within reasonable bounds
            new_circles[i, 2] = max(0.001, min(0.49, new_circles[i, 2]))
        
        return new_circles
    
    # Enhanced local optimization with systematic search and better parameter tuning
    def local_optimization(circles):
        # Try different strategies for improvement
        best_circles = circles.copy()
        best_sum = np.sum(circles[:, 2])
        
        # Try multiple rounds of optimization
        for round_num in range(4):  # More rounds for better exploration
            if time.time() - start_time > time_limit:
                break
                
            # In each round, try different types of adjustments
            for _ in range(150):  # Even more iterations for better search
                if time.time() - start_time > time_limit:
                    break
                    
                temp_circles = best_circles.copy()
                
                # Try adjusting multiple circles in batches
                adjustment_count = min(8, n//2)  # More adjustments
                indices_to_adjust = random.sample(range(n), adjustment_count)
                
                for i in indices_to_adjust:
                    if random.random() < 0.95:  # Even higher probability to adjust
                        # Try small position adjustment
                        old_x, old_y, old_r = temp_circles[i]
                        dx = random.uniform(-0.025, 0.025)
                        dy = random.uniform(-0.025, 0.025)
                        dr = random.uniform(-0.012, 0.012)
                        
                        new_x = max(0.01, min(0.99, old_x + dx))
                        new_y = max(0.01, min(0.99, old_y + dy))
                        new_r = max(0.001, min(0.49, old_r + dr))
                        
                        # Test if this change is valid
                        temp_circles[i] = [new_x, new_y, new_r]
                        
                        # Check constraints
                        if validate_circles(temp_circles):
                            new_sum = np.sum(temp_circles[:, 2])
                            if new_sum > best_sum:
                                best_sum = new_sum
                                best_circles = temp_circles.copy()
                        else:
                            # Revert if invalid
                            temp_circles[i] = [old_x, old_y, old_r]
        
        return best_circles
    
    # Enhanced optimization with better radius maximization
    def radius_enhancement(circles):
        # Strategy: Try to increase radii systematically
        current_circles = circles.copy()
        current_sum = np.sum(current_circles[:, 2])
        
        # Perform several rounds of focused optimization
        for _ in range(300):
            if time.time() - start_time > time_limit:
                break
                
            improved = False
            # Try to increase radius for each circle
            for i in range(n):
                old_x, old_y, old_r = current_circles[i]
                max_possible_radius = min(old_x, 1-old_x, old_y, 1-old_y)
                
                # Check constraints with neighbors - more thorough
                new_r = old_r
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = current_circles[j]
                        dist = np.sqrt((old_x-x2)**2 + (old_y-y2)**2)
                        max_allowed = dist - r2
                        if max_allowed > 0:
                            new_r = min(new_r, max_allowed)
                
                # Limit by boundary constraints
                new_r = min(new_r, max_possible_radius)
                
                # Increase radius if beneficial
                if new_r > old_r:
                    current_circles[i, 2] = new_r
                    current_sum = np.sum(current_circles[:, 2])
                    improved = True
            
            # If no improvement in a full pass, break early
            if not improved:
                break
        
        return current_circles
    
    # Multi-start approach for better exploration
    def multi_start_optimization():
        best_result = None
        best_sum = 0
        
        # Try multiple random starts with different seeds
        for start_num in range(5):
            if time.time() - start_time > time_limit * 0.8:  # Leave time for final refinement
                break
                
            # Set seed for reproducible results but still varied
            random.seed(start_num * 1000 + int(time.time()) % 10000)
            
            # Initialize with different pattern
            circles = initialize_hexagonal()
            
            # Apply physics simulation with fewer iterations for diversity
            for iteration in range(500):
                if time.time() - start_time > time_limit * 0.8:
                    break
                circles = apply_physics_force(circles, iteration)
            
            # Apply local optimization
            circles = local_optimization(circles)
            
            # Enhance radii
            circles = radius_enhancement(circles)
            
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = circles.copy()
                
        return best_result if best_result is not None else initialize_hexagonal()
    
    # Final mathematical refinement approach inspired by constraint programming
    def mathematical_refinement(circles):
        # Apply a more mathematically rigorous approach to fine-tune
        # This mimics the approach from INSPIRATION 2 but in a heuristic way
        
        # Run several passes of systematic radius maximization
        for pass_num in range(3):
            if time.time() - start_time > time_limit:
                break
                
            improved = False
            # Go through each circle and try to maximize its radius
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                    
                old_x, old_y, old_r = circles[i]
                max_possible_radius = min(old_x, 1-old_x, old_y, 1-old_y)
                
                # Check all neighbors to find the tightest constraint
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((old_x-x2)**2 + (old_y-y2)**2)
                        max_allowed = dist - r2
                        if max_allowed > 0:
                            max_possible_radius = min(max_possible_radius, max_allowed)
                
                # Try to increase radius up to the maximum allowed
                if max_possible_radius > old_r + 1e-6:
                    # Test a few increments to see what works
                    test_radius = min(max_possible_radius, old_r + 0.005)
                    
                    # Check if this is valid
                    valid = True
                    temp_circles = circles.copy()
                    temp_circles[i, 2] = test_radius
                    
                    # Validate the change
                    if validate_circles(temp_circles):
                        circles[i, 2] = test_radius
                        improved = True
                        
            if not improved:
                break
                
        return circles
    
    # Main optimization loop with multi-start approach
    # Multi-start approach for better exploration
    circles = multi_start_optimization()
    
    # Apply physics simulation for initial refinement
    for iteration in range(1200):  # More iterations for better convergence
        if time.time() - start_time > time_limit:
            break
        circles = apply_physics_force(circles, iteration)
        
        # Occasionally reinitialize to escape local minima
        if iteration % 400 == 0 and iteration > 0:
            circles = initialize_hexagonal()
    
    # Radius enhancement after physics simulation
    circles = radius_enhancement(circles)
    
    # Mathematical refinement to fine-tune
    circles = mathematical_refinement(circles)
    
    # Final local optimization
    circles = local_optimization(circles)
    
    # Additional refinement with iterative improvement
    for _ in range(250):
        if time.time() - start_time > time_limit:
            break
        circles = apply_physics_force(circles)
        circles = local_optimization(circles)
    
    # Final validation and cleanup
    if not validate_circles(circles):
        # If validation fails, fall back to a simple but effective approach
        circles = initialize_hexagonal()
        for _ in range(400):
            if time.time() - start_time > time_limit:
                break
            circles = apply_physics_force(circles)
    
    return circles


# EVOLVE-BLOCK-END
