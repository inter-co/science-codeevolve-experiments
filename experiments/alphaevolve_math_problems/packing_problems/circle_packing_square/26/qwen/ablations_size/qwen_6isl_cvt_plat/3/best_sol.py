# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from scipy.spatial import cKDTree
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with gradient-based optimization
    and local refinement.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 26
    
    # Try multiple optimization runs with different initializations to find the best solution
    best_circles = None
    best_sum = -float('inf')
    
    # Run multiple optimization attempts with different initializations to find the best solution
    # Following INSPIRATION 1 approach with more attempts and better refinement
    for attempt in range(20):  # Increase attempts to 20 for better chance of finding good solution
        # Initialize circles using improved hexagonal packing pattern
        circles = initialize_improved_hexagonal_packing(n)
        
        # Optimize using gradient-based method with constraints
        optimized_circles = optimize_circles(circles)
        
        # Apply local refinement for further improvement
        refined_circles = local_refinement(optimized_circles)
        
        # Evaluate this solution
        current_sum = np.sum(refined_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = refined_circles.copy()
    
    return best_circles

def initialize_improved_hexagonal_packing(n: int) -> np.ndarray:
    """Initialize circles using a more sophisticated hexagonal packing pattern"""
    # Create a 5x5 grid pattern for 25 circles, then add one more
    rows = 5
    cols = 5
    
    # Calculate spacing
    spacing_x = 0.9 / (cols - 1) if cols > 1 else 0.5
    spacing_y = 0.9 / (rows - 1) if rows > 1 else 0.5
    
    # Determine radius based on spacing
    min_radius = min(spacing_x, spacing_y) * 0.3
    
    circles = []
    count = 0
    
    # Create hexagonal grid pattern
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
                
            # Hexagonal offset for odd rows
            x_offset = 0.5 if i % 2 == 1 else 0.0
            x = 0.05 + (j + x_offset) * spacing_x
            y = 0.05 + i * spacing_y
            
            # Add small random perturbation to avoid perfect symmetry
            x += np.random.uniform(-0.005, 0.005)
            y += np.random.uniform(-0.005, 0.005)
            
            # Ensure within bounds
            x = max(min_radius, min(1.0 - min_radius, x))
            y = max(min_radius, min(1.0 - min_radius, y))
            
            circles.append([x, y, min_radius])
            count += 1
            
        if count >= n:
            break
    
    # Fill remaining circles with random positions
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        # Larger initial radius for diversity
        r = np.random.uniform(0.02, 0.08)
        circles.append([x, y, r])
    
    return np.array(circles)

def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using constrained optimization"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(params):
        circles = params.reshape((n, 3))
        radii_sum = np.sum(circles[:, 2])
        return -radii_sum  # Negative because we minimize
    
    # Define constraints with better handling
    def containment_constraint(params):
        circles = params.reshape((n, 3))
        constraints = []
        
        # Each circle must be fully contained in unit square
        for i in range(n):
            x, y, r = circles[i]
            # r <= x <= 1-r and r <= y <= 1-r
            constraints.append(x - r)  # x >= r
            constraints.append(1 - x - r)  # 1-x >= r, so x+r <= 1
            constraints.append(y - r)  # y >= r
            constraints.append(1 - y - r)  # 1-y >= r, so y+r <= 1
            
        return np.array(constraints)
    
    def non_overlap_constraint(params):
        circles = params.reshape((n, 3))
        constraints = []
        
        # Use spatial indexing to reduce number of checks
        # But still do full check for correctness
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Distance between centers
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # We want dist >= r1 + r2, so we constrain dist_sq >= min_dist_sq
                # This means: dist_sq - min_dist_sq >= 0
                constraints.append(dist_sq - min_dist_sq)
                
        return np.array(constraints)
    
    # Set up constraints for scipy.optimize
    cons = []
    
    # Containment constraints (each must be >= 0)
    cons.append({'type': 'ineq', 'fun': lambda p: containment_constraint(p)})
    
    # Non-overlap constraints (each must be >= 0)
    cons.append({'type': 'ineq', 'fun': lambda p: non_overlap_constraint(p)})
    
    # Bounds for parameters: [x, y, r] for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Perform optimization with better settings - inspired by INSPIRATION 1
    try:
        # Try multiple optimization methods to get better results
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_sum = -float('inf')
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}
                )
                
                if result.success:
                    # Check if this is better than our previous best
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception:
                continue
                
        if best_result is not None and best_result.success:
            optimized_circles = best_result.x.reshape((n, 3))
            # Ensure all circles are valid
            for i in range(n):
                x, y, r = optimized_circles[i]
                # Clamp to valid ranges
                optimized_circles[i] = [
                    max(0.001, min(0.999, x)),
                    max(0.001, min(0.999, y)), 
                    max(0.001, min(0.499, r))
                ]
            return optimized_circles
    except Exception as e:
        pass
    
    # Fallback: return initial configuration if optimization fails
    return initial_circles

def local_refinement(circles: np.ndarray) -> np.ndarray:
    """Apply local refinement to improve the solution with enhanced strategies"""
    # Use a more sophisticated local search approach inspired by INSPIRATION 1
    n = len(circles)
    best_circles = circles.copy()
    best_sum = np.sum(best_circles[:, 2])
    
    # Enhanced local search with multiple strategies - following INSPIRATION 1 approach
    for round_num in range(20):  # More rounds for better refinement
        # Strategy 1: Random single-circle perturbations (larger perturbations early)
        for _ in range(3000):  # Many more attempts for better search
            # Copy current solution
            test_circles = best_circles.copy()
            
            # Select a random circle to perturb
            idx = random.randint(0, n - 1)
            
            # Different perturbation sizes based on round - more aggressive early on
            # Start with larger perturbations and decrease over time
            perturbation_scale = 0.02 * (1.0 - round_num * 0.03)
            
            # Small random perturbation
            test_circles[idx, 0] += np.random.normal(0, perturbation_scale)
            test_circles[idx, 1] += np.random.normal(0, perturbation_scale)
            test_circles[idx, 2] += np.random.normal(0, perturbation_scale * 0.3)
            
            # Ensure constraints
            x, y, r = test_circles[idx]
            max_r = min(x, 1-x, y, 1-y)
            test_circles[idx, 2] = min(max_r, max(0.001, r))
            
            # Clamp positions
            test_circles[idx, 0] = max(0.001, min(0.999, test_circles[idx, 0]))
            test_circles[idx, 1] = max(0.001, min(0.999, test_circles[idx, 1]))
            
            # Check if the new configuration is valid
            if is_valid_configuration(test_circles):
                new_sum = np.sum(test_circles[:, 2])
                if new_sum > best_sum:
                    best_circles = test_circles
                    best_sum = new_sum
    
    return best_circles

def is_valid_configuration(circles):
    """Check if the entire configuration is valid"""
    n = len(circles)
    
    # Check containment with stricter bounds to prevent numerical issues
    for i in range(n):
        x, y, r = circles[i]
        if not (r + 1e-10 <= x <= 1 - r - 1e-10 and r + 1e-10 <= y <= 1 - r - 1e-10):
            return False
    
    # Check overlaps with tolerance for floating-point precision
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # Use a small tolerance to handle floating point precision issues
            if distance < (r1 + r2) - 1e-10:
                return False
    
    return True


# EVOLVE-BLOCK-END
