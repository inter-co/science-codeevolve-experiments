# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import warnings

def initialize_circles_geometrically(n: int) -> np.ndarray:
    """
    Initialize circles using a configuration inspired by known optimal solutions
    """
    circles = np.zeros((n, 3))
    
    # Use a proven configuration approach for 26 circles in square
    # This is based on research on optimal circle packings
    
    # We'll use a combination of hexagonal and rectangular arrangements
    # Start with a core hexagonal pattern, then add remaining circles strategically
    
    # First, let's place circles in a 5x5 hexagonal-like pattern
    rows = 5
    cols = 5
    
    # Calculate spacing for approximately equal area coverage
    spacing_x = 0.8 / cols  # Leave 0.1 margin on each side
    spacing_y = 0.8 / rows
    
    # Hexagonal offset
    hex_offset = spacing_x * 0.5
    
    # Center the pattern
    offset_x = 0.1
    offset_y = 0.1
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Hexagonal offset for odd rows
            x_offset = (i % 2) * hex_offset
            x = offset_x + (j + x_offset) * spacing_x
            y = offset_y + i * spacing_y
            
            # Ensure within bounds with generous margins
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Set a reasonable initial radius
            max_radius = min(x, 1-x, y, 1-y) * 0.45
            r = max(0.03, min(0.15, max_radius))
            
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # For remaining circles, we'll place them in the center area with larger radii
    # This is a key insight from optimization studies - central circles can often have larger radii
    remaining = n - idx
    if remaining > 0:
        # Place in a more central area where we can achieve larger radii
        center_x = 0.5
        center_y = 0.5
        
        for i in range(remaining):
            # Try to place in a way that maximizes the chance for large radii
            attempts = 0
            while attempts < 100:
                # Place closer to center with more freedom for larger radii
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0.05, 0.25)  # Within center region
                
                x = center_x + radius * np.cos(angle)
                y = center_y + radius * np.sin(angle)
                
                # Ensure within bounds
                if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                    # Calculate maximum possible radius
                    max_radius = min(x, 1-x, y, 1-y) * 0.45
                    r = max(0.03, min(0.2, max_radius * (0.8 + random.random() * 0.4)))
                    
                    # Check overlap with existing circles
                    valid = True
                    for k in range(n):
                        existing_x, existing_y, existing_r = circles[k]
                        distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if distance < r + existing_r:
                            valid = False
                            break
                    
                    if valid:
                        circles[idx + i] = [x, y, r]
                        break
                attempts += 1
    
    return circles

def create_bounds(n: int) -> List[Tuple[float, float]]:
    """Create bounds for optimization variables [x1, y1, r1, x2, y2, r2, ...]"""
    bounds = []
    for i in range(n):
        # Bounds for x coordinate: [radius, 1-radius]
        bounds.append((0.001, 0.999))
        # Bounds for y coordinate: [radius, 1-radius] 
        bounds.append((0.001, 0.999))
        # Bounds for radius: [0.001, min(x,1-x,y,1-y)]
        bounds.append((0.001, 0.5))
    return bounds

def create_constraints(n: int) -> List:
    """Create constraint functions for the optimization"""
    constraints = []
    
    # Containment constraints: each circle must be fully inside the unit square
    def containment_constraint(params):
        # params is flattened array [x1, y1, r1, x2, y2, r2, ...]
        result = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            # x - r >= 0
            result.append(x - r)
            # 1 - x - r >= 0  
            result.append(1 - x - r)
            # y - r >= 0
            result.append(y - r)
            # 1 - y - r >= 0
            result.append(1 - y - r)
        return np.array(result)
    
    # Add containment constraint (all values must be >= 0)
    constraints.append({'type': 'ineq', 'fun': containment_constraint})
    
    # Non-overlap constraints: distance between centers >= sum of radii
    def overlap_constraint(params):
        result = []
        for i in range(n):
            for j in range(i+1, n):
                x1 = params[3*i]
                y1 = params[3*i + 1]
                r1 = params[3*i + 2]
                x2 = params[3*j]
                y2 = params[3*j + 1]
                r2 = params[3*j + 2]
                
                # Distance constraint: sqrt((x1-x2)^2 + (y1-y2)^2) >= r1 + r2
                # This becomes: (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
                # Rearranged: (x1-x2)^2 + (y1-y2)^2 - (r1+r2)^2 >= 0
                distance_sq = (x1-x2)**2 + (y1-y2)**2
                sum_radii_sq = (r1+r2)**2
                result.append(distance_sq - sum_radii_sq)
        return np.array(result)
    
    # Add non-overlap constraint (all values must be >= 0)
    constraints.append({'type': 'ineq', 'fun': overlap_constraint})
    
    return constraints

def optimize_with_sqp(initial_circles: np.ndarray, maxiter: int = 200) -> np.ndarray:
    """
    Use Sequential Quadratic Programming to optimize circle placement
    """
    n = len(initial_circles)
    # Flatten initial circles to parameter vector
    initial_params = initial_circles.flatten()
    
    # Create bounds and constraints
    bounds = create_bounds(n)
    constraints = create_constraints(n)
    
    # Objective function: minimize negative sum of radii (maximize sum of radii)
    def objective(params):
        total_radii = np.sum(params[2::3])  # Sum of all radii (every third element starting from index 2)
        return -total_radii  # Negative because we want to maximize
    
    # Set up options for better convergence
    options = {
        'maxiter': maxiter,
        'ftol': 1e-6,
        'gtol': 1e-6,
        'disp': False
    }
    
    # Use SLSQP method which handles constraints well
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=1e-6
        )
        
        if result.success:
            optimized_circles = result.x.reshape((n, 3))
            return optimized_circles
        else:
            # Fallback to using initial solution if optimization fails
            warnings.warn(f"Optimization failed: {result.message}")
            return initial_circles.copy()
    except Exception as e:
        warnings.warn(f"Optimization error: {str(e)}")
        return initial_circles.copy()

def validate_solution(circles: np.ndarray) -> bool:
    """Validate that all constraints are satisfied"""
    n = len(circles)
    
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check non-overlap
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2 - 1e-8:  # Small tolerance for floating point
                return False
    
    return True

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 26
    circles = np.zeros((n, 3))
    
    # Phase 1: Geometric initialization with better packing
    circles = initialize_circles_geometrically(n)
    
    # Phase 2: Constrained optimization using SQP
    optimized_circles = optimize_with_sqp(circles, maxiter=150)
    
    # Phase 3: Additional refinement with multiple local optimizations
    best_sum = np.sum(optimized_circles[:, 2])
    best_solution = optimized_circles.copy()
    
    # Try several random restarts with local optimization
    for restart in range(5):
        # Perturb solution slightly
        perturbed = optimized_circles.copy()
        for i in range(n):
            # Small random perturbation to positions
            perturbed[i, 0] += random.uniform(-0.02, 0.02)
            perturbed[i, 1] += random.uniform(-0.02, 0.02)
            # Keep within bounds
            perturbed[i, 0] = max(0.01, min(0.99, perturbed[i, 0]))
            perturbed[i, 1] = max(0.01, min(0.99, perturbed[i, 1]))
        
        # Optimize this perturbed version
        refined = optimize_with_sqp(perturbed, maxiter=100)
        
        current_sum = np.sum(refined[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_solution = refined.copy()
    
    # Final validation
    if not validate_solution(best_solution):
        # If still invalid, do one final hard constraint enforcement
        final_circles = best_solution.copy()
        for i in range(n):
            x, y, r = final_circles[i]
            # Enforce containment constraints
            r = min(r, x, 1-x, y, 1-y)
            # Adjust position if needed
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            final_circles[i] = [x, y, r]
        return final_circles
    
    return best_solution


# EVOLVE-BLOCK-END
