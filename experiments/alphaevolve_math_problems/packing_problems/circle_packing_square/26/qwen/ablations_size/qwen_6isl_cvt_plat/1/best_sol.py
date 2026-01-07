# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal packing initialization with local optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For deterministic results
    random.seed(42)
    
    n = 26
    
    # Multiple initialization strategies to find good starting points
    best_circles = None
    best_sum = 0
    
    # Try different initialization strategies
    for strategy in range(4):  # Use 4 strategies as in inspirations
        if strategy == 0:
            # Standard hexagonal packing
            circles = initialize_hexagonal_packing(n)
        elif strategy == 1:
            # Slightly perturbed hexagonal packing
            circles = initialize_hexagonal_packing(n)
            circles = perturb_configuration(circles, 0.1, 0.1)
        elif strategy == 2:
            # Corner + center initialization
            circles = initialize_corner_center_packing(n)
        else:
            # Random initialization with good distribution
            circles = initialize_random_distribution(n)
        
        # Multiple refinement attempts for each strategy
        for attempt in range(4):  # Use 4 attempts per strategy as in inspirations
            # Refine using optimization
            optimized = optimize_circles(circles)
            
            # Local search refinement
            refined = local_search_refinement(optimized)
            
            # Check if this attempt improved the solution
            current_sum = np.sum(refined[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = refined.copy()
            
            # Add small random perturbation for next attempt to escape local minima
            if attempt < 3:  # Don't perturb on last attempt
                circles = perturb_configuration(best_circles, 0.05, 0.05)
    
    # Final additional refinement with specialized approach
    if best_circles is not None:
        # Try one final optimization with different settings to fine-tune
        final_refinement = optimize_circles(best_circles)
        final_refinement = local_search_refinement(final_refinement)
        final_sum = np.sum(final_refinement[:, 2])
        if final_sum > best_sum:
            best_circles = final_refinement
    
    return best_circles if best_circles is not None else initialize_hexagonal_packing(n)

def initialize_hexagonal_packing(n: int) -> np.ndarray:
    """Initialize circle positions using a hexagonal packing pattern."""
    # Determine grid dimensions for approximately n circles
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Create hexagonal grid
    circles = np.zeros((n, 3))
    
    # Hexagonal packing parameters
    sqrt3 = math.sqrt(3)
    spacing_x = 1.0 / max(cols, 1)
    spacing_y = sqrt3 * spacing_x / 2
    
    # Place circles in hexagonal pattern
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
                
            # Offset every other row
            offset = (i % 2) * spacing_x / 2
            x = offset + (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Ensure we're within the unit square
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Initial radius - small enough to fit in the cell
                r = min(spacing_x, spacing_y) / 4
                circles[count] = [x, y, r]
                count += 1
                
        if count >= n:
            break
    
    # Fill remaining positions with small circles if needed
    for i in range(count, n):
        circles[i] = [0.5, 0.5, 0.01]
        
    return circles

def initialize_corner_center_packing(n: int) -> np.ndarray:
    """Initialize circles using corner and center positions."""
    circles = np.zeros((n, 3))
    
    # Place circles at strategic positions
    # Corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    # Edges
    edges = [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]
    # Center
    center = [(0.5, 0.5)]
    
    positions = corners + edges + center
    
    # Place circles at strategic positions with larger initial radii
    for i in range(min(len(positions), n)):
        x, y = positions[i]
        # Use larger initial radius for these positions
        r = min(x, y, 1-x, 1-y) * 0.3
        circles[i] = [x, y, r]
    
    # Fill remaining positions with hexagonal pattern
    remaining = n - len(positions)
    if remaining > 0:
        rows = int(math.ceil(math.sqrt(remaining)))
        cols = int(math.ceil(remaining / rows))
        
        spacing_x = 0.8 / (cols + 1)
        spacing_y = 0.8 / (rows + 1)
        
        count = len(positions)
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                    
                # Offset every other row
                offset = (i % 2) * spacing_x / 2
                x = 0.1 + offset + (j + 1) * spacing_x
                y = 0.1 + (i + 1) * spacing_y
                
                # Ensure we're within the unit square
                if 0 <= x <= 1 and 0 <= y <= 1:
                    # Initial radius - smaller than corner ones
                    r = min(spacing_x, spacing_y) * 0.25
                    circles[count] = [x, y, r]
                    count += 1
                    
            if count >= n:
                break
    
    # Fill any remaining positions with small circles
    for i in range(count, n):
        circles[i] = [0.5, 0.5, 0.01]
        
    return circles

def initialize_random_distribution(n: int) -> np.ndarray:
    """Initialize circles with a more random but structured distribution."""
    circles = np.zeros((n, 3))
    
    # Place some circles in a more scattered pattern
    for i in range(n):
        # Distribute more evenly in the square
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        # Use a moderate radius to start with
        r = random.uniform(0.02, 0.15)
        circles[i] = [x, y, r]
        
    return circles

def perturb_configuration(circles: np.ndarray, dx_range: float = 0.015, dy_range: float = 0.015) -> np.ndarray:
    """Add small random perturbations to escape local minima."""
    perturbed = circles.copy()
    
    # Perturb positions slightly
    for i in range(len(perturbed)):
        if random.random() < 0.3:  # 30% chance to perturb each circle
            dx = random.uniform(-dx_range, dx_range)
            dy = random.uniform(-dy_range, dy_range)
            
            x, y, r = perturbed[i]
            new_x = max(0.001, min(0.999, x + dx))
            new_y = max(0.001, min(0.999, y + dy))
            perturbed[i] = [new_x, new_y, r]
    
    return perturbed

def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using scipy.optimize."""
    n = len(initial_circles)
    
    # Flatten initial circles for optimization
    initial_flat = initial_circles.flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds  
        bounds.append((0.001, 0.999))
        # r bounds (positive but less than min of x, y, 1-x, 1-y)
        bounds.append((0.001, 0.499))
    
    # Objective function: minimize negative sum of radii (equivalent to maximizing sum of radii)
    def objective(x):
        circles = x.reshape((n, 3))
        return -np.sum(circles[:, 2])
    
    # Constraint functions
    def constraint_func(x):
        circles = x.reshape((n, 3))
        constraints = []
        
        # Position constraints: circle must fit completely in unit square
        for i in range(n):
            x_pos, y_pos, r = circles[i]
            # r <= x, r <= y, r <= 1-x, r <= 1-y
            constraints.extend([
                x_pos - r,      # x >= r
                y_pos - r,      # y >= r
                1 - x_pos - r,  # 1-x >= r
                1 - y_pos - r   # 1-y >= r
            ])
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                constraints.append(dist - r1 - r2)  # distance >= r1 + r2
                
        return np.array(constraints)
    
    # Create constraint dictionary
    constraints = {'type': 'ineq', 'fun': constraint_func}
    
    try:
        # Use SLSQP optimizer for constrained optimization with more iterations and tighter tolerances
        # Try multiple optimization runs with different settings for better convergence
        best_result = None
        best_value = float('inf')
        
        for run in range(3):  # Run optimization 3 times with different settings
            # Vary the optimizer settings slightly for each run
            options = {'maxiter': 5000, 'ftol': 1e-8, 'eps': 1e-8}
            
            # Add a small random perturbation to initial point for diversity
            if run > 0:
                perturbed_initial = initial_flat.copy()
                for i in range(0, len(perturbed_initial), 3):  # Perturb x, y, r for each circle
                    perturbed_initial[i] += random.uniform(-0.001, 0.001)  # x
                    perturbed_initial[i+1] += random.uniform(-0.001, 0.001)  # y
                    # Don't perturb radius as it might be invalid
                result = minimize(
                    objective,
                    perturbed_initial,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options=options
                )
            else:
                result = minimize(
                    objective,
                    initial_flat,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options=options
                )
            
            if result.success and result.fun < best_value:
                best_value = result.fun
                best_result = result
        
        if best_result is not None:
            optimized_circles = best_result.x.reshape((n, 3))
            # Ensure all values are valid
            for i in range(n):
                optimized_circles[i, 0] = max(0.001, min(0.999, optimized_circles[i, 0]))
                optimized_circles[i, 1] = max(0.001, min(0.999, optimized_circles[i, 1]))
                optimized_circles[i, 2] = max(0.001, min(0.499, optimized_circles[i, 2]))
            return optimized_circles
        else:
            # If optimization fails, return initial configuration
            return initial_circles
    except Exception as e:
        # Return initial configuration if optimization fails
        return initial_circles

def local_search_refinement(circles: np.ndarray) -> np.ndarray:
    """Apply local search refinement to improve the solution."""
    n = len(circles)
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Try to locally improve each circle
    for iteration in range(3000):  # Even more iterations
        improved = False
        # Try to increase radii first
        for i in range(n):
            test_circles = best_circles.copy()
            x, y, r = test_circles[i]
            
            # Try to increase radius with more careful step size
            new_r = min(r + 0.0002, 0.499, x, y, 1-x, 1-y)  # Smaller step size for precision
            test_circles[i] = [x, y, new_r]
            
            # Check if all constraints are satisfied
            if is_valid_placement(test_circles):
                new_sum = np.sum(test_circles[:, 2])
                if new_sum > best_sum:
                    best_circles = test_circles.copy()
                    best_sum = new_sum
                    improved = True
        
        # If no improvement was made, try small position adjustments
        if not improved:
            for i in range(n):
                test_circles = best_circles.copy()
                x, y, r = test_circles[i]
                
                # Even smaller random movements
                dx = random.uniform(-0.001, 0.001)
                dy = random.uniform(-0.001, 0.001)
                
                new_x = max(0.001, min(0.999, x + dx))
                new_y = max(0.001, min(0.999, y + dy))
                
                test_circles[i] = [new_x, new_y, r]
                
                # Check if all constraints are satisfied
                if is_valid_placement(test_circles):
                    new_sum = np.sum(test_circles[:, 2])
                    if new_sum > best_sum:
                        best_circles = test_circles.copy()
                        best_sum = new_sum
                        improved = True
        
        if not improved:
            break
    
    return best_circles

def is_valid_placement(circles: np.ndarray) -> bool:
    """Check if all circles are valid (inside square and non-overlapping)."""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints efficiently using pairwise comparison
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            if dist_sq < (r1 + r2)**2:
                return False
                
    return True


# EVOLVE-BLOCK-END
