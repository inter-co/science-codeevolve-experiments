# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import math
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid geometric construction and mathematical programming approach.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Start with a systematic geometric construction
    circles = construct_geometrically(n)
    
    # Refine using mathematical programming with proper constraints
    circles = optimize_mathematically(circles)
    
    # Final local refinement
    circles = local_refinement(circles)
    
    return circles

def construct_geometrically(n):
    """Construct initial configuration using systematic geometric approach"""
    circles = np.zeros((n, 3))
    
    # Method: Arrange in a grid-like pattern but allow for optimization
    # For 32 circles, we can arrange in a roughly 5x7 grid with some adjustments
    
    # Grid dimensions
    rows = 5
    cols = 7
    
    # Calculate spacing to fit in unit square
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Initial radius - based on spacing and some margin for optimization
    max_radius = min(spacing_x, spacing_y) * 0.4
    
    # Place circles in grid with slight randomization to avoid symmetry
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Position in grid
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Add small random perturbation
            x += np.random.uniform(-spacing_x/8, spacing_x/8)
            y += np.random.uniform(-spacing_y/8, spacing_y/8)
            
            # Ensure within bounds
            x = np.clip(x, max_radius, 1 - max_radius)
            y = np.clip(y, max_radius, 1 - max_radius)
            
            circles[idx] = [x, y, max_radius]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with careful placement
    for i in range(idx, n):
        circles[i] = place_circle_carefully(circles[:i], max_radius)
    
    return circles

def place_circle_carefully(existing_circles, max_radius):
    """Place a new circle with proper constraint checking"""
    best_radius = 0
    best_pos = None
    
    # Sample positions more intelligently
    for attempt in range(1000):
        # Try to place near the edge or in gaps
        if attempt < 500:
            # Random placement in valid region
            x = np.random.uniform(max_radius, 1 - max_radius)
            y = np.random.uniform(max_radius, 1 - max_radius)
        else:
            # Try to place near existing circles to find gaps
            if len(existing_circles) > 0:
                # Pick a random existing circle
                idx = np.random.randint(len(existing_circles))
                cx, cy, cr = existing_circles[idx]
                # Place nearby
                angle = np.random.uniform(0, 2*np.pi)
                distance = cr + max_radius + np.random.uniform(0, max_radius/2)
                x = cx + distance * np.cos(angle)
                y = cy + distance * np.sin(angle)
                # Clamp to valid region
                x = np.clip(x, max_radius, 1 - max_radius)
                y = np.clip(y, max_radius, 1 - max_radius)
            else:
                x = np.random.uniform(max_radius, 1 - max_radius)
                y = np.random.uniform(max_radius, 1 - max_radius)
        
        # Calculate maximum possible radius at this position
        max_possible_radius = min(x, 1-x, y, 1-y)
        
        # Check distance to existing circles
        for circle in existing_circles:
            cx, cy, cr = circle
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            max_possible_radius = min(max_possible_radius, dist - cr)
        
        # Accept if this gives us a better radius
        if max_possible_radius > best_radius and max_possible_radius > 0.001:
            best_radius = max_possible_radius
            best_pos = (x, y, max_possible_radius)
    
    # If we couldn't find a good placement, just return a basic one
    if best_pos is None:
        x = np.random.uniform(max_radius, 1 - max_radius)
        y = np.random.uniform(max_radius, 1 - max_radius)
        return [x, y, max_radius]
    
    return best_pos

def optimize_mathematically(circles):
    """Use mathematical programming approach with strict constraint handling"""
    n = len(circles)
    
    # Set up variables and bounds
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
    initial_params = []
    bounds = []
    
    for i in range(n):
        x, y, r = circles[i]
        initial_params.extend([x, y, r])
        # Bounds: x in [r, 1-r], y in [r, 1-r], r in [0.001, 0.5]
        bounds.extend([(r, 1-r), (r, 1-r), (0.001, 0.5)])
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        total_radius = sum(params[3*i + 2] for i in range(n))
        return -total_radius  # Negative because we want to maximize
    
    # Constraint function: check all pairwise non-overlap constraints
    def constraint_func(params):
        constraints = []
        for i in range(n):
            x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
            
            # Boundary constraints (hard constraints)
            constraints.append(x1 - r1)  # x >= r
            constraints.append(1 - x1 - r1)  # x <= 1-r
            constraints.append(y1 - r1)  # y >= r
            constraints.append(1 - y1 - r1)  # y <= 1-r
            
            # Non-overlap constraints
            for j in range(i+1, n):
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                # We want dist_sq >= min_dist_sq (so that dist >= r1 + r2)
                # This constraint is satisfied when: dist_sq - min_dist_sq >= 0
                constraints.append(dist_sq - min_dist_sq)
        
        return np.array(constraints)
    
    # Try different optimization approaches
    try:
        # First try with SLSQP which handles constraints well
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1500, 'ftol': 1e-8, 'eps': 1e-8}
        )
        
        if result.success:
            # Convert back to circles
            optimized_circles = np.zeros((n, 3))
            for i in range(n):
                optimized_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            return optimized_circles
    except Exception as e:
        # Fall back to simpler approach if optimization fails
        pass
    
    # If optimization failed, return the original configuration with some refinement
    return circles

def local_refinement(circles):
    """Apply local refinement to improve the solution"""
    n = len(circles)
    
    # Create a copy to work with
    refined = circles.copy()
    
    # Multiple rounds of local improvement
    for round_num in range(3):
        improved = True
        iteration = 0
        while improved and iteration < 500:
            improved = False
            iteration += 1
            
            # Try to increase all radii where possible
            for i in range(n):
                current_x, current_y, current_r = refined[i]
                
                # Try to increase radius
                max_possible_radius = min(current_x, 1-current_x, current_y, 1-current_y)
                
                # Find closest circle
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        x, y, r = refined[j]
                        dist = math.sqrt((current_x - x)**2 + (current_y - y)**2)
                        min_dist = min(min_dist, dist)
                
                # If there's a close circle, our radius is limited by that
                if min_dist < float('inf'):
                    max_possible_radius = min(max_possible_radius, min_dist - 0.001)
                
                # Try to increase radius
                if max_possible_radius > current_r and max_possible_radius > 0.001:
                    new_r = min(current_r + 0.002, max_possible_radius)
                    
                    # Check if we can actually place it
                    valid = True
                    for j in range(n):
                        if i != j:
                            x, y, r = refined[j]
                            dist = math.sqrt((current_x - x)**2 + (current_y - y)**2)
                            if dist < new_r + r:
                                valid = False
                                break
                    
                    if valid:
                        refined[i] = [current_x, current_y, new_r]
                        improved = True
        
        # Try a more aggressive improvement for the final round
        if round_num == 2:
            # Do a more thorough search for improvements
            for i in range(n):
                current_x, current_y, current_r = refined[i]
                
                # Try to slightly adjust position to increase radius
                best_r = current_r
                best_x, best_y = current_x, current_y
                
                # Try several small movements
                for _ in range(50):
                    dx = np.random.uniform(-0.005, 0.005)
                    dy = np.random.uniform(-0.005, 0.005)
                    
                    new_x = current_x + dx
                    new_y = current_y + dy
                    
                    # Ensure within bounds
                    new_x = np.clip(new_x, current_r, 1 - current_r)
                    new_y = np.clip(new_y, current_r, 1 - current_r)
                    
                    # Calculate new radius we could have at this new position
                    max_new_r = min(new_x, 1-new_x, new_y, 1-new_y)
                    
                    # Check distances to others
                    for j in range(n):
                        if i != j:
                            x, y, r = refined[j]
                            dist = math.sqrt((new_x - x)**2 + (new_y - y)**2)
                            max_new_r = min(max_new_r, dist - r)
                    
                    if max_new_r > best_r and max_new_r > 0.001:
                        best_r = max_new_r
                        best_x, best_y = new_x, new_y
                
                # Apply the improvement if found
                if best_r > current_r:
                    refined[i] = [best_x, best_y, best_r]
    
    return refined


# EVOLVE-BLOCK-END
