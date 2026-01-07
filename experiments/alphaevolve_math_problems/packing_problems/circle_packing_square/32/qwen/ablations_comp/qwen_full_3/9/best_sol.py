# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
from typing import Tuple

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def _validate_circles(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and non-overlapping - highly optimized version"""
    n = len(circles)
    
    # Early exit if too few circles
    if n <= 1:
        return True
    
    # Vectorized containment check
    x_coords, y_coords, radii = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Check containment constraints efficiently
    containment_check = (
        (radii <= x_coords) & 
        (radii <= y_coords) & 
        (x_coords <= 1 - radii) & 
        (y_coords <= 1 - radii)
    )
    if not np.all(containment_check):
        return False
    
    # Check overlap constraints using vectorized operations
    positions = circles[:, :2]
    radii_vec = circles[:, 2]
    
    # Compute pairwise distances efficiently
    distances = cdist(positions, positions)
    
    # Create mask for upper triangle to avoid double counting
    upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Check overlap conditions
    radii_sum = np.add.outer(radii_vec, radii_vec)
    overlap_mask = distances < radii_sum
    overlap_mask[np.logical_not(upper_triangle)] = False
    
    # Return True if no overlaps found
    return not np.any(overlap_mask)

def _compute_radius_sum(circles: np.ndarray) -> float:
    """Compute sum of all radii"""
    return np.sum(circles[:, 2])

def _initialize_hexagonal_placement(n: int) -> np.ndarray:
    """Initialize circles using hexagonal packing pattern - improved version"""
    # Create a hexagonal grid pattern that fits within the unit square
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    if rows * cols < n:
        rows += 1
        cols = math.ceil(n / rows)
    
    # Calculate spacing with padding to avoid boundary issues
    padding = 0.05
    width = 1 - 2*padding
    height = 1 - 2*padding
    
    spacing_x = width / cols
    spacing_y = height / rows
    
    # Adjust for hexagonal packing (alternate rows offset)
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            x = padding + (j + 0.5) * spacing_x
            y = padding + (i + 0.5) * spacing_y
            
            # Offset odd rows for hexagonal packing
            if i % 2 == 1:
                x += spacing_x * 0.5
                
            # Ensure within bounds
            x = max(padding, min(1-padding, x))
            y = max(padding, min(1-padding, y))
            
            # Set initial radius - start with small values to allow for optimization
            r = min(spacing_x, spacing_y) * 0.3
            
            circles.append([x, y, r])
    
    # Fill remaining circles if needed
    while len(circles) < n:
        circles.append([0.5, 0.5, 0.05])
        
    return np.array(circles[:n])

def _initialize_focused_placement(n: int) -> np.ndarray:
    """Initialize circles with focused placement near center - improved version"""
    circles = []
    
    # Place some circles near the center with larger radii
    center_count = min(8, n)
    for i in range(center_count):
        angle = 2 * np.pi * i / center_count
        radius = 0.15 * random.uniform(0.7, 1.0)
        x = 0.5 + radius * np.cos(angle) * random.uniform(0.5, 1.0)
        y = 0.5 + radius * np.sin(angle) * random.uniform(0.5, 1.0)
        r = radius * random.uniform(0.8, 1.0)
        
        # Ensure within bounds
        r = min(r, x, 1-x, y, 1-y)
        circles.append([x, y, r])
    
    # Fill remaining with hexagonal approach
    remaining = n - len(circles)
    if remaining > 0:
        hex_circles = _initialize_hexagonal_placement(remaining)
        circles.extend(hex_circles.tolist())
    
    return np.array(circles)

def _initialize_voronoi_placement(n: int) -> np.ndarray:
    """Initialize circles using a Voronoi-inspired approach similar to inspiration program"""
    # Create points distributed in a grid pattern with some randomness
    grid_size = int(np.ceil(np.sqrt(n)))
    points = []
    
    # Generate grid points with slight perturbation
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n:
                x = (i + 0.5 + (random.random() - 0.5) * 0.3) / grid_size
                y = (j + 0.5 + (random.random() - 0.5) * 0.3) / grid_size
                points.append([x, y])
    
    points = np.array(points[:n])
    
    # Initialize radii based on distance to neighbors
    radii = np.full(n, 0.05)  # Default small radius
    
    # For each point, set radius based on proximity to neighbors
    for i in range(n):
        distances = cdist([points[i]], points)[0]
        distances = distances[distances > 0]  # Exclude self-distance
        if len(distances) > 0:
            min_dist = np.min(distances)
            # Set radius to a fraction of the minimum distance to neighbors
            radii[i] = min(min_dist / 3.0, 0.2)
    
    # Create circles array
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i][0] = points[i][0]     # x coordinate
        circles[i][1] = points[i][1]     # y coordinate
        circles[i][2] = radii[i]         # radius
    
    return circles

def _constraint_violation_penalty(circles: np.ndarray) -> float:
    """Calculate penalty for constraint violations"""
    penalty = 0.0
    
    # Check containment violations
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x < r or x > 1-r or y < r or y > 1-r:
            penalty += 1000.0
    
    # Check overlap violations
    positions = circles[:, :2]
    radii = circles[:, 2]
    distances = cdist(positions, positions)
    
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            dist = distances[i, j]
            if dist < (radii[i] + radii[j]):
                penalty += (radii[i] + radii[j] - dist)**2
    
    return penalty

def _optimize_circles(circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using constrained optimization"""
    n = len(circles)
    
    # Flatten parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = circles.flatten()
    
    # Define bounds for parameters (x, y, r)
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x coordinate
        bounds.append((0.001, 0.999))  # y coordinate
        bounds.append((0.001, 0.499))  # radius (max radius is 0.5 when placed at corner)
    
    # Define constraint functions for non-overlap and containment
    def constraint_func(params):
        circles_test = params.reshape(-1, 3)
        constraints = []
        
        # Non-overlap constraints: distance between centers >= sum of radii
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_test[i]
                x2, y2, r2 = circles_test[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                # Constraint: dist >= r1 + r2, so we want: dist - (r1 + r2) >= 0
                constraints.append(dist - (r1 + r2))
        
        # Boundary constraints (containment): x-r >= 0, y-r >= 0, 1-x-r >= 0, 1-y-r >= 0
        for i in range(n):
            x, y, r = circles_test[i]
            constraints.append(x - r)  # x >= r
            constraints.append(y - r)  # y >= r
            constraints.append(1 - x - r)  # 1-x >= r
            constraints.append(1 - y - r)  # 1-y >= r
        
        return np.array(constraints)
    
    # Objective function: minimize negative sum of radii (to maximize sum of radii)
    def objective(params):
        circles_test = params.reshape(-1, 3)
        return -np.sum(circles_test[:, 2])
    
    # Try multiple optimization approaches
    try:
        # First try SLSQP
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        # Fall back to a simpler approach
        pass
    
    # If optimization fails, return original
    return circles

def _physics_based_refinement(circles: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Apply simple physics-based refinement to improve packing"""
    # This is a simplified version inspired by the second inspiration program
    current = circles.copy()
    
    for iteration in range(max_iterations):
        # Simple force-directed approach for basic improvement
        improved = False
        
        # Try to increase radii where possible
        for i in range(len(current)):
            orig_r = current[i, 2]
            # Try to increase radius by small amount
            test_r = min(orig_r + 0.005, 0.499)
            test_r = max(test_r, 0.001)
            
            # Check if we can increase this radius without violating constraints
            valid = True
            for j in range(len(current)):
                if i != j:
                    dx = current[i, 0] - current[j, 0]
                    dy = current[i, 1] - current[j, 1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (test_r + current[j, 2])**2
                    
                    if dist_sq < min_dist_sq:
                        valid = False
                        break
            
            if valid:
                test_circles = current.copy()
                test_circles[i, 2] = test_r
                if _validate_circles(test_circles):
                    current = test_circles.copy()
                    improved = True
        
        # If no improvement, try small position adjustments
        if not improved:
            for i in range(len(current)):
                orig_x, orig_y, orig_r = current[i, 0], current[i, 1], current[i, 2]
                
                # Try small adjustments
                adjustments = [(0.002, 0), (-0.002, 0), (0, 0.002), (0, -0.002)]
                for dx, dy in adjustments:
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    
                    # Keep within bounds
                    new_x = np.clip(new_x, orig_r, 1 - orig_r)
                    new_y = np.clip(new_y, orig_r, 1 - orig_r)
                    
                    # Check if this change is valid
                    valid = True
                    for j in range(len(current)):
                        if i != j:
                            dx = new_x - current[j, 0]
                            dy = new_y - current[j, 1]
                            dist_sq = dx*dx + dy*dy
                            min_dist_sq = (orig_r + current[j, 2])**2
                            
                            if dist_sq < min_dist_sq:
                                valid = False
                                break
                    
                    if valid:
                        test_circles = current.copy()
                        test_circles[i, 0] = new_x
                        test_circles[i, 1] = new_y
                        if _validate_circles(test_circles):
                            current = test_circles.copy()
                            improved = True
                            break
        
        if not improved:
            break
    
    return current

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 32
    best_circles = None
    best_sum = 0
    
    # Multi-start approach with different initialization strategies
    initializers = [
        _initialize_hexagonal_placement,
        _initialize_focused_placement,
        _initialize_voronoi_placement
    ]
    
    # Run more iterations to find better solutions
    for start_iter in range(15):  # Increase iterations for better exploration
        # Choose random initializer
        initializer = random.choice(initializers)
        circles = initializer(n)
        
        # Apply physics-based refinement
        circles = _physics_based_refinement(circles, 20)
        
        # Optimize with constrained optimization
        circles = _optimize_circles(circles)
        
        # Validate and check quality
        if _validate_circles(circles):
            current_sum = _compute_radius_sum(circles)
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
    
    # Final refinement passes
    if best_circles is not None:
        # Try a few more optimization passes
        for _ in range(3):
            refined = _optimize_circles(best_circles)
            if _validate_circles(refined):
                refined_sum = _compute_radius_sum(refined)
                if refined_sum > best_sum:
                    best_sum = refined_sum
                    best_circles = refined.copy()
    
    # Ensure final validation
    if best_circles is None:
        # Fallback to Voronoi initialization
        best_circles = _initialize_voronoi_placement(n)
        best_circles = _optimize_circles(best_circles)
    
    return best_circles


# EVOLVE-BLOCK-END
