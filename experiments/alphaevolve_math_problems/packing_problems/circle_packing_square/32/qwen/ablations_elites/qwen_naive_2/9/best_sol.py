# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, multi-start optimization, 
    and physics-inspired refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Multi-start approach with diverse initializations (from INSPIRATION 2)
    best_sum = -np.inf
    best_circles = None
    
    # Try multiple initialization strategies
    for start_iter in range(8):  # Increase from 5 to 8 for better exploration
        # Strategy 1: Hexagonal pattern with randomness (like INSPIRATION 1)
        if start_iter % 3 == 0:
            circles = initialize_hexagonal_pattern(n)
        # Strategy 2: Grid pattern with randomness (like INSPIRATION 2)  
        elif start_iter % 3 == 1:
            circles = initialize_grid_pattern(n)
        # Strategy 3: Random placement with boundary awareness
        else:
            circles = initialize_random_pattern(n)
        
        # Phase 2: Multi-start optimization with perturbations (enhanced from INSPIRATION 1)
        optimized = optimize_with_constraints(circles)
        
        # Phase 3: Physics-inspired refinement (like INSPIRATION 2)
        refined = refine_with_forces(optimized)
        
        # Evaluate solution
        current_sum = np.sum(refined[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = refined.copy()
    
    # Final validation and cleanup
    if best_circles is not None:
        # Ensure all circles are valid
        for i in range(n):
            x, y, r = best_circles[i]
            # Ensure valid bounds
            best_circles[i] = [
                max(r, min(1-r, x)), 
                max(r, min(1-r, y)), 
                max(1e-6, min(0.5, r))
            ]
        return best_circles
    
    # Fallback to simple initialization if everything fails
    return initialize_hexagonal_pattern(n)

def initialize_hexagonal_pattern(n: int) -> np.ndarray:
    """Initialize circles in a hexagonal pattern that's more likely to yield good results"""
    # Create a hexagonal grid that better fits the unit square
    rows = 6
    cols = 6
    
    circles = []
    
    # Hexagonal packing parameters
    sqrt3 = math.sqrt(3)
    spacing_x = 1.0 / cols
    spacing_y = spacing_x * sqrt3 / 2.0  # Vertical spacing for hexagonal packing
    
    # Create hexagonal arrangement with some randomness for diversity
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            # Offset every other row
            x_offset = (i % 2) * spacing_x * 0.5
            x = x_offset + j * spacing_x + spacing_x * 0.5 + random.uniform(-0.02, 0.02)
            y = i * spacing_y + spacing_y * 0.5 + random.uniform(-0.02, 0.02)
            
            # Ensure within bounds
            if x >= 0 and x <= 1 and y >= 0 and y <= 1:
                # Calculate maximum possible radius at this position
                max_radius = min(x, 1-x, y, 1-y)
                # Use a reasonable fraction of maximum radius to allow for optimization
                r = max_radius * 0.85  # Slightly higher initial radius
                circles.append([x, y, r])
    
    # Fill remaining positions with random placements near boundaries for better coverage
    while len(circles) < n:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        max_radius = min(x, 1-x, y, 1-y)
        r = max_radius * 0.8  # Higher initial radius
        circles.append([x, y, r])
    
    return np.array(circles[:n])

def initialize_grid_pattern(n: int) -> np.ndarray:
    """Initialize circles in a grid pattern with randomness"""
    rows = math.ceil(math.sqrt(n))
    cols = math.ceil(n / rows)
    
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Add randomness to avoid symmetric solutions
            x = (j + 0.5 + random.uniform(-0.15, 0.15)) / cols
            y = (i + 0.5 + random.uniform(-0.15, 0.15)) / rows
            positions.append([x, y])
    
    # Adjust to ensure we have exactly n points
    positions = positions[:n]
    
    # Ensure positions are within bounds
    positions = [[max(0.01, min(0.99, x)), max(0.01, min(0.99, y))] for x, y in positions]
    
    # Initial radii based on distance to nearest neighbors and boundaries
    radii = []
    for i, (x, y) in enumerate(positions):
        # Calculate minimum distance to boundaries
        min_dist_to_bound = min(x, 1-x, y, 1-y)
        
        # Calculate minimum distance to other points
        min_dist_to_other = float('inf')
        for k, (px, py) in enumerate(positions):
            if k != i:
                dist = math.sqrt((x - px)**2 + (y - py)**2)
                min_dist_to_other = min(min_dist_to_other, dist)
        
        # Set initial radius to be half the minimum of boundary and neighbor distances
        initial_radius = min(min_dist_to_bound, min_dist_to_other/2) if min_dist_to_other != float('inf') else min_dist_to_bound
        radii.append(max(0.001, initial_radius * 0.9))  # Slightly smaller initial radius
    
    return np.column_stack([positions, radii])

def initialize_random_pattern(n: int) -> np.ndarray:
    """Initialize circles with random positions and radii"""
    circles = []
    for _ in range(n):
        # Place near center but with some variation
        x = 0.5 + random.uniform(-0.3, 0.3)
        y = 0.5 + random.uniform(-0.3, 0.3)
        # Ensure within bounds
        x = max(0.01, min(0.99, x))
        y = max(0.01, min(0.99, y))
        # Calculate max radius
        max_radius = min(x, 1-x, y, 1-y)
        # Use a smaller fraction for more room to grow
        r = max_radius * 0.6
        circles.append([x, y, r])
    return np.array(circles)

def optimize_with_constraints(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize using scipy's constrained optimization with proper constraint handling"""
    n = len(initial_circles)
    
    # Flatten initial parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = initial_circles.flatten()
    
    # Define bounds for each parameter (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # x bounds: [r, 1-r] to ensure circle fits within square
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])
    
    def objective(params):
        # Extract circles from flattened parameters
        circles = params.reshape(-1, 3)
        # We want to maximize sum of radii, so minimize negative sum
        return -np.sum(circles[:, 2])
    
    def constraint_func(params):
        """Non-overlap constraints: distance >= sum of radii"""
        circles = params.reshape(-1, 3)
        # Calculate pairwise distances between all circle centers
        distances = cdist(circles[:, :2], circles[:, :2])
        
        # Create constraint vector: for each pair of circles, 
        # constraint is (distance - (r1 + r2)) >= 0
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                radius_sum = circles[i, 2] + circles[j, 2]
                constraints.append(dist - radius_sum)
        
        return np.array(constraints)
    
    def containment_constraint(params):
        """Boundary constraints: r <= x <= 1-r and r <= y <= 1-r"""
        circles = params.reshape(-1, 3)
        constraints = []
        for i in range(n):
            x, y, r = circles[i]
            # x >= r and x <= 1-r
            constraints.append(x - r)
            constraints.append(1 - x - r)
            # y >= r and y <= 1-r  
            constraints.append(y - r)
            constraints.append(1 - y - r)
        return np.array(constraints)
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: constraint_func(x)},
        {'type': 'ineq', 'fun': lambda x: containment_constraint(x)}
    ]
    
    try:
        # Use SLSQP which handles both constraints and bounds well
        # Increase maxiter and tighten tolerances for better convergence (from INSPIRATION 1)
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-9}  # More iterations and tighter tolerance
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Final validation and clamping
            for i in range(n):
                x, y, r = optimized_circles[i]
                # Ensure valid ranges
                optimized_circles[i] = [
                    max(r, min(1-r, x)), 
                    max(r, min(1-r, y)), 
                    max(1e-6, min(0.5, r))
                ]
            return optimized_circles
    except Exception as e:
        pass
    
    # Return initial configuration if optimization fails
    return initial_circles

def refine_with_forces(circles: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Apply physics-inspired refinement to improve solution quality (from INSPIRATION 2)"""
    n = len(circles)
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()
    
    # Simple force-based refinement
    for _ in range(max_iter):
        forces = np.zeros_like(pos)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist > 0 and dist < rad[i] + rad[j]:
                        # Repulsion force - stronger when circles are very close
                        force_magnitude = 0.01 * (rad[i] + rad[j] - dist) / (dist + 1e-10)
                        forces[i, 0] += force_magnitude * dx
                        forces[i, 1] += force_magnitude * dy
        
        # Apply forces with boundary constraints
        pos += forces * 0.01
        
        # Boundary constraints
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], rad[i], 1-rad[i])
            pos[i, 1] = np.clip(pos[i, 1], rad[i], 1-rad[i])
    
    return np.column_stack([pos, rad])


# EVOLVE-BLOCK-END
