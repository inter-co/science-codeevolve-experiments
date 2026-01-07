# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, constraint propagation, and 
    enhanced optimization with spatial indexing for better performance.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Multi-start approach with diverse initializations
    best_sum = -np.inf
    best_circles = None
    
    # Try 15 different initialization strategies for better exploration
    for start_iter in range(15):
        # Strategy 1: Hexagonal pattern with randomness
        if start_iter % 5 == 0:
            circles = initialize_hexagonal_pattern(n)
        # Strategy 2: Grid pattern with randomness  
        elif start_iter % 5 == 1:
            circles = initialize_grid_pattern(n)
        # Strategy 3: Random pattern with boundary awareness
        elif start_iter % 5 == 2:
            circles = initialize_random_pattern(n)
        # Strategy 4: Improved grid pattern with better distribution
        elif start_iter % 5 == 3:
            circles = initialize_improved_grid_pattern(n)
        # Strategy 5: Pattern based on constraint propagation
        else:
            circles = initialize_constraint_propagation_pattern(n)
        
        # Phase 2: Constraint propagation to tighten initial solution
        circles = propagate_bounds(circles)
        
        # Phase 3: Multi-start optimization with perturbations
        optimized = optimize_with_constraints(circles)
        
        # Phase 4: Constraint-aware refinement to improve quality
        refined = constraint_aware_refinement(optimized)
        
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
            x = x_offset + j * spacing_x + spacing_x * 0.5 + random.uniform(-0.015, 0.015)
            y = i * spacing_y + spacing_y * 0.5 + random.uniform(-0.015, 0.015)
            
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
            x = (j + 0.5 + random.uniform(-0.12, 0.12)) / cols
            y = (i + 0.5 + random.uniform(-0.12, 0.12)) / rows
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
        radii.append(max(0.001, initial_radius * 0.85))  # Slightly smaller initial radius
    
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
        r = max_radius * 0.65
        circles.append([x, y, r])
    return np.array(circles)

def initialize_improved_grid_pattern(n: int) -> np.ndarray:
    """Initialize with an improved grid pattern that's more evenly distributed"""
    # Use a more sophisticated grid layout
    sqrt_n = math.ceil(math.sqrt(n))
    rows = sqrt_n
    cols = sqrt_n
    
    # Ensure we have enough positions
    if rows * cols < n:
        cols += 1
    
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Distribute more evenly across the square
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            # Add slight jitter for better distribution
            x += random.uniform(-0.01, 0.01)
            y += random.uniform(-0.01, 0.01)
            positions.append([x, y])
    
    # Trim to exact count
    positions = positions[:n]
    
    # Ensure positions are within bounds
    positions = [[max(0.01, min(0.99, x)), max(0.01, min(0.99, y))] for x, y in positions]
    
    # Compute initial radii based on local density
    radii = []
    for i, (x, y) in enumerate(positions):
        min_dist_to_bound = min(x, 1-x, y, 1-y)
        
        # Find minimum distance to other points
        min_dist_to_other = float('inf')
        for k, (px, py) in enumerate(positions):
            if k != i:
                dist = math.sqrt((x - px)**2 + (y - py)**2)
                min_dist_to_other = min(min_dist_to_other, dist)
        
        # Set radius based on local density and boundary constraints
        initial_radius = min(min_dist_to_bound, min_dist_to_other/2) if min_dist_to_other != float('inf') else min_dist_to_bound
        radii.append(max(0.001, initial_radius * 0.9))
    
    return np.column_stack([positions, radii])

def initialize_constraint_propagation_pattern(n: int) -> np.ndarray:
    """Initialize using constraint propagation to get better starting point"""
    # Start with a basic hexagonal pattern
    circles = initialize_hexagonal_pattern(n)
    
    # Apply constraint propagation to tighten bounds
    circles = propagate_bounds(circles)
    
    # Slightly perturb to avoid symmetry
    for i in range(len(circles)):
        circles[i, 0] += random.uniform(-0.005, 0.005)
        circles[i, 1] += random.uniform(-0.005, 0.005)
        circles[i, 2] *= (1 + random.uniform(-0.02, 0.02))
        
        # Keep within bounds
        circles[i, 0] = np.clip(circles[i, 0], 1e-6, 1-1e-6)
        circles[i, 1] = np.clip(circles[i, 1], 1e-6, 1-1e-6)
        circles[i, 2] = np.clip(circles[i, 2], 1e-6, 0.5)
    
    return circles

def propagate_bounds(circles):
    """Apply constraint propagation to tighten bounds on radii"""
    new_circles = circles.copy()
    
    # Iteratively improve bounds until convergence or max iterations
    max_iterations = 15
    for iteration in range(max_iterations):
        changed = False
        for i in range(len(new_circles)):
            x, y, r = new_circles[i]
            
            # Calculate maximum possible radius due to boundaries
            max_radius_bound = min(x, 1-x, y, 1-y)
            
            # Calculate maximum possible radius due to other circles using spatial indexing
            max_radius_circles = max_radius_bound
            
            # Use spatial indexing for efficient neighbor search
            positions = new_circles[:, :2]
            radii = new_circles[:, 2]
            
            if len(positions) > 1:
                # Create KDTree for efficient neighbor queries
                tree = cKDTree(positions)
                # Query nearby points within reasonable distance
                nearby_indices = tree.query_ball_point([x, y], 2 * max_radius_bound)
                
                for j in nearby_indices:
                    if i != j:
                        dist = np.sqrt((x - positions[j, 0])**2 + (y - positions[j, 1])**2)
                        max_radius_circles = min(max_radius_circles, dist - radii[j])
            
            # Update radius to be the minimum of all constraints
            new_radius = max(0.001, min(max_radius_bound, max_radius_circles))
            if abs(new_radius - r) > 1e-8:
                new_circles[i, 2] = new_radius
                changed = True
        
        if not changed:
            break
            
    return new_circles

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
                # Small safety margin to ensure constraints are met
                constraints.append(dist - radius_sum - 1e-8)
        
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
        # Increased maxiter and tightened tolerances for better convergence
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 2500, 'ftol': 1e-12, 'gtol': 1e-12}
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

def constraint_aware_refinement(circles: np.ndarray) -> np.ndarray:
    """Apply constraint-aware refinement to improve solution quality"""
    n = len(circles)
    refined_circles = circles.copy()
    
    # More sophisticated refinement with better constraint handling
    for iteration in range(600):  # More iterations for better refinement
        improved = False
        # Shuffle circle indices for better exploration
        indices = list(range(n))
        np.random.shuffle(indices)
        
        for i in indices:
            # Get current circle info
            x, y, r = refined_circles[i]
            
            # Calculate maximum possible radius at current position
            max_radius = min(x, 1-x, y, 1-y)
            
            # Check if we can increase radius without violating constraints
            can_increase = True
            
            # Check overlap with all other circles more carefully using spatial indexing
            positions = refined_circles[:, :2]
            radii = refined_circles[:, 2]
            
            # Use spatial indexing to find nearby circles efficiently
            if len(positions) > 1:
                tree = cKDTree(positions)
                # Query nearby points within reasonable distance
                nearby_indices = tree.query_ball_point([x, y], 2 * max_radius)
                
                for j in nearby_indices:
                    if i != j:
                        dist = np.sqrt((x - positions[j, 0])**2 + (y - positions[j, 1])**2)
                        # If too close to another circle, we can't increase radius
                        if dist < radii[j] + max_radius:
                            can_increase = False
                            break
            
            if can_increase:
                # Try to increase radius more aggressively but safely
                new_r = min(max_radius, r * 1.02)  # Slightly larger increase factor
                
                # Verify that this change doesn't cause overlaps
                valid = True
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = refined_circles[j]
                        dist = math.sqrt((x2-x)**2 + (y2-y)**2)
                        if dist < new_r + r2:
                            valid = False
                            break
                
                if valid and new_r > r:
                    refined_circles[i, 2] = new_r
                    improved = True
        
        # Stop if no improvement made in several consecutive iterations
        if not improved:
            break
    
    return refined_circles


# EVOLVE-BLOCK-END
