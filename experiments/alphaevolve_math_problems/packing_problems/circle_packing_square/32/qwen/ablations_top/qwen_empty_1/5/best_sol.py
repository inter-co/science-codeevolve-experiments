# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-start optimization approach with multiple initialization strategies and 
    physics-based refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Multiple initialization strategies to find better starting points
    best_result = None
    best_sum = 0
    
    # Strategy 1: Hexagonal packing with physics refinement (like INSPIRATION 2)
    hex_result = initialize_improved_hexagonal_packing(n)
    hex_result = refine_with_physics_simulation(hex_result, max_iterations=500)
    hex_result = optimize_circles(hex_result)
    hex_sum = np.sum(hex_result[:, 2])
    
    if hex_sum > best_sum:
        best_sum = hex_sum
        best_result = hex_result
    
    # Strategy 2: Grid-based initialization with physics refinement
    grid_result = initialize_grid_packing(n)
    grid_result = refine_with_physics_simulation(grid_result, max_iterations=300)
    grid_result = optimize_circles(grid_result)
    grid_sum = np.sum(grid_result[:, 2])
    
    if grid_sum > best_sum:
        best_sum = grid_sum
        best_result = grid_result
    
    # Strategy 3: Random initialization with physics refinement
    random_result = initialize_random_packing(n)
    random_result = refine_with_physics_simulation(random_result, max_iterations=400)
    random_result = optimize_circles(random_result)
    random_sum = np.sum(random_result[:, 2])
    
    if random_sum > best_sum:
        best_sum = random_sum
        best_result = random_result
    
    # Strategy 4: Improved hexagonal with more aggressive refinement
    if best_result is not None:
        final_refinement = refine_with_physics_simulation(best_result, max_iterations=300, 
                                                           repulsion_strength=100.0, 
                                                           boundary_strength=1000.0)
        final_refinement = optimize_circles(final_refinement)
        final_sum = np.sum(final_refinement[:, 2])
        if final_sum > best_sum:
            best_result = final_refinement
    
    # If no good result was found, return a fallback
    if best_result is None:
        # Fallback to a known good configuration
        best_result = np.zeros((n, 3))
        # Place in a grid-like pattern with decreasing radii
        grid_size = int(np.ceil(np.sqrt(n)))
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx < n:
                    x = (i + 0.5) / grid_size
                    y = (j + 0.5) / grid_size
                    # Adjust for better packing
                    x = max(0.05, min(0.95, x))
                    y = max(0.05, min(0.95, y))
                    # Use a decreasing radius pattern
                    r = 0.15 * (1.0 - 0.05 * idx)  # Decreasing radii
                    r = max(0.01, min(0.15, r))
                    best_result[idx] = [x, y, r]
                    idx += 1
    
    return best_result

def initialize_improved_hexagonal_packing(n):
    """Initialize circle positions using an improved hexagonal packing pattern."""
    # Use a more systematic approach for hexagonal packing
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Ensure we have enough space for all circles
    while rows * cols < n:
        rows += 1
        cols = int(np.ceil(n / rows))
    
    # Create hexagonal grid with better spacing
    circles = np.zeros((n, 3))
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Use sqrt(3)/2 for better hexagonal packing
    hex_height = spacing_y * np.sqrt(3) / 2
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Hexagonal offset for alternating rows
            x = (j + (i % 2) * 0.5) * spacing_x
            y = i * hex_height
            
            # Adjust to fit within unit square with margin
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius based on available space
            max_radius = min(x, 1-x, y, 1-y)
            # Use a larger initial radius to start with better packing
            radius = max_radius * 0.6
            circles[idx] = [x, y, radius]
            idx += 1
    
    return circles

def initialize_grid_packing(n):
    """Initialize using a regular grid pattern with slight perturbations."""
    # Calculate grid dimensions
    rows = int(math.sqrt(n))
    cols = int(math.ceil(n / rows))
    
    # Ensure we have enough space for all circles
    while rows * cols < n:
        rows += 1
        cols = int(math.ceil(n / rows))
    
    # Create regular grid
    circles = np.zeros((n, 3))
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Add some randomness to positions
    np.random.seed(42)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Add small random perturbation to grid positions
            x = max(0.05, min(0.95, (j + 0.5 + np.random.uniform(-0.15, 0.15)) * spacing_x))
            y = max(0.05, min(0.95, (i + 0.5 + np.random.uniform(-0.15, 0.15)) * spacing_y))
            
            # Initial radius based on available space
            radius = min(x, 1-x, y, 1-y) * 0.4
            circles[idx] = [x, y, radius]
            idx += 1
    
    return circles

def initialize_random_packing(n):
    """Initialize with random positions and small radii."""
    np.random.seed(42)
    circles = np.zeros((n, 3))
    
    # Initialize with random positions and small radii
    for i in range(n):
        circles[i] = [
            np.random.uniform(0.05, 0.95),  # x coordinate
            np.random.uniform(0.05, 0.95),  # y coordinate
            np.random.uniform(0.01, 0.05)   # initial radius
        ]
    
    return circles

def refine_with_physics_simulation(circles, max_iterations=500, repulsion_strength=100.0, boundary_strength=1000.0):
    """Apply physics-based refinement to improve initial configuration."""
    np.random.seed(42)
    
    # Physics simulation parameters
    dt = 0.001
    
    n = len(circles)
    
    # Track improvement for early stopping
    previous_sum = np.sum(circles[:, 2])
    improvement_count = 0
    
    for iteration in range(max_iterations):
        # Calculate pairwise distances efficiently
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Initialize forces
        forces = np.zeros_like(positions)
        
        # Compute repulsion forces between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                # Check if circles overlap
                if dist < (circles[i, 2] + circles[j, 2]):
                    # Repulsion force (stronger when more overlapping)
                    force_magnitude = repulsion_strength * (circles[i, 2] + circles[j, 2] - dist) / (dist + 1e-8)
                    forces[i, 0] += force_magnitude * dx / (dist + 1e-8)
                    forces[i, 1] += force_magnitude * dy / (dist + 1e-8)
                    forces[j, 0] -= force_magnitude * dx / (dist + 1e-8)
                    forces[j, 1] -= force_magnitude * dy / (dist + 1e-8)
        
        # Apply boundary constraints (repulsive forces from edges)
        for i in range(n):
            # Left boundary
            if positions[i, 0] < circles[i, 2]:
                forces[i, 0] += boundary_strength * (circles[i, 2] - positions[i, 0])
            # Right boundary
            if positions[i, 0] > 1 - circles[i, 2]:
                forces[i, 0] -= boundary_strength * (positions[i, 0] - (1 - circles[i, 2]))
            # Bottom boundary
            if positions[i, 1] < circles[i, 2]:
                forces[i, 1] += boundary_strength * (circles[i, 2] - positions[i, 1])
            # Top boundary
            if positions[i, 1] > 1 - circles[i, 2]:
                forces[i, 1] -= boundary_strength * (positions[i, 1] - (1 - circles[i, 2]))
        
        # Update positions and radii
        for i in range(n):
            # Update position based on forces
            circles[i, 0] += dt * forces[i, 0]
            circles[i, 1] += dt * forces[i, 1]
            
            # Ensure circles stay within bounds
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
        # Adaptive radius adjustment with more aggressive increases
        for i in range(n):
            # Check if we can safely increase radius
            safe_to_increase = True
            for j in range(n):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < circles[i, 2] + circles[j, 2] + 0.005:  # Tighter buffer
                        safe_to_increase = False
                        break
            
            # More aggressive radius increase when safe
            if safe_to_increase and circles[i, 2] < 0.15:  # Cap maximum radius
                circles[i, 2] += 0.001  # Increased increment
        
        # Early stopping based on improvement
        current_sum = np.sum(circles[:, 2])
        if abs(current_sum - previous_sum) < 1e-5:
            improvement_count += 1
        else:
            improvement_count = 0
            previous_sum = current_sum
            
        if improvement_count > 20 and iteration > 100:
            break
    
    return circles

def calculate_constraints(circles):
    """Calculate all constraint violations."""
    n = len(circles)
    constraints = []
    
    # Boundary constraints: radius must be such that circle fits in unit square
    for i in range(n):
        x, y, r = circles[i]
        boundary_violation = max(0, r - x, r - (1-x), r - y, r - (1-y))
        if boundary_violation > 0:
            constraints.append(boundary_violation)
    
    # Circle-to-circle constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            overlap = distance - (r1 + r2)
            if overlap < 0:
                constraints.append(-overlap)
    
    return constraints

def objective_function(circles_flat):
    """Objective function to maximize sum of radii."""
    # Reshape flat array back to circles
    circles = circles_flat.reshape(-1, 3)
    return -np.sum(circles[:, 2])  # Negative because we minimize

def constraint_function(circles_flat):
    """Constraint function returning negative overlap values."""
    circles = circles_flat.reshape(-1, 3)
    n = len(circles)
    
    # Check boundary constraints
    bounds = []
    for i in range(n):
        x, y, r = circles[i]
        bounds.extend([x - r, (1-x) - r, y - r, (1-y) - r])
    
    # Check circle-to-circle constraints
    overlaps = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            overlap = distance - (r1 + r2)
            overlaps.append(overlap)
    
    return np.concatenate([bounds, overlaps])

def optimize_circles(initial_circles):
    """Optimize circle positions and radii using scipy optimization."""
    # Flatten initial circles for optimization
    initial_flat = initial_circles.flatten()
    
    # Set up constraints
    # Boundary constraints: each circle must fit in the unit square
    # Circle-to-circle constraints: circles cannot overlap
    
    # Create bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(len(initial_circles)):
        # x coordinate bounds
        bounds.append((0.001, 0.999))  # Small buffer to prevent exact boundary
        # y coordinate bounds  
        bounds.append((0.001, 0.999))
        # radius bounds (must be positive and small enough to fit)
        bounds.append((0.001, 0.499))
    
    # Define constraints
    # We want all constraints to be >= 0 (so negative values indicate violations)
    constraints = {'type': 'ineq', 'fun': constraint_function}
    
    try:
        # Perform optimization with better parameters
        result = minimize(
            objective_function,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1500, 'ftol': 1e-7}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure all radii are valid
            optimized_circles[:, 2] = np.maximum(0.001, optimized_circles[:, 2])
            return optimized_circles
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        pass
    
    return initial_circles


# EVOLVE-BLOCK-END
