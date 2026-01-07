# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
from scipy.spatial.distance import cdist
import random

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with robust optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    # Set random seed for reproducibility (as in inspiration programs)
    random.seed(42)
    np.random.seed(42)
    
    # Strategy: Multi-start local search with geometric initialization
    best_sum_radii = 0
    best_circles = None
    
    # Run multiple random starts with different strategies
    num_starts = 15  # More starts to increase chance of finding better solution
    
    for start_idx in range(num_starts):
        # Different initialization strategies
        if start_idx < 3:
            # Grid-based initialization
            circles = initialize_grid(26)
        elif start_idx < 6:
            # Random with careful spacing
            circles = initialize_spaced_random(26)
        elif start_idx < 9:
            # Semi-structured initialization
            circles = initialize_semi_structured(26)
        elif start_idx < 12:
            # Hexagonal arrangement
            circles = initialize_hexagonal(26)
        else:
            # Dense pack initialization
            circles = initialize_dense_pack(26)
        
        # Local optimization for this starting point
        optimized_circles = local_optimization(circles)
        
        # Calculate sum of radii
        sum_radii = np.sum(optimized_circles[:, 2])
        
        if sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_circles = optimized_circles.copy()
    
    # Final refinement of the best solution
    if best_circles is not None:
        final_circles = final_refinement(best_circles)
        return final_circles
    
    # Fallback to default if something went wrong
    return initialize_grid(26)

def initialize_grid(n):
    """Initialize circles in a grid pattern"""
    circles = np.zeros((n, 3))
    
    # Arrange in roughly a 5x5 grid with some randomness
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Space the circles evenly
    spacing_x = 0.8 / cols
    spacing_y = 0.8 / rows
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = 0.1 + j * spacing_x + random.uniform(-spacing_x*0.1, spacing_x*0.1)
            y = 0.1 + i * spacing_y + random.uniform(-spacing_y*0.1, spacing_y*0.1)
            # Initial radius based on available space
            r = min(0.05, 0.5 * min(spacing_x, spacing_y) - 0.01)
            circles[idx] = [x, y, max(0.001, r)]
            idx += 1
    
    return circles

def initialize_spaced_random(n):
    """Initialize circles with random placement but ensuring minimum spacing"""
    circles = np.zeros((n, 3))
    
    # Place first circle in center
    circles[0] = [0.5, 0.5, 0.1]
    
    # Place remaining circles ensuring minimum spacing
    for i in range(1, n):
        max_attempts = 1000
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts:
            # Random position
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            
            # Minimum radius based on proximity to existing circles
            min_dist = float('inf')
            for j in range(i):
                existing_x, existing_y, existing_r = circles[j]
                dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                min_dist = min(min_dist, dist)
            
            # Radius is limited by boundaries and minimum distance to others
            r = min(
                x, 1-x, y, 1-y,  # Boundary constraints
                min_dist / 2  # Spacing constraint
            )
            
            if r > 0.001:
                circles[i] = [x, y, max(0.001, r)]
                placed = True
            
            attempts += 1
    
    return circles

def initialize_semi_structured(n):
    """Initialize with some structured pattern and random variation"""
    circles = np.zeros((n, 3))
    
    # Create a basic pattern and add randomness
    # Start with a hexagonal-like arrangement
    centers = []
    rows = 5
    cols = 5
    
    # Generate a grid of points
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= n:
                break
            x = 0.1 + j * 0.18 + (i % 2) * 0.09
            y = 0.1 + i * 0.18
            centers.append([x, y])
    
    # Add some randomness and set initial radii
    for i in range(min(n, len(centers))):
        x, y = centers[i]
        # Add slight random perturbation
        x += random.uniform(-0.02, 0.02)
        y += random.uniform(-0.02, 0.02)
        # Keep within bounds
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        
        # Set radius based on surrounding space
        min_dist = float('inf')
        for j in range(i):
            existing_x, existing_y, existing_r = circles[j]
            dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
            min_dist = min(min_dist, dist)
        
        r = min(0.05, min_dist/2 - 0.005)
        circles[i] = [x, y, max(0.001, r)]
    
    # Fill remaining slots with random circles respecting constraints
    for i in range(len(centers), n):
        max_attempts = 100
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            
            min_dist = float('inf')
            for j in range(i):
                existing_x, existing_y, existing_r = circles[j]
                dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                min_dist = min(min_dist, dist)
            
            r = min(0.05, min_dist/2 - 0.005)
            if r > 0.001:
                circles[i] = [x, y, max(0.001, r)]
                placed = True
            attempts += 1
    
    return circles

def initialize_hexagonal(n):
    """Initialize with a hexagonal arrangement"""
    circles = np.zeros((n, 3))
    
    # Use 5 rows with 6 columns for 30 positions, then trim to 26
    rows = 5
    cols = 6
    
    positions = []
    
    # Generate hexagonal grid with proper spacing
    sqrt3 = math.sqrt(3)
    row_height = sqrt3 / 2.0
    
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Offset every other row for hexagonal packing
            x = j + (i % 2) * 0.5
            y = i * row_height
            
            # Scale to unit square with margins
            x_scaled = 0.1 + (x / (cols - 1)) * 0.8
            y_scaled = 0.1 + (y / (rows * row_height - row_height)) * 0.8
            
            # Ensure within bounds
            x_scaled = max(0.05, min(0.95, x_scaled))
            y_scaled = max(0.05, min(0.95, y_scaled))
            
            positions.append([x_scaled, y_scaled])
        if len(positions) >= n:
            break
    
    # Fill any remaining positions randomly
    while len(positions) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        positions.append([x, y])
    
    # Set initial radii based on available space
    total_area = 0.8 * 0.8  # Inner area excluding 0.1 margin
    avg_area_per_circle = total_area / n
    initial_radius = math.sqrt(avg_area_per_circle / math.pi) * 0.8
    
    for i in range(n):
        circles[i][0] = positions[i][0]  # x
        circles[i][1] = positions[i][1]  # y
        circles[i][2] = min(0.2, initial_radius)  # Initial radius
    
    return circles

def initialize_dense_pack(n):
    """Initialize with a more dense packing approach"""
    circles = np.zeros((n, 3))
    
    # Try to create a more densely packed arrangement
    # Start with a grid pattern and then adjust
    rows = 5
    cols = 5
    
    positions = []
    
    # Create a dense grid with some variation
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Add some randomness to create a more natural dense packing
            x = 0.1 + j * 0.18 + np.random.uniform(-0.02, 0.02)
            y = 0.1 + i * 0.18 + np.random.uniform(-0.02, 0.02)
            
            # Ensure within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            positions.append([x, y])
        if len(positions) >= n:
            break
    
    # Fill remaining positions
    while len(positions) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        positions.append([x, y])
    
    # Set initial radii - start with larger initial values to encourage growth
    for i in range(n):
        circles[i][0] = positions[i][0]  # x
        circles[i][1] = positions[i][1]  # y
        # Start with larger initial radius to promote better optimization
        circles[i][2] = 0.08  # Larger initial radius
    
    return circles

def local_optimization(initial_circles):
    """Perform local optimization using constrained optimization"""
    circles = initial_circles.copy()
    n = len(circles)
    
    # Define the objective function to maximize sum of radii
    def objective(params):
        # Reshape parameters back to circles
        circles_flat = params.reshape(-1, 3)
        return -np.sum(circles_flat[:, 2])  # Negative because we minimize
    
    # Define constraints
    def constraint_containment(params):
        circles_flat = params.reshape(-1, 3)
        constraints = []
        
        for i in range(n):
            x, y, r = circles_flat[i]
            # Left boundary: x >= r
            constraints.append(x - r)
            # Right boundary: x + r <= 1
            constraints.append(1 - x - r)
            # Bottom boundary: y >= r
            constraints.append(y - r)
            # Top boundary: y + r <= 1
            constraints.append(1 - y - r)
        
        return np.array(constraints)
    
    def constraint_overlap(params):
        circles_flat = params.reshape(-1, 3)
        constraints = []
        
        # Use distance matrix for efficiency
        positions = circles_flat[:, :2]
        radii = circles_flat[:, 2]
        
        # Compute distance matrix once
        dist_matrix = cdist(positions, positions)
        
        # Check all pairs
        for i in range(n):
            for j in range(i+1, n):
                distance = dist_matrix[i, j]
                min_distance = radii[i] + radii[j]
                # We want distance >= r1 + r2, so constraint: distance - r1 - r2 >= 0
                constraints.append(distance - min_distance - 1e-10)
        
        return np.array(constraints)
    
    # Set up bounds for variables
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds
        bounds.append((0.001, 0.999))
        # r bounds
        bounds.append((0.001, 0.499))
    
    # Flatten initial circles for optimization
    initial_params = circles.flatten()
    
    # Use SLSQP optimizer which handles constraints well
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': constraint_containment},
                {'type': 'ineq', 'fun': constraint_overlap}
            ],
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            return result.x.reshape(-1, 3)
    except:
        pass
    
    # Fallback to iterative improvement if optimization fails
    return iterative_improvement(circles)

def iterative_improvement(circles):
    """Iteratively improve solution by local search"""
    n = len(circles)
    max_iter = 300
    
    for iteration in range(max_iter):
        improved = False
        
        # Try to increase radii while maintaining feasibility
        for i in range(n):
            old_radius = circles[i, 2]
            max_radius = calculate_max_feasible_radius(circles, i)
            
            if max_radius > old_radius + 1e-6:
                circles[i, 2] = max_radius
                improved = True
        
        # Try position adjustments
        for i in range(n):
            old_pos = circles[i, :2].copy()
            old_radius = circles[i, 2]
            
            # Try small perturbations to improve packing
            best_pos = old_pos.copy()
            best_radius = old_radius
            
            for _ in range(10):
                dx = random.uniform(-0.01, 0.01)
                dy = random.uniform(-0.01, 0.01)
                new_pos = old_pos + np.array([dx, dy])
                
                # Keep within bounds
                new_pos[0] = np.clip(new_pos[0], 0.001, 0.999)
                new_pos[1] = np.clip(new_pos[1], 0.001, 0.999)
                
                # Test if this improves the configuration
                temp_circles = circles.copy()
                temp_circles[i, :2] = new_pos
                
                max_radius = calculate_max_feasible_radius(temp_circles, i)
                
                if max_radius > best_radius + 1e-6:
                    best_pos = new_pos
                    best_radius = max_radius
                    improved = True
            
            if improved:
                circles[i, :2] = best_pos
                circles[i, 2] = best_radius
        
        if not improved:
            break
    
    return circles

def calculate_max_feasible_radius(circles, index):
    """Calculate maximum feasible radius for circle at given index"""
    pos = circles[index, :2]
    current_radius = circles[index, 2]
    
    # Start with current radius as minimum
    max_radius = current_radius
    
    # Check containment constraints
    max_radius = min(max_radius, pos[0])  # x >= radius
    max_radius = min(max_radius, 1 - pos[0])  # x <= 1 - radius
    max_radius = min(max_radius, pos[1])  # y >= radius
    max_radius = min(max_radius, 1 - pos[1])  # y <= 1 - radius
    
    # Check overlap constraints with all other circles
    for i in range(len(circles)):
        if i != index:
            other_pos = circles[i, :2]
            other_radius = circles[i, 2]
            
            # Distance to other circle center
            dx = pos[0] - other_pos[0]
            dy = pos[1] - other_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # To avoid overlap: distance >= radius + other_radius
            # So: radius <= distance - other_radius
            max_radius_for_this_circle = distance - other_radius
            
            if max_radius_for_this_circle > 0:
                max_radius = min(max_radius, max_radius_for_this_circle)
    
    # Make sure we don't go negative
    return max(0.001, max_radius)

def final_refinement(circles):
    """Apply final refinement to improve the solution"""
    # Perform one final round of optimization with better constraints
    n = len(circles)
    
    # Create a more refined optimization approach
    for _ in range(100):
        improved = False
        
        # Focus on improving radii first
        for i in range(n):
            old_radius = circles[i, 2]
            max_radius = calculate_max_feasible_radius(circles, i)
            
            if max_radius > old_radius + 1e-6:
                circles[i, 2] = max_radius
                improved = True
        
        # Then fine-tune positions
        for i in range(n):
            old_pos = circles[i, :2].copy()
            old_radius = circles[i, 2]
            
            # Try to slightly adjust position to potentially increase radius
            best_pos = old_pos.copy()
            best_radius = old_radius
            
            # Try several small moves
            for _ in range(5):
                dx = random.uniform(-0.005, 0.005)
                dy = random.uniform(-0.005, 0.005)
                new_pos = old_pos + np.array([dx, dy])
                
                # Keep within bounds
                new_pos[0] = np.clip(new_pos[0], 0.001, 0.999)
                new_pos[1] = np.clip(new_pos[1], 0.001, 0.999)
                
                temp_circles = circles.copy()
                temp_circles[i, :2] = new_pos
                
                max_radius = calculate_max_feasible_radius(temp_circles, i)
                
                if max_radius > best_radius + 1e-6:
                    best_pos = new_pos
                    best_radius = max_radius
                    improved = True
            
            if improved:
                circles[i, :2] = best_pos
                circles[i, 2] = best_radius
    
    return circles


# EVOLVE-BLOCK-END
