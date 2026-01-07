# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import math
from sklearn.cluster import KMeans
import warnings
from numba import jit, prange

# Set random seed for reproducibility
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-stage approach combining geometric initialization and advanced optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Stage 1: Multi-start initialization with better strategies
    best_circles = None
    best_sum = 0
    
    # Try multiple initialization strategies
    for i in range(10):  # More random starts for better exploration
        if i == 0:
            # Use hexagonal grid with better spacing
            circles = initialize_better_hexagonal_grid(n)
        elif i == 1:
            # Use cluster-based initialization
            circles = initialize_clustered(n)
        elif i == 2:
            # Use a more systematic grid approach
            circles = initialize_systematic_grid(n)
        elif i == 3:
            # Use a physics-inspired approach with repulsion
            circles = initialize_physics_based(n)
        else:
            # Random initialization with better constraints
            circles = initialize_random_better(n)
        
        # Stage 2: Optimization with multiple approaches
        optimized = optimize_with_improved_methods(circles)
        
        # Keep the best solution
        current_sum = np.sum(optimized[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized.copy()
    
    return best_circles

def initialize_systematic_grid(n: int) -> np.ndarray:
    """Initialize using a systematic grid approach."""
    # Try to fill the square systematically with circles
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Make sure we have enough space
    while rows * cols < n:
        rows += 1
        cols = int(np.ceil(n / rows))
    
    # Calculate spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    circles = []
    
    # Create a more sophisticated grid with some randomness
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            # Add slight randomness to positions for better distribution
            x_offset = (np.random.random() - 0.5) * spacing_x * 0.2
            y_offset = (np.random.random() - 0.5) * spacing_y * 0.2
            x = (j + 0.5) * spacing_x + x_offset
            y = (i + 0.5) * spacing_y + y_offset
            
            # Ensure we're within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Estimate radius based on proximity to boundaries
                min_dist = min(x, 1-x, y, 1-y)
                max_radius = min_dist / 2.0
                # Use a more aggressive but safe estimate
                radius = max_radius * 0.7
                
                circles.append([x, y, radius])
    
    # Fill remaining slots if needed
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        min_dist = min(x, 1-x, y, 1-y)
        # More aggressive radius estimate
        radius = min_dist * 0.4
        circles.append([x, y, radius])
    
    return np.array(circles[:n])

def initialize_physics_based(n: int) -> np.ndarray:
    """Initialize using a physics-inspired approach."""
    # Start with a regular grid and apply forces
    circles = initialize_systematic_grid(n)
    
    # Apply simple repulsion force to spread circles
    for _ in range(50):  # Apply repulsion iterations
        for i in range(n):
            # Calculate repulsion from other circles
            total_force_x, total_force_y = 0.0, 0.0
            x, y, r = circles[i]
            
            for j in range(n):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x - x2
                    dy = y - y2
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance < (r + r2) * 1.1:  # Within interaction distance
                        # Repulsion force
                        if distance > 0.001:
                            force_magnitude = 0.001 / (distance * distance)
                            total_force_x += force_magnitude * dx / distance
                            total_force_y += force_magnitude * dy / distance
            
            # Apply forces with some damping
            new_x = x + total_force_x * 0.01
            new_y = y + total_force_y * 0.01
            
            # Keep within bounds
            new_x = np.clip(new_x, r, 1-r)
            new_y = np.clip(new_y, r, 1-r)
            
            circles[i] = [new_x, new_y, r]
    
    return circles

def initialize_better_hexagonal_grid(n: int) -> np.ndarray:
    """Initialize circle positions using a refined hexagonal grid pattern."""
    # More precise hexagonal grid calculation
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Ensure we have enough space
    while rows * cols < n:
        rows += 1
        cols = int(np.ceil(n / rows))
    
    # Calculate spacing with better distribution
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Create hexagonal pattern with proper offsetting
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = (i % 2) * spacing_x / 2
            x = (j + 0.5) * spacing_x + x_offset
            y = (i + 0.5) * spacing_y
            
            # Ensure we're within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                # Better radius estimation based on nearest neighbors
                min_dist = min(x, 1-x, y, 1-y)
                max_radius = min_dist / 2.0
                # Use a more conservative estimate to allow room for optimization
                radius = max_radius * 0.6
                
                circles.append([x, y, radius])
    
    # Fill remaining slots if needed
    while len(circles) < n:
        # Place randomly with better constraints
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        min_dist = min(x, 1-x, y, 1-y)
        # Conservative radius estimate
        radius = min_dist * 0.3
        circles.append([x, y, radius])
    
    return np.array(circles[:n])

def initialize_clustered(n: int) -> np.ndarray:
    """Initialize circles using k-means clustering approach."""
    # Generate random points first
    points = np.random.rand(n, 2)
    
    # Use k-means to find cluster centers (we'll use fewer clusters for better distribution)
    k = min(8, n)  # Use fewer clusters to get well-separated centers
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(points)
    
    # Get cluster centers and assign radii
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Create circles with appropriate radii
    circles = []
    for i in range(min(len(centers), n)):
        x, y = centers[i]
        # Estimate radius based on proximity to other centers
        distances = [np.sqrt((x - centers[j][0])**2 + (y - centers[j][1])**2) 
                     for j in range(len(centers)) if j != i]
        if distances:
            min_dist = min(distances)
            # Radius should be about half the minimum distance to neighbors
            radius = min_dist / 4.0
        else:
            radius = 0.1
            
        # Ensure within bounds
        radius = min(radius, x, 1-x, y, 1-y)
        radius = max(radius, 0.01)
        
        circles.append([x, y, radius])
    
    # Fill any remaining slots
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        min_dist = min(x, 1-x, y, 1-y)
        radius = min_dist * 0.25
        circles.append([x, y, radius])
    
    return np.array(circles[:n])

def initialize_random_better(n: int) -> np.ndarray:
    """Better random initialization with constraint awareness."""
    circles = []
    
    # Start with some circles placed in corners and center
    corner_positions = [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]
    center_positions = [(0.5, 0.5)]
    
    for x, y in corner_positions:
        if len(circles) < n:
            radius = min(x, 1-x, y, 1-y) * 0.3
            circles.append([x, y, radius])
    
    for x, y in center_positions:
        if len(circles) < n:
            radius = min(x, 1-x, y, 1-y) * 0.4
            circles.append([x, y, radius])
    
    # Fill remaining with random positions respecting constraints
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        # Estimate maximum possible radius considering nearby circles
        min_dist = min(x, 1-x, y, 1-y)
        radius = min_dist * 0.25
        circles.append([x, y, radius])
    
    return np.array(circles)

def create_optimization_variables(circles: np.ndarray) -> np.ndarray:
    """Convert circle data to optimization variables (x, y, r for each circle)."""
    return circles.flatten()

def unpack_optimization_variables(vars: np.ndarray) -> np.ndarray:
    """Convert optimization variables back to circle data."""
    return vars.reshape(-1, 3)

@jit(nopython=True, parallel=True)
def compute_constraints_fast(circles: np.ndarray) -> tuple:
    """Fast computation of constraints using numba."""
    n = len(circles)
    containment_constraints = np.zeros(n * 4)
    overlap_constraints = np.zeros(n * (n - 1) // 2)
    
    # Containment constraints
    for i in range(n):
        x, y, r = circles[i]
        containment_constraints[i*4] = x - r  # x - r >= 0
        containment_constraints[i*4 + 1] = 1 - x - r  # 1 - x - r >= 0
        containment_constraints[i*4 + 2] = y - r  # y - r >= 0
        containment_constraints[i*4 + 3] = 1 - y - r  # 1 - y - r >= 0
    
    # Overlap constraints
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            overlap_constraints[idx] = distance - (r1 + r2)
            idx += 1
    
    return containment_constraints, overlap_constraints

def objective_function(vars: np.ndarray) -> float:
    """Objective function to maximize sum of radii."""
    circles = unpack_optimization_variables(vars)
    return -np.sum(circles[:, 2])  # Negative because we minimize

def constraint_containment(vars: np.ndarray) -> np.ndarray:
    """Constraint function for containment (all circles within unit square)."""
    circles = unpack_optimization_variables(vars)
    n = len(circles)
    
    # Each circle must satisfy: r <= x <= 1-r and r <= y <= 1-r
    # This means: x - r >= 0, 1-x - r >= 0, y - r >= 0, 1-y - r >= 0
    constraints = np.empty(n * 4)
    
    for i in range(n):
        x, y, r = circles[i]
        constraints[i*4] = x - r           # x - r >= 0
        constraints[i*4 + 1] = 1 - x - r   # 1 - x - r >= 0
        constraints[i*4 + 2] = y - r       # y - r >= 0
        constraints[i*4 + 3] = 1 - y - r   # 1 - y - r >= 0
    
    return constraints

def constraint_overlaps(vars: np.ndarray) -> np.ndarray:
    """Constraint function for non-overlapping (distance >= sum of radii)."""
    circles = unpack_optimization_variables(vars)
    n = len(circles)
    
    # For each pair of circles, ensure distance >= sum of radii
    # This creates a constraint: distance - (r1 + r2) >= 0
    constraints = np.empty(n * (n - 1) // 2)
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # We want: distance >= r1 + r2, which means: distance - (r1 + r2) >= 0
            constraints[idx] = distance - (r1 + r2)
            idx += 1
    
    return constraints

def optimize_with_improved_methods(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle placement using multiple strategies."""
    
    # Convert to optimization variables
    initial_vars = create_optimization_variables(initial_circles)
    
    # Define constraints
    # Containment constraints: all must be >= 0
    containment_cons = {
        'type': 'ineq',
        'fun': constraint_containment
    }
    
    # Overlap constraints: all must be >= 0
    overlap_cons = {
        'type': 'ineq', 
        'fun': constraint_overlaps
    }
    
    # Bounds for variables: [x, y, r] for each circle
    # x: [0, 1], y: [0, 1], r: [0, 0.5] (radius bounded by square size)
    bounds = []
    for i in range(len(initial_circles)):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Try multiple optimization methods with better parameters
    methods_to_try = ['trust-constr', 'SLSQP']
    best_result = None
    best_sum = -np.inf
    
    for method in methods_to_try:
        try:
            # Use a more aggressive optimization approach
            result = minimize(
                objective_function,
                initial_vars,
                method=method,
                bounds=bounds,
                constraints=[containment_cons, overlap_cons],
                options={
                    'maxiter': 300, 
                    'ftol': 1e-8, 
                    'gtol': 1e-8,
                    'eps': 1e-6
                },
                # callback=lambda x: print(f"Method {method} - Current objective: {-objective_function(x)}")
            )
            
            if result.success:
                current_sum = -objective_function(result.x)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            continue
    
    # If we found a good result, refine it
    if best_result is not None:
        optimized_circles = unpack_optimization_variables(best_result.x)
        return validate_and_refine(optimized_circles)
    else:
        # Fallback to the initial configuration with refinement
        return validate_and_refine(initial_circles)

def validate_and_refine(circles: np.ndarray) -> np.ndarray:
    """Validate constraints and perform final refinement."""
    # Ensure all circles are within bounds and have positive radii
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Keep circle within bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        r = max(r, 0.001)  # Ensure positive radius
        circles[i] = [x, y, r]
    
    # Perform more sophisticated iterative improvement
    improved = True
    iterations = 0
    max_iterations = 50  # Reduced for speed
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to improve each circle individually
        for i in range(len(circles)):
            original = circles[i].copy()
            
            # Try to increase radius while maintaining constraints
            step_size = 0.001
            test_radius = min(original[2] + step_size, 0.5)
            
            # Check if we can increase radius without violating constraints
            valid = True
            for j in range(len(circles)):
                if i != j:
                    dist = np.sqrt((original[0] - circles[j][0])**2 + 
                                 (original[1] - circles[j][1])**2)
                    if dist < test_radius + circles[j][2]:
                        valid = False
                        break
            
            if valid:
                # Try to move the circle slightly to accommodate larger radius
                x_new = np.clip(original[0], test_radius, 1-test_radius)
                y_new = np.clip(original[1], test_radius, 1-test_radius)
                
                # Check if this movement still maintains constraints
                valid_move = True
                for j in range(len(circles)):
                    if i != j:
                        dist = np.sqrt((x_new - circles[j][0])**2 + 
                                     (y_new - circles[j][1])**2)
                        if dist < test_radius + circles[j][2]:
                            valid_move = False
                            break
                
                if valid_move:
                    circles[i] = [x_new, y_new, test_radius]
                    improved = True
                    continue
            
            # Try adjusting position while keeping same radius
            if not improved:
                # Try to move to a better location while preserving radius
                x_test = np.clip(original[0], original[2], 1-original[2])
                y_test = np.clip(original[1], original[2], 1-original[2])
                
                # Only update if there's a meaningful change
                if abs(x_test - original[0]) > 1e-6 or abs(y_test - original[1]) > 1e-6:
                    circles[i] = [x_test, y_test, original[2]]
                    improved = True
    
    # Final pass: try to slightly increase all radii if possible
    # This is a greedy improvement step
    for _ in range(30):  # Reduced iterations for speed
        improved_local = False
        for i in range(len(circles)):
            original = circles[i].copy()
            # Try to increase radius slightly
            test_radius = min(original[2] + 0.0005, 0.5)
            
            # Check if we can increase radius without conflicts
            valid = True
            for j in range(len(circles)):
                if i != j:
                    dist = np.sqrt((original[0] - circles[j][0])**2 + 
                                 (original[1] - circles[j][1])**2)
                    if dist < test_radius + circles[j][2]:
                        valid = False
                        break
            
            if valid:
                circles[i] = [original[0], original[1], test_radius]
                improved_local = True
        
        if not improved_local:
            break
    
    return circles


# EVOLVE-BLOCK-END
