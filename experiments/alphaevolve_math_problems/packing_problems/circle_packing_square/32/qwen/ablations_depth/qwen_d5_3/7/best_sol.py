# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations
import time

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    best_sum = 0
    best_circles = None
    
    # Multi-start strategy with different initialization methods and time limits
    start_time = time.time()
    seeds = [42, 123, 456, 789, 987, 111, 222, 333]
    
    for seed in seeds:
        if time.time() - start_time > 55:  # Leave some time for final processing
            break
            
        np.random.seed(seed)
        circles = _optimize_single_start(n)
        
        # Calculate sum of radii
        radii_sum = np.sum(circles[:, 2])
        
        if radii_sum > best_sum:
            best_sum = radii_sum
            best_circles = circles.copy()
    
    # If no good solution found, try one more optimized run with better initialization
    if best_circles is None or best_sum < 2.9:
        best_circles = _optimized_initialization_and_optimization(n)
    
    return best_circles if best_circles is not None else _optimize_single_start(n)

def _optimized_initialization_and_optimization(n: int) -> np.ndarray:
    """Use a more sophisticated initialization approach."""
    # Generate positions using a hexagonal packing approach for better initial distribution
    positions = _hexagonal_grid_initialization(n)
    
    # Initialize radii to a more reasonable value based on spacing
    initial_radii = []
    for i in range(n):
        # Estimate initial radius based on available space
        initial_radii.append(min(0.1, 0.5))
    
    # Combine into single vector: [x1, y1, r1, x2, y2, r2, ...]
    initial_guess = []
    for i in range(n):
        initial_guess.extend([positions[i][0], positions[i][1], initial_radii[i]])
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x_vec):
        return -np.sum(x_vec[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Constraints for scipy.optimize
    def constraint_func(x_vec):
        # Return both containment and non-overlap constraints
        return _constraint_function(x_vec, n)
    
    # Bounds for variables: [0,1] for x and y, [0,0.5] for r (since max radius is 0.5)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Set up constraints
    constraints = {'type': 'ineq', 'fun': constraint_func}
    
    # Solve optimization problem with more iterations
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6},
            callback=lambda x: None  # No callback needed
        )
        
        if result.success:
            final_solution = result.x
        else:
            # Fallback: use initial guess if optimization fails
            final_solution = initial_guess
            
    except Exception as e:
        # If optimization fails, return initial guess
        final_solution = initial_guess
    
    # Extract final positions and radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [final_solution[3*i], final_solution[3*i+1], final_solution[3*i+2]]
    
    # Ensure minimum radius is not too small
    for i in range(n):
        if circles[i, 2] < 0.001:
            circles[i, 2] = 0.001
    
    # Apply advanced local refinement
    circles = _advanced_local_refinement(circles)
    
    return circles

def _hexagonal_grid_initialization(n: int) -> list:
    """Generate initial positions using a hexagonal grid approach."""
    positions = []
    
    # Hexagonal packing parameters
    side_length = 0.15  # Initial spacing
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Create hexagonal grid
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Hexagonal offset
            x = (j + 0.5 * (i % 2)) * side_length * 2
            y = i * side_length * np.sqrt(3)
            
            # Add jitter to avoid perfect patterns
            x += np.random.uniform(-side_length*0.1, side_length*0.1)
            y += np.random.uniform(-side_length*0.1, side_length*0.1)
            
            # Clamp to valid range
            x = max(side_length, min(1-side_length, x))
            y = max(side_length, min(1-side_length, y))
            
            positions.append([x, y])
    
    # If we don't have enough points, fill with random positions
    while len(positions) < n:
        positions.append([np.random.uniform(side_length, 1-side_length), 
                         np.random.uniform(side_length, 1-side_length)])
    
    return positions[:n]

def _optimize_single_start(n: int) -> np.ndarray:
    """Perform optimization from a single starting configuration."""
    
    # Initialize with a good starting configuration using multiple methods
    initial_positions = _generate_initial_positions(n)
    
    # Initialize radii to small values
    initial_radii = [0.02] * n
    
    # Combine into single vector: [x1, y1, r1, x2, y2, r2, ...]
    initial_guess = []
    for i in range(n):
        initial_guess.extend([initial_positions[i][0], initial_positions[i][1], initial_radii[i]])
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(x_vec):
        return -np.sum(x_vec[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Constraints for scipy.optimize
    def constraint_func(x_vec):
        # Return both containment and non-overlap constraints
        return _constraint_function(x_vec, n)
    
    # Bounds for variables: [0,1] for x and y, [0,0.5] for r (since max radius is 0.5)
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Set up constraints
    constraints = {'type': 'ineq', 'fun': constraint_func}
    
    # Solve optimization problem
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6},
            callback=lambda x: None  # No callback needed
        )
        
        if result.success:
            final_solution = result.x
        else:
            # Fallback: use initial guess if optimization fails
            final_solution = initial_guess
            
    except Exception as e:
        # If optimization fails, return initial guess
        final_solution = initial_guess
    
    # Extract final positions and radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [final_solution[3*i], final_solution[3*i+1], final_solution[3*i+2]]
    
    # Ensure minimum radius is not too small
    for i in range(n):
        if circles[i, 2] < 0.001:
            circles[i, 2] = 0.001
    
    # Apply local refinement to improve solution
    circles = _local_refinement(circles)
    
    return circles

def _generate_initial_positions(n: int) -> list:
    """Generate initial positions using multiple strategies."""
    # Strategy 1: Grid-based with jitter
    positions = []
    grid_size = int(np.ceil(np.sqrt(n)))
    
    # Create a grid pattern with some jitter
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) < n:
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                # Add small random jitter to avoid perfect symmetry
                x += np.random.uniform(-0.02, 0.02)
                y += np.random.uniform(-0.02, 0.02)
                # Clamp to valid range
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                positions.append([x, y])
    
    # If we don't have enough points, fill with random positions
    while len(positions) < n:
        positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
    
    return positions[:n]

def _constraint_function(x_vec, n):
    """Combined constraint function for containment and non-overlap."""
    constraints = []
    
    # Containment constraints: ensure all circles are fully contained
    for i in range(n):
        x = x_vec[3*i]
        y = x_vec[3*i+1]
        r = x_vec[3*i+2]
        # Circle must fit inside square
        constraints.extend([
            x - r,           # x - r >= 0
            1 - x - r,       # 1 - x - r >= 0  
            y - r,           # y - r >= 0
            1 - y - r        # 1 - y - r >= 0
        ])
    
    # Non-overlap constraints: ensure no overlap between circles
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = x_vec[3*i], x_vec[3*i+1], x_vec[3*i+2]
            x2, y2, r2 = x_vec[3*j], x_vec[3*j+1], x_vec[3*j+2]
            # Distance between centers must be >= sum of radii
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_dist_sq = (r1 + r2)**2
            # We want dist_sq >= min_dist_sq, so we return dist_sq - min_dist_sq
            constraints.append(dist_sq - min_dist_sq)
    
    return np.array(constraints)

def _local_refinement(circles: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """Apply basic local refinement to improve the solution."""
    n = len(circles)
    
    # Simple local search: try to increase radii while maintaining constraints
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            # Try to increase radius of circle i
            current_radius = circles[i, 2]
            if current_radius < 0.5:  # Can still grow
                # Find the maximum possible radius
                max_radius = 0.5
                
                # Check containment constraints
                max_radius = min(max_radius, circles[i, 0])  # x - r >= 0
                max_radius = min(max_radius, 1 - circles[i, 0])  # 1 - x - r >= 0
                max_radius = min(max_radius, circles[i, 1])  # y - r >= 0
                max_radius = min(max_radius, 1 - circles[i, 1])  # 1 - y - r >= 0
                
                # Check non-overlap constraints with other circles
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + 
                                     (circles[i, 1] - circles[j, 1])**2)
                        max_radius = min(max_radius, dist - circles[j, 2])
                
                # Increase radius if beneficial
                if max_radius > current_radius + 1e-6:
                    circles[i, 2] = max_radius
                    improved = True
        
        if not improved:
            break
    
    return circles

def _advanced_local_refinement(circles: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """Apply advanced local refinement with better constraint handling."""
    n = len(circles)
    
    # More sophisticated refinement that considers all constraints together
    for iteration in range(max_iter):
        improved = False
        # Try to optimize each circle individually
        for i in range(n):
            # Store original values
            original_x, original_y, original_r = circles[i, 0], circles[i, 1], circles[i, 2]
            
            # Try to increase radius while respecting constraints
            max_radius = 0.5
            
            # Containment constraints
            max_radius = min(max_radius, original_x)
            max_radius = min(max_radius, 1 - original_x)
            max_radius = min(max_radius, original_y)
            max_radius = min(max_radius, 1 - original_y)
            
            # Non-overlap constraints with all other circles
            for j in range(n):
                if i != j:
                    dist = np.sqrt((original_x - circles[j, 0])**2 + 
                                 (original_y - circles[j, 1])**2)
                    max_radius = min(max_radius, dist - circles[j, 2])
            
            # If we can increase radius, do it
            if max_radius > original_r + 1e-6:
                circles[i, 2] = max_radius
                improved = True
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
