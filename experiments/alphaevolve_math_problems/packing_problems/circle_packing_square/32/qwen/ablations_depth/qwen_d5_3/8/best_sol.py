# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations
from collections import defaultdict

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
    
    # Multi-start strategy with different initialization methods
    seeds = [42, 123, 456, 789, 987, 111, 222, 333]
    
    for seed in seeds:
        np.random.seed(seed)
        circles = _optimize_single_start(n)
        
        # Calculate sum of radii
        radii_sum = np.sum(circles[:, 2])
        
        if radii_sum > best_sum:
            best_sum = radii_sum
            best_circles = circles.copy()
    
    # If no improvement found, try a more systematic approach
    if best_circles is None:
        best_circles = _optimize_with_improved_initialization(n)
    
    return best_circles

def _optimize_single_start(n: int) -> np.ndarray:
    """Perform optimization from a single starting configuration."""
    
    # Initialize with a good starting configuration
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
            options={'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-6},
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
    
    # Apply advanced local refinement to improve solution
    circles = _advanced_local_refinement(circles)
    
    return circles

def _optimize_with_improved_initialization(n: int) -> np.ndarray:
    """Try a more systematic approach with better initialization."""
    # Start with a denser grid initialization
    positions = []
    grid_size = int(np.ceil(np.sqrt(n)) + 1)
    
    # Create a dense grid pattern
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) < n:
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                # Add more substantial jitter
                x += np.random.uniform(-0.03, 0.03)
                y += np.random.uniform(-0.03, 0.03)
                # Clamp to valid range
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                positions.append([x, y])
    
    # Fill remaining positions randomly
    while len(positions) < n:
        positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
    
    positions = positions[:n]
    
    # Initialize radii based on density - smaller for more crowded areas
    initial_radii = []
    for i in range(n):
        # Start with small radius, then adjust based on neighbors
        initial_radii.append(0.03)
    
    # Combine into single vector
    initial_guess = []
    for i in range(n):
        initial_guess.extend([positions[i][0], positions[i][1], initial_radii[i]])
    
    # Objective function
    def objective(x_vec):
        return -np.sum(x_vec[2::3])
    
    # Constraints
    def constraint_func(x_vec):
        return _constraint_function(x_vec, n)
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
    
    # Set up constraints
    constraints = {'type': 'ineq', 'fun': constraint_func}
    
    # Solve optimization problem
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            final_solution = result.x
        else:
            final_solution = initial_guess
            
    except Exception as e:
        final_solution = initial_guess
    
    # Extract final positions and radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [final_solution[3*i], final_solution[3*i+1], final_solution[3*i+2]]
    
    # Ensure minimum radius
    for i in range(n):
        if circles[i, 2] < 0.001:
            circles[i, 2] = 0.001
    
    # Final refinement
    circles = _advanced_local_refinement(circles)
    
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
    # Use more efficient pairwise comparison by checking only nearby pairs
    # Convert to numpy array for faster processing
    positions = np.array(x_vec).reshape(-1, 3)
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = positions[i]
            x2, y2, r2 = positions[j]
            # Distance between centers must be >= sum of radii
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_dist_sq = (r1 + r2)**2
            # We want dist_sq >= min_dist_sq, so we return dist_sq - min_dist_sq
            constraints.append(dist_sq - min_dist_sq)
    
    return np.array(constraints)

def _advanced_local_refinement(circles: np.ndarray, max_iter: int = 15) -> np.ndarray:
    """Apply advanced local refinement to improve the solution."""
    n = len(circles)
    
    # Use a more sophisticated approach that considers neighborhood effects
    for iteration in range(max_iter):
        improved = False
        
        # Try to increase all radii simultaneously considering constraints
        # This is a simplified version of a more complex optimization
        for i in range(n):
            current_radius = circles[i, 2]
            
            if current_radius < 0.45:  # Reasonable upper bound for growth
                # Find maximum possible radius for this circle
                max_radius = 0.5
                
                # Check containment constraints
                max_radius = min(max_radius, circles[i, 0])  # x - r >= 0
                max_radius = min(max_radius, 1 - circles[i, 0])  # 1 - x - r >= 0
                max_radius = min(max_radius, circles[i, 1])  # y - r >= 0
                max_radius = min(max_radius, 1 - circles[i, 1])  # 1 - y - r >= 0
                
                # Check non-overlap constraints with all other circles
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + 
                                     (circles[i, 1] - circles[j, 1])**2)
                        # Allow a small safety margin
                        max_radius = min(max_radius, dist - circles[j, 2] - 1e-6)
                
                # Only increase if there's significant improvement
                if max_radius > current_radius + 1e-5:
                    circles[i, 2] = max_radius
                    improved = True
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
