# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from scipy.spatial import cKDTree
import random
from sklearn.cluster import KMeans

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining clustering initialization, gradient-based optimization
    and sophisticated local refinement.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 26
    
    # Try multiple optimization runs with different initializations to find the best solution
    best_circles = None
    best_sum = -float('inf')
    
    # Run multiple optimization attempts with different initial conditions
    # Following INSPIRATION 1 approach with 8 attempts
    for attempt in range(8):  # Increase from 5 to 8 to match INSPIRATION 1
        # Initialize circles using multiple strategies
        circles = initialize_multiple_strategies(n)
        
        # Optimize using gradient-based method with constraints
        optimized_circles = optimize_circles(circles)
        
        # Apply local refinement for further improvement
        refined_circles = local_refinement(optimized_circles)
        
        # Evaluate this solution
        current_sum = np.sum(refined_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = refined_circles.copy()
    
    return best_circles

def initialize_multiple_strategies(n: int) -> np.ndarray:
    """Initialize circles using multiple strategies and return the best one"""
    # Strategy 1: Clustering approach like INSPIRATION 1
    circles1 = cluster_initial_placement(n)
    sum1 = np.sum(circles1[:, 2])
    
    # Strategy 2: Grid approach
    circles2 = create_grid_initial_placement(n)
    sum2 = np.sum(circles2[:, 2])
    
    # Strategy 3: Improved hexagonal packing
    circles3 = initialize_improved_hexagonal_packing(n)
    sum3 = np.sum(circles3[:, 2])
    
    # Strategy 4: Random valid placement
    circles4 = create_random_valid_placement(n)
    sum4 = np.sum(circles4[:, 2])
    
    # Return the best initialization
    max_sum = max(sum1, sum2, sum3, sum4)
    if max_sum == sum1:
        return circles1
    elif max_sum == sum2:
        return circles2
    elif max_sum == sum3:
        return circles3
    else:
        return circles4

def cluster_initial_placement(n: int) -> np.ndarray:
    """Use clustering to get good initial placement like INSPIRATION 1"""
    # Generate candidate points for clustering
    n_points = 1500
    candidates = np.random.rand(n_points, 2)
    
    # Use k-means to find dense regions
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=30)
    kmeans.fit(candidates)
    
    # Initialize circles at cluster centers with small radii
    circles = np.zeros((n, 3))
    centers = kmeans.cluster_centers_
    
    # Distribute circles with reasonable radii
    for i in range(n):
        circles[i, 0] = centers[i, 0]  # x
        circles[i, 1] = centers[i, 1]  # y
        # Start with small radii, will be optimized later
        circles[i, 2] = 0.02
    
    return circles

def create_grid_initial_placement(n: int) -> np.ndarray:
    """Create a structured initial placement like INSPIRATION 2"""
    circles = np.zeros((n, 3))
    
    # Create a 5x6 grid pattern
    rows = 5
    cols = 6
    
    # Calculate spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Place circles in a grid pattern with jitter
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Position with slight jitter to avoid perfect grid
            x = (j + 0.5) * spacing_x + np.random.normal(0, 0.01)
            y = (i + 0.5) * spacing_y + np.random.normal(0, 0.01)
            
            # Clip to valid range
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            
            # Initial radius
            r = 0.05
            
            circles[idx] = [x, y, r]
            idx += 1
    
    return circles

def create_random_valid_placement(n: int) -> np.ndarray:
    """Create a random valid initial placement"""
    circles = np.zeros((n, 3))
    max_attempts = 1000
    
    for attempt in range(max_attempts):
        # Generate random positions and radii
        for i in range(n):
            # Random position within bounds (considering radius)
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            # Random radius (small initial value)
            r = random.uniform(0.01, 0.1)
            
            circles[i] = [x, y, r]
        
        # Check validity
        if is_valid_configuration(circles):
            return circles
    
    # If we couldn't generate a valid configuration, use a simple heuristic
    # Place circles in a grid-like pattern with small radii
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = 0.1 + j * 0.8 / (cols - 1) if cols > 1 else 0.5
            y = 0.1 + i * 0.8 / (rows - 1) if rows > 1 else 0.5
            r = 0.05
            
            circles[idx] = [x, y, r]
            idx += 1
    
    # Ensure it's valid
    if not is_valid_configuration(circles):
        # If still invalid, just fill with minimal valid setup
        for i in range(n):
            circles[i] = [0.5, 0.5, 0.01]
    
    return circles

def initialize_improved_hexagonal_packing(n: int) -> np.ndarray:
    """Initialize circles using a more sophisticated hexagonal packing pattern"""
    # Create a 5x5 grid pattern for 25 circles, then add one more
    rows = 5
    cols = 5
    
    # Calculate spacing
    spacing_x = 0.9 / (cols - 1) if cols > 1 else 0.5
    spacing_y = 0.9 / (rows - 1) if rows > 1 else 0.5
    
    # Determine radius based on spacing
    min_radius = min(spacing_x, spacing_y) * 0.3
    
    circles = []
    count = 0
    
    # Create hexagonal grid pattern
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
                
            # Hexagonal offset for odd rows
            x_offset = 0.5 if i % 2 == 1 else 0.0
            x = 0.05 + (j + x_offset) * spacing_x
            y = 0.05 + i * spacing_y
            
            # Add small random perturbation to avoid perfect symmetry
            x += np.random.uniform(-0.005, 0.005)
            y += np.random.uniform(-0.005, 0.005)
            
            # Ensure within bounds
            x = max(min_radius, min(1.0 - min_radius, x))
            y = max(min_radius, min(1.0 - min_radius, y))
            
            circles.append([x, y, min_radius])
            count += 1
            
        if count >= n:
            break
    
    # Fill remaining circles with random positions
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        # Larger initial radius for diversity
        r = np.random.uniform(0.02, 0.08)
        circles.append([x, y, r])
    
    return np.array(circles)

def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using constrained optimization"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(params):
        circles = params.reshape((n, 3))
        radii_sum = np.sum(circles[:, 2])
        return -radii_sum  # Negative because we minimize
    
    # Define constraints with better handling
    def containment_constraint(params):
        circles = params.reshape((n, 3))
        constraints = []
        
        # Each circle must be fully contained in unit square
        for i in range(n):
            x, y, r = circles[i]
            # r <= x <= 1-r and r <= y <= 1-r
            constraints.append(x - r)  # x >= r
            constraints.append(1 - x - r)  # 1-x >= r, so x+r <= 1
            constraints.append(y - r)  # y >= r
            constraints.append(1 - y - r)  # 1-y >= r, so y+r <= 1
            
        return np.array(constraints)
    
    def non_overlap_constraint(params):
        circles = params.reshape((n, 3))
        constraints = []
        
        # Use spatial indexing to reduce number of checks but still do full check for correctness
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Distance between centers
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # We want dist >= r1 + r2, so we constrain dist_sq >= min_dist_sq
                # This means: dist_sq - min_dist_sq >= 0
                constraints.append(dist_sq - min_dist_sq)
                
        return np.array(constraints)
    
    # Set up constraints for scipy.optimize
    cons = []
    
    # Containment constraints (each must be >= 0)
    cons.append({'type': 'ineq', 'fun': lambda p: containment_constraint(p)})
    
    # Non-overlap constraints (each must be >= 0)
    cons.append({'type': 'ineq', 'fun': lambda p: non_overlap_constraint(p)})
    
    # Bounds for parameters: [x, y, r] for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Perform optimization with better settings
    try:
        # Try multiple optimization methods to get better results
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_sum = -float('inf')
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 2500, 'ftol': 1e-9, 'gtol': 1e-9, 'disp': False}
                )
                
                if result.success:
                    # Check if this is better than our previous best
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception:
                continue
                
        if best_result is not None and best_result.success:
            optimized_circles = best_result.x.reshape((n, 3))
            # Ensure all circles are valid
            for i in range(n):
                x, y, r = optimized_circles[i]
                # Clamp to valid ranges
                optimized_circles[i] = [
                    max(0.001, min(0.999, x)),
                    max(0.001, min(0.999, y)), 
                    max(0.001, min(0.499, r))
                ]
            return optimized_circles
    except Exception as e:
        pass
    
    # Fallback: return initial configuration if optimization fails
    return initial_circles

def local_refinement(circles: np.ndarray) -> np.ndarray:
    """Apply local refinement to improve the solution like INSPIRATION 1"""
    # Create a copy to work with
    refined = circles.copy()
    
    # More thorough iterative improvement like INSPIRATION 1
    for _ in range(150):  # More iterations for better refinement
        improved = False
        # Try to improve each circle individually
        for i in range(len(circles)):
            # Save original
            orig_x, orig_y, orig_r = refined[i, 0], refined[i, 1], refined[i, 2]
            
            # Try small adjustments with more diverse perturbations like INSPIRATION 1
            for _ in range(30):  # More tries per circle
                # Perturb position and radius with varying magnitudes
                dx = np.random.normal(0, 0.005)
                dy = np.random.normal(0, 0.005)
                dr = np.random.normal(0, 0.003)
                
                new_x = np.clip(orig_x + dx, 0.01, 0.99)
                new_y = np.clip(orig_y + dy, 0.01, 0.99)
                new_r = np.clip(orig_r + dr, 0.001, 0.4)
                
                # Test if this change helps
                test_circles = refined.copy()
                test_circles[i, 0] = new_x
                test_circles[i, 1] = new_y
                test_circles[i, 2] = new_r
                
                # Check if valid and if improvement is made
                if is_valid_configuration(test_circles):
                    # Calculate improvement with better comparison
                    old_obj = -np.sum(refined[:, 2])  # Current sum (negative)
                    new_obj = -np.sum(test_circles[:, 2])  # New sum (negative)
                    
                    if new_obj < old_obj:  # Better (lower) objective means improvement
                        refined = test_circles.copy()
                        improved = True
                        break
        
        # If no improvements were made, stop early
        if not improved:
            break
            
    return refined

def is_valid_configuration(circles):
    """Check if the entire configuration is valid"""
    n = len(circles)
    
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if not (r <= x <= 1 - r and r <= y <= 1 - r):
            return False
    
    # Check overlaps using more efficient approach
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Calculate pairwise distances
    distances = cdist(positions, positions)
    
    # Check for overlaps
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            if dist < radii[i] + radii[j]:
                return False
    
    return True


# EVOLVE-BLOCK-END
