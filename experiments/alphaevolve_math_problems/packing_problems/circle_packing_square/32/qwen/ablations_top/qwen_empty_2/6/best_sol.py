# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
from scipy.spatial import cKDTree

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining grid initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = 32
    circles = np.zeros((n, 3))
    
    # Phase 1: Grid-based initialization (like INSPIRATION 1)
    # Place circles in a grid pattern to get a good starting configuration
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing_x = 1.0 / (grid_size + 1)
    spacing_y = 1.0 / (grid_size + 1)
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= n:
                break
            x = (i + 1) * spacing_x
            y = (j + 1) * spacing_y
            # Initial radius - small enough to fit in the grid cell
            r = min(spacing_x, spacing_y) * 0.4
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Ensure we have exactly n circles
    while idx < n:
        # Fill remaining positions with small random circles
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        r = 0.01
        circles[idx] = [x, y, r]
        idx += 1
    
    # Phase 2: Enhanced optimization with multiple strategies and high precision (like INSPIRATION 1 & 2)
    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
        # Convert to array for easy access
        circles_array = params.reshape((n, 3))
        
        # Extract positions and radii
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        # Calculate total radius (we want to maximize this)
        total_radius = np.sum(radii)
        
        # Add penalty for overlapping circles using vectorized operations for efficiency
        penalty = 0
        
        # Compute all pairwise distances at once
        distances = cdist(positions, positions)
        
        # Create mask for upper triangle (avoid double counting)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Get distances and radii for overlapping pairs
        overlap_distances = distances[mask]
        overlap_radii = (radii[:, None] + radii[None, :])[mask]
        
        # Calculate overlap violations (positive when there's overlap)
        overlap_violations = overlap_radii - overlap_distances
        
        # Apply cubic penalty for overlaps (very strong enforcement)
        overlap_penalty = np.sum(overlap_violations[overlap_violations > 0]**3) * 1000000
        
        # Add penalty for circles going outside bounds (very strong penalty)
        bound_penalty = 0
        for i in range(n):
            x, y, r = circles_array[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                bound_penalty += 10000000000  # Extremely strong penalty
                
        return -total_radius + overlap_penalty + bound_penalty  # negative because we want to maximize
    
    # Set up initial parameters
    initial_params = circles.flatten()
    
    # Create bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # r capped at 0.499 to prevent overlap issues
    
    # Run optimization with multiple strategies and high precision
    try:
        best_result = None
        best_value = float('inf')
        
        # Try L-BFGS-B with very high precision
        try:
            result1 = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
            if result1.success:
                current_value = -result1.fun  # Convert back to maximization value
                if current_value < best_value:
                    best_value = current_value
                    best_result = result1
        except:
            pass
            
        # Try Trust-Region Constrained with very high precision
        try:
            result2 = minimize(objective, initial_params, method='trust-constr', bounds=bounds,
                              options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12})
            if result2.success:
                current_value = -result2.fun  # Convert back to maximization value
                if current_value < best_value:
                    best_value = current_value
                    best_result = result2
        except:
            pass
            
        # Multi-start approach - run optimization from multiple random starting points
        for _ in range(3):  # Try 3 random restarts
            # Generate random starting point
            random_params = initial_params.copy()
            for i in range(n):
                # Randomize positions and radii slightly
                random_params[3*i] = random.uniform(0.05, 0.95)  # x
                random_params[3*i+1] = random.uniform(0.05, 0.95)  # y  
                random_params[3*i+2] = random.uniform(0.01, 0.2)  # r
            
            try:
                result = minimize(objective, random_params, method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10})
                if result.success:
                    current_value = -result.fun
                    if current_value < best_value:
                        best_value = current_value
                        best_result = result
            except:
                continue
                
        # If we found a good optimization result, update circles
        if best_result is not None:
            final_params = best_result.x
            circles = final_params.reshape((n, 3))
    except Exception as e:
        # If optimization fails, keep the initial configuration
        pass
    
    # Phase 3: Enhanced local search with strategic move types and increased iterations (like INSPIRATION 2)
    best_circles = circles.copy()
    best_sum = np.sum(best_circles[:, 2])
    
    # Even more sophisticated local search with strategic weighting and higher iterations
    for iteration in range(30000):  # Significantly more iterations for better search
        # Strategic move type selection with better weights
        move_type = random.choices(
            ['single', 'multiple', 'aggressive'], 
            weights=[0.6, 0.3, 0.1]  # Slight emphasis on single moves for stability
        )[0]
        
        if move_type == 'single':
            # Single circle modification (most common)
            test_circles = best_circles.copy()
            idx = random.randint(0, n-1)
            
            # Use adaptive perturbation sizes based on current radius
            current_radius = test_circles[idx, 2]
            perturbation_scale = min(0.03, current_radius * 0.2)  # Slightly larger scale
            
            test_circles[idx, 0] += random.uniform(-perturbation_scale, perturbation_scale)
            test_circles[idx, 1] += random.uniform(-perturbation_scale, perturbation_scale)
            test_circles[idx, 2] += random.uniform(-perturbation_scale*0.3, perturbation_scale*0.3)
            
        elif move_type == 'multiple':
            # Multiple circle modification (more aggressive exploration)
            test_circles = best_circles.copy()
            # Modify 3-5 random circles
            num_modifications = random.randint(3, 5)
            for _ in range(num_modifications):
                idx = random.randint(0, n-1)
                current_radius = test_circles[idx, 2]
                perturbation_scale = min(0.02, current_radius * 0.15)
                test_circles[idx, 0] += random.uniform(-perturbation_scale, perturbation_scale)
                test_circles[idx, 1] += random.uniform(-perturbation_scale, perturbation_scale)
                test_circles[idx, 2] += random.uniform(-perturbation_scale*0.2, perturbation_scale*0.2)
                
        else:  # aggressive
            # Aggressive move - larger perturbations
            test_circles = best_circles.copy()
            # Modify 2-4 random circles with larger perturbations
            num_modifications = random.randint(2, 4)
            for _ in range(num_modifications):
                idx = random.randint(0, n-1)
                current_radius = test_circles[idx, 2]
                perturbation_scale = min(0.08, current_radius * 0.3)  # Even larger perturbations
                test_circles[idx, 0] += random.uniform(-perturbation_scale, perturbation_scale)
                test_circles[idx, 1] += random.uniform(-perturbation_scale, perturbation_scale)
                test_circles[idx, 2] += random.uniform(-perturbation_scale*0.5, perturbation_scale*0.5)
        
        # Ensure valid bounds
        for i in range(n):
            test_circles[i, 0] = max(0.01, min(0.99, test_circles[i, 0]))
            test_circles[i, 1] = max(0.01, min(0.99, test_circles[i, 1]))
            test_circles[i, 2] = max(0.001, min(0.499, test_circles[i, 2]))
        
        # Check constraints more efficiently using spatial indexing
        valid = True
        for i in range(n):
            x, y, r = test_circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                valid = False
                break
        
        if valid:
            # Check for overlaps more efficiently using spatial indexing
            overlap = False
            
            # Create KDTree for fast neighbor queries
            positions = test_circles[:, :2]
            tree = cKDTree(positions)
            
            # Check for overlaps with nearby circles only - more efficient than full pairwise
            for i in range(n):
                x, y, r = test_circles[i]
                # Search for neighbors within 4*(r + epsilon) distance (even wider search)
                neighbors = tree.query_ball_point([x, y], 4*(r + 0.001))
                
                for j in neighbors:
                    if i != j:
                        dist = np.sqrt((x - test_circles[j, 0])**2 + (y - test_circles[j, 1])**2)
                        if dist < r + test_circles[j, 2]:
                            overlap = True
                            break
                if overlap:
                    break
            
            if not overlap:
                test_sum = np.sum(test_circles[:, 2])
                if test_sum > best_sum:
                    best_circles = test_circles
                    best_sum = test_sum
    
    return best_circles


# EVOLVE-BLOCK-END
