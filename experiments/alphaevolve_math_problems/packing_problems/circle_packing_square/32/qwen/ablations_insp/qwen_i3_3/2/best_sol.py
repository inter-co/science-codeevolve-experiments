# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
import random
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal initialization, multiple optimization attempts,
    and post-processing refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    n = 32
    
    # Better hexagonal initialization with proper spacing
    def initialize_hexagonal_layout():
        circles = []
        sqrt3 = np.sqrt(3)
        
        # Determine grid dimensions for approximately 32 circles
        rows = int(np.ceil(np.sqrt(n * 2 / sqrt3)))
        cols = int(np.ceil(n / rows))
        
        # Spacing based on hexagonal packing
        radius = 0.08  # Starting radius estimate
        horizontal_spacing = 2 * radius
        vertical_spacing = sqrt3 * radius
        
        # Create hexagonal grid
        for i in range(rows):
            y = radius + i * vertical_spacing
            if y > 1 - radius:
                break
            for j in range(cols):
                x = radius + j * horizontal_spacing
                if x > 1 - radius:
                    break
                # Offset every other row
                if i % 2 == 1:
                    x += horizontal_spacing / 2
                if x <= 1 - radius and y <= 1 - radius:
                    circles.append([x, y, radius])
        
        # Fill remaining circles with random placement near grid points
        while len(circles) < n:
            # Add random placements near existing grid points
            if circles:
                base_idx = np.random.randint(len(circles))
                base_x, base_y, base_r = circles[base_idx]
                x = np.clip(base_x + np.random.normal(0, 0.03), base_r, 1-base_r)
                y = np.clip(base_y + np.random.normal(0, 0.03), base_r, 1-base_r)
                circles.append([x, y, base_r])
            else:
                # If no circles yet, place randomly
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                circles.append([x, y, 0.05])
                
        return np.array(circles[:n])
    
    # Alternative: Voronoi-based initialization
    def initialize_voronoi():
        np.random.seed(42)  # For reproducibility
        points = np.random.rand(50, 2)  # More points than needed for Voronoi
        
        try:
            from scipy.spatial import Voronoi
            vor = Voronoi(points)
            # Use Voronoi cell centroids as initial positions (but keep within bounds)
            positions = []
            for i in range(min(n, len(vor.points))):
                point = vor.points[i]
                x = np.clip(point[0], 0.05, 0.95)
                y = np.clip(point[1], 0.05, 0.95)
                positions.append([x, y])
            
            # Fill remaining positions randomly
            while len(positions) < n:
                positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
                
            return np.array(positions[:n])
            
        except Exception:
            # Fallback to grid initialization
            positions = []
            grid_size = int(np.ceil(np.sqrt(n)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(positions) >= n:
                        break
                    x = (j + 1) * spacing_x
                    y = (i + 1) * spacing_y
                    positions.append([x, y])
                    
            # Fill remaining positions randomly
            while len(positions) < n:
                positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
                
            return np.array(positions[:n])
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        return -np.sum(params[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Constraint functions
    def constraint_containment(params):
        # Ensure all circles fit inside the unit square
        constraints = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            # Circle must stay inside square: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
            constraints.extend([
                x - r,           # x >= r
                1 - x - r,       # 1 - x >= r
                y - r,           # y >= r
                1 - y - r        # 1 - y >= r
            ])
        return np.array(constraints)
    
    # Non-overlap constraints with optimized vectorized computation
    def constraint_nonoverlap(params):
        # Reshape params into (n, 3) array for easier access
        coords = params.reshape((n, 3))
        x = coords[:, 0]
        y = coords[:, 1] 
        r = coords[:, 2]
        
        # Compute pairwise distances efficiently using broadcasting
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        dist_sq = diff_x**2 + diff_y**2
        dist = np.sqrt(dist_sq)
        
        # Create constraint values for all pairs (i,j) where i < j
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                # Non-overlap constraint: distance >= r1 + r2
                constraints.append(dist[i, j] - (r[i] + r[j]))
        return np.array(constraints)
    
    # Bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])  # x, y, r
    
    # Constraints list
    cons = [
        {'type': 'ineq', 'fun': constraint_containment},
        {'type': 'ineq', 'fun': constraint_nonoverlap}
    ]
    
    # Enhanced optimization approach with multiple methods
    def optimize_circles(initial_circles):
        # Flatten for optimization
        initial_params = initial_circles.flatten()
        
        # Try multiple optimization methods for better results
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_value = float('inf')
        
        for method in methods:
            try:
                result = minimize(
                    objective, 
                    initial_params, 
                    method=method, 
                    bounds=bounds, 
                    constraints=cons,
                    options={'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-9, 'disp': False}
                )
                
                if result.success:
                    # Check if this is better than previous attempts
                    if result.fun < best_value:
                        best_value = result.fun
                        best_result = result
                        
            except Exception:
                continue
        
        # Return best result or fallback to initial
        if best_result is not None and best_result.success:
            return best_result.x.reshape((n, 3))
        else:
            # If optimization fails, return initial configuration with corrected radii
            corrected = initial_circles.copy()
            for i in range(n):
                x, y, r = corrected[i]
                corrected[i, 2] = min(r, x, 1-x, y, 1-y)
            return corrected
    
    # Enhanced refinement with better convergence criteria
    def refine_circles(circles_in):
        circles_refined = circles_in.copy()
        improved = True
        iteration = 0
        max_iterations = 150
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try to increase radii while maintaining constraints
            for i in range(n):
                current_x, current_y, current_r = circles_refined[i]
                
                # Calculate maximum possible radius for this circle
                max_radius = min(current_x, 1-current_x, current_y, 1-current_y)
                
                # Check overlap with all other circles more carefully
                for j in range(n):
                    if i != j:
                        other_x, other_y, other_r = circles_refined[j]
                        dist = np.sqrt((current_x - other_x)**2 + (current_y - other_y)**2)
                        # Add small epsilon to prevent numerical issues
                        max_radius = min(max_radius, dist - other_r - 1e-8)
                
                # Increase radius if beneficial with better tolerance
                if max_radius > current_r + 1e-6:
                    circles_refined[i, 2] = max_radius
                    improved = True
                    
        return circles_refined
    
    # Final validation
    def validate_and_correct(circles_array):
        corrected = circles_array.copy()
        for i in range(n):
            x, y, r = corrected[i]
            # Ensure circle fits in unit square
            r = min(r, x, 1-x, y, 1-y)
            # Ensure positive radius
            r = max(1e-6, r)
            corrected[i] = [x, y, r]
        return corrected
    
    # Try multiple initialization strategies and optimization attempts
    best_circles = None
    best_sum = 0
    
    # Try different initialization methods with more aggressive attempts
    init_methods = [initialize_hexagonal_layout, initialize_voronoi]
    
    for init_method in init_methods:
        # Try several optimization attempts with same initialization
        for attempt in range(5):  # Increased attempts
            try:
                # Get initial configuration
                if attempt == 0:
                    circles = init_method()
                else:
                    # Perturb previous result with more variation in later attempts
                    circles = best_circles.copy() if best_circles is not None else init_method()
                    for i in range(n):
                        # More significant perturbations in later attempts
                        strength = 0.02 if attempt < 3 else 0.05
                        circles[i, 0] += np.random.normal(0, strength)
                        circles[i, 1] += np.random.normal(0, strength)
                        circles[i, 0] = np.clip(circles[i, 0], 0.01, 0.99)
                        circles[i, 1] = np.clip(circles[i, 1], 0.01, 0.99)
                
                # Optimize with enhanced parameters
                optimized_circles = optimize_circles(circles)
                
                # Refine aggressively
                refined_circles = refine_circles(optimized_circles)
                
                # Validate
                final_circles = validate_and_correct(refined_circles)
                
                # Calculate sum of radii
                radii_sum = np.sum(final_circles[:, 2])
                if radii_sum > best_sum:
                    best_sum = radii_sum
                    best_circles = final_circles
                    
            except Exception:
                continue
    
    # If we still don't have a good solution, return fallback
    if best_circles is None:
        # Final fallback: grid-based solution with better spacing
        best_circles = np.zeros((n, 3))
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                r = min(x, y, 1-x, 1-y) * 0.4
                best_circles[idx] = [x, y, r]
                idx += 1
        # Fill remaining circles
        for i in range(idx, n):
            best_circles[i] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), 0.05]
    
    return best_circles


# EVOLVE-BLOCK-END
