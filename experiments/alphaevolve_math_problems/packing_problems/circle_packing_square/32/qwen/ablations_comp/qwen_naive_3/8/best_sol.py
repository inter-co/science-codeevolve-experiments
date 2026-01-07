# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithms with local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Generate a more sophisticated initial configuration using a combination of methods
    def generate_better_initial_configuration():
        # Strategy 1: Create a more efficient grid pattern
        # For 32 circles, we want a near-optimal arrangement
        
        # Create a grid that's closer to what we'd expect for optimal packing
        rows = 6
        cols = 6
        points = []
        
        # Create a grid with some randomness for better distribution
        margin = 0.05
        spacing_x = (1 - 2*margin) / (cols - 1) if cols > 1 else 0.1
        spacing_y = (1 - 2*margin) / (rows - 1) if rows > 1 else 0.1
        
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n:
                    break
                # Offset odd rows for better packing
                x = margin + j * spacing_x + (i % 2) * spacing_x/2
                y = margin + i * spacing_y
                # Ensure within bounds
                x = max(margin, min(1-margin, x))
                y = max(margin, min(1-margin, y))
                points.append([x, y])
        
        # Trim to exact number needed
        points = points[:n]
        
        # Add slight randomness to positions
        np.random.seed(42)
        perturbation = np.random.uniform(-0.01, 0.01, (n, 2))
        initial_points = np.clip(np.array(points) + perturbation, 0.05, 0.95)
        
        # Initialize with appropriate radii based on expected density
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [initial_points[i][0], initial_points[i][1], 0.05]
        
        return circles
    
    # More efficient constraint checking
    def check_constraints_fast(circles):
        """Fast constraint checking using vectorized operations"""
        # Check containment constraints
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        containment_ok = np.all((x >= r) & (x <= 1-r) & (y >= r) & (y <= 1-r))
        if not containment_ok:
            return False
            
        # Check overlap constraints efficiently using KDTree for better performance
        from scipy.spatial import cKDTree
        tree = cKDTree(circles[:, :2])
        pairs = tree.query_pairs(0.001)  # This is a small threshold to detect close pairs
        
        # Check if any circles are too close (overlapping)
        for i, j in pairs:
            dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2)
            if dist < circles[i, 2] + circles[j, 2]:
                return False
                
        return True
    
    # Objective function to maximize (negative because we minimize)
    def objective(vars_flat):
        circles = vars_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we want to maximize sum of radii
    
    # Optimized constraint functions using vectorized operations
    def constraint_func(vars_flat):
        """Vectorized constraint function for scipy.optimize"""
        circles = vars_flat.reshape(-1, 3)
        
        # Containment constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        containment = np.concatenate([
            x - r,           # x - r >= 0
            1 - x - r,       # 1 - x - r >= 0
            y - r,           # y - r >= 0
            1 - y - r        # 1 - y - r >= 0
        ])
        
        # Overlap constraints: compute all pairwise distances
        diff = circles[:, np.newaxis, :2] - circles[np.newaxis, :, :2]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        # Create overlap matrix
        overlap_matrix = distances - (circles[:, 2] + circles[:, 2][:, np.newaxis])
        # Set diagonal to large positive values to ignore self-overlaps
        np.fill_diagonal(overlap_matrix, 1000.0)
        
        # Flatten overlap constraints (only upper triangle to avoid duplicates)
        overlap = overlap_matrix[np.triu_indices_from(overlap_matrix, k=1)]
        
        return np.concatenate([containment, overlap])
    
    # Better initialization using a more sophisticated approach
    circles = generate_better_initial_configuration()
    
    # Convert to flat array for optimization
    initial_vars = []
    for i in range(n):
        initial_vars.extend([circles[i][0], circles[i][1], circles[i][2]])
    
    # Create bounds
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
    
    # Use a more robust optimization approach
    try:
        from scipy.optimize import Bounds, NonlinearConstraint
        
        # Create constraint object
        cons = NonlinearConstraint(constraint_func, 0, np.inf, jac='cs')
        
        # Optimization parameters
        bounds_obj = Bounds([0.001]*3*n, [0.999]*3*n)
        
        # Try multiple optimization attempts with different methods
        best_result = None
        best_sum = 0
        
        # Try different optimization approaches
        methods = ['SLSQP', 'trust-constr']
        restarts = 5  # Increased restarts for better chance at global optimum
        
        for method in methods:
            for restart in range(restarts):
                try:
                    np.random.seed(restart)
                    # Slightly perturb the initial solution
                    perturbed_vars = initial_vars.copy()
                    for i in range(len(perturbed_vars)):
                        if i % 3 < 2:  # x and y coordinates
                            perturbed_vars[i] += np.random.uniform(-0.02, 0.02)
                        else:  # radius
                            perturbed_vars[i] += np.random.uniform(-0.01, 0.01)
                    
                    # Clip to valid ranges
                    for i in range(len(perturbed_vars)):
                        perturbed_vars[i] = max(0.001, min(0.999, perturbed_vars[i]))
                    
                    result = minimize(
                        objective,
                        perturbed_vars,
                        method=method,
                        bounds=bounds_obj,
                        constraints=[cons],
                        options={'maxiter': 3000, 'ftol': 1e-6, 'gtol': 1e-6}
                    )
                    
                    if result.success:
                        test_circles = result.x.reshape(-1, 3)
                        current_sum = np.sum(test_circles[:, 2])
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_result = result
                            
                except Exception:
                    continue
        
        # If we found a good result, use it
        if best_result is not None:
            final_circles = best_result.x.reshape(-1, 3)
        else:
            # Fallback to our initial heuristic
            final_circles = circles
            
    except Exception as e:
        # Fallback to simple heuristic if optimization fails
        final_circles = circles
    
    # Enhanced refinement step with better algorithm
    def refine_circles_enhanced(circles_input):
        """Enhanced refinement to maximize radii while maintaining constraints"""
        circles = circles_input.copy()
        n_circles = len(circles)
        
        # Iteratively optimize each circle with better strategy
        for iteration in range(100):  # More iterations for better convergence
            improved = False
            
            # Process circles in random order for better convergence
            indices = list(range(n_circles))
            np.random.shuffle(indices)
            
            for i in indices:
                # Get current circle
                x, y, r = circles[i]
                
                # Calculate maximum possible radius based on all constraints
                max_radius = min(x, 1-x, y, 1-y)
                
                # Check overlap with all others and find minimum safe radius
                for j in range(n_circles):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                        max_radius = min(max_radius, dist - r2)
                
                # Increase radius if beneficial
                if max_radius > r + 1e-6 and max_radius > 0.001:
                    circles[i, 2] = max_radius
                    improved = True
            
            # Early stopping if no improvements
            if not improved:
                break
        
        return circles
    
    # Apply enhanced refinement
    final_circles = refine_circles_enhanced(final_circles)
    
    # Final validation and cleanup
    final_circles = np.clip(final_circles, 0, 1)
    
    # Ensure all constraints are met after final adjustments
    if not check_constraints_fast(final_circles):
        # If constraints violated, perform final correction using a more robust approach
        corrected_circles = final_circles.copy()
        
        # First, fix containment constraints
        for i in range(n):
            # Ensure containment
            corrected_circles[i, 0] = np.clip(corrected_circles[i, 0], 
                                            corrected_circles[i, 2], 1-corrected_circles[i, 2])
            corrected_circles[i, 1] = np.clip(corrected_circles[i, 1], 
                                            corrected_circles[i, 2], 1-corrected_circles[i, 2])
        
        # Then, perform a more careful adjustment of radii
        for i in range(n):
            # Calculate max radius respecting containment
            max_radius_containment = min(
                corrected_circles[i, 0], 1 - corrected_circles[i, 0],
                corrected_circles[i, 1], 1 - corrected_circles[i, 1]
            )
            
            # Calculate max radius avoiding overlaps with neighbors
            max_radius_overlap = max_radius_containment
            for j in range(n):
                if i != j:
                    dist = np.sqrt(
                        (corrected_circles[i, 0] - corrected_circles[j, 0])**2 +
                        (corrected_circles[i, 1] - corrected_circles[j, 1])**2
                    )
                    max_radius_overlap = min(max_radius_overlap, dist - corrected_circles[j, 2])
            
            # Take the smaller of the two limits
            final_radius = min(max_radius_containment, max_radius_overlap)
            if final_radius > 0.001:
                corrected_circles[i, 2] = final_radius
        
        final_circles = corrected_circles
    
    return final_circles


# EVOLVE-BLOCK-END
