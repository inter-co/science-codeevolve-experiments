# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Even better initialization using a more sophisticated approach
    def generate_advanced_initialization(n_points):
        # Create a more sophisticated initialization that mimics known good packings
        # Start with a grid pattern, then apply a more complex perturbation
        
        # Create a grid that's close to optimal for 32 circles
        rows = 6  # 6 rows seems appropriate for 32 circles
        cols = 6  # 6 columns gives 36 positions, more than enough
        
        points = []
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Generate grid points
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n_points:
                    break
                # Create a staggered pattern for better packing
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Apply more significant but controlled perturbations
                x += (np.random.random() - 0.5) * 0.15 * spacing_x
                y += (np.random.random() - 0.5) * 0.15 * spacing_y
                
                # Keep within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                
                points.append([x, y])
        
        return np.array(points[:n_points])
    
    # Generate initial points
    initial_points = generate_advanced_initialization(n)
    
    # Precompute neighbor relationships for efficiency
    def compute_neighbors(points, max_neighbors=20):
        """Compute nearest neighbors for each point to reduce constraint checking"""
        neighbors = {}
        for i in range(len(points)):
            distances = []
            for j in range(len(points)):
                if i != j:
                    dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                    distances.append((dist, j))
            
            # Sort by distance and keep only closest neighbors
            distances.sort()
            neighbors[i] = [j for _, j in distances[:min(max_neighbors, len(distances))]]
        return neighbors
    
    neighbors_dict = compute_neighbors(initial_points)
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(params):
        # params contains [x1, y1, r1, x2, y2, r2, ...]
        total_radius = 0
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            if r <= 0:
                return 1e10  # Invalid configuration penalty
            total_radius += r
        return -total_radius  # Negative because we're minimizing
    
    # Define constraints with better numerical handling
    def boundary_constraint(params):
        # Ensure all circles fit within the unit square
        constraints = []
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            if r <= 0:
                return 1e10
            # Circle must be fully inside the unit square
            constraints.extend([
                x - r,  # x >= r
                1 - x - r,  # 1 - x >= r
                y - r,  # y >= r
                1 - y - r   # 1 - y >= r
            ])
        return np.array(constraints)
    
    def overlap_constraint(params):
        # Ensure no overlapping circles - optimized version
        constraints = []
        for i in range(n):
            x_i, y_i, r_i = params[3*i], params[3*i+1], params[3*i+2]
            if r_i <= 0:
                return 1e10
                
            # Check against nearby neighbors only for efficiency
            neighbor_indices = neighbors_dict[i]
            for j in neighbor_indices:
                if i >= j:  # Avoid double counting
                    continue
                x_j, y_j, r_j = params[3*j], params[3*j+1], params[3*j+2]
                if r_j <= 0:
                    continue
                    
                dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                # Distance between centers must be >= sum of radii
                # Add small epsilon to handle numerical precision issues
                overlap = dist - (r_i + r_j)
                constraints.append(overlap)  # Should be >= 0
        return np.array(constraints)
    
    # Initialize parameters with even better starting values
    initial_params = []
    for i in range(n):
        x, y = initial_points[i]
        
        # Estimate initial radius based on proximity to neighbors
        distances = []
        neighbor_indices = neighbors_dict[i]
        for j in neighbor_indices:
            dist = np.sqrt((x - initial_points[j][0])**2 + (y - initial_points[j][1])**2)
            distances.append(dist)
        
        # Set initial radius with more careful consideration
        if len(distances) > 0:
            min_dist = min(distances)
            # Allow radius to be up to 1/5 of the minimum neighbor distance to start
            r = min(0.2, min_dist/5.0, 0.4)
        else:
            r = 0.15
            
        # Ensure radius is positive and reasonable
        r = max(0.005, min(0.3, r))
        initial_params.extend([x, y, r])
    
    # Create bounds: [x_min, x_max, y_min, y_max, r_min, r_max]
    bounds = []
    for i in range(n):
        bounds.extend([(0.005, 0.995), (0.005, 0.995), (0.005, 0.3)])  # x, y, r bounds (tighter bounds)
    
    # Define constraints for scipy
    cons = [
        {'type': 'ineq', 'fun': lambda p: boundary_constraint(p)},
        {'type': 'ineq', 'fun': lambda p: overlap_constraint(p)}
    ]
    
    # Try multiple optimization approaches for better results
    best_result = None
    best_sum = 0
    
    # First attempt with SLSQP - use high precision settings
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1500, 'ftol': 1e-9, 'eps': 1e-9, 'iprint': -1},
            callback=lambda x: None
        )
        
        if result.success:
            # Validate the result
            total_radius = -objective(result.x)
            if total_radius > best_sum:
                best_sum = total_radius
                best_result = result
    except Exception as e:
        pass
    
    # Second attempt with L-BFGS-B if SLSQP fails or doesn't give good results
    if best_result is None or best_sum < 2.9:
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-9},
                callback=lambda x: None
            )
            
            if result.success:
                total_radius = -objective(result.x)
                if total_radius > best_sum:
                    best_sum = total_radius
                    best_result = result
        except Exception as e:
            pass
    
    # Third attempt with a different algorithm if still no success
    if best_result is None or best_sum < 2.9:
        try:
            # Try trust-constr which often works well for constrained problems
            result = minimize(
                objective,
                initial_params,
                method='trust-constr',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9},
                callback=lambda x: None
            )
            
            if result.success:
                total_radius = -objective(result.x)
                if total_radius > best_sum:
                    best_sum = total_radius
                    best_result = result
        except Exception as e:
            pass
    
    # If still no good result, try a different initialization approach
    if best_result is None or best_sum < 2.9:
        try:
            # Generate a completely different initialization
            np.random.seed(42)  # Fixed seed for consistency
            initial_params_alt = []
            for i in range(n):
                # Place points randomly but with better distribution
                x = 0.1 + 0.8 * np.random.random()  # Avoid edges
                y = 0.1 + 0.8 * np.random.random()  # Avoid edges
                r = 0.05 + 0.15 * np.random.random()  # Reasonable starting radius
                initial_params_alt.extend([x, y, r])
            
            result = minimize(
                objective,
                initial_params_alt,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-8},
                callback=lambda x: None
            )
            
            if result.success:
                total_radius = -objective(result.x)
                if total_radius > best_sum:
                    best_sum = total_radius
                    best_result = result
        except Exception as e:
            pass
    
    # Fallback to initial parameters if optimization failed
    if best_result is None:
        final_params = initial_params
    else:
        final_params = best_result.x
    
    # Extract final results
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
    
    # Final validation and cleanup
    for i in range(n):
        x, y, r = circles[i]
        # Make sure circles stay within bounds
        if r < 0.005:
            r = 0.005
        if x < 0.005:
            x = 0.005
        if x > 0.995:
            x = 0.995
        if y < 0.005:
            y = 0.005
        if y > 0.995:
            y = 0.995
        circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
