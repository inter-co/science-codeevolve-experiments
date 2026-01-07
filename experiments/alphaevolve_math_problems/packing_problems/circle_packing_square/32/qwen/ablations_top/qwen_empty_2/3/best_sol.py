# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-start approach with improved initialization and local refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    best_sum = 0
    best_circles = None
    
    # Improved initialization with multiple strategies
    def initialize_configurations():
        configs = []
        
        # Strategy 1: Grid with random perturbations (inspiration 1 approach)
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= n:
                    break
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                # Add small random perturbation
                x += np.random.uniform(-spacing/4, spacing/4)
                y += np.random.uniform(-spacing/4, spacing/4)
                # Ensure within bounds
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                positions.append([x, y])
        if len(positions) < n:
            # Fill remaining spots with random points
            for _ in range(n - len(positions)):
                positions.append([np.random.uniform(0.01, 0.99), np.random.uniform(0.01, 0.99)])
        configs.append(np.array(positions[:n]))
        
        # Strategy 2: More evenly distributed points (Voronoi-inspired)
        np.random.seed(42)
        positions = []
        for _ in range(n):
            positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
        configs.append(np.array(positions))
        
        # Strategy 3: Better structured grid with more careful spacing
        np.random.seed(123)
        positions = []
        # Create a more sophisticated grid layout
        for i in range(6):
            for j in range(6):
                if len(positions) >= n:
                    break
                # Add slight randomness to avoid perfect grids
                x = 0.1 + (i + 0.5 + np.random.normal(0, 0.1)) * 0.8 / 6
                y = 0.1 + (j + 0.5 + np.random.normal(0, 0.1)) * 0.8 / 6
                # Ensure within bounds
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                positions.append([x, y])
        if len(positions) < n:
            # Fill remaining spots with random points
            for _ in range(n - len(positions)):
                positions.append([np.random.uniform(0.01, 0.99), np.random.uniform(0.01, 0.99)])
        configs.append(np.array(positions[:n]))
        
        return configs
    
    # Constraint functions with better handling
    def get_constraints_and_bounds():
        # Bounds for each circle: (x, y, r) where 0 <= x,y <= 1 and 0 <= r <= min(x, y, 1-x, 1-y)
        bounds = []
        for i in range(n):
            # x, y, r bounds - tighter bounds to improve convergence
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Non-overlap constraints - more efficient computation
        def non_overlap_constraint(params):
            # Extract positions and radii
            positions = params.reshape(-1, 3)[:, :2]
            radii = params.reshape(-1, 3)[:, 2]
            
            # Compute pairwise distances efficiently
            distances = cdist(positions, positions)
            constraints = []
            
            # Only check upper triangle to avoid duplicates
            for i in range(n):
                for j in range(i+1, n):
                    # Constraint: distance >= radii[i] + radii[j]
                    dist = distances[i, j]
                    min_dist = radii[i] + radii[j]
                    constraints.append(dist - min_dist)
            
            return np.array(constraints)
        
        # Boundary constraints
        def boundary_constraint(params):
            positions = params.reshape(-1, 3)[:, :2]
            radii = params.reshape(-1, 3)[:, 2]
            
            constraints = []
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                # Circle must be contained in unit square
                constraints.extend([
                    x - r,           # x >= r
                    y - r,           # y >= r
                    1 - x - r,       # 1-x >= r
                    1 - y - r        # 1-y >= r
                ])
            
            return np.array(constraints)
        
        return bounds, non_overlap_constraint, boundary_constraint
    
    # Enhanced local refinement procedure
    def local_refinement(circles):
        # Make a copy to avoid modifying original
        refined = circles.copy()
        
        # More aggressive refinement with larger search space
        for iteration in range(200):  # Increased iterations
            improved = False
            for i in range(n):
                best_x, best_y, best_r = refined[i]
                best_obj = best_r
                
                # Try a wider range of adjustments
                dx_range = [-0.01, -0.005, -0.002, 0, 0.002, 0.005, 0.01]
                dy_range = [-0.01, -0.005, -0.002, 0, 0.002, 0.005, 0.01]
                dr_range = [-0.01, -0.005, -0.002, 0, 0.002, 0.005, 0.01]
                
                for dx in dx_range:
                    for dy in dy_range:
                        for dr in dr_range:
                            x, y, r = refined[i]
                            new_x = x + dx
                            new_y = y + dy
                            new_r = r + dr
                            
                            # Check bounds
                            if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
                                continue
                            if new_r <= 0:
                                continue
                                
                            # Check containment
                            if new_x - new_r < 0 or new_x + new_r > 1 or new_y - new_r < 0 or new_y + new_r > 1:
                                continue
                                
                            # Check overlap with others
                            valid = True
                            for j in range(n):
                                if i != j:
                                    x2, y2, r2 = refined[j]
                                    dist = np.sqrt((new_x - x2)**2 + (new_y - y2)**2)
                                    if dist < new_r + r2:
                                        valid = False
                                        break
                            if valid:
                                # Calculate objective improvement
                                obj = new_r
                                if obj > best_obj:
                                    best_obj = obj
                                    best_x, best_y, best_r = new_x, new_y, new_r
                                    improved = True
                                    
            if not improved:
                break
        
        return refined
    
    # Multi-start optimization with enhanced strategy
    initial_configs = initialize_configurations()
    
    for config_idx, initial_positions in enumerate(initial_configs):
        # Set up optimization variables: [x1, y1, r1, x2, y2, r2, ...]
        initial_params = []
        for i in range(n):
            x, y = initial_positions[i]
            # Initial radius: try to fit in a small area around the point, but with more variation
            r = min(x, y, 1-x, 1-y) * np.random.uniform(0.2, 0.4)
            initial_params.extend([x, y, r])
        
        # Get constraints
        bounds, non_overlap, boundary = get_constraints_and_bounds()
        
        # Define objective function (negative because we want to maximize)
        def objective(params):
            radii = params.reshape(-1, 3)[:, 2]
            return -np.sum(radii)
        
        # Define constraint dictionaries
        constraints = [
            {'type': 'ineq', 'fun': boundary},
            {'type': 'ineq', 'fun': non_overlap}
        ]
        
        # Optimize using SLSQP method with better parameters
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    initial_params,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6, 'disp': False}
                )
            
            if result.success:
                final_params = result.x
            else:
                # Fallback to initial configuration if optimization fails
                final_params = initial_params
                
        except Exception as e:
            # Fallback to initial configuration if optimization fails
            final_params = initial_params
        
        # Convert back to circles array
        circles = final_params.reshape(-1, 3)
        
        # Ensure valid ranges (just in case)
        for i in range(n):
            x, y, r = circles[i]
            # Clamp values to valid ranges
            x = np.clip(x, 0.001, 0.999)
            y = np.clip(y, 0.001, 0.999)
            r = np.clip(r, 0.001, min(x, y, 1-x, 1-y) - 0.001)
            circles[i] = [x, y, r]
        
        # Apply local refinement
        refined_circles = local_refinement(circles)
        
        # Check if this is better
        current_sum = np.sum(refined_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = refined_circles.copy()
    
    # If no good solution found, return the best we have
    if best_circles is None:
        # Fallback to simple grid initialization with refinement
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= n:
                    break
                x = (i + 1) * spacing
                y = (j + 1) * spacing
                # Add small random perturbation
                x += np.random.uniform(-spacing/4, spacing/4)
                y += np.random.uniform(-spacing/4, spacing/4)
                # Ensure within bounds
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                positions.append([x, y])
        if len(positions) < n:
            # Fill remaining spots with random points
            for _ in range(n - len(positions)):
                positions.append([np.random.uniform(0.01, 0.99), np.random.uniform(0.01, 0.99)])
        
        initial_params = []
        for i in range(n):
            x, y = positions[i]
            r = min(x, y, 1-x, 1-y) * 0.3
            initial_params.extend([x, y, r])
        
        # Simple refinement approach
        circles = np.array(initial_params).reshape(-1, 3)
        best_circles = local_refinement(circles)
    
    return best_circles


# EVOLVE-BLOCK-END
