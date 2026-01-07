# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import random
import time

# Global constants for the problem
N_CIRCLES = 32

def validate_circles(circles: np.ndarray) -> bool:
    """Check if all circles are within bounds and non-overlapping."""
    if circles.shape != (N_CIRCLES, 3):
        return False
    
    # Check containment constraints
    for i in range(N_CIRCLES):
        x, y, r = circles[i]
        if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
            return False
    
    # Check overlap constraints using vectorized computation for efficiency
    centers = circles[:, :2]
    radii = circles[:, 2]
    
    # Compute pairwise distances
    distances = cdist(centers, centers)
    radius_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Create mask for pairs that would overlap
    overlap_mask = distances < radius_sums
    # Set diagonal to False (circle doesn't overlap with itself)
    np.fill_diagonal(overlap_mask, False)
    
    return not np.any(overlap_mask)

def check_constraints_simple(circles):
    """Simple constraint checking - faster version for refinement"""
    for i, (x, y, r) in enumerate(circles):
        # Containment constraints
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
        
        # Overlap constraints with all previous circles
        for j in range(i):
            px, py, pr = circles[j]
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            if distance < r + pr:
                return False
    return True

def initialize_hexagonal_config():
    """Initialize with a hexagonal packing pattern for better starting configuration."""
    circles = np.zeros((N_CIRCLES, 3))
    
    # Create a hexagonal grid pattern - optimized for 32 circles
    rows = 6
    cols = 6
    spacing_x = 0.8 / (cols - 0.5) if cols > 1 else 0.8
    spacing_y = 0.8 / (rows - 1) if rows > 1 else 0.8
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= N_CIRCLES:
                break
            x = 0.1 + (j + 0.5*(i%2)) * spacing_x
            y = 0.1 + i * spacing_y
            r = min(spacing_x, spacing_y) * 0.3
            
            # Ensure it's within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= N_CIRCLES:
            break
    
    # Fill remaining positions with random placements near center for diversity
    for i in range(idx, N_CIRCLES):
        x = 0.5 + (np.random.random() - 0.5) * 0.4
        y = 0.5 + (np.random.random() - 0.5) * 0.4
        r = 0.02 + np.random.random() * 0.08
        # Ensure within bounds
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
        
    return circles

def initialize_grid_config():
    """Initialize with a grid-based configuration."""
    circles = np.zeros((N_CIRCLES, 3))
    
    # Generate points in a grid pattern with some jitter
    rows_cols = int(np.ceil(np.sqrt(N_CIRCLES)))
    spacing = 1.0 / (rows_cols + 1)
    
    idx = 0
    for i in range(rows_cols):
        for j in range(rows_cols):
            if idx >= N_CIRCLES:
                break
            x = (i + 1) * spacing + random.uniform(-spacing/4, spacing/4)
            y = (j + 1) * spacing + random.uniform(-spacing/4, spacing/4)
            # Initial radius - small enough to fit in square
            r = min(x, 1-x, y, 1-y) * 0.4
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= N_CIRCLES:
            break
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining mathematical programming and multiple initializations.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    # Multi-start optimization with different initial configurations
    best_result = None
    best_sum = -float('inf')
    
    # Try several different initial configurations
    initial_strategies = [
        initialize_hexagonal_config,  # Hexagonal packing (from inspiration)
        initialize_grid_config,       # Grid-based
        lambda: np.array([[random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.01, 0.1)] for _ in range(N_CIRCLES)])
    ]
    
    # Use a reduced number of iterations to meet time constraints while maintaining quality
    max_attempts = 8  # Reduced to ensure faster execution
    
    for attempt in range(max_attempts):
        # Select initialization strategy
        strategy_idx = attempt % len(initial_strategies)
        circles = initial_strategies[strategy_idx]()
        
        # Flatten for optimization
        initial_params = circles.flatten()
        
        # Optimization bounds
        bounds = []
        for i in range(N_CIRCLES):
            # x, y, r bounds - tighter bounds for better convergence
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Objective function to maximize (negative because we minimize)
        def objective(params):
            # Reshape params into circles array
            circles_local = params.reshape(-1, 3)
            return -np.sum(circles_local[:, 2])  # Negative because we want to maximize
        
        # Constraint functions - optimized for performance
        def constraint_containment(params):
            circles_local = params.reshape(-1, 3)
            # Return array of constraint violations (should be >= 0 for feasibility)
            violations = []
            for i in range(len(circles_local)):
                x, y, r = circles_local[i]
                violations.extend([
                    x - r,           # x >= r
                    1 - x - r,       # 1-x >= r  
                    y - r,           # y >= r
                    1 - y - r        # 1-y >= r
                ])
            return np.array(violations)
        
        def constraint_overlap(params):
            circles_local = params.reshape(-1, 3)
            # Return array of constraint violations (should be >= 0 for feasibility)
            violations = []
            for i in range(len(circles_local)):
                for j in range(i+1, len(circles_local)):
                    x1, y1, r1 = circles_local[i]
                    x2, y2, r2 = circles_local[j]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    # dist >= r1 + r2 (non-overlapping) -> dist - r1 - r2 >= 0
                    violations.append(dist - r1 - r2)
            return np.array(violations)
        
        try:
            # Use SLSQP method which handles constraints well
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': constraint_containment},
                    {'type': 'ineq', 'fun': constraint_overlap}
                ],
                options={'maxiter': 400, 'ftol': 1e-6}  # Reduced iterations for speed
            )
            
            if result.success:
                final_circles = result.x.reshape(-1, 3)
                total_radius = np.sum(final_circles[:, 2])
                if total_radius > best_sum:
                    best_sum = total_radius
                    best_result = final_circles.copy()
                    
        except Exception as e:
            continue
    
    # If we still have no good result, return a basic configuration
    if best_result is None:
        best_result = initialize_hexagonal_config()
    
    # Try differential evolution as a global optimization method (like INSPIRATION 2)
    try:
        bounds_de = []
        for i in range(N_CIRCLES):
            bounds_de.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        def de_objective(params):
            circles_flat = params.reshape((N_CIRCLES, 3))
            return -np.sum(circles_flat[:, 2])
        
        # Run differential evolution for global optimization - reduced iterations to stay fast
        result_de = differential_evolution(
            de_objective,
            bounds_de,
            maxiter=30,  # Reduced iterations to save time
            popsize=10,  # Smaller population for speed
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False,
            polish=True
        )
        
        if result_de.success:
            de_solution = result_de.x.reshape((N_CIRCLES, 3))
            if check_constraints_simple(de_solution):
                de_sum = np.sum(de_solution[:, 2])
                if de_sum > best_sum:
                    best_sum = de_sum
                    best_result = de_solution.copy()
                    
    except Exception as e:
        pass
    
    # Final refinement with L-BFGS-B method for better results
    try:
        # Use a more direct approach with better initial guess
        final_params = best_result.flatten()
        
        # Apply bounds to ensure valid ranges
        for i in range(N_CIRCLES):
            x, y, r = final_params[i*3:i*3+3]
            # Ensure proper bounds
            final_params[i*3] = max(0.001, min(0.999, x))
            final_params[i*3+1] = max(0.001, min(0.999, y))
            final_params[i*3+2] = max(0.001, min(0.499, r))
        
        # Final optimization with L-BFGS-B method - reduced iterations for time
        result = minimize(
            objective,
            final_params,
            method='L-BFGS-B',
            bounds=[(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * N_CIRCLES,
            options={'maxiter': 200, 'ftol': 1e-6}
        )
        
        if result.success:
            refined_circles = result.x.reshape(-1, 3)
            # Verify constraints and adjust if needed
            if validate_circles(refined_circles):
                return refined_circles
    except Exception:
        pass
    
    return best_result


# EVOLVE-BLOCK-END
