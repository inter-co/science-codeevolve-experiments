# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with gradient-based optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 32
    circles = np.zeros((n, 3))
    
    # Phase 1: Generate multiple initial configurations and pick the best
    initial_circles = generate_multiple_initial_placements(n)
    
    # Phase 2: Use a more focused optimization approach
    optimized_circles = optimize_with_constraints(initial_circles)
    
    # Phase 3: Very aggressive refinement with multiple strategies
    final_circles = very_aggressive_refinement(optimized_circles)
    
    return final_circles

def generate_multiple_initial_placements(n):
    """Generate several initial configurations and pick the best one"""
    best_circles = None
    best_sum = 0
    
    for attempt in range(5):
        # Create different initial placements
        circles = np.zeros((n, 3))
        
        # Different pattern for each attempt
        if attempt == 0:
            # Regular hexagonal pattern
            rows = 6
            cols = 6
        elif attempt == 1:
            # Slightly different pattern
            rows = 5
            cols = 7
        elif attempt == 2:
            # Another variation
            rows = 7
            cols = 5
        else:
            # Randomized version
            rows = 6
            cols = 6
            
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                x = 0.1 + 0.8 * j / (cols - 1) if cols > 1 else 0.5
                y = 0.1 + 0.8 * i / (rows - 1) if rows > 1 else 0.5
                
                # Apply hexagonal offset for odd rows
                if i % 2 == 1:
                    x += 0.4 / cols
                
                # Add larger random perturbation for variety
                x += random.uniform(-0.02, 0.02)
                y += random.uniform(-0.02, 0.02)
                
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                positions.append([x, y])
        
        # Initialize with equal small radii
        for i in range(n):
            circles[i] = [positions[i][0], positions[i][1], 0.05]
        
        # Calculate initial sum of radii
        current_sum = np.sum(circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles.copy()
    
    return best_circles if best_circles is not None else generate_hexagonal_initial_placement(n)

def generate_hexagonal_initial_placement(n):
    """Generate initial configuration using hexagonal close packing pattern"""
    circles = np.zeros((n, 3))
    
    # Create a hexagonal grid pattern
    # For 32 circles, we'll use a 6x6 grid with some adjustment
    rows = 6
    cols = 6
    
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            # Hexagonal offset pattern
            x = 0.1 + 0.8 * j / (cols - 1) if cols > 1 else 0.5
            y = 0.1 + 0.8 * i / (rows - 1) if rows > 1 else 0.5
            
            # Apply hexagonal offset for odd rows
            if i % 2 == 1:
                x += 0.4 / cols
            
            # Add small random perturbation
            x += random.uniform(-0.01, 0.01)
            y += random.uniform(-0.01, 0.01)
            
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            positions.append([x, y])
    
    # Initialize with equal small radii
    for i in range(n):
        circles[i] = [positions[i][0], positions[i][1], 0.05]
    
    return circles

def optimize_with_constraints(initial_circles):
    """Use scipy optimization with proper constraint handling"""
    n = len(initial_circles)
    
    # Flatten initial circles for optimization
    initial_params = initial_circles.flatten()
    
    # Objective function (negative since we want to maximize sum of radii)
    def objective(params):
        circles = params.reshape((n, 3))
        return -np.sum(circles[:, 2])  # Negative because we want to maximize
    
    # Constraint function for optimization
    def constraint_func(params):
        circles = params.reshape((n, 3))
        constraints = []
        
        # Containment constraints (each circle must fit inside unit square)
        for i in range(n):
            x, y, r = circles[i]
            # Circle center must be at least radius away from edges
            constraints.append(x - r)  # x >= r
            constraints.append(1 - x - r)  # 1-x >= r
            constraints.append(y - r)  # y >= r
            constraints.append(1 - y - r)  # 1-y >= r
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                # Distance between centers should be >= sum of radii
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                constraints.append(dist_sq - (r1 + r2)**2)
                
        return np.array(constraints)
    
    # Bounds for parameters: x,y in [0.05, 0.95], r in [0.01, 0.45]
    bounds = []
    for i in range(n):
        bounds.extend([(0.05, 0.95), (0.05, 0.95), (0.01, 0.45)])
    
    # Perform optimization with multiple attempts
    best_result = None
    best_value = float('inf')
    
    for attempt in range(4):  # More attempts
        try:
            # Try with different solver settings
            if attempt == 0:
                result = minimize(objective, initial_params, method='SLSQP', 
                                 bounds=bounds, constraints={'type': 'ineq', 'fun': constraint_func},
                                 options={'maxiter': 2000, 'ftol': 1e-8, 'eps': 1e-8})
            elif attempt == 1:
                result = minimize(objective, initial_params, method='L-BFGS-B', 
                                 bounds=bounds,
                                 options={'maxiter': 2000, 'ftol': 1e-8})
            elif attempt == 2:
                result = minimize(objective, initial_params, method='TNC', 
                                 bounds=bounds,
                                 options={'maxiter': 2000, 'ftol': 1e-8})
            else:
                # Try with a different approach - trust-constr
                result = minimize(objective, initial_params, method='trust-constr', 
                                 bounds=bounds, constraints={'type': 'ineq', 'fun': constraint_func},
                                 options={'maxiter': 2000, 'ftol': 1e-8})
            
            if result.success:
                # Evaluate the actual value
                current_value = -result.fun  # Convert back to maximization
                if current_value < best_value:
                    best_value = current_value
                    best_result = result
        except Exception:
            continue
    
    if best_result is not None and best_result.success:
        optimized_params = best_result.x
        return optimized_params.reshape((n, 3))
    
    # If optimization fails, return initial configuration
    return initial_circles

def very_aggressive_refinement(circles):
    """Apply extremely aggressive refinement to squeeze out every bit of improvement"""
    n = len(circles)
    
    # Multiple passes with different strategies
    for pass_num in range(10):  # More passes
        improved = False
        
        # Strategy 1: Global improvement with careful checking
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            x, y, r = circles[i]
            # Calculate maximum possible radius
            max_r = min(x, 1-x, y, 1-y)
            
            # Check overlap with all other circles
            new_r = r
            for j in range(n):
                if i != j:
                    xj, yj, rj = circles[j]
                    dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                    # If overlapping, reduce radius significantly
                    if dist < (r + rj):
                        new_r = min(new_r, dist - rj - 0.0005)
                        if new_r < 0.01:
                            new_r = 0.01
                        break
            
            # Aggressive attempt to increase radius
            if new_r > r:
                # Try multiple increments
                increments = [0.015, 0.01, 0.008, 0.005]
                for incr in increments:
                    test_r = min(max_r, new_r + incr)
                    if test_r <= max_r:
                        # Check if increasing radius causes overlap
                        valid = True
                        for j in range(n):
                            if i != j:
                                xj, yj, rj = circles[j]
                                dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                                if dist < (test_r + rj):
                                    valid = False
                                    break
                        
                        if valid and test_r > r:
                            new_r = test_r
                            break
            
            # Apply the change if beneficial
            if new_r > r and new_r <= max_r:
                circles[i] = [x, y, new_r]
                improved = True
        
        # Strategy 2: Simultaneous optimization of nearby groups
        if pass_num % 2 == 0 and pass_num > 0:
            # Group optimization - find clusters and optimize them together
            tree = cKDTree(circles[:, :2])
            
            # Process clusters of 4-6 circles
            for i in range(0, n, 4):
                if i + 4 <= n:
                    # Get cluster around circle i
                    cluster_indices = tree.query_ball_point(circles[i, :2], 0.2)
                    if len(cluster_indices) >= 3:
                        # Try to optimize this cluster
                        cluster_circles = circles[cluster_indices]
                        # Simple heuristic: increase all radii a little bit if possible
                        for idx in cluster_indices:
                            x, y, r = circles[idx]
                            max_r = min(x, 1-x, y, 1-y)
                            
                            # Check overlap with others in cluster
                            new_r = r
                            for j in cluster_indices:
                                if j != idx:
                                    xj, yj, rj = circles[j]
                                    dist = np.sqrt((x - xj)**2 + (y - yj)**2)
                                    if dist < (r + rj):
                                        new_r = min(new_r, dist - rj - 0.001)
                                        if new_r < 0.01:
                                            new_r = 0.01
                                        break
                            
                            if new_r > r and new_r <= max_r:
                                circles[idx] = [x, y, new_r]
                                improved = True
        
        # Stop early if no significant improvement
        if not improved:
            break
    
    # Final validation and cleanup
    for i in range(n):
        x, y, r = circles[i]
        # Ensure containment
        r = min(r, x, 1-x, y, 1-y)
        circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
