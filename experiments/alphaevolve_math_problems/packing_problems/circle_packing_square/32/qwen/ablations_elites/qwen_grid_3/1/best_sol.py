# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def initialize_hexagonal_placement(n: int) -> np.ndarray:
    """Initialize circles using a hexagonal grid pattern for better initial placement"""
    circles = np.zeros((n, 3))
    
    # Create a hexagonal grid pattern with better spacing
    rows = int(np.ceil(np.sqrt(n * 1.2)))  # Slightly denser than pure sqrt
    cols = int(np.ceil(n / rows))
    
    # Adjust grid size to fit exactly n circles
    spacing_x = 0.9 / cols
    spacing_y = 0.9 / rows
    offset_x = 0.05
    offset_y = 0.05
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Hexagonal offset for even rows
            x_offset = 0.5 * (i % 2)
            x = offset_x + (j + x_offset) * spacing_x
            y = offset_y + i * spacing_y
            
            # Add more randomness to avoid perfect symmetry
            x += np.random.uniform(-spacing_x/8, spacing_x/8)
            y += np.random.uniform(-spacing_y/8, spacing_y/8)
            
            # Ensure circles stay within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius - start with a more informed value
            circles[idx] = [x, y, 0.025]
            idx += 1
        if idx >= n:
            break
            
    return circles

def evaluate_constraints(circles: np.ndarray) -> tuple:
    """Evaluate constraint satisfaction and return violation information"""
    n = len(circles)
    
    # Check containment constraints
    containment_violations = 0
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            containment_violations += 1
    
    # Check overlap constraints efficiently using KDTree
    overlap_violations = 0
    tree = cKDTree(circles[:, :2])
    # Use a reasonable threshold for neighbor search
    pairs = tree.query_pairs(0.0001)  # Very small threshold for tight checking
    for i, j in pairs:
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if distance < r1 + r2 - 1e-8:  # Small tolerance
            overlap_violations += 1
    
    return containment_violations, overlap_violations

def penalty_method_objective(circles: np.ndarray, penalty_weight: float = 1000.0) -> float:
    """Objective function with penalty terms for constraint violations"""
    # Objective: maximize sum of radii
    objective_value = -np.sum(circles[:, 2])
    
    # Penalty for containment violations
    penalty = 0
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0:
            penalty += penalty_weight * (x - r)**2
        if x + r > 1:
            penalty += penalty_weight * (x + r - 1)**2
        if y - r < 0:
            penalty += penalty_weight * (y - r)**2
        if y + r > 1:
            penalty += penalty_weight * (y + r - 1)**2
    
    # Penalty for overlap violations
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                penalty += penalty_weight * (r1 + r2 - distance)**2
    
    return objective_value + penalty

def constraint_containment(circles: np.ndarray) -> np.ndarray:
    """Ensure all circles are contained within the unit square"""
    n = len(circles)
    constraints = []
    
    for i in range(n):
        x, y, r = circles[i]
        # r <= x <= 1-r and r <= y <= 1-r
        constraints.extend([
            x - r,           # x >= r
            1 - x - r,       # x <= 1-r
            y - r,           # y >= r
            1 - y - r        # y <= 1-r
        ])
    
    return np.array(constraints)

def constraint_nonoverlap(circles: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """Ensure no two circles overlap"""
    n = len(circles)
    constraints = []
    
    # Calculate pairwise distances
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # For each pair of circles, ensure distance >= sum of radii
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance = np.sqrt(dx*dx + dy*dy)
            min_distance = radii[i] + radii[j]
            
            # Constraint: distance >= min_distance (so we want distance - min_distance >= 0)
            constraints.append(distance - min_distance)
    
    return np.array(constraints)

def optimize_with_scipy(circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Use scipy optimization with proper constraints"""
    n = len(circles)
    
    # Flatten circles array for optimization
    initial_flat = circles.flatten()
    
    def objective(flat_params):
        # Reshape back to circles
        temp_circles = flat_params.reshape(-1, 3)
        return penalty_method_objective(temp_circles)
    
    # Define bounds for optimization (radius must be positive, positions bounded)
    bounds = []
    for i in range(n):
        # x coordinate bounds (r <= x <= 1-r)
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.499))  # r (max radius is 0.5)
    
    # Set up constraints properly
    def containment_constraint(flat_params):
        temp_circles = flat_params.reshape(-1, 3)
        return constraint_containment(temp_circles)
    
    def nonoverlap_constraint(flat_params):
        temp_circles = flat_params.reshape(-1, 3)
        return constraint_nonoverlap(temp_circles)
    
    cons = [
        {'type': 'ineq', 'fun': containment_constraint},
        {'type': 'ineq', 'fun': nonoverlap_constraint}
    ]
    
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        # Fallback to simpler approach if optimization fails
        pass
    
    # If optimization fails, return original
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Strategy 1: Multiple initialization attempts with different patterns
    best_circles = None
    best_sum = 0
    
    # Try multiple initialization strategies
    for attempt in range(5):
        # Initialize with different patterns
        if attempt == 0:
            # Hexagonal pattern
            circles = initialize_hexagonal_placement(n)
        elif attempt == 1:
            # Grid pattern with some randomness
            circles = np.zeros((n, 3))
            rows = 6
            cols = 6
            spacing_x = 0.9 / cols
            spacing_y = 0.9 / rows
            for i in range(rows):
                for j in range(cols):
                    if i * cols + j >= n:
                        break
                    x = 0.05 + (j + 0.5) * spacing_x + np.random.uniform(-spacing_x/10, spacing_x/10)
                    y = 0.05 + (i + 0.5) * spacing_y + np.random.uniform(-spacing_y/10, spacing_y/10)
                    circles[i * cols + j] = [x, y, 0.02]
        else:
            # Random initialization with some structure
            circles = np.zeros((n, 3))
            for i in range(n):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                circles[i] = [x, y, 0.02]
        
        # Apply iterative improvement with local search
        current_sum = np.sum(circles[:, 2])
        improved = True
        iterations = 0
        
        while improved and iterations < 20:
            improved = False
            # Try to increase all radii more aggressively
            for i in range(n):
                x, y, r = circles[i]
                
                # Calculate maximum possible radius
                max_possible_r = min(x, 1-x, y, 1-y)
                
                # Check neighbors for potential radius increase
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dx = x - x2
                        dy = y - y2
                        distance = np.sqrt(dx*dx + dy*dy)
                        # Radius cannot exceed distance to neighbor minus their radius
                        max_possible_r = min(max_possible_r, distance - r2 - 1e-6)
                
                # Try to increase radius more aggressively
                if max_possible_r > r + 0.0005 and max_possible_r > 0.001:
                    # Use a more aggressive increment
                    new_r = min(max_possible_r, r + min(0.01, max_possible_r - r))
                    if new_r > r:
                        circles[i, 2] = new_r
                        improved = True
            
            if improved:
                current_sum = np.sum(circles[:, 2])
            iterations += 1
        
        # Apply scipy optimization to the best local result
        optimized_circles = optimize_with_scipy(circles, max_iter=1000)
        final_sum = np.sum(optimized_circles[:, 2])
        
        if final_sum > best_sum:
            best_sum = final_sum
            best_circles = optimized_circles.copy()
    
    # Strategy 2: Physics-inspired refinement if needed
    if best_circles is not None:
        # Apply additional physics-inspired refinement
        for _ in range(5):
            improved = False
            for i in range(n):
                x, y, r = best_circles[i]
                
                # Try to find better position that maximizes radius
                max_possible_r = min(x, 1-x, y, 1-y)
                
                # Check neighbors for constraints
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = best_circles[j]
                        dx = x - x2
                        dy = y - y2
                        distance = np.sqrt(dx*dx + dy*dy)
                        max_possible_r = min(max_possible_r, distance - r2 - 1e-6)
                
                # If we can increase radius significantly, do it
                if max_possible_r > r + 0.001 and max_possible_r > 0.001:
                    new_r = min(max_possible_r, r + 0.003)
                    best_circles[i, 2] = new_r
                    improved = True
            
            if not improved:
                break
    
    # Final validation and return best result
    if best_circles is None:
        # Fallback to simple initialization
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95), 0.02]
        return circles
    
    return best_circles


# EVOLVE-BLOCK-END
