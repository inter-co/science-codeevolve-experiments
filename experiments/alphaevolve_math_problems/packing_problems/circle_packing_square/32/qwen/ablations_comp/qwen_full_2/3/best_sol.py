# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def _compute_max_radius(x, y, circles, min_radius=0.001):
    """Compute maximum possible radius for a circle at (x,y) without overlapping existing circles."""
    if len(circles) == 0:
        # If no circles exist, we can place at maximum distance from boundaries
        return min(x, 1-x, y, 1-y)
    
    # Get centers and radii of existing circles
    centers = np.array([[c[0], c[1]] for c in circles])
    radii = np.array([c[2] for c in circles])
    
    # Compute distances to all existing centers
    distances = np.sqrt(np.sum((centers - [x, y])**2, axis=1))
    
    # Maximum radius is limited by the closest circle and boundaries
    if len(distances) > 0:
        min_dist_to_center = np.min(distances)
        # Max radius is (distance to closest center - existing radius) / 2
        # But we also need to respect boundaries
        max_radius_from_centers = (min_dist_to_center - np.min(radii)) / 2.0 if len(radii) > 0 else min_dist_to_center
        max_radius_from_boundaries = min(x, 1-x, y, 1-y)
        return max(min(max_radius_from_centers, max_radius_from_boundaries), min_radius)
    else:
        return min(x, 1-x, y, 1-y)

def _initialize_hexagonal_pattern(n):
    """Initialize circles using a hexagonal packing pattern with some randomness."""
    circles = np.zeros((n, 3))
    
    # Create a hexagonal lattice pattern with appropriate spacing
    # For 32 circles, we'll use a more strategic approach
    rows = 6
    cols = 6
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
                
            # Hexagonal offset for even rows
            x_offset = (row % 2) * 0.15
            x = 0.1 + col * 0.15 + x_offset
            y = 0.1 + row * 0.15
            
            # Add slight randomness to improve optimization convergence
            x += random.uniform(-0.01, 0.01)
            y += random.uniform(-0.01, 0.01)
            
            # Initial radius - start with values that allow room for growth
            r = 0.04 + random.uniform(-0.01, 0.01)
            
            # Ensure within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[idx] = [x, y, r]
            idx += 1
            
        if idx >= n:
            break
    
    return circles

def _initialize_grid_pattern(n):
    """Initialize circles using a grid pattern with some randomness."""
    circles = np.zeros((n, 3))
    
    # Create a 6x6 grid pattern but adjust for exactly 32 circles
    rows = 6
    cols = 6
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
                
            x = 0.1 + col * 0.15
            y = 0.1 + row * 0.15
            
            # Add slight randomness
            x += random.uniform(-0.01, 0.01)
            y += random.uniform(-0.01, 0.01)
            
            # Initial radius
            r = 0.03 + random.uniform(0, 0.02)
            
            # Ensure within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[idx] = [x, y, r]
            idx += 1
            
        if idx >= n:
            break
    
    return circles

def _evaluate_objective(circles_flat, n):
    """Evaluate objective function (negative sum of radii)."""
    # Reshape flat array back to circles format
    circles = circles_flat.reshape(-1, 3)
    return -np.sum(circles[:, 2])  # Negative because we minimize

def _constraint_func(params, n):
    """Constraint function for optimization."""
    # Convert flat params back to circles
    circles = params.reshape(-1, 3)
    
    # Constraint: containment
    containment = []
    for i in range(n):
        x, y, r = circles[i]
        containment.extend([
            x - r,  # x >= r
            (1-x) - r,  # 1-x >= r
            y - r,  # y >= r
            (1-y) - r   # 1-y >= r
        ])
    
    # Constraint: non-overlap
    overlap = []
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1-x2)**2 + (y1-y2)**2
            overlap_dist = np.sqrt(dist_sq)
            overlap.append(overlap_dist - (r1 + r2))  # Should be >= 0
    
    return np.concatenate([containment, overlap])

def _optimize_with_sqp(initial_circles, n):
    """Refine circle positions using Sequential Quadratic Programming."""
    # Flatten initial circles for optimization
    initial_flat = initial_circles.flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # Bounds for x and y (0.01 to 0.99 to leave margin for radii)
        bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.001, 0.49)])
    
    try:
        result = minimize(
            _evaluate_objective,
            initial_flat,
            args=(n,),
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda x: _constraint_func(x, n)},
            options={'maxiter': 2000, 'ftol': 1e-10, 'eps': 1e-10}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure valid radii
            optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], 0.001)
            return optimized_circles
    except Exception as e:
        pass
    
    # Return initial if optimization fails
    return initial_circles

def _local_search_improvement(circles, n):
    """Apply local search improvement to enhance solution quality."""
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Try different improvement strategies
    for iteration in range(200):  # Increased iterations for better improvement
        improved = False
        
        # Try adjusting each circle individually
        for i in range(n):
            # Store original
            original_x, original_y, original_r = best_circles[i]
            
            # Try various moves
            best_move = (original_x, original_y, original_r)
            best_move_radius_sum = best_sum
            
            # Try multiple variations for this circle
            for _ in range(30):  # More variations
                # Try moving in different directions
                dx = random.uniform(-0.015, 0.015)
                dy = random.uniform(-0.015, 0.015)
                dr = random.uniform(-0.008, 0.008)
                
                new_x = original_x + dx
                new_y = original_y + dy
                new_r = original_r + dr
                
                # Keep within bounds
                new_x = max(new_r, min(1-new_r, new_x))
                new_y = max(new_r, min(1-new_r, new_y))
                new_r = max(0.001, min(0.49, new_r))
                
                # Test this move
                test_circles = best_circles.copy()
                test_circles[i] = [new_x, new_y, new_r]
                
                # Check if it improves the solution
                if _check_constraints(test_circles, n):
                    test_radius_sum = np.sum(test_circles[:, 2])
                    if test_radius_sum > best_move_radius_sum:
                        best_move = (new_x, new_y, new_r)
                        best_move_radius_sum = test_radius_sum
                
            # Apply the best move
            best_circles[i] = list(best_move)
            
            # Update best solution
            current_radius_sum = np.sum(best_circles[:, 2])
            if current_radius_sum > best_sum:
                best_circles = best_circles.copy()
                best_sum = current_radius_sum
                improved = True
        
        # If no improvement, break early
        if not improved:
            break
    
    return best_circles

def _check_constraints(circles, n):
    """Check if all constraints are satisfied."""
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if r > x or r > (1-x) or r > y or r > (1-y):
            return False
    
    # Check non-overlap using efficient spatial data structure
    try:
        tree = cKDTree(circles[:, :2])
        pairs = tree.query_pairs(r=1e-8)  # Find very close points
        
        # More precise overlap checking
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < r1 + r2 - 1e-8:  # Small tolerance for numerical errors
                    return False
    except:
        # Fallback to brute force checking if KDTree fails
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < r1 + r2:
                    return False
    
    return True

def _multi_start_optimization(n, max_starts=15):
    """Run optimization from multiple starting points with different initialization strategies."""
    best_circles = None
    best_sum = -np.inf
    
    # Use only grid initialization for consistency and simplicity
    for start_idx in range(max_starts):
        # Initialize circles with grid pattern
        circles = _initialize_grid_pattern(n)
        
        # Refine using optimization
        refined_circles = _optimize_with_sqp(circles, n)
        
        # Evaluate
        radii_sum = np.sum(refined_circles[:, 2])
        
        if radii_sum > best_sum:
            best_sum = radii_sum
            best_circles = refined_circles.copy()
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies with mathematical optimization
    and local search improvements.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))
    
    # Use multi-start optimization approach with different initialization strategies
    circles = _multi_start_optimization(n, max_starts=10)
    
    # Final validation and cleanup
    if circles is not None:
        # Ensure final constraints are met
        for i in range(n):
            x, y, r = circles[i]
            # Make sure radius is valid
            if r <= 0:
                circles[i][2] = 0.01
            
            # Make sure circle is contained
            circles[i][0] = np.clip(x, r, 1-r)
            circles[i][1] = np.clip(y, r, 1-r)
    
    return circles


# EVOLVE-BLOCK-END
