# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def _compute_circle_radius(x, y, circles, min_radius=0.001):
    """Compute maximum possible radius for a circle at (x,y) without overlapping existing circles."""
    if len(circles) > 0:
        centers = np.array([[c[0], c[1]] for c in circles])
        distances = cdist([[x, y]], centers)[0]
        # Minimum distance to any center minus the existing radius
        min_dist = np.min(distances) - circles[np.argmin(distances)][2]
        max_radius = min_dist / 2.0
        return max(max_radius, min_radius)
    else:
        # If no circles exist, we can place at maximum distance from boundaries
        return min(x, 1-x, y, 1-y)

def _initialize_grid_placement(n):
    """Initialize circle positions using a grid approach."""
    # Create a structured grid pattern
    grid_size = int(np.ceil(np.sqrt(n)))
    positions = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) < n:
                # Place points in a grid with slight randomization
                x = 0.1 + 0.8 * j / (grid_size - 1) if grid_size > 1 else 0.5
                y = 0.1 + 0.8 * i / (grid_size - 1) if grid_size > 1 else 0.5
                # Add small random perturbation to avoid perfect grid artifacts
                x += np.random.normal(0, 0.01)
                y += np.random.normal(0, 0.01)
                # Ensure within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                positions.append([x, y])
    
    return np.array(positions[:n])

def _initialize_random_placement(n):
    """Initialize circle positions using random placement."""
    return np.random.uniform(0.05, 0.95, (n, 2))

def _evaluate_objective(circles_flat, n):
    """Evaluate objective function (negative sum of radii)."""
    # Reshape flat array back to circles format
    circles = circles_flat.reshape(-1, 3)
    return -np.sum(circles[:, 2])  # Negative because we minimize

def _check_constraints(circles, n):
    """Check if all constraints are satisfied."""
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if r > x or r > (1-x) or r > y or r > (1-y):
            return False
    
    # Check non-overlap using vectorized computation for better performance
    if n > 1:
        circles_array = np.array(circles)
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        # Vectorized distance calculation
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        # Create overlap matrix (should be >= 0 for valid non-overlapping)
        overlap_matrix = distances - (radii[:, np.newaxis] + radii[np.newaxis, :])
        
        # Check if any overlaps exist (diagonal is zero, so ignore it)
        overlap_values = overlap_matrix[~np.eye(n, dtype=bool)]
        
        if np.any(overlap_values < 0):
            return False
    
    return True

def _optimize_circles_slsqp(initial_circles, n):
    """Refine circle positions using constrained optimization with SLSQP."""
    # Flatten initial circles for optimization
    initial_flat = initial_circles.flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        # Bounds for x and y (0.01 to 0.99 to leave margin for radii)
        bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.001, 0.49)])
    
    def constraint_func(params):
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
        
        # Constraint: non-overlap (vectorized for efficiency)
        if n > 1:
            positions = circles[:, :2]
            radii = circles[:, 2]
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            overlap_matrix = distances - (radii[:, np.newaxis] + radii[np.newaxis, :])
            overlap_values = overlap_matrix[~np.eye(n, dtype=bool)]
            overlap = overlap_values.tolist()
        else:
            overlap = []
        
        return np.concatenate([containment, overlap])
    
    # Use SLSQP optimizer with very tight tolerances for better precision
    try:
        result = minimize(
            _evaluate_objective,
            initial_flat,
            args=(n,),
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Ensure valid radii (they might go negative due to numerical issues)
            optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], 0.001)
            return optimized_circles
    except Exception as e:
        pass
    
    # Return initial if optimization fails
    return initial_circles

def _optimize_circles_simulated_annealing(initial_circles, n, max_iter=2500):
    """Refine circle positions using simulated annealing with improved parameters."""
    # Copy initial circles
    current_circles = initial_circles.copy()
    
    # Get initial objective value
    current_obj = -np.sum(current_circles[:, 2])
    
    # Parameters for simulated annealing - more aggressive
    temp = 1.0
    cooling_rate = 0.996  # Slightly faster cooling
    min_temp = 1e-12
    stagnation_count = 0
    max_stagnation = 150
    
    for iteration in range(max_iter):
        # Perturb one circle at random
        idx = random.randint(0, n-1)
        old_x, old_y, old_r = current_circles[idx]
        
        # Adaptive perturbation with better scaling
        adaptive_factor = temp * 0.07
        delta_x = random.uniform(-adaptive_factor, adaptive_factor)
        delta_y = random.uniform(-adaptive_factor, adaptive_factor)
        delta_r = random.uniform(-adaptive_factor * 0.4, adaptive_factor * 0.4)
        
        # Apply perturbation with bounds checking
        new_x = np.clip(old_x + delta_x, old_r, 1-old_r)
        new_y = np.clip(old_y + delta_y, old_r, 1-old_r)
        new_r = np.clip(old_r + delta_r, 0.001, 0.49)
        
        # Create candidate solution
        candidate_circles = current_circles.copy()
        candidate_circles[idx] = [new_x, new_y, new_r]
        
        # Check constraints
        if _check_constraints(candidate_circles, n):
            candidate_obj = -np.sum(candidate_circles[:, 2])
            
            # Accept or reject based on Metropolis criterion
            if candidate_obj > current_obj or \
               random.random() < np.exp((candidate_obj - current_obj) / temp):
                current_circles = candidate_circles
                current_obj = candidate_obj
                stagnation_count = 0  # Reset stagnation counter
            else:
                stagnation_count += 1
        else:
            stagnation_count += 1
            
        # Cool down temperature
        temp *= cooling_rate
        if temp < min_temp:
            break
            
        # Early stopping if no improvement for many iterations
        if stagnation_count > max_stagnation:
            break
    
    return current_circles

def _multi_start_optimization(n, max_starts=30):
    """Run optimization from multiple starting points with improved hybrid approach."""
    best_circles = None
    best_sum = -np.inf
    
    # Even more diverse initialization strategies
    init_strategies = [
        lambda x: _initialize_grid_placement(x),
        lambda x: _initialize_random_placement(x),
        lambda x: _initialize_grid_placement(x) * 0.9 + 0.05,
        lambda x: _initialize_grid_placement(x) * 0.8 + 0.1,
        lambda x: np.random.uniform(0.1, 0.9, (x, 2)),
    ]
    
    for start_idx in range(max_starts):
        # Choose initialization strategy
        strategy_idx = start_idx % len(init_strategies)
        initial_positions = init_strategies[strategy_idx](n)
        
        # Create initial circles with reasonable radii
        circles = np.zeros((n, 3))
        for i in range(n):
            x, y = initial_positions[i]
            circles[i] = [x, y, _compute_circle_radius(x, y, circles[:i])]
        
        # First optimize with SLSQP
        refined_circles = _optimize_circles_slsqp(circles, n)
        
        # Then refine with simulated annealing for better local search
        refined_circles = _optimize_circles_simulated_annealing(refined_circles, n, max_iter=2500)
        
        # Evaluate
        radii_sum = np.sum(refined_circles[:, 2])
        
        if radii_sum > best_sum:
            best_sum = radii_sum
            best_circles = refined_circles.copy()
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining grid initialization, SLSQP optimization, and simulated annealing.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))
    
    # Use multi-start optimization approach with improved methods
    circles = _multi_start_optimization(n, max_starts=30)
    
    # Final validation
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
