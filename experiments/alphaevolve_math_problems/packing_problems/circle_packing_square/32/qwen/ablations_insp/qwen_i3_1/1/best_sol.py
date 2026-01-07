# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
import math
import random

# Global constants
N_CIRCLES = 32
BENCHMARK = 2.937944526205518

def initialize_circles_enhanced_hexagonal() -> np.ndarray:
    """Enhanced hexagonal initialization with better packing density"""
    circles = np.zeros((N_CIRCLES, 3))
    
    # Try multiple hexagonal layouts to find the best starting configuration
    best_radius_sum = 0
    best_circles = None
    
    # Test several hexagonal arrangements
    layouts = [
        (6, 6),  # 6x6 grid (36 positions)
        (5, 7),  # 5x7 grid (35 positions) 
        (7, 5),  # 7x5 grid (35 positions)
        (4, 8),  # 4x8 grid (32 positions)
        (8, 4),  # 8x4 grid (32 positions),
    ]
    
    for rows, cols in layouts:
        if rows * cols >= N_CIRCLES:
            # Calculate spacing with better padding
            spacing_x = 0.9 / cols
            spacing_y = 0.9 / rows
            
            # Create hexagonal pattern
            circle_list = []
            for i in range(rows):
                for j in range(cols):
                    if len(circle_list) >= N_CIRCLES:
                        break
                    # Hexagonal offset for odd rows
                    x_offset = 0.0 if i % 2 == 0 else 0.5
                    x = 0.05 + (j + x_offset) * spacing_x
                    y = 0.05 + i * spacing_y
                    
                    # Ensure circle fits in unit square with margin
                    if x >= 0.01 and x <= 0.99 and y >= 0.01 and y <= 0.99:
                        # Estimate radius based on available space
                        radius = min(
                            spacing_x / 2.0, 
                            spacing_y / 2.0,
                            x, y, 1-x, 1-y
                        ) * 0.85  # Slightly higher ratio for better packing
                        
                        if radius > 0.005:
                            circle_list.append([x, y, radius])
                if len(circle_list) >= N_CIRCLES:
                    break
            
            if len(circle_list) >= N_CIRCLES:
                # Convert to numpy array and apply force relaxation
                positions = np.array([[c[0], c[1]] for c in circle_list[:N_CIRCLES]])
                radii = np.array([c[2] for c in circle_list[:N_CIRCLES]])
                
                # Apply enhanced force relaxation
                positions, radii = force_relaxation_enhanced(positions, radii, max_iterations=100)
                
                total_radius = np.sum(radii)
                
                if total_radius > best_radius_sum:
                    best_radius_sum = total_radius
                    best_circles = np.column_stack([positions, radii])
    
    # Fallback to simpler grid if needed
    if best_circles is None:
        rows = int(np.ceil(np.sqrt(N_CIRCLES)))
        cols = int(np.ceil(N_CIRCLES / rows))
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        radius = min(spacing_x, spacing_y) / 2.0
        
        circle_list = []
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= N_CIRCLES:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                if x - radius >= 0 and x + radius <= 1 and y - radius >= 0 and y + radius <= 1:
                    circle_list.append([x, y, radius])
                    count += 1
            if count >= N_CIRCLES:
                break
                
        positions = np.array([[c[0], c[1]] for c in circle_list[:N_CIRCLES]])
        radii = np.array([c[2] for c in circle_list[:N_CIRCLES]])
        best_circles = np.column_stack([positions, radii])
    
    return best_circles

def force_relaxation_enhanced(positions, radii, max_iterations=150):
    """Enhanced force relaxation with better convergence properties"""
    pos_copy = positions.copy()
    rad_copy = radii.copy()
    
    prev_total_radius = np.sum(rad_copy)
    
    # Use adaptive damping schedule for better convergence
    damping_schedule = np.linspace(0.15, 0.02, max_iterations)
    
    for iteration in range(max_iterations):
        # Calculate forces between circles
        forces = np.zeros_like(pos_copy)
        
        # Vectorized force computation
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                dx = pos_copy[i][0] - pos_copy[j][0]
                dy = pos_copy[i][1] - pos_copy[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > 0 and dist < rad_copy[i] + rad_copy[j]:
                    # Stronger repulsive force for close circles
                    force_magnitude = 1.0 / (dist * dist + 0.0001)
                    forces[i][0] += force_magnitude * dx / dist
                    forces[i][1] += force_magnitude * dy / dist
                    forces[j][0] -= force_magnitude * dx / dist
                    forces[j][1] -= force_magnitude * dy / dist
        
        # Apply forces with adaptive damping
        damping = damping_schedule[iteration]
        pos_copy += damping * forces
        
        # Boundary constraints and radius adjustments
        for i in range(N_CIRCLES):
            # Keep within bounds with margin
            pos_copy[i][0] = np.clip(pos_copy[i][0], rad_copy[i] + 0.001, 1 - rad_copy[i] - 0.001)
            pos_copy[i][1] = np.clip(pos_copy[i][1], rad_copy[i] + 0.001, 1 - rad_copy[i] - 0.001)
            
            # Adjust radius based on available space
            max_radius = min(
                pos_copy[i][0], 1-pos_copy[i][0],
                pos_copy[i][1], 1-pos_copy[i][1]
            )
            
            # Check overlap with all other circles to determine safe radius
            safe_radius = max_radius
            for j in range(N_CIRCLES):
                if i != j:
                    dx = pos_copy[i][0] - pos_copy[j][0]
                    dy = pos_copy[i][1] - pos_copy[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    safe_radius = min(safe_radius, dist - rad_copy[j])
            
            # Allow small improvements to radius with more aggressive adjustment
            rad_copy[i] = np.clip(safe_radius, 0.001, max_radius)
        
        # Early stopping criteria
        current_total_radius = np.sum(rad_copy)
        if abs(current_total_radius - prev_total_radius) < 1e-6:
            break
        prev_total_radius = current_total_radius
    
    return pos_copy, rad_copy

def check_constraints_fast(circles: np.ndarray) -> tuple:
    """Fast constraint checking using vectorized operations"""
    n = len(circles)
    
    # Check containment constraints
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    containment_ok = np.all((r <= x) & (x <= 1-r) & (r <= y) & (y <= 1-r))
    
    if not containment_ok:
        return False, 0
    
    # Check overlap constraints using vectorized computation
    # Compute distance matrix
    distances = cdist(circles[:, :2], circles[:, :2])
    # Create mask for upper triangle (avoid double counting)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    # Compute minimum distances for overlapping
    min_distances = np.add.outer(circles[:, 2], circles[:, 2])
    # Check if any distances are too small
    overlaps = distances[mask] < min_distances[mask]
    
    return not np.any(overlaps), np.sum(circles[:, 2])

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate the sum of all circle radii"""
    return np.sum(circles[:, 2])

def optimize_with_scipy_enhanced(initial_circles: np.ndarray) -> np.ndarray:
    """Enhanced scipy optimization with better parameter handling"""
    n = len(initial_circles)
    
    # Flatten initial circles for scipy
    initial_flat = []
    for i in range(n):
        initial_flat.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define bounds for optimization
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.499)])
    
    def objective(vars):
        # Reshape back to circles
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [vars[3*i], vars[3*i+1], vars[3*i+2]]
        
        # Objective: minimize negative sum of radii (maximize sum)
        return -np.sum(circles[:, 2])
    
    def constraint_func(vars):
        # Reshape back to circles
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [vars[3*i], vars[3*i+1], vars[3*i+2]]
        
        # Check containment constraints
        constraints = []
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([
                x - r - 1e-6,           # x - r >= 0
                1 - x - r - 1e-6,       # 1 - x - r >= 0
                y - r - 1e-6,           # y - r >= 0
                1 - y - r - 1e-6        # 1 - y - r >= 0
            ])
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2 + 1e-12
                constraints.append(min_dist_sq - dist_sq)
        
        return constraints
    
    # Try multiple optimization methods with better settings
    methods_to_try = ['trust-constr', 'L-BFGS-B']
    for method in methods_to_try:
        try:
            result = minimize(
                objective, 
                initial_flat, 
                method=method,
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                circles = np.zeros((n, 3))
                for i in range(n):
                    circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
                return circles
        except Exception:
            continue
    
    # Fallback to original if all methods fail
    return initial_circles

def optimize_with_local_improvement(initial_circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Improved local search with more aggressive radius increases"""
    circles = initial_circles.copy()
    
    for iteration in range(max_iter):
        improved = False
        # Try to improve each circle systematically
        for i in range(N_CIRCLES):
            old_x, old_y, old_r = circles[i]
            
            # Calculate maximum possible radius
            max_radius = min(old_x, 1-old_x, old_y, 1-old_y)
            
            # Try multiple step sizes for more thorough exploration
            step_sizes = [0.005, 0.01, 0.02, 0.03, 0.05]
            for step in step_sizes:
                test_r = old_r + step
                if test_r > max_radius:
                    continue
                    
                # Check if this change is valid
                valid = True
                for j in range(N_CIRCLES):
                    if i != j:
                        xj, yj, rj = circles[j]
                        dx = old_x - xj
                        dy = old_y - yj
                        dist_sq = dx*dx + dy*dy
                        min_dist_sq = (test_r + rj)**2
                        
                        # Check containment and overlap
                        if old_x - test_r < 0 or old_x + test_r > 1 or old_y - test_r < 0 or old_y + test_r > 1:
                            valid = False
                            break
                            
                        if dist_sq < min_dist_sq:
                            valid = False
                            break
                
                if valid:
                    circles[i] = [old_x, old_y, test_r]
                    improved = True
                    break
        
        if not improved:
            break
    
    return circles

def optimize_with_simulated_annealing_enhanced(initial_circles: np.ndarray, max_time: float = 55.0) -> np.ndarray:
    """Enhanced simulated annealing with better parameters and strategy"""
    start_time = time.time()
    
    # Start with current configuration
    current_circles = initial_circles.copy()
    current_score = calculate_radius_sum(current_circles)
    best_circles = current_circles.copy()
    best_score = current_score
    
    # Better SA parameters for this problem
    temp = 0.15  # Higher initial temperature for more exploration
    cooling_rate = 0.9999  # Very slow cooling for better convergence
    min_temp = 1e-9
    step_size = 0.01  # Larger step size for faster exploration
    
    iteration = 0
    while time.time() - start_time < max_time and temp > min_temp:
        # Try a neighbor solution
        new_circles = current_circles.copy()
        
        # Perturb one circle at random
        idx = random.randint(0, N_CIRCLES-1)
        
        # More aggressive perturbations
        new_circles[idx, 0] += random.uniform(-step_size, step_size)  # x
        new_circles[idx, 1] += random.uniform(-step_size, step_size)  # y
        new_circles[idx, 2] += random.uniform(-step_size*0.4, step_size*0.4)  # r
        
        # Keep within bounds
        new_circles[idx, 0] = np.clip(new_circles[idx, 0], 1e-6, 0.999)
        new_circles[idx, 1] = np.clip(new_circles[idx, 1], 1e-6, 0.999)
        new_circles[idx, 2] = np.clip(new_circles[idx, 2], 1e-6, 0.499)
        
        # Check constraints
        is_valid, new_score = check_constraints_fast(new_circles)
        
        if is_valid:
            delta = new_score - current_score
            
            # Accept or reject based on Metropolis criterion
            if delta > 0 or random.random() < math.exp(delta / temp):
                current_circles = new_circles
                current_score = new_score
                
                if new_score > best_score:
                    best_circles = new_circles
                    best_score = new_score
                    
        # Cool down
        temp *= cooling_rate
        iteration += 1
        
        # More frequent local improvements
        if iteration % 20 == 0:
            local_best = optimize_with_local_improvement(best_circles, 50)
            local_score = calculate_radius_sum(local_best)
            if local_score > best_score:
                best_circles = local_best
                best_score = local_score
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining enhanced hexagonal initialization, force relaxation, 
    and advanced optimization techniques.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Initialize with enhanced hexagonal pattern
    circles = initialize_circles_enhanced_hexagonal()
    
    # Apply enhanced force relaxation
    positions = circles[:, :2]
    radii = circles[:, 2]
    positions, radii = force_relaxation_enhanced(positions, radii, max_iterations=100)
    circles = np.column_stack([positions, radii])
    
    # Apply enhanced scipy optimization
    circles = optimize_with_scipy_enhanced(circles)
    
    # Apply local improvement
    circles = optimize_with_local_improvement(circles, 500)
    
    # Apply enhanced simulated annealing for further improvement
    circles = optimize_with_simulated_annealing_enhanced(circles, 45.0)
    
    # Final refinement with more aggressive local search
    circles = optimize_with_local_improvement(circles, 500)
    
    # Final validation and cleanup
    is_valid, final_score = check_constraints_fast(circles)
    if not is_valid:
        # If still invalid, use a more robust approach
        circles = initialize_circles_enhanced_hexagonal()
        circles = optimize_with_local_improvement(circles, 500)
        circles = optimize_with_simulated_annealing_enhanced(circles, 30.0)
    
    return circles


# EVOLVE-BLOCK-END
