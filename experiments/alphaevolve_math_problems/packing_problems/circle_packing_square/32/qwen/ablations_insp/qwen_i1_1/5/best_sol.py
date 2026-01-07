# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize
import math
from typing import Tuple
from scipy.spatial import cKDTree

def circle_packing32() -> np.ndarray:
    """
    Enhanced approach combining geometric initialization with advanced optimization
    techniques to maximize sum of radii for 32 non-overlapping circles in unit square.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize circles using a more informed hexagonal grid pattern
    circles = initialize_better_hexagonal_packing(n)
    
    # Multiple optimization strategies to find better solutions
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Strategy 1: Improved differential evolution with better bounds
    try:
        de_result = optimize_with_improved_differential_evolution(circles)
        de_sum = np.sum(de_result[:, 2])
        if de_sum > best_sum:
            best_sum = de_sum
            best_circles = de_result.copy()
    except Exception as e:
        pass
    
    # Strategy 2: Multi-start local optimization with better constraint handling
    try:
        local_result = optimize_locally_restarted_improved(best_circles)
        local_sum = np.sum(local_result[:, 2])
        if local_sum > best_sum:
            best_sum = local_sum
            best_circles = local_result.copy()
    except Exception as e:
        pass
    
    # Strategy 3: Enhanced iterative improvement with smarter neighborhood search
    try:
        iterative_result = enhanced_iterative_improvement(best_circles)
        iterative_sum = np.sum(iterative_result[:, 2])
        if iterative_sum > best_sum:
            best_sum = iterative_sum
            best_circles = iterative_result.copy()
    except Exception as e:
        pass
    
    # Strategy 4: Global search with hybrid approach
    try:
        global_result = hybrid_global_search(best_circles)
        global_sum = np.sum(global_result[:, 2])
        if global_sum > best_sum:
            best_sum = global_sum
            best_circles = global_result.copy()
    except Exception as e:
        pass
    
    return best_circles

def initialize_better_hexagonal_packing(n: int) -> np.ndarray:
    """Initialize circle positions using a better hexagonal packing pattern"""
    circles = np.zeros((n, 3))
    
    # Use a more optimal approach: 6 rows, 6 columns (36 positions)
    rows = 6
    cols = 6
    
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    # Calculate max radius based on spacing
    max_radius = min(spacing_x, spacing_y) * 0.4
    
    # Fill positions systematically in hexagonal pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Position with offset for hexagonal packing
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            
            # Offset every other row for better packing
            if i % 2 == 1:
                x += spacing_x / 2
                
            # Ensure we stay within bounds and have valid radius
            safe_x = np.clip(x, max_radius, 1 - max_radius)
            safe_y = np.clip(y, max_radius, 1 - max_radius)
            
            circles[idx] = [safe_x, safe_y, max_radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining positions with strategic placement
    np.random.seed(42)  # Fixed seed for reproducibility
    for i in range(idx, n):
        # Try to place in a region that allows larger radii
        attempts = 0
        while attempts < 50:
            # Prioritize center regions with better radius potential
            x = np.random.beta(2, 2) * 0.8 + 0.1  # Beta distribution centered around 0.5
            y = np.random.beta(2, 2) * 0.8 + 0.1
            
            # Calculate maximum possible radius at this location
            max_r = min(x, 1-x, y, 1-y) * 0.8
            
            # If we can place a reasonable sized circle here, do so
            if max_r > 0.01:
                circles[i] = [x, y, max_r]
                break
            attempts += 1
            
        # If we couldn't find a good spot, use a default small circle
        if attempts >= 50:
            circles[i] = [0.5, 0.5, 0.05]
    
    return circles

def optimize_with_improved_differential_evolution(initial_circles: np.ndarray) -> np.ndarray:
    """Use improved differential evolution for global optimization"""
    n = len(initial_circles)
    
    # Flatten initial configuration
    flat_init = initial_circles.flatten()
    
    # Define bounds for each parameter (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x bounds: [0.001, 0.999] 
        # y bounds: [0.001, 0.999]
        # r bounds: [0.001, 0.499]
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    def objective(params):
        # Reshape parameters back to circles
        circles = params.reshape((n, 3))
        # Minimize negative of sum of radii (maximize sum of radii)
        return -np.sum(circles[:, 2])
    
    def constraint_containment(params):
        circles = params.reshape((n, 3))
        constraints = []
        for i in range(n):
            x, y, r = circles[i]
            # r <= x, r <= 1-x, r <= y, r <= 1-y
            constraints.extend([
                x - r,           # x >= r
                1 - x - r,       # 1-x >= r
                y - r,           # y >= r
                1 - y - r        # 1-y >= r
            ])
        return np.array(constraints)
    
    def constraint_overlap(params):
        circles = params.reshape((n, 3))
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                # Distance >= r1 + r2 (we want to prevent overlap, so this should be >= 0)
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(dist - (r1 + r2))
        return np.array(constraints)
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': lambda p: constraint_containment(p)},
        {'type': 'ineq', 'fun': lambda p: constraint_overlap(p)}
    ]
    
    # Use differential evolution with better parameters
    result = differential_evolution(
        objective,
        bounds,
        args=(),
        strategy='best1bin',
        maxiter=150,
        popsize=20,
        tol=1e-7,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42,
        callback=None,
        disp=False,
        polish=True,
        init='latinhypercube'
    )
    
    if result.success:
        optimized_circles = result.x.reshape((n, 3))
        # Validate and fix any constraint violations
        if is_valid_configuration(optimized_circles):
            return optimized_circles
        else:
            # Try to repair the solution
            return repair_solution(optimized_circles)
    
    return initial_circles

def optimize_locally_restarted_improved(initial_circles: np.ndarray) -> np.ndarray:
    """Perform improved local optimization with multiple restarts"""
    n = len(initial_circles)
    best_circles = initial_circles.copy()
    best_sum = np.sum(initial_circles[:, 2])
    
    # Try multiple local optimizations with different starting points
    for restart in range(8):  # More restarts for better exploration
        # Perturb the initial solution slightly with better distribution
        perturbed = initial_circles.copy()
        np.random.seed(restart * 42)  # Different seed for each restart
        for i in range(n):
            # Larger random perturbations to escape local optima
            perturbed[i, 0] += np.random.normal(0, 0.02)  # Increased perturbation
            perturbed[i, 1] += np.random.normal(0, 0.02)
            # Keep within bounds
            perturbed[i, 0] = np.clip(perturbed[i, 0], 0.001, 0.999)
            perturbed[i, 1] = np.clip(perturbed[i, 1], 0.001, 0.999)
        
        # Run local optimization
        try:
            local_result = optimize_circles_local_improved(perturbed)
            local_sum = np.sum(local_result[:, 2])
            if local_sum > best_sum:
                best_sum = local_sum
                best_circles = local_result.copy()
        except Exception:
            continue
    
    return best_circles

def optimize_circles_local_improved(initial_circles: np.ndarray) -> np.ndarray:
    """Improved optimization using local optimization"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_params = initial_circles.flatten()
    
    def objective(params):
        # Reshape parameters back to circles
        circles = params.reshape((n, 3))
        # Minimize negative of sum of radii (maximize sum of radii)
        return -np.sum(circles[:, 2])
    
    def constraint_containment(params):
        circles = params.reshape((n, 3))
        constraints = []
        for i in range(n):
            x, y, r = circles[i]
            # r <= x, r <= 1-x, r <= y, r <= 1-y
            constraints.extend([
                x - r,           # x >= r
                1 - x - r,       # 1-x >= r
                y - r,           # y >= r
                1 - y - r        # 1-y >= r
            ])
        return np.array(constraints)
    
    def constraint_overlap(params):
        circles = params.reshape((n, 3))
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                # Distance >= r1 + r2 (we want to prevent overlap, so this should be >= 0)
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(dist - (r1 + r2))
        return np.array(constraints)
    
    # Define bounds for each parameter (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # x bounds: [0.001, 0.999] 
        # y bounds: [0.001, 0.999]
        # r bounds: [0.001, 0.499]
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': lambda p: constraint_containment(p)},
        {'type': 'ineq', 'fun': lambda p: constraint_overlap(p)}
    ]
    
    # Perform optimization with better solver settings
    try:
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-7, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape((n, 3))
            return optimized_circles
        else:
            # Fallback to initial configuration if optimization fails
            return initial_circles
            
    except Exception as e:
        # Return initial configuration if optimization fails
        return initial_circles

def enhanced_iterative_improvement(initial_circles: np.ndarray) -> np.ndarray:
    """Enhanced iterative improvement with smarter neighborhood search"""
    circles = initial_circles.copy()
    n = len(circles)
    max_iter = 1500  # More iterations for better convergence
    
    # Use k-d tree for faster neighbor queries
    tree = cKDTree(circles[:, :2])
    
    for iteration in range(max_iter):
        improved = False
        
        # Try to improve each circle individually
        for i in range(n):
            old_circle = circles[i].copy()
            
            # Try to increase radius
            max_radius = calculate_max_radius_for_circle_enhanced(circles, i, tree)
            current_radius = circles[i, 2]
            
            if max_radius > current_radius + 0.0001:
                # Try several increments to find best
                best_radius = current_radius
                best_sum = np.sum(circles[:, 2])
                
                # Test different radius increases with more aggressive steps
                test_increments = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03]
                for dr in test_increments:
                    test_radius = min(max_radius, current_radius + dr)
                    if test_radius > current_radius:
                        # Create test configuration
                        test_circles = circles.copy()
                        test_circles[i, 2] = test_radius
                        
                        if is_valid_configuration(test_circles):
                            test_sum = np.sum(test_circles[:, 2])
                            if test_sum > best_sum:
                                best_sum = test_sum
                                best_radius = test_radius
                
                if best_radius > current_radius:
                    circles[i, 2] = best_radius
                    improved = True
        
        # If no improvement, try position adjustments with more comprehensive search
        if not improved:
            for i in range(n):
                old_pos = circles[i, :2].copy()
                old_radius = circles[i, 2]
                
                # Try position adjustments with wider search space
                best_pos = old_pos.copy()
                best_radius = old_radius
                best_sum = np.sum(circles[:, 2])
                
                # Test more comprehensive adjustments
                adjustments = [(-0.01, -0.01), (-0.01, 0), (-0.01, 0.01),
                              (0, -0.01), (0, 0), (0, 0.01),
                              (0.01, -0.01), (0.01, 0), (0.01, 0.01),
                              (-0.005, -0.005), (-0.005, 0.005), (0.005, -0.005), (0.005, 0.005)]
                
                for dx, dy in adjustments:
                    new_pos = old_pos + np.array([dx, dy])
                    # Check bounds
                    if (new_pos[0] >= old_radius and 
                        new_pos[0] <= 1 - old_radius and
                        new_pos[1] >= old_radius and 
                        new_pos[1] <= 1 - old_radius):
                        
                        # Check if we can increase radius
                        test_circles = circles.copy()
                        test_circles[i, :2] = new_pos
                        max_r = calculate_max_radius_for_circle_enhanced(test_circles, i, tree)
                        test_circles[i, 2] = max_r
                        
                        if is_valid_configuration(test_circles):
                            test_sum = np.sum(test_circles[:, 2])
                            if test_sum > best_sum:
                                best_sum = test_sum
                                best_pos = new_pos.copy()
                                best_radius = max_r
                
                circles[i, :2] = best_pos
                circles[i, 2] = best_radius
        
        # Early stopping if no significant improvement
        if not improved and iteration > 200:
            break
    
    return circles

def hybrid_global_search(initial_circles: np.ndarray) -> np.ndarray:
    """Hybrid approach combining global and local search"""
    # Start with a greedy approach to get a good baseline
    circles = initial_circles.copy()
    
    # Apply a more aggressive iterative improvement first
    circles = enhanced_iterative_improvement(circles)
    
    # Then apply a few rounds of local optimization
    try:
        for _ in range(3):
            circles = optimize_circles_local_improved(circles)
    except:
        pass
    
    return circles

def calculate_max_radius_for_circle_enhanced(circles: np.ndarray, index: int, tree: cKDTree) -> float:
    """Enhanced calculation of maximum possible radius with k-d tree for efficiency"""
    # Calculate how much we can increase radius before violating constraints
    pos = circles[index, :2]
    radius = circles[index, 2]
    
    # Minimum distance to boundaries
    min_boundary_dist = min(pos[0], 1 - pos[0], pos[1], 1 - pos[1])
    
    # Find nearest neighbors using k-d tree for efficiency
    # Look for nearby circles that might constrain our radius
    nearby_indices = tree.query_ball_point(pos, 0.2)  # Search within 0.2 distance
    
    min_other_dist = float('inf')
    for i in nearby_indices:
        if i != index:
            dist = np.sqrt(
                (pos[0] - circles[i, 0])**2 +
                (pos[1] - circles[i, 1])**2
            )
            min_other_dist = min(min_other_dist, dist)
    
    # Maximum possible radius
    max_radius = min(min_boundary_dist, min_other_dist)
    return max(0.01, min(0.45, max_radius))

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if a configuration of circles is valid"""
    n = len(circles)
    
    # Check containment constraints
    if not (np.all(circles[:, 2] <= circles[:, 0]) and 
            np.all(circles[:, 0] <= 1 - circles[:, 2]) and
            np.all(circles[:, 2] <= circles[:, 1]) and
            np.all(circles[:, 1] <= 1 - circles[:, 2])):
        return False
    
    # Check overlap constraints using distance matrix for small n
    if n <= 50:  # Use direct computation for smaller arrays
        distances = cdist(circles[:, :2], circles[:, :2])
        np.fill_diagonal(distances, np.inf)
        
        for i in range(n):
            for j in range(i+1, n):
                if distances[i,j] < (circles[i, 2] + circles[j, 2]):
                    return False
    else:  # For larger arrays, use a more efficient approach
        # Use cKDTree for more efficient distance checks
        tree = cKDTree(circles[:, :2])
        for i in range(n):
            # Find neighbors within the sum of radii
            neighbors = tree.query_ball_point(circles[i, :2], circles[i, 2] + 0.001)
            for j in neighbors:
                if i != j:
                    dist = np.sqrt(
                        (circles[i, 0] - circles[j, 0])**2 +
                        (circles[i, 1] - circles[j, 1])**2
                    )
                    if dist < (circles[i, 2] + circles[j, 2]):
                        return False
    
    return True

def repair_solution(circles: np.ndarray) -> np.ndarray:
    """Attempt to repair an invalid solution by reducing radii"""
    repaired = circles.copy()
    n = len(repaired)
    
    # Reduce all radii until valid
    for _ in range(100):  # Limit iterations
        if is_valid_configuration(repaired):
            break
            
        # Reduce all radii proportionally
        for i in range(n):
            repaired[i, 2] *= 0.99
            
        # Ensure they don't go below minimum
        repaired[:, 2] = np.maximum(repaired[:, 2], 0.001)
    
    return repaired


# EVOLVE-BLOCK-END
