# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Global constants for the optimization
MAX_ITER = 2000
TOL = 1e-8

def compute_distances(circles):
    """Compute pairwise distances between circle centers"""
    centers = circles[:, :2]
    return cdist(centers, centers)

def objective(circles):
    """Objective function to maximize sum of radii"""
    return -np.sum(circles[:, 2])  # Negative because we minimize

def constraint_containment(circles):
    """Constraint function for containment"""
    n = len(circles)
    cons = []
    for i in range(n):
        x, y, r = circles[i]
        # r <= x <= 1-r and r <= y <= 1-r
        cons.append(x - r)  # x >= r
        cons.append(1 - x - r)  # x <= 1-r
        cons.append(y - r)  # y >= r
        cons.append(1 - y - r)  # y <= 1-r
    return np.array(cons)

def constraint_overlap(circles):
    """Constraint function for non-overlap"""
    n = len(circles)
    cons = []
    distances = compute_distances(circles)
    
    # Only check upper triangle to avoid double counting
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            r_i, r_j = circles[i, 2], circles[j, 2]
            # We want dist >= r_i + r_j, so we add the constraint: dist - r_i - r_j >= 0
            cons.append(dist - r_i - r_j)
    
    return np.array(cons)

def initialize_hexagonal_packing():
    """Initialize circles using a hexagonal packing pattern"""
    n = 32
    circles = np.zeros((n, 3))
    
    # Place in hexagonal lattice pattern
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Ensure enough space
    while rows * cols < n:
        rows += 1
    
    spacing_x = 0.9 / cols
    spacing_y = 0.9 / rows
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = 0.05 + j * spacing_x
            y = 0.05 + i * spacing_y
            
            # Offset every other row for hexagonal packing
            if i % 2 == 1:
                x += spacing_x / 2
            
            # Set initial radius based on spacing
            r = min(spacing_x, spacing_y) / 3
            
            # Ensure it fits in the unit square
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                r = min(x, 1-x, y, 1-y) * 0.7
                
            circles[idx] = [x, y, r]
            idx += 1
            
            if idx >= n:
                break
    
    return circles

def initialize_grid():
    """Initialize circles in a grid pattern"""
    n = 32
    circles = np.zeros((n, 3))
    
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            r = min(spacing_x, spacing_y) * 0.4
            circles[idx] = [x, y, r]
            idx += 1
    
    return circles

def initialize_random():
    """Initialize circles with random positions and small radii"""
    n = 32
    circles = np.zeros((n, 3))
    for i in range(n):
        # Random position with small radius
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = np.random.uniform(0.01, 0.1)
        circles[i] = [x, y, r]
    return circles

def optimize_circles(circles):
    """Refine the circle configuration using optimization"""
    n = len(circles)
    
    # Flatten the circles array for optimization
    initial_vars = np.column_stack([circles[:, 0], circles[:, 1], circles[:, 2]]).flatten()
    
    # Define bounds for variables: x, y, r for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r
    
    # Define constraints
    def constraint_func(vars):
        # Reconstruct circles from flattened vars
        reconstructed = np.zeros((n, 3))
        for i in range(n):
            reconstructed[i] = [vars[3*i], vars[3*i+1], vars[3*i+2]]
        
        # Check containment constraints
        containment = constraint_containment(reconstructed)
        
        # Check overlap constraints  
        overlap = constraint_overlap(reconstructed)
        
        return np.concatenate([containment, overlap])
    
    # Optimization using SLSQP with better parameters
    try:
        result = minimize(
            lambda x: objective(x.reshape(-1, 3)),
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': MAX_ITER, 'ftol': TOL, 'eps': 1e-7, 'disp': False}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Clip values to ensure they're within bounds
            for i in range(len(optimized_circles)):
                x, y, r = optimized_circles[i]
                optimized_circles[i] = [
                    np.clip(x, r, 1-r),
                    np.clip(y, r, 1-r),
                    np.clip(r, 0.001, 0.499)
                ]
            return optimized_circles
    except Exception as e:
        pass
    
    return circles

def aggressive_local_refinement(circles):
    """Apply aggressive local refinement to squeeze out maximum improvement"""
    n = len(circles)
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Try multiple rounds of local search with different strategies
    for round_num in range(5):
        # More aggressive perturbation scales
        perturbation_scale = 0.03 if round_num == 0 else 0.02 if round_num == 1 else \
                           0.015 if round_num == 2 else 0.01 if round_num == 3 else 0.005
        
        for _ in range(50):  # More iterations
            # Create perturbed version
            perturbed = best_circles.copy()
            
            # Perturb positions and radii more aggressively
            for i in range(n):
                if np.random.rand() < 0.8:  # 80% chance to perturb (more aggressive)
                    # Larger random change to position and radius
                    perturbed[i, 0] += np.random.normal(0, perturbation_scale)
                    perturbed[i, 1] += np.random.normal(0, perturbation_scale)
                    perturbed[i, 2] += np.random.normal(0, perturbation_scale * 0.4)
                    
                    # Keep within bounds
                    r = perturbed[i, 2]
                    perturbed[i, 0] = np.clip(perturbed[i, 0], r, 1-r)
                    perturbed[i, 1] = np.clip(perturbed[i, 1], r, 1-r)
                    perturbed[i, 2] = np.clip(r, 0.001, 0.499)
            
            # Optimize the perturbed version
            optimized = optimize_circles(perturbed)
            current_sum = np.sum(optimized[:, 2])
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Try multiple initialization strategies and optimization approaches
    best_circles = None
    best_sum = -np.inf
    
    # Strategy 1: Hexagonal packing with optimization
    circles1 = initialize_hexagonal_packing()
    optimized1 = optimize_circles(circles1)
    sum1 = np.sum(optimized1[:, 2])
    
    if sum1 > best_sum:
        best_sum = sum1
        best_circles = optimized1.copy()
    
    # Strategy 2: Random initialization with optimization  
    circles2 = initialize_random()
    optimized2 = optimize_circles(circles2)
    sum2 = np.sum(optimized2[:, 2])
    
    if sum2 > best_sum:
        best_sum = sum2
        best_circles = optimized2.copy()
    
    # Strategy 3: Grid initialization with optimization
    circles3 = initialize_grid()
    optimized3 = optimize_circles(circles3)
    sum3 = np.sum(optimized3[:, 2])
    
    if sum3 > best_sum:
        best_sum = sum3
        best_circles = optimized3.copy()
    
    # Apply aggressive local refinement to the best solution
    if best_circles is not None:
        best_circles = aggressive_local_refinement(best_circles)
    
    # If nothing worked, return default
    if best_circles is None:
        best_circles = initialize_hexagonal_packing()
        
    return best_circles


# EVOLVE-BLOCK-END
