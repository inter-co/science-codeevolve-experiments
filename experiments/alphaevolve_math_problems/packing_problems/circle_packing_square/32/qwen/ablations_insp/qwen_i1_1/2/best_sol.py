# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: geometric initialization + local optimization + evolutionary refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    N_CIRCLES = 32
    MAX_TIME = 55  # Leave some buffer for cleanup
    
    # Phase 1: Geometric initialization with hexagonal packing
    def initialize_hexagonal():
        circles = np.zeros((N_CIRCLES, 3))
        
        # Arrange in roughly a hexagonal pattern
        rows = 6
        cols = 6
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N_CIRCLES:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                # Initial radius - small enough to fit in the grid cell
                r = min(spacing_x, spacing_y) * 0.3
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= N_CIRCLES:
                break
        
        # Fill remaining positions with random placements
        for i in range(idx, N_CIRCLES):
            circles[i] = [
                random.uniform(0.05, 0.95),
                random.uniform(0.05, 0.95),
                random.uniform(0.01, 0.1)
            ]
        
        return circles
    
    # Phase 2: Constraint validation and penalty calculation
    def validate_and_evaluate(circles):
        """Calculate fitness with proper penalties for constraint violations"""
        total_radius = 0
        penalty = 0
        
        # Calculate total radius
        for x, y, r in circles:
            total_radius += r
            
        # Check containment penalties
        for i, (x, y, r) in enumerate(circles):
            # Check containment: circle must be fully within [0,1]x[0,1]
            containment_penalty = 0
            containment_penalty += max(0, r - x)  # left boundary
            containment_penalty += max(0, r - (1 - x))  # right boundary
            containment_penalty += max(0, r - y)  # bottom boundary
            containment_penalty += max(0, r - (1 - y))  # top boundary
            
            penalty += containment_penalty * 1000
        
        # Check overlap penalties
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                # Calculate distance between centers
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                # Overlap penalty if circles intersect
                if dist < r1 + r2:
                    overlap = (r1 + r2) - dist
                    penalty += overlap * 10000  # Heavy penalty for overlap
        
        return total_radius - penalty
    
    # Phase 3: Local optimization using scipy
    def optimize_local(circles):
        """Refine the configuration using local optimization"""
        def objective(params):
            # Reshape parameters back into circles
            circles_flat = params.reshape(-1, 3)
            # Calculate negative of sum of radii (since we want to maximize)
            total_radius = sum(circles_flat[:, 2])
            return -total_radius
        
        def constraint_func(params):
            circles_flat = params.reshape(-1, 3)
            
            # Containment constraints (each circle must be fully within unit square)
            containment_constraints = []
            for i, (x, y, r) in enumerate(circles_flat):
                containment_constraints.extend([
                    x - r,  # x - r >= 0
                    1 - x - r,  # 1 - x - r >= 0
                    y - r,  # y - r >= 0
                    1 - y - r  # 1 - y - r >= 0
                ])
            
            # Overlap constraints (distance between centers >= sum of radii)
            overlap_constraints = []
            for i in range(len(circles_flat)):
                for j in range(i + 1, len(circles_flat)):
                    x1, y1, r1 = circles_flat[i]
                    x2, y2, r2 = circles_flat[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    # We want dist >= r1 + r2, so constraint is dist - r1 - r2 >= 0
                    overlap_constraints.append(dist - r1 - r2)
            
            return np.array(containment_constraints + overlap_constraints)
        
        # Flatten current circles for optimization
        initial_params = circles.flatten()
        
        # Set up bounds for optimization
        bounds = []
        for i in range(len(initial_params)):
            if i % 3 == 0 or i % 3 == 1:  # x, y coordinates
                bounds.append((0.001, 0.999))
            else:  # radius
                bounds.append((0.001, 0.4))
        
        try:
            # Use SLSQP optimizer which handles constraints well
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                # Ensure valid ranges
                for i in range(len(optimized_circles)):
                    x, y, r = optimized_circles[i]
                    optimized_circles[i] = [
                        max(0.001, min(0.999, x)),
                        max(0.001, min(0.999, y)),
                        max(0.001, min(0.4, r))
                    ]
                return optimized_circles
        except Exception:
            pass
        
        return circles
    
    # Phase 4: Multi-start local search with improved initialization
    start_time = time.time()
    
    # Start with geometric initialization
    circles = initialize_hexagonal()
    
    # Evaluate initial solution
    best_fitness = validate_and_evaluate(circles)
    best_circles = circles.copy()
    
    # Run local optimization on initial solution
    circles = optimize_local(circles)
    current_fitness = validate_and_evaluate(circles)
    
    if current_fitness > best_fitness:
        best_fitness = current_fitness
        best_circles = circles.copy()
    
    # Additional refinement with multiple restarts
    for restart in range(10):
        if time.time() - start_time > MAX_TIME:
            break
            
        # Perturb current solution slightly
        perturbed = best_circles.copy()
        for i in range(len(perturbed)):
            x, y, r = perturbed[i]
            # Add small random perturbation
            perturbed[i] = [
                max(0.001, min(0.999, x + random.uniform(-0.02, 0.02))),
                max(0.001, min(0.999, y + random.uniform(-0.02, 0.02))),
                max(0.001, min(0.4, r + random.uniform(-0.01, 0.01)))
            ]
        
        # Optimize this perturbed version
        optimized = optimize_local(perturbed)
        fitness = validate_and_evaluate(optimized)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_circles = optimized.copy()
    
    return best_circles


# EVOLVE-BLOCK-END
