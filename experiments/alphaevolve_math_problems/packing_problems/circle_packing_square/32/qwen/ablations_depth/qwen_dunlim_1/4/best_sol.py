# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a robust optimization approach with carefully tuned parameters and improved constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Generate high-quality initial configuration using a proven method
    def generate_high_quality_initialization():
        # Use a method inspired by the "best known" solutions for circle packing
        # Create a configuration that's already close to optimal
        
        # Create a hexagonal-like grid with some randomness
        points = []
        
        # Grid size that works well for 32 circles
        rows = 6
        cols = 6
        
        # Calculate spacing
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Fill grid with points
        for i in range(rows):
            for j in range(cols):
                if len(points) < n:
                    x = (j + 1) * spacing_x
                    # Offset odd rows for hexagonal packing
                    if i % 2 == 1:
                        x += spacing_x * 0.5
                    y = (i + 1) * spacing_y
                    points.append([x, y])
        
        # Make sure we have exactly n points
        points_array = np.array(points[:n])
        
        # Add significant randomness to break symmetry
        noise = np.random.normal(0, 0.015, points_array.shape)
        points_array += noise
        points_array = np.clip(points_array, 0, 1)
        
        # Initialize radii using a more sophisticated approach
        radii = []
        for i in range(n):
            # Find minimum distance to any other point
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dx = points_array[i, 0] - points_array[j, 0]
                    dy = points_array[i, 1] - points_array[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
            
            # Set radius to a safe fraction of the minimum distance
            radius = min(min_dist/4.0, 0.12)
            radii.append(max(radius, 0.005))
        
        # Construct initial guess
        initial_guess = []
        for i in range(n):
            initial_guess.extend([points_array[i, 0], points_array[i, 1], radii[i]])
        
        return initial_guess
    
    # Generate high quality initial configuration
    initial_guess = generate_high_quality_initialization()
    
    # Define constraint functions with optimized performance
    def containment_constraints(x):
        """Ensure all circles are within the unit square"""
        constraints = []
        for i in range(n):
            xi = x[3*i]
            yi = x[3*i+1]
            ri = x[3*i+2]
            # Add very small buffer to prevent numerical issues
            constraints.append(xi - ri - 1e-15)  # xi - ri >= 0
            constraints.append(1 - xi - ri - 1e-15)  # 1 - xi - ri >= 0
            constraints.append(yi - ri - 1e-15)  # yi - ri >= 0
            constraints.append(1 - yi - ri - 1e-15)  # 1 - yi - ri >= 0
        return np.array(constraints)
    
    def non_overlap_constraints(x):
        """Ensure no overlap between circles with numerical tolerance"""
        constraints = []
        positions = x.reshape(-1, 3)[:, :2]
        radii = x.reshape(-1, 3)[:, 2]
        
        # Vectorized constraint checking for better performance
        # Use explicit loops to avoid memory issues with large arrays
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                r_sum = radii[i] + radii[j]
                
                # Only compute sqrt when necessary and avoid numerical errors
                if dist_sq > 0:
                    dist = np.sqrt(dist_sq)
                    # We want: dist >= r_sum, so: dist - r_sum >= 0
                    constraints.append(dist - r_sum - 1e-15)
        return np.array(constraints)
    
    # Define objective function (negative because we want to maximize)
    def objective(x):
        return -np.sum(x[2::3])  # Sum of all radii (every third element starting from index 2)
    
    # Set up bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(1e-15, 1-1e-15), (1e-15, 1-1e-15), (1e-15, 0.5)])
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: containment_constraints(x)},
        {'type': 'ineq', 'fun': lambda x: non_overlap_constraints(x)}
    ]
    
    # Multi-start optimization with maximum effort
    best_result = None
    best_sum = -float('inf')
    
    # Run with many different starting points to ensure we don't miss good solutions
    for start_run in range(8):
        # Create variation in initial guess
        current_initial = initial_guess.copy()
        
        if start_run > 0:
            # Apply different perturbations for each run
            for i in range(n):
                if i * 3 + 0 < len(current_initial):
                    # Apply different perturbations based on run number
                    perturbation_factor = 0.01 + (start_run * 0.002)
                    current_initial[i*3] += np.random.normal(0, perturbation_factor)
                    current_initial[i*3+1] += np.random.normal(0, perturbation_factor)
                    # Keep within bounds
                    current_initial[i*3] = np.clip(current_initial[i*3], 1e-15, 1-1e-15)
                    current_initial[i*3+1] = np.clip(current_initial[i*3+1], 1e-15, 1-1e-15)
        
        # Try multiple solvers and configurations
        solver_configs = [
            ('SLSQP', {'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}),
            ('trust-constr', {'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8})
        ]
        
        for solver, options in solver_configs:
            try:
                result = minimize(objective, current_initial, method=solver, bounds=bounds, constraints=cons, 
                                options=options)
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception:
                continue
    
    # If optimization failed, fall back to initial guess
    if best_result is None:
        final_solution = initial_guess
    else:
        final_solution = best_result.x
    
    # Final aggressive refinement
    try:
        # Try one final optimization with even tighter tolerances
        final_refinement = minimize(objective, final_solution, method='SLSQP', bounds=bounds, constraints=cons,
                                  options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
        
        if final_refinement.success:
            final_solution = final_refinement.x
    except Exception:
        pass
    
    # Final extraction
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i][0] = final_solution[3*i]
        circles[i][1] = final_solution[3*i+1]
        circles[i][2] = final_solution[3*i+2]
    
    return circles


# EVOLVE-BLOCK-END
