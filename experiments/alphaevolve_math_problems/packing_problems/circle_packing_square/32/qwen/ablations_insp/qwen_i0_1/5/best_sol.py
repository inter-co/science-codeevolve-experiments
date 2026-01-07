# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import math
from sklearn.cluster import KMeans
import warnings
from itertools import combinations
from deap import base, creator, tools, algorithms
import random
from scipy.spatial import cKDTree
import numba

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining physics-based simulation, geometric initialization, and 
    advanced optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Use physics-based initialization with hexagonal packing
    initial_circles = initialize_with_hexagonal_packing(n)
    
    # Refine with physics simulation
    physics_refined = refine_with_physics_simulation(initial_circles)
    
    # Final optimization with constrained optimization
    final_solution = refine_with_local_optimization(physics_refined)
    
    return final_solution

def initialize_with_hexagonal_packing(n: int) -> np.ndarray:
    """Initialize circles using a hexagonal packing pattern as starting point."""
    # For 32 circles, arrange in approximately 6 rows and 5 columns with hexagonal offset
    rows = 6
    cols = 6
    if rows * cols < n:
        rows = 7
        cols = 5
    
    # Create hexagonal grid
    circles = []
    spacing_x = 0.8 / cols
    spacing_y = 0.8 / rows
    
    # Hexagonal offset for alternating rows
    hex_offset = spacing_x * 0.5
    
    for i in range(rows):
        y = 0.1 + i * spacing_y
        offset = hex_offset if i % 2 == 1 else 0
        for j in range(cols):
            x = 0.1 + j * spacing_x + offset
            if len(circles) < n:
                # Start with a reasonable initial radius
                r = min(spacing_x, spacing_y) * 0.3
                circles.append([x, y, r])
    
    # Fill remaining circles with random positions near the center
    center_x, center_y = 0.5, 0.5
    for i in range(len(circles), n):
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0, 0.2)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        r = np.random.uniform(0.02, 0.1)
        circles.append([x, y, r])
    
    return np.array(circles)

def refine_with_physics_simulation(initial_circles: np.ndarray, steps: int = 1000) -> np.ndarray:
    """Refine using a physics-based simulation with custom potentials."""
    circles = initial_circles.copy()
    
    # Physics parameters
    dt = 0.01
    repulsion_strength = 100.0
    attraction_strength = 1.0
    boundary_strength = 1000.0
    
    # Spatial indexing for efficient neighbor lookups
    tree = cKDTree(circles[:, :2])
    
    for step in range(steps):
        # Calculate forces
        forces = np.zeros_like(circles)
        
        # Repulsion forces between overlapping circles
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            # Find neighbors within 2*(r1+r2) distance
            indices = tree.query_ball_point([x1, y1], 2*(r1 + 0.1), p=2)
            
            for j in indices:
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0 and distance < (r1 + r2):
                        # Repulsion force
                        force_magnitude = repulsion_strength * (1 - distance/(r1 + r2))
                        forces[i, 0] -= force_magnitude * dx / distance
                        forces[i, 1] -= force_magnitude * dy / distance
                        
                        # Increase radius if space allows
                        if distance > 0.1 and r1 < 0.45:
                            # Try to increase radius slightly
                            max_r_increase = min(0.01, 0.45 - r1)
                            r1 += max_r_increase * 0.1
                        
        # Boundary forces (push away from edges)
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Force to stay within boundaries
            boundary_force = boundary_strength * 0.1
            
            if x - r < 0.01:
                forces[i, 0] += boundary_force * (0.01 - (x - r))
            elif x + r > 0.99:
                forces[i, 0] -= boundary_force * ((x + r) - 0.99)
                
            if y - r < 0.01:
                forces[i, 1] += boundary_force * (0.01 - (y - r))
            elif y + r > 0.99:
                forces[i, 1] -= boundary_force * ((y + r) - 0.99)
        
        # Update positions
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Apply forces with some damping
            damping = 0.9
            x += forces[i, 0] * dt * damping
            y += forces[i, 1] * dt * damping
            
            # Keep within bounds
            x = np.clip(x, r, 1-r)
            y = np.clip(y, r, 1-r)
            
            circles[i] = [x, y, r]
            
        # Update spatial index
        tree = cKDTree(circles[:, :2])
    
    return circles

def refine_with_local_optimization(initial_circles: np.ndarray) -> np.ndarray:
    """Refine the solution with local optimization techniques."""
    
    # Convert to optimization variables
    initial_vars = initial_circles.flatten()
    
    # Define constraints more efficiently using numba for performance
    @numba.jit(nopython=True)
    def compute_constraints_fast(circles):
        """Compute constraints efficiently."""
        n = len(circles)
        containment_constraints = np.empty(4*n)
        overlap_constraints = np.empty(n*(n-1)//2)
        
        # Containment constraints
        for i in range(n):
            x, y, r = circles[i]
            containment_constraints[4*i] = x - r  # x - r >= 0
            containment_constraints[4*i+1] = 1 - x - r  # 1 - x - r >= 0
            containment_constraints[4*i+2] = y - r  # y - r >= 0
            containment_constraints[4*i+3] = 1 - y - r  # 1 - y - r >= 0
        
        # Overlap constraints
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                overlap_constraints[idx] = distance - (r1 + r2)
                idx += 1
                
        return containment_constraints, overlap_constraints
    
    def constraint_containment(vars):
        """Constraint function for containment (all circles within unit square)."""
        circles = vars.reshape(-1, 3)
        n = len(circles)
        
        # Each circle must satisfy: r <= x <= 1-r and r <= y <= 1-r
        constraints = []
        
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([
                x - r,           # x - r >= 0
                1 - x - r,       # 1 - x - r >= 0
                y - r,           # y - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        
        return np.array(constraints)
    
    def constraint_overlaps(vars):
        """Constraint function for non-overlapping (distance >= sum of radii)."""
        circles = vars.reshape(-1, 3)
        n = len(circles)
        
        # For each pair of circles, ensure distance >= sum of radii
        constraints = []
        
        # Use efficient pairwise checking
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # We want: distance >= r1 + r2, which means: distance - (r1 + r2) >= 0
                constraints.append(distance - (r1 + r2))
        
        return np.array(constraints)
    
    def objective_function(vars):
        """Objective function to maximize sum of radii."""
        circles = vars.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we minimize
    
    # Define constraints
    containment_cons = {
        'type': 'ineq',
        'fun': constraint_containment
    }
    
    overlap_cons = {
        'type': 'ineq', 
        'fun': constraint_overlaps
    }
    
    # Bounds for variables: [x, y, r] for each circle
    bounds = []
    for i in range(len(initial_circles)):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.49)])  # x, y, r bounds
    
    # Try optimization with multiple methods
    methods_to_try = ['trust-constr', 'SLSQP']
    best_result = None
    best_sum = -np.inf
    
    for method in methods_to_try:
        try:
            result = minimize(
                objective_function,
                initial_vars,
                method=method,
                bounds=bounds,
                constraints=[containment_cons, overlap_cons],
                options={
                    'maxiter': 300, 
                    'ftol': 1e-6, 
                    'gtol': 1e-6,
                }
            )
            
            if result.success:
                current_sum = -objective_function(result.x)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            continue
    
    # If optimization succeeded, return the result, otherwise return the initial
    if best_result is not None:
        optimized_circles = best_result.x.reshape(-1, 3)
        return validate_and_refine(optimized_circles)
    else:
        return validate_and_refine(initial_circles)

def validate_and_refine(circles: np.ndarray) -> np.ndarray:
    """Validate constraints and perform final refinement."""
    # Ensure all circles are within bounds and have positive radii
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Keep circle within bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        r = max(r, 0.001)  # Ensure positive radius
        circles[i] = [x, y, r]
    
    # Perform iterative improvement with more aggressive approach
    improved = True
    iterations = 0
    max_iterations = 50
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to improve each circle individually
        for i in range(len(circles)):
            original = circles[i].copy()
            
            # Try to increase radius while maintaining constraints
            step_size = 0.001
            test_radius = min(original[2] + step_size, 0.49)
            
            # Check if we can increase radius without violating constraints
            valid = True
            for j in range(len(circles)):
                if i != j:
                    dist = np.sqrt((original[0] - circles[j][0])**2 + 
                                 (original[1] - circles[j][1])**2)
                    if dist < test_radius + circles[j][2]:
                        valid = False
                        break
            
            if valid:
                # Try to move the circle slightly to accommodate larger radius
                x_new = np.clip(original[0], test_radius, 1-test_radius)
                y_new = np.clip(original[1], test_radius, 1-test_radius)
                
                # Check if this movement still maintains constraints
                valid_move = True
                for j in range(len(circles)):
                    if i != j:
                        dist = np.sqrt((x_new - circles[j][0])**2 + 
                                     (y_new - circles[j][1])**2)
                        if dist < test_radius + circles[j][2]:
                            valid_move = False
                            break
                
                if valid_move:
                    circles[i] = [x_new, y_new, test_radius]
                    improved = True
                    continue
            
            # Try adjusting position while keeping same radius
            if not improved:
                # Try to move to a better location while preserving radius
                x_test = np.clip(original[0], original[2], 1-original[2])
                y_test = np.clip(original[1], original[2], 1-original[2])
                
                # Only update if there's a meaningful change
                if abs(x_test - original[0]) > 1e-6 or abs(y_test - original[1]) > 1e-6:
                    circles[i] = [x_test, y_test, original[2]]
                    improved = True
    
    # Final pass: try to slightly increase all radii if possible
    # This is a greedy improvement step
    for _ in range(30):  # Fewer iterations to save time
        improved_local = False
        for i in range(len(circles)):
            original = circles[i].copy()
            # Try to increase radius slightly
            test_radius = min(original[2] + 0.0005, 0.49)
            
            # Check if we can increase radius without conflicts
            valid = True
            for j in range(len(circles)):
                if i != j:
                    dist = np.sqrt((original[0] - circles[j][0])**2 + 
                                 (original[1] - circles[j][1])**2)
                    if dist < test_radius + circles[j][2]:
                        valid = False
                        break
            
            if valid:
                circles[i] = [original[0], original[1], test_radius]
                improved_local = True
        
        if not improved_local:
            break
    
    return circles


# EVOLVE-BLOCK-END
