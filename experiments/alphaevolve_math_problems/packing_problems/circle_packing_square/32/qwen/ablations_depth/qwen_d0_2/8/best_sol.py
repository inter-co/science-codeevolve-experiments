# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
import time
from itertools import combinations

# Global constants
N_CIRCLES = 32
BOUNDARY_MARGIN = 1e-6  # Small margin to prevent numerical issues
MAX_ITERATIONS = 1000

def generate_initial_config():
    """Generate a good initial configuration using a more sophisticated approach"""
    # Use a hexagonal packing pattern as starting point for better density
    circles = []
    
    # Hexagonal lattice parameters
    sqrt3 = np.sqrt(3)
    radius_estimate = 0.1  # Initial estimate
    spacing = 2 * radius_estimate
    
    # Create a hexagonal grid pattern
    rows = 6
    cols = 6
    
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= N_CIRCLES:
                break
            # Hexagonal offset
            x_offset = (j + 0.5 * (i % 2)) * spacing
            y_offset = i * spacing * sqrt3 / 2
            x = x_offset + np.random.uniform(-0.1*spacing, 0.1*spacing)
            y = y_offset + np.random.uniform(-0.1*spacing, 0.1*spacing)
            positions.append([x, y])
    
    # Ensure we have enough positions
    while len(positions) < N_CIRCLES:
        positions.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    
    # Initialize with calculated radii based on density
    initial_radii = []
    for i in range(N_CIRCLES):
        # Estimate radius based on area density
        estimated_area = 1.0 / N_CIRCLES
        estimated_radius = np.sqrt(estimated_area / np.pi)
        initial_radii.append(max(0.02, min(0.2, estimated_radius)))
    
    # Use a more sophisticated initialization based on the idea of maximizing 
    # density while maintaining minimal overlaps
    circles = []
    for i in range(min(N_CIRCLES, len(positions))):
        x, y = positions[i]
        # Clamp to valid range
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        circles.append([x, y, initial_radii[i]])
    
    return np.array(circles[:N_CIRCLES])

def objective_function(circles_flat):
    """Objective function to maximize sum of radii"""
    # Reshape flat array back to circles
    circles = circles_flat.reshape(-1, 3)
    # Return negative because we're minimizing
    return -np.sum(circles[:, 2])

def constraint_functions(circles_flat):
    """Constraint functions for optimization"""
    circles = circles_flat.reshape(-1, 3)
    n = len(circles)
    constraints = []
    
    # Boundary constraints (positive means feasible)
    for i in range(n):
        x, y, r = circles[i]
        constraints.append(x - r)  # x - r >= 0
        constraints.append(1 - x - r)  # 1 - x - r >= 0
        constraints.append(y - r)  # y - r >= 0
        constraints.append(1 - y - r)  # 1 - y - r >= 0
    
    # Non-overlap constraints (positive means feasible)
    for i, j in combinations(range(n), 2):
        dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
        overlap = dist - (circles[i][2] + circles[j][2])
        constraints.append(overlap)  # dist >= r_i + r_j
    
    return np.array(constraints)

def optimize_with_scipy(circles):
    """Use scipy optimization for better convergence with improved constraints"""
    # Flatten circles for scipy optimization
    circles_flat = circles.flatten()
    
    # Define bounds: [x, y, r] for each circle
    bounds = []
    for i in range(N_CIRCLES):
        # x bounds: [r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN]
        bounds.append((BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN))
        # y bounds: [r + BOUNDARY_MARGIN, 1 - r - BOUNDARY_MARGIN]  
        bounds.append((BOUNDARY_MARGIN, 1 - BOUNDARY_MARGIN))
        # r bounds: [BOUNDARY_MARGIN, 0.5]
        bounds.append((BOUNDARY_MARGIN, 0.5))
    
    # Define constraints
    def constraint_func(x):
        return constraint_functions(x)
    
    constraints = [{'type': 'ineq', 'fun': constraint_func}]
    
    # Optimization options with tighter tolerances
    options = {'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
    
    try:
        result = minimize(
            objective_function,
            circles_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options,
            tol=1e-8
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
    
    return circles

def validate_and_correct_configuration(circles):
    """Validate configuration and correct any violations"""
    # First check for boundary violations and fix them
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Correct boundary violations
        if x - r < BOUNDARY_MARGIN:
            x = r + BOUNDARY_MARGIN
        if x + r > 1 - BOUNDARY_MARGIN:
            x = 1 - r - BOUNDARY_MARGIN
        if y - r < BOUNDARY_MARGIN:
            y = r + BOUNDARY_MARGIN
        if y + r > 1 - BOUNDARY_MARGIN:
            y = 1 - r - BOUNDARY_MARGIN
        circles[i] = [x, y, r]
    
    # Now resolve overlaps through iterative improvement
    improved = True
    iteration = 0
    max_iterations = 50
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try to decrease radii of overlapping circles
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            for j in range(i+1, len(circles)):
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                # If circles overlap
                if dist < r1 + r2:
                    # Reduce both radii proportionally
                    overlap = (r1 + r2) - dist
                    reduction = min(overlap * 0.3, r1 * 0.1, r2 * 0.1)
                    
                    if r1 > reduction and r2 > reduction:
                        circles[i][2] -= reduction
                        circles[j][2] -= reduction
                        improved = True
                        
                        # Ensure radii remain positive
                        circles[i][2] = max(BOUNDARY_MARGIN, circles[i][2])
                        circles[j][2] = max(BOUNDARY_MARGIN, circles[j][2])
                        
                        # Reposition slightly to reduce overlap
                        if dist > 0:
                            dx = (x1 - x2) / dist * overlap * 0.05
                            dy = (y1 - y2) / dist * overlap * 0.05
                            circles[i][0] += dx
                            circles[i][1] += dy
                            circles[j][0] -= dx
                            circles[j][1] -= dy
                            
                            # Keep within bounds
                            circles[i][0] = max(BOUNDARY_MARGIN + circles[i][2], 
                                              min(1 - BOUNDARY_MARGIN - circles[i][2], circles[i][0]))
                            circles[i][1] = max(BOUNDARY_MARGIN + circles[i][2], 
                                              min(1 - BOUNDARY_MARGIN - circles[i][2], circles[i][1]))
                            circles[j][0] = max(BOUNDARY_MARGIN + circles[j][2], 
                                              min(1 - BOUNDARY_MARGIN - circles[j][2], circles[j][0]))
                            circles[j][1] = max(BOUNDARY_MARGIN + circles[j][2], 
                                              min(1 - BOUNDARY_MARGIN - circles[j][2], circles[j][1]))
    
    return circles

def advanced_local_search(circles):
    """Enhanced local search to improve solution quality"""
    # Try to locally improve by adjusting individual circles
    improved = True
    iterations = 0
    max_iterations = 30
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to increase radii of circles
        for i in range(len(circles)):
            x, y, r = circles[i]
            
            # Compute maximum possible radius for this circle
            max_radius = min(1 - x, x, 1 - y, y) - BOUNDARY_MARGIN
            
            # Check constraints with neighbors
            for j in range(len(circles)):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    max_radius_neighbor = dist - r2 - BOUNDARY_MARGIN
                    max_radius = min(max_radius, max_radius_neighbor)
            
            # Try to increase radius if beneficial
            if max_radius > r and max_radius - r > 0.0005:
                # Increase radius but don't go too far
                new_radius = min(max_radius, r + (max_radius - r) * 0.2)
                circles[i][2] = new_radius
                improved = True
    
    return circles

def gradient_based_refinement(circles):
    """Use gradient-based refinement to improve the solution"""
    # This is a simplified version that tries to move circles to reduce overlaps
    # without using complex gradients
    improved = True
    iterations = 0
    max_iterations = 20
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # For each circle, try to find a better position
        for i in range(len(circles)):
            x, y, r = circles[i]
            
            # Simple neighborhood search
            best_x, best_y = x, y
            best_radius = r
            best_score = -np.inf
            
            # Try small movements in 8 directions
            directions = [(0,0), (0.01,0), (-0.01,0), (0,0.01), (0,-0.01),
                         (0.01,0.01), (-0.01,0.01), (0.01,-0.01), (-0.01,-0.01)]
            
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                
                # Check if new position is valid
                if (new_x - r >= BOUNDARY_MARGIN and 
                    new_x + r <= 1 - BOUNDARY_MARGIN and
                    new_y - r >= BOUNDARY_MARGIN and 
                    new_y + r <= 1 - BOUNDARY_MARGIN):
                    
                    # Score based on how much overlap is reduced
                    score = 0
                    for j in range(len(circles)):
                        if i != j:
                            x2, y2, r2 = circles[j]
                            dist = np.sqrt((new_x - x2)**2 + (new_y - y2)**2)
                            overlap = max(0, r + r2 - dist)
                            score -= overlap
                    
                    if score > best_score:
                        best_score = score
                        best_x, best_y = new_x, new_y
            
            # Apply the best movement if it improves the situation
            if best_x != x or best_y != y:
                circles[i][0] = best_x
                circles[i][1] = best_y
                improved = True
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initialization, scipy optimization, and local search.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Step 1: Generate initial configuration
    circles = generate_initial_config()
    
    # Step 2: Validate initial configuration
    circles = validate_and_correct_configuration(circles)
    
    # Step 3: Optimize using scipy with SLSQP
    circles = optimize_with_scipy(circles)
    
    # Step 4: Validate after optimization
    circles = validate_and_correct_configuration(circles)
    
    # Step 5: Apply enhanced local search
    circles = advanced_local_search(circles)
    
    # Step 6: Apply gradient-based refinement
    circles = gradient_based_refinement(circles)
    
    # Step 7: Final validation and cleanup
    circles = validate_and_correct_configuration(circles)
    
    # Step 8: Additional fine-tuning
    circles = advanced_local_search(circles)
    
    # Ensure all circles are within bounds and have valid radii
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Clamp coordinates to valid range
        circles[i] = [
            max(BOUNDARY_MARGIN, min(1-BOUNDARY_MARGIN, x)),
            max(BOUNDARY_MARGIN, min(1-BOUNDARY_MARGIN, y)),
            max(0.001, min(0.5, r))
        ]
    
    return circles


# EVOLVE-BLOCK-END
