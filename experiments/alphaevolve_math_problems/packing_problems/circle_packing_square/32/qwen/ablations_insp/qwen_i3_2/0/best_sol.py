# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random

# Global constants for the problem
N_CIRCLES = 32
UNIT_SQUARE_SIZE = 1.0
BENCHMARK = 2.937944526205518

def create_improved_hexagonal_initial():
    """Create initial configuration using enhanced hexagonal packing pattern."""
    # Try different arrangements to find a better starting point
    best_circles = None
    best_sum = 0
    
    # Try multiple configurations
    for attempt in range(3):
        circles = []
        
        # Try different grid sizes
        if attempt == 0:
            rows, cols = 6, 6
        elif attempt == 1:
            rows, cols = 5, 7
        else:
            rows, cols = 7, 5
            
        spacing_x = 1.0 / cols
        spacing_y = spacing_x * np.sqrt(3) / 2
        
        # Adjust spacing slightly to allow for better packing
        max_radius = min(spacing_x, spacing_y) / 2 * 0.95  # Slightly smaller to allow adjustments
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= N_CIRCLES:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Offset odd rows
                if i % 2 == 1:
                    x += spacing_x / 2
                    
                # Ensure we're within bounds
                if x - max_radius >= 0 and x + max_radius <= 1 and y - max_radius >= 0 and y + max_radius <= 1:
                    circles.append([x, y, max_radius])
                else:
                    # Place near boundaries with smaller radii
                    x = max(max_radius, min(1 - max_radius, x))
                    y = max(max_radius, min(1 - max_radius, y))
                    circles.append([x, y, max_radius])
                    
        # Fill remaining slots with strategic placements
        while len(circles) < N_CIRCLES:
            # Place some near corners and edges for better utilization
            corner_positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
            edge_positions = [(0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)]
            
            if len(circles) < N_CIRCLES and len(corner_positions) > 0:
                pos = corner_positions.pop(0)
                circles.append([pos[0], pos[1], 0.02])
            elif len(circles) < N_CIRCLES and len(edge_positions) > 0:
                pos = edge_positions.pop(0)
                circles.append([pos[0], pos[1], 0.02])
            else:
                circles.append([0.5, 0.5, 0.01])
                
        circles_array = np.array(circles[:N_CIRCLES])
        current_sum = np.sum(circles_array[:, 2])
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = circles_array
    
    return best_circles if best_circles is not None else create_simple_initial()

def create_simple_initial():
    """Fallback simple initial configuration."""
    circles = []
    # Create a more balanced initial configuration
    for i in range(N_CIRCLES):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = np.random.uniform(0.02, 0.15)
        circles.append([x, y, r])
    return np.array(circles)

def optimize_with_scipy(initial_circles):
    """Optimize using scipy with improved constraints handling."""
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(params):
        # Reshape params into circles array
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because minimize
    
    # Define constraints more efficiently
    def constraint_containment(i):
        def c(params):
            circles = params.reshape(-1, 3)
            x, y, r = circles[i]
            # Return positive values when constraint satisfied (>= 0)
            # Use a small tolerance to avoid numerical issues
            return min(x - r + 1e-8, 1 - x - r + 1e-8, y - r + 1e-8, 1 - y - r + 1e-8)
        return c
    
    def constraint_nonoverlap(i, j):
        def c(params):
            circles = params.reshape(-1, 3)
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            # Return positive when constraint satisfied (distance >= radii sum)
            # Use a small tolerance to avoid numerical issues
            return dist_sq - (ri + rj)**2 + 1e-8
        return c
    
    # Build constraints list
    constraints = []
    
    # Add containment constraints
    for i in range(N_CIRCLES):
        constraints.append({'type': 'ineq', 'fun': constraint_containment(i)})
    
    # Add non-overlap constraints
    # Reduce number of constraints by using a more intelligent approach
    for i in range(N_CIRCLES):
        for j in range(i+1, N_CIRCLES):
            constraints.append({'type': 'ineq', 'fun': constraint_nonoverlap(i, j)})
    
    # Flatten initial circles for optimization
    initial_params = initial_circles.flatten()
    
    # Try different optimization methods
    methods = ['SLSQP', 'L-BFGS-B']
    
    for method in methods:
        try:
            result = minimize(
                objective,
                initial_params,
                method=method,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6},
                callback=lambda x: None
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                # Ensure all circles are valid
                for i in range(N_CIRCLES):
                    # Make sure radii are positive and reasonable
                    optimized_circles[i, 2] = max(0.001, optimized_circles[i, 2])
                    # Make sure positions are within bounds
                    optimized_circles[i, 0] = np.clip(optimized_circles[i, 0], 
                                                     optimized_circles[i, 2], 
                                                     1 - optimized_circles[i, 2])
                    optimized_circles[i, 1] = np.clip(optimized_circles[i, 1], 
                                                     optimized_circles[i, 2], 
                                                     1 - optimized_circles[i, 2])
                return optimized_circles
                
        except Exception:
            continue
    
    # Fallback to initial configuration if optimization fails
    return initial_circles

def iterative_refinement(circles):
    """Perform iterative refinement to improve the solution."""
    current_circles = circles.copy()
    
    # Try several rounds of improvement
    for round_num in range(5):
        improved = True
        iteration = 0
        
        while improved and iteration < 50:
            improved = False
            iteration += 1
            
            # Try to increase radii
            for i in range(N_CIRCLES):
                old_radius = current_circles[i, 2]
                # Try to increase radius by small amounts
                test_radius = min(old_radius + 0.002, 0.45)
                
                # Check if this would cause any overlaps
                valid = True
                for j in range(N_CIRCLES):
                    if i != j:
                        dx = current_circles[i, 0] - current_circles[j, 0]
                        dy = current_circles[i, 1] - current_circles[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = test_radius + current_circles[j, 2]
                        if dist < min_dist:
                            valid = False
                            break
                
                # Check containment
                if (test_radius > current_circles[i, 0] or 
                    test_radius > current_circles[i, 1] or
                    test_radius > 1 - current_circles[i, 0] or
                    test_radius > 1 - current_circles[i, 1]):
                    valid = False
                
                if valid:
                    current_circles[i, 2] = test_radius
                    improved = True
            
            # Try to adjust positions to allow larger radii
            for i in range(N_CIRCLES):
                old_pos = current_circles[i, :2].copy()
                # Try small movements
                for _ in range(5):
                    dx = np.random.uniform(-0.003, 0.003)
                    dy = np.random.uniform(-0.003, 0.003)
                    new_pos = old_pos + np.array([dx, dy])
                    
                    # Check if new position is valid
                    if (new_pos[0] >= current_circles[i, 2] and 
                        new_pos[0] <= 1 - current_circles[i, 2] and
                        new_pos[1] >= current_circles[i, 2] and 
                        new_pos[1] <= 1 - current_circles[i, 2]):
                        
                        # Check overlap constraints with other circles
                        valid = True
                        for j in range(N_CIRCLES):
                            if i != j:
                                dx = new_pos[0] - current_circles[j, 0]
                                dy = new_pos[1] - current_circles[j, 1]
                                dist = np.sqrt(dx*dx + dy*dy)
                                min_dist = current_circles[i, 2] + current_circles[j, 2]
                                if dist < min_dist:
                                    valid = False
                                    break
                        
                        if valid:
                            current_circles[i, 0] = new_pos[0]
                            current_circles[i, 1] = new_pos[1]
                            improved = True
                            break
    
    return current_circles

def validate_and_fix(circles):
    """Validate solution and fix any constraint violations."""
    validated_circles = circles.copy()
    
    # Ensure no overlaps and all circles are within bounds
    max_attempts = 10
    for attempt in range(max_attempts):
        valid = True
        positions = validated_circles[:, :2]
        radii = validated_circles[:, 2]
        
        # Check containment
        for i in range(N_CIRCLES):
            if (radii[i] > positions[i, 0] or 
                radii[i] > positions[i, 1] or
                radii[i] > 1 - positions[i, 0] or
                radii[i] > 1 - positions[i, 1]):
                valid = False
                break
        
        # Check non-overlap
        if valid:
            distances = cdist(positions, positions)
            for i in range(N_CIRCLES):
                for j in range(i+1, N_CIRCLES):
                    if distances[i, j] < radii[i] + radii[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            break
            
        # Fix violations by adjusting positions
        for i in range(N_CIRCLES):
            # Ensure containment
            validated_circles[i, 0] = np.clip(validated_circles[i, 0], 
                                             validated_circles[i, 2], 
                                             1 - validated_circles[i, 2])
            validated_circles[i, 1] = np.clip(validated_circles[i, 1], 
                                             validated_circles[i, 2], 
                                             1 - validated_circles[i, 2])
    
    return validated_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: improved hexagonal initialization + multi-stage optimization with 
    iterative refinement and adaptive constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Stage 1: Create improved initial configuration
    initial_circles = create_improved_hexagonal_initial()
    
    # Stage 2: Global optimization with scipy
    optimized_circles = optimize_with_scipy(initial_circles)
    
    # Stage 3: Local refinement with iterative improvement
    refined_circles = iterative_refinement(optimized_circles)
    
    # Stage 4: Final validation and adjustment
    final_circles = validate_and_fix(refined_circles)
    
    return final_circles


# EVOLVE-BLOCK-END
