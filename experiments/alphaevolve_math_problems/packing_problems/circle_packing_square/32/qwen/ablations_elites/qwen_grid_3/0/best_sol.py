# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time
import random

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def initialize_better_placement(n: int) -> np.ndarray:
    """Initialize circles using a better structured approach inspired by known packings"""
    circles = np.zeros((n, 3))
    
    # Use a refined hexagonal packing pattern for better initial distribution
    # Based on INSPIRATION 1's approach but with better spacing and distribution
    
    # Create a more systematic hexagonal grid
    rows = 6
    cols = 6
    spacing_x = 0.15
    spacing_y = 0.15 * np.sqrt(3)/2  # Vertical spacing for hexagon
    offset_x = 0.15
    offset_y = 0.15
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = offset_x + j * spacing_x + (i % 2) * spacing_x/2  # Hexagonal offset
            y = offset_y + i * spacing_y
            
            # Make sure we're still within bounds
            if x <= 1 - 0.05 and y <= 1 - 0.05:
                # Add controlled randomness to avoid perfect symmetry
                x += random.uniform(-0.005, 0.005) * spacing_x
                y += random.uniform(-0.005, 0.005) * spacing_y
                # Set radius to a reasonable value
                r = 0.04 * (0.8 + random.random() * 0.6)
                circles[idx] = [x, y, r]
                idx += 1
    
    # If we didn't fill all positions, fill remaining with better spaced grid
    if idx < n:
        # Use a denser grid for the remainder
        rows_remaining = 5
        cols_remaining = int(np.ceil((n - idx) / rows_remaining))
        spacing_x = 0.8 / cols_remaining
        spacing_y = 0.8 / rows_remaining
        offset_x = 0.1
        offset_y = 0.1
        
        for i in range(idx, n):
            row = (i - idx) // cols_remaining
            col = (i - idx) % cols_remaining
            x = offset_x + col * spacing_x
            y = offset_y + row * spacing_y
            # Add slight randomness
            x += random.uniform(-spacing_x/6, spacing_x/6)
            y += random.uniform(-spacing_y/6, spacing_y/6)
            circles[i] = [x, y, 0.03]
    
    # Ensure all circles are within bounds and valid
    for i in range(n):
        x, y, r = circles[i]
        # Clip to valid range
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        r = np.clip(r, 0.01, 0.5)
        circles[i] = [x, y, r]
    
    return circles

def validate_and_correct_solution(circles: np.ndarray) -> np.ndarray:
    """Ensure solution satisfies constraints and correct any violations"""
    n = len(circles)
    corrected = circles.copy()
    
    # First pass: fix containment violations by adjusting positions
    for i in range(n):
        x, y, r = corrected[i]
        # Ensure circle fits within bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        corrected[i] = [x, y, r]
    
    # Second pass: adjust radii to avoid overlaps using more efficient neighbor search
    max_iter = 30  # Reduced iterations for speed
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            x, y, r = corrected[i]
            
            # Calculate max possible radius without violating constraints
            max_radius = min(x, 1-x, y, 1-y)
            
            # Check overlap with neighbors using KDTree for efficiency
            points = corrected[:, :2]
            tree = cKDTree(points)
            # Find neighbors within a reasonable distance (use a conservative estimate)
            neighbors = tree.query_ball_point([x, y], 2*max_radius)
            
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = corrected[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if distance > 0:
                        max_radius = min(max_radius, distance - r2)
            
            # Reduce radius if necessary
            if max_radius < r and max_radius > 0.001:
                corrected[i, 2] = max_radius
                improved = True
                
        if not improved:
            break
    
    return corrected

def constraint_penalty_objective(circles: np.ndarray, penalty_weight: float = 50000.0) -> float:
    """Objective function with strong penalty for constraint violations"""
    # Objective: maximize sum of radii (negative because we minimize)
    objective_value = -np.sum(circles[:, 2])
    
    # Penalty for containment violations
    penalty = 0
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Violation penalties for boundary constraints
        if x - r < 0:
            penalty += penalty_weight * (x - r)**2
        if x + r > 1:
            penalty += penalty_weight * (x + r - 1)**2
        if y - r < 0:
            penalty += penalty_weight * (y - r)**2
        if y + r > 1:
            penalty += penalty_weight * (y + r - 1)**2
    
    # Penalty for overlap violations - stronger penalty for severe violations
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                # Strong penalty for overlaps
                overlap_amount = (r1 + r2 - distance)
                penalty += penalty_weight * overlap_amount**2
    
    return objective_value + penalty

def compute_constraints(circles: np.ndarray) -> np.ndarray:
    """Compute constraint violations as a single array for optimization"""
    n = len(circles)
    constraints = []
    
    # Containment constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    for i in range(n):
        x, y, r = circles[i]
        constraints.extend([
            x - r,           # x >= r
            1 - x - r,       # x <= 1-r
            y - r,           # y >= r
            1 - y - r        # y <= 1-r
        ])
    
    # Non-overlap constraints: distance >= r1 + r2 (we want distance - (r1+r2) >= 0)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            constraints.append(distance - (r1 + r2))
    
    return np.array(constraints)

def optimize_with_differential_evolution(circles: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Use differential evolution for global optimization"""
    n = len(circles)
    
    # Flatten circles array for optimization (x, y, r for each circle)
    initial_flat = circles.flatten()
    
    # Define bounds for optimization (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    def objective(flat_params):
        temp_circles = flat_params.reshape(-1, 3)
        return constraint_penalty_objective(temp_circles)
    
    try:
        # Use differential evolution for global optimization
        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iter,
            popsize=15,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False,
            atol=1e-6,
            rtol=1e-6
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return validate_and_correct_solution(optimized_circles)
    except Exception as e:
        pass
    
    # If optimization fails, return original
    return circles

def optimize_with_slsqp(circles: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Use SLSQP optimization with proper constraints"""
    n = len(circles)
    
    # Flatten circles array for optimization
    initial_flat = circles.flatten()
    
    def objective(flat_params):
        temp_circles = flat_params.reshape(-1, 3)
        return constraint_penalty_objective(temp_circles)
    
    # Define tighter bounds to improve optimization
    bounds = []
    for i in range(n):
        # x coordinate bounds (r <= x <= 1-r)
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.499))  # r (max radius ~0.5)
    
    # Set up constraints with proper structure
    def constraint_func(flat_params):
        temp_circles = flat_params.reshape(-1, 3)
        return compute_constraints(temp_circles)
    
    cons = [{'type': 'ineq', 'fun': constraint_func}]
    
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return validate_and_correct_solution(optimized_circles)
    except Exception as e:
        pass
    
    # If optimization fails, return original
    return circles

def local_improvement_step(circles: np.ndarray, max_iterations: int = 30) -> np.ndarray:
    """Perform local improvement to increase radii where possible"""
    n = len(circles)
    improved_circles = circles.copy()
    
    # Use more efficient approach with early termination and better search
    for iteration in range(max_iterations):
        improved = False
        
        # Shuffle the order to avoid systematic bias
        indices = list(range(n))
        random.shuffle(indices)
        
        # Try to increase radii iteratively in shuffled order
        for i in indices:
            x, y, r = improved_circles[i]
            
            # Compute maximum possible radius
            max_radius = min(x, 1-x, y, 1-y)
            
            # Check constraints with neighbors using efficient search
            points = improved_circles[:, :2]
            tree = cKDTree(points)
            # Query neighbors with a reasonable distance bound
            neighbors = tree.query_ball_point([x, y], 2*max_radius)
            
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = improved_circles[j]
                    dx = x - x2
                    dy = y - y2
                    distance = np.sqrt(dx*dx + dy*dy)
                    # Can't get closer than sum of radii
                    if distance > 0:
                        max_radius = min(max_radius, distance - r2)
            
            # Try to increase radius slightly if beneficial
            if max_radius > r and max_radius > r + 1e-6:
                # Use a more aggressive increment for better improvement
                increment = min(0.003, max_radius - r)
                new_r = min(max_radius, r + increment)
                # Verify that this change doesn't violate constraints
                valid = True
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = improved_circles[j]
                        dx = new_r + r2 - np.sqrt((x - x2)**2 + (y - y2)**2)
                        if dx > 0:  # Would cause overlap
                            valid = False
                            break
                
                if valid:
                    improved_circles[i, 2] = new_r
                    improved = True
        
        # Early termination if no improvements made
        if not improved:
            break
    
    return improved_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Step 1: Initialize with better structured placement (inspired by hexagonal pattern)
    circles = initialize_better_placement(n)
    
    # Step 2: Apply multiple rounds of local improvement
    for round_num in range(2):
        circles = local_improvement_step(circles, 25)
    
    # Step 3: Apply global optimization with differential evolution
    circles = optimize_with_differential_evolution(circles, 150)
    
    # Step 4: Apply local improvement again
    circles = local_improvement_step(circles, 20)
    
    # Step 5: Apply SLSQP optimization for final refinement
    circles = optimize_with_slsqp(circles, 100)
    
    # Step 6: Final local improvement
    circles = local_improvement_step(circles, 15)
    
    # Step 7: Additional fine-tuning with enhanced local search
    # Inspired by INSPIRATION PROGRAM 2's approach but with better efficiency
    for iteration in range(5):  # Reduced iterations for speed
        temp_circles = circles.copy()
        improved = False
        
        # Try to slightly increase radii of circles that have room
        for i in range(n):
            x, y, r = temp_circles[i]
            
            # Try to increase radius slightly if there's space
            max_radius = min(x, 1-x, y, 1-y)
            
            # Check neighbors efficiently with KDTree
            points = temp_circles[:, :2]
            tree = cKDTree(points)
            neighbors = tree.query_ball_point([x, y], 2*max_radius)
            
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = temp_circles[j]
                    distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                    max_radius = min(max_radius, distance - r2)
            
            # Only increase if beneficial and safe
            if max_radius > r and max_radius - r > 0.0005:
                new_r = min(max_radius, r + 0.005)  # Slightly smaller increments
                temp_circles[i, 2] = new_r
                improved = True
        
        if improved:
            temp_circles = validate_and_correct_solution(temp_circles)
            circles = temp_circles.copy()
    
    return circles


# EVOLVE-BLOCK-END
