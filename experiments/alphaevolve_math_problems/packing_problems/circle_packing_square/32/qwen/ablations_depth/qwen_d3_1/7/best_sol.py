# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from scipy.spatial import cKDTree
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-stage approach: strategic initialization + hybrid optimization + fine-tuning.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Stage 1: Strategic initialization using a more informed approach
    circles = initialize_strategic_grid(n)
    
    # Stage 2: Hybrid optimization combining multiple strategies
    circles = hybrid_optimization(circles)
    
    # Stage 3: Fine-tuning with specialized refinement
    circles = fine_tuning_refinement(circles)
    
    return circles

def initialize_strategic_grid(n: int) -> np.ndarray:
    """Initialize circle positions using a strategic approach inspired by known good packings."""
    # Create a configuration that tries to balance density and spacing
    # For 32 circles, let's try a 5x7 grid with special adjustments
    
    rows = 5
    cols = 7
    
    # Ensure we have enough positions
    while rows * cols < n:
        rows += 1
        cols = int(n / rows) + 1
    
    # Create positions with strategic spacing
    circles = np.zeros((n, 3))
    
    # Calculate spacing with some padding
    spacing_x = 0.9 / cols
    spacing_y = 0.9 / rows
    
    # Create positions in a strategic pattern
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Position with strategic offset
        x = (col + 0.5) * spacing_x
        y = (row + 0.5) * spacing_y
        
        # Add small random perturbations to avoid perfect patterns
        x += random.uniform(-spacing_x*0.05, spacing_x*0.05)
        y += random.uniform(-spacing_y*0.05, spacing_y*0.05)
        
        # Ensure within bounds
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        
        # Initial radius - start with a more informed value
        r = min(spacing_x, spacing_y) * 0.25
        
        circles[i] = [x, y, r]
    
    return circles

def hybrid_optimization(initial_circles: np.ndarray) -> np.ndarray:
    """Use a hybrid optimization approach combining global and local search."""
    current_circles = initial_circles.copy()
    
    # Multiple phases with different optimization strategies
    # Phase 1: Heavy global exploration
    current_circles = heavy_global_exploration(current_circles)
    
    # Phase 2: Refinement with gradient methods
    current_circles = gradient_refinement(current_circles)
    
    # Phase 3: Local improvement with constraint-aware moves
    current_circles = constraint_aware_local_improvement(current_circles)
    
    return current_circles

def heavy_global_exploration(initial_circles: np.ndarray, max_iter: int = 2000) -> np.ndarray:
    """Aggressive global exploration using multiple move types."""
    current_circles = initial_circles.copy()
    best_circles = current_circles.copy()
    best_sum = -objective_function(current_circles)
    
    # Very aggressive temperature schedule
    temp = 0.15
    cooling_rate = 0.998
    min_temp = 1e-7
    
    for iteration in range(max_iter):
        new_circles = current_circles.copy()
        
        # Choose move type with bias towards more impactful moves
        move_types = ['position', 'radius', 'combined', 'global_shift']
        weights = [0.3, 0.3, 0.3, 0.1]  # Bias towards position/radius moves
        move_type = random.choices(move_types, weights=weights)[0]
        
        # Select a random circle to modify
        idx = random.randint(0, len(new_circles) - 1)
        
        if move_type == 'position' or move_type == 'combined':
            # Larger position perturbations for global exploration
            new_circles[idx, 0] += random.uniform(-0.02, 0.02)
            new_circles[idx, 1] += random.uniform(-0.02, 0.02)
            
            # Keep within bounds
            new_circles[idx, 0] = max(0.05, min(0.95, new_circles[idx, 0]))
            new_circles[idx, 1] = max(0.05, min(0.95, new_circles[idx, 1]))
        
        if move_type == 'radius' or move_type == 'combined':
            # Aggressive radius adjustment
            max_radius = min(
                new_circles[idx, 0], 
                new_circles[idx, 1], 
                1 - new_circles[idx, 0], 
                1 - new_circles[idx, 1]
            )
            
            # Check overlap constraints
            valid_radius = max_radius
            for j in range(len(new_circles)):
                if j != idx:
                    dist = np.sqrt(
                        (new_circles[idx, 0] - new_circles[j, 0])**2 +
                        (new_circles[idx, 1] - new_circles[j, 1])**2
                    )
                    min_dist = new_circles[idx, 2] + new_circles[j, 2]
                    if dist > 0 and dist < min_dist:
                        max_radius_for_overlap = dist - new_circles[j, 2]
                        valid_radius = min(valid_radius, max_radius_for_overlap)
            
            # Update radius more aggressively
            if valid_radius > new_circles[idx, 2]:
                new_circles[idx, 2] = min(valid_radius, max_radius)
        
        if move_type == 'global_shift':
            # Move multiple circles together to explore different configurations
            shift_amount = random.uniform(0.005, 0.02)
            direction = random.choice(['x', 'y', 'both'])
            
            # Apply shift to a subset of circles
            selected_indices = random.sample(range(len(new_circles)), 
                                           max(2, len(new_circles) // 8))
            
            for idx in selected_indices:
                if direction == 'x' or direction == 'both':
                    new_circles[idx, 0] += random.choice([-shift_amount, shift_amount])
                    new_circles[idx, 0] = max(0.05, min(0.95, new_circles[idx, 0]))
                
                if direction == 'y' or direction == 'both':
                    new_circles[idx, 1] += random.choice([-shift_amount, shift_amount])
                    new_circles[idx, 1] = max(0.05, min(0.95, new_circles[idx, 1]))
        
        # Accept or reject based on simulated annealing
        current_sum = -objective_function(current_circles)
        new_sum = -objective_function(new_circles)
        
        delta = new_sum - current_sum
        
        # Acceptance with adaptive probability
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_circles = new_circles
            if new_sum > best_sum:
                best_sum = new_sum
                best_circles = new_circles.copy()
        
        # Cool down temperature
        temp = max(min_temp, temp * cooling_rate)
    
    return best_circles

def gradient_refinement(initial_circles: np.ndarray) -> np.ndarray:
    """Use gradient-based refinement with better constraint handling."""
    n = len(initial_circles)
    refined_circles = initial_circles.copy()
    
    # Run a more thorough optimization loop
    for iteration in range(5):
        # Run scipy optimization
        try:
            # Flatten parameters
            initial_params = refined_circles.flatten()
            
            # Define bounds
            bounds = []
            for i in range(n):
                bounds.extend([(0.05, 0.95), (0.05, 0.95), (0.01, 0.45)])
            
            def obj(params):
                circles = params.reshape(-1, 3)
                return -np.sum(circles[:, 2])
            
            def constraint_func(params):
                circles = params.reshape(-1, 3)
                constraints = []
                
                # Containment constraints (should be >= 0)
                for i in range(n):
                    x, y, r = circles[i]
                    max_r = min(x, y, 1-x, 1-y)
                    constraints.append(max_r - r)
                
                # Overlap constraints (should be <= 0)
                positions = circles[:, :2]
                distances = cdist(positions, positions)
                for i in range(n):
                    for j in range(i+1, n):
                        if i != j:
                            dist = distances[i, j]
                            min_dist = circles[i, 2] + circles[j, 2]
                            constraints.append(dist - min_dist)
                
                return constraints
            
            # Run optimization with more iterations and stricter tolerance
            result = minimize(
                obj,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                refined_circles = result.x.reshape(-1, 3)
            
        except Exception:
            # If optimization fails, continue with current state
            pass
    
    return refined_circles

def constraint_aware_local_improvement(initial_circles: np.ndarray) -> np.ndarray:
    """Perform local improvement that respects constraints more carefully."""
    refined_circles = initial_circles.copy()
    n = len(refined_circles)
    
    # Multiple rounds of careful improvement
    for round_num in range(5):
        improved = True
        iteration = 0
        while improved and iteration < 400:
            improved = False
            # Process circles in random order for better exploration
            indices = list(range(n))
            random.shuffle(indices)
            
            for i in indices:
                old_radius = refined_circles[i, 2]
                
                # Calculate maximum possible radius for this circle
                max_radius = min(
                    refined_circles[i, 0], 
                    refined_circles[i, 1], 
                    1 - refined_circles[i, 0], 
                    1 - refined_circles[i, 1]
                )
                
                # Check overlap constraints with all other circles
                valid_radius = max_radius
                for j in range(n):
                    if i != j:
                        dist = np.sqrt(
                            (refined_circles[i, 0] - refined_circles[j, 0])**2 +
                            (refined_circles[i, 1] - refined_circles[j, 1])**2
                        )
                        min_dist = refined_circles[i, 2] + refined_circles[j, 2]
                        if dist > 0 and dist < min_dist:
                            max_radius_for_overlap = dist - refined_circles[j, 2]
                            valid_radius = min(valid_radius, max_radius_for_overlap)
                
                # Aggressive update with safety margin
                if valid_radius > old_radius * 1.01:  # Only if significantly better
                    refined_circles[i, 2] = valid_radius
                    improved = True
            
            iteration += 1
    
    return refined_circles

def fine_tuning_refinement(initial_circles: np.ndarray) -> np.ndarray:
    """Final fine-tuning with specialized techniques."""
    refined_circles = initial_circles.copy()
    
    # Post-processing step: try to squeeze out every bit of improvement
    # This involves more exhaustive local search around promising areas
    
    # Try to improve individual circles more systematically
    for i in range(len(refined_circles)):
        # For each circle, try to find the exact maximum radius
        max_radius = min(
            refined_circles[i, 0], 
            refined_circles[i, 1], 
            1 - refined_circles[i, 0], 
            1 - refined_circles[i, 1]
        )
        
        # Check all overlap constraints
        valid_radius = max_radius
        for j in range(len(refined_circles)):
            if i != j:
                dist = np.sqrt(
                    (refined_circles[i, 0] - refined_circles[j, 0])**2 +
                    (refined_circles[i, 1] - refined_circles[j, 1])**2
                )
                min_dist = refined_circles[i, 2] + refined_circles[j, 2]
                if dist > 0 and dist < min_dist:
                    max_radius_for_overlap = dist - refined_circles[j, 2]
                    valid_radius = min(valid_radius, max_radius_for_overlap)
        
        # Set to the computed maximum radius
        refined_circles[i, 2] = valid_radius
    
    # Run one final optimization pass
    try:
        n = len(refined_circles)
        initial_params = refined_circles.flatten()
        
        bounds = []
        for i in range(n):
            bounds.extend([(0.05, 0.95), (0.05, 0.95), (0.01, 0.45)])
        
        def obj(params):
            circles = params.reshape(-1, 3)
            return -np.sum(circles[:, 2])
        
        def constraint_func(params):
            circles = params.reshape(-1, 3)
            constraints = []
            
            # Containment constraints
            for i in range(n):
                x, y, r = circles[i]
                max_r = min(x, y, 1-x, 1-y)
                constraints.append(max_r - r)
            
            # Overlap constraints
            positions = circles[:, :2]
            distances = cdist(positions, positions)
            for i in range(n):
                for j in range(i+1, n):
                    if i != j:
                        dist = distances[i, j]
                        min_dist = circles[i, 2] + circles[j, 2]
                        constraints.append(dist - min_dist)
            
            return constraints
        
        result = minimize(
            obj,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
            options={'maxiter': 150, 'ftol': 1e-8}
        )
        
        if result.success:
            refined_circles = result.x.reshape(-1, 3)
            
    except Exception:
        pass
    
    return refined_circles

def objective_function(circles: np.ndarray) -> float:
    """Objective function to maximize sum of radii."""
    return -np.sum(circles[:, 2])  # Negative because we're minimizing


# EVOLVE-BLOCK-END
