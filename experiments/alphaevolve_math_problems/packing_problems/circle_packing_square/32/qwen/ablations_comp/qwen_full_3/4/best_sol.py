# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from sklearn.cluster import KMeans
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, local optimization, and 
    constraint handling improvements.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Even better initialization using a more sophisticated approach
    def initialize_sophisticated_layout():
        circles = np.zeros((n, 3))
        
        # Try to create a more optimal initial configuration
        # Start with a hexagonal packing pattern but adapt to 32 circles
        # In a hexagonal packing, we can place roughly 1 + 3*(k-1) circles in k rows
        # For 32 circles, let's try 6 rows (1 + 3*5 = 16) and add more
        
        # Create a better hexagonal pattern
        rows = 6
        cols = 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Adjust spacing to be slightly larger to allow for optimization
        hex_radius = min(spacing_x, spacing_y) * 0.45
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Offset every other row for hexagonal packing
                x = (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Ensure we're within bounds and adjust for radius
                x = max(hex_radius, min(1 - hex_radius, x))
                y = max(hex_radius, min(1 - hex_radius, y))
                
                circles[idx] = [x, y, hex_radius]
                idx += 1
                
                if idx >= n:
                    break
                    
        # If we don't have enough circles, fill remaining with random positions
        # but still respecting constraints
        if idx < n:
            for i in range(idx, n):
                # Random positions with proper radius constraints
                x = np.random.uniform(hex_radius, 1 - hex_radius)
                y = np.random.uniform(hex_radius, 1 - hex_radius)
                circles[i] = [x, y, hex_radius]
                    
        return circles
    
    # Create better initial configuration
    circles = initialize_sophisticated_layout()
    
    # Improved constraint functions with better numerical stability
    def containment_constraints(params):
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Vectorized containment constraints
        # Each constraint should be >= 0
        constraints = []
        
        # Left boundary: x - r >= 0
        constraints.extend(positions[:, 0] - radii)
        # Right boundary: 1 - x - r >= 0  
        constraints.extend(1 - positions[:, 0] - radii)
        # Bottom boundary: y - r >= 0
        constraints.extend(positions[:, 1] - radii)
        # Top boundary: 1 - y - r >= 0
        constraints.extend(1 - positions[:, 1] - radii)
        
        return np.array(constraints)
    
    def non_overlap_constraints(params):
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Vectorized non-overlap constraints
        constraints = []
        
        # More efficient approach using broadcasting
        # Create all pairwise differences
        pos_i = positions[:, np.newaxis, :]  # Shape: (n, 1, 2)
        pos_j = positions[np.newaxis, :, :]  # Shape: (1, n, 2)
        radii_i = radii[:, np.newaxis]       # Shape: (n, 1)
        radii_j = radii[np.newaxis, :]       # Shape: (1, n)
        
        # Compute all pairwise distances
        diff = pos_i - pos_j  # Shape: (n, n, 2)
        distances = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (n, n)
        
        # Compute all required distances
        required_distances = radii_i + radii_j  # Shape: (n, n)
        
        # Compute violations (negative when overlapping)
        violations = distances - required_distances
        
        # For each pair (i,j) with i<j, we want violations[i,j] >= 0
        # Extract upper triangle (excluding diagonal)
        upper_triangle = np.triu(violations, k=1)
        # Filter out zeros (diagonal elements)
        valid_violations = upper_triangle[upper_triangle != 0]
        
        # Convert to constraints (positive when satisfied)
        constraints = valid_violations
        
        return np.array(constraints)
    
    # Improved objective function with better numerical handling
    def objective(params):
        # Reshape params back to circles array
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Calculate negative sum of radii (we'll minimize this)
        # Add small regularization to prevent degenerate solutions
        return -np.sum(radii) - 1e-10 * np.sum(radii**2)
    
    # Set up optimization variables
    initial_positions = circles[:, :2].flatten()
    initial_radii = circles[:, 2]
    initial_params = np.concatenate([initial_positions, initial_radii])
    
    # Set bounds for optimization (positions and radii)
    bounds = []
    # Position bounds: [0,1] for both x and y
    for _ in range(2*n):
        bounds.extend([(0, 1)])
    # Radius bounds: [0.001, 0.5] (avoid zero radius to prevent numerical issues)
    for _ in range(n):
        bounds.extend([(0.001, 0.5)])
    
    # Define constraints
    constraints = [
        {'type': 'ineq', 'fun': lambda x: containment_constraints(x)},
        {'type': 'ineq', 'fun': lambda x: non_overlap_constraints(x)}
    ]
    
    # Perform optimization with multiple attempts and better solver selection
    best_result = None
    best_sum = -np.inf
    
    # Try multiple optimization runs with different initializations
    for attempt in range(5):
        # Use different random seeds and slight perturbations
        np.random.seed(attempt * 100 + 42)
        
        # Perturb initial parameters more significantly
        perturbed_positions = initial_positions.copy()
        perturbed_radii = initial_radii.copy()
        
        # Add noise to positions and radii
        noise_pos = np.random.normal(0, 0.02, len(initial_positions))
        noise_rad = np.random.normal(0, 0.01, len(initial_radii))
        
        perturbed_positions = perturbed_positions + noise_pos
        perturbed_radii = perturbed_radii + noise_rad
        
        # Clip to valid ranges
        perturbed_positions = np.clip(perturbed_positions, 0, 1)
        perturbed_radii = np.clip(perturbed_radii, 0.001, 0.5)
        
        perturbed_params = np.concatenate([perturbed_positions, perturbed_radii])
        
        try:
            # Try different optimization methods
            result = minimize(
                objective,
                perturbed_params,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-6},
                callback=None
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception as e:
            continue
    
    # If we have a valid result, use it; otherwise fallback to initial
    if best_result is not None and best_result.success:
        final_positions = best_result.x[:2*n].reshape(-1, 2)
        final_radii = best_result.x[2*n:]
        
        # Update circles with optimized values
        for i in range(n):
            circles[i, 0] = final_positions[i, 0]
            circles[i, 1] = final_positions[i, 1]
            circles[i, 2] = final_radii[i]
    else:
        # Final fallback to original initialization if optimization fails
        pass
    
    return circles


# EVOLVE-BLOCK-END
