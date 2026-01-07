# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a multi-stage approach with improved seeding, force relaxation, and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Improved seeding using a more systematic approach
    def generate_improved_seeding():
        # Try to create a good starting configuration
        circles = []
        
        # Create a grid-like pattern with some randomness
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        
        # Place points on a grid with slight perturbations
        for i in range(grid_size):
            for j in range(grid_size):
                if len(circles) >= n:
                    break
                # Add small random perturbation
                x = spacing * (j + 1) + np.random.normal(0, spacing * 0.1)
                y = spacing * (i + 1) + np.random.normal(0, spacing * 0.1)
                
                # Ensure within bounds
                x = max(spacing, min(1 - spacing, x))
                y = max(spacing, min(1 - spacing, y))
                
                # Initial radius based on distance to boundaries
                r = min(x, 1-x, y, 1-y) * 0.3
                
                circles.append([x, y, r])
        
        # Fill remaining with random positions
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Initial radius
            r = min(x, 1-x, y, 1-y) * 0.2
            circles.append([x, y, r])
            
        return np.array(circles)
    
    # More efficient constraint checking
    def check_constraints(circles):
        """Fast constraint checking using vectorized operations"""
        # Check containment
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        if np.any(r > x) or np.any(r > 1 - x) or np.any(r > y) or np.any(r > 1 - y):
            return False
            
        # Check non-overlap using distance matrix
        if len(circles) > 1:
            distances = cdist(circles[:, :2], circles[:, :2])
            # Create mask for upper triangle (avoid double counting)
            mask = np.triu(np.ones((len(circles), len(circles))), k=1)
            # Get minimum distances between all pairs
            min_distances = np.min(distances * mask)
            # Check if any pair violates non-overlap constraint
            if min_distances < np.sum(circles[:, 2]) / len(circles):
                # This is a rough check - more precise would be needed
                for i in range(len(circles)):
                    for j in range(i+1, len(circles)):
                        dist_sq = (circles[i, 0] - circles[j, 0])**2 + (circles[i, 1] - circles[j, 1])**2
                        if dist_sq < (circles[i, 2] + circles[j, 2])**2:
                            return False
        return True
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        # Reshape flat array back to circles
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we minimize
    
    # Constraint functions - more efficient versions
    def constraint_containment(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        # Each circle must be fully contained
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        # r <= x <= 1-r and r <= y <= 1-r
        # Convert to inequality constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
        result = np.column_stack([
            x - r,           # x - r >= 0
            1 - x - r,       # 1 - x - r >= 0
            y - r,           # y - r >= 0
            1 - y - r        # 1 - y - r >= 0
        ]).flatten()
        return result
    
    def constraint_overlap(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        # Non-overlap constraints - for all pairs
        result = []
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1-x2)**2 + (y1-y2)**2
                # We want sqrt(dist_sq) >= r1 + r2
                # So we want (r1 + r2)^2 <= dist_sq
                # Which means dist_sq - (r1 + r2)^2 >= 0
                overlap = dist_sq - (r1 + r2)**2
                result.append(overlap)
        return np.array(result)
    
    # Improved force relaxation with better physics
    def apply_improved_relaxation(circles, iterations=100):
        # Better force relaxation with adaptive step sizes
        for iteration in range(iterations):
            forces = np.zeros_like(circles)
            
            # Calculate forces between all pairs
            for i in range(len(circles)):
                x1, y1, r1 = circles[i]
                for j in range(len(circles)):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dx = x2 - x1
                        dy = y2 - y1
                        dist_sq = dx*dx + dy*dy
                        
                        if dist_sq > 0:
                            dist = np.sqrt(dist_sq)
                            if dist < r1 + r2:
                                # Repulsive force - stronger when closer
                                force_magnitude = 0.01 * (r1 + r2 - dist) / (dist + 1e-10)
                                forces[i, 0] += force_magnitude * dx
                                forces[i, 1] += force_magnitude * dy
                            elif dist > 0.05:  # Attractive force for distant circles to cluster
                                force_magnitude = -0.001 / (dist_sq + 1e-10)
                                forces[i, 0] += force_magnitude * dx
                                forces[i, 1] += force_magnitude * dy
            
            # Apply forces with adaptive damping
            damping = 0.05 + 0.05 * (1 - iteration / iterations)  # Decreasing damping
            circles[:, :2] += damping * forces[:, :2]
            
            # Keep within bounds and adjust radii
            for k in range(len(circles)):
                x, y, r = circles[k]
                # Clamp to bounds
                circles[k, 0] = max(r, min(1-r, x))
                circles[k, 1] = max(r, min(1-r, y))
                
                # Try to increase radius if possible and valid
                temp_circles = circles.copy()
                temp_circles[k, 2] = min(0.5, r + 0.002)
                if check_constraints(temp_circles):
                    circles[k, 2] = temp_circles[k, 2]
        
        return circles
    
    # Initialize
    circles = generate_improved_seeding()
    
    # Apply improved relaxation
    circles = apply_improved_relaxation(circles)
    
    # Flatten for optimization
    circles_flat = circles.flatten()
    
    # Optimize using a more robust approach
    try:
        # Create bounds for optimization
        bounds = []
        for i in range(n):
            # x bounds
            bounds.append((0.001, 0.999))
            # y bounds
            bounds.append((0.001, 0.999))
            # r bounds (positive, reasonable upper bound)
            bounds.append((0.001, 0.499))
        
        # Set up constraints more carefully
        def containment_constraint(x):
            return constraint_containment(x)
            
        def overlap_constraint(x):
            return constraint_overlap(x)
        
        cons = [
            {'type': 'ineq', 'fun': containment_constraint},
            {'type': 'ineq', 'fun': overlap_constraint}
        ]
        
        # Run optimization with multiple attempts
        best_result = None
        best_sum = -np.inf
        
        # Try multiple optimization runs with different starting points
        for attempt in range(3):
            # Slightly perturb the current solution for different starts
            perturbed = circles_flat.copy()
            if attempt > 0:
                perturbed += np.random.normal(0, 0.001, len(perturbed))
            
            result = minimize(
                objective,
                perturbed,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                current_sum = -objective(result.x)  # Convert back to positive sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        
        if best_result is not None and best_result.success:
            circles_opt = best_result.x.reshape(-1, 3)
        else:
            circles_opt = circles
            
    except Exception as e:
        # Fallback to just the relaxed version if optimization fails
        circles_opt = circles
    
    # Final constraint check and adjustment
    if not check_constraints(circles_opt):
        # Try to fix constraints by reducing radii where needed
        for i in range(len(circles_opt)):
            x, y, r = circles_opt[i]
            # Reduce radius if needed to satisfy containment
            max_r = min(x, 1-x, y, 1-y)
            circles_opt[i, 2] = min(r, max_r * 0.99)
            
            # Try to increase radius if possible while maintaining constraints
            test_r = min(max_r, r + 0.005)
            temp_circles = circles_opt.copy()
            temp_circles[i, 2] = test_r
            
            if check_constraints(temp_circles):
                circles_opt[i, 2] = test_r
    
    # Ensure all circles are valid
    for i in range(len(circles_opt)):
        x, y, r = circles_opt[i]
        # Make sure radii aren't too large
        max_r = min(x, 1-x, y, 1-y)
        circles_opt[i, 2] = min(r, max_r * 0.99)
    
    return circles_opt


# EVOLVE-BLOCK-END
