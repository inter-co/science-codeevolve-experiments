# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from scipy.spatial import cKDTree
import random
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, local search, and constrained optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Even better initialization using a more sophisticated approach
    def initialize_sophisticated_placement():
        # Use a combination of grid and spiral placement for better distribution
        positions = []
        
        # Grid-based initialization with refinement
        rows = 6
        cols = 6
        padding = 0.05
        
        # Create a more sophisticated grid pattern with better spacing
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                # Apply golden ratio-like spacing for better distribution
                x = padding + (j + 0.5) * (1 - 2*padding) / cols
                y = padding + (i + 0.5) * (1 - 2*padding) / rows
                
                # Add more sophisticated jitter based on position
                jitter_factor = 0.015
                # More jitter near edges, less in center
                edge_factor = 1.0 - 0.5 * (abs(x - 0.5) + abs(y - 0.5))
                x += random.uniform(-jitter_factor, jitter_factor) * edge_factor
                y += random.uniform(-jitter_factor, jitter_factor) * edge_factor
                
                # Keep within bounds
                x = max(padding, min(1-padding, x))
                y = max(padding, min(1-padding, y))
                positions.append([x, y])
        
        # Trim to exactly n positions
        positions = positions[:n]
        
        # Ensure we have enough positions
        while len(positions) < n:
            positions.append([0.5, 0.5])  # Center fallback
            
        return np.array(positions)
    
    # Initialize with sophisticated placement
    initial_positions = initialize_sophisticated_placement()
    
    # Better estimation of initial radii with improved algorithm
    def estimate_initial_radii_better(positions):
        radii = []
        tree = cKDTree(positions)
        
        for i, pos in enumerate(positions):
            # Find the nearest neighbor distance
            distances, indices = tree.query(pos, k=4)  # Get 3 closest neighbors
            # Use the second closest (index 1) to avoid self-distance
            min_dist = distances[1] if len(distances) > 1 else 0.5
            
            # Set radius to be a fraction of the minimum distance to neighbors, but capped
            max_radius = min(0.5, min_dist/2.0)
            # Use a more conservative initial value to ensure feasibility
            radius = max(0.01, min(max_radius, 0.07))
            radii.append(radius)
            
        return np.array(radii)
    
    initial_radii = estimate_initial_radii_better(initial_positions)
    
    # Flatten initial parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = np.concatenate([initial_positions.flatten(), initial_radii])
    
    # Improved constraint functions with better error handling
    def containment_constraints(params):
        """Ensure all circles are within the unit square"""
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Vectorized constraints
        left_bound = positions[:, 0] - radii
        right_bound = 1 - positions[:, 0] - radii
        bottom_bound = positions[:, 1] - radii
        top_bound = 1 - positions[:, 1] - radii
        
        # Return all constraints (should all be >= 0)
        return np.concatenate([left_bound, right_bound, bottom_bound, top_bound])
    
    def non_overlap_constraints(params):
        """Ensure no overlap between circles - vectorized version"""
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Use scipy's cdist for efficient pairwise distance computation
        distances = cdist(positions, positions)
        
        # Create constraint matrix: for each pair (i,j), constraint is 
        # sqrt((x_i-x_j)^2 + (y_i-y_j)^2) - (r_i + r_j) >= 0
        constraints = []
        
        # Only check upper triangle to avoid duplicates
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                r_sum = radii[i] + radii[j]
                constraints.append(dist - r_sum)
                
        return np.array(constraints)
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        radii = params[2*n:]
        return -np.sum(radii)
    
    # Create bounds for parameters
    bounds = []
    # Position bounds [0, 1]
    for i in range(2*n):
        bounds.append((0, 1))
    # Radius bounds [0, 0.5] (maximum possible for one circle)
    for i in range(n):
        bounds.append((0, 0.5))
    
    # Define constraint dictionaries
    containment_cons = {
        'type': 'ineq',
        'fun': lambda p: containment_constraints(p)
    }
    
    non_overlap_cons = {
        'type': 'ineq', 
        'fun': lambda p: non_overlap_constraints(p)
    }
    
    # Try multiple optimization approaches with better error handling
    best_result = None
    best_sum = 0
    
    # Strategy 1: Direct optimization with SLSQP - most reliable for this problem
    try:
        result_slsqp = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=[containment_cons, non_overlap_cons],
            options={'maxiter': 400, 'ftol': 1e-7, 'eps': 1e-7},
            tol=1e-7
        )
        
        if result_slsqp.success:
            final_radii = result_slsqp.x[2*n:]
            sum_radii = np.sum(final_radii)
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_result = result_slsqp
                
    except Exception as e:
        pass
    
    # Strategy 2: Enhanced local search followed by optimization
    if best_result is None:
        try:
            # Enhanced local search improvement with more thorough iterations
            current_params = initial_params.copy()
            
            # More extensive iterative improvement
            for iteration in range(100):
                positions = current_params[:2*n].reshape(-1, 2)
                radii = current_params[2*n:]
                
                # Compute constraints to see how much we can increase radii
                tree = cKDTree(positions)
                new_radii = radii.copy()
                
                # For each circle, try to increase radius without violating constraints
                for i in range(n):
                    # Find closest neighbors
                    distances, indices = tree.query(positions[i], k=4)
                    if len(indices) > 1:
                        min_dist = distances[1]  # Second closest neighbor
                        max_possible_radius = min(0.5, min_dist/2.0)
                        # Increase radius more aggressively but safely
                        new_radii[i] = min(max_possible_radius, new_radii[i] * 1.03)
                
                # Update parameters
                current_params[2*n:] = new_radii
                
                # Occasionally run optimization to correct any violations
                if iteration % 15 == 0:
                    # Run a quick optimization step with tighter tolerances
                    try:
                        temp_result = minimize(
                            objective,
                            current_params,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 30, 'ftol': 1e-6},
                            tol=1e-6
                        )
                        if temp_result.success:
                            current_params = temp_result.x
                    except:
                        pass
            
            # Final optimization with this improved starting point
            result_local = minimize(
                objective,
                current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-7},
                tol=1e-7
            )
            
            if result_local.success:
                final_radii = result_local.x[2*n:]
                sum_radii = np.sum(final_radii)
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = result_local
                    
        except Exception as e:
            pass
    
    # Strategy 3: Try a completely different approach with better initial guess
    if best_result is None:
        try:
            # Try a more aggressive optimization with different starting configuration
            # Try placing circles in a more compact arrangement with better clustering
            compact_positions = []
            
            # Use a more structured spiral approach
            for i in range(n):
                if len(compact_positions) >= n:
                    break
                # Create a spiral that fills space more effectively
                angle = i * 0.8  # Different angular spacing
                radius = 0.4 * (1 - i/(n*1.5))  # Radius decreases toward center
                x = 0.5 + radius * math.cos(angle)
                y = 0.5 + radius * math.sin(angle)
                # Keep within bounds with padding
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                compact_positions.append([x, y])
                
            # Trim to exactly n positions
            compact_positions = compact_positions[:n]
            
            # Ensure we have enough positions
            while len(compact_positions) < n:
                compact_positions.append([0.5, 0.5])  # Center fallback
            
            compact_positions = np.array(compact_positions)
            compact_radii = estimate_initial_radii_better(compact_positions)
            compact_params = np.concatenate([compact_positions.flatten(), compact_radii])
            
            result_compact = minimize(
                objective,
                compact_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-7},
                tol=1e-7
            )
            
            if result_compact.success:
                final_radii = result_compact.x[2*n:]
                sum_radii = np.sum(final_radii)
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = result_compact
                    
        except Exception as e:
            pass
    
    # Strategy 4: If all else fails, try to improve the best result so far
    if best_result is None:
        # Last resort: Try to optimize the initial configuration more thoroughly
        try:
            # Run optimization with very tight tolerances on initial configuration
            result_final = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=[containment_cons, non_overlap_cons],
                options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-8},
                tol=1e-8
            )
            
            if result_final.success:
                final_radii = result_final.x[2*n:]
                sum_radii = np.sum(final_radii)
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = result_final
                    
        except Exception as e:
            pass
    
    # If still no good result, return the initial configuration
    if best_result is None:
        circles = np.column_stack([initial_positions, initial_radii])
        return circles
    
    # Extract the best result
    final_positions = best_result.x[:2*n].reshape(-1, 2)
    final_radii = best_result.x[2*n:]
    
    # Create output array
    circles = np.column_stack([final_positions, final_radii])
    return circles


# EVOLVE-BLOCK-END
