# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import random
from deap import base, creator, tools, algorithms
import time
from numba import jit
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def fast_compute_constraints_jit(positions, radii, n):
    """Fast constraint computation using numba"""
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        x, y, r = positions[i, 0], positions[i, 1], radii[i]
        constraints.extend([
            x - r,           # x >= r
            1 - x - r,       # x <= 1 - r  
            y - r,           # y >= r
            1 - y - r        # y <= 1 - r
        ])
    
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx*dx + dy*dy
            r1, r2 = radii[i], radii[j]
            # Distance squared should be >= (r1 + r2)^2 for no overlap
            constraints.append(dist_sq - (r1 + r2)**2)
    
    return np.array(constraints)

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach with better initialization and targeted optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    random.seed(42)
    np.random.seed(42)
    
    # Better initialization using a more sophisticated approach
    def initialize_better():
        circles = np.zeros((n, 3))
        
        # Use a grid-based approach with adaptive spacing
        # Try to create a more uniform distribution
        rows = 5
        cols = 5
        
        # Calculate spacing to fit circles nicely in the unit square
        padding = 0.05
        available_width = 1.0 - 2 * padding
        available_height = 1.0 - 2 * padding
        
        spacing_x = available_width / (cols - 1) if cols > 1 else 0.5
        spacing_y = available_height / (rows - 1) if rows > 1 else 0.5
        
        # Adjust spacing to ensure circles don't exceed boundaries
        max_radius = min(spacing_x, spacing_y) * 0.4
        actual_spacing_x = spacing_x
        actual_spacing_y = spacing_y
        
        # Create a more sophisticated hexagonal packing pattern
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset pattern
                x_offset = 0 if i % 2 == 0 else actual_spacing_x * 0.5
                x = padding + j * actual_spacing_x + x_offset
                y = padding + i * actual_spacing_y
                
                # Add more significant random perturbation to escape local minima
                x += np.random.normal(0, actual_spacing_x * 0.05)
                y += np.random.normal(0, actual_spacing_y * 0.05)
                
                # Ensure within bounds
                x = np.clip(x, padding + max_radius, 1 - padding - max_radius)
                y = np.clip(y, padding + max_radius, 1 - padding - max_radius)
                
                # Use larger initial radius to encourage better packing
                circles[idx] = [x, y, max_radius * 0.8]
                idx += 1
                
                if idx >= n:
                    break
        
        # Fill remaining positions with careful placement
        for i in range(idx, n):
            placed = False
            attempts = 0
            while not placed and attempts < 200:  # More attempts for better placement
                x = np.random.uniform(padding + max_radius, 1 - padding - max_radius)
                y = np.random.uniform(padding + max_radius, 1 - padding - max_radius)
                
                # Check distance to all existing circles
                min_dist = float('inf')
                for k in range(i):
                    dx = x - circles[k, 0]
                    dy = y - circles[k, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = min(min_dist, dist)
                
                # Place if sufficiently distant
                if min_dist > max_radius * 1.5 or attempts > 100:  # Larger minimum distance
                    circles[i] = [x, y, max_radius * 0.7]
                    placed = True
                attempts += 1
            
            if not placed:
                circles[i] = [np.random.uniform(padding + max_radius, 1 - padding - max_radius), 
                             np.random.uniform(padding + max_radius, 1 - padding - max_radius), 
                             max_radius * 0.7]
            
        return circles
    
    # Optimized constraint checking with better numerical stability
    def compute_constraints(circles):
        """Compute all constraints efficiently"""
        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Use numba-optimized version for speed
        try:
            return fast_compute_constraints_jit(positions, radii, n)
        except:
            # Fallback to pure Python version
            constraints = []
            
            # Boundary constraints: each circle must stay within bounds
            for i in range(n):
                x, y, r = circles[i]
                # Each circle must stay within bounds (r <= x <= 1-r, r <= y <= 1-r)
                constraints.extend([
                    x - r,           # x >= r
                    1 - x - r,       # x <= 1 - r  
                    y - r,           # y >= r
                    1 - y - r        # y <= 1 - r
                ])
            
            # Overlap constraints using vectorized computation
            try:
                distances = cdist(positions, positions, 'sqeuclidean')
                # Overlap constraints: distance^2 >= (r1 + r2)^2 for all pairs
                for i in range(n):
                    for j in range(i+1, n):
                        dist_sq = distances[i, j]
                        r1, r2 = radii[i], radii[j]
                        # Distance squared should be >= (r1 + r2)^2 for no overlap
                        constraints.append(dist_sq - (r1 + r2)**2)
            except:
                # Fallback for any computation errors
                for i in range(n):
                    for j in range(i+1, n):
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist_sq = dx*dx + dy*dy
                        r1, r2 = radii[i], radii[j]
                        constraints.append(dist_sq - (r1 + r2)**2)
                
            return np.array(constraints)
    
    # Optimization objective - maximize sum of radii
    def objective(circles_flat):
        # Extract radii from flattened array
        radii = circles_flat[2::3]
        return -np.sum(radii)  # Negative because we want to maximize
    
    # Enhanced constraint function with proper handling
    def constraint_func(circles_flat):
        # Convert flat array back to circles
        circles = circles_flat.reshape(-1, 3)
        return compute_constraints(circles)
    
    # Improved optimization with better local search
    def improved_local_search(initial_solution):
        """Use a more aggressive and diverse local search approach"""
        try:
            best_solution = initial_solution.copy()
            best_value = -objective(best_solution)
            
            # Multiple optimization attempts with different strategies
            for attempt in range(15):  # Increase from 10 to 15 for more exploration
                # Perturb solution slightly
                perturbed = initial_solution.copy()
                # Add random noise with adaptive scaling
                noise_scale = 0.005 if attempt < 8 else 0.002  # More noise initially
                for i in range(len(perturbed)):
                    if i % 3 != 2:  # Don't perturb radius too much
                        perturbed[i] += np.random.normal(0, noise_scale)
                    else:  # Perturb radius with smaller scale
                        perturbed[i] += np.random.normal(0, noise_scale * 0.5)
                
                # Clip to valid ranges
                for i in range(0, len(perturbed), 3):
                    perturbed[i] = np.clip(perturbed[i], 0.001, 0.999)  # x
                    perturbed[i+1] = np.clip(perturbed[i+1], 0.001, 0.999)  # y
                    perturbed[i+2] = np.clip(perturbed[i+2], 0.001, 0.499)  # r
                
                # Try different optimization methods with varying settings
                bounds = []
                for i in range(n):
                    bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                
                # Alternate between multiple methods for better exploration
                methods = ['L-BFGS-B', 'TNC', 'SLSQP']
                method = methods[attempt % len(methods)]
                options = {'maxiter': 2500, 'ftol': 1e-9} if method in ['L-BFGS-B', 'TNC'] else {'maxiter': 2000, 'ftol': 1e-9}
                
                try:
                    if method in ['L-BFGS-B', 'TNC']:
                        result = minimize(
                            objective,
                            perturbed,
                            method=method,
                            bounds=bounds,
                            options=options
                        )
                    else:
                        result = minimize(
                            objective,
                            perturbed,
                            method=method,
                            bounds=bounds,
                            constraints={'type': 'ineq', 'fun': constraint_func},
                            options=options
                        )
                    
                    if result.success:
                        new_value = -objective(result.x)
                        if new_value > best_value:
                            best_value = new_value
                            best_solution = result.x.copy()
                except:
                    continue
            
            return best_solution
        except Exception as e:
            return initial_solution
    
    # More robust constraint checking with early termination
    @jit(nopython=True)
    def fast_constraint_check(positions, radii, n):
        """More efficient constraint checking with early exit"""
        # Check boundary constraints
        for i in range(n):
            x, y, r = positions[i, 0], positions[i, 1], radii[i]
            if x - r < -1e-10 or 1 - x - r < -1e-10 or y - r < -1e-10 or 1 - y - r < -1e-10:
                return False
        
        # Check overlap constraints with early exit
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                r1, r2 = radii[i], radii[j]
                if dist_sq < (r1 + r2)**2 - 1e-10:
                    return False
        return True
    
    # Main optimization approach
    def optimized_approach():
        # Strategy 1: Start with good initialization
        initial_circles = initialize_better()
        initial_flat = initial_circles.flatten()
        
        # Strategy 2: Try multiple optimization approaches
        best_result = None
        best_sum = -float('inf')
        
        # Try multiple optimization runs with different parameters
        for run in range(12):  # Increase from 8 to 12
            try:
                # Start with a slightly perturbed version to avoid local minima
                if run > 0:
                    perturbed = initial_flat.copy()
                    # Add small random noise to all parameters
                    noise = np.random.normal(0, 0.003, len(perturbed))  # Slightly larger noise
                    perturbed += noise
                    # Clip to valid ranges
                    for i in range(0, len(perturbed), 3):
                        perturbed[i] = np.clip(perturbed[i], 0.001, 0.999)  # x
                        perturbed[i+1] = np.clip(perturbed[i+1], 0.001, 0.999)  # y
                        perturbed[i+2] = np.clip(perturbed[i+2], 0.001, 0.499)  # r
                    start_solution = perturbed
                else:
                    start_solution = initial_flat
                
                # Try multiple methods with different settings
                methods = ['L-BFGS-B', 'TNC', 'SLSQP']
                bounds = []
                for i in range(n):
                    bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                
                # Try different optimization methods
                for method_idx, method in enumerate(methods):
                    try:
                        options = {'maxiter': 3000, 'ftol': 1e-9} if method in ['L-BFGS-B', 'TNC'] else {'maxiter': 2500, 'ftol': 1e-9}
                        
                        if method in ['L-BFGS-B', 'TNC']:
                            result = minimize(
                                objective,
                                start_solution,
                                method=method,
                                bounds=bounds,
                                options=options
                            )
                        else:
                            result = minimize(
                                objective,
                                start_solution,
                                method=method,
                                bounds=bounds,
                                constraints={'type': 'ineq', 'fun': constraint_func},
                                options=options
                            )
                        
                        if result.success:
                            circles = result.x.reshape(-1, 3)
                            constraints = compute_constraints(circles)
                            if np.all(constraints >= -1e-6):
                                radii_sum = np.sum(circles[:, 2])
                                if radii_sum > best_sum:
                                    best_sum = radii_sum
                                    best_result = result.x.copy()
                    
                    except Exception as e:
                        continue
                
                # Also try with a more aggressive refinement step
                if best_result is not None and run > 0:
                    try:
                        # Do one more aggressive optimization on the best result
                        result = minimize(
                            objective,
                            best_result,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 4000, 'ftol': 1e-10}
                        )
                        
                        if result.success:
                            circles = result.x.reshape(-1, 3)
                            constraints = compute_constraints(circles)
                            if np.all(constraints >= -1e-6):
                                radii_sum = np.sum(circles[:, 2])
                                if radii_sum > best_sum:
                                    best_sum = radii_sum
                                    best_result = result.x.copy()
                    except:
                        pass
                    
            except Exception as e:
                continue
        
        # If we have a good result, do additional refinement
        if best_result is not None:
            # Apply additional local search refinement with more aggressive approach
            refined_result = improved_local_search(best_result)
            circles = np.array(refined_result).reshape(-1, 3)
            constraints = compute_constraints(circles)
            if np.all(constraints >= -1e-6):
                return circles
            
        # Fallback to the best result found so far or initial solution
        if best_result is not None:
            return np.array(best_result).reshape(-1, 3)
        else:
            return initial_circles
    
    # Execute main optimization
    try:
        # Run the optimized approach
        circles = optimized_approach()
        
        # Final validation and additional optimization if needed
        constraints = compute_constraints(circles)
        if np.any(constraints < -1e-6):
            # If constraints still violated, try one final optimization
            try:
                circles_flat = circles.flatten()
                bounds = []
                for i in range(n):
                    bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                
                result = minimize(
                    objective,
                    circles_flat,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-10}
                )
                
                if result.success:
                    circles = result.x.reshape(-1, 3)
            except:
                pass
        
        return circles
        
    except Exception as e:
        # Fallback to simple initialization and basic optimization
        circles = initialize_better()
        try:
            circles_flat = circles.flatten()
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
                
            result = minimize(
                objective,
                circles_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-10}
            )
            
            if result.success:
                circles_optimized = result.x.reshape(-1, 3)
                return circles_optimized
        except Exception as e2:
            pass
            
        return circles


# EVOLVE-BLOCK-END
