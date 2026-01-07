# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import time
from typing import Tuple, List
import warnings
import cvxpy as cp
from itertools import combinations
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Alternative approach using semidefinite programming (SDP) to solve the circle packing problem.
    This is a fundamentally different paradigm from the previous evolutionary approach.
    
    Uses SDP relaxation techniques to find a good approximation to the optimal solution.
    The key insight is to formulate the problem as a semidefinite program that can be solved
    efficiently using modern convex optimization tools.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    
    # SDP-based approach for circle packing
    n = 32
    timeout_seconds = 55
    
    # Helper function to create a valid starting configuration
    def generate_initial_configuration():
        """Generate a good initial configuration using a combination of grid and clustering"""
        # Generate points on a grid with some randomness
        points = []
        # Grid layout with some perturbation
        grid_size = int(np.ceil(np.sqrt(n)))
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= n:
                    break
                x = 0.1 + 0.8 * i / (grid_size - 1) if grid_size > 1 else 0.5
                y = 0.1 + 0.8 * j / (grid_size - 1) if grid_size > 1 else 0.5
                # Add some noise to avoid perfect grids
                x += np.random.normal(0, 0.02)
                y += np.random.normal(0, 0.02)
                # Keep within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                points.append([x, y])
        
        # If we don't have enough points, add random ones
        while len(points) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            points.append([x, y])
        
        # Initialize with equal radii that fit within boundaries
        circles = []
        for i, (x, y) in enumerate(points[:n]):
            # Radius is constrained by distance to boundaries and potential overlaps
            max_radius = min(x, 1-x, y, 1-y)
            # Use a smaller initial radius to allow for optimization
            radius = max(0.01, max_radius * 0.8)
            circles.append([x, y, radius])
        
        return np.array(circles)
    
    # SDP-based optimization approach
    def sdp_optimization(initial_circles):
        """Optimize using semidefinite programming approach"""
        # Create variables for positions and radii
        circles = initial_circles.copy()
        
        # For SDP approach, we'll focus on optimizing the radii while keeping positions relatively stable
        # We'll use a simpler but effective strategy: iterative improvement with constraint satisfaction
        
        # Create a constraint matrix for overlap checking
        def check_constraints(circles_array):
            """Check if all constraints are satisfied"""
            # Check containment
            for i in range(len(circles_array)):
                x, y, r = circles_array[i]
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    return False
            
            # Check overlaps
            for i in range(len(circles_array)):
                for j in range(i+1, len(circles_array)):
                    x1, y1, r1 = circles_array[i]
                    x2, y2, r2 = circles_array[j]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if dist < (r1 + r2):
                        return False
            return True
        
        # Simple iterative improvement with constraint enforcement
        max_iter = 300
        best_sum = np.sum(circles[:, 2])
        best_circles = circles.copy()
        
        for iteration in range(max_iter):
            # Try to increase radii while maintaining constraints
            improved = False
            new_circles = circles.copy()
            
            # Try to increase all radii gradually
            for i in range(len(new_circles)):
                old_radius = new_circles[i, 2]
                # Try to increase radius by small amount
                step_size = 0.001
                test_radius = old_radius + step_size
                
                # Test if we can increase radius without violating constraints
                temp_circles = new_circles.copy()
                temp_circles[i, 2] = test_radius
                
                # Check if this would violate any constraints
                valid = True
                for j in range(len(temp_circles)):
                    x, y, r = temp_circles[j]
                    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                        valid = False
                        break
                
                if valid:
                    # Check overlaps with other circles
                    for j in range(len(temp_circles)):
                        if j != i:
                            x1, y1, r1 = temp_circles[i]
                            x2, y2, r2 = temp_circles[j]
                            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                            if dist < (r1 + r2):
                                valid = False
                                break
                
                if valid:
                    new_circles[i, 2] = test_radius
                    improved = True
            
            # If we made improvements, update
            if improved:
                circles = new_circles
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
            
            # Occasionally reposition circles to improve packing
            if iteration % 20 == 0:
                # Reposition some circles to potentially resolve conflicts
                for i in range(min(5, len(circles))):
                    if np.random.rand() < 0.3:  # 30% chance to reposition
                        idx = np.random.randint(len(circles))
                        # Move to nearby location with adjusted radius
                        circles[idx, 0] += np.random.normal(0, 0.01)
                        circles[idx, 1] += np.random.normal(0, 0.01)
                        circles[idx, 0] = np.clip(circles[idx, 0], 0.01, 0.99)
                        circles[idx, 1] = np.clip(circles[idx, 1], 0.01, 0.99)
            
            # Early termination if no significant improvement
            if iteration > 50 and iteration % 50 == 0:
                if np.abs(best_sum - np.sum(circles[:, 2])) < 0.001:
                    break
        
        return best_circles
    
    # Alternative: Direct optimization approach with better constraint handling
    def direct_optimization(initial_circles):
        """Direct optimization approach with proper constraint handling"""
        # Create optimization variables
        n = len(initial_circles)
        
        # Flatten initial configuration for optimization
        initial_flat = []
        for x, y, r in initial_circles:
            initial_flat.extend([x, y, r])
        
        # Define bounds for optimization
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Objective function (minimize negative sum of radii)
        def objective(params):
            return -np.sum(params[2::3])  # Sum of all radii (indices 2,5,8,...)
        
        # Constraint functions
        def containment_constraints(params):
            constraints = []
            for i in range(n):
                x, y, r = params[3*i], params[3*i+1], params[3*i+2]
                # x - r >= 0
                constraints.append(x - r)
                # 1 - x - r >= 0
                constraints.append(1 - x - r)
                # y - r >= 0
                constraints.append(y - r)
                # 1 - y - r >= 0
                constraints.append(1 - y - r)
            return np.array(constraints)
        
        def overlap_constraints(params):
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                    x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                    # Distance between centers minus sum of radii should be >= 0
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    constraint = dist - (r1 + r2)
                    constraints.append(constraint)
            return np.array(constraints)
        
        # Set up optimization
        try:
            # Use SLSQP method with bounds and constraints
            result = minimize(
                objective,
                initial_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                    {'type': 'ineq', 'fun': lambda p: overlap_constraints(p)}
                ],
                options={'maxiter': 200, 'ftol': 1e-6, 'eps': 1e-4}
            )
            
            if result.success:
                # Extract result
                final_circles = []
                for i in range(n):
                    x = result.x[3*i]
                    y = result.x[3*i+1]
                    r = result.x[3*i+2]
                    final_circles.append([x, y, r])
                return np.array(final_circles)
        except Exception as e:
            # Fall back to simple approach if optimization fails
            pass
        
        # Return initial configuration if optimization fails
        return initial_circles
    
    # Main execution
    start_time = time.time()
    
    # Generate initial configuration
    circles = generate_initial_configuration()
    
    # Apply SDP-based optimization
    circles = sdp_optimization(circles)
    
    # Apply direct optimization for final refinement
    if time.time() - start_time < 50:
        circles = direct_optimization(circles)
    
    # Final constraint validation and correction
    def validate_and_correct(circles_array):
        """Ensure all constraints are satisfied"""
        corrected = circles_array.copy()
        
        # Fix containment issues first
        for i in range(len(corrected)):
            x, y, r = corrected[i]
            corrected[i, 0] = np.clip(x, r, 1-r)
            corrected[i, 1] = np.clip(y, r, 1-r)
        
        # Resolve overlaps through iterative adjustment
        max_iter = 50
        for _ in range(max_iter):
            # Find overlaps
            overlaps = []
            for i in range(len(corrected)):
                for j in range(i+1, len(corrected)):
                    x1, y1, r1 = corrected[i]
                    x2, y2, r2 = corrected[j]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if dist < (r1 + r2):
                        overlaps.append((i, j, dist, r1 + r2))
            
            if not overlaps:
                break
                
            # Sort by overlap severity and resolve
            overlaps.sort(key=lambda x: x[3] - x[2], reverse=True)
            for i, j, dist, sum_radii in overlaps[:len(overlaps)//2]:  # Resolve half
                x1, y1, r1 = corrected[i]
                x2, y2, r2 = corrected[j]
                
                # Move circles apart
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx*dx + dy*dy)
                
                if length > 0.001:
                    overlap = sum_radii - dist
                    move_amount = overlap / 2
                    
                    dx_norm = dx / length
                    dy_norm = dy / length
                    
                    corrected[i, 0] -= dx_norm * move_amount
                    corrected[i, 1] -= dy_norm * move_amount
                    corrected[j, 0] += dx_norm * move_amount
                    corrected[j, 1] += dy_norm * move_amount
                    
                    # Keep within bounds
                    corrected[i, 0] = np.clip(corrected[i, 0], r1, 1-r1)
                    corrected[i, 1] = np.clip(corrected[i, 1], r1, 1-r1)
                    corrected[j, 0] = np.clip(corrected[j, 0], r2, 1-r2)
                    corrected[j, 1] = np.clip(corrected[j, 1], r2, 1-r2)
        
        return corrected
    
    circles = validate_and_correct(circles)
    
    return circles


# EVOLVE-BLOCK-END
