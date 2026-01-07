# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random
from scipy.spatial.distance import cdist

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize positions using a better hexagonal packing heuristic
    def initialize_hexagonal():
        # Create a more refined hexagonal grid pattern
        circles = []
        
        # For 32 circles, we'll use a 6x6 grid with offset rows but carefully place
        rows = 6
        cols = 6
        
        # Calculate spacing that allows for good circle packing
        spacing_x = 0.9 / cols  # Leave some margin
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        # Max radius based on spacing
        max_radius = min(spacing_x, spacing_y) / 2
        
        # Place circles in hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Offset every other row for hexagonal packing
                x = 0.05 + (j + (i % 2) * 0.5) * spacing_x
                y = 0.05 + i * spacing_y
                
                # Ensure we're within bounds and circle fits
                if (x - max_radius >= 0.05 and x + max_radius <= 0.95 and 
                    y - max_radius >= 0.05 and y + max_radius <= 0.95):
                    circles.append([x, y, max_radius])
                    
        # Fill remaining spots with random positions but still respecting spacing
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            # Give a reasonable initial radius
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Efficient constraint checking using spatial acceleration
    def create_constraints(params):
        """Create constraint functions for scipy optimization"""
        circles = params.reshape(-1, 3)
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        def boundary_constraints(params):
            circles = params.reshape(-1, 3)
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # For each circle, check containment constraints
            constraints = []
            for i in range(len(circles)):
                x, y, r = circles[i]
                # Circle must be within bounds: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
                constraints.extend([
                    x - r,              # x-r >= 0
                    1 - x - r,          # 1-x-r >= 0
                    y - r,              # y-r >= 0
                    1 - y - r           # 1-y-r >= 0
                ])
            return np.array(constraints)
        
        def overlap_constraints(params):
            circles = params.reshape(-1, 3)
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            constraints = []
            # Check all pairs for overlap constraints
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (radii[i] + radii[j])**2
                    
                    # We want dist_sq >= min_dist_sq (non-overlap constraint)
                    # So we want: dist_sq - min_dist_sq >= 0
                    constraints.append(dist_sq - min_dist_sq)
                    
            return np.array(constraints)
        
        return boundary_constraints, overlap_constraints
    
    # Compute objective function (negative because we want to maximize)
    def objective(params):
        circles = params.reshape(-1, 3)
        # Sum of radii (we want to maximize this, so return negative)
        return -np.sum(circles[:, 2])
    
    # Improved post-processing to enhance solution quality
    def post_process(circles):
        # Simple local optimization to slightly improve the solution
        improved = True
        iterations = 0
        while improved and iterations < 10:
            improved = False
            iterations += 1
            
            # Try small adjustments to each circle
            for i in range(len(circles)):
                old_x, old_y, old_r = circles[i]
                best_x, best_y, best_r = old_x, old_y, old_r
                best_sum = np.sum(circles[:, 2])
                
                # Try small movements in 8 directions
                moves = [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001),
                         (0.0005, 0.0005), (-0.0005, -0.0005), (0.0005, -0.0005), (-0.0005, 0.0005)]
                
                for dx, dy in moves:
                    new_x = old_x + dx
                    new_y = old_y + dy
                    
                    # Check bounds
                    if (new_x - old_r < 0 or new_x + old_r > 1 or 
                        new_y - old_r < 0 or new_y + old_r > 1):
                        continue
                    
                    # Temporarily adjust position
                    circles[i, 0] = new_x
                    circles[i, 1] = new_y
                    
                    # Check if still valid (no overlaps)
                    valid = True
                    for j in range(len(circles)):
                        if i != j:
                            dx_ij = circles[i, 0] - circles[j, 0]
                            dy_ij = circles[i, 1] - circles[j, 1]
                            dist_sq = dx_ij*dx_ij + dy_ij*dy_ij
                            min_dist_sq = (circles[i, 2] + circles[j, 2])**2
                            if dist_sq < min_dist_sq:
                                valid = False
                                break
                    
                    if valid:
                        current_sum = np.sum(circles[:, 2])
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_x, best_y = new_x, new_y
                            improved = True
                    
                    # Restore original position
                    circles[i, 0] = old_x
                    circles[i, 1] = old_y
                
                # Apply best adjustment if found
                circles[i, 0] = best_x
                circles[i, 1] = best_y
        
        return circles
    
    # Initialize with hexagonal pattern
    initial_circles = initialize_hexagonal()
    
    # Extract initial parameters (x, y, r) for each circle
    initial_params = initial_circles.flatten()
    
    # Optimize using scipy's minimize with constraints
    try:
        # Define bounds for x, y positions and radii
        bounds = []
        for i in range(n):
            # Bounds for x and y positions (slightly inside square to ensure containment)
            bounds.extend([(0.001, 0.999), (0.001, 0.999)])
            # Bounds for radii (must be positive and reasonable)
            bounds.append((0.001, 0.499))
        
        # Define constraints using the helper function
        def boundary_constraints(params):
            circles = params.reshape(-1, 3)
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            constraints = []
            for i in range(len(circles)):
                x, y, r = circles[i]
                # Circle must be within bounds: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
                constraints.extend([
                    x - r,              # x-r >= 0
                    1 - x - r,          # 1-x-r >= 0
                    y - r,              # y-r >= 0
                    1 - y - r           # 1-y-r >= 0
                ])
            return np.array(constraints)
        
        def overlap_constraints(params):
            circles = params.reshape(-1, 3)
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            constraints = []
            # Check all pairs for overlap constraints
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (radii[i] + radii[j])**2
                    
                    # We want dist_sq >= min_dist_sq (non-overlap constraint)
                    constraints.append(dist_sq - min_dist_sq)
                    
            return np.array(constraints)
        
        # Define constraints
        cons = [
            {'type': 'ineq', 'fun': boundary_constraints},
            {'type': 'ineq', 'fun': overlap_constraints}
        ]
        
        # Optimize with multiple attempts for better results
        best_result = None
        best_sum = -float('inf')
        
        # Try multiple optimization runs with different initializations
        for attempt in range(3):  # Reduced attempts for faster execution
            # Slightly perturb initial solution for diversity
            perturbed_params = initial_params.copy()
            if attempt > 0:
                # Add small random perturbations
                for i in range(0, len(perturbed_params), 3):
                    perturbed_params[i] += random.uniform(-0.01, 0.01)  # x
                    perturbed_params[i+1] += random.uniform(-0.01, 0.01)  # y
                    perturbed_params[i+2] += random.uniform(-0.005, 0.005)  # r
            
            # Try different optimization methods
            methods = ['SLSQP']  # Simplified to just one method for speed
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        perturbed_params,
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 200, 'ftol': 1e-6, 'eps': 1e-6}
                    )
                    
                    if result.success:
                        current_sum = -objective(result.x)
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_result = result
                except:
                    continue  # Skip failed optimization attempts
        
        if best_result is not None and best_result.success:
            optimized_circles = best_result.x.reshape(-1, 3)
            # Apply post-processing to fine-tune
            final_circles = post_process(optimized_circles)
            return final_circles
        else:
            # If optimization fails, return the initial configuration with post-processing
            return post_process(initial_circles)
            
    except Exception as e:
        # Fallback to initial configuration with post-processing if optimization fails
        return post_process(initial_circles)


# EVOLVE-BLOCK-END
