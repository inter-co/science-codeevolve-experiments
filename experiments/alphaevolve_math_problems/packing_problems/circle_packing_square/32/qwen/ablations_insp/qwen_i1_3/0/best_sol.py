# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import time
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: hexagonal grid initialization + constrained optimization + refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using hexagonal grid pattern with better spacing based on inspiration
    def initialize_hexagonal_grid():
        # Create a hexagonal grid pattern that fits in the unit square
        # Using a 6x6 grid for 32 circles (better distribution)
        rows = 6
        cols = 6
        
        # Hexagon parameters optimized for 32 circles
        side_length = 0.18  # Adjusted for better packing density
        hex_height = side_length * math.sqrt(3) / 2
        hex_width = side_length
        
        # Generate hexagonal grid points
        points = []
        for i in range(rows):
            for j in range(cols):
                x = (j + 0.5 * (i % 2)) * hex_width + 0.05  # Add offset to center
                y = i * hex_height + 0.05
                if x <= 0.95 and y <= 0.95:  # Leave margin for boundary constraints
                    points.append([x, y])
        
        # Take first 32 points
        points = points[:n]
        
        # Initialize with varying small radii for better distribution
        circles = np.zeros((n, 3))
        for i, (x, y) in enumerate(points):
            # Start with slightly different initial radii to avoid symmetry issues
            base_radius = 0.03 + (i % 4) * 0.002  # Slightly larger starting radius
            circles[i] = [x, y, base_radius]
            
        return circles
    
    # More robust constraint handling
    def create_constraints():
        """Create constraint functions for scipy optimization."""
        cons = []
        
        # Boundary constraints: radius <= x <= 1-radius and radius <= y <= 1-radius
        for i in range(n):
            # x >= r
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: x[3*i] - x[3*i+2] - 1e-6
            })
            # y >= r  
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: x[3*i+1] - x[3*i+2] - 1e-6
            })
            # 1-x >= r
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2] - 1e-6
            })
            # 1-y >= r
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2] - 1e-6
            })
        
        # Non-overlap constraints: sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        for i in range(n):
            for j in range(i+1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (
                        np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) 
                        - x[3*i+2] - x[3*j+2] - 1e-6
                    )
                })
        
        return cons
    
    # Objective function (negative because we want to maximize sum of radii)
    def objective(circles_flat):
        return -np.sum(circles_flat[2::3])  # Negative because minimize
    
    # Initialize
    circles = initialize_hexagonal_grid()
    
    # Flatten for optimization
    initial_guess = circles.flatten()
    
    # Define bounds for variables (x, y, r for each circle)
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
    
    # Create constraints
    constraints = create_constraints()
    
    # Run optimization with bounds using SLSQP (more robust than L-BFGS-B for constrained problems)
    try:
        result = minimize(
            objective, 
            initial_guess, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
        else:
            # Fallback to initial configuration if optimization fails
            optimized_circles = circles
            
    except Exception as e:
        # If optimization fails, return initial configuration
        optimized_circles = circles
    
    # Final validation and refinement with more comprehensive local search
    def validate_and_refine(circles_array):
        """Validate constraints and refine the solution."""
        # First ensure all circles are within bounds
        for i in range(n):
            x, y, r = circles_array[i]
            # Clamp values to valid ranges
            x = np.clip(x, r + 1e-6, 1 - r - 1e-6)
            y = np.clip(y, r + 1e-6, 1 - r - 1e-6)
            r = np.clip(r, 0.001, min(x, 1-x, y, 1-y) - 1e-6)
            circles_array[i] = [x, y, r]
        
        # Comprehensive local search improvement with better neighborhood exploration
        improved = True
        max_iterations = 100  # Increased iterations for better optimization
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try to improve each circle systematically
            for i in range(n):
                current_circle = circles_array[i].copy()
                best_x, best_y, best_r = current_circle[0], current_circle[1], current_circle[2]
                best_sum = np.sum(circles_array[:, 2])
                
                # Generate candidate moves with more thorough exploration
                candidates = []
                
                # Radius increases
                candidates.extend([(0, 0, 0.001), (0, 0, 0.002), (0, 0, 0.003)])
                
                # Position moves - more diverse
                candidates.extend([
                    (-0.005, 0, 0), (0.005, 0, 0), (0, -0.005, 0), (0, 0.005, 0),
                    (-0.003, -0.003, 0.001), (-0.003, 0.003, 0.001), 
                    (0.003, -0.003, 0.001), (0.003, 0.003, 0.001),
                    (-0.01, 0, 0), (0.01, 0, 0), (0, -0.01, 0), (0, 0.01, 0),
                    (-0.002, 0, 0.001), (0.002, 0, 0.001), (0, -0.002, 0.001), (0, 0.002, 0.001)
                ])
                
                # Random perturbations for diversity - increased number
                for _ in range(15):
                    dx = random.uniform(-0.005, 0.005)
                    dy = random.uniform(-0.005, 0.005)
                    dr = random.uniform(-0.002, 0.002)
                    candidates.append((dx, dy, dr))
                
                for dx, dy, dr in candidates:
                    new_x = current_circle[0] + dx
                    new_y = current_circle[1] + dy
                    new_r = current_circle[2] + dr
                    
                    # Check bounds and validity
                    if (0 <= new_x <= 1 and 0 <= new_y <= 1 and 
                        new_r > 0.001 and new_r <= 0.5 and
                        new_x - new_r >= 0 and new_x + new_r <= 1 and
                        new_y - new_r >= 0 and new_y + new_r <= 1):
                        
                        # Check overlap with all other circles
                        valid = True
                        test_circles = circles_array.copy()
                        test_circles[i] = [new_x, new_y, new_r]
                        
                        for j in range(n):
                            if i != j:
                                dist = math.sqrt((new_x - circles_array[j, 0])**2 + 
                                                (new_y - circles_array[j, 1])**2)
                                if dist < (new_r + circles_array[j, 2]):
                                    valid = False
                                    break
                        
                        if valid:
                            # Calculate new sum
                            new_sum = np.sum(test_circles[:, 2])
                            if new_sum > best_sum + 1e-8:
                                best_sum = new_sum
                                best_x, best_y, best_r = new_x, new_y, new_r
                                improved = True
                
                circles_array[i] = [best_x, best_y, best_r]
        
        return circles_array
    
    # Apply final refinement
    refined_circles = validate_and_refine(optimized_circles.copy())
    
    # Final check for constraints
    def check_constraints(circles_array):
        """Check if all circles satisfy containment and non-overlap constraints."""
        # Check containment
        for i in range(n):
            x, y, r = circles_array[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check non-overlap
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < r1 + r2:
                    return False
        return True
    
    if not check_constraints(refined_circles):
        # If constraints violated, use initial configuration as final fallback
        refined_circles = circles
    
    return refined_circles


# EVOLVE-BLOCK-END
