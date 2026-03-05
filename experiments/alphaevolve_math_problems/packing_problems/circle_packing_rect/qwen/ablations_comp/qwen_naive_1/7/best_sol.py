# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import random
import time
from scipy.spatial.distance import cdist
import math
from scipy.spatial import cKDTree
import itertools

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining evolutionary optimization with local refinement and improved initialization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    # Rectangle dimensions: width + height = 2 (perimeter = 4)
    # Try different width/height ratios to find optimal configuration
    best_ratio = 1.0  # Start with square
    width = 1.0
    height = 1.0
    
    # Improved initialization using hexagonal packing pattern
    def initialize_hexagonal_packing():
        circles = []
        
        # Try different rectangle aspect ratios
        ratios = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]
        best_config = None
        best_sum = 0
        
        for ratio in ratios:
            test_width = 2.0 / (1 + ratio)
            test_height = test_width * ratio
            
            # Create hexagonal packing
            circles_test = []
            max_radius = min(test_width, test_height) * 0.3
            
            # Hexagonal lattice parameters
            hex_radius = max_radius * 0.8
            hex_spacing_x = hex_radius * 2
            hex_spacing_y = hex_radius * math.sqrt(3)
            
            rows = int(test_height / hex_spacing_y) + 2
            cols = int(test_width / hex_spacing_x) + 2
            
            placed = 0
            for i in range(rows):
                for j in range(cols):
                    if placed >= n:
                        break
                        
                    # Offset odd rows
                    x = (j + 0.5) * hex_spacing_x
                    if i % 2 == 1:
                        x += hex_spacing_x / 2
                        
                    y = (i + 0.5) * hex_spacing_y
                    
                    # Ensure within bounds
                    if (x >= hex_radius and x <= test_width - hex_radius and
                        y >= hex_radius and y <= test_height - hex_radius):
                        
                        circles_test.append([x, y, hex_radius])
                        placed += 1
                        
                if placed >= n:
                    break
            
            # Fill remaining with random placement
            while len(circles_test) < n:
                x = random.uniform(hex_radius, test_width - hex_radius)
                y = random.uniform(hex_radius, test_height - hex_radius)
                circles_test.append([x, y, hex_radius * 0.8])
            
            sum_radii = sum(circle[2] for circle in circles_test)
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_config = (circles_test, test_width, test_height)
        
        if best_config:
            circles, test_width, test_height = best_config
            width, height = test_width, test_height
        else:
            # Fallback to basic initialization
            width, height = 1.0, 1.0
            max_radius = 0.15
            circles = [[random.uniform(max_radius, width-max_radius), 
                       random.uniform(max_radius, height-max_radius), 
                       max_radius] for _ in range(n)]
        
        return np.array(circles), width, height
    
    # Initialize with better configuration
    circles, width, height = initialize_hexagonal_packing()
    
    # More efficient constraint checking using spatial data structures
    def constraint_func(params):
        """Constraint function returning positive values when satisfied"""
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Non-overlap constraints using KDTree for efficiency
        constraints = []
        
        # Build KDTree for faster neighbor search
        points = positions
        tree = cKDTree(points)
        
        # Check all pairs efficiently
        pairs = list(itertools.combinations(range(n), 2))
        for i, j in pairs:
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            distance = math.sqrt(dx*dx + dy*dy)
            min_distance = radii[i] + radii[j]
            # Constraint should be positive when satisfied (distance >= min_distance)
            constraints.append(distance - min_distance)
        
        # Boundary constraints (positive when satisfied)
        for i in range(n):
            # Left boundary
            constraints.append(positions[i][0] - radii[i])
            # Right boundary  
            constraints.append(width - positions[i][0] - radii[i])
            # Bottom boundary
            constraints.append(positions[i][1] - radii[i])
            # Top boundary
            constraints.append(height - positions[i][1] - radii[i])
            
        return np.array(constraints)
    
    # Optimized objective function
    def objective(params):
        """Minimize negative sum of radii (maximize sum of radii)"""
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        # We want to maximize sum of radii, so minimize negative sum
        return -np.sum(radii)
    
    # Create initial parameter vector: [x1, y1, x2, y2, ..., xn, yn, r1, r2, ..., rn]
    initial_params = np.concatenate([
        circles[:, :2].flatten(),  # Positions
        circles[:, 2]              # Radii
    ])
    
    # Set bounds for positions and radii
    bounds = [(0, width) for _ in range(2*n)] + [(1e-6, width/2) for _ in range(n)]
    
    # Define constraints
    cons = {
        'type': 'ineq',  # Inequality constraints (g(x) >= 0)
        'fun': constraint_func
    }
    
    # Try different optimization approaches with better parameter tuning
    best_result = None
    best_sum = -float('inf')
    
    # Try differential evolution first for global search
    try:
        # More aggressive differential evolution settings
        de_result = differential_evolution(
            objective,
            bounds,
            args=(),
            maxiter=100,
            popsize=20,
            mutation=(0.8, 1),
            recombination=0.7,
            seed=42,
            disp=False,
            atol=1e-6,
            rtol=1e-6
        )
        
        if de_result.success:
            # Evaluate the result from DE
            de_positions = de_result.x[:-n].reshape(-1, 2)
            de_radii = de_result.x[-n:]
            de_sum = np.sum(de_radii)
            
            if de_sum > best_sum:
                best_sum = de_sum
                best_result = de_result
                
    except Exception as e:
        pass
    
    # Then try local optimization from the best found solution
    if best_result is not None:
        try:
            # Refine with local optimization
            refined_result = minimize(
                objective,
                best_result.x,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if refined_result.success:
                final_positions = refined_result.x[:-n].reshape(-1, 2)
                final_radii = refined_result.x[-n:]
                final_sum = np.sum(final_radii)
                
                if final_sum > best_sum:
                    best_sum = final_sum
                    circles[:, 0] = final_positions[:, 0]
                    circles[:, 1] = final_positions[:, 1]
                    circles[:, 2] = final_radii
                    
        except Exception as e:
            pass
    else:
        # Fall back to direct local optimization with better settings
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result.success:
                final_positions = result.x[:-n].reshape(-1, 2)
                final_radii = result.x[-n:]
                
                # Update circles array with optimized values
                circles[:, 0] = final_positions[:, 0]
                circles[:, 1] = final_positions[:, 1]
                circles[:, 2] = final_radii
                
                # Ensure all radii are positive
                circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
                
        except Exception as e:
            pass
    
    # Final validation and cleanup
    # Ensure all circles are within bounds
    for i in range(n):
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], width - circles[i, 2])
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], height - circles[i, 2])
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
