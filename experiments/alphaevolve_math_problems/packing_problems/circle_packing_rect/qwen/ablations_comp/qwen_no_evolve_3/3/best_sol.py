# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Voronoi
import random
from typing import Tuple
import time
from itertools import combinations
import warnings

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a constraint satisfaction approach with systematic grid search and mathematical optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Optimal rectangle dimensions - we'll optimize this too
    rect_width = 1.2
    rect_height = 0.8
    
    # Number of circles
    n = 21
    
    def generate_grid_points(width, height, n_circles):
        """Generate initial circle positions using a grid-based approach"""
        # Determine grid dimensions
        grid_rows = int(np.ceil(np.sqrt(n_circles)))
        grid_cols = int(np.ceil(n_circles / grid_rows))
        
        # Create grid points
        x_spacing = width / (grid_cols + 1)
        y_spacing = height / (grid_rows + 1)
        
        points = []
        for i in range(grid_rows):
            for j in range(grid_cols):
                if len(points) >= n_circles:
                    break
                x = (j + 1) * x_spacing
                y = (i + 1) * y_spacing
                points.append([x, y])
        
        # Fill remaining positions randomly
        while len(points) < n_circles:
            x = np.random.uniform(0.1, width - 0.1)
            y = np.random.uniform(0.1, height - 0.1)
            points.append([x, y])
            
        return np.array(points[:n_circles])
    
    def calculate_initial_radii(points, width, height):
        """Calculate initial radii based on distance to nearest neighbors"""
        n = len(points)
        radii = np.zeros(n)
        
        # For each point, calculate the minimum distance to others
        for i in range(n):
            min_dist = float('inf')
            for j in range(n):
                if i != j:
                    dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                    min_dist = min(min_dist, dist)
            
            # Set radius to a fraction of the minimum distance, bounded by container size
            r = min(min_dist / 3, width * 0.2, height * 0.2)
            radii[i] = max(0.01, min(r, 0.3))
            
        return radii
    
    def setup_constraint_bounds(width, height, n_circles):
        """Set up bounds for optimization variables"""
        # Each circle has 3 parameters: x, y, radius
        # Bounds: x in [radius, width-radius], y in [radius, height-radius], radius in [0.01, min(width,height)/2]
        bounds = []
        for i in range(n_circles):
            # x coordinate bounds
            bounds.extend([(0.01, width - 0.01), (0.01, height - 0.01), (0.01, min(width, height) / 2)])
        return bounds
    
    def objective_function(params, width, height, n_circles):
        """Objective function to maximize sum of radii"""
        # Extract x, y, radius values for each circle
        total_radius = 0
        for i in range(n_circles):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            total_radius += r
            
        # We want to maximize the sum of radii, so return negative
        return -total_radius
    
    def constraint_overlap(params, width, height, n_circles):
        """Constraint function ensuring no overlaps"""
        # Extract positions and radii
        positions = []
        radii = []
        for i in range(n_circles):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            positions.append([x, y])
            radii.append(r)
            
        # Check all pairs of circles for overlap
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                pos_i = positions[i]
                pos_j = positions[j]
                r_i = radii[i]
                r_j = radii[j]
                
                # Distance between centers
                dist = np.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                # Minimum distance to avoid overlap
                min_dist = r_i + r_j
                
                # Constraint violation: distance < min_dist (we want dist >= min_dist)
                # So return value should be <= 0 when valid
                yield dist - min_dist
    
    def constraint_bounds(params, width, height, n_circles):
        """Constraint function ensuring all circles are within bounds"""
        for i in range(n_circles):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            
            # Circle must be within bounds
            yield x - r  # x >= r
            yield y - r  # y >= r
            yield width - x - r  # width >= x + r
            yield height - y - r  # height >= y + r
    
    def solve_with_differential_evolution(width, height, n_circles):
        """Use differential evolution for global optimization"""
        # Generate initial population
        initial_points = generate_grid_points(width, height, n_circles)
        initial_radii = calculate_initial_radii(initial_points, width, height)
        
        # Initialize parameters
        initial_params = []
        for i in range(n_circles):
            x, y = initial_points[i]
            r = initial_radii[i]
            initial_params.extend([x, y, r])
        
        # Set up bounds
        bounds = setup_constraint_bounds(width, height, n_circles)
        
        # Define constraints
        constraints = []
        
        # Add overlap constraints
        def overlap_constraint(params):
            results = list(constraint_overlap(params, width, height, n_circles))
            return results
        
        # Add boundary constraints
        def bound_constraint(params):
            results = list(constraint_bounds(params, width, height, n_circles))
            return results
            
        # Add both constraints
        constraints.append({'type': 'ineq', 'fun': overlap_constraint})
        constraints.append({'type': 'ineq', 'fun': bound_constraint})
        
        # Run optimization
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                args=(width, height, n_circles),
                constraints=constraints,
                seed=42,
                maxiter=100,
                popsize=15,
                disp=False,
                polish=True
            )
            
            if result.success:
                return result.x
        except Exception as e:
            warnings.warn(f"Differential evolution failed: {e}")
            
        # Return initial guess if optimization fails
        return initial_params
    
    def refine_solution(params, width, height, n_circles):
        """Refine solution using local optimization"""
        # Convert to array for easier manipulation
        solution = np.array(params)
        
        # Extract and validate the solution
        circles = []
        for i in range(n_circles):
            x = max(0.01, min(width - 0.01, solution[3*i]))
            y = max(0.01, min(height - 0.01, solution[3*i + 1]))
            r = max(0.01, min(min(width, height)/2, solution[3*i + 2]))
            circles.append([x, y, r])
            
        # Check validity and fix if needed
        circles = np.array(circles)
        
        # Perform local optimization using L-BFGS
        bounds = setup_constraint_bounds(width, height, n_circles)
        
        # Optimization using scipy minimize
        def obj_func(x):
            return -np.sum(x[2::3])  # Maximize sum of radii
            
        def constraint_func(x):
            # Check constraints manually for scipy
            positions = []
            radii = []
            for i in range(n_circles):
                positions.append([x[3*i], x[3*i+1]])
                radii.append(x[3*i+2])
                
            # Check all overlap constraints
            violations = []
            for i in range(n_circles):
                for j in range(i+1, n_circles):
                    dist = np.sqrt((positions[i][0]-positions[j][0])**2 + (positions[i][1]-positions[j][1])**2)
                    min_dist = radii[i] + radii[j]
                    violations.append(dist - min_dist)  # Should be >= 0
                    
            # Check boundary constraints
            for i in range(n_circles):
                x_pos, y_pos, r = x[3*i], x[3*i+1], x[3*i+2]
                violations.append(x_pos - r)  # x >= r
                violations.append(y_pos - r)  # y >= r
                violations.append(width - x_pos - r)  # width >= x + r
                violations.append(height - y_pos - r)  # height >= y + r
                
            return violations
            
        # Create constraint dictionary for scipy
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        try:
            result = minimize(obj_func, solution, method='SLSQP', bounds=bounds, constraints=cons, 
                            options={'maxiter': 100, 'ftol': 1e-6})
            if result.success:
                return result.x
        except Exception as e:
            warnings.warn(f"Local optimization failed: {e}")
            
        return solution
    
    def create_final_circles(params, width, height, n_circles):
        """Convert optimization parameters to final circle array"""
        circles = np.zeros((n_circles, 3))
        for i in range(n_circles):
            x = max(0.01, min(width - 0.01, params[3*i]))
            y = max(0.01, min(height - 0.01, params[3*i + 1]))
            r = max(0.01, min(min(width, height)/2, params[3*i + 2]))
            circles[i] = [x, y, r]
        return circles
    
    # Main optimization process
    # Try several different rectangle dimensions to find optimal aspect ratio
    best_sum = 0
    best_circles = None
    
    # Test different aspect ratios
    aspect_ratios = [(1.0, 1.0), (1.2, 0.8), (0.8, 1.2), (1.5, 0.5), (0.5, 1.5)]
    
    for w, h in aspect_ratios:
        # Scale to perimeter = 4
        scale_factor = 2 / (w + h)
        width = w * scale_factor
        height = h * scale_factor
        
        try:
            # Solve using differential evolution
            params = solve_with_differential_evolution(width, height, n)
            
            # Refine solution
            refined_params = refine_solution(params, width, height, n)
            
            # Create final circles
            circles = create_final_circles(refined_params, width, height, n)
            
            # Validate
            total_radius = np.sum(circles[:, 2])
            if total_radius > best_sum:
                best_sum = total_radius
                best_circles = circles.copy()
                
        except Exception as e:
            warnings.warn(f"Failed for aspect ratio ({w}, {h}): {e}")
            continue
    
    # If no good solution found, use fallback
    if best_circles is None:
        # Use simple grid initialization
        points = generate_grid_points(rect_width, rect_height, n)
        radii = calculate_initial_radii(points, rect_width, rect_height)
        
        best_circles = np.column_stack([points, radii])
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
