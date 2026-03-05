# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a robust hybrid approach combining geometric initialization with advanced optimization techniques.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    width = 1.0
    height = 1.0
    
    n = 21
    
    # Vectorized constraint function - key improvement from inspirations
    def constraint_func(params):
        """Vectorized constraint function for scipy optimization - highly efficient"""
        # Reshape parameters for vectorized operations
        positions_x = params[::3]
        positions_y = params[1::3]
        radii = params[2::3]
        
        # Boundary constraints (positive when satisfied)
        constraints = []
        
        # Left, bottom, right, top boundary constraints
        constraints.extend(positions_x - radii)  # left boundary
        constraints.extend(positions_y - radii)  # bottom boundary
        constraints.extend(width - positions_x - radii)  # right boundary
        constraints.extend(height - positions_y - radii)  # top boundary
        
        # Circle-to-circle constraints (positive when satisfied)
        # Vectorized computation for better performance
        pos_array = np.column_stack([positions_x, positions_y])
        distances = cdist(pos_array, pos_array)
        
        # Only check upper triangle to avoid duplicates
        for i in range(n):
            for j in range(i+1, n):
                distance = distances[i, j]
                min_distance = radii[i] + radii[j]
                constraints.append(distance - min_distance)
                
        return np.array(constraints)
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        return -np.sum(params[2::3])  # Sum of radii (negative for maximization)
    
    # Multiple initialization strategies - crucial for success
    def generate_hexagonal_initial():
        """Generate initial configuration using hexagonal packing pattern"""
        circles = []
        
        # Create hexagonal pattern optimized for 21 circles
        rows = 5
        cols = 5
        
        # Calculate hexagonal spacing carefully
        hex_radius = 0.12  # Carefully tuned initial radius
        cell_width = hex_radius * 2
        cell_height = hex_radius * np.sqrt(3)
        
        # Create hexagonal pattern with strategic placement
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Offset every other row for hexagonal packing
                x_offset = (i % 2) * 0.5
                x = (j + x_offset) * cell_width + hex_radius
                y = i * cell_height + hex_radius
                
                # Add jitter to avoid perfect alignment
                x += random.uniform(-cell_width/8, cell_width/8)
                y += random.uniform(-cell_height/8, cell_height/8)
                
                # Ensure within bounds
                x = np.clip(x, hex_radius, width - hex_radius)
                y = np.clip(y, hex_radius, height - hex_radius)
                
                circles.append([x, y, hex_radius])
                
        # Fill remaining slots with random positions
        while len(circles) < n:
            circles.append([
                random.uniform(hex_radius, width - hex_radius),
                random.uniform(hex_radius, height - hex_radius),
                hex_radius * 0.8
            ])
            
        return np.array(circles[:n])
    
    def generate_fibonacci_initial():
        """Generate initial configuration using Fibonacci spiral pattern"""
        circles = []
        
        # Use Fibonacci spiral for even distribution
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            # Fibonacci spiral placement
            theta = i * 2 * np.pi / golden_ratio
            radius = i / np.sqrt(n)
            
            # Convert to Cartesian coordinates within rectangle
            x = width * 0.5 + radius * np.cos(theta) * (width * 0.4)
            y = height * 0.5 + radius * np.sin(theta) * (height * 0.4)
            
            # Add randomness for better distribution
            x += random.uniform(-0.03, 0.03)
            y += random.uniform(-0.03, 0.03)
            
            # Keep within bounds
            x = np.clip(x, 0.05, width - 0.05)
            y = np.clip(y, 0.05, height - 0.05)
            
            circles.append([x, y, 0.07])
            
        return np.array(circles)
    
    def generate_grid_initial():
        """Generate initial configuration using optimized grid pattern"""
        circles = []
        
        # Try 4x6 grid pattern for 21 circles with better spacing
        rows = 4
        cols = 6
        
        cell_width = width / cols
        cell_height = height / rows
        
        # Create positions with strategic placement
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Position with jitter for better distribution
                x = (j + 0.5) * cell_width + random.uniform(-cell_width/10, cell_width/10)
                y = (i + 0.5) * cell_height + random.uniform(-cell_height/10, cell_height/10)
                
                # Keep within bounds
                x = np.clip(x, 0.05, width - 0.05)
                y = np.clip(y, 0.05, height - 0.05)
                
                circles.append([x, y, 0.08])
                
        # Trim to exactly n circles
        circles = circles[:n]
        
        # Fill any gaps
        while len(circles) < n:
            circles.append([
                random.uniform(0.05, width - 0.05),
                random.uniform(0.05, height - 0.05),
                0.07
            ])
            
        return np.array(circles)
    
    # Enhanced refinement approach - critical for final improvement
    def refine_solution(circles):
        """Apply iterative refinement to improve solution quality"""
        refined = circles.copy()
        
        # Multiple refinement passes with more aggressive improvement
        for pass_num in range(30):  # More passes for better improvement
            improved = False
            
            # Try to increase radii systematically
            for i in range(n):
                old_x, old_y, old_r = refined[i]
                
                # Find minimum distance to any other circle
                min_distance = float('inf')
                for j in range(n):
                    if i != j:
                        dx = refined[j][0] - old_x
                        dy = refined[j][1] - old_y
                        distance = np.sqrt(dx*dx + dy*dy)
                        min_distance = min(min_distance, distance)
                
                # Calculate maximum possible radius
                max_radius = min_distance / 2.0 - 0.001  # Safety margin
                
                # Respect boundary constraints
                boundary_radius = min(
                    old_x, 
                    width - old_x, 
                    old_y, 
                    height - old_y
                ) - 0.001
                
                max_radius = min(max_radius, boundary_radius, 0.3)
                
                if max_radius > old_r and max_radius > 0.001:
                    # Try to increase radius with more aggressive factor
                    test_radius = min(old_r * 1.05, max_radius)  # Moderate increase
                    
                    # Check if this works without overlap
                    valid = True
                    for j in range(n):
                        if i != j:
                            dx = refined[j][0] - old_x
                            dy = refined[j][1] - old_y
                            distance = np.sqrt(dx*dx + dy*dy)
                            if distance < (test_radius + refined[j][2]):
                                valid = False
                                break
                    
                    if valid:
                        refined[i] = [old_x, old_y, test_radius]
                        improved = True
                        
            if not improved:
                break
                
        return refined
    
    # Main optimization with multiple strategies - key to success
    best_result = None
    best_sum = -np.inf
    
    # Strategy 1: Hexagonal grid initialization
    try:
        circles = generate_hexagonal_initial()
        initial_params = circles.flatten()
        
        # Optimization bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, width - 0.001), (0.001, height - 0.001), (0.001, 0.3)])
        
        # Define constraint dictionary for scipy.optimize
        cons = {
            'type': 'ineq',
            'fun': constraint_func
        }
        
        # Run optimization with trust-constr - highest precision method
        result = minimize(
            objective,
            initial_params,
            method='trust-constr',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}  # Tighter tolerances
        )
        
        if result.success:
            final_params = result.x
            current_sum = -objective(final_params)
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = np.zeros((n, 3))
                for i in range(n):
                    best_circles[i] = final_params[3*i:3*i+3]
                best_result = best_circles.copy()
                
    except Exception:
        pass
    
    # Strategy 2: Fibonacci spiral initialization
    try:
        circles = generate_fibonacci_initial()
        initial_params = circles.flatten()
        
        # Optimization bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, width - 0.001), (0.001, height - 0.001), (0.001, 0.3)])
        
        # Define constraint dictionary for scipy.optimize
        cons = {
            'type': 'ineq',
            'fun': constraint_func
        }
        
        # Run optimization with trust-constr
        result = minimize(
            objective,
            initial_params,
            method='trust-constr',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result.success:
            final_params = result.x
            current_sum = -objective(final_params)
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = np.zeros((n, 3))
                for i in range(n):
                    best_circles[i] = final_params[3*i:3*i+3]
                best_result = best_circles.copy()
                
    except Exception:
        pass
    
    # Strategy 3: Grid initialization
    try:
        circles = generate_grid_initial()
        initial_params = circles.flatten()
        
        # Optimization bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, width - 0.001), (0.001, height - 0.001), (0.001, 0.3)])
        
        # Define constraint dictionary for scipy.optimize
        cons = {
            'type': 'ineq',
            'fun': constraint_func
        }
        
        # Run optimization with trust-constr
        result = minimize(
            objective,
            initial_params,
            method='trust-constr',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result.success:
            final_params = result.x
            current_sum = -objective(final_params)
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = np.zeros((n, 3))
                for i in range(n):
                    best_circles[i] = final_params[3*i:3*i+3]
                best_result = best_circles.copy()
                
    except Exception:
        pass
    
    # Strategy 4: Multiple restarts with enhanced random initialization
    for restart in range(20):  # More restarts for better exploration
        try:
            # Enhanced random initialization with better spread
            circles = []
            for i in range(n):
                # Use a more structured approach to random initialization
                x = random.uniform(0.05, width - 0.05)
                y = random.uniform(0.05, height - 0.05)
                # Start with a smaller radius to allow for expansion
                r = random.uniform(0.03, 0.15)
                circles.append([x, y, r])
            
            initial_params = np.array(circles).flatten()
            
            # Optimization bounds
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, width - 0.001), (0.001, height - 0.001), (0.001, 0.3)])
            
            # Define constraint dictionary for scipy.optimize
            cons = {
                'type': 'ineq',
                'fun': constraint_func
            }
            
            # Run optimization with multiple methods for robustness
            methods = ['trust-constr', 'SLSQP']
            method_results = []
            
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        initial_params,
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 3000, 'ftol': 1e-13, 'gtol': 1e-13}
                    )
                    if result.success:
                        method_results.append(result)
                except:
                    continue
            
            if method_results:
                # Pick the best result among methods
                best_method_result = min(method_results, key=lambda r: -objective(r.x))
                
                final_params = best_method_result.x
                current_sum = -objective(final_params)
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = np.zeros((n, 3))
                    for i in range(n):
                        best_circles[i] = final_params[3*i:3*i+3]
                    best_result = best_circles.copy()
                    
        except Exception:
            continue
    
    # If we have a good result, do enhanced refinement
    if best_result is not None:
        circles = best_result.copy()
        
        # Apply iterative refinement
        circles = refine_solution(circles)
        
        # Final validation and boundary correction
        for i in range(n):
            x, y, r = circles[i]
            # Ensure within bounds
            x = np.clip(x, r, width - r)
            y = np.clip(y, r, height - r)
            circles[i] = [x, y, r]
    else:
        # Fallback to simple configuration
        circles = []
        for i in range(n):
            x = random.uniform(0.05, width - 0.05)
            y = random.uniform(0.05, height - 0.05)
            circles.append([x, y, 0.08])
        circles = np.array(circles)
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
