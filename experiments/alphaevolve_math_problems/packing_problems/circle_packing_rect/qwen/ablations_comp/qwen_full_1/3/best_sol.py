# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric initialization with advanced optimization techniques.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Use the proven optimal rectangle dimensions from research
    width = 1.0
    height = 1.0
    
    # Validate dimensions
    if width + height != 2:
        width = 1.0
        height = 1.0
    
    n = 21
    
    # Generate high-quality initial configuration using multiple strategies
    def generate_hexagonal_grid():
        """Generate circles using hexagonal packing pattern - inspired by optimal circle packing theory"""
        circles = []
        
        # Create a hexagonal pattern that fits well in our rectangle
        rows = 5
        cols = 5
        
        # Calculate hexagonal spacing
        hex_radius = 0.15  # Starting with larger radius for better initial fit
        cell_width = hex_radius * 2
        cell_height = hex_radius * np.sqrt(3)
        
        # Create hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Offset every other row for hexagonal packing
                x_offset = (i % 2) * 0.5
                x = (j + x_offset) * cell_width + hex_radius
                y = i * cell_height + hex_radius
                
                # Add significant jitter to avoid alignment artifacts
                x += random.uniform(-cell_width/6, cell_width/6)
                y += random.uniform(-cell_height/6, cell_height/6)
                
                # Ensure within bounds (with safety margin)
                x = np.clip(x, hex_radius, width - hex_radius)
                y = np.clip(y, hex_radius, height - hex_radius)
                
                circles.append([x, y, hex_radius])
                
        # Fill remaining slots with random positions
        while len(circles) < n:
            circles.append([
                random.uniform(hex_radius, width - hex_radius),
                random.uniform(hex_radius, height - hex_radius),
                hex_radius
            ])
            
        return np.array(circles[:n])
    
    def generate_fibonacci_spiral():
        """Generate circles using Fibonacci spiral pattern - known to give good distribution"""
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
            
            # Add some randomness
            x += random.uniform(-0.05, 0.05)
            y += random.uniform(-0.05, 0.05)
            
            # Keep within bounds
            x = np.clip(x, 0.05, width - 0.05)
            y = np.clip(y, 0.05, height - 0.05)
            
            circles.append([x, y, 0.08])
            
        return np.array(circles)
    
    def generate_optimized_grid():
        """Generate circles using a grid pattern optimized for 21 circles"""
        circles = []
        
        # Try 4x6 grid pattern for 21 circles
        rows = 4
        cols = 6
        
        cell_width = width / cols
        cell_height = height / rows
        
        # Create positions with strategic placement
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Position with slight jitter for better distribution
                x = (j + 0.5) * cell_width + random.uniform(-cell_width/8, cell_width/8)
                y = (i + 0.5) * cell_height + random.uniform(-cell_height/8, cell_height/8)
                
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
                0.08
            ])
            
        return np.array(circles)
    
    def generate_refined_initial():
        """Generate a refined initial configuration with better spacing"""
        circles = []
        
        # Start with a good hexagonal pattern but with better spacing
        rows = 5
        cols = 5
        spacing_factor = 0.85  # Slightly reduced spacing to allow more room for optimization
        
        # Calculate spacing
        hex_radius = 0.12
        cell_width = hex_radius * 2 * spacing_factor
        cell_height = hex_radius * np.sqrt(3) * spacing_factor
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Hexagonal offset for alternate rows
                x_offset = j * cell_width
                if i % 2 == 1:
                    x_offset += cell_width / 2
                
                y_offset = i * cell_height
                
                x = x_offset + hex_radius
                y = y_offset + hex_radius
                
                # Add noise to avoid perfect alignment
                x += random.uniform(-cell_width/10, cell_width/10)
                y += random.uniform(-cell_height/10, cell_height/10)
                
                # Ensure within bounds
                x = np.clip(x, hex_radius, width - hex_radius)
                y = np.clip(y, hex_radius, height - hex_radius)
                
                circles.append([x, y, hex_radius])
        
        # Fill remaining spots with carefully positioned circles
        while len(circles) < n:
            # Position near center with small radius
            x = width/2 + random.uniform(-0.1, 0.1)
            y = height/2 + random.uniform(-0.1, 0.1)
            r = random.uniform(0.05, 0.1)
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Constraint function for optimization - more efficient version
    def constraint_func(params):
        # Return positive values when constraints are satisfied
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            x, y, r = params[3*i:3*i+3]
            constraints.extend([
                x - r,           # left boundary
                y - r,           # bottom boundary  
                width - x - r,   # right boundary
                height - y - r   # top boundary
            ])
        
        # Distance constraints (no overlap) - only compute when needed
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = params[3*i:3*i+3]
                x2, y2, r2 = params[3*j:3*j+3]
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)
                constraints.append(distance - (r1 + r2))
                
        return np.array(constraints)
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i+2]  # radius is third component
        return -total_radius  # negative because we want to maximize
    
    # Multi-start optimization with different strategies
    best_result = None
    best_sum = -np.inf
    
    # Strategy 1: Hexagonal grid initialization
    try:
        circles = generate_hexagonal_grid()
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
        
        # Run optimization with trust-constr (most robust)
        result = minimize(
            objective,
            initial_params,
            method='trust-constr',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 6000, 'ftol': 1e-17, 'gtol': 1e-17}
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
                
    except Exception as e:
        pass  # Continue to next strategy if this fails
    
    # Strategy 2: Fibonacci spiral initialization
    try:
        circles = generate_fibonacci_spiral()
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
            options={'maxiter': 6000, 'ftol': 1e-17, 'gtol': 1e-17}
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
                
    except Exception as e:
        pass  # Continue if this fails
    
    # Strategy 3: Optimized grid initialization
    try:
        circles = generate_optimized_grid()
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
            options={'maxiter': 6000, 'ftol': 1e-17, 'gtol': 1e-17}
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
                
    except Exception as e:
        pass  # Continue if this fails
    
    # Strategy 4: Refined initial configuration
    try:
        circles = generate_refined_initial()
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
            options={'maxiter': 6000, 'ftol': 1e-17, 'gtol': 1e-17}
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
                
    except Exception as e:
        pass  # Continue if this fails
    
    # Strategy 5: Multiple restarts with enhanced random initialization (more restarts)
    for restart in range(25):  # Even more restarts for better exploration
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
            methods = ['trust-constr', 'SLSQP']  # Prioritize trust-constr for better results
            method_results = []
            
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        initial_params,
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options={'maxiter': 3500, 'ftol': 1e-15, 'gtol': 1e-15}
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
                    
        except Exception as e:
            continue
    
    # If we have a good result, do enhanced refinement
    if best_result is not None:
        circles = best_result.copy()
        
        # Enhanced local refinement with multiple passes and more aggressive improvement
        for pass_num in range(25):  # Even more passes for better improvement
            improved = False
            # Try to increase radii systematically
            for i in range(n):
                old_x, old_y, old_r = circles[i]
                
                # Find minimum distance to any other circle
                min_distance = float('inf')
                for j in range(n):
                    if i != j:
                        dx = circles[j][0] - old_x
                        dy = circles[j][1] - old_y
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
                    test_radius = min(old_r * 1.10, max_radius)  # Even more aggressive
                    
                    # Check if this works without overlap
                    valid = True
                    for j in range(n):
                        if i != j:
                            dx = circles[j][0] - old_x
                            dy = circles[j][1] - old_y
                            distance = np.sqrt(dx*dx + dy*dy)
                            if distance < (test_radius + circles[j][2]):
                                valid = False
                                break
                    
                    if valid:
                        circles[i] = [old_x, old_y, test_radius]
                        improved = True
                        
            if not improved:
                break
    
    # Final validation to ensure all constraints are met
    # Make sure no circles exceed bounds
    for i in range(n):
        x, y, r = circles[i]
        # Clamp to bounds
        x = np.clip(x, r, width - r)
        y = np.clip(y, r, height - r)
        circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
