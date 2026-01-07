# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, multi-start optimization, 
    and local refinement for robustness and quality.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Improved initialization using a more sophisticated geometric approach
    def initialize_better_placement():
        # Create a more evenly distributed initial configuration
        circles = []
        
        # Try different grid sizes to find a good distribution
        grid_sizes = [(6, 6), (7, 5), (5, 7), (8, 4), (4, 8)]
        
        best_grid = (6, 6)
        best_density = 0
        
        for rows, cols in grid_sizes:
            if rows * cols >= n:
                # Calculate spacing
                spacing_x = 1.0 / cols
                spacing_y = 1.0 / rows
                
                # Check density estimate
                estimated_density = min(spacing_x, spacing_y) ** 2 * n / 1.0
                
                if estimated_density > best_density:
                    best_density = estimated_density
                    best_grid = (rows, cols)
        
        rows, cols = best_grid
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Generate positions with some randomness to avoid regular patterns
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Add slight randomness to positions
                x = (j + 0.5 + random.uniform(-0.2, 0.2)) * spacing_x
                y = (i + 0.5 + random.uniform(-0.2, 0.2)) * spacing_y
                # Keep within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                circles.append([x, y])
        
        # Trim to exact number
        circles = circles[:n]
        
        # Initialize with appropriate radii based on spacing
        avg_spacing = (spacing_x + spacing_y) / 2
        initial_radii = [avg_spacing * 0.3] * n
        
        # Adjust radii to respect boundary constraints
        for i in range(n):
            x, y = circles[i]
            # Radius cannot exceed distance to nearest boundary
            max_radius = min(x, 1-x, y, 1-y)
            initial_radii[i] = min(initial_radii[i], max_radius * 0.9)
        
        # Combine into solution vector
        solution = []
        for i in range(n):
            solution.extend([circles[i][0], circles[i][1], initial_radii[i]])
            
        return solution
    
    # More robust constraint checking
    def check_constraints(sol):
        """Check if solution satisfies all constraints"""
        # Check boundary constraints
        for i in range(n):
            x = sol[3*i]
            y = sol[3*i+1]
            r = sol[3*i+2]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check overlap constraints manually for better control
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = sol[3*i], sol[3*i+1], sol[3*i+2]
                x2, y2, r2 = sol[3*j], sol[3*j+1], sol[3*j+2]
                # Distance between centers must be >= sum of radii
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                if dist_sq < min_dist_sq:
                    return False
                    
        return True
    
    # Objective function (negative because we minimize to maximize sum)
    def objective(sol):
        return -np.sum(sol[2::3])  # Sum of all radii (every third element starting at index 2)
    
    # Constraint functions with better numerical stability
    def contain_constraints(sol):
        """Ensure all circles are within the unit square"""
        constraints = []
        for i in range(n):
            x = sol[3*i]
            y = sol[3*i+1]
            r = sol[3*i+2]
            # Circle must be contained in square [0,1]x[0,1]
            # x - r >= 0  =>  x >= r
            # y - r >= 0  =>  y >= r  
            # 1 - x - r >= 0  =>  x <= 1 - r
            # 1 - y - r >= 0  =>  y <= 1 - r
            constraints.extend([
                x - r,           # x >= r
                y - r,           # y >= r
                1 - x - r,       # x <= 1 - r
                1 - y - r        # y <= 1 - r
            ])
        return np.array(constraints)
    
    def overlap_constraints(sol):
        """Ensure no two circles overlap"""
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = sol[3*i], sol[3*i+1], sol[3*i+2]
                x2, y2, r2 = sol[3*j], sol[3*j+1], sol[3*j+2]
                # Distance between centers must be >= sum of radii
                # We want: sqrt((x1-x2)^2 + (y1-y2)^2) >= r1 + r2
                # Squared form: (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
                # Constraint is: (x1-x2)^2 + (y1-y2)^2 - (r1+r2)^2 >= 0
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)
    
    # Define bounds for variables
    bounds = []
    for i in range(n):
        # x, y in [r, 1-r], so r <= 0.5
        # r > 0
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Define constraints for scipy.optimize
    cons = []
    # Add containment constraints (all must be >= 0)
    cons.append({'type': 'ineq', 'fun': lambda x: contain_constraints(x)})
    # Add overlap constraints (all must be >= 0)
    cons.append({'type': 'ineq', 'fun': lambda x: overlap_constraints(x)})
    
    # Run multiple restarts with different strategies
    best_solution = None
    best_sum = -float('inf')
    
    # Try multiple initialization strategies and restarts
    for restart in range(25):  # Increase restarts significantly
        # Different initialization approaches
        if restart < 5:
            # Grid-based initialization
            solution = initialize_better_placement()
        elif restart < 10:
            # Random initialization with good constraints
            solution = []
            for i in range(n):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                # Initial radius based on proximity to boundaries
                r = min(x, 1-x, y, 1-y) * 0.4
                r = max(0.001, min(0.499, r))
                solution.extend([x, y, r])
        else:
            # Perturbed previous best solution
            if best_solution is not None:
                solution = best_solution.copy()
                # Add more substantial perturbation
                for i in range(n):
                    solution[3*i] += random.uniform(-0.03, 0.03)
                    solution[3*i+1] += random.uniform(-0.03, 0.03)
                    solution[3*i] = np.clip(solution[3*i], 0.001, 0.999)
                    solution[3*i+1] = np.clip(solution[3*i+1], 0.001, 0.999)
            else:
                solution = initialize_better_placement()
        
        try:
            # Use different optimization methods for robustness
            methods = ['SLSQP', 'trust-constr']
            method_idx = restart % len(methods)
            
            # Optimize using chosen method
            result = minimize(
                objective, 
                solution, 
                method=methods[method_idx], 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 2000, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
            )
            
            # Evaluate solution quality
            current_sum = -objective(result.x) if result.success else -objective(solution)
            
            # If we get a better solution, keep it
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution = result.x.copy() if result.success else solution.copy()
                
        except Exception as e:
            # Continue with current best if optimization fails
            continue
    
    # Final refinement using enhanced local search
    if best_solution is not None:
        # Apply enhanced local search refinement
        final_solution = best_solution.copy()
        
        # Try more comprehensive local search
        improved = True
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try improving each circle individually
            for i in range(n):
                # Store original state
                orig_x, orig_y, orig_r = final_solution[3*i], final_solution[3*i+1], final_solution[3*i+2]
                
                # Try various adjustments to position and radius
                best_x, best_y, best_r = orig_x, orig_y, orig_r
                best_sum = -objective(final_solution)
                
                # Test various small moves - more thorough exploration
                moves = [
                    (0, 0, 0),      # No change
                    (-0.002, 0, 0), (0.002, 0, 0), (0, -0.002, 0), (0, 0.002, 0),
                    (-0.005, 0, 0), (0.005, 0, 0), (0, -0.005, 0), (0, 0.005, 0),
                    (-0.01, 0, 0), (0.01, 0, 0), (0, -0.01, 0), (0, 0.01, 0),
                    (-0.005, -0.005, 0), (0.005, 0.005, 0), (-0.005, 0.005, 0), (0.005, -0.005, 0),
                    (0, 0, 0.001), (0, 0, -0.001), (0, 0, 0.003), (0, 0, -0.003),
                    (0, 0, 0.005), (0, 0, -0.005)
                ]
                
                for dx, dy, dr in moves:
                    new_x = max(0.001, min(0.999, orig_x + dx))
                    new_y = max(0.001, min(0.999, orig_y + dy))
                    new_r = max(0.001, min(0.499, orig_r + dr))
                    
                    # Temporarily update this circle
                    temp_solution = final_solution.copy()
                    temp_solution[3*i] = new_x
                    temp_solution[3*i+1] = new_y
                    temp_solution[3*i+2] = new_r
                    
                    # Check if this maintains feasibility
                    if check_constraints(temp_solution):
                        new_sum = -objective(temp_solution)
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_x, best_y, best_r = new_x, new_y, new_r
                            improved = True
                
                # Update if we found an improvement
                if improved:
                    final_solution[3*i] = best_x
                    final_solution[3*i+1] = best_y
                    final_solution[3*i+2] = best_r
    
    # Convert back to circles array
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [final_solution[3*i], final_solution[3*i+1], final_solution[3*i+2]]
    
    return circles


# EVOLVE-BLOCK-END
