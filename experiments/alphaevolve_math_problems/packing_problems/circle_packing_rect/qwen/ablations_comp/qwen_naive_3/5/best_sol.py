# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import random
from deap import base, creator, tools, algorithms
import time
from platypus import NSGAII, Problem, Real
from scipy.spatial import distance_matrix

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid evolutionary approach with advanced optimization techniques.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Try different aspect ratios to find optimal
    best_ratio = 1.0  # Start with square
    width, height = 1.0, 1.0
    
    # Number of circles
    n = 21
    
    # Better initialization using a more structured approach
    def generate_better_initialization():
        # Try multiple strategies and pick the best
        strategies = []
        
        # Strategy 1: Hexagonal packing
        def hexagonal_pack():
            # Calculate optimal spacing for hexagonal packing
            total_area = width * height
            circle_area = total_area / n * 0.85  # Leave some margin
            avg_radius = np.sqrt(circle_area / np.pi)
            spacing = 2 * avg_radius
            
            # Generate hexagonal grid
            hex_radius = spacing * np.sqrt(3) / 2
            rows = int(np.ceil(height / hex_radius)) + 2
            cols = int(np.ceil(width / spacing)) + 2
            
            centers = []
            for i in range(rows):
                for j in range(cols):
                    x = 0.1 + j * spacing + (i % 2) * spacing / 2
                    y = 0.1 + i * hex_radius
                    if x <= width - 0.1 and y <= height - 0.1:
                        centers.append([x, y])
            
            # Take first n centers
            if len(centers) >= n:
                selected_centers = np.array(centers[:n])
            else:
                # Add random points for remaining circles
                selected_centers = np.array(centers)
                remaining = n - len(centers)
                for _ in range(remaining):
                    x = random.uniform(0.1, width - 0.1)
                    y = random.uniform(0.1, height - 0.1)
                    selected_centers = np.vstack([selected_centers, [x, y]])
            
            return selected_centers
        
        # Strategy 2: Golden spiral with better scaling
        def golden_spiral():
            golden_angle = np.pi * (3 - np.sqrt(5))
            centers = []
            
            # Generate points along a spiral
            for i in range(n):
                radius = np.sqrt(i / (n - 1)) * 0.4  # Scale to fit in rectangle
                angle = i * golden_angle
                x = width/2 + radius * np.cos(angle) * 0.8
                y = height/2 + radius * np.sin(angle) * 0.8
                centers.append([x, y])
            
            return np.array(centers)
        
        # Strategy 3: Grid with perturbation
        def grid_perturbed():
            # Create regular grid
            side_length = int(np.ceil(np.sqrt(n)))
            spacing_x = width / (side_length + 1)
            spacing_y = height / (side_length + 1)
            
            centers = []
            for i in range(side_length):
                for j in range(side_length):
                    if len(centers) >= n:
                        break
                    x = spacing_x * (j + 1) + random.uniform(-spacing_x/4, spacing_x/4)
                    y = spacing_y * (i + 1) + random.uniform(-spacing_y/4, spacing_y/4)
                    if x <= width - 0.1 and y <= height - 0.1:
                        centers.append([x, y])
            
            # Fill remaining if needed
            while len(centers) < n:
                x = random.uniform(0.1, width - 0.1)
                y = random.uniform(0.1, height - 0.1)
                centers.append([x, y])
            
            return np.array(centers[:n])
        
        # Try all strategies and return the best one
        strategies = [hexagonal_pack(), golden_spiral(), grid_perturbed()]
        
        # Evaluate each strategy by how many circles fit without overlap
        best_strategy = strategies[0]
        best_score = 0
        
        for strategy in strategies:
            # Simple heuristic evaluation - count valid circles
            score = 0
            radii = [0.05] * n  # Initial guess
            for i in range(n):
                x, y = strategy[i]
                valid = True
                for j in range(n):
                    if i != j:
                        x2, y2 = strategy[j]
                        dist_sq = (x - x2)**2 + (y - y2)**2
                        if dist_sq < (radii[i] + radii[j])**2:
                            valid = False
                            break
                if valid:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy
    
    # Generate initial configuration
    initial_centers = generate_better_initialization()
    
    # Set initial radii based on available space
    initial_radii = np.full(n, 0.05)
    
    # Combine into one array for optimization
    initial_params = np.column_stack([initial_centers, initial_radii])
    
    # Define constraint checking functions
    def check_containment(x, y, r, w, h):
        """Check if circle fits entirely within rectangle"""
        return (x - r >= 0 and y - r >= 0 and x + r <= w and y + r <= h)
    
    def check_non_overlap(x1, y1, r1, x2, y2, r2):
        """Check if two circles don't overlap"""
        dist_sq = (x1 - x2)**2 + (y1 - y2)**2
        return dist_sq >= (r1 + r2)**2
    
    # Multi-objective optimization approach
    def multi_objective_optimization():
        # Define problem with multiple objectives
        def evaluate_multi_objective(individual):
            # Reshape individual to (n, 3) format
            circles = individual.reshape((n, 3))
            
            # Extract parameters
            x_coords = circles[:, 0]
            y_coords = circles[:, 1]
            radii = circles[:, 2]
            
            # Check constraints
            valid = True
            penalty = 0
            
            # Check containment
            for i in range(n):
                if not check_containment(x_coords[i], y_coords[i], radii[i], width, height):
                    valid = False
                    penalty += 1000  # Large penalty for containment violations
            
            # Check non-overlapping
            for i in range(n):
                for j in range(i+1, n):
                    if not check_non_overlap(x_coords[i], y_coords[i], radii[i], 
                                           x_coords[j], y_coords[j], radii[j]):
                        valid = False
                        penalty += 1000  # Large penalty for overlap violations
            
            # If invalid, return penalties
            if not valid:
                return [-np.sum(radii) - penalty, penalty]
            
            # Return negative sum of radii and penalty
            return [-np.sum(radii), 0]
        
        # Use NSGA-II for multi-objective optimization
        # Create problem
        class CirclePackingProblem(Problem):
            def __init__(self):
                super().__init__(3*n, 2)  # 3*n variables, 2 objectives
                # Bounds for each variable
                for i in range(3*n):
                    self.types[i] = Real(0.01, width - 0.01 if i % 3 == 0 or i % 3 == 1 else min(width, height)/2 - 0.01)
            
            def evaluate(self, solution):
                # Convert solution to circles array
                circles = np.array(solution).reshape((n, 3))
                
                # Extract parameters
                x_coords = circles[:, 0]
                y_coords = circles[:, 1]
                radii = circles[:, 2]
                
                # Check constraints
                valid = True
                penalty = 0
                
                # Check containment
                for i in range(n):
                    if not check_containment(x_coords[i], y_coords[i], radii[i], width, height):
                        valid = False
                        penalty += 1000
                
                # Check non-overlapping
                for i in range(n):
                    for j in range(i+1, n):
                        if not check_non_overlap(x_coords[i], y_coords[i], radii[i], 
                                               x_coords[j], y_coords[j], radii[j]):
                            valid = False
                            penalty += 1000
                
                # If invalid, return penalties
                if not valid:
                    return [-np.sum(radii) - penalty, penalty]
                
                # Return negative sum of radii and penalty
                return [-np.sum(radii), 0]
        
        try:
            # Create problem instance
            problem = CirclePackingProblem()
            
            # Run NSGA-II optimization
            algorithm = NSGAII(problem, population_size=100, termination_condition=100)
            algorithm.run(100)
            
            # Get best solution
            if algorithm.result:
                best_solution = algorithm.result[0].variables
                circles = np.array(best_solution).reshape((n, 3))
                return circles
        except:
            pass
        
        return None
    
    # Try multi-objective optimization first
    try:
        optimized_circles = multi_objective_optimization()
        if optimized_circles is not None:
            # Validate the result
            valid = True
            for i in range(n):
                x, y, r = optimized_circles[i]
                if not check_containment(x, y, r, width, height):
                    valid = False
                    break
                for j in range(i+1, n):
                    x2, y2, r2 = optimized_circles[j]
                    if not check_non_overlap(x, y, r, x2, y2, r2):
                        valid = False
                        break
            
            if valid:
                return optimized_circles
    except:
        pass
    
    # Fallback to enhanced optimization approach
    def enhanced_optimization():
        # Objective function to maximize (negative because minimize)
        def objective(params):
            # Extract radii
            radii = params[2::3]
            # Return negative sum of radii (we want to maximize)
            return -np.sum(radii)
        
        # Constraint functions
        def containment_constraint(params):
            results = []
            for i in range(n):
                x, y, r = params[3*i], params[3*i+1], params[3*i+2]
                # Ensure circle is within bounds
                results.append(x - r)  # Should be >= 0
                results.append(y - r)  # Should be >= 0
                results.append(width - x - r)  # Should be >= 0
                results.append(height - y - r)  # Should be >= 0
            return np.array(results)
        
        def non_overlap_constraint(params):
            # Check pairwise distances
            results = []
            for i in range(n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                for j in range(i+1, n):
                    x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                    # Distance between centers
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    # Should be >= sum of radii (for no overlap)
                    results.append(dist - (r1 + r2))
            return np.array(results)
        
        # Bounds for variables
        bounds = []
        for i in range(n):
            bounds.extend([
                (0.01, width - 0.01),   # x bounds
                (0.01, height - 0.01),  # y bounds
                (0.001, min(width, height)/2 - 0.01)  # r bounds
            ])
        
        constraints = [
            {'type': 'ineq', 'fun': containment_constraint},
            {'type': 'ineq', 'fun': non_overlap_constraint}
        ]
        
        # Run optimization with multiple attempts
        best_result = None
        best_sum = 0
        
        # Try multiple optimization runs with different starting points
        for attempt in range(8):  # Increased attempts
            # Generate better starting point
            if attempt == 0:
                # Use the initial configuration
                start_params = initial_params.flatten()
            else:
                # Perturb the initial configuration
                start_params = initial_params.flatten().copy()
                for i in range(len(start_params)):
                    if i % 3 == 0:  # x coordinate
                        start_params[i] += random.uniform(-0.1, 0.1)
                        start_params[i] = np.clip(start_params[i], 0.01, width - 0.01)
                    elif i % 3 == 1:  # y coordinate
                        start_params[i] += random.uniform(-0.1, 0.1)
                        start_params[i] = np.clip(start_params[i], 0.01, height - 0.01)
                    else:  # radius
                        start_params[i] += random.uniform(-0.02, 0.02)
                        start_params[i] = np.clip(start_params[i], 0.001, min(width, height)/2 - 0.01)
            
            try:
                # Try different optimization methods
                result = minimize(
                    objective,
                    start_params,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-6, 'eps': 1e-4}
                )
                
                if result.success:
                    # Check if this result is better
                    final_params = result.x
                    test_radii = final_params[2::3]
                    test_sum = np.sum(test_radii)
                    if test_sum > best_sum:
                        best_sum = test_sum
                        best_result = result
                        
            except Exception as e:
                continue
        
        # If we found a good result, use it
        if best_result is not None:
            final_params = best_result.x
            circles = np.reshape(final_params, (n, 3))
            return circles
        else:
            # Return initial configuration if nothing worked
            return initial_params
    
    # Try enhanced optimization approach
    circles = enhanced_optimization()
    
    # Final refinement with local search using a more sophisticated approach
    try:
        current_circles = circles.copy()
        
        # Use a greedy local search to improve the solution
        improved = True
        max_iter = 100
        
        for iteration in range(max_iter):
            if not improved:
                break
            improved = False
            
            # Try to increase radii for each circle
            for i in range(n):
                # Save current state
                x, y, r = current_circles[i]
                current_radius = r
                
                # Try to increase radius while maintaining constraints
                # We'll try to increase by small amounts
                step_size = 0.005
                new_radius = min(current_radius + step_size, min(width, height)/2 - 0.01)
                
                # Check if we can actually increase radius
                valid = True
                for j in range(n):
                    if i != j:
                        x_j, y_j, r_j = current_circles[j]
                        # Check containment
                        if not check_containment(x, y, new_radius, width, height):
                            valid = False
                            break
                        # Check overlap
                        dist_sq = (x - x_j)**2 + (y - y_j)**2
                        if dist_sq < (new_radius + r_j)**2:
                            valid = False
                            break
                
                if valid and new_radius > current_radius:
                    current_circles[i, 2] = new_radius
                    improved = True
                    
        # Also try to slightly adjust positions to improve packing
        for iteration in range(20):
            # For each circle, try to move it slightly to reduce conflicts
            for i in range(n):
                x, y, r = current_circles[i]
                old_x, old_y = x, y
                
                # Try small adjustments
                dx = random.uniform(-0.01, 0.01)
                dy = random.uniform(-0.01, 0.01)
                
                new_x = x + dx
                new_y = y + dy
                
                # Keep within bounds
                new_x = np.clip(new_x, r + 0.01, width - r - 0.01)
                new_y = np.clip(new_y, r + 0.01, height - r - 0.01)
                
                # Check if this improves things
                valid = True
                for j in range(n):
                    if i != j:
                        x_j, y_j, r_j = current_circles[j]
                        # Check containment
                        if not check_containment(new_x, new_y, r, width, height):
                            valid = False
                            break
                        # Check overlap
                        dist_sq = (new_x - x_j)**2 + (new_y - y_j)**2
                        if dist_sq < (r + r_j)**2:
                            valid = False
                            break
                
                if valid:
                    current_circles[i, 0] = new_x
                    current_circles[i, 1] = new_y
                    
    except Exception as e:
        pass  # Continue with current solution
    
    return current_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
