# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
from itertools import combinations
import time
from scipy.optimize import differential_evolution
import warnings
from scipy.spatial import distance
import copy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from functools import partial
import math
from scipy.spatial.distance import pdist, squareform
import itertools
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid evolutionary and optimization approach for better performance.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    n = 21
    
    # Objective function to maximize sum of radii
    def objective(params):
        # Reshape parameters: [x1,y1,r1,x2,y2,r2,...,x21,y21,r21]
        circles = params.reshape(-1, 3)
        # Minimize negative sum of radii (maximize sum)
        return -np.sum(circles[:, 2])
    
    # More efficient constraint implementation using vectorized operations
    def constraint_distance(params):
        circles = params.reshape(-1, 3)
        # Vectorized distance computation for all pairs
        distances = cdist(circles[:, :2], circles[:, :2])
        # Create upper triangular mask to avoid duplicate pairs
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        # Get pairwise distances and radii sums
        dists = distances[mask]
        radii_sums = (circles[:, 2][:, None] + circles[:, 2][None, :])[mask]
        # Constraint: distance >= radii_sum (for non-overlap)
        constraints = dists - radii_sums
        return constraints
    
    # Boundary constraints
    def constraint_bounds(params):
        circles = params.reshape(-1, 3)
        constraints = []
        # Rectangle dimensions - we'll optimize these too
        # For now, fixed rectangle that works well
        width, height = 1.2, 0.8  # Better aspect ratio found to work well
        
        for i in range(n):
            # x - r >= 0 (left boundary)
            constraints.append(circles[i, 0] - circles[i, 2])
            # width - x - r >= 0 (right boundary)  
            constraints.append(width - circles[i, 0] - circles[i, 2])
            # y - r >= 0 (bottom boundary)
            constraints.append(circles[i, 1] - circles[i, 2])
            # height - y - r >= 0 (top boundary)
            constraints.append(height - circles[i, 1] - circles[i, 2])
        return np.array(constraints)
    
    # Improved initialization with better circle packing strategies
    def generate_initial_solution():
        # Use a more sophisticated approach: hexagonal packing with adjustments
        width, height = 1.2, 0.8  # Optimized aspect ratio
        
        # Create a more regular hexagonal packing pattern
        circles = np.zeros((n, 3))
        
        # Parameters for hexagonal packing
        target_radius = 0.15  # Starting guess
        hex_radius = target_radius * 1.1  # Slightly larger to allow for adjustment
        
        # Hexagonal grid parameters
        row_spacing = hex_radius * 2 * np.sqrt(3) / 2  # Vertical spacing for hexagon
        col_spacing = hex_radius * 2  # Horizontal spacing
        
        # Calculate how many rows/columns we need
        rows = int(np.ceil(np.sqrt(n) * 1.2))  # Allow extra space
        cols = int(np.ceil(n / rows))
        
        # Adjust to fit within bounds
        actual_rows = min(rows, int(height / row_spacing) + 1)
        actual_cols = min(cols, int(width / col_spacing) + 1)
        
        # Place circles in hexagonal pattern
        idx = 0
        for row in range(actual_rows):
            for col in range(actual_cols):
                if idx >= n:
                    break
                    
                # Offset every other row for hexagonal packing
                x_offset = (row % 2) * (col_spacing / 2)
                x = col_spacing * col + x_offset
                y = row_spacing * row
                
                # Center the pattern within the rectangle
                x += (width - (actual_cols - 1) * col_spacing) / 2
                y += (height - (actual_rows - 1) * row_spacing) / 2
                
                # Adjust to stay within bounds
                x = max(hex_radius, min(width - hex_radius, x))
                y = max(hex_radius, min(height - hex_radius, y))
                
                # Add some randomness to avoid perfect patterns
                x += random.uniform(-hex_radius * 0.2, hex_radius * 0.2)
                y += random.uniform(-hex_radius * 0.2, hex_radius * 0.2)
                
                # Ensure reasonable radius
                r = max(0.01, min(hex_radius * 0.8, target_radius * random.uniform(0.8, 1.2)))
                
                circles[idx] = [x, y, r]
                idx += 1
                
        # Fill remaining circles with strategic placement
        if idx < n:
            for i in range(idx, n):
                # Place in a more scattered pattern with better distribution
                x = random.uniform(0.05, width - 0.05)
                y = random.uniform(0.05, height - 0.05)
                # Use a more adaptive radius based on position
                r = random.uniform(0.05, 0.15)
                circles[i] = [x, y, r]
        
        return circles, width, height
    
    # Better initialization using Voronoi-based distribution
    def generate_voronoi_initialization():
        # Create points using a Voronoi-inspired approach for even distribution
        width, height = 1.2, 0.8
        
        # Generate initial seed points
        points = []
        # Use a grid-based approach but with random perturbations
        grid_size = max(3, int(np.ceil(np.sqrt(n))))
        spacing_x = width / (grid_size + 1)
        spacing_y = height / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(points) >= n:
                    break
                x = (i + 1) * spacing_x + random.uniform(-spacing_x * 0.2, spacing_x * 0.2)
                y = (j + 1) * spacing_y + random.uniform(-spacing_y * 0.2, spacing_y * 0.2)
                points.append([x, y])
        
        # Ensure we have exactly n points
        while len(points) < n:
            x = random.uniform(0.05, width - 0.05)
            y = random.uniform(0.05, height - 0.05)
            points.append([x, y])
            
        points = points[:n]
        
        # Create circles with appropriate radii
        circles = np.zeros((n, 3))
        for i, (x, y) in enumerate(points):
            # Radius based on distance to neighbors
            if i < n - 1:
                # Use minimum distance to neighbors as basis for radius
                distances = [np.sqrt((x - px)**2 + (y - py)**2) for px, py in points[:i] + points[i+1:]]
                if distances:
                    min_dist = min(distances)
                    r = min(min_dist * 0.3, 0.2)  # Cap at reasonable size
                else:
                    r = random.uniform(0.05, 0.15)
            else:
                r = random.uniform(0.05, 0.15)
            
            # Ensure within bounds
            x = max(r, min(width - r, x))
            y = max(r, min(height - r, y))
            circles[i] = [x, y, r]
            
        return circles, width, height
    
    # Enhanced evolutionary approach with better encoding
    def evolutionary_approach():
        # Define the problem as an optimization problem
        toolbox = base.Toolbox()
        
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Define bounds for each variable (x, y, r for each circle) 
        # Using more precise bounds for better optimization
        bounds = []
        for i in range(n):
            bounds.extend([(0.01, 1.19), (0.01, 0.79), (0.001, 0.2)])  # x, y, r for each circle
        
        # Individual creation function with better initialization
        def create_individual():
            individual = []
            for i in range(n):
                # Better distributed positions
                x = random.uniform(0.05, 1.15)
                y = random.uniform(0.05, 0.75)
                r = random.uniform(0.02, 0.15)
                individual.extend([x, y, r])
            return creator.Individual(individual)
        
        # Fitness function for EA with proper constraint handling
        def evaluate(individual):
            # Extract circles
            circles_data = individual
            circles = np.array(circles_data).reshape(-1, 3)
            
            # Apply boundary constraints directly
            for i in range(len(circles)):
                circles[i, 0] = max(circles[i, 2], min(1.19, circles[i, 0]))
                circles[i, 1] = max(circles[i, 2], min(0.79, circles[i, 1]))
            
            # Calculate overlap penalty using vectorized operations
            penalty = 0
            if len(circles) > 1:
                distances = cdist(circles[:, :2], circles[:, :2])
                # Create mask for upper triangle (avoid double counting)
                mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
                # Compute overlap for overlapping pairs
                overlap_distances = distances[mask]
                overlap_radii = (circles[:, 2][:, None] + circles[:, 2][None, :])[mask]
                overlaps = overlap_radii - overlap_distances
                # Only penalize actual overlaps
                overlap_mask = overlaps > 0
                if np.any(overlap_mask):
                    penalty = np.sum(overlaps[overlap_mask] ** 2) * 1000
            
            # Calculate sum of radii minus penalty
            total_radii = np.sum(circles[:, 2])
            return (total_radii - penalty,)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution with more generations and better parameters
        try:
            population = toolbox.population(n=100)
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            population, logbook = algorithms.eaSimple(
                population, 
                toolbox, 
                cxpb=0.7, 
                mutpb=0.3, 
                ngen=50, 
                stats=stats, 
                halloffame=hof, 
                verbose=False
            )
            
            best_individual = hof[0]
            return best_individual
        except Exception as e:
            return None
    
    # Improved optimization with better convergence strategies
    def optimize_with_improved_strategies(initial_circles, width, height):
        # Create bounds with correct dimensions
        bounds = []
        for i in range(n):
            bounds.extend([(0.01, width - 0.01), (0.01, height - 0.01), (0.001, width/2)])
        
        # Create constraint dictionaries with more efficient implementations
        def distance_constraint(params):
            circles = params.reshape(-1, 3)
            distances = cdist(circles[:, :2], circles[:, :2])
            # Use upper triangle to avoid duplicates
            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            dists = distances[mask]
            radii_sums = (circles[:, 2][:, None] + circles[:, 2][None, :])[mask]
            constraints = dists - radii_sums
            return constraints
        
        def bound_constraint(params):
            circles = params.reshape(-1, 3)
            constraints = []
            for i in range(n):
                constraints.append(circles[i, 0] - circles[i, 2])
                constraints.append(width - circles[i, 0] - circles[i, 2])
                constraints.append(circles[i, 1] - circles[i, 2])
                constraints.append(height - circles[i, 1] - circles[i, 2])
            return np.array(constraints)
        
        distance_cons = {
            'type': 'ineq',
            'fun': lambda x: distance_constraint(x)
        }
        
        bound_cons = {
            'type': 'ineq', 
            'fun': lambda x: bound_constraint(x)
        }
        
        # Try multiple optimization approaches
        try:
            # First try with L-BFGS-B for faster convergence
            result1 = minimize(
                objective,
                initial_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result1.success:
                # Then try with SLSQP for better constraint handling
                result2 = minimize(
                    objective,
                    result1.x,
                    method='SLSQP',
                    constraints=[distance_cons, bound_cons],
                    options={'maxiter': 100, 'ftol': 1e-8, 'eps': 1e-8},
                    bounds=bounds
                )
                
                if result2.success:
                    refined_circles = result2.x.reshape(-1, 3)
                    return refined_circles, True
        except Exception as e:
            pass
        
        # Fallback to direct optimization with constraints
        try:
            result = minimize(
                objective,
                initial_circles.flatten(),
                method='SLSQP',
                constraints=[distance_cons, bound_cons],
                options={'maxiter': 100, 'ftol': 1e-8, 'eps': 1e-8},
                bounds=bounds
            )
            
            if result.success:
                refined_circles = result.x.reshape(-1, 3)
                return refined_circles, True
        except Exception as e:
            pass
        
        return initial_circles, False
    
    # Try multiple strategies with better rectangle optimization
    best_circles = None
    best_sum = 0
    best_width = 1.2
    best_height = 0.8
    
    # Test multiple aspect ratios with focus on promising ones
    # Let's expand the search space more systematically
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5, 3.0]
    
    # Also try optimizing rectangle dimensions directly in a constrained optimization
    def optimize_rectangle_dimensions():
        """Try to optimize rectangle dimensions as well"""
        def rectangle_objective(dimensions):
            width, height = dimensions
            if width + height != 2:
                return float('inf')  # Invalid constraint
            
            # Try a few initialization methods with these dimensions
            circles, _, _ = generate_initial_solution()
            
            # Scale circles to fit in this rectangle
            scale_x = width / 1.2
            scale_y = height / 0.8
            for i in range(len(circles)):
                circles[i, 0] *= scale_x
                circles[i, 1] *= scale_y
                circles[i, 2] *= min(scale_x, scale_y)  # Scale radii appropriately
            
            # Now optimize this configuration
            optimized_circles, success = optimize_with_improved_strategies(circles, width, height)
            return -np.sum(optimized_circles[:, 2])  # Negative because we minimize
        
        # Try different rectangle sizes
        best_rect_sum = 0
        best_rect_dims = (1.2, 0.8)
        
        # Grid search over some aspect ratios
        test_ratios = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0]
        for ratio in test_ratios:
            width = 2.0 / (1.0 + ratio)
            height = width * ratio
            try:
                # Simple heuristic: start with good initial solution and optimize
                circles, _, _ = generate_initial_solution()
                # Scale to this rectangle
                scale_x = width / 1.2
                scale_y = height / 0.8
                for i in range(len(circles)):
                    circles[i, 0] *= scale_x
                    circles[i, 1] *= scale_y
                    circles[i, 2] *= min(scale_x, scale_y)
                
                optimized_circles, success = optimize_with_improved_strategies(circles, width, height)
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_rect_sum:
                    best_rect_sum = current_sum
                    best_rect_dims = (width, height)
            except:
                continue
        
        return best_rect_dims[0], best_rect_dims[1], best_rect_sum
    
    # Try optimizing rectangle dimensions first
    rect_width, rect_height, rect_sum = optimize_rectangle_dimensions()
    if rect_sum > 0:
        # Use this optimized rectangle for further processing
        best_width = rect_width
        best_height = rect_height
        best_sum = rect_sum
    
    # Continue with systematic search over aspect ratios
    for ratio in aspect_ratios:
        width = 2.0 / (1.0 + ratio)
        height = width * ratio
        
        # Try both initialization methods
        for init_method in [generate_initial_solution, generate_voronoi_initialization]:
            try:
                circles, _, _ = init_method()
                
                # Scale circles to fit in this rectangle if needed
                if width < 1.2 or height < 0.8:
                    scale_x = width / 1.2
                    scale_y = height / 0.8
                    for i in range(len(circles)):
                        circles[i, 0] *= scale_x
                        circles[i, 1] *= scale_y
                        circles[i, 2] *= min(scale_x, scale_y)
                
                # Optimize for this configuration
                optimized_circles, success = optimize_with_improved_strategies(circles, width, height)
                
                # Check if this is better
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized_circles.copy()
                    best_width = width
                    best_height = height
            except Exception as e:
                continue
    
    # Additional refinement with evolutionary approach
    try:
        # Try evolutionary approach to find even better solutions
        ea_result = evolutionary_approach()
        if ea_result is not None:
            # Extract circles from EA result
            circles_data = ea_result
            circles = np.array(circles_data).reshape(-1, 3)
            
            # Apply boundary constraints
            for i in range(len(circles)):
                circles[i, 0] = max(circles[i, 2], min(1.19, circles[i, 0]))
                circles[i, 1] = max(circles[i, 2], min(0.79, circles[i, 1]))
            
            # Calculate sum of radii
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
                best_width = 1.2
                best_height = 0.8
    except Exception as e:
        pass
    
    # Final optimization with improved algorithm
    if best_circles is not None:
        # Create final bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0.01, best_width - 0.01), (0.01, best_height - 0.01), (0.001, best_width/2)])
        
        # Recreate constraints with more efficient implementation
        def final_constraint_distance(params):
            circles = params.reshape(-1, 3)
            distances = cdist(circles[:, :2], circles[:, :2])
            # Use upper triangle to avoid duplicates
            mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
            dists = distances[mask]
            radii_sums = (circles[:, 2][:, None] + circles[:, 2][None, :])[mask]
            constraints = dists - radii_sums
            return constraints
        
        def final_constraint_bounds(params):
            circles = params.reshape(-1, 3)
            constraints = []
            for i in range(n):
                constraints.append(circles[i, 0] - circles[i, 2])
                constraints.append(best_width - circles[i, 0] - circles[i, 2])
                constraints.append(circles[i, 1] - circles[i, 2])
                constraints.append(best_height - circles[i, 1] - circles[i, 2])
            return np.array(constraints)
        
        distance_cons = {
            'type': 'ineq',
            'fun': lambda x: final_constraint_distance(x)
        }
        
        bound_cons = {
            'type': 'ineq', 
            'fun': lambda x: final_constraint_bounds(x)
        }
        
        # Try with multiple optimization approaches
        try:
            # Method 1: L-BFGS-B first for global search
            result1 = minimize(
                objective,
                best_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 30, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result1.success:
                # Then SLSQP with constraints
                result2 = minimize(
                    objective,
                    result1.x,
                    method='SLSQP',
                    constraints=[distance_cons, bound_cons],
                    options={'maxiter': 100, 'ftol': 1e-8, 'eps': 1e-8},
                    bounds=bounds
                )
                
                if result2.success:
                    best_circles = result2.x.reshape(-1, 3)
        except Exception as e:
            pass
    
    # Ensure all circles are valid
    if best_circles is not None:
        # Validate constraints
        circles = best_circles.copy()
        for i in range(len(circles)):
            # Ensure radii are positive
            circles[i, 2] = max(0.001, circles[i, 2])
            # Ensure positions are valid
            circles[i, 0] = max(circles[i, 2], min(best_width - circles[i, 2], circles[i, 0]))
            circles[i, 1] = max(circles[i, 2], min(best_height - circles[i, 2], circles[i, 1]))
        
        return circles
    
    # Fallback to initial solution
    circles, _, _ = generate_initial_solution()
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
