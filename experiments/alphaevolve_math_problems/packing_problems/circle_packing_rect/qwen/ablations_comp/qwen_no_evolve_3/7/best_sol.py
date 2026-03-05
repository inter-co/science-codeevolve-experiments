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
from scipy.optimize import Bounds
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
    
    # Constraints for non-overlapping - optimized version
    def constraint_distance(params):
        circles = params.reshape(-1, 3)
        # More efficient implementation using vectorized operations
        # Create all pairwise differences
        diff = circles[:, np.newaxis, :2] - circles[np.newaxis, :, :2]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        # Create constraints for all pairs
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                radii_sum = circles[i, 2] + circles[j, 2]
                # Constraint: distance >= radii_sum (for non-overlap)
                constraints.append(dist - radii_sum)
        return np.array(constraints)
    
    # Constraints for boundary conditions
    def constraint_bounds(params):
        circles = params.reshape(-1, 3)
        constraints = []
        # Rectangle dimensions will be optimized as part of the problem
        # For now, we'll work with a fixed rectangle and optimize positions/radii
        width, height = 1.0, 1.0  # Will be optimized later
        
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
    
    # Better initialization approach using hexagonal packing with improved geometry
    def generate_hexagonal_initial_solution():
        # Use more sophisticated hexagonal packing approach
        # Try aspect ratios that typically work well for circle packing
        best_aspect_ratios = [1.0, 1.2, 1.3, 1.5, 1.8, 2.0]
        
        best_circles = None
        best_sum = 0
        best_width = 1.0
        best_height = 1.0
        
        for ratio in best_aspect_ratios:
            width = 2.0 / (1.0 + ratio)
            height = width * ratio
            
            # Create hexagonal arrangement
            circles = np.zeros((n, 3))
            
            # Try to place circles in a hexagonal lattice pattern
            # For 21 circles, we can arrange in approximately 5 rows of 4-5 columns
            rows = 5
            cols = 5
            
            # Calculate spacing based on how many circles we can fit
            cell_width = width / cols
            cell_height = height / rows
            
            # Maximum radius based on grid spacing
            max_radius = min(cell_width, cell_height) / 2.0
            
            # Create hexagonal pattern with staggered rows
            idx = 0
            for row in range(rows):
                for col in range(cols):
                    if idx >= n:
                        break
                        
                    # Staggered pattern
                    x_offset = 0.5 * (row % 2)
                    x = (col + 0.5 + x_offset) * cell_width
                    y = (row + 0.5) * cell_height
                    
                    # Ensure within bounds
                    x = max(max_radius, min(width - max_radius, x))
                    y = max(max_radius, min(height - max_radius, y))
                    
                    # Use slightly varied radii to improve packing
                    r = max_radius * random.uniform(0.8, 0.95)
                    
                    circles[idx] = [x, y, r]
                    idx += 1
                    
                if idx >= n:
                    break
            
            # Fill remaining circles
            while idx < n:
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                r = random.uniform(max_radius * 0.7, max_radius * 0.9)
                circles[idx] = [x, y, r]
                idx += 1
            
            # Evaluate this configuration
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
                best_width = width
                best_height = height
        
        return best_circles, best_width, best_height
    
    # Even better initialization using a more strategic approach
    def generate_strategic_initial_solution():
        # Create a more balanced initialization that considers the constraint
        # of having perimeter = 4 (so width + height = 2)
        
        # Try a range of aspect ratios
        aspect_ratios_to_try = [0.8, 1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
        
        best_circles = None
        best_sum = 0
        best_width = 1.0
        best_height = 1.0
        
        for ratio in aspect_ratios_to_try:
            width = 2.0 / (1.0 + ratio)
            height = width * ratio
            
            # Use a more intelligent approach for circle placement
            circles = np.zeros((n, 3))
            
            # Create a more uniform distribution using a grid-like approach
            # but with strategic adjustments
            rows = 5
            cols = 5
            
            # Adjust grid size based on actual number of circles
            actual_rows = min(rows, int(np.ceil(np.sqrt(n))))
            actual_cols = min(cols, int(np.ceil(n / actual_rows)))
            
            if actual_rows * actual_cols < n:
                actual_rows = int(np.ceil(n / actual_cols))
            
            cell_width = width / actual_cols
            cell_height = height / actual_rows
            
            # Maximum possible radius
            max_radius = min(cell_width, cell_height) / 2.0
            
            # Place circles with slight randomization to avoid perfect grid
            idx = 0
            for row in range(actual_rows):
                for col in range(actual_cols):
                    if idx >= n:
                        break
                        
                    # Position with slight offset for better packing
                    x = (col + 0.5 + random.uniform(-0.1, 0.1)) * cell_width
                    y = (row + 0.5 + random.uniform(-0.1, 0.1)) * cell_height
                    
                    # Ensure within bounds
                    x = max(max_radius, min(width - max_radius, x))
                    y = max(max_radius, min(height - max_radius, y))
                    
                    # Use a more diverse range of radii
                    r = max_radius * random.uniform(0.7, 0.95)
                    
                    circles[idx] = [x, y, r]
                    idx += 1
                    
                if idx >= n:
                    break
            
            # Fill remaining circles
            while idx < n:
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                r = random.uniform(max_radius * 0.6, max_radius * 0.9)
                circles[idx] = [x, y, r]
                idx += 1
            
            # Evaluate
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
                best_width = width
                best_height = height
        
        return best_circles, best_width, best_height
    
    # Enhanced optimization approach with better handling of constraints
    def optimize_with_improved_strategies(initial_circles, width, height):
        # Strategy: Multiple-stage optimization with better constraint handling
        bounds = []
        for i in range(n):
            # x bounds
            bounds.extend([(0, width), (0, height), (1e-6, width/2)])
        
        # Create constraints more efficiently
        def distance_constraint(params):
            circles = params.reshape(-1, 3)
            # Vectorized distance computation for efficiency
            diff = circles[:, np.newaxis, :2] - circles[np.newaxis, :, :2]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            constraints = []
            # Only compute upper triangle to avoid duplicates
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    radii_sum = circles[i, 2] + circles[j, 2]
                    constraints.append(dist - radii_sum)
            return np.array(constraints)
        
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
            # Method 1: L-BFGS-B first for global search
            result1 = minimize(
                objective,
                initial_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result1.success:
                # Then SLSQP with constraints for better results
                result2 = minimize(
                    objective,
                    result1.x,
                    method='SLSQP',
                    constraints=[distance_cons, bound_cons],
                    options={'maxiter': 300, 'ftol': 1e-8, 'eps': 1e-8},
                    bounds=bounds
                )
                
                if result2.success:
                    refined_circles = result2.x.reshape(-1, 3)
                    return refined_circles, True
        except Exception as e:
            pass
        
        # Fallback to direct optimization with initial solution
        try:
            result = minimize(
                objective,
                initial_circles.flatten(),
                method='SLSQP',
                constraints=[distance_cons, bound_cons],
                options={'maxiter': 200, 'ftol': 1e-8, 'eps': 1e-8},
                bounds=bounds
            )
            
            if result.success:
                refined_circles = result.x.reshape(-1, 3)
                return refined_circles, True
        except Exception as e:
            pass
        
        return initial_circles, False
    
    # Evolutionary algorithm approach with better fitness evaluation
    def evolutionary_approach():
        # Define the problem as an optimization problem
        toolbox = base.Toolbox()
        
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Individual creation function with proper bounds
        def create_individual():
            individual = []
            # Create circles with proper bounds
            for i in range(n):
                # Position within bounds (with margin for radius)
                x = random.uniform(0.01, 1.99)
                y = random.uniform(0.01, 1.99)
                r = random.uniform(0.01, 0.5)
                individual.extend([x, y, r])
            
            # Add rectangle dimensions (aspect ratio ~1.0 for perimeter 4)
            width = random.uniform(0.8, 1.2)
            height = 2.0 - width  # Maintain perimeter constraint
            individual.extend([width, height])
            
            return creator.Individual(individual)
        
        # Fitness function for EA - improved with better penalty
        def evaluate(individual):
            # Extract circles and rectangle dimensions
            circles_data = individual[:-2]
            width = individual[-2]
            height = individual[-1]
            
            # Ensure perimeter constraint
            if abs(width + height - 2.0) > 0.01:
                # Adjust to satisfy constraint
                width = max(0.1, min(1.9, width))
                height = 2.0 - width
            
            # Create circles array
            circles = np.array(circles_data).reshape(-1, 3)
            
            # Apply boundary constraints
            for i in range(len(circles)):
                circles[i, 0] = max(circles[i, 2], min(width - circles[i, 2], circles[i, 0]))
                circles[i, 1] = max(circles[i, 2], min(height - circles[i, 2], circles[i, 1]))
            
            # Calculate overlap penalty using a more sophisticated approach
            penalty = 0
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    radii_sum = circles[i, 2] + circles[j, 2]
                    if distance < radii_sum:
                        # Use softer penalty to allow for gradual improvement
                        overlap = radii_sum - distance
                        penalty += overlap * overlap * 100  # Quadratic penalty
            
            # Calculate sum of radii minus penalty
            total_radii = np.sum(circles[:, 2])
            return (total_radii - penalty,)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution with better parameters
        population = toolbox.population(n=30)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3, 
                                                    ngen=25, stats=stats, halloffame=hof, verbose=False)
            best_individual = hof[0]
            return best_individual
        except:
            return None
    
    # Try multiple strategies with better rectangle optimization
    best_circles = None
    best_sum = 0
    best_width = 1.0
    best_height = 1.0
    
    # Test multiple initialization strategies
    init_methods = [generate_hexagonal_initial_solution, generate_strategic_initial_solution]
    
    for init_method in init_methods:
        try:
            circles, width, height = init_method()
            
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
    
    # Additional refinement with evolutionary approach - use a more targeted strategy
    try:
        # Try evolutionary approach to find even better solutions
        ea_result = evolutionary_approach()
        if ea_result is not None:
            # Extract circles and rectangle dimensions from EA result
            circles_data = ea_result[:-2]
            width = ea_result[-2]
            height = ea_result[-1]
            
            # Ensure perimeter constraint
            if abs(width + height - 2.0) > 0.01:
                width = max(0.1, min(1.9, width))
                height = 2.0 - width
            
            # Create circles array
            circles = np.array(circles_data).reshape(-1, 3)
            
            # Apply boundary constraints
            for i in range(len(circles)):
                circles[i, 0] = max(circles[i, 2], min(width - circles[i, 2], circles[i, 0]))
                circles[i, 1] = max(circles[i, 2], min(height - circles[i, 2], circles[i, 1]))
            
            # Calculate sum of radii
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
                best_width = width
                best_height = height
    except Exception as e:
        pass
    
    # Final optimization with improved algorithm
    if best_circles is not None:
        # Create final bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0, best_width), (0, best_height), (1e-6, best_width/2)])
        
        # Recreate constraints with more efficient implementation
        def final_constraint_distance(params):
            circles = params.reshape(-1, 3)
            # Vectorized distance computation
            diff = circles[:, np.newaxis, :2] - circles[np.newaxis, :, :2]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            constraints = []
            # Only compute upper triangle to avoid duplicates
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    radii_sum = circles[i, 2] + circles[j, 2]
                    constraints.append(dist - radii_sum)
            return np.array(constraints)
        
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
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result1.success:
                # Then SLSQP with constraints
                result2 = minimize(
                    objective,
                    result1.x,
                    method='SLSQP',
                    constraints=[distance_cons, bound_cons],
                    options={'maxiter': 200, 'ftol': 1e-8, 'eps': 1e-8},
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
            circles[i, 2] = max(1e-6, circles[i, 2])
            # Ensure positions are valid
            circles[i, 0] = max(circles[i, 2], min(best_width - circles[i, 2], circles[i, 0]))
            circles[i, 1] = max(circles[i, 2], min(best_height - circles[i, 2], circles[i, 1]))
        
        return circles
    
    # Fallback to initial solution
    circles, _, _ = generate_hexagonal_initial_solution()
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
