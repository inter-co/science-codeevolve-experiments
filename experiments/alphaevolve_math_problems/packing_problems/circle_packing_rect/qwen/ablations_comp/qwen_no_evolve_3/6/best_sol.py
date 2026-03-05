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
import math
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid evolutionary and optimization approach.

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
    
    # Constraints for non-overlapping
    def constraint_distance(params):
        circles = params.reshape(-1, 3)
        # Use more efficient distance calculation
        distances = cdist(circles[:, :2], circles[:, :2])
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
        # Rectangle dimensions are fixed at 1.0 width, 1.0 height for simplicity
        width, height = 1.0, 1.0
        
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
    
    # More efficient constraint implementation using vectorization
    def fast_constraint_distance(params):
        circles = params.reshape(-1, 3)
        # Vectorized computation of all pairwise distances
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]
        
        # Compute all pairwise distances efficiently
        diff_x = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
        diff_y = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
        distances = np.sqrt(diff_x**2 + diff_y**2)
        
        # Compute radii sums
        radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Get upper triangular part (avoiding double counting)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        constraints = distances[mask] - radii_sum[mask]
        
        return constraints
    
    def fast_constraint_bounds(params):
        circles = params.reshape(-1, 3)
        constraints = []
        width, height = 1.0, 1.0
        
        # Vectorized bounds checking
        constraints.extend(circles[:, 0] - circles[:, 2])  # Left boundary
        constraints.extend(width - circles[:, 0] - circles[:, 2])  # Right boundary
        constraints.extend(circles[:, 1] - circles[:, 2])  # Bottom boundary
        constraints.extend(height - circles[:, 1] - circles[:, 2])  # Top boundary
        
        return np.array(constraints)
    
    # Better initialization using a more sophisticated approach
    def generate_advanced_initial_solution():
        # Use a more systematic approach based on circle packing theory
        # Start with a square grid pattern and then refine
        
        # Rectangle dimensions - we'll try different aspect ratios
        width, height = 1.0, 1.0
        
        # Create a grid pattern for initial placement
        rows = 5
        cols = 5
        if rows * cols < n:
            rows = int(np.ceil(n / cols))
        
        # Adjust grid to fit exactly n circles
        actual_cols = min(cols, n)
        actual_rows = int(np.ceil(n / actual_cols))
        
        # Calculate spacing
        cell_width = width / actual_cols
        cell_height = height / actual_rows
        
        # Maximum radius based on cell size
        max_radius = min(cell_width, cell_height) / 2.0
        
        # Place circles in a grid pattern
        circles = np.zeros((n, 3))
        idx = 0
        
        for row in range(actual_rows):
            for col in range(actual_cols):
                if idx >= n:
                    break
                    
                # Staggered pattern to improve packing
                x_offset = 0.5 * (row % 2)  # Offset every other row
                x = (col + 0.5 + x_offset) * cell_width
                y = (row + 0.5) * cell_height
                
                # Ensure within bounds
                x = max(max_radius, min(width - max_radius, x))
                y = max(max_radius, min(height - max_radius, y))
                
                # Use varying radii for better initial solution
                r = max_radius * random.uniform(0.7, 0.9)
                
                circles[idx] = [x, y, r]
                idx += 1
                
        # Refine the initial solution by slightly adjusting positions
        # and trying to increase radii where possible
        for i in range(n):
            # Try to increase radius if possible without overlap
            # This is a simple heuristic - in practice, we'd want more sophisticated approach
            circles[i, 2] = min(circles[i, 2], 
                               min(circles[i, 0], width - circles[i, 0], 
                                   circles[i, 1], height - circles[i, 1]) * 0.9)
        
        return circles, width, height
    
    # Evolutionary algorithm approach for better global search
    def evolutionary_approach():
        # Define the optimization problem using DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Define bounds for each parameter (x, y, r for each circle)
        # Each circle has 3 parameters: x, y, r
        # Bounds: x in [0, 1], y in [0, 1], r in [0.001, 0.5]
        bounds = []
        for i in range(n):
            bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])
        
        def eval_circle_packing(individual):
            # Convert individual to circles array
            circles = np.array(individual).reshape(-1, 3)
            
            # Check constraints
            # Boundary constraints
            for i in range(n):
                if (circles[i, 0] - circles[i, 2] < 0 or 
                    circles[i, 0] + circles[i, 2] > 1 or
                    circles[i, 1] - circles[i, 2] < 0 or
                    circles[i, 1] + circles[i, 2] > 1):
                    return -1000000,  # Invalid solution
            
            # Non-overlapping constraints
            distances = cdist(circles[:, :2], circles[:, :2])
            penalty = 0
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    radii_sum = circles[i, 2] + circles[j, 2]
                    if dist < radii_sum:
                        # Penalize overlapping circles
                        penalty += (radii_sum - dist) * 1000000
            
            # Objective: maximize sum of radii
            return np.sum(circles[:, 2]) - penalty,  # Return fitness (higher is better)
        
        toolbox.register("attr_float", random.uniform, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n*3)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", eval_circle_packing)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution
        population = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        
        try:
            # Use simple evolution for faster execution
            stats = tools.Statistics(lambda ind: ind.fitness.values[0])
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.5, mutpb=0.2, 
                ngen=30, stats=stats, halloffame=hof, verbose=False
            )
            
            if len(hof) > 0:
                best_individual = hof[0]
                circles = np.array(best_individual).reshape(-1, 3)
                return circles, True
        except Exception as e:
            pass
        
        return None, False
    
    # Enhanced optimization with multiple strategies
    def enhanced_optimization(initial_circles, width, height):
        # Strategy 1: Try evolutionary approach first
        try:
            evolved_circles, success = evolutionary_approach()
            if success and evolved_circles is not None:
                # Evaluate the evolved solution
                current_sum = np.sum(evolved_circles[:, 2])
                return evolved_circles, current_sum, True
        except Exception as e:
            pass
        
        # Strategy 2: Local optimization with better constraints
        bounds = []
        for i in range(n):
            bounds.extend([(0, width), (0, height), (1e-6, width/2)])
        
        # Create constraints dictionary with better error handling
        def distance_constraint(params):
            try:
                circles = params.reshape(-1, 3)
                distances = cdist(circles[:, :2], circles[:, :2])
                constraints = []
                # Only compute upper triangle to avoid duplicates and improve performance
                for i in range(n):
                    for j in range(i+1, n):
                        dist = distances[i, j]
                        radii_sum = circles[i, 2] + circles[j, 2]
                        constraints.append(dist - radii_sum)
                return np.array(constraints)
            except:
                return np.array([0.0] * (n*(n-1)//2))
        
        def bound_constraint(params):
            try:
                circles = params.reshape(-1, 3)
                constraints = []
                for i in range(n):
                    constraints.append(circles[i, 0] - circles[i, 2])
                    constraints.append(width - circles[i, 0] - circles[i, 2])
                    constraints.append(circles[i, 1] - circles[i, 2])
                    constraints.append(height - circles[i, 1] - circles[i, 2])
                return np.array(constraints)
            except:
                return np.array([0.0] * (4*n))
        
        distance_cons = {
            'type': 'ineq',
            'fun': lambda x: distance_constraint(x)
        }
        
        bound_cons = {
            'type': 'ineq', 
            'fun': lambda x: bound_constraint(x)
        }
        
        try:
            # First try with L-BFGS-B for global search
            result1 = minimize(
                objective,
                initial_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result1.success:
                # Then try with SLSQP for better constraint handling
                result2 = minimize(
                    objective,
                    result1.x,
                    method='SLSQP',
                    constraints=[distance_cons, bound_cons],
                    options={'maxiter': 200, 'ftol': 1e-8, 'eps': 1e-8},
                    bounds=bounds
                )
                
                if result2.success:
                    refined_circles = result2.x.reshape(-1, 3)
                    current_sum = np.sum(refined_circles[:, 2])
                    return refined_circles, current_sum, True
        except Exception as e:
            pass
        
        # Fallback to simpler optimization
        try:
            result = minimize(
                objective,
                initial_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                refined_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(refined_circles[:, 2])
                return refined_circles, current_sum, True
        except Exception as e:
            pass
        
        return initial_circles, np.sum(initial_circles[:, 2]), False
    
    # Try multiple initialization strategies and aspect ratios
    best_circles = None
    best_sum = 0
    
    # Try different rectangle aspect ratios
    aspect_ratios = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    
    for ratio in aspect_ratios:
        width = 2.0 / (1.0 + ratio)
        height = width * ratio
        
        # Try advanced initialization
        try:
            circles, _, _ = generate_advanced_initial_solution()
            
            # Optimize this configuration
            optimized_circles, current_sum, success = enhanced_optimization(circles, width, height)
            
            # Check if this is better
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized_circles.copy()
        except Exception as e:
            continue
    
    # Final refinement if we have a solution
    if best_circles is not None:
        # Apply final validation and refinement
        try:
            # Create final bounds
            bounds = []
            width = 1.0  # Fixed for simplicity
            height = 1.0
            
            for i in range(n):
                bounds.extend([(0, width), (0, height), (1e-6, width/2)])
            
            # Create constraints for final optimization
            def final_distance_constraint(params):
                circles = params.reshape(-1, 3)
                distances = cdist(circles[:, :2], circles[:, :2])
                constraints = []
                for i in range(n):
                    for j in range(i+1, n):
                        dist = distances[i, j]
                        radii_sum = circles[i, 2] + circles[j, 2]
                        constraints.append(dist - radii_sum)
                return np.array(constraints)
            
            def final_bound_constraint(params):
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
                'fun': lambda x: final_distance_constraint(x)
            }
            
            bound_cons = {
                'type': 'ineq', 
                'fun': lambda x: final_bound_constraint(x)
            }
            
            # Final optimization
            result = minimize(
                objective,
                best_circles.flatten(),
                method='SLSQP',
                constraints=[distance_cons, bound_cons],
                options={'maxiter': 100, 'ftol': 1e-8, 'eps': 1e-8},
                bounds=bounds
            )
            
            if result.success:
                best_circles = result.x.reshape(-1, 3)
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
            circles[i, 0] = max(circles[i, 2], min(1.0 - circles[i, 2], circles[i, 0]))
            circles[i, 1] = max(circles[i, 2], min(1.0 - circles[i, 2], circles[i, 1]))
        
        return circles
    
    # Fallback to simple initialization
    circles, _, _ = generate_advanced_initial_solution()
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
