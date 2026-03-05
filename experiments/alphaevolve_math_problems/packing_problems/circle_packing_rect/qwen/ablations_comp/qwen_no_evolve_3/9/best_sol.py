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
from numba import jit
import math
from deap import base, creator, tools, algorithms
import multiprocessing as mp
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_distances_numba(circles):
    """Fast distance computation using numba"""
    n = len(circles)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric initialization and global optimization.

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
    
    # Constraint function for non-overlapping
    def constraint_distance(params):
        circles = params.reshape(-1, 3)
        # Use faster distance calculation
        distances = cdist(circles[:, :2], circles[:, :2])
        constraints = []
        # Only check upper triangle to avoid duplicates
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                radii_sum = circles[i, 2] + circles[j, 2]
                # Constraint: distance >= radii_sum (for non-overlap)
                constraints.append(dist - radii_sum)
        return np.array(constraints)
    
    # Constraint function for boundary conditions
    def constraint_bounds(params):
        circles = params.reshape(-1, 3)
        constraints = []
        # Rectangle dimensions will be optimized
        width, height = 1.0, 1.0  # These will be optimized
        
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
    
    # Better initialization using hexagonal packing with adaptive sizing
    def generate_hexagonal_initialization():
        # Try different aspect ratios to find optimal rectangle
        best_aspect_ratio = 1.3
        width = 2.0 / (1.0 + best_aspect_ratio)
        height = width * best_aspect_ratio
        
        # Create hexagonal arrangement
        circles = np.zeros((n, 3))
        
        # Calculate grid size for hexagonal packing
        # For 21 circles, we want something like 5x4 or similar
        rows = int(np.ceil(np.sqrt(n * 1.2)))  # Allow for spacing
        cols = int(np.ceil(n / rows))
        
        # Ensure proper dimensions
        if cols * rows < n:
            rows = int(np.ceil(n / cols))
            
        # Grid spacing
        cell_width = width / cols
        cell_height = height / rows
        
        # Maximum possible radius based on grid spacing
        max_radius = min(cell_width, cell_height) / 2.0
        
        # Place circles in a staggered pattern for better packing
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                    
                # Staggered pattern to improve packing
                x_offset = 0.5 * (row % 2)  # Offset every other row
                x = (col + 0.5 + x_offset) * cell_width
                y = (row + 0.5) * cell_height
                
                # Ensure within bounds
                x = max(max_radius, min(width - max_radius, x))
                y = max(max_radius, min(height - max_radius, y))
                
                # Use a more aggressive radius assignment to start with
                # But make sure they're reasonable
                r = max_radius * random.uniform(0.8, 0.95)
                
                circles[idx] = [x, y, r]
                idx += 1
                
        # Fill remaining circles with strategic placement
        if idx < n:
            for i in range(idx, n):
                # Place in a more scattered pattern with better distribution
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                r = random.uniform(max_radius * 0.7, max_radius * 0.9)
                circles[i] = [x, y, r]
        
        return circles, width, height
    
    # Even better initialization using a more advanced approach
    def generate_advanced_initialization():
        # Start with a more systematic approach using better spacing
        # Try aspect ratios that tend to work well for circle packing
        aspect_ratios_to_try = [0.8, 1.0, 1.2, 1.3, 1.5, 1.8, 2.0]
        best_circles = None
        best_sum = 0
        
        for ratio in aspect_ratios_to_try:
            width = 2.0 / (1.0 + ratio)
            height = width * ratio
            
            # Create a more structured initialization
            circles = np.zeros((n, 3))
            
            # Use a more sophisticated approach: place circles in rings or layers
            # This helps avoid having too many circles in small areas
            
            # First, try placing in a circular pattern with varying radii
            center_x, center_y = width/2, height/2
            max_radius = min(width, height) / 4
            
            # Layered approach: concentric rings
            layer_count = 4
            circles_per_layer = n // layer_count + 1
            
            idx = 0
            for layer in range(layer_count):
                if idx >= n:
                    break
                    
                # Radius for this layer
                layer_radius = (layer + 1) * max_radius / layer_count
                
                # Number of circles in this layer
                angle_step = 2 * np.pi / min(circles_per_layer, n - idx)
                num_circles_in_layer = min(circles_per_layer, n - idx)
                
                for i in range(num_circles_in_layer):
                    if idx >= n:
                        break
                        
                    angle = i * angle_step
                    x = center_x + layer_radius * np.cos(angle)
                    y = center_y + layer_radius * np.sin(angle)
                    
                    # Ensure within bounds
                    x = max(max_radius, min(width - max_radius, x))
                    y = max(max_radius, min(height - max_radius, y))
                    
                    # Radius decreases with distance from center
                    r = max_radius * 0.7 * (1 - layer / layer_count) * random.uniform(0.8, 1.0)
                    r = max(0.01, min(max_radius * 0.5, r))
                    
                    circles[idx] = [x, y, r]
                    idx += 1
            
            # Fill remaining circles
            if idx < n:
                for i in range(idx, n):
                    x = random.uniform(max_radius, width - max_radius)
                    y = random.uniform(max_radius, height - max_radius)
                    r = random.uniform(max_radius * 0.5, max_radius * 0.8)
                    circles[i] = [x, y, r]
            
            # Evaluate this configuration
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
        
        if best_circles is None:
            # Fallback to hexagonal
            return generate_hexagonal_initialization()
        
        return best_circles, width, height
    
    # Evolutionary algorithm approach for better optimization
    def evolutionary_optimization(initial_circles, width, height):
        # Create a more efficient representation for EA
        def create_individual():
            individual = []
            for i in range(n):
                # x, y, radius
                x = random.uniform(0, width)
                y = random.uniform(0, height)
                r = random.uniform(0.01, min(width, height) / 4)
                individual.extend([x, y, r])
            return individual
        
        def evaluate(individual):
            # Convert to circles array
            circles = np.array(individual).reshape(-1, 3)
            
            # Check constraints
            for i in range(n):
                if (circles[i, 0] - circles[i, 2] < 0 or 
                    circles[i, 0] + circles[i, 2] > width or
                    circles[i, 1] - circles[i, 2] < 0 or 
                    circles[i, 1] + circles[i, 2] > height):
                    return float('inf')  # Invalid solution
            
            # Check overlap constraints
            distances = cdist(circles[:, :2], circles[:, :2])
            penalty = 0
            for i in range(n):
                for j in range(i+1, n):
                    dist = distances[i, j]
                    radii_sum = circles[i, 2] + circles[j, 2]
                    if dist < radii_sum:
                        # Penalty based on how much they overlap
                        penalty += (radii_sum - dist) * 1000
            
            # Return negative sum of radii (we want to maximize)
            return -np.sum(circles[:, 2]) + penalty
        
        # Setup evolutionary algorithm
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        toolbox.register("individual", creator.Individual, create_individual())
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution
        population = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        
        # Use a smaller number of generations to stay within time limits
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.5, mutpb=0.2, 
                ngen=30, stats=stats, halloffame=hof, verbose=False
            )
            
            if hof:
                best_individual = hof[0]
                circles = np.array(best_individual).reshape(-1, 3)
                return circles, True
        except Exception:
            pass
        
        return initial_circles, False
    
    # Enhanced optimization approach with better handling of constraints
    def optimize_with_improved_strategies(initial_circles, width, height):
        # Strategy 1: Try evolutionary approach first
        try:
            evolved_circles, evolved_success = evolutionary_optimization(initial_circles, width, height)
            if evolved_success:
                current_sum = np.sum(evolved_circles[:, 2])
                initial_sum = np.sum(initial_circles[:, 2])
                if current_sum > initial_sum * 1.05:  # Only accept significant improvements
                    return evolved_circles, True
        except Exception:
            pass
        
        # Strategy 2: Use global optimization with bounds
        bounds = []
        for i in range(n):
            # x bounds
            bounds.extend([(0, width), (0, height), (1e-6, width/2)])
        
        # Create constraints dictionary
        def distance_constraint(params):
            circles = params.reshape(-1, 3)
            distances = cdist(circles[:, :2], circles[:, :2])
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
                # x - r >= 0 (left boundary)
                constraints.append(circles[i, 0] - circles[i, 2])
                # width - x - r >= 0 (right boundary)  
                constraints.append(width - circles[i, 0] - circles[i, 2])
                # y - r >= 0 (bottom boundary)
                constraints.append(circles[i, 1] - circles[i, 2])
                # height - y - r >= 0 (top boundary)
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
        
        # Try different optimization approaches
        try:
            # First, try L-BFGS-B for global search
            result1 = minimize(
                objective,
                initial_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result1.success:
                # Then try SLSQP with constraints for fine-tuning
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
    
    # Try multiple strategies with better rectangle optimization
    best_circles = None
    best_sum = 0
    best_width = 1.0
    best_height = 1.0
    
    # Test a wider range of aspect ratios with more focus on promising ones
    aspect_ratios = [0.6, 0.8, 1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]
    
    # Try several initialization approaches
    init_methods = [generate_hexagonal_initialization, generate_advanced_initialization]
    
    # Try multiple random starting points for better exploration
    for attempt in range(3):
        for ratio in aspect_ratios:
            width = 2.0 / (1.0 + ratio)
            height = width * ratio
            
            # Try multiple initialization methods
            for init_method in init_methods:
                try:
                    circles, _, _ = init_method()
                    
                    # Add some randomness to initial configuration
                    if attempt > 0:
                        for i in range(n):
                            circles[i, 0] += random.uniform(-0.1, 0.1) * width
                            circles[i, 1] += random.uniform(-0.1, 0.1) * height
                            circles[i, 2] *= random.uniform(0.9, 1.1)
                    
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
    
    # Final refinement with improved algorithm
    if best_circles is not None:
        # Create final bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0, best_width), (0, best_height), (1e-6, best_width/2)])
        
        # Recreate constraints with more efficient implementation
        def final_constraint_distance(params):
            circles = params.reshape(-1, 3)
            distances = cdist(circles[:, :2], circles[:, :2])
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
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result1.success:
                # Then SLSQP with constraints
                result2 = minimize(
                    objective,
                    result1.x,
                    method='SLSQP',
                    constraints=[distance_cons, bound_cons],
                    options={'maxiter': 300, 'ftol': 1e-8, 'eps': 1e-8},
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
    circles, _, _ = generate_hexagonal_initialization()
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
