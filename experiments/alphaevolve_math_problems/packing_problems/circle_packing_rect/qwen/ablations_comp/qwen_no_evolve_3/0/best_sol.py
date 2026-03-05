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
from scipy.spatial import Voronoi, voronoi_plot_2d
import heapq
from numba import jit
warnings.filterwarnings('ignore')

@jit(nopython=True)
def fast_distance_matrix(xys):
    """Fast computation of pairwise Euclidean distances"""
    n = xys.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = xys[i, 0] - xys[j, 0]
            dy = xys[i, 1] - xys[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric initialization and efficient local optimization.

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
    
    # More efficient constraint handling using vectorized operations
    def constraint_distance(params):
        circles = params.reshape(-1, 3)
        # Use vectorized distance calculation
        distances = cdist(circles[:, :2], circles[:, :2])
        constraints = []
        # Only compute upper triangle to avoid duplicates and self-interactions
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                radii_sum = circles[i, 2] + circles[j, 2]
                # Constraint: distance >= radii_sum (for non-overlap)
                constraints.append(dist - radii_sum)
        return np.array(constraints)
    
    # Boundary constraints
    def constraint_bounds(params, width, height):
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
    
    # Generate initial configuration using a more sophisticated approach
    def generate_advanced_initialization():
        # Try multiple aspect ratios to find good configuration
        aspect_ratios = [0.8, 1.0, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
        best_circles = None
        best_sum = 0
        best_ratio = 1.0
        
        # Precompute potential circle arrangements for different aspect ratios
        for ratio in aspect_ratios:
            width = 2.0 / (1.0 + ratio)
            height = width * ratio
            
            # Create a grid-based initialization with better spacing
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            
            # Ensure we have enough space for all circles
            if rows * cols < n:
                rows = int(np.ceil(n / cols))
            
            # Calculate spacing
            cell_width = width / cols
            cell_height = height / rows
            
            # Max radius based on spacing
            max_radius = min(cell_width, cell_height) / 2.0
            
            # Create circles with more strategic positioning
            circles = np.zeros((n, 3))
            idx = 0
            
            # Use a more sophisticated grid with staggered rows
            for row in range(rows):
                for col in range(cols):
                    if idx >= n:
                        break
                    # Stagger odd rows for better packing
                    offset = 0.5 * (row % 2)
                    x = (col + 0.5 + offset) * cell_width
                    y = (row + 0.5) * cell_height
                    
                    # Keep within bounds
                    x = max(max_radius, min(width - max_radius, x))
                    y = max(max_radius, min(height - max_radius, y))
                    
                    # Use a more conservative but effective radius
                    r = max_radius * random.uniform(0.85, 0.95)
                    circles[idx] = [x, y, r]
                    idx += 1
            
            # Fill any remaining spots
            while idx < n:
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                r = max_radius * random.uniform(0.8, 0.9)
                circles[idx] = [x, y, r]
                idx += 1
            
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
                best_ratio = ratio
        
        if best_circles is not None:
            width = 2.0 / (1.0 + best_ratio)
            height = width * best_ratio
            return best_circles, width, height
        else:
            # Fallback to basic grid
            width = 1.0
            height = 1.0
            circles = np.zeros((n, 3))
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            cell_width = width / cols
            cell_height = height / rows
            max_radius = min(cell_width, cell_height) / 2.0
            
            idx = 0
            for row in range(rows):
                for col in range(cols):
                    if idx >= n:
                        break
                    x = (col + 0.5) * cell_width
                    y = (row + 0.5) * cell_height
                    x = max(max_radius, min(width - max_radius, x))
                    y = max(max_radius, min(height - max_radius, y))
                    r = max_radius * random.uniform(0.85, 0.95)
                    circles[idx] = [x, y, r]
                    idx += 1
                    
            while idx < n:
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                r = max_radius * random.uniform(0.8, 0.9)
                circles[idx] = [x, y, r]
                idx += 1
            
            return circles, width, height
    
    # Improved optimization with better handling of constraints
    def optimize_with_improved_sqp(initial_circles, width, height):
        # Create bounds for optimization
        bounds = []
        for i in range(n):
            bounds.extend([(0, width), (0, height), (1e-6, width/2)])
        
        # Create constraint dictionaries with better error handling
        def distance_constraint(params):
            try:
                circles = params.reshape(-1, 3)
                distances = cdist(circles[:, :2], circles[:, :2])
                constraints = []
                for i in range(n):
                    for j in range(i+1, n):
                        dist = distances[i, j]
                        radii_sum = circles[i, 2] + circles[j, 2]
                        constraints.append(dist - radii_sum)
                return np.array(constraints)
            except:
                # Return a large penalty if constraint evaluation fails
                return np.full(n*(n-1)//2, -1000.0)
        
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
                return np.full(4*n, -1000.0)
        
        # Create constraint objects
        distance_cons = {
            'type': 'ineq',
            'fun': distance_constraint
        }
        
        bound_cons = {
            'type': 'ineq', 
            'fun': bound_constraint
        }
        
        # Try multiple optimization approaches
        try:
            # First try L-BFGS-B for global search with tighter tolerances
            result1 = minimize(
                objective,
                initial_circles.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            
            if result1.success:
                # Then use SLSQP with constraints for fine tuning
                result2 = minimize(
                    objective,
                    result1.x,
                    method='SLSQP',
                    constraints=[distance_cons, bound_cons],
                    options={'maxiter': 500, 'ftol': 1e-12, 'eps': 1e-12},
                    bounds=bounds
                )
                
                if result2.success:
                    return result2.x.reshape(-1, 3)
        except Exception as e:
            pass
        
        # Try a different approach with trust-constr
        try:
            result = minimize(
                objective,
                initial_circles.flatten(),
                method='trust-constr',
                constraints=[distance_cons, bound_cons],
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12},
                bounds=bounds
            )
            
            if result.success:
                return result.x.reshape(-1, 3)
        except Exception as e:
            pass
        
        # Fallback to basic optimization
        try:
            result = minimize(
                objective,
                initial_circles.flatten(),
                method='SLSQP',
                constraints=[distance_cons, bound_cons],
                options={'maxiter': 200, 'ftol': 1e-10, 'eps': 1e-10},
                bounds=bounds
            )
            
            if result.success:
                return result.x.reshape(-1, 3)
        except Exception as e:
            pass
        
        return initial_circles
    
    # Advanced clustering-based initialization
    def generate_clustered_initialization():
        # Generate points using k-means clustering approach
        width = 1.0
        height = 1.0
        
        # Create a set of candidate points
        candidates = []
        for _ in range(1000):
            x = random.uniform(0.1, width - 0.1)
            y = random.uniform(0.1, height - 0.1)
            candidates.append([x, y])
        
        # Cluster these points to find good distribution
        candidates = np.array(candidates)
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(candidates)
        
        # Get cluster centers
        centers = kmeans.cluster_centers_
        
        # Initialize circles around these centers with varying radii
        circles = np.zeros((n, 3))
        for i in range(n):
            x, y = centers[i]
            # Use smaller radius near cluster centers
            r = random.uniform(0.05, 0.15)
            circles[i] = [x, y, r]
        
        return circles, width, height
    
    # Enhanced evolutionary algorithm approach for global optimization
    def evolutionary_optimization():
        # Set up DEAP for evolutionary optimization
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Define individual generation with better bounds
        def create_individual():
            # Generate random circles with more careful positioning
            circles = []
            for _ in range(n):
                # Use a wider range but then clip to valid bounds
                x = random.uniform(0.0, 2.0)
                y = random.uniform(0.0, 2.0)
                r = random.uniform(0.01, 0.4)
                circles.extend([x, y, r])
            return creator.Individual(circles)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Fitness function with improved penalty system
        def eval_fitness(individual):
            circles = np.array(individual).reshape(-1, 3)
            
            # Check constraints
            total_radius = np.sum(circles[:, 2])
            
            # Penalize overlapping circles with a more sophisticated penalty
            penalty = 0
            for i in range(n):
                for j in range(i+1, n):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    radii_sum = circles[i, 2] + circles[j, 2]
                    
                    if distance < radii_sum:
                        overlap = radii_sum - distance
                        # Quadratic penalty for overlap
                        penalty += overlap * overlap * 10000
            
            # Penalize boundary violations
            for i in range(n):
                if (circles[i, 0] - circles[i, 2] < 0 or 
                    circles[i, 0] + circles[i, 2] > 2 or
                    circles[i, 1] - circles[i, 2] < 0 or
                    circles[i, 1] + circles[i, 2] > 2):
                    # Linear penalty for boundary violations
                    penalty += 50000
            
            # Add penalty for very small radii to encourage larger circles
            for i in range(n):
                if circles[i, 2] < 0.02:
                    penalty += 1000 * (0.02 - circles[i, 2])
            
            return (total_radius - penalty,)
        
        toolbox.register("evaluate", eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Run evolution with more generations and better parameters
        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, 
                                             ngen=50, stats=stats, halloffame=hof, 
                                             verbose=False)
            return np.array(hof[0]).reshape(-1, 3)
        except:
            return None
    
    # Improved initialization strategy with hexagonal packing approach
    def generate_hexagonal_initialization():
        # Try a hexagonal pattern which often works well for circle packing
        width = 1.5
        height = 1.5
        aspect_ratio = width / height
        
        # Hexagonal packing parameters
        rows = int(np.ceil(np.sqrt(n) * 1.2))
        cols = int(np.ceil(n / rows))
        
        # Make sure we have enough space
        if rows * cols < n:
            rows = int(np.ceil(n / cols))
        
        # Calculate cell size
        cell_width = width / cols
        cell_height = height / rows
        
        # Adjust for hexagonal packing
        hex_radius = min(cell_width, cell_height) * 0.4  # Reduce slightly for better packing
        
        circles = np.zeros((n, 3))
        idx = 0
        
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset for alternate rows
                offset = (row % 2) * (cell_width / 2)
                x = (col + 0.5 + offset) * cell_width
                y = (row + 0.5) * cell_height
                
                # Keep within bounds
                x = max(hex_radius, min(width - hex_radius, x))
                y = max(hex_radius, min(height - hex_radius, y))
                
                # Use a slightly smaller radius for better spacing
                r = hex_radius * random.uniform(0.9, 1.0)
                circles[idx] = [x, y, r]
                idx += 1
        
        # Fill any remaining spots
        while idx < n:
            x = random.uniform(hex_radius, width - hex_radius)
            y = random.uniform(hex_radius, height - hex_radius)
            r = hex_radius * random.uniform(0.8, 0.9)
            circles[idx] = [x, y, r]
            idx += 1
        
        # Scale to perimeter = 4
        scale_factor = 2.0 / (width + height)
        circles[:, 0] *= scale_factor
        circles[:, 1] *= scale_factor
        circles[:, 2] *= scale_factor
        
        return circles, width * scale_factor, height * scale_factor
    
    # Main optimization routine with improved strategies
    best_circles = None
    best_sum = 0
    best_width = 1.0
    best_height = 1.0
    
    # Strategy 1: Hexagonal initialization followed by local optimization
    try:
        circles, width, height = generate_hexagonal_initialization()
        optimized_circles = optimize_with_improved_sqp(circles, width, height)
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
            best_width = width
            best_height = height
    except Exception as e:
        pass
    
    # Strategy 2: Advanced initialization followed by local optimization
    try:
        circles, width, height = generate_advanced_initialization()
        optimized_circles = optimize_with_improved_sqp(circles, width, height)
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
            best_width = width
            best_height = height
    except Exception as e:
        pass
    
    # Strategy 3: Clustered initialization
    try:
        circles, width, height = generate_clustered_initialization()
        optimized_circles = optimize_with_improved_sqp(circles, width, height)
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
            best_width = width
            best_height = height
    except Exception as e:
        pass
    
    # Strategy 4: Evolutionary optimization (as backup)
    try:
        evolved_circles = evolutionary_optimization()
        if evolved_circles is not None:
            current_sum = np.sum(evolved_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = evolved_circles.copy()
                best_width = 1.0
                best_height = 1.0
    except Exception as e:
        pass
    
    # Final validation and cleanup
    if best_circles is not None:
        # Validate constraints and fix any boundary violations
        circles = best_circles.copy()
        for i in range(len(circles)):
            # Ensure radii are positive and reasonable
            circles[i, 2] = max(1e-6, min(circles[i, 2], best_width/2, best_height/2))
            # Ensure positions are valid
            circles[i, 0] = max(circles[i, 2], min(best_width - circles[i, 2], circles[i, 0]))
            circles[i, 1] = max(circles[i, 2], min(best_height - circles[i, 2], circles[i, 1]))
        
        return circles
    
    # Fallback to simple initialization
    circles, _, _ = generate_advanced_initialization()
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
