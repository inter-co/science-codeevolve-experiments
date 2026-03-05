# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import time
from typing import Tuple
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
from itertools import combinations
from deap import base, creator, tools, algorithms
import multiprocessing as mp
import copy
from scipy.spatial import Voronoi
import math
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining smart initialization with evolutionary algorithm optimization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Optimize rectangle dimensions - try different ratios to find better packing
    # Based on research and previous results, a rectangle with ratio ~1.33:1 (square-like) often works well
    width, height = 1.0, 1.0  # Square rectangle with perimeter 4
    
    # Number of circles
    n = 21
    
    # Helper function to check if a circle fits within the rectangle
    def is_valid_circle(x, y, r):
        return (r <= x <= width - r and 
                r <= y <= height - r)
    
    # Helper function to compute total radius sum
    def compute_radius_sum(circles_array):
        return np.sum(circles_array[:, 2])
    
    # More efficient overlap checking using spatial data structures
    def check_all_overlaps_fast(circles_array):
        # Use spatial indexing for faster overlap detection
        if len(circles_array) < 2:
            return False
            
        # Create KDTree for fast nearest neighbor queries
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        # Build KDTree once
        tree = cKDTree(positions)
        
        # Check for overlaps using tree queries
        # For each circle, find neighbors within 2*(max_radius) distance
        max_radius = np.max(radii)
        pairs = tree.query_pairs(2 * max_radius)
        
        # Verify actual overlaps
        for i, j in pairs:
            x1, y1, r1 = circles_array[i]
            x2, y2, r2 = circles_array[j]
            distance_sq = (x2 - x1)**2 + (y2 - y1)**2
            min_distance_sq = (r1 + r2)**2
            if distance_sq < min_distance_sq:
                return True
        return False
    
    # Even faster overlap checking for small arrays
    def check_all_overlaps_simple(circles_array):
        # For small arrays, use simple pairwise checking
        if len(circles_array) < 2:
            return False
            
        # Get all circle positions and radii
        positions = circles_array[:, :2]
        radii = circles_array[:, 2]
        
        # Compute pairwise distances
        distances = cdist(positions, positions)
        
        # Compute minimum distances needed to avoid overlap
        min_distances = np.outer(radii, np.ones_like(radii)) + np.outer(np.ones_like(radii), radii)
        
        # Set diagonal to infinity to ignore self-overlaps
        np.fill_diagonal(distances, np.inf)
        
        # Check if any circles overlap
        return np.any(distances < min_distances)
    
    # Better initialization using a more systematic approach
    def initialize_better_pattern():
        circles = []
        
        # Try a hexagonal packing approach which is known to be efficient
        rows = 5
        cols = 5
        
        # Calculate spacing for hexagonal packing
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)
        
        # Place circles in a hexagonal pattern
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Offset every other row for hexagonal packing
                offset = 0 if i % 2 == 0 else spacing_x * 0.5
                x = offset + (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Add some randomness to positions to avoid perfect patterns
                x += random.uniform(-spacing_x * 0.15, spacing_x * 0.15)
                y += random.uniform(-spacing_y * 0.15, spacing_y * 0.15)
                
                # Calculate max possible radius
                max_r = min(x, width - x, y, height - y)
                if max_r > 0.005:  # Minimum radius threshold
                    # Use a more adaptive approach to radius selection
                    r = max_r * (0.4 + 0.4 * random.random())
                    
                    # Check if this circle fits and doesn't overlap with existing ones
                    candidate_circle = [x, y, r]
                    temp_circles = circles + [candidate_circle]
                    temp_array = np.array(temp_circles)
                    
                    if is_valid_circle(x, y, r) and not check_all_overlaps_simple(temp_array):
                        circles.append(candidate_circle)
        
        # Fill remaining positions with strategic random placement
        attempts = 0
        while len(circles) < n and attempts < 1000:
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            # Calculate max possible radius
            max_r = min(x, width - x, y, height - y)
            if max_r > 0.005:
                r = max_r * (0.3 + 0.4 * random.random())
                if is_valid_circle(x, y, r):
                    # Check overlap before adding
                    candidate_circle = [x, y, r]
                    temp_circles = circles + [candidate_circle]
                    temp_array = np.array(temp_circles)
                    if not check_all_overlaps_simple(temp_array):
                        circles.append(candidate_circle)
            attempts += 1
        
        # If still not enough circles, fill with minimal valid circles
        while len(circles) < n:
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            max_r = min(x, width - x, y, height - y)
            if max_r > 0.005:
                r = max_r * 0.15
                if is_valid_circle(x, y, r):
                    circles.append([x, y, r])
        
        return np.array(circles)
    
    # Evolutionary Algorithm approach for global optimization
    def evolutionary_optimization(initial_circles):
        # Create DEAP toolbox
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Define individual representation: [x1,y1,r1,x2,y2,r2,...,x21,y21,r21]
        def create_individual():
            # Start with initial circles
            individual = []
            for i in range(n):
                x, y, r = initial_circles[i]
                individual.extend([x, y, r])
            return creator.Individual(individual)
        
        def evaluate(individual):
            # Convert individual back to circles array
            circles_array = np.array(individual).reshape(-1, 3)
            
            # Check constraints
            if check_all_overlaps_simple(circles_array):
                return -1000000,  # Penalty for overlaps
            
            # Check boundary constraints
            for i in range(n):
                x, y, r = circles_array[i]
                if not is_valid_circle(x, y, r):
                    return -1000000,  # Penalty for boundary violations
            
            # Return negative sum (since we're minimizing in DEAP)
            return compute_radius_sum(circles_array),
        
        def mutate(individual):
            # Mutate by slightly adjusting positions and radii
            for i in range(len(individual)):
                if random.random() < 0.3:  # 30% chance to mutate
                    if i % 3 == 0:  # x coordinate
                        individual[i] += random.uniform(-0.03, 0.03)
                        individual[i] = max(0.01, min(width - 0.01, individual[i]))
                    elif i % 3 == 1:  # y coordinate
                        individual[i] += random.uniform(-0.03, 0.03)
                        individual[i] = max(0.01, min(height - 0.01, individual[i]))
                    else:  # radius
                        individual[i] *= random.uniform(0.9, 1.1)
                        individual[i] = max(0.005, min(min(width/2, height/2) - 0.01, individual[i]))
            return individual,
        
        def crossover(ind1, ind2):
            # Simple uniform crossover
            for i in range(len(ind1)):
                if random.random() < 0.5:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
            return ind1, ind2
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create population with fewer individuals for faster computation
        population = toolbox.population(n=30)
        
        # Evolve with fewer generations
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.7, mutpb=0.3, 
                ngen=20, stats=stats, halloffame=hof, verbose=False
            )
            best_individual = hof[0]
            return np.array(best_individual).reshape(-1, 3)
        except:
            return initial_circles
    
    # Improved local refinement using a more robust optimization approach
    def local_refinement(circles_array):
        # Create a flattened parameter vector [x1,y1,r1,x2,y2,r2,...,x21,y21,r21]
        def flatten_circles(circles_array):
            return circles_array.flatten()
        
        def unflatten_circles(params):
            return params.reshape(-1, 3)
        
        # Objective function to maximize (negative because we minimize)
        def objective(params):
            circles_array = unflatten_circles(params)
            return -compute_radius_sum(circles_array)
        
        # Constraint functions
        def constraint_positions(params):
            """Ensure all circles stay within bounds"""
            circles_array = unflatten_circles(params)
            constraints = []
            for i in range(len(circles_array)):
                x, y, r = circles_array[i]
                # Circle must fit within rectangle
                constraints.extend([
                    x - r,  # x >= r
                    width - x - r,  # width - x >= r
                    y - r,  # y >= r
                    height - y - r  # height - y >= r
                ])
            return np.array(constraints)
        
        def constraint_overlaps(params):
            """Ensure no overlaps between circles"""
            circles_array = unflatten_circles(params)
            constraints = []
            for i, j in combinations(range(len(circles_array)), 2):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                distance_sq = (x2 - x1)**2 + (y2 - y1)**2
                min_distance_sq = (r1 + r2)**2
                # We want distance^2 >= min_distance^2, so constraint is distance^2 - min_distance^2 >= 0
                constraints.append(distance_sq - min_distance_sq)
            return np.array(constraints)
        
        # Flatten initial configuration
        initial_params = flatten_circles(circles_array)
        
        # Create bounds for parameters (x, y, r) for each circle
        bounds = []
        for i in range(n):
            # x bounds
            bounds.append((0.001, width - 0.001))
            # y bounds  
            bounds.append((0.001, height - 0.001))
            # r bounds (positive and bounded by available space)
            bounds.append((0.001, min(width/2, height/2) - 0.001))
        
        # Define constraints
        cons = [
            {'type': 'ineq', 'fun': lambda p: constraint_positions(p)},
            {'type': 'ineq', 'fun': lambda p: constraint_overlaps(p)}
        ]
        
        # Try multiple optimization methods
        try:
            # First try SLSQP with good starting point
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 300, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_circles = unflatten_circles(result.x)
                return optimized_circles
        except Exception as e:
            pass
        
        # Fallback to L-BFGS-B if SLSQP fails
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_circles = unflatten_circles(result.x)
                return optimized_circles
        except Exception as e:
            pass
            
        return circles_array
    
    # Enhanced multi-start local search with better initialization strategies
    def enhanced_multi_start_local_search():
        best_circles = None
        best_sum = -np.inf
        
        # Try multiple initialization strategies
        strategies = [
            "hexagonal",
            "random"
        ]
        
        for strategy in strategies:
            for _ in range(5):  # More runs for better exploration
                # Initialize based on strategy
                if strategy == "hexagonal":
                    circles = initialize_better_pattern()
                else:  # random
                    # Initialize with random valid circles
                    circles = []
                    attempts = 0
                    while len(circles) < n and attempts < 1000:
                        x = random.uniform(0.01, width - 0.01)
                        y = random.uniform(0.01, height - 0.01)
                        # Calculate max possible radius
                        max_r = min(x, width - x, y, height - y)
                        if max_r > 0.005:
                            r = max_r * (0.3 + 0.4 * random.random())
                            
                            # Check if circle would fit without overlapping
                            candidate_circle = [x, y, r]
                            temp_circles = circles + [candidate_circle]
                            temp_array = np.array(temp_circles)
                            
                            if is_valid_circle(x, y, r) and not check_all_overlaps_simple(temp_array):
                                circles.append(candidate_circle)
                        attempts += 1
                    
                    # Fill remaining positions
                    while len(circles) < n:
                        x = random.uniform(0.01, width - 0.01)
                        y = random.uniform(0.01, height - 0.01)
                        max_r = min(x, width - x, y, height - y)
                        if max_r > 0.005:
                            r = max_r * 0.15
                            if is_valid_circle(x, y, r):
                                circles.append([x, y, r])
                    circles = np.array(circles)
                
                # Apply local refinement
                refined_circles = local_refinement(circles)
                refined_sum = compute_radius_sum(refined_circles)
                
                if refined_sum > best_sum:
                    best_sum = refined_sum
                    best_circles = refined_circles
                    
        return best_circles
    
    # Run the optimization
    start_time = time.time()
    
    # Use enhanced multi-start local search
    final_circles = enhanced_multi_start_local_search()
    
    # Apply evolutionary optimization for further improvement
    final_circles = evolutionary_optimization(final_circles)
    
    # Final local refinement
    final_circles = local_refinement(final_circles)
    
    # Ensure we have exactly 21 circles
    if len(final_circles) < 21:
        # If somehow we don't have enough, create more
        current_count = len(final_circles)
        additional_circles = []
        for _ in range(21 - current_count):
            x = random.uniform(0.01, width - 0.01)
            y = random.uniform(0.01, height - 0.01)
            max_r = min(x, width - x, y, height - y)
            if max_r > 0.005:
                r = max_r * 0.15
                if is_valid_circle(x, y, r):
                    additional_circles.append([x, y, r])
        
        final_circles = np.vstack([final_circles, additional_circles])
    
    return final_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
