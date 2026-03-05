# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import math
from scipy.spatial import distance_matrix
import warnings
from deap import base, creator, tools, algorithms
import random
from multiprocessing import Pool
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses evolutionary algorithm with local refinement for better optimization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Try different aspect ratios - test several candidates to find optimal
    # Based on circle packing theory, a square (1x1) is often good, but let's try some ratios
    candidates = [(1.0, 1.0), (1.2, 0.8), (0.8, 1.2), (1.5, 0.5), (0.5, 1.5)]
    best_result = None
    best_sum = 0
    
    # Test multiple rectangle configurations
    for width, height in candidates:
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Problem constants
        n = 21
        max_radius = min(width, height) / 4.0
        
        # Define DEAP structures for evolutionary algorithm
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Define bounds for each parameter: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
        def create_individual():
            individual = []
            for i in range(n):
                # x coordinate between 0.01 and width - 0.01
                x = random.uniform(0.01, width - 0.01)
                # y coordinate between 0.01 and height - 0.01
                y = random.uniform(0.01, height - 0.01)
                # radius between 0.01 and max_radius
                r = random.uniform(0.01, max_radius)
                individual.extend([x, y, r])
            return creator.Individual(individual)
        
        def evaluate(individual):
            """Evaluate fitness of an individual (sum of radii)"""
            # Convert individual to circles array
            circles = np.array(individual).reshape(-1, 3)
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # Check boundary constraints
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                    return (0,)  # Invalid configuration
            
            # Check overlap constraints using a more efficient approach
            # Use vectorized operations for better performance
            penalty = 0
            
            # Vectorized overlap checking - compute all pairwise distances efficiently
            # We'll use a smarter approach: only check close neighbors
            try:
                # Create distance matrix for all pairs
                dist_matrix = cdist(positions, positions)
                
                # For each pair, check if they overlap
                for i in range(n):
                    for j in range(i+1, n):
                        dist = dist_matrix[i, j]
                        min_dist = radii[i] + radii[j]
                        if dist < min_dist:
                            # Penalty for overlapping circles - quadratic penalty
                            overlap = min_dist - dist
                            penalty += overlap * overlap * 1000  # Strong penalty
                            
            except Exception:
                # Fallback to pairwise checking if vectorization fails
                for i in range(n):
                    for j in range(i+1, n):
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        dist_sq = dx*dx + dy*dy
                        min_dist = radii[i] + radii[j]
                        if dist_sq < min_dist * min_dist:  # Compare squared distances to avoid sqrt
                            # Penalty for overlapping circles - quadratic penalty
                            overlap = min_dist - math.sqrt(dist_sq)
                            penalty += overlap * overlap * 1000  # Strong penalty
            
            # Fitness is sum of radii minus penalty
            total_radius = np.sum(radii)
            fitness = total_radius - penalty
            
            # Ensure fitness is not negative
            return (max(0, fitness),)
        
        def mutate(individual):
            """Mutate an individual"""
            for i in range(len(individual)):
                if random.random() < 0.15:  # Increased mutation rate
                    if i % 3 == 0:  # x coordinate
                        individual[i] = max(0.01, min(width - 0.01, individual[i] + random.gauss(0, 0.03)))
                    elif i % 3 == 1:  # y coordinate
                        individual[i] = max(0.01, min(height - 0.01, individual[i] + random.gauss(0, 0.03)))
                    else:  # radius
                        individual[i] = max(0.01, min(max_radius, individual[i] + random.gauss(0, 0.015)))
            return individual,
        
        def crossover(ind1, ind2):
            """Crossover two individuals"""
            size = len(ind1)
            cxpoint1 = random.randint(1, size)
            cxpoint2 = random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1
                
            ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
            return ind1, ind2
        
        # Initialize the evolutionary algorithm
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", crossover)
        toolbox.register("mutate", mutate)
        toolbox.register("select", tools.selTournament, tournsize=5)  # Larger tournament size
        
        # Run evolutionary algorithm with more generations and better parameters
        population = toolbox.population(n=100)  # Larger population
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run evolution for 100 generations with better parameters
        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.3, 
                                                 ngen=100, stats=stats, halloffame=hof, verbose=False)
        
        # Get the best individual from evolution
        best_individual = hof[0]
        
        # Enhanced local optimization with better constraint handling
        def constraint_func(params):
            # params: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
            positions = params.reshape(-1, 3)[:, :2]
            radii = params.reshape(-1, 3)[:, 2]
            
            constraints = []
            
            # Non-overlap constraints
            # Use vectorized approach for better performance
            try:
                dist_matrix = cdist(positions, positions)
                for i in range(n):
                    for j in range(i+1, n):
                        dist = dist_matrix[i, j]
                        min_dist = radii[i] + radii[j]
                        # Constraint is satisfied when dist >= min_dist (dist >= min_dist)
                        # So we want: dist - min_dist >= 0
                        constraints.append(dist - min_dist)
            except Exception:
                # Fallback to pairwise checking
                for i in range(n):
                    for j in range(i+1, n):
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        min_dist = radii[i] + radii[j]
                        constraints.append(dist - min_dist)
            
            # Boundary constraints (positive means violation)
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                # Circle must fit within rectangle
                constraints.append(x - r)  # left boundary
                constraints.append(width - x - r)  # right boundary
                constraints.append(y - r)  # bottom boundary
                constraints.append(height - y - r)  # top boundary
            
            return np.array(constraints)
        
        # Objective function to maximize (negative because minimize)
        def objective(params):
            radii = params.reshape(-1, 3)[:, 2]
            return -np.sum(radii)
        
        # Flatten the best individual
        initial_params = np.array(best_individual)
        
        # Use scipy's minimize with constraints for final refinement
        bounds = []
        for i in range(n):
            # x bounds
            bounds.extend([(0.001, width - 0.001), (0.001, height - 0.001), (0.001, max_radius)])
        
        try:
            # Try multiple optimization approaches
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = optimized_circles
        except Exception as e:
            warnings.warn(f"Local optimization failed: {str(e)}")
    
    # If no good result found, return the last best from evolution
    if best_result is None:
        # Run just one more focused optimization on the best from first run
        # This is a fallback case
        random.seed(42)
        np.random.seed(42)
        width, height = 1.0, 1.0  # Default to square
        max_radius = min(width, height) / 4.0
        
        # Simple run with better parameters
        toolbox = base.Toolbox()
        
        def create_individual():
            individual = []
            for i in range(n):
                x = random.uniform(0.01, width - 0.01)
                y = random.uniform(0.01, height - 0.01)
                r = random.uniform(0.01, max_radius)
                individual.extend([x, y, r])
            return creator.Individual(individual)
        
        def evaluate(individual):
            circles = np.array(individual).reshape(-1, 3)
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # Check boundary constraints
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                    return (0,)
            
            # Efficient overlap checking with early termination
            penalty = 0
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist_sq = dx*dx + dy*dy
                    min_dist = radii[i] + radii[j]
                    if dist_sq < min_dist * min_dist:
                        overlap = min_dist - math.sqrt(dist_sq)
                        penalty += overlap * overlap * 1000
                        
            total_radius = np.sum(radii)
            fitness = total_radius - penalty
            return (max(0, fitness),)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.02, indpb=0.15)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        population = toolbox.population(n=150)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        population, _ = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3, 
                                          ngen=150, stats=stats, halloffame=hof, verbose=False)
        
        best_individual = hof[0]
        return np.array(best_individual).reshape(-1, 3)
    
    return best_result


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
