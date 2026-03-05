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
from sklearn.cluster import KMeans
from numba import jit
import heapq

@jit(nopython=True)
def fast_distance_squared(x1, y1, x2, y2):
    """Fast squared distance calculation"""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy

@jit(nopython=True)
def fast_check_overlap_fast(pos1, pos2, r1, r2):
    """Fast overlap check using squared distances"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dist_sq = dx * dx + dy * dy
    min_dist = r1 + r2
    return dist_sq < min_dist * min_dist

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining evolutionary algorithm with local optimization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Try different aspect ratios - test several candidates to find optimal
    candidates = [(1.0, 1.0), (1.2, 0.8), (0.8, 1.2), (1.5, 0.5), (0.5, 1.5), (1.3, 0.7), (0.7, 1.3)]
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
            
            # Check overlap constraints efficiently
            penalty = 0
            
            # Use spatial hashing for faster overlap checking
            # Create a simple spatial grid for acceleration
            grid_size = min(width, height) / 5.0
            grid = {}
            
            # Place circles into grid cells
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                cell_x = int(x / grid_size)
                cell_y = int(y / grid_size)
                if (cell_x, cell_y) not in grid:
                    grid[(cell_x, cell_y)] = []
                grid[(cell_x, cell_y)].append((i, x, y, r))
                
                # Check neighboring cells as well
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_cell = (cell_x + dx, cell_y + dy)
                        if neighbor_cell in grid:
                            for j, nx, ny, nr in grid[neighbor_cell]:
                                if i < j:  # Only check each pair once
                                    # Use fast distance check
                                    if fast_check_overlap_fast((x, y), (nx, ny), r, nr):
                                        overlap = (r + nr) - math.sqrt(fast_distance_squared(x, y, nx, ny))
                                        penalty += overlap * overlap * 1000  # Strong penalty
                            
            # If spatial hashing didn't work, fall back to full check
            if penalty == 0:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = positions[i][0] - positions[j][0]
                        dy = positions[i][1] - positions[j][1]
                        dist_sq = dx*dx + dy*dy
                        min_dist = radii[i] + radii[j]
                        if dist_sq < min_dist * min_dist:
                            overlap = min_dist - math.sqrt(dist_sq)
                            penalty += overlap * overlap * 1000
            
            # Fitness is sum of radii minus penalty
            total_radius = np.sum(radii)
            fitness = total_radius - penalty
            
            # Ensure fitness is not negative
            return (max(0, fitness),)
        
        def mutate(individual):
            """Mutate an individual"""
            for i in range(len(individual)):
                if random.random() < 0.1:  # Reduced mutation rate for stability
                    if i % 3 == 0:  # x coordinate
                        individual[i] = max(0.01, min(width - 0.01, individual[i] + random.gauss(0, 0.02)))
                    elif i % 3 == 1:  # y coordinate
                        individual[i] = max(0.01, min(height - 0.01, individual[i] + random.gauss(0, 0.02)))
                    else:  # radius
                        individual[i] = max(0.01, min(max_radius, individual[i] + random.gauss(0, 0.01)))
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
        toolbox.register("select", tools.selTournament, tournsize=3)  # Smaller tournament for more diversity
        
        # Run evolutionary algorithm with optimized parameters
        population = toolbox.population(n=100)  # Reduced population size for faster runs
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run evolution for fewer generations with better parameters
        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.2, 
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
            # Try multiple optimization approaches with better settings
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
    
    # If no good result found, run a focused optimization with better parameters
    if best_result is None:
        # Run a focused optimization with better parameters
        random.seed(42)
        np.random.seed(42)
        width, height = 1.0, 1.0  # Default to square
        max_radius = min(width, height) / 4.0
        
        # Better initialization with more structured placement
        def create_better_initial():
            individual = []
            # Use hexagonal packing pattern for better initial distribution
            rows = 5
            cols = 5
            spacing_x = width / (cols + 1)
            spacing_y = height / (rows + 1)
            
            for i in range(n):
                row = i // cols
                col = i % cols
                x = spacing_x * (col + 1)
                y = spacing_y * (row + 1)
                # Add slight randomness based on position
                if row % 2 == 1:  # Offset odd rows
                    x += spacing_x / 2
                x += random.uniform(-spacing_x/6, spacing_x/6)
                y += random.uniform(-spacing_y/6, spacing_y/6)
                # Keep within bounds
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                r = random.uniform(0.01, max_radius)
                individual.extend([x, y, r])
            return creator.Individual(individual)
        
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
            
            # Efficient overlap checking with spatial acceleration
            penalty = 0
            
            # Spatial hash for acceleration
            grid_size = min(width, height) / 4.0
            grid = {}
            
            # Place circles into grid cells
            for i in range(n):
                x, y, r = positions[i][0], positions[i][1], radii[i]
                cell_x = int(x / grid_size)
                cell_y = int(y / grid_size)
                if (cell_x, cell_y) not in grid:
                    grid[(cell_x, cell_y)] = []
                grid[(cell_x, cell_y)].append((i, x, y, r))
                
                # Check neighboring cells as well
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor_cell = (cell_x + dx, cell_y + dy)
                        if neighbor_cell in grid:
                            for j, nx, ny, nr in grid[neighbor_cell]:
                                if i < j:  # Only check each pair once
                                    if fast_check_overlap_fast((x, y), (nx, ny), r, nr):
                                        overlap = (r + nr) - math.sqrt(fast_distance_squared(x, y, nx, ny))
                                        penalty += overlap * overlap * 1000  # Strong penalty
                            
            # If spatial hashing didn't catch everything, do full check
            if penalty == 0:
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
        toolbox.register("mate", tools.cxUniform, indpb=0.15)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.015, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        population = toolbox.population(n=150)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        population, _ = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.25, 
                                          ngen=150, stats=stats, halloffame=hof, verbose=False)
        
        best_individual = hof[0]
        return np.array(best_individual).reshape(-1, 3)
    
    return best_result


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
