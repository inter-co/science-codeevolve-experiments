# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple, List

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining spatial partitioning and gradient-based optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Initialize with a good heuristic layout
    circles = initialize_circles_heuristic(n)
    
    # Refine using optimization
    circles = optimize_circles(circles)
    
    return circles

def initialize_circles_heuristic(n: int) -> np.ndarray:
    """Initialize circle positions using a hexagonal packing heuristic"""
    # Create a hexagonal grid pattern
    circles = np.zeros((n, 3))
    
    # For 26 circles, we can arrange them in approximately 5 rows and 5 columns
    # But we'll use a more sophisticated approach
    
    # Calculate approximate spacing based on total area
    total_area = n * (math.pi / 4)  # Assuming max possible area
    side_length = 1.0
    
    # Place circles in a grid pattern with some randomness
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    
    # Adjust grid size to fit in unit square
    grid_size_x = 1.0 / cols
    grid_size_y = 1.0 / rows
    
    # Add padding to prevent boundary issues
    padding = 0.02
    
    for i in range(n):
        row = i // cols
        col = i % cols
        
        # Position in grid with slight jitter
        x = (col + 0.5) * grid_size_x
        y = (row + 0.5) * grid_size_y
        
        # Apply padding
        x = max(padding, min(1-padding, x))
        y = max(padding, min(1-padding, y))
        
        # Initial radius - start with small values and adjust
        r = min(grid_size_x, grid_size_y) * 0.3
        
        circles[i] = [x, y, r]
    
    return circles

def get_constraints(circles: np.ndarray) -> Tuple[List, List]:
    """Generate constraint functions for optimization"""
    n = len(circles)
    
    # Boundary constraints: each circle must fit inside unit square
    def boundary_constraint(i):
        def constraint(xyr):
            x, y, r = xyr
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        return constraint
    
    # Circle-to-circle constraints: circles must not overlap
    def overlap_constraint(i, j):
        def constraint(xyr_i, xyr_j):
            x1, y1, r1 = xyr_i
            x2, y2, r2 = xyr_j
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            return dist_sq - (r1 + r2)**2
        return constraint
    
    # Generate all constraints
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda xyr, i=i: boundary_constraint(i)(xyr)
        })
    
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda xyr, i=i, j=j: overlap_constraint(i, j)(xyr[:3], xyr[3:])
            })
    
    return constraints

def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions using scipy's minimize"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_flat = initial_circles.flatten()
    
    # Define objective function (negative because we want to maximize sum of radii)
    def objective(flat_params):
        circles = flat_params.reshape(n, 3)
        return -np.sum(circles[:, 2])  # Negative because we minimize
    
    # Define bounds for each parameter (x, y, r)
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))  # Small padding to prevent boundary issues
        # y bounds
        bounds.append((0.001, 0.999))
        # r bounds
        bounds.append((0.001, 0.5))  # Reasonable upper bound
    
    # Constraints for optimization
    def boundary_constraint(i):
        def constraint(params):
            x, y, r = params[3*i:3*i+3]
            return min(x - r, 1 - x - r, y - r, 1 - y - r)
        return constraint
    
    def overlap_constraint(i, j):
        def constraint(params):
            x1, y1, r1 = params[3*i:3*i+3]
            x2, y2, r2 = params[3*j:3*j+3]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            return dist_sq - (r1 + r2)**2
        return constraint
    
    # Create constraints list
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': boundary_constraint(i)
        })
    
    # Overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': overlap_constraint(i, j)
            })
    
    # Run optimization
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(n, 3)
            return optimized_circles
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        pass
    
    return initial_circles

# Alternative implementation using a more robust evolutionary approach
def circle_packing26_evolutionary() -> np.ndarray:
    """Alternative approach using evolutionary algorithms"""
    from deap import base, creator, tools, algorithms
    
    # Set up evolutionary algorithm
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define individual (x, y, r) for each of 26 circles
    def create_individual():
        individual = []
        for _ in range(26):
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            r = np.random.uniform(0.01, 0.2)
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_fitness(individual):
        circles = np.array(individual).reshape(26, 3)
        
        # Check constraints
        total_radius = 0
        valid = True
        
        # Boundary check
        for i in range(26):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                valid = False
                break
            total_radius += r
            
        if not valid:
            return 0.0,
            
        # Overlap check
        for i in range(26):
            for j in range(i+1, 26):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                if dist_sq < (r1 + r2)**2:
                    valid = False
                    break
            if not valid:
                return 0.0,
                
        return total_radius,
    
    toolbox.register("evaluate", eval_fitness)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolution
    population = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    
    try:
        algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.3, ngen=100, 
                           stats=None, halloffame=hof, verbose=False)
    except:
        pass
    
    best_individual = hof[0]
    circles = np.array(best_individual).reshape(26, 3)
    return circles

# Final implementation using the most promising approach
def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining spatial initialization and gradient-based optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Try multiple initialization strategies and pick the best
    best_circles = None
    best_sum = 0
    
    # Strategy 1: Hexagonal grid initialization
    circles1 = initialize_circles_heuristic(n)
    circles1_optimized = optimize_circles(circles1)
    sum1 = np.sum(circles1_optimized[:, 2])
    
    if sum1 > best_sum:
        best_sum = sum1
        best_circles = circles1_optimized
    
    # Strategy 2: Random initialization with constraints
    np.random.seed(42)  # For reproducibility
    circles2 = np.zeros((n, 3))
    for i in range(n):
        circles2[i] = [
            np.random.uniform(0.05, 0.95),
            np.random.uniform(0.05, 0.95),
            np.random.uniform(0.02, 0.15)
        ]
    circles2_optimized = optimize_circles(circles2)
    sum2 = np.sum(circles2_optimized[:, 2])
    
    if sum2 > best_sum:
        best_sum = sum2
        best_circles = circles2_optimized
    
    # Return the best result found
    if best_circles is None:
        # Fallback to simple initialization
        circles_fallback = np.zeros((n, 3))
        for i in range(n):
            circles_fallback[i] = [0.5, 0.5, 0.1]
        return circles_fallback
    
    return best_circles


# EVOLVE-BLOCK-END
