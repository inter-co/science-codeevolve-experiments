# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from deap import base, creator, tools, algorithms
import random
import time
from itertools import combinations
from scipy.spatial import cKDTree

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses evolutionary algorithm with geometric constraints.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    random.seed(42)
    np.random.seed(42)
    
    # Better initial configuration using a more dense packing approach
    def generate_initial_config():
        # Use a more sophisticated initial configuration based on known good packings
        circles = np.zeros((n, 3))
        
        # Create a hexagonal-like pattern with some randomness
        rows = 6
        cols = 6
        
        # Generate points in a hexagonal pattern
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= n:
                    break
                # Offset odd rows
                offset = 0.5 if row % 2 == 1 else 0.0
                x = (col + 0.5 + offset) * spacing_x
                y = (row + 0.5) * spacing_y
                
                # Add small random jitter
                jitter = 0.02
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)
                
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Initial radius - start with a reasonable value
                circles[idx] = [x, y, 0.04]
                idx += 1
                
                if idx >= n:
                    break
        
        # Fill remaining positions randomly but with good spacing
        for i in range(idx, n):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles[i] = [x, y, 0.04]
            
        return circles
    
    # Improved fitness function with better penalty system
    def evaluate_individual(individual):
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)
        
        # Check containment constraints
        total_radius = 0
        penalty = 0
        
        for i in range(n):
            x, y, r = circles[i]
            # Check containment
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 10000  # Large penalty for containment violation
            total_radius += r
            
        # Check overlap constraints more efficiently using spatial indexing
        # First, build KDTree for faster neighbor searches
        coords = circles[:, :2]
        tree = cKDTree(coords)
        
        # Find neighbors within a reasonable distance
        for i in range(n):
            x1, y1, r1 = circles[i]
            
            # Query nearby points (this is much faster than checking all pairs)
            nearby_indices = tree.query_ball_point([x1, y1], 2*(r1 + 0.01))
            
            for j in nearby_indices:
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist_sq = (x1-x2)**2 + (y1-y2)**2
                    min_dist_sq = (r1 + r2)**2
                    
                    if dist_sq < min_dist_sq:
                        # Calculate overlap amount
                        overlap = min_dist_sq - dist_sq
                        penalty += 1000 * overlap  # Penalty based on overlap
                        
        # Return negative sum of radii (minimize negative = maximize sum)
        # Add penalty for constraint violations
        return -(total_radius - penalty), 
    
    # Create evolutionary algorithm components
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Attribute generator - each circle has (x, y, r)
    def create_circle_gene():
        # x, y in [0.05, 0.95] to give some margin, r in [0.01, 0.15] for better packing
        return [
            random.uniform(0.05, 0.95),
            random.uniform(0.05, 0.95),
            random.uniform(0.01, 0.15)
        ]
    
    # Structure initializers
    toolbox.register("attr_circle", create_circle_gene)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_circle, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register genetic operators with better parameters
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolution with more generations
    population = toolbox.population(n=150)
    hof = tools.HallOfFame(1)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run the evolutionary algorithm with more generations
    try:
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=0.7, mutpb=0.2, 
            ngen=200, stats=stats, halloffame=hof, verbose=False
        )
        
        best_individual = hof[0]
        circles = np.array(best_individual).reshape(-1, 3)
        
    except Exception as e:
        # Fallback to improved initial configuration
        circles = generate_initial_config()
    
    # Apply enhanced local optimization to final solution
    try:
        # Convert to flat parameter vector for scipy optimization
        def objective(params):
            # Reconstruct circles from params
            circles = params.reshape(-1, 3)
            total_radius = np.sum(circles[:, 2])
            return -total_radius  # negative because we want to maximize
        
        def constraint_func(params):
            # Check all constraints
            circles = params.reshape(-1, 3)
            constraints = []
            
            # Containment constraints
            for i in range(n):
                x, y, r = circles[i]
                constraints.extend([
                    x - r,           # x >= r
                    1 - x - r,       # 1 - x >= r
                    y - r,           # y >= r
                    1 - y - r        # 1 - y >= r
                ])
            
            # Non-overlap constraints - use more efficient pairwise checks with early termination
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                    constraints.append(dist - r1 - r2)  # dist >= r1 + r2
                    
            return np.array(constraints)
        
        # Initial parameters
        initial_params = circles.flatten()
        
        # Bounds: x, y in [r, 1-r], r in [0.001, 0.3]
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.3)])
        
        # Try local optimization with bounds and better method
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraint_func}],
            options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4},
            tol=1e-6
        )
        
        if result.success:
            circles = result.x.reshape(-1, 3)
            
    except Exception:
        pass
    
    # Final validation and cleanup
    for i in range(n):
        # Ensure radii are valid
        circles[i][2] = max(0.001, min(0.3, circles[i][2]))
        
        # Ensure positions respect containment
        x, y, r = circles[i]
        circles[i] = [max(r, min(1-r, x)), max(r, min(1-r, y)), r]
    
    return circles


# EVOLVE-BLOCK-END
