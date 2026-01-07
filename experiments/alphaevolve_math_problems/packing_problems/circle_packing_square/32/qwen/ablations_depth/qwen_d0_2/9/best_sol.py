# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from scipy.spatial import Voronoi
import random
from itertools import combinations
from deap import base, creator, tools, algorithms
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: multiple initialization strategies with advanced optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Try multiple initialization strategies and select the best
    best_result = None
    best_sum = 0
    
    # Strategy 1: Genetic Algorithm initialization
    ga_result = initialize_ga_packing(n)
    if ga_result is not None:
        optimized = optimize_circles(ga_result)
        current_sum = np.sum(optimized[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = optimized
    
    # Strategy 2: Voronoi-based initialization
    voronoi_result = initialize_voronoi_packing(n)
    if voronoi_result is not None:
        optimized = optimize_circles(voronoi_result)
        current_sum = np.sum(optimized[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = optimized
    
    # Strategy 3: Grid-based with random perturbations
    grid_result = initialize_grid_packing(n)
    if grid_result is not None:
        optimized = optimize_circles(grid_result)
        current_sum = np.sum(optimized[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = optimized
    
    # Strategy 4: Random initialization with refinement
    random_result = initialize_random_packing(n)
    if random_result is not None:
        optimized = optimize_circles(random_result)
        current_sum = np.sum(optimized[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = optimized
    
    # If none worked, fall back to a good heuristic
    if best_result is None:
        # Start with a simple but effective configuration
        circles = np.zeros((n, 3))
        # Place in a roughly hexagonal pattern with varying radii
        for i in range(n):
            # Distribute points more evenly
            row = i // 6
            col = i % 6
            x = 0.1 + col * 0.15
            y = 0.1 + row * 0.15
            r = 0.05
            # Adjust for boundaries
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[i] = [x, y, r]
        best_result = circles
    
    return best_result

def initialize_ga_packing(n: int) -> np.ndarray:
    """Initialize circles using genetic algorithm approach"""
    # Define the optimization problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define gene ranges: [x1, y1, r1, x2, y2, r2, ...]
    def create_individual():
        individual = []
        for i in range(n):
            # x coordinate: [0.01, 0.99]
            x = np.random.uniform(0.01, 0.99)
            # y coordinate: [0.01, 0.99] 
            y = np.random.uniform(0.01, 0.99)
            # radius: [0.01, 0.1]
            r = np.random.uniform(0.01, 0.1)
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    def evaluate(individual):
        # Convert individual to circles
        circles = []
        for i in range(n):
            x = individual[3*i]
            y = individual[3*i+1]
            r = individual[3*i+2]
            circles.append([x, y, r])
        
        # Check constraints
        total_radius = 0
        valid = True
        
        # Check boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                valid = False
                break
            total_radius += r
        
        if not valid:
            return (-1000,)  # Penalty for invalid solution
        
        # Check overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance_sq = (x1-x2)**2 + (y1-y2)**2
                min_distance_sq = (r1 + r2)**2
                
                if distance_sq < min_distance_sq:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return (-1000,)
            
        return (total_radius,)
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.02, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run GA for a short time to get a good initial solution
    try:
        pop = toolbox.population(n=50)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run for fewer generations to save time
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                          ngen=20, stats=stats, halloffame=hof, verbose=False)
        
        if len(hof) > 0:
            best_individual = hof[0]
            circles = []
            for i in range(n):
                x = best_individual[3*i]
                y = best_individual[3*i+1]
                r = best_individual[3*i+2]
                circles.append([x, y, r])
            return np.array(circles)
    except:
        pass
    
    return None

def initialize_voronoi_packing(n: int) -> np.ndarray:
    """Initialize circles based on Voronoi diagram of randomly distributed points"""
    # Generate random points
    np.random.seed(42)
    points = np.random.rand(100, 2)  # More points for better Voronoi
    
    try:
        # Compute Voronoi diagram
        vor = Voronoi(points)
        
        # Select n points that give good circle placement
        # Get centroids of Voronoi cells
        centroids = []
        for region in vor.regions:
            if len(region) > 0 and -1 not in region:
                vertices = [vor.vertices[i] for i in region]
                centroid = np.mean(vertices, axis=0)
                centroids.append(centroid)
        
        # Take first n centroids that are inside unit square
        selected_centroids = []
        for centroid in centroids:
            if 0 <= centroid[0] <= 1 and 0 <= centroid[1] <= 1:
                selected_centroids.append(centroid)
                if len(selected_centroids) >= n:
                    break
        
        if len(selected_centroids) < n:
            return None
            
        # Create circles with appropriate radii
        circles = []
        for i, (x, y) in enumerate(selected_centroids[:n]):
            # Estimate radius based on Voronoi cell area
            # For now, use a reasonable starting value
            r = min(0.1, 0.5 * min(x, 1-x, y, 1-y))
            circles.append([x, y, r])
        
        return np.array(circles)
    except:
        return None

def initialize_grid_packing(n: int) -> np.ndarray:
    """Initialize circles in a grid pattern with some randomness"""
    # Create a grid pattern with some variation
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Ensure we have enough space
    if rows * cols < n:
        rows += 1
    
    # Create grid with some randomization
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            # Grid position with slight randomization
            x = 0.1 + j * 0.8 / (cols - 1) if cols > 1 else 0.5
            y = 0.1 + i * 0.8 / (rows - 1) if rows > 1 else 0.5
            
            # Add some randomness to avoid perfect grid
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            
            # Ensure within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Radius based on proximity to boundaries
            r = min(0.05, 0.5 * min(x, 1-x, y, 1-y))
            circles.append([x, y, r])
    
    return np.array(circles[:n]) if len(circles) >= n else None

def initialize_random_packing(n: int) -> np.ndarray:
    """Initialize circles with random positions and small radii"""
    np.random.seed(123)
    circles = []
    
    # Try to place circles avoiding overlaps initially
    attempts = 0
    max_attempts = 1000
    
    while len(circles) < n and attempts < max_attempts:
        # Random position and small radius
        x = np.random.uniform(0.01, 0.99)
        y = np.random.uniform(0.01, 0.99)
        r = np.random.uniform(0.01, 0.05)
        
        # Check if this circle would overlap with any existing ones
        valid = True
        for cx, cy, cr in circles:
            distance = math.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < r + cr:
                valid = False
                break
        
        if valid:
            # Ensure it fits in the square
            if r <= x <= 1-r and r <= y <= 1-r:
                circles.append([x, y, r])
        
        attempts += 1
    
    if len(circles) < n:
        # Fill remaining with smaller circles
        while len(circles) < n:
            x = np.random.uniform(0.01, 0.99)
            y = np.random.uniform(0.01, 0.99)
            r = 0.01
            circles.append([x, y, r])
    
    return np.array(circles) if len(circles) >= n else None

def optimize_circles(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using advanced constrained optimization"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    def objective(params):
        # Convert params back to circles
        circles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            circles.append([x, y, r])
        
        # Maximize sum of radii (minimize negative sum)
        return -sum(circle[2] for circle in circles)
    
    # Create constraint functions that are more numerically stable
    def boundary_constraints(params):
        """Ensure all circles fit in the unit square"""
        constraints = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            
            # Circle must fit in square (boundary constraints)
            constraints.append(x - r)          # x >= r
            constraints.append(1 - x - r)      # 1 - x >= r
            constraints.append(y - r)          # y >= r
            constraints.append(1 - y - r)      # 1 - y >= r
            
        return np.array(constraints)
    
    def overlap_constraints(params):
        """Ensure no overlaps between circles"""
        constraints = []
        circles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            circles.append([x, y, r])
        
        # Check all pairs of circles for overlap
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance_squared = (x1-x2)**2 + (y1-y2)**2
                # Use squared distance to avoid sqrt computation
                min_distance_squared = (r1 + r2)**2
                
                # We want distance >= r1 + r2, so distance^2 >= (r1+r2)^2
                # Therefore: (distance^2 - (r1+r2)^2) >= 0
                constraints.append(distance_squared - min_distance_squared)
                
        return np.array(constraints)
    
    # Set up bounds for optimization (x, y, r for each circle)
    bounds = [(0.001, 0.999) for _ in range(3*n)]
    
    # Create constraint objects for scipy
    cons = []
    
    # Boundary constraints
    cons.append({'type': 'ineq', 'fun': boundary_constraints})
    
    # Overlap constraints
    cons.append({'type': 'ineq', 'fun': overlap_constraints})
    
    # Try multiple optimization approaches
    try:
        # First try with SLSQP which is generally good for this type of problem
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-8}
        )
        
        if result.success:
            final_circles = []
            for i in range(n):
                x = result.x[3*i]
                y = result.x[3*i+1]
                r = result.x[3*i+2]
                final_circles.append([x, y, r])
            return np.array(final_circles)
            
    except Exception as e:
        pass
    
    # If optimization fails, return initial configuration
    return initial_circles


# EVOLVE-BLOCK-END
