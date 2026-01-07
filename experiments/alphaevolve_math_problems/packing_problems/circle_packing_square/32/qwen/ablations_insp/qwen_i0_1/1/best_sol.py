# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time
from itertools import combinations
from deap import base, creator, tools, algorithms
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Multi-start approach: try several different initializations
    best_circles = None
    best_sum = 0
    
    # Try different initialization strategies
    initial_configs = [
        initialize_grid_pack(n),
        initialize_hexagonal_pack(n),
        initialize_random_pack(n),
        initialize_fibonacci_pack(n)
    ]
    
    # Add evolutionary algorithm initialization
    try:
        evolutive_initial = initialize_evolutionary_pack(n)
        initial_configs.append(evolutive_initial)
    except:
        pass
    
    for i, initial_config in enumerate(initial_configs):
        try:
            # Optimize this initial configuration
            optimized = optimize_circles(initial_config)
            
            # Validate and get final sum
            validated = validate_and_adjust(optimized)
            current_sum = np.sum(validated[:, 2])
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = validated.copy()
                
        except Exception as e:
            continue
    
    # Final refinement with a more sophisticated approach
    if best_circles is not None:
        # Try one more optimization with better constraints
        final_result = refine_solution(best_circles)
        final_sum = np.sum(final_result[:, 2])
        
        if final_sum > best_sum:
            best_circles = final_result
    
    return best_circles if best_circles is not None else initialize_grid_pack(n)

def initialize_grid_pack(n):
    """Initialize circles in a grid pattern"""
    # Create a rectangular grid
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Ensure we have enough space for all circles
    while rows * cols < n:
        rows += 1
    
    # Calculate spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Initialize positions
    circles = np.zeros((n, 3))
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Ensure we're within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius based on spacing
            radius = min(spacing_x, spacing_y) * 0.4
            
            circles[idx] = [x, y, radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Set remaining circles with random positions but reasonable radii
    for i in range(idx, n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = np.random.uniform(0.01, 0.1)
        circles[i] = [x, y, radius]
    
    return circles

def initialize_hexagonal_pack(n):
    """Initialize circles using hexagonal packing pattern"""
    # Create a hexagonal grid pattern
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Adjust for better packing
    if rows * cols < n:
        rows += 1
    
    # Calculate spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Initialize positions
    circles = np.zeros((n, 3))
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = 0.5 * (i % 2)
            x = (j + x_offset) * spacing_x
            y = i * spacing_y
            
            # Ensure we're within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius based on spacing
            radius = min(spacing_x, spacing_y) * 0.4
            
            circles[idx] = [x, y, radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Set remaining circles with random positions but reasonable radii
    for i in range(idx, n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = np.random.uniform(0.01, 0.1)
        circles[i] = [x, y, radius]
    
    return circles

def initialize_random_pack(n):
    """Initialize circles with random positions and small radii"""
    circles = np.zeros((n, 3))
    for i in range(n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = np.random.uniform(0.01, 0.05)
        circles[i] = [x, y, radius]
    return circles

def initialize_fibonacci_pack(n):
    """Initialize circles using Fibonacci spiral for even distribution"""
    circles = np.zeros((n, 3))
    
    # Golden ratio
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in range(n):
        # Fibonacci spiral placement
        theta = i * 2 * np.pi / golden_ratio
        radius = np.sqrt(i / (n - 1)) if i < n - 1 else 1.0
        
        # Convert to Cartesian coordinates
        x = 0.5 + 0.45 * radius * np.cos(theta)
        y = 0.5 + 0.45 * radius * np.sin(theta)
        
        # Ensure within bounds
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        
        # Small initial radius
        radius = 0.02 + 0.03 * np.random.random()
        
        circles[i] = [x, y, radius]
    
    return circles

def initialize_evolutionary_pack(n):
    """Initialize circles using evolutionary algorithm approach"""
    # Set up DEAP for circle packing optimization
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define individual genes: [x, y, r] for each circle
    def create_individual():
        individual = []
        for _ in range(n):
            # Random position and radius
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.1)
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def eval_circle_pack(individual):
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)
        
        # Check constraints
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for violations
        penalty = 0
        
        # Containment penalties
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 1000
        
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < r1 + r2:
                    penalty += 10000 * (r1 + r2 - dist)
        
        return (total_radius - penalty,)
    
    toolbox.register("evaluate", eval_circle_pack)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolutionary algorithm for a short time
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    
    # Run for limited generations
    try:
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=20, 
                           halloffame=hof, verbose=False)
    except:
        pass
    
    # Return the best individual
    best_individual = hof[0] if len(hof) > 0 else create_individual()
    circles = np.array(best_individual).reshape(-1, 3)
    
    # Ensure all circles are valid
    for i in range(n):
        x, y, r = circles[i]
        # Keep radius positive and reasonable
        r = max(0.001, min(0.5, r))
        # Keep center within bounds
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    return circles

def optimize_circles(initial_circles):
    """Optimize circle positions using scipy with better constraints"""
    n = len(initial_circles)
    
    # Flatten parameters: [x0,y0,r0,x1,y1,r1,...]
    initial_params = initial_circles.flatten()
    
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we want to maximize
    
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        
        # Constraint: all circles must be within the unit square
        # x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
        constraints = []
        
        # Containment constraints
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([x - r, 1 - x - r, y - r, 1 - y - r])
        
        # Overlap constraints: distance >= r1 + r2
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(dist - r1 - r2)
        
        return np.array(constraints)
    
    # Define bounds: [x_min, x_max, y_min, y_max, r_min, r_max] for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])  # x, y, r bounds
    
    # Try different optimization approaches
    try:
        # First try with bounds and constraints
        cons = {'type': 'ineq', 'fun': constraint_func}
        result = minimize(objective, initial_params, method='SLSQP', 
                         bounds=bounds, constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        pass
    
    # Fallback: use L-BFGS-B which often works better for this type of problem
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 1000})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        pass
    
    # If all else fails, return initial configuration
    return initial_circles

def refine_solution(circles):
    """Apply advanced refinement to improve solution quality"""
    # Apply iterative improvement with better overlap resolution
    n = len(circles)
    
    # Create KDTree for efficient neighbor search
    tree = cKDTree(circles[:, :2])
    
    # Perform multiple rounds of refinement
    for iteration in range(100):  # More iterations for better refinement
        improved = False
        
        # For each circle, check neighbors and adjust
        for i in range(n):
            # Find nearby circles using KDTree
            neighbors = tree.query_ball_point(circles[i, :2], 2 * np.max(circles[:, 2]))
            
            # Remove self from neighbors
            neighbors = [idx for idx in neighbors if idx != i]
            
            # Resolve overlaps with neighbors
            for j in neighbors:
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                if dist < r1 + r2:
                    # Move circles apart
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        dx, dy = 1, 0
                    norm = np.sqrt(dx*dx + dy*dy)
                    dx /= norm
                    dy /= norm
                    
                    # Move them apart (smaller movement to preserve structure)
                    move_dist = (r1 + r2 - dist) * 0.5
                    circles[i, 0] -= dx * move_dist
                    circles[i, 1] -= dy * move_dist
                    circles[j, 0] += dx * move_dist
                    circles[j, 1] += dy * move_dist
                    
                    improved = True
        
        if not improved:
            break
    
    return circles

def validate_and_adjust(circles):
    """Ensure final solution is valid"""
    # Make sure all circles are within bounds and don't overlap
    n = len(circles)
    
    # Simple validation and adjustment
    for i in range(n):
        x, y, r = circles[i]
        # Keep radius positive and reasonable
        r = max(0.001, min(0.5, r))
        # Keep center within bounds
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    # Refine to avoid overlaps more carefully
    for _ in range(100):  # More iterations for better overlap resolution
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                if dist < r1 + r2:
                    # Move circles apart
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 and dy == 0:
                        dx, dy = 1, 0
                    norm = np.sqrt(dx*dx + dy*dy)
                    dx /= norm
                    dy /= norm
                    
                    # Move them apart with more aggressive adjustment
                    move_dist = (r1 + r2 - dist) * 0.7
                    circles[i, 0] -= dx * move_dist
                    circles[i, 1] -= dy * move_dist
                    circles[j, 0] += dx * move_dist
                    circles[j, 1] += dy * move_dist
                    
                    improved = True
        
        if not improved:
            break
    
    return circles


# EVOLVE-BLOCK-END
