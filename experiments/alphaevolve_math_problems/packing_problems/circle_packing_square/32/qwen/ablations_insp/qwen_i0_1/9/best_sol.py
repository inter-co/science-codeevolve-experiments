# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time
from itertools import combinations
import random
from deap import base, creator, tools, algorithms
import copy

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining physics simulation and evolutionary algorithms.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Strategy 1: Physics-based simulation
    circles1 = simulate_physics_pack(n, seed=42)
    sum1 = np.sum(circles1[:, 2])
    
    # Strategy 2: Evolutionary algorithm
    circles2 = evolutionary_pack(n, seed=42)
    sum2 = np.sum(circles2[:, 2])
    
    # Strategy 3: Multi-start with different initialization patterns
    best_sum = 0
    best_circles = None
    
    # Try several initialization strategies
    strategies = [
        initialize_hexagonal_pack(n),
        initialize_square_pack(n),
        initialize_random_pack(n),
        initialize_dense_pack(n)
    ]
    
    for i, init_circles in enumerate(strategies):
        # Apply local optimization
        optimized = optimize_circles_local(init_circles)
        sum_val = np.sum(optimized[:, 2])
        
        if sum_val > best_sum:
            best_sum = sum_val
            best_circles = optimized.copy()
    
    # Strategy 4: Hybrid approach - combine best results
    # Take the best of all approaches
    strategies_results = [circles1, circles2, best_circles]
    strategy_sums = [sum1, sum2, best_sum]
    
    best_idx = np.argmax(strategy_sums)
    final_circles = strategies_results[best_idx]
    
    # Final validation and refinement
    final_circles = validate_and_adjust(final_circles)
    
    return final_circles

def simulate_physics_pack(n, seed=None):
    """Use physics-based simulation to pack circles"""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize circles with random positions and small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = np.random.uniform(0.005, 0.05)
        circles[i] = [x, y, radius]
    
    # Physics simulation parameters
    dt = 0.01
    max_steps = 1000
    temperature = 1.0
    
    for step in range(max_steps):
        # Compute forces between all pairs
        forces = np.zeros((n, 2))
        
        for i in range(n):
            x1, y1, r1 = circles[i]
            
            # Repulsion force from boundaries
            fx_boundary = 0
            fy_boundary = 0
            
            # Left boundary
            if x1 - r1 < 0:
                fx_boundary += (0 - (x1 - r1)) * 100
            # Right boundary  
            if x1 + r1 > 1:
                fx_boundary += (1 - (x1 + r1)) * 100
                
            # Bottom boundary
            if y1 - r1 < 0:
                fy_boundary += (0 - (y1 - r1)) * 100
            # Top boundary
            if y1 + r1 > 1:
                fy_boundary += (1 - (y1 + r1)) * 100
                
            forces[i] += np.array([fx_boundary, fy_boundary])
            
            # Repulsion forces from other circles
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > 0 and dist < r1 + r2:
                    # Repulsive force when overlapping
                    force_magnitude = 1000 * (r1 + r2 - dist) / dist
                    fx = force_magnitude * dx / dist
                    fy = force_magnitude * dy / dist
                    
                    forces[i] += np.array([-fx, -fy])
                    forces[j] += np.array([fx, fy])
        
        # Update positions with damping
        damping = 0.9
        for i in range(n):
            x, y, r = circles[i]
            fx, fy = forces[i]
            
            # Apply forces with temperature (random component)
            noise = np.random.normal(0, temperature * 0.1)
            fx += noise
            fy += noise
            
            # Update position
            x += fx * dt
            y += fy * dt
            
            # Keep within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[i] = [x, y, r]
        
        # Decrease temperature
        temperature *= 0.999
        
        # Occasionally adjust radii to maximize sum
        if step % 50 == 0:
            # Simple greedy adjustment: slightly increase radii if possible
            for i in range(n):
                x, y, r = circles[i]
                # Try to increase radius while maintaining constraints
                new_r = min(0.5, r + 0.001)
                # Check if this would cause overlaps
                valid = True
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                        if dist < new_r + r2:
                            valid = False
                            break
                if valid:
                    circles[i, 2] = new_r
    
    return circles

def evolutionary_pack(n, seed=None):
    """Use evolutionary algorithm to find optimal circle packing"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Define the fitness function
    def eval_fitness(individual):
        # Convert individual to circles array
        circles = np.array(individual).reshape(-1, 3)
        
        # Check constraints and calculate fitness
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for constraint violations
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
                    penalty += 1000 * (r1 + r2 - dist)
        
        # Return negative because we want to maximize
        return (total_radius - penalty,)
    
    # Create the evolutionary algorithm components
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Attribute generator
    def create_individual():
        individual = []
        for i in range(n):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.005, 0.1)
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register the evaluation function
    toolbox.register("evaluate", eval_fitness)
    
    # Register the crossover and mutation operators
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create population and run evolution
    population = toolbox.population(n=50)
    
    # Run evolution for a few generations
    for gen in range(20):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    
    # Get the best individual
    best_individual = tools.selBest(population, k=1)[0]
    circles = np.array(best_individual).reshape(-1, 3)
    
    # Validate and adjust
    circles = validate_and_adjust(circles)
    
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

def initialize_square_pack(n):
    """Initialize circles using square lattice pattern"""
    side = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / side
    
    circles = np.zeros((n, 3))
    
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            x = (j + 0.5) * spacing
            y = (i + 0.5) * spacing
            
            # Ensure we're within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius
            radius = spacing * 0.4
            
            circles[idx] = [x, y, radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Set remaining circles with random positions
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
        radius = np.random.uniform(0.005, 0.05)
        circles[i] = [x, y, radius]
    
    return circles

def initialize_dense_pack(n):
    """Initialize circles with denser packing approach"""
    # Start with a good initial configuration
    circles = np.zeros((n, 3))
    
    # Place some circles near corners and edges to encourage better distribution
    # Corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i in range(min(len(corners), n)):
        x, y = corners[i]
        radius = 0.05
        circles[i] = [x, y, radius]
    
    # Fill remaining positions randomly
    for i in range(len(corners), n):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = np.random.uniform(0.01, 0.08)
        circles[i] = [x, y, radius]
    
    return circles

def optimize_circles_local(initial_circles):
    """Apply local optimization to improve initial solution"""
    n = len(initial_circles)
    
    # Flatten parameters: [x0,y0,r0,x1,y1,r1,...]
    initial_params = initial_circles.flatten()
    
    def objective(params):
        circles = params.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we want to maximize
    
    def constraint_func(params):
        circles = params.reshape(-1, 3)
        constraints = []
        
        # Containment constraints: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
        for i in range(n):
            x, y, r = circles[i]
            # These should be >= 0 for feasibility
            constraints.extend([x - r, 1 - x - r, y - r, 1 - y - r])
        
        # Overlap constraints: distance >= r1 + r2 (so distance - r1 - r2 >= 0)
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
    
    # Try multiple optimization approaches
    try:
        # Try with SLSQP first (good for constrained problems)
        cons = {'type': 'ineq', 'fun': constraint_func}
        result = minimize(objective, initial_params, method='SLSQP', 
                         bounds=bounds, constraints=cons, 
                         options={'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-4})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        pass
    
    # If that fails, try L-BFGS-B with bounds only
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B',
                         bounds=bounds,
                         options={'maxiter': 300, 'ftol': 1e-6})
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        pass
    
    # If all else fails, return original
    return initial_circles

def validate_and_adjust(circles):
    """Ensure final solution is valid and improve quality"""
    # Make sure all circles are within bounds and don't overlap
    n = len(circles)
    
    # First, clean up positions and radii
    for i in range(n):
        x, y, r = circles[i]
        # Keep radius positive and reasonable
        r = max(0.001, min(0.5, r))
        # Keep center within bounds
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    # More aggressive refinement to avoid overlaps
    improved = True
    iterations = 0
    max_iterations = 100
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Use spatial data structure for efficient neighbor search
        positions = circles[:, :2]
        tree = cKDTree(positions)
        
        # Find close pairs and resolve overlaps
        pairs = tree.query_pairs(0.01, output_type='ndarray')  # Very small distance threshold
        
        for i, j in pairs:
            if i >= j:  # Only process each pair once
                continue
                
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
                
                # Move them apart proportionally to their radii
                move_dist = (r1 + r2 - dist) * 0.5
                circles[i, 0] -= dx * move_dist * 0.5
                circles[i, 1] -= dy * move_dist * 0.5
                circles[j, 0] += dx * move_dist * 0.5
                circles[j, 1] += dy * move_dist * 0.5
                
                improved = True
        
        # Boundary correction
        for i in range(n):
            x, y, r = circles[i]
            # Keep within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
