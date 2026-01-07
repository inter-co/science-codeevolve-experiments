# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import random
from deap import base, creator, tools, algorithms
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    
    Uses an evolutionary algorithm approach combined with local optimization for better results.
    """
    n = 32
    random.seed(42)
    np.random.seed(42)
    
    # Create initial configuration using multiple strategies
    def create_initial_placement():
        best_circles = None
        best_sum = 0
        
        # Strategy 1: Hexagonal close packing approximation
        circles = []
        rows, cols = 6, 6
        spacing_x = 0.15
        spacing_y = 0.15 * math.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, 0.05])
                    
        # Fill remaining slots if needed
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        initial_sum = sum(circle[2] for circle in circles)
        if initial_sum > best_sum:
            best_sum = initial_sum
            best_circles = circles.copy()
        
        # Strategy 2: Better hexagonal packing
        circles = []
        rows, cols = 5, 7
        spacing_x = 0.18
        spacing_y = 0.18 * math.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.05 + j * spacing_x
                y = 0.05 + i * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                if x <= 0.95 and y <= 0.95:
                    circles.append([x, y, 0.05])
                    
        # Fill remaining slots if needed
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        initial_sum = sum(circle[2] for circle in circles)
        if initial_sum > best_sum:
            best_sum = initial_sum
            best_circles = circles.copy()
        
        # Strategy 3: Concentric rings approach
        circles = []
        center_x, center_y = 0.5, 0.5
        radius_step = 0.15
        angle_step = 0.2
        
        # Place circles in concentric rings
        ring_radius = 0.05
        for i in range(n):
            if len(circles) >= n:
                break
            angle = i * angle_step
            x = center_x + ring_radius * math.cos(angle)
            y = center_y + ring_radius * math.sin(angle)
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                circles.append([x, y, 0.05])
            if (i + 1) % 10 == 0:  # Move to next ring
                ring_radius += radius_step
                
        # Fill remaining slots if needed
        while len(circles) < n:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            circles.append([x, y, 0.05])
            
        initial_sum = sum(circle[2] for circle in circles)
        if initial_sum > best_sum:
            best_sum = initial_sum
            best_circles = circles.copy()
        
        return np.array(best_circles[:n])
    
    # Constraint checking function
    def check_constraints(circles):
        """Check if all constraints are satisfied"""
        n = len(circles)
        
        # Check boundary constraints
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                r_sum = r1 + r2
                if dist_sq < r_sum * r_sum:
                    return False
                    
        return True
    
    # Evaluate fitness function
    def evaluate_fitness(individual):
        """Evaluate how good a solution is"""
        circles = np.array(individual).reshape(-1, 3)
        if not check_constraints(circles):
            # Penalize invalid solutions heavily
            return (-1000000,)
        # Return negative sum of radii (since we minimize in DEAP)
        return (-np.sum(circles[:, 2]),)
    
    # Mutation function
    def mutate_individual(individual):
        """Mutate an individual by slightly adjusting positions and radii"""
        mutated = individual.copy()
        n = len(mutated) // 3
        
        # Mutate some circles
        for i in range(n):
            if random.random() < 0.3:  # 30% chance to mutate
                # Randomly choose what to mutate
                if random.random() < 0.7:  # Mutate position
                    idx = random.randint(0, 2)  # 0=x, 1=y, 2=r
                    if idx < 2:  # Position
                        mutated[i*3 + idx] += random.uniform(-0.02, 0.02)
                        # Keep within bounds
                        mutated[i*3 + idx] = max(0.05, min(0.95, mutated[i*3 + idx]))
                    else:  # Radius
                        mutated[i*3 + idx] += random.uniform(-0.01, 0.01)
                        # Keep within reasonable bounds
                        mutated[i*3 + idx] = max(0.01, min(0.45, mutated[i*3 + idx]))
                else:  # Mutate radius only
                    mutated[i*3 + 2] += random.uniform(-0.01, 0.01)
                    mutated[i*3 + 2] = max(0.01, min(0.45, mutated[i*3 + 2]))
        
        return tuple(mutated),
    
    # Create evolutionary algorithm components
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     lambda: random.uniform(0.05, 0.95) if i < 64 else random.uniform(0.01, 0.45), 
                     n=96)  # 32 circles * 3 parameters
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fitness)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolutionary algorithm for initial optimization
    try:
        # Create initial population
        pop = toolbox.population(n=50)
        
        # Initialize with our best placement
        initial_circles = create_initial_placement()
        for i in range(min(10, len(pop))):
            for j in range(32):
                pop[i][j*3] = initial_circles[j][0]  # x
                pop[i][j*3+1] = initial_circles[j][1]  # y
                pop[i][j*3+2] = initial_circles[j][2]  # r
        
        # Run evolution
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Run for fewer generations due to time constraints
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, 
                                          ngen=30, stats=stats, halloffame=hof, verbose=False)
        
        # Get best solution
        best_individual = hof[0]
        circles = np.array(best_individual).reshape(-1, 3)
        
    except Exception as e:
        # Fallback to initial placement if evolutionary algorithm fails
        circles = create_initial_placement()
    
    # Local optimization with scipy for final refinement
    def objective(vars):
        # Sum of radii (we want to maximize this)
        circles = vars.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we minimize
    
    def constraint_func(vars):
        # Return array of constraint values (positive means violated)
        constraints = []
        n = 32
        
        # Reshape variables
        circles = vars.reshape(-1, 3)
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        radii = circles[:, 2]
        
        # Boundary constraints
        for i in range(n):
            x, y, r = x_coords[i], y_coords[i], radii[i]
            constraints.extend([
                x - r,           # left boundary
                1 - x - r,       # right boundary
                y - r,           # bottom boundary
                1 - y - r        # top boundary
            ])
        
        # Non-overlap constraints - use efficient pairwise computation
        # Vectorized distance calculation
        for i in range(n):
            for j in range(i+1, n):
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dist_sq = dx*dx + dy*dy
                r_sum = radii[i] + radii[j]
                constraints.append(dist_sq - r_sum*r_sum)
                
        return np.array(constraints)
    
    # Try to refine with optimization if we have valid circles
    if len(circles) == 32:
        try:
            # Flatten into variables [x1, y1, r1, x2, y2, r2, ...]
            initial_vars = circles.flatten()
            
            # Set bounds for variables (x, y, r)
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            # Try optimization with better bounds and constraints
            result = minimize(
                objective,
                initial_vars,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                optimized_vars = result.x
                circles = optimized_vars.reshape(-1, 3)
        except:
            pass
    
    # Final validation and adjustment
    # Make sure all circles are valid
    for i in range(n):
        x, y, r = circles[i]
        # Ensure valid bounds
        circles[i][0] = max(r, min(1-r, x))
        circles[i][1] = max(r, min(1-r, y))
        circles[i][2] = max(0.001, min(0.499, r))
    
    return circles


# EVOLVE-BLOCK-END
