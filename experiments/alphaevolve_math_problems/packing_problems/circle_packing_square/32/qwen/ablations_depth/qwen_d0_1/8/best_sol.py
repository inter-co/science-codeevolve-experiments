# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from deap import base, creator, tools, algorithms
import random
import time
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach: advanced evolutionary algorithm + physics-inspired refinement + 
    mathematical programming for final optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    max_time = 60  # seconds
    start_time = time.time()
    
    # Use evolutionary algorithm for global search with improved strategy
    toolbox = base.Toolbox()
    
    # Create individual (32 circles = 96 parameters: x1,y1,r1,x2,y2,r2,...)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    def create_individual():
        # Generate better initial configuration using hexagonal packing inspiration
        individual = []
        
        # Try to create a more uniform initial distribution
        # Start with a more systematic approach
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        # Add some randomness but keep good distribution
        for i in range(n):
            # Grid positions with more significant randomization
            row = i // grid_size
            col = i % grid_size
            x = spacing_x * (col + 1) + random.uniform(-0.02, 0.02)
            y = spacing_y * (row + 1) + random.uniform(-0.02, 0.02)
            
            # Ensure positions are within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            
            # Initial radius based on density considerations
            # Start with a more informed initial guess
            r = min(0.08, 0.4 / np.sqrt(n))  # Slightly larger initial radii
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    def evaluate(individual):
        """Evaluate fitness as sum of radii, penalize violations with better penalty scheme"""
        circles = np.array(individual).reshape(-1, 3)
        total_radius = np.sum(circles[:, 2])
        
        # Penalty for constraint violations - more sophisticated penalty system
        penalty = 0
        
        # Boundary penalties - stronger penalty for severe violations
        for i in range(n):
            x, y, r = circles[i]
            # Calculate how much we violate boundaries
            left_violation = max(0, r - x)
            right_violation = max(0, x + r - 1)
            bottom_violation = max(0, r - y)
            top_violation = max(0, y + r - 1)
            
            # Quadratic penalty for boundary violations (stronger penalties for larger violations)
            penalty += (left_violation**2 + right_violation**2 + bottom_violation**2 + top_violation**2) * 1000000
        
        # Overlap penalties - quadratic penalty for overlap amounts
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < r1 + r2:
                    # Quadratic penalty based on overlap amount
                    overlap = (r1 + r2) - dist
                    penalty += overlap**2 * 10000000  # Much higher penalty for overlaps
                    
        # Additional penalty for very small radii (encourage larger circles)
        small_radius_penalty = 0
        for i in range(n):
            r = circles[i, 2]
            if r < 0.005:
                small_radius_penalty += (0.005 - r)**2 * 100000
                
        penalty += small_radius_penalty
        
        # Bonus for larger radii (encourage growth)
        bonus = 0
        for i in range(n):
            r = circles[i, 2]
            if r > 0.05:
                bonus += r * 100  # Encourage larger radii
        
        return (total_radius - penalty + bonus,)
    
    def mutate(individual):
        # More sophisticated mutation with adaptive strategies
        idx = random.randint(0, len(individual)-1)
        if idx % 3 == 0:  # x coordinate
            # Larger mutations for x with adaptive scaling
            mutation_strength = 0.03 if random.random() < 0.7 else 0.06
            individual[idx] = max(0.001, min(0.999, individual[idx] + random.gauss(0, mutation_strength)))
        elif idx % 3 == 1:  # y coordinate
            # Larger mutations for y with adaptive scaling
            mutation_strength = 0.03 if random.random() < 0.7 else 0.06
            individual[idx] = max(0.001, min(0.999, individual[idx] + random.gauss(0, mutation_strength)))
        else:  # radius
            # Smaller mutations for radius with adaptive scaling
            mutation_strength = 0.01 if random.random() < 0.7 else 0.02
            individual[idx] = max(0.001, min(0.499, individual[idx] + random.gauss(0, mutation_strength)))
        return individual,
    
    def crossover(ind1, ind2):
        # Improved crossover with adaptive recombination
        # Use heuristic crossover with preference for better individuals
        for i in range(len(ind1)):
            if random.random() < 0.6:  # More selective crossover
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Initialize population with better diversity and quality
    population = toolbox.population(n=150)  # Even larger population for better exploration
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolution parameters optimized for better convergence
    CXPB = 0.7   # Moderate crossover probability
    MUTPB = 0.5  # Higher mutation probability for better exploration
    NGEN = 150   # More generations
    
    # Main evolution loop with enhanced early stopping criteria
    best_fitness_history = []
    stagnation_count = 0
    max_stagnation = 25
    
    for gen in range(NGEN):
        if time.time() - start_time > max_time * 0.6:  # Leave more time for final refinement
            break
            
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Replace the old population with the new one
        population[:] = offspring
        
        # Track best fitness for early stopping
        current_best = max([ind.fitness.values[0] for ind in population])
        best_fitness_history.append(current_best)
        
        # Check for stagnation with more stringent criteria
        if len(best_fitness_history) > 15:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-15]
            if recent_improvement < 1e-6:
                stagnation_count += 1
                if stagnation_count > max_stagnation:
                    break
            else:
                stagnation_count = 0
    
    # Get the best individual
    best_individual = tools.selBest(population, 1)[0]
    circles = np.array(best_individual).reshape(-1, 3)
    
    # Refine with enhanced local optimization using multiple approaches
    refined_circles = refine_with_advanced_local_optimization(circles, max_time - (time.time() - start_time))
    
    return refined_circles

def refine_with_advanced_local_optimization(initial_circles, remaining_time):
    """Advanced refinement using multiple sophisticated optimization approaches"""
    n = len(initial_circles)
    
    # Try multiple approaches and return the best
    best_result = initial_circles.copy()
    best_radius_sum = np.sum(initial_circles[:, 2])
    
    # Approach 1: Physics-inspired optimization with force-based relaxation
    try:
        result1 = physics_based_relaxation(initial_circles)
        if result1 is not None:
            radius_sum = np.sum(result1[:, 2])
            if radius_sum > best_radius_sum:
                best_result = result1
                best_radius_sum = radius_sum
    except:
        pass
    
    # Approach 2: Enhanced SLSQP optimization with better initial setup
    try:
        result2 = enhanced_slqp_optimization(initial_circles)
        if result2 is not None:
            radius_sum = np.sum(result2[:, 2])
            if radius_sum > best_radius_sum:
                best_result = result2
                best_radius_sum = radius_sum
    except:
        pass
    
    # Approach 3: Simulated annealing for further improvement
    try:
        result3 = simulated_annealing_refinement(initial_circles)
        if result3 is not None:
            radius_sum = np.sum(result3[:, 2])
            if radius_sum > best_radius_sum:
                best_result = result3
                best_radius_sum = radius_sum
    except:
        pass
    
    return best_result

def physics_based_relaxation(initial_circles):
    """Physics-inspired relaxation method with repulsive forces"""
    circles = initial_circles.copy()
    n = len(circles)
    
    # Parameters for physics simulation
    max_iterations = 1000
    dt = 0.01
    k_repel = 1000.0  # Repulsion constant
    k_boundary = 10000.0  # Boundary attraction constant
    
    # Spatial index for efficient neighbor search
    tree = KDTree(circles[:, :2])
    
    for iteration in range(max_iterations):
        forces = np.zeros((n, 2))  # Force on each circle
        
        # Calculate repulsion forces from other circles
        for i in range(n):
            x1, y1, r1 = circles[i]
            
            # Find nearby circles using spatial indexing
            neighbors = tree.query_ball_point([x1, y1], 2*(r1 + 0.01))
            
            for j in neighbors:
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < r1 + r2 and dist > 0:
                        # Repulsion force
                        force_magnitude = k_repel * (r1 + r2 - dist) / (dist + 1e-10)
                        forces[i, 0] += force_magnitude * dx / dist
                        forces[i, 1] += force_magnitude * dy / dist
        
        # Add boundary forces
        for i in range(n):
            x, y, r = circles[i]
            # Boundary forces (attract to edges when close)
            boundary_force_x = 0
            boundary_force_y = 0
            
            if x - r < 0.01:
                boundary_force_x += k_boundary * (0.01 - (x - r))
            if x + r > 0.99:
                boundary_force_x -= k_boundary * ((x + r) - 0.99)
                
            if y - r < 0.01:
                boundary_force_y += k_boundary * (0.01 - (y - r))
            if y + r > 0.99:
                boundary_force_y -= k_boundary * ((y + r) - 0.99)
            
            forces[i, 0] += boundary_force_x
            forces[i, 1] += boundary_force_y
        
        # Update positions
        for i in range(n):
            x, y, r = circles[i]
            # Limit movement to prevent overshooting
            dx = forces[i, 0] * dt
            dy = forces[i, 1] * dt
            
            # Clamp forces to prevent large jumps
            dx = np.clip(dx, -0.01, 0.01)
            dy = np.clip(dy, -0.01, 0.01)
            
            new_x = max(0.001, min(0.999, x + dx))
            new_y = max(0.001, min(0.999, y + dy))
            
            circles[i] = [new_x, new_y, r]
        
        # Early stopping if forces are small
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force < 1e-5:
            break
    
    return circles

def enhanced_slqp_optimization(initial_circles):
    """Enhanced SLSQP optimization with better constraint handling and preprocessing"""
    n = len(initial_circles)
    
    # Flatten for optimization
    x0 = initial_circles.flatten()
    
    def objective(x_flat):
        # Extract circles
        circles = x_flat.reshape(-1, 3)
        # Maximize sum of radii (minimize negative)
        return -np.sum(circles[:, 2])
    
    def constraint_func(x_flat):
        """Return constraint violations (positive means violation)"""
        circles = x_flat.reshape(-1, 3)
        constraints = []
        
        # Boundary constraints: x-r >= 0, y-r >= 0, 1-x-r >= 0, 1-y-r >= 0
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([
                x - r,           # x - r >= 0
                y - r,           # y - r >= 0  
                1 - x - r,       # 1 - x - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        
        # Overlap constraints: distance - r1 - r2 >= 0
        # Use a more efficient approach for overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(dist - r1 - r2)
                
        return np.array(constraints)
    
    # Set up bounds with tighter ranges
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Set up constraints
    cons = [{'type': 'ineq', 'fun': constraint_func}]
    
    # Run optimization with better settings and multiple attempts
    # First try with default settings
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                     options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-6})
    
    if result.success:
        final_circles = result.x.reshape(-1, 3)
        return final_circles
    
    return None

def simulated_annealing_refinement(initial_circles):
    """Simulated annealing refinement for escaping local optima"""
    circles = initial_circles.copy()
    n = len(circles)
    
    current_solution = circles.copy()
    current_value = np.sum(circles[:, 2])
    
    # Annealing parameters
    temperature = 1.0
    min_temperature = 1e-6
    cooling_rate = 0.995
    max_iterations = 10000
    
    for iteration in range(max_iterations):
        # Cool down temperature
        if temperature < min_temperature:
            break
        temperature *= cooling_rate
        
        # Generate neighbor solution
        neighbor = current_solution.copy()
        
        # Choose random circle to perturb
        i = random.randint(0, n-1)
        x, y, r = neighbor[i]
        
        # Perturb position and radius
        dx = random.uniform(-0.01, 0.01)
        dy = random.uniform(-0.01, 0.01)
        dr = random.uniform(-0.01, 0.01)
        
        new_x = max(0.001, min(0.999, x + dx))
        new_y = max(0.001, min(0.999, y + dy))
        new_r = max(0.001, min(0.499, r + dr))
        
        neighbor[i] = [new_x, new_y, new_r]
        
        # Check constraints for neighbor
        valid = True
        # Check boundary constraints
        for j in range(n):
            nx, ny, nr = neighbor[j]
            if nx - nr < 0 or nx + nr > 1 or ny - nr < 0 or ny + nr > 1:
                valid = False
                break
        
        # Check overlap constraints
        if valid:
            for j in range(n):
                for k in range(j+1, n):
                    x1, y1, r1 = neighbor[j]
                    x2, y2, r2 = neighbor[k]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    if dist < r1 + r2:
                        valid = False
                        break
        
        if valid:
            neighbor_value = np.sum(neighbor[:, 2])
            delta = neighbor_value - current_value
            
            # Accept or reject based on Metropolis criterion
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_solution = neighbor
                current_value = neighbor_value
    
    return current_solution


# EVOLVE-BLOCK-END
