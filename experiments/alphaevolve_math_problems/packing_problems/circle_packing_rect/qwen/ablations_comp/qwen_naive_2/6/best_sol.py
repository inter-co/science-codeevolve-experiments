# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import math
from itertools import combinations
import time
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import random
from deap import base, creator, tools, algorithms
import multiprocessing as mp

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses an evolutionary algorithm approach combined with local optimization to achieve better results.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Rectangle dimensions: perimeter = 4, so width + height = 2
    best_sum_radii = 0
    best_circles = None
    
    # More sophisticated initialization with multiple strategies
    def create_multiple_initializations(width, height, n=21):
        """Create several initial configurations using different strategies"""
        initial_configs = []
        
        # Strategy 1: Hexagonal packing
        def hexagonal_packing():
            circles = []
            rows = 5
            cols_per_row = [4, 5, 4, 5, 4]
            
            cell_width = width / max(cols_per_row)
            cell_height = height / rows
            
            idx = 0
            for row in range(rows):
                cols = cols_per_row[row]
                row_y = (row + 0.5) * cell_height
                
                x_offset = (row % 2) * cell_width * 0.5
                
                for col in range(cols):
                    if idx >= n:
                        break
                    row_x = (col + 0.5) * cell_width + x_offset
                    
                    row_x = max(cell_width/2, min(width - cell_width/2, row_x))
                    row_y = max(cell_height/2, min(height - cell_height/2, row_y))
                    
                    max_radius = min(row_x, width - row_x, row_y, height - row_y)
                    radius = min(max_radius, min(cell_width, cell_height) * 0.35)
                    
                    circles.append([row_x, row_y, radius])
                    idx += 1
                    
                if idx >= n:
                    break
            
            # Fill remaining positions if needed
            while len(circles) < n:
                x = np.random.uniform(0.05, width - 0.05)
                y = np.random.uniform(0.05, height - 0.05)
                max_radius = min(x, width - x, y, height - y)
                radius = min(max_radius, 0.1)
                circles.append([x, y, radius])
            
            return np.array(circles)
        
        # Strategy 2: Grid-based packing
        def grid_packing():
            circles = []
            rows = 5
            cols = 5
            
            cell_width = width / cols
            cell_height = height / rows
            
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= n:
                        break
                    x = (j + 0.5) * cell_width
                    y = (i + 0.5) * cell_height
                    
                    x = max(cell_width/2, min(width - cell_width/2, x))
                    y = max(cell_height/2, min(height - cell_height/2, y))
                    
                    max_radius = min(x, width - x, y, height - y)
                    radius = min(max_radius, min(cell_width, cell_height) * 0.4)
                    
                    circles.append([x, y, radius])
            
            # Fill remaining positions
            while len(circles) < n:
                x = np.random.uniform(0.05, width - 0.05)
                y = np.random.uniform(0.05, height - 0.05)
                max_radius = min(x, width - x, y, height - y)
                radius = min(max_radius, 0.1)
                circles.append([x, y, radius])
            
            return np.array(circles)
        
        # Strategy 3: Random with density control
        def random_density_control():
            circles = []
            max_attempts = 1000
            
            for _ in range(n):
                attempts = 0
                placed = False
                
                while not placed and attempts < max_attempts:
                    x = np.random.uniform(0.05, width - 0.05)
                    y = np.random.uniform(0.05, height - 0.05)
                    
                    # Estimate max radius at this position
                    max_radius = min(x, width - x, y, height - y)
                    
                    # Check if this position conflicts with existing circles
                    valid = True
                    for cx, cy, cr in circles:
                        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                        if dist < cr + max_radius * 0.5:  # Conservative check
                            valid = False
                            break
                    
                    if valid:
                        radius = min(max_radius, 0.15)
                        circles.append([x, y, radius])
                        placed = True
                    
                    attempts += 1
                
                # If couldn't place, use fallback
                if not placed:
                    x = np.random.uniform(0.05, width - 0.05)
                    y = np.random.uniform(0.05, height - 0.05)
                    max_radius = min(x, width - x, y, height - y)
                    radius = min(max_radius, 0.1)
                    circles.append([x, y, radius])
            
            return np.array(circles)
        
        # Generate multiple initial configurations
        initial_configs.append(hexagonal_packing())
        initial_configs.append(grid_packing())
        initial_configs.append(random_density_control())
        
        # Add some diversity with slight perturbations
        for config in initial_configs[:]:
            perturbed = config.copy()
            for i in range(len(perturbed)):
                perturbed[i][0] += np.random.normal(0, width * 0.02)
                perturbed[i][1] += np.random.normal(0, height * 0.02)
                perturbed[i][2] += np.random.normal(0, 0.01)
                perturbed[i][2] = max(0.01, perturbed[i][2])
            initial_configs.append(perturbed)
        
        return initial_configs
    
    # Constraint checking functions
    def check_overlap(circles):
        """Check if any circles overlap"""
        coords = circles[:, :2]
        radii = circles[:, 2]
        
        if len(circles) < 2:
            return False
            
        dist_matrix = distance_matrix(coords, coords)
        
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dist = dist_matrix[i, j]
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    return True
        return False
    
    def check_bounds(circles, width, height):
        """Check if all circles are within bounds"""
        for x, y, r in circles:
            if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
                return False
        return True
    
    def fitness_function(circles, width, height):
        """Calculate fitness: sum of radii with penalties for violations"""
        if check_overlap(circles) or not check_bounds(circles, width, height):
            return -1e10  # Invalid solution penalty
        
        return np.sum(circles[:, 2])
    
    # Evolutionary algorithm implementation
    def evolutionary_optimization(initial_circles, width, height, max_generations=50):
        """Use evolutionary algorithm to optimize circle placement"""
        
        # Individual representation: [x1, y1, r1, x2, y2, r2, ...]
        def individual_to_circles(individual):
            circles = []
            for i in range(0, len(individual), 3):
                circles.append([individual[i], individual[i+1], individual[i+2]])
            return np.array(circles)
        
        def circles_to_individual(circles):
            individual = []
            for x, y, r in circles:
                individual.extend([x, y, r])
            return individual
        
        # Create DEAP toolbox
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Attribute generator
        def generate_individual():
            # Take one of the initial configurations and slightly perturb it
            config = initial_circles[np.random.randint(len(initial_circles))]
            individual = circles_to_individual(config)
            
            # Add small random perturbations
            for i in range(len(individual)):
                if i % 3 == 2:  # Radius
                    individual[i] += np.random.normal(0, 0.02)
                    individual[i] = max(0.01, individual[i])
                else:  # Position
                    individual[i] += np.random.normal(0, min(width, height) * 0.05)
                    # Keep within bounds
                    if i % 3 == 0:  # X coordinate
                        individual[i] = max(0.01, min(width - 0.01, individual[i]))
                    elif i % 3 == 1:  # Y coordinate
                        individual[i] = max(0.01, min(height - 0.01, individual[i]))
            
            return creator.Individual(individual)
        
        toolbox.register("individual", generate_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def eval_fitness(individual):
            circles = individual_to_circles(individual)
            return fitness_function(circles, width, height),
        
        toolbox.register("evaluate", eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=min(width, height) * 0.02, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create population
        pop = toolbox.population(n=50)
        
        # Run evolution
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        try:
            pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                             ngen=max_generations, stats=stats, 
                                             halloffame=hof, verbose=False)
            
            if hof:
                best_individual = hof[0]
                best_circles = individual_to_circles(best_individual)
                return best_circles
        except Exception:
            pass
        
        # If evolutionary fails, return the best from initial configs
        return initial_circles[0]
    
    # Local optimization refinement
    def refine_solution(circles, width, height, max_iter=100):
        """Apply local optimization to improve solution"""
        def objective(params):
            circles = params.reshape(-1, 3)
            return -np.sum(circles[:, 2])  # Negative because we want to maximize
        
        def constraint_func(params):
            circles = params.reshape(-1, 3)
            constraints = []
            
            # Distance constraints (no overlaps)
            coords = circles[:, :2]
            radii = circles[:, 2]
            
            if len(circles) > 1:
                dist_matrix = distance_matrix(coords, coords)
                for i in range(len(circles)):
                    for j in range(i+1, len(circles)):
                        dist = dist_matrix[i, j]
                        min_dist = radii[i] + radii[j]
                        constraints.append(dist - min_dist)  # Should be >= 0
            
            # Boundary constraints
            for i in range(len(circles)):
                x, y, r = circles[i]
                constraints.extend([
                    x - r,              # left boundary
                    width - x - r,      # right boundary
                    y - r,              # bottom boundary
                    height - y - r      # top boundary
                ])
            
            return np.array(constraints)
        
        # Set up bounds
        bounds = []
        for i in range(21):
            bounds.extend([(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)])
        
        # Initial parameters
        initial_params = circles.flatten()
        
        # Define constraints
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        # Optimize
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6},
                tol=1e-6
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                return optimized_circles
        except Exception:
            pass
        
        return circles
    
    # Try multiple aspect ratios
    aspect_ratios = [0.8, 1.0, 1.2, 1.5, 1.618, 2.0]
    
    for ratio in aspect_ratios:
        width = 2.0 / (1 + ratio)  # width + height = 2, and width/height = ratio
        height = 2.0 / (1 + 1/ratio)
        
        # Create multiple initial configurations
        initial_configs = create_multiple_initializations(width, height)
        
        # Try evolutionary optimization on each
        for config in initial_configs:
            try:
                # Apply evolutionary optimization
                evolved_circles = evolutionary_optimization(config, width, height, max_generations=30)
                
                # Refine with local optimization
                refined_circles = refine_solution(evolved_circles, width, height)
                
                # Check fitness
                current_sum = np.sum(refined_circles[:, 2])
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_circles = refined_circles.copy()
                    print(f"Found better solution: {best_sum_radii}")
                    
            except Exception as e:
                continue
    
    # If no improvement found, return best from initial configurations
    if best_circles is None:
        # Use a more systematic approach
        width, height = 1.0, 1.0
        circles = np.zeros((21, 3))
        
        # Dense hexagonal packing
        rows = 4
        cols_per_row = [5, 6, 5, 5]
        
        cell_width = width / max(cols_per_row)
        cell_height = height / rows
        
        idx = 0
        for row in range(rows):
            cols = cols_per_row[row]
            row_y = (row + 0.5) * cell_height
            
            x_offset = (row % 2) * cell_width * 0.5
            
            for col in range(cols):
                if idx >= 21:
                    break
                row_x = (col + 0.5) * cell_width + x_offset
                
                row_x = max(cell_width/2, min(width - cell_width/2, row_x))
                row_y = max(cell_height/2, min(height - cell_height/2, row_y))
                
                max_radius = min(row_x, width - row_x, row_y, height - row_y)
                radius = min(max_radius, min(cell_width, cell_height) * 0.38)
                
                circles[idx] = [row_x, row_y, radius]
                idx += 1
                
            if idx >= 21:
                break
        
        best_circles = refine_solution(circles, width, height)
        best_sum_radii = np.sum(best_circles[:, 2])
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
