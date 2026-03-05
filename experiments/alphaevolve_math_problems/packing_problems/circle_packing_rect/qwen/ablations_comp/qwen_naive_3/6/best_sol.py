# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import random
import time
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from numba import jit
import math
from scipy.spatial import distance_matrix
import optuna
from sklearn.cluster import KMeans
from typing import Tuple

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid optimization approach combining global search with local refinement.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Try different aspect ratios to find optimal
    # Better approach: optimize both width and height
    width, height = 1.2, 0.8  # Initial guess for 1.5:1 ratio
    
    # Number of circles
    n = 21
    
    # Improved initialization using a more systematic approach
    def generate_better_initialization(width, height):
        circles = []
        
        # Calculate area and target radius
        total_area = width * height
        circle_area = total_area / n * 0.85  # Leave some margin for optimal packing
        avg_radius = np.sqrt(circle_area / np.pi)
        
        # Try to create a more efficient packing pattern
        # Use a combination of hexagonal and grid packing strategies
        spacing = 2 * avg_radius * 0.95  # Slightly less than diameter
        hex_radius = spacing * np.sqrt(3) / 2
        
        # Create a dense hexagonal pattern
        rows = int(np.ceil(height / hex_radius)) + 3
        cols = int(np.ceil(width / spacing)) + 3
        
        placed_circles = []
        for i in range(rows):
            for j in range(cols):
                # Offset odd rows
                x_offset = (i % 2) * spacing / 2
                x = spacing/2 + j * spacing + x_offset
                y = hex_radius/2 + i * hex_radius
                
                # Only place if within bounds with margin
                if 0.05 <= x <= width - 0.05 and 0.05 <= y <= height - 0.05:
                    placed_circles.append([x, y, avg_radius * 0.9])
        
        # If we have enough circles, use them; otherwise add random ones
        if len(placed_circles) >= n:
            return np.array(placed_circles[:n])
        else:
            # Fill with random circles in valid positions
            result = np.array(placed_circles)
            remaining = n - len(placed_circles)
            for _ in range(remaining):
                x = random.uniform(0.05, width - 0.05)
                y = random.uniform(0.05, height - 0.05)
                # Use smaller radius initially
                r = min(avg_radius * 0.5, min(width, height) / 4)
                result = np.vstack([result, [x, y, r]])
            return result
    
    # More efficient constraint checking using vectorized operations
    @jit(nopython=True)
    def check_containment_jit(x, y, r, w, h):
        """Check if circle is contained within rectangle"""
        return (x - r >= 0.001 and y - r >= 0.001 and x + r <= w - 0.001 and y + r <= h - 0.001)
    
    @jit(nopython=True)
    def check_non_overlap_jit(x1, y1, r1, x2, y2, r2):
        """Check if two circles don't overlap"""
        dx = x1 - x2
        dy = y1 - y2
        dist_sq = dx*dx + dy*dy
        radii_sum = r1 + r2
        return dist_sq >= radii_sum * radii_sum
    
    # Optimized evaluation function with better penalty system
    def evaluate_individual(individual):
        """Evaluate fitness of individual (negative sum of radii for minimization)"""
        # Reshape individual to (n, 3) array
        circles = individual.reshape((n, 3))
        
        # Extract positions and radii
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Check constraints efficiently
        total_penalty = 0.0
        
        # Containment penalty - vectorized check
        for i in range(n):
            x, y, r = positions[i][0], positions[i][1], radii[i]
            if not check_containment_jit(x, y, r, width, height):
                total_penalty += 1000000.0  # Large penalty
        
        # Overlap penalty - more efficient version using vectorized operations
        # For each circle, check against all others
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = positions[i][0], positions[i][1], radii[i]
                x2, y2, r2 = positions[j][0], positions[j][1], radii[j]
                if not check_non_overlap_jit(x1, y1, r1, x2, y2, r2):
                    # Penalty based on how much they overlap
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_sq = dx*dx + dy*dy
                    radii_sum = r1 + r2
                    overlap = radii_sum * radii_sum - dist_sq
                    # Use exponential penalty to strongly discourage overlaps
                    total_penalty += 100000.0 * max(0.0, overlap)**2
        
        # Return negative sum of radii plus penalties (minimize this value)
        return (-np.sum(radii) + total_penalty,)
    
    # Enhanced evolutionary algorithm with better parameters
    def run_evolution():
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Attribute generator with better initialization
        def create_individual():
            individual = np.zeros((n, 3))
            init_circles = generate_better_initialization(width, height)
            
            # Copy from initialization
            for i in range(n):
                individual[i, 0] = init_circles[i, 0]  # x
                individual[i, 1] = init_circles[i, 1]  # y
                individual[i, 2] = init_circles[i, 2]  # r
            
            # Add small random perturbations to avoid getting stuck
            for i in range(n):
                individual[i, 0] += random.uniform(-0.02, 0.02)
                individual[i, 1] += random.uniform(-0.02, 0.02)
                individual[i, 2] += random.uniform(-0.003, 0.003)
                
                # Keep within bounds
                individual[i, 0] = max(0.05, min(width - 0.05, individual[i, 0]))
                individual[i, 1] = max(0.05, min(height - 0.05, individual[i, 1]))
                individual[i, 2] = max(0.001, min(min(width, height)/2 - 0.05, individual[i, 2]))
            
            return creator.Individual(individual)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register genetic operators with better settings
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.25)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.15)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        pop = toolbox.population(n=100)  # Larger population for better exploration
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Evolution parameters - more generations and better controls
        CXPB = 0.8   # Higher crossover probability
        MUTPB = 0.2  # Lower mutation probability for more stability
        NGEN = 100   # More generations
        
        # Run evolution
        for gen in range(NGEN):
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
            
            # Apply crossover and mutation on the offspring
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
            
            # Replace the old population with the new population
            pop[:] = offspring
            
            # Print progress every 20 generations
            if gen % 20 == 0:
                best_fitness = max([ind.fitness.values[0] for ind in pop])
                print(f"Generation {gen}: Best fitness = {-best_fitness}")
        
        # Return best individual
        best_ind = tools.selBest(pop, 1)[0]
        return best_ind
    
    # Alternative: Use Bayesian optimization for better local search
    def optimize_with_optimization_methods():
        # Use Optuna for hyperparameter tuning and better optimization
        def objective(trial):
            # Sample rectangle dimensions
            w = trial.suggest_float('width', 0.8, 1.8)
            h = 2.0 - w  # Perimeter constraint
            
            # Sample initial configuration parameters
            init_radius = trial.suggest_float('radius', 0.05, 0.2)
            
            # Generate initial configuration
            circles = generate_better_initialization(w, h)
            
            # Set up optimization parameters
            bounds = []
            for i in range(n):
                bounds.extend([
                    (0.001, w - 0.001),   # x bounds
                    (0.001, h - 0.001),   # y bounds
                    (0.001, min(w, h)/2 - 0.001)  # r bounds
                ])
            
            # Simple local optimization
            try:
                # Just use the initial configuration for now as a baseline
                return -np.sum(circles[:, 2])
            except:
                return float('inf')
        
        # Run optimization to find good starting point
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        
        return study.best_params
    
    # Main optimization process - focus on simpler but more effective approach
    try:
        start_time = time.time()
        
        # Start with a better initialization approach
        # Try multiple different configurations to find good starting points
        best_result = None
        best_sum = -float('inf')
        
        # Try different rectangle dimensions
        aspect_ratios = [(1.0, 1.0), (1.2, 0.8), (1.5, 0.5), (0.8, 1.2), (0.5, 1.5)]
        
        for w, h in aspect_ratios:
            # Generate initial configuration
            circles = generate_better_initialization(w, h)
            
            # Simple optimization with better bounds
            try:
                # Create flattened parameter array for optimization
                flat_params = circles.flatten()
                
                # Define bounds more carefully
                bounds = []
                for i in range(n):
                    bounds.extend([
                        (0.001, w - 0.001),   # x bounds
                        (0.001, h - 0.001),   # y bounds
                        (0.001, min(w, h)/2 - 0.001)  # r bounds
                    ])
                
                # Objective function for local optimization
                def objective(params):
                    # Reshape and extract values
                    circles_flat = params.reshape((n, 3))
                    radii = circles_flat[:, 2]
                    return -np.sum(radii)  # Minimize negative sum (maximize sum)
                
                # Constraint function to check containment
                def containment_constraint(params):
                    circles_flat = params.reshape((n, 3))
                    results = []
                    for i in range(n):
                        x, y, r = circles_flat[i]
                        # Circle must be within bounds
                        results.append(x - r)  # >= 0
                        results.append(y - r)  # >= 0
                        results.append(w - x - r)  # >= 0
                        results.append(h - y - r)  # >= 0
                    return np.array(results)
                
                # Constraint function to check non-overlap
                def overlap_constraint(params):
                    circles_flat = params.reshape((n, 3))
                    results = []
                    for i in range(n):
                        x1, y1, r1 = circles_flat[i]
                        for j in range(i+1, n):
                            x2, y2, r2 = circles_flat[j]
                            # Distance squared between centers
                            dx = x1 - x2
                            dy = y1 - y2
                            dist_sq = dx*dx + dy*dy
                            # Must be >= (r1 + r2)^2 for no overlap
                            radii_sum_sq = (r1 + r2) * (r1 + r2)
                            results.append(dist_sq - radii_sum_sq)
                    return np.array(results)
                
                # Optimization constraints
                constraints = [
                    {'type': 'ineq', 'fun': containment_constraint},
                    {'type': 'ineq', 'fun': overlap_constraint}
                ]
                
                # Try different optimization approaches
                result = minimize(
                    objective,
                    flat_params,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6}
                )
                
                if result.success:
                    final_params = result.x
                    final_circles = final_params.reshape((n, 3))
                    sum_radii = np.sum(final_circles[:, 2])
                    
                    if sum_radii > best_sum:
                        best_sum = sum_radii
                        best_result = final_circles
                        
            except Exception as e:
                continue  # Skip this attempt if optimization fails
        
        # If we didn't find a good solution, fall back to evolutionary approach
        if best_result is None:
            best_individual = run_evolution()
            best_result = best_individual.reshape((n, 3))
        
        eval_time = time.time() - start_time
        
    except Exception as e:
        # Fallback to better hexagonal packing with more careful optimization
        circles = np.zeros((n, 3))
        # Use more precise hexagonal packing with better parameters
        total_area = width * height
        circle_area = total_area / n * 0.85
        avg_radius = np.sqrt(circle_area / np.pi)
        
        spacing = 2 * avg_radius * 0.9
        hex_radius = spacing * np.sqrt(3) / 2
        
        row_count = int(np.ceil(height / hex_radius)) + 1
        col_count = int(np.ceil(width / spacing)) + 1
        
        idx = 0
        for i in range(row_count):
            for j in range(col_count):
                if idx >= n:
                    break
                x = 0.05 + j * spacing + (i % 2) * spacing / 2
                y = 0.05 + i * hex_radius
                # Make sure we stay within bounds
                if x <= width - 0.05 and y <= height - 0.05:
                    circles[idx] = [x, y, avg_radius * 0.85]
                    idx += 1
                else:
                    circles[idx] = [width/2, height/2, avg_radius * 0.5]
                    idx += 1
                if idx >= n:
                    break
        
        best_result = circles
    
    return best_result


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
