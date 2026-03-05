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
import itertools

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid optimization approach combining global search with local refinement.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Try different aspect ratios - test several configurations
    # For circle packing, a square-like configuration often works best
    width, height = 1.0, 1.0  # Square configuration
    
    # Number of circles
    n = 21
    
    # Better initialization using more systematic approaches
    def generate_better_initialization():
        circles = []
        
        # Start with a good hexagonal packing approach
        # Calculate optimal circle size based on available area
        total_area = width * height
        circle_area = total_area / n * 0.85  # Leave some margin for optimal packing
        avg_radius = np.sqrt(circle_area / np.pi)
        
        # Create a more efficient hexagonal grid pattern
        spacing = 2 * avg_radius * 0.95  # Slightly less than diameter to allow overlap in planning
        hex_radius = spacing * np.sqrt(3) / 2
        
        # Place circles in hexagonal pattern
        rows = int(np.ceil(height / hex_radius)) + 2
        cols = int(np.ceil(width / spacing)) + 2
        
        placed_circles = []
        for i in range(rows):
            for j in range(cols):
                # Offset odd rows
                x_offset = (i % 2) * spacing / 2
                x = spacing/2 + j * spacing + x_offset
                y = hex_radius/2 + i * hex_radius
                
                # Only place if within bounds with margin
                if 0.01 <= x <= width - 0.01 and 0.01 <= y <= height - 0.01:
                    placed_circles.append([x, y, avg_radius * 0.9])
        
        # If we have enough circles, use them; otherwise add random ones
        if len(placed_circles) >= n:
            return np.array(placed_circles[:n])
        else:
            # Fill with random circles in valid positions
            result = np.array(placed_circles)
            remaining = n - len(placed_circles)
            for _ in range(remaining):
                x = random.uniform(0.01, width - 0.01)
                y = random.uniform(0.01, height - 0.01)
                # Use smaller radius initially
                r = min(avg_radius * 0.3, min(width, height) / 4)
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
    
    # Vectorized constraint checking for performance
    def compute_distance_matrix(positions):
        """Compute distance matrix for all pairs of positions"""
        return cdist(positions, positions)
    
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
                # Heavy penalty for containment violations
                total_penalty += 1000000.0
        
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
                    # Use a quadratic penalty to encourage large overlaps to be reduced
                    total_penalty += 100000.0 * max(0.0, overlap)**2
        
        # Return negative sum of radii plus penalties (minimize this value)
        return (-np.sum(radii) + total_penalty,)
    
    # Enhanced evolutionary algorithm with better parameters and strategies
    def run_evolution():
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Attribute generator with better initialization
        def create_individual():
            individual = np.zeros((n, 3))
            init_circles = generate_better_initialization()
            
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
                individual[i, 0] = max(0.01, min(width - 0.01, individual[i, 0]))
                individual[i, 1] = max(0.01, min(height - 0.01, individual[i, 1]))
                individual[i, 2] = max(0.001, min(min(width, height)/2 - 0.01, individual[i, 2]))
            
            return creator.Individual(individual)
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Register genetic operators with better settings
        toolbox.register("evaluate", evaluate_individual)
        toolbox.register("mate", tools.cxUniform, indpb=0.4)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.3)
        toolbox.register("select", tools.selTournament, tournsize=5)
        
        # Create initial population with higher diversity
        pop = toolbox.population(n=100)
        
        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # Evolution parameters - more generations and better controls
        CXPB = 0.8   # Higher crossover probability
        MUTPB = 0.4  # Higher mutation probability
        NGEN = 100   # More generations for better convergence
        
        # Run evolution with adaptive strategy
        best_fitness_history = []
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
            
            # Track best fitness
            current_best = max([ind.fitness.values[0] for ind in pop])
            best_fitness_history.append(current_best)
            
            # Print progress every 10 generations
            if gen % 10 == 0:
                print(f"Generation {gen}: Best fitness = {-current_best}")
                
                # Adaptive early stopping - if no improvement for 30 generations, stop
                if len(best_fitness_history) > 30:
                    recent_improvement = current_best - max(best_fitness_history[-30:])
                    if recent_improvement < 1e-6:
                        print("Early stopping due to no improvement")
                        break
        
        # Return best individual
        best_ind = tools.selBest(pop, 1)[0]
        return best_ind
    
    # Try evolutionary approach first
    try:
        start_time = time.time()
        best_individual = run_evolution()
        eval_time = time.time() - start_time
        
        # Convert back to circles array
        circles = best_individual.reshape((n, 3))
        
        # Local refinement using a more sophisticated optimization approach
        # Use a two-phase local search: first global, then fine-tuning
        refined_params = circles.flatten()
        
        # Define a more robust objective function for local optimization
        def objective_local(params):
            # Extract radii
            radii = params[2::3]
            # Return negative sum of radii (we want to maximize)
            return -np.sum(radii)
        
        # Constraint functions for local optimization
        def constraint_containment(params):
            results = []
            for i in range(n):
                x, y, r = params[3*i], params[3*i+1], params[3*i+2]
                # Ensure circle is within bounds (with margin)
                results.append(x - r - 0.001)  # Should be >= 0
                results.append(y - r - 0.001)  # Should be >= 0
                results.append(width - x - r - 0.001)  # Should be >= 0
                results.append(height - y - r - 0.001)  # Should be >= 0
            return np.array(results)
        
        def constraint_non_overlap(params):
            results = []
            for i in range(n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                for j in range(i+1, n):
                    x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                    # Distance between centers squared
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_sq = dx*dx + dy*dy
                    # Should be >= sum of radii squared (for no overlap)
                    radii_sum_sq = (r1 + r2) * (r1 + r2)
                    results.append(dist_sq - radii_sum_sq)
            return np.array(results)
        
        # Get bounds - tighter bounds for better optimization
        bounds = []
        for i in range(n):
            bounds.extend([
                (0.001, width - 0.001),   # x bounds
                (0.001, height - 0.001),  # y bounds
                (0.001, min(width, height)/2 - 0.001)  # r bounds
            ])
        
        constraints = [
            {'type': 'ineq', 'fun': constraint_containment},
            {'type': 'ineq', 'fun': constraint_non_overlap}
        ]
        
        # Try multiple local optimization methods for robustness
        # Method 1: SLSQP
        try:
            result = minimize(
                objective_local,
                refined_params,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-5}
            )
            
            if result.success:
                final_params = result.x
                circles = np.reshape(final_params, (n, 3))
            else:
                # If SLSQP fails, fall back to L-BFGS-B
                result = minimize(
                    objective_local,
                    refined_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                
                if result.success:
                    final_params = result.x
                    circles = np.reshape(final_params, (n, 3))
        except:
            pass  # Continue with current circles if optimization fails
    
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
                x = 0.01 + j * spacing + (i % 2) * spacing / 2
                y = 0.01 + i * hex_radius
                # Make sure we stay within bounds
                if x <= width - 0.01 and y <= height - 0.01:
                    circles[idx] = [x, y, avg_radius * 0.85]
                    idx += 1
                else:
                    circles[idx] = [width/2, height/2, avg_radius * 0.5]
                    idx += 1
                if idx >= n:
                    break
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
