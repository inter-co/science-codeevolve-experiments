# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import distance
from scipy.optimize import minimize
import random
from typing import Tuple
import time
from itertools import combinations
import math
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist
import numba
from numba import jit
from deap import base, creator, tools, algorithms
import copy
from scipy.optimize import differential_evolution
import heapq

@jit(nopython=True)
def compute_distance_squared_numba(p1, p2):
    """Fast squared distance computation for numba"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + dy*dy

@jit(nopython=True)
def check_overlap_fast(pos1, pos2, r1, r2):
    """Fast overlap checking for numba"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dist_sq = dx*dx + dy*dy
    return dist_sq < (r1 + r2)*(r1 + r2)

@jit(nopython=True)
def compute_total_radius_fast(circles):
    """Fast computation of total radius sum for numba"""
    total = 0.0
    for i in range(len(circles)):
        total += circles[i, 2]
    return total

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric initialization, evolutionary algorithm, and local optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Try different aspect ratios to find optimal configuration
    best_result = None
    best_sum = 0
    
    # Test several aspect ratios - focusing on more promising ones
    ratios = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0]
    
    # Use fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    for ratio in ratios:
        width = 1.0
        height = 1.0 / ratio if ratio > 1 else ratio
        
        # Multi-scale approach: start with better initialization
        circles = initialize_better(width, height, 21)
        
        # Apply improved evolutionary optimization for global optimization
        circles = evolutionary_optimization_improved(circles, width, height, generations=100, population_size=60)
        
        # Refine with local optimization
        circles = refine_circles(circles, width, height)
        
        # Calculate sum of radii
        total_radius = compute_total_radius_fast(circles)
        
        if total_radius > best_sum:
            best_sum = total_radius
            best_result = circles.copy()
    
    return best_result if best_result is not None else generate_default_solution(1.0, 1.0, 21)

def initialize_better(width: float, height: float, n: int) -> np.ndarray:
    """Better initialization using hexagonal packing idea and clustering"""
    circles = np.zeros((n, 3))
    
    # Strategy: Use more sophisticated hexagonal packing approach for 21 circles
    if n == 21:
        # Use a known good configuration inspired by hexagonal packing
        # Try to place in a pattern similar to 21 circles in hexagonal arrangement
        
        # Arrange in 5 rows (3, 4, 5, 4, 3) for a hexagonal-like pattern
        rows = [3, 4, 5, 4, 3]
        total_circles = sum(rows)
        
        # Determine appropriate spacing
        max_radius_guess = min(width, height) * 0.15  # Larger initial guess
        
        # Position circles in hexagonal pattern
        count = 0
        row_height = max_radius_guess * 1.732  # sqrt(3) * radius for vertical spacing
        row_width = max_radius_guess * 2.0      # horizontal spacing
        
        for i, row_size in enumerate(rows):
            # Offset for alternating rows
            offset = (i % 2) * (row_width / 2)
            
            for j in range(row_size):
                if count >= n:
                    break
                    
                x = offset + (j * row_width) + row_width / 2
                y = (i * row_height) + row_height / 2
                
                # Ensure within bounds
                x = max(max_radius_guess, min(width - max_radius_guess, x))
                y = max(max_radius_guess, min(height - max_radius_guess, y))
                
                # Add slight randomization to avoid perfect patterns
                x += np.random.uniform(-max_radius_guess*0.2, max_radius_guess*0.2)
                y += np.random.uniform(-max_radius_guess*0.2, max_radius_guess*0.2)
                
                # Ensure still within bounds after randomization
                x = max(max_radius_guess, min(width - max_radius_guess, x))
                y = max(max_radius_guess, min(height - max_radius_guess, y))
                
                # Set radius based on available space
                max_radius = min(x, width - x, y, height - y)
                radius = max_radius * 0.6  # Larger initial radius to start with
                
                circles[count] = [x, y, radius]
                count += 1
                
            if count >= n:
                break
    else:
        # For other sizes, use grid-based approach
        # Distribute circles more evenly using a grid with some randomness
        grid_rows = int(np.ceil(np.sqrt(n)))
        grid_cols = int(np.ceil(n / grid_rows))
        
        # Calculate spacing
        spacing_x = width / (grid_cols + 1)
        spacing_y = height / (grid_rows + 1)
        
        # Adjust spacing to be slightly smaller for better packing
        spacing_x *= 0.85
        spacing_y *= 0.85
        
        count = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                if count >= n:
                    break
                x = spacing_x * (j + 1)
                y = spacing_y * (i + 1)
                
                # Add randomness to positions
                x += np.random.uniform(-spacing_x/4, spacing_x/4)
                y += np.random.uniform(-spacing_y/4, spacing_y/4)
                
                # Ensure within bounds
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                
                # Initial radius based on available space
                max_radius = min(x, width - x, y, height - y)
                radius = max_radius * 0.35
                
                circles[count] = [x, y, radius]
                count += 1
                
            if count >= n:
                break
    
    # Improve by adjusting radii to reduce overlap potential
    if n > 10:
        # Make a second pass to adjust radii based on proximity to neighbors
        for i in range(n):
            x, y, r = circles[i]
            # Find nearest neighbors
            distances = []
            for j in range(n):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dx = x - x2
                    dy = y - y2
                    dist = np.sqrt(dx*dx + dy*dy)
                    distances.append((j, dist, r2))
            
            # Sort by distance
            distances.sort(key=lambda x: x[1])
            
            # Adjust radius to be smaller than minimum distance to neighbors minus some buffer
            if len(distances) > 0:
                min_dist = distances[0][1]
                if min_dist > 0.01:
                    # Allow for some space (buffer factor)
                    new_radius = min(r, min_dist * 0.45)
                    circles[i, 2] = max(0.001, new_radius)
    
    return circles

def evolutionary_optimization_improved(circles: np.ndarray, width: float, height: float, generations: int = 100, population_size: int = 60) -> np.ndarray:
    """Improved evolutionary algorithm with better selection and crossover"""
    # Use DEAP for better evolutionary optimization
    # Define fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Create individual from circles array
    def create_individual():
        individual = []
        for i in range(len(circles)):
            individual.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
        return creator.Individual(individual)
    
    # Evaluate individual fitness
    def evaluate(individual):
        # Convert back to circles format
        circles_array = np.array(individual).reshape(-1, 3)
        
        # Check constraints
        if not check_all_constraints(circles_array, width, height):
            return (-1000,)  # Penalize infeasible solutions heavily
        
        # Return sum of radii as fitness
        return (compute_total_radius_fast(circles_array),)
    
    # Mutate individual - more sophisticated mutation
    def mutate(individual):
        for i in range(len(individual)):
            if random.random() < 0.25:  # Lower mutation rate
                if i % 3 == 0:  # x coordinate
                    # Use adaptive mutation step size
                    mutation_step = width * 0.05
                    individual[i] += np.random.normal(0, mutation_step)
                    # Ensure within bounds
                    individual[i] = max(individual[i], individual[i+2])
                    individual[i] = min(individual[i], width - individual[i+2])
                elif i % 3 == 1:  # y coordinate
                    # Use adaptive mutation step size
                    mutation_step = height * 0.05
                    individual[i] += np.random.normal(0, mutation_step)
                    # Ensure within bounds
                    individual[i] = max(individual[i], individual[i-1])
                    individual[i] = min(individual[i], height - individual[i-1])
                else:  # radius
                    # Adaptive radius mutation
                    old_radius = individual[i]
                    individual[i] += np.random.normal(0, old_radius * 0.15)
                    individual[i] = max(0.001, individual[i])
        return individual,
    
    # Crossover function - more effective crossover
    def crossover(ind1, ind2):
        # Uniform crossover with higher probability for better mixing
        for i in range(len(ind1)):
            if random.random() < 0.6:  # Higher crossover rate
                ind1[i], ind2[i] = ind2[i], ind1[i]
        return ind1, ind2
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", crossover)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=4)  # Larger tournament size
    
    # Create initial population
    population = toolbox.population(n=population_size)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolution loop with early stopping
    best_fitness_history = []
    stagnation_counter = 0
    
    for generation in range(generations):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:  # Crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if random.random() < 0.25:  # Mutation probability
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population with offspring
        population[:] = offspring
        
        # Track best fitness
        current_best = max(population, key=lambda ind: ind.fitness.values[0])
        best_fitness_history.append(current_best.fitness.values[0])
        
        # Early stopping if no improvement
        if len(best_fitness_history) > 10:
            if abs(best_fitness_history[-1] - best_fitness_history[-10]) < 1e-6:
                stagnation_counter += 1
                if stagnation_counter > 5:
                    break
            else:
                stagnation_counter = 0
    
    # Return best solution
    best_ind = tools.selBest(population, 1)[0]
    best_circles = np.array(best_ind).reshape(-1, 3)
    return best_circles

def compute_total_radius(circles: np.ndarray) -> float:
    """Compute the sum of all radii"""
    return np.sum(circles[:, 2])

def check_all_constraints(circles: np.ndarray, width: float, height: float) -> bool:
    """Check if all constraints are satisfied"""
    n = len(circles)
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > width - r or y < r or y > height - r:
            return False
    
    # Check overlap constraints efficiently using fast numba version
    if n > 1:
        # Use faster numba-based checking
        coords = circles[:, :2]
        radii = circles[:, 2]
        
        # Check overlaps using vectorized approach
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap_fast(coords[i], coords[j], radii[i], radii[j]):
                    return False
    
    return True

def optimize_circle_positions(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize circle positions using constrained optimization"""
    n = len(circles)
    
    # Flatten parameters: [x0, y0, r0, x1, y1, r1, ...]
    initial_params = circles.flatten()
    
    def objective(params):
        # Reconstruct circles
        reconstructed = params.reshape(-1, 3)
        # Objective: maximize sum of radii (minimize negative sum)
        return -np.sum(reconstructed[:, 2])
    
    def constraint_func(params):
        # Reconstruct circles
        reconstructed = params.reshape(-1, 3)
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            x, y, r = reconstructed[i]
            # Ensure circles don't exceed boundaries
            constraints.extend([
                x - r,  # left boundary
                width - x - r,  # right boundary
                y - r,  # bottom boundary
                height - y - r  # top boundary
            ])
        
        # Non-overlap constraints
        coords = reconstructed[:, :2]
        radii = reconstructed[:, 2]
        
        # Use fast distance calculation
        for i in range(n):
            for j in range(i+1, n):
                # Constraint: dist^2 >= (r1 + r2)^2
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dist_sq = dx*dx + dy*dy
                sum_radii_sq = (radii[i] + radii[j])**2
                constraints.append(dist_sq - sum_radii_sq)
        
        return np.array(constraints)
    
    # Create constraints dictionary
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Optimize with better parameters
    try:
        # Use differential evolution for global optimization first
        bounds = [(0, width) if i % 3 == 0 else (0, height) if i % 3 == 1 else (0.001, width/2) for i in range(len(initial_params))]
        result = differential_evolution(objective, bounds, constraints=cons, maxiter=50, popsize=15, seed=42)
        if result.success:
            return result.x.reshape(-1, 3)
    except Exception as e:
        pass
    
    # Fallback to SLSQP if needed
    try:
        result = minimize(objective, initial_params, method='SLSQP', constraints=cons, 
                         options={'maxiter': 200, 'ftol': 1e-6, 'eps': 1e-4})
        if result.success:
            return result.x.reshape(-1, 3)
    except Exception as e:
        pass
    
    # Return original if optimization fails
    return circles

def refine_circles(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Refine circle configuration using multi-stage optimization"""
    refined = circles.copy()
    
    # Stage 1: Global optimization using constrained optimization
    refined = optimize_circle_positions(refined, width, height)
    
    # Stage 2: Local refinement with boundary-aware adjustments
    for _ in range(30):  # More iterations for better refinement
        # Adjust positions to avoid overlaps and respect boundaries
        for i in range(len(refined)):
            x, y, r = refined[i]
            
            # Keep within bounds
            x = max(r, min(width - r, x))
            y = max(r, min(height - r, y))
            
            # Adjust radius to maximize it while respecting constraints
            max_radius = min(x, width - x, y, height - y)
            
            # Check overlaps with other circles
            new_radius = max_radius
            for j in range(len(refined)):
                if i != j:
                    x2, y2, r2 = refined[j]
                    dx = x - x2
                    dy = y - y2
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 0:
                        # Maximum radius without overlapping this circle
                        max_radius_for_this = dist - r2
                        new_radius = min(new_radius, max_radius_for_this)
            
            # Ensure positive radius
            new_radius = max(0.001, new_radius)
            refined[i] = [x, y, new_radius]
    
    # Stage 3: Final validation and adjustment with better overlap resolution
    refined = validate_and_correct(refined, width, height)
    
    return refined

def validate_and_correct(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Ensure all constraints are satisfied with better overlap resolution"""
    corrected_circles = circles.copy()
    
    # First, handle boundary violations
    for i in range(len(corrected_circles)):
        x, y, r = corrected_circles[i]
        # Correct positions that violate boundaries
        corrected_circles[i, 0] = max(r, min(width - r, x))
        corrected_circles[i, 1] = max(r, min(height - r, y))
    
    # Then resolve overlaps through iterative correction with better strategy
    max_iterations = 100
    for iteration in range(max_iterations):
        overlaps = []
        for i in range(len(corrected_circles)):
            for j in range(i+1, len(corrected_circles)):
                x1, y1, r1 = corrected_circles[i]
                x2, y2, r2 = corrected_circles[j]
                
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < (r1 + r2):
                    overlaps.append((i, j, distance, r1 + r2))
        
        if not overlaps:
            break
            
        # Resolve the most severe overlap first
        overlaps.sort(key=lambda x: x[3] - x[2])  # Sort by overlap amount
        i, j, dist, sum_radii = overlaps[-1]
        
        # Push circles apart along the line connecting centers with better strategy
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > 0.001:  # Avoid division by zero
            # Push with more careful amount to prevent oscillation
            push_amount = (sum_radii - distance) * 0.7  # Slightly more aggressive
            dx_norm = dx / distance
            dy_norm = dy / distance
            
            # Move both circles away from each other
            corrected_circles[i, 0] -= dx_norm * push_amount
            corrected_circles[i, 1] -= dy_norm * push_amount
            corrected_circles[j, 0] += dx_norm * push_amount
            corrected_circles[j, 1] += dy_norm * push_amount
            
            # Keep within bounds
            corrected_circles[i, 0] = max(corrected_circles[i, 2], 
                                        min(width - corrected_circles[i, 2], 
                                            corrected_circles[i, 0]))
            corrected_circles[i, 1] = max(corrected_circles[i, 2], 
                                        min(height - corrected_circles[i, 2], 
                                            corrected_circles[i, 1]))
            corrected_circles[j, 0] = max(corrected_circles[j, 2], 
                                        min(width - corrected_circles[j, 2], 
                                            corrected_circles[j, 0]))
            corrected_circles[j, 1] = max(corrected_circles[j, 2], 
                                        min(height - corrected_circles[j, 2], 
                                            corrected_circles[j, 1]))
    
    return corrected_circles

def generate_default_solution(width: float, height: float, n: int) -> np.ndarray:
    """Fallback solution if optimization fails"""
    circles = np.zeros((n, 3))
    
    # Simple grid approach with better spacing
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing_x = width / (grid_size + 1)
    spacing_y = height / (grid_size + 1)
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n:
                break
            x = spacing_x * (i + 1)
            y = spacing_y * (j + 1)
            radius = min(spacing_x, spacing_y) / 2.5
            
            # Ensure it's within bounds
            x = max(radius, min(width - radius, x))
            y = max(radius, min(height - radius, y))
            
            circles[count] = [x, y, radius]
            count += 1
            
        if count >= n:
            break
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
