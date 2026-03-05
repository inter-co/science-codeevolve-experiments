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
from scipy.spatial import cKDTree

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
    ratios = [0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]
    
    # Use fixed seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    for ratio in ratios:
        width = 1.0
        height = 1.0 / ratio if ratio > 1 else ratio
        
        # Multi-scale approach: start with better initialization
        circles = initialize_better(width, height, 21)
        
        # Apply improved evolutionary optimization for global optimization
        circles = evolutionary_optimization_improved(circles, width, height, generations=150, population_size=80)
        
        # Refine with local optimization
        circles = refine_circles(circles, width, height)
        
        # Calculate sum of radii
        total_radius = compute_total_radius_fast(circles)
        
        if total_radius > best_sum:
            best_sum = total_radius
            best_result = circles.copy()
    
    return best_result if best_result is not None else generate_default_solution(1.0, 1.0, 21)

def initialize_better(width: float, height: float, n: int) -> np.ndarray:
    """Better initialization using hexagonal packing and smart distribution"""
    circles = np.zeros((n, 3))
    
    # Strategy: Use a combination of hexagonal packing and strategic placement
    if n <= 12:
        # For small numbers, use hexagonal packing pattern
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Calculate spacing for hexagonal packing
        max_radius_guess = min(width, height) * 0.12
        
        # Place circles in a hexagonal pattern
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                    
                # Hexagonal packing offset
                offset = (i % 2) * (max_radius_guess * 1.5)  # Horizontal offset for even rows
                x = (j * 2 + 1) * max_radius_guess + offset
                y = (i * 1.732 + 1) * max_radius_guess  # sqrt(3) ≈ 1.732
                
                # Ensure within bounds
                x = max(max_radius_guess, min(width - max_radius_guess, x))
                y = max(max_radius_guess, min(height - max_radius_guess, y))
                
                # Add slight randomization to avoid perfect patterns
                x += np.random.uniform(-max_radius_guess*0.15, max_radius_guess*0.15)
                y += np.random.uniform(-max_radius_guess*0.15, max_radius_guess*0.15)
                
                # Ensure still within bounds after randomization
                x = max(max_radius_guess, min(width - max_radius_guess, x))
                y = max(max_radius_guess, min(height - max_radius_guess, y))
                
                # Set radius with better scaling
                max_radius = min(x, width - x, y, height - y)
                radius = max_radius * 0.7  # Slightly larger radius for better packing
                
                circles[count] = [x, y, radius]
                count += 1
                
            if count >= n:
                break
    elif n <= 21:
        # For medium numbers, use a more sophisticated approach
        # Use a grid with irregular spacing to get better packing
        
        # Try to distribute circles more intelligently
        # Start with a hexagonal-like pattern
        rows = 4
        cols = 6
        
        # Calculate spacing based on container size
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)
        
        # Adjust spacing to be slightly smaller for better packing
        spacing_x *= 0.85
        spacing_y *= 0.85
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                x = spacing_x * (j + 1)
                y = spacing_y * (i + 1)
                
                # Add randomness to positions with more controlled variance
                x += np.random.uniform(-spacing_x/6, spacing_x/6)
                y += np.random.uniform(-spacing_y/6, spacing_y/6)
                
                # Ensure within bounds
                x = max(0.01, min(width - 0.01, x))
                y = max(0.01, min(height - 0.01, y))
                
                # Initial radius based on available space
                max_radius = min(x, width - x, y, height - y)
                radius = max_radius * 0.5
                
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

def evolutionary_optimization_improved(circles: np.ndarray, width: float, height: float, generations: int = 150, population_size: int = 80) -> np.ndarray:
    """Improved evolutionary algorithm with better selection and crossover"""
    # Create initial population
    population = []
    for _ in range(population_size):
        individual = circles.copy()
        # Add noise to create diversity
        for i in range(len(individual)):
            # Add more significant variation for first few generations
            scale_factor = 0.03
            individual[i, 0] += np.random.normal(0, scale_factor * width)
            individual[i, 1] += np.random.normal(0, scale_factor * height)
            individual[i, 2] += np.random.normal(0, scale_factor * 0.1)
            
            # Keep within bounds
            individual[i, 0] = max(individual[i, 2], min(width - individual[i, 2], individual[i, 0]))
            individual[i, 1] = max(individual[i, 2], min(height - individual[i, 2], individual[i, 1]))
            individual[i, 2] = max(0.001, individual[i, 2])
        
        population.append(individual)
    
    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness for each individual
        fitness_scores = []
        for individual in population:
            if check_all_constraints(individual, width, height):
                fitness = compute_total_radius_fast(individual)
                fitness_scores.append((individual, fitness))
            else:
                # Penalize infeasible solutions heavily
                fitness_scores.append((individual, -1000))
        
        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top performers with adaptive selection
        top_performers_count = max(15, population_size // 4)
        top_performers = [ind for ind, fit in fitness_scores[:top_performers_count]]
        
        # Create new population through better crossover and mutation
        new_population = top_performers[:]
        
        # Elitism: keep best solution
        best_individual = fitness_scores[0][0]
        new_population.append(best_individual)
        
        # Generate offspring with better crossover strategy
        while len(new_population) < population_size:
            # Tournament selection with larger tournament size for better exploration
            tournament_size = 6
            parent1 = tournament_selection_with_size(fitness_scores, tournament_size)
            parent2 = tournament_selection_with_size(fitness_scores, tournament_size)
            
            # Use uniform crossover with higher probability for better mixing
            child = crossover_uniform(parent1, parent2)
            
            # Mutation with adaptive rate
            child = mutate_adaptive(child, width, height, generation, generations)
            
            new_population.append(child)
        
        population = new_population[:population_size]
    
    # Return best solution from final population
    final_fitness = [(ind, compute_total_radius_fast(ind)) for ind in population if check_all_constraints(ind, width, height)]
    if final_fitness:
        best_final = max(final_fitness, key=lambda x: x[1])
        return best_final[0]
    else:
        return population[0] if population else circles

def tournament_selection_with_size(fitness_scores, tournament_size: int = 6) -> np.ndarray:
    """Select an individual using tournament selection with specified size"""
    tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
    return max(tournament, key=lambda x: x[1])[0]

def crossover_uniform(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform uniform crossover between two parents"""
    child = parent1.copy()
    
    # Uniform crossover with higher probability for better mixing
    for i in range(len(parent1)):
        if random.random() < 0.7:  # Higher crossover probability
            # Blend positions and radii with weighted average
            alpha = random.random()
            child[i, 0] = alpha * parent1[i, 0] + (1 - alpha) * parent2[i, 0]
            child[i, 1] = alpha * parent1[i, 1] + (1 - alpha) * parent2[i, 1]
            child[i, 2] = alpha * parent1[i, 2] + (1 - alpha) * parent2[i, 2]
    
    return child

def mutate_adaptive(individual: np.ndarray, width: float, height: float, generation: int, max_generations: int) -> np.ndarray:
    """Mutate an individual with adaptive mutation rate"""
    mutated = individual.copy()
    
    # Mutation rate decreases over time but starts higher
    # Use a more aggressive initial mutation rate
    mutation_rate = 0.6 * (1 - generation / max_generations) + 0.15
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate position with higher variance initially
            position_variance = 0.04 * width if generation < max_generations/2 else 0.02 * width
            radius_variance = 0.03
            
            mutated[i, 0] += np.random.normal(0, position_variance)
            mutated[i, 1] += np.random.normal(0, position_variance)
            mutated[i, 2] += np.random.normal(0, radius_variance)
            
            # Keep within bounds
            mutated[i, 0] = max(mutated[i, 2], min(width - mutated[i, 2], mutated[i, 0]))
            mutated[i, 1] = max(mutated[i, 2], min(height - mutated[i, 2], mutated[i, 1]))
            mutated[i, 2] = max(0.001, mutated[i, 2])
    
    return mutated

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
        
        # Check overlaps using vectorized approach with early termination
        for i in range(n):
            for j in range(i+1, n):
                if check_overlap_fast(coords[i], coords[j], radii[i], radii[j]):
                    return False
    
    return True

def optimize_circle_positions(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize circle positions using constrained optimization with better approach"""
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
        result = minimize(objective, initial_params, method='SLSQP', constraints=cons, 
                         options={'maxiter': 500, 'ftol': 1e-7, 'eps': 1e-5})
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
    for _ in range(50):  # More refinement passes for better results
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
    max_iterations = 150
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
            push_amount = (sum_radii - distance) * 0.8  # More aggressive pushing
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
            radius = min(spacing_x, spacing_y) / 3
            
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
