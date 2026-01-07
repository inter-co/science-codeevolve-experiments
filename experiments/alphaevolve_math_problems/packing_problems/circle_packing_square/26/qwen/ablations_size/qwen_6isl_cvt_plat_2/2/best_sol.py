# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import time

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

# Global constants
TIME_LIMIT = 175  # seconds
MAX_LOCAL_ITERATIONS = 600  # Increased iterations for better convergence

def calculate_fitness(circles: np.ndarray) -> float:
    """Calculate fitness based on sum of radii with penalty for constraint violations."""
    n = len(circles)
    
    # Calculate sum of radii
    total_radius = np.sum(circles[:, 2])
    
    # Penalty for constraint violations
    penalty = 0.0
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            penalty += 1000000000.0  # Increased penalty
    
    # Check overlap constraints efficiently using KDTree
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Build KDTree for fast neighbor lookup
    tree = KDTree(positions)
    
    # Find neighbors within 2*(max_radius) distance to check for overlaps
    for i in range(n):
        x, y, r = circles[i]
        # Query nearby points - only look at close neighbors
        indices = tree.query_ball_point([x, y], 2 * r)
        for j in indices:
            if i != j:
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < r1 + r2:
                    overlap = (r1 + r2) - distance
                    penalty += 1000000000.0 * overlap**4  # Even stronger penalty
    
    return total_radius - penalty

def initialize_hexagonal_config(n: int) -> np.ndarray:
    """Initialize circles using hexagonal packing pattern for better distribution."""
    # Create a hexagonal grid pattern for good initial coverage
    rows = 5
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            
            # Adjust for hexagonal pattern
            if i % 2 == 1:
                x += spacing_x / 2
                
            # Ensure within bounds and set appropriate radius
            max_radius = min(x, 1-x, y, 1-y)
            r = min(max_radius * 0.88, 0.13)  # Slightly larger initial radii
            
            circles.append([x, y, r])
    
    # Fill remaining positions if needed with random valid positions
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        max_radius = min(x, 1-x, y, 1-y)
        r = min(max_radius * 0.58, 0.13)  # Slightly larger initial radii
        circles.append([x, y, r])
        
    return np.array(circles[:n])

def local_optimization(circles: np.ndarray) -> np.ndarray:
    """Apply local optimization using scipy minimize to refine the solution."""
    n = len(circles)
    
    def objective(params):
        # Reshape parameters back to circles
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Apply bounds to radii
        radii = np.maximum(0.001, np.minimum(0.5, radii))
        
        # Calculate sum of radii (negative because we want to maximize)
        return -np.sum(radii)
    
    def constraint_non_overlap(params):
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Calculate distances efficiently using cdist
        distances = cdist(positions, positions)
        constraints = []
        
        # Only check pairs that might overlap (distance < sum of radii)
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                # Constraint violation: distance should be >= min_dist
                constraints.append(dist - min_dist)
        
        return np.array(constraints)
    
    def constraint_containment(params):
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            
            # Left boundary: x - r >= 0
            constraints.append(x - r)
            # Right boundary: 1 - x - r >= 0  
            constraints.append(1 - x - r)
            # Bottom boundary: y - r >= 0
            constraints.append(y - r)
            # Top boundary: 1 - y - r >= 0
            constraints.append(1 - y - r)
        
        return np.array(constraints)
    
    # Flatten initial circles for optimization
    initial_positions = circles[:, :2].flatten()
    initial_radii = circles[:, 2]
    initial_params = np.concatenate([initial_positions, initial_radii])
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_non_overlap},
        {'type': 'ineq', 'fun': constraint_containment}
    ]
    
    # Bounds for positions (0,1) and radii (0.001, 0.5)
    bounds = [(0, 1) for _ in range(len(initial_params))]
    for i in range(len(initial_params) - n, len(initial_params)):
        bounds[i] = (0.001, 0.5)
    
    try:
        # Try different optimization methods for robustness
        methods = ['SLSQP', 'L-BFGS-B', 'TNC']
        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': MAX_LOCAL_ITERATIONS, 'ftol': 1e-10, 'gtol': 1e-10}
                )
                if result.success:
                    final_positions = result.x[:-n].reshape(-1, 2)
                    final_radii = result.x[-n:]
                    
                    # Create final circle array
                    circles = np.column_stack([final_positions, final_radii])
                    return circles
            except:
                continue
    except:
        pass
    
    return circles

def mutate_individual(circles: np.ndarray, mutation_rate: float = 0.20) -> np.ndarray:
    """Mutate a single individual by perturbing circle positions and radii."""
    mutated = circles.copy()
    n = len(mutated)
    
    for i in range(n):
        if random.random() < mutation_rate:
            # Randomly decide what to mutate
            action = random.choice(['position', 'radius'])
            
            if action == 'position':
                # Mutate position with larger steps for exploration
                mutated[i, 0] = max(0.01, min(0.99, mutated[i, 0] + np.random.normal(0, 0.08)))
                mutated[i, 1] = max(0.01, min(0.99, mutated[i, 1] + np.random.normal(0, 0.08)))
            else:
                # Mutate radius with larger step for exploration
                mutated[i, 2] = max(0.001, min(0.4, mutated[i, 2] + np.random.normal(0, 0.05)))
    
    return mutated

def repair_circles(circles: np.ndarray) -> np.ndarray:
    """Repair circles to satisfy boundary constraints."""
    repaired = circles.copy()
    n = len(repaired)
    
    for i in range(n):
        x, y, r = repaired[i]
        
        # Ensure circle fits within boundaries
        r = min(r, x, 1-x, y, 1-y)
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        
        repaired[i] = [x, y, r]
    
    return repaired

def optimize_with_hybrid_approach(circles: np.ndarray, start_time: float) -> np.ndarray:
    """Use hybrid approach: local optimization + evolutionary refinement."""
    current_solution = circles.copy()
    best_solution = circles.copy()
    best_fitness = calculate_fitness(best_solution)
    
    # First apply local optimization to get a good baseline
    current_solution = local_optimization(current_solution)
    current_fitness = calculate_fitness(current_solution)
    
    if current_fitness > best_fitness:
        best_solution = current_solution.copy()
        best_fitness = current_fitness
    
    # Use a more sophisticated evolutionary algorithm with better parameters
    population_size = 40  # Increased population size
    max_generations = 200  # More generations
    
    # Generate initial population
    population = [current_solution.copy()]
    for _ in range(population_size - 1):
        individual = mutate_individual(current_solution, 0.40)  # Higher mutation rate
        individual = repair_circles(individual)
        population.append(individual)
    
    generation = 0
    while generation < max_generations and time.time() - start_time < TIME_LIMIT:
        # Evaluate fitness of entire population
        fitness_scores = []
        for individual in population:
            fitness = calculate_fitness(individual)
            fitness_scores.append(fitness)
        
        # Update best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = population[max_fitness_idx].copy()
        
        # Selection and reproduction (tournament selection with adaptive tournament size)
        new_population = []
        
        # Elitism: keep best individual
        new_population.append(best_solution.copy())
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection with adaptive size
            tournament_size = min(7, population_size // 3)  # Larger tournament size
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            # Select another parent
            tournament_indices2 = random.sample(range(population_size), tournament_size)
            tournament_fitness2 = [fitness_scores[i] for i in tournament_indices2]
            winner_idx2 = tournament_indices2[np.argmax(tournament_fitness2)]
            
            # Crossover (uniform with higher probability for better parents)
            child = population[winner_idx].copy()
            for i in range(len(child)):
                if random.random() < 0.60:  # Slight bias towards better parent
                    child[i] = population[winner_idx2][i]
            
            # Mutation with adaptive rate
            mutation_rate = 0.30 if generation < max_generations // 4 else 0.25
            child = mutate_individual(child, mutation_rate)
            
            # Repair
            child = repair_circles(child)
            
            new_population.append(child)
        
        # Apply local optimization to top individuals with increased frequency
        top_count = min(12, len(new_population) // 2)
        for i in range(top_count):
            new_population[i] = local_optimization(new_population[i])
        
        population = new_population
        generation += 1
    
    return best_solution

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses hexagonal initialization followed by hybrid optimization (local + evolutionary).

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Initialize with hexagonal pattern for better starting configuration
    circles = initialize_hexagonal_config(26)
    
    # Optimize using hybrid approach
    optimized_circles = optimize_with_hybrid_approach(circles, start_time)
    
    # Final refinement with focused optimization
    try:
        final_refinement = local_optimization(optimized_circles)
        final_fitness = calculate_fitness(final_refinement)
        original_fitness = calculate_fitness(optimized_circles)
        
        if final_fitness > original_fitness:
            optimized_circles = final_refinement
    except:
        pass
    
    return optimized_circles


# EVOLVE-BLOCK-END
