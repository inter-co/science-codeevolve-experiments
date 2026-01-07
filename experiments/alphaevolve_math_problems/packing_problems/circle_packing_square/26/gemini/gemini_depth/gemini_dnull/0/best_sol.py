# EVOLVE-BLOCK-START
import numpy as np
import time

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

N_CIRCLES = 26
UNIT_SQUARE_SIZE = 1.0
MIN_RADIUS = 1e-6 # To avoid zero or negative radii
MAX_RADIUS_INIT = 0.12 # Max radius for initial population circles, slightly more generous to allow for a better constructive start
MAX_RADIUS_GLOBAL = 0.25 # Max theoretical radius for any circle in this context (for 26 circles, a single one won't be 0.5)

def calculate_overlap_penalty(circles: np.ndarray) -> float:
    """
    Calculates the penalty for overlapping circles.
    The penalty is the sum of (overlap_depth)^2 for all overlapping pairs.
    """
    penalty = 0.0
    n = circles.shape[0]
    
    centers = circles[:, :2] # (n, 2)
    radii = circles[:, 2]    # (n,)

    # Calculate squared distances between all pairs of centers
    # This uses broadcasting: (n, 1, 2) - (1, n, 2) -> (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_distances = np.sum(diff**2, axis=-1)
    
    # Calculate sum of radii for all pairs
    radii_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]

    # Get upper triangle indices (excluding diagonal for self-comparison)
    upper_triangle_indices = np.triu_indices(n, k=1)
    
    sq_distances_pairs = sq_distances[upper_triangle_indices]
    radii_sum_pairs = radii_sum_matrix[upper_triangle_indices]

    # Overlap condition: (r_i + r_j)^2 > d_ij^2
    # The penalty is `(r_i + r_j)^2 - d_ij^2`. If this is positive, there's overlap.
    overlap_squared_depth = (radii_sum_pairs**2) - sq_distances_pairs
    
    # Only positive values contribute to penalty
    penalty = np.sum(np.maximum(0, overlap_squared_depth))
    
    return penalty

def calculate_containment_penalty(circles: np.ndarray) -> float:
    """
    Calculates the penalty for circles exceeding the unit square boundaries.
    The penalty is the sum of (boundary_violation_depth)^2.
    """
    penalty = 0.0
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # Check x-axis boundaries: x < r or x > 1-r
    penalty += np.sum(np.maximum(0, r - x)**2) # Violation on left boundary
    penalty += np.sum(np.maximum(0, x + r - UNIT_SQUARE_SIZE)**2) # Violation on right boundary

    # Check y-axis boundaries: y < r or y > 1-r
    penalty += np.sum(np.maximum(0, r - y)**2) # Violation on bottom boundary
    penalty += np.sum(np.maximum(0, y + r - UNIT_SQUARE_SIZE)**2) # Violation on top boundary
    
    # Check for negative or too small radii
    penalty += np.sum(np.maximum(0, MIN_RADIUS - r)**2) 

    return penalty

def calculate_fitness(circles: np.ndarray, overlap_penalty_factor: float, containment_penalty_factor: float) -> float:
    """
    Evaluates the fitness of a configuration of circles.
    Objective: maximize sum of radii. Penalize for constraint violations.
    Higher values are better.
    """
    sum_radii = np.sum(circles[:, 2])

    overlap_pen = calculate_overlap_penalty(circles)
    contain_pen = calculate_containment_penalty(circles)

    # Penalize heavily for violations. The penalty factors should be large enough
    # to push solutions towards feasibility. Using a quadratic penalty helps
    # gradient-free optimizers converge more smoothly towards feasibility.
    fitness = sum_radii - overlap_penalty_factor * overlap_pen - containment_penalty_factor * contain_pen
    
    return fitness

# --- Genetic Algorithm Components ---

def initialize_population(pop_size: int, n_circles: int, min_r: float, max_r_init: float) -> np.ndarray:
    """
    Initializes a population of circle configurations, with a portion generated constructively.
    Circles are placed such that they are contained within the square initially.
    """
    population = np.zeros((pop_size, n_circles, 3))
    
    # Generate a portion of the population using a constructive grid heuristic
    num_constructive = pop_size // 4 # e.g., 25% of population
    if num_constructive == 0 and pop_size > 0: num_constructive = 1 # ensure at least one constructive
    
    if num_constructive > 0:
        grid_side_count = 5 
        # Calculate base radius for a 5x5 grid of tangent circles
        r_grid_init = UNIT_SQUARE_SIZE / (2.0 * grid_side_count) # 0.1 for 5x5
        
        # Create a base constructive configuration for N_CIRCLES
        base_constructive_circles = np.zeros((n_circles, 3))
        
        k = 0
        # Place 25 circles in a 5x5 grid
        for i in range(grid_side_count):
            for j in range(grid_side_count):
                if k < n_circles: # Ensure we don't exceed N_CIRCLES
                    x = r_grid_init + i * 2 * r_grid_init
                    y = r_grid_init + j * 2 * r_grid_init
                    base_constructive_circles[k] = [x, y, r_grid_init]
                    k += 1
        
        # For the remaining circles (for N=26, this is 1 circle), place them centrally or randomly
        for rc_idx in range(k, n_circles):
            extra_r_init = np.random.uniform(min_r, max_r_init * 0.8) # Smaller initial radius for extra circles
            extra_x = UNIT_SQUARE_SIZE / 2.0 + np.random.uniform(-0.05, 0.05)
            extra_y = UNIT_SQUARE_SIZE / 2.0 + np.random.uniform(-0.05, 0.05)
            base_constructive_circles[rc_idx] = [extra_x, extra_y, extra_r_init]

        # Apply the constructive pattern to `num_constructive` individuals
        for idx_constructive in range(num_constructive):
            temp_circles = base_constructive_circles.copy()
            
            # Apply slight random perturbation to constructive placements
            # The magnitude of perturbation can vary slightly per individual
            perturb_factor = 0.05 + idx_constructive * (0.1 / num_constructive) # Vary perturbation strength
            temp_circles[:, :2] += np.random.normal(0, r_grid_init * perturb_factor * UNIT_SQUARE_SIZE, (n_circles, 2))
            temp_circles[:, 2] += np.random.normal(0, r_grid_init * perturb_factor * MAX_RADIUS_GLOBAL, n_circles)
            
            # Ensure radii are within bounds
            temp_circles[:, 2] = np.clip(temp_circles[:, 2], min_r, max_r_init)
            
            # Ensure positions are contained within the square based on their (possibly new) radii
            for c_idx in range(n_circles):
                r_val = temp_circles[c_idx, 2]
                temp_circles[c_idx, 0] = np.clip(temp_circles[c_idx, 0], r_val, UNIT_SQUARE_SIZE - r_val)
                temp_circles[c_idx, 1] = np.clip(temp_circles[c_idx, 1], r_val, UNIT_SQUARE_SIZE - r_val)
            
            population[idx_constructive] = temp_circles

    # Fill the rest of the population with purely random configurations
    for i in range(num_constructive, pop_size):
        radii = np.random.uniform(min_r, max_r_init, n_circles)
        x_coords = np.random.uniform(radii, UNIT_SQUARE_SIZE - radii)
        y_coords = np.random.uniform(radii, UNIT_SQUARE_SIZE - radii)
        population[i, :, 0] = x_coords
        population[i, :, 1] = y_coords
        population[i, :, 2] = radii
        
    return population

def local_jiggle_search(individual: np.ndarray, iterations: int, jiggle_strength_pos: float, jiggle_strength_r: float, 
                        overlap_penalty_factor: float, containment_penalty_factor: float) -> np.ndarray:
    """
    Applies a local random walk (jiggle) search to an individual to improve fitness or resolve violations.
    Perturbs positions and radii, accepting changes that improve fitness.
    """
    best_local_individual = individual.copy()
    best_local_fitness = calculate_fitness(individual, overlap_penalty_factor, containment_penalty_factor)

    for _ in range(iterations):
        temp_individual = best_local_individual.copy()
        
        # Jiggle positions
        pos_mut = np.random.normal(0, jiggle_strength_pos * UNIT_SQUARE_SIZE, (N_CIRCLES, 2))
        temp_individual[:, :2] += pos_mut
        
        # Jiggle radii
        r_mut = np.random.normal(0, jiggle_strength_r * MAX_RADIUS_GLOBAL, N_CIRCLES)
        temp_individual[:, 2] += r_mut
        
        # Re-clip all values to ensure basic bounds (without considering containment yet)
        temp_individual[:, 2] = np.clip(temp_individual[:, 2], MIN_RADIUS, MAX_RADIUS_GLOBAL)
        
        # Re-clip x,y based on new radii to ensure containment
        for c_idx in range(N_CIRCLES):
            r_val = temp_individual[c_idx, 2]
            temp_individual[c_idx, 0] = np.clip(temp_individual[c_idx, 0], r_val, UNIT_SQUARE_SIZE - r_val)
            temp_individual[c_idx, 1] = np.clip(temp_individual[c_idx, 1], r_val, UNIT_SQUARE_SIZE - r_val)

        current_local_fitness = calculate_fitness(temp_individual, overlap_penalty_factor, containment_penalty_factor)
        
        # Accept if strictly better, or if it resolves violations significantly (even if sum_radii is slightly lower)
        if current_local_fitness > best_local_fitness:
            best_local_fitness = current_local_fitness
            best_local_individual = temp_individual
            
    return best_local_individual

def select_parents_tournament(population: np.ndarray, fitnesses: np.ndarray, num_parents: int, tournament_size: int) -> np.ndarray:
    """Selects parents using tournament selection."""
    parents = np.zeros((num_parents, population.shape[1], population.shape[2]))
    pop_len = len(population)
    
    for i in range(num_parents):
        contenders_indices = np.random.choice(pop_len, tournament_size, replace=False)
        contenders_fitness = fitnesses[contenders_indices]
        winner_index_in_contenders = np.argmax(contenders_fitness)
        winner_index = contenders_indices[winner_index_in_contenders]
        parents[i] = population[winner_index]
    return parents

def crossover_uniform(parent1: np.ndarray, parent2: np.ndarray, crossover_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs uniform crossover on two parent circle configurations.
    Each circle (x,y,r) is taken from either parent with 50% probability.
    Returns two children.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    if np.random.rand() < crossover_rate:
        # For each circle, decide which parent it comes from
        mask = np.random.rand(N_CIRCLES) < 0.5
        child1[mask] = parent2[mask]
        child2[mask] = parent1[mask]
    
    return child1, child2

def mutate_gaussian(individual: np.ndarray, mutation_rate: float, mutation_strength: float, min_r: float, max_r_global: float) -> np.ndarray:
    """
    Applies Gaussian mutation to an individual circle configuration.
    Mutates x, y, r for selected circles, ensuring constraints are met.
    """
    mutated_individual = individual.copy()
    
    # Identify circles to mutate
    mutate_mask = np.random.rand(N_CIRCLES) < mutation_rate
    
    if np.any(mutate_mask):
        mutated_circles = mutated_individual[mutate_mask]
        
        # Mutate radii
        current_radii = mutated_circles[:, 2]
        # Use mutation_strength directly as std dev for radii changes
        r_mutation_amounts = np.random.normal(0, mutation_strength * max_r_global, mutated_circles.shape[0])
        new_radii = current_radii + r_mutation_amounts
        new_radii = np.clip(new_radii, min_r, max_r_global)
        mutated_circles[:, 2] = new_radii

        # Mutate x, y positions
        current_x = mutated_circles[:, 0]
        current_y = mutated_circles[:, 1]
        
        # Use mutation_strength directly as std dev for position changes
        x_mutation_amounts = np.random.normal(0, mutation_strength * UNIT_SQUARE_SIZE, mutated_circles.shape[0])
        y_mutation_amounts = np.random.normal(0, mutation_strength * UNIT_SQUARE_SIZE, mutated_circles.shape[0])

        new_x = current_x + x_mutation_amounts
        new_y = current_y + y_mutation_amounts
        
        # Bounds for x and y depend on the new_r
        x_lower_bound = new_radii
        x_upper_bound = UNIT_SQUARE_SIZE - new_radii
        y_lower_bound = new_radii
        y_upper_bound = UNIT_SQUARE_SIZE - new_radii

        mutated_circles[:, 0] = np.clip(new_x, x_lower_bound, x_upper_bound)
        mutated_circles[:, 1] = np.clip(new_y, y_lower_bound, y_upper_bound)

        mutated_individual[mutate_mask] = mutated_circles
            
    return mutated_individual

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a Genetic Algorithm approach.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42) # For reproducibility

    # Genetic Algorithm Parameters
    POPULATION_SIZE = 350 # Slightly increased population size for more diversity
    GENERATIONS = 1700 # Slightly reduced generations to keep total evals similar
    
    # Dynamic Penalty factors (exponential increase)
    INITIAL_PENALTY_FACTOR = 1000.0 # Start lower to allow exploration
    FINAL_PENALTY_FACTOR = 100000.0 # End higher to enforce strict feasibility
    
    CROSSOVER_RATE = 0.8
    
    # Dynamic Mutation parameters (linear decrease)
    INITIAL_MUTATION_RATE = 0.2 # Probability of a circle being mutated
    FINAL_MUTATION_RATE = 0.05 # Reduced mutation rate for fine-tuning
    
    INITIAL_MUTATION_STRENGTH = 0.05 # Controls the magnitude of mutation
    FINAL_MUTATION_STRENGTH = 0.005 # Reduced strength for fine-tuning

    start_time = time.time()

    # 1. Initialize Population
    population = initialize_population(POPULATION_SIZE, N_CIRCLES, MIN_RADIUS, MAX_RADIUS_INIT)
    
    best_individual = population[0].copy()
    best_fitness = -np.inf # Initialize with a very low fitness
    
    for generation in range(GENERATIONS):
        # Stop early if approaching time limit
        if time.time() - start_time > 170: # 170 seconds buffer
            break

        # Calculate adaptive parameters for current generation
        generation_ratio = generation / GENERATIONS
        
        # Exponential increase for penalty factors to enforce constraints more strongly later
        current_overlap_penalty_factor = INITIAL_PENALTY_FACTOR * (FINAL_PENALTY_FACTOR / INITIAL_PENALTY_FACTOR)**generation_ratio
        current_containment_penalty_factor = INITIAL_PENALTY_FACTOR * (FINAL_PENALTY_FACTOR / INITIAL_PENALTY_FACTOR)**generation_ratio
        
        # Linear decrease for mutation parameters
        current_mutation_rate = INITIAL_MUTATION_RATE + (FINAL_MUTATION_RATE - INITIAL_MUTATION_RATE) * generation_ratio
        current_mutation_strength = INITIAL_MUTATION_STRENGTH + (FINAL_MUTATION_STRENGTH - INITIAL_MUTATION_STRENGTH) * generation_ratio

        # 2. Evaluate Fitness
        fitnesses = np.array([calculate_fitness(ind, current_overlap_penalty_factor, current_containment_penalty_factor) for ind in population])

        current_best_idx = np.argmax(fitnesses)
        current_best_fitness = fitnesses[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best_idx].copy()


        # 3. Select Parents
        num_parents = POPULATION_SIZE // 2 # Select half the population as parents
        parents = select_parents_tournament(population, fitnesses, num_parents, tournament_size=5)

        # 4. Create Next Generation (Offspring)
        offspring = []
        # Create children by pairing parents
        for i in range(0, num_parents, 2):
            parent1 = parents[i]
            # Ensure we have a second parent for crossover, or just mutate the single parent
            if i + 1 < num_parents:
                parent2 = parents[i+1]
                child1, child2 = crossover_uniform(parent1, parent2, CROSSOVER_RATE)
                child1 = mutate_gaussian(child1, current_mutation_rate, current_mutation_strength, MIN_RADIUS, MAX_RADIUS_GLOBAL)
                child2 = mutate_gaussian(child2, current_mutation_rate, current_mutation_strength, MIN_RADIUS, MAX_RADIUS_GLOBAL)
                offspring.extend([child1, child2])
            else: # Odd number of parents, just mutate the last one
                child = mutate_gaussian(parent1.copy(), current_mutation_rate, current_mutation_strength, MIN_RADIUS, MAX_RADIUS_GLOBAL)
                offspring.append(child)

        # 5. Form new population using elitism and offspring
        next_population_list = [best_individual.copy()] # Elitism: carry over the absolute best
        
        # Evaluate offspring fitness to select the best ones using current penalties
        offspring_fitnesses = np.array([calculate_fitness(ind, current_overlap_penalty_factor, current_containment_penalty_factor) for ind in offspring])
        
        # Sort offspring by fitness and take the top `POPULATION_SIZE - 1` to fill the new population
        sorted_offspring_indices = np.argsort(offspring_fitnesses)[::-1] # Sort in descending order
        
        # Take min(len(offspring), POPULATION_SIZE - 1) best offspring
        num_offspring_to_take = min(len(offspring), POPULATION_SIZE - 1)
        for idx in sorted_offspring_indices[:num_offspring_to_take]:
            next_population_list.append(offspring[idx])
            
        # If still not enough individuals for the next population, pad with copies of the best.
        while len(next_population_list) < POPULATION_SIZE:
             next_population_list.append(best_individual.copy())

        population = np.array(next_population_list)

    # Final verification/adjustment:
    # Apply a local jiggle search to the best individual to fine-tune and resolve any remaining violations.
    # Use even higher penalty factors for local search to strictly enforce feasibility.
    final_overlap_pen_factor = FINAL_PENALTY_FACTOR * 10 
    final_contain_pen_factor = FINAL_PENALTY_FACTOR * 10
    
    best_individual = local_jiggle_search(best_individual, 
                                          iterations=750, # Increased iterations for final fine-tuning
                                          jiggle_strength_pos=0.002, 
                                          jiggle_strength_r=0.001,
                                          overlap_penalty_factor=final_overlap_pen_factor, 
                                          containment_penalty_factor=final_contain_pen_factor)

    # One last check and shrink radii if still not feasible after local search
    final_overlap_pen = calculate_overlap_penalty(best_individual)
    final_contain_pen = calculate_containment_penalty(best_individual)
    
    if final_overlap_pen > 1e-6 or final_contain_pen > 1e-6: # If there's any significant violation
        temp_individual = best_individual.copy()
        shrink_factor = 0.99 # More aggressive reduction if local search failed
        max_repair_iterations = 20 # Limit iterations to save time
        
        for _ in range(max_repair_iterations):
            current_overlap_pen = calculate_overlap_penalty(temp_individual)
            current_contain_pen = calculate_containment_penalty(temp_individual)
            
            if current_overlap_pen < 1e-6 and current_contain_pen < 1e-6:
                break # Feasible
            
            # Reduce all radii
            temp_individual[:, 2] *= shrink_factor
            # Ensure radii don't go below MIN_RADIUS
            temp_individual[:, 2] = np.maximum(temp_individual[:, 2], MIN_RADIUS)
            
            # Re-clip positions based on new (smaller) radii to ensure containment
            for c_idx in range(N_CIRCLES):
                r_val = temp_individual[c_idx, 2]
                x_val = temp_individual[c_idx, 0]
                y_val = temp_individual[c_idx, 1]
                temp_individual[c_idx, 0] = np.clip(x_val, r_val, UNIT_SQUARE_SIZE - r_val)
                temp_individual[c_idx, 1] = np.clip(y_val, r_val, UNIT_SQUARE_SIZE - r_val)
        
        # If repair was successful, use the repaired one
        if calculate_overlap_penalty(temp_individual) < 1e-6 and calculate_containment_penalty(temp_individual) < 1e-6:
            best_individual = temp_individual

    return best_individual


# EVOLVE-BLOCK-END
