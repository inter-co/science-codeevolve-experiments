# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining constructive geometry, evolutionary algorithms, 
    and advanced optimization strategies inspired by successful approaches in the literature.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    np.random.seed(42)  # For reproducibility
    
    # Enhanced constructive approaches from successful inspirations
    def construct_algebraic_points():
        """Construct points using algebraic number theory for optimal angular distribution"""
        points = []
        # Use 16th roots of unity for excellent angular distribution
        for i in range(n):
            angle = 2 * np.pi * i / n
            # Slightly adjust to avoid degenerate cases
            radius = 0.4 + 0.05 * np.sin(3 * angle)  # Add slight variation
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def construct_hexagonal_grid():
        """Construct points in a hexagonal grid pattern"""
        points = []
        # Create a 4x4 grid with hexagonal offsets
        for i in range(4):
            for j in range(4):
                x = 0.1 + j * 0.225
                y = 0.1 + i * 0.225
                # Offset every other row for hexagonal packing
                if i % 2 == 1:
                    x += 0.1125
                points.append([x, y])
        
        return np.array(points[:n])  # Take only 16 points
    
    def construct_fibonacci_spiral():
        """Construct points using Fibonacci spiral with improved distribution"""
        points = []
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Distribute points more evenly using logarithmic spiral
        for i in range(n):
            # Use a more controlled spiral approach
            angle = 2 * np.pi * i / phi
            # Non-linear radial distribution to avoid clustering
            r = 0.4 * np.sqrt(i / (n - 1)) if n > 1 else 0
            # Add some randomness to break symmetry
            r += 0.02 * np.sin(5 * angle)
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            points.append([x, y])
            
        return np.array(points)
    
    # Advanced evolutionary algorithm with better operators
    def advanced_evolutionary_optimization():
        """Enhanced genetic algorithm with better operators and adaptive parameters"""
        population_size = 30
        generations = 50
        mutation_rate = 0.15
        
        # Initialize with better starting points
        def create_individual():
            # Mix multiple good construction methods
            choice = random.randint(0, 2)
            if choice == 0:
                return construct_algebraic_points() + np.random.normal(0, 0.015, (n, 2))
            elif choice == 1:
                return construct_hexagonal_grid() + np.random.normal(0, 0.015, (n, 2))
            else:
                return construct_fibonacci_spiral() + np.random.normal(0, 0.015, (n, 2))
        
        # Fitness function with better numerical stability
        def fitness(individual):
            # Ensure points are within bounds
            individual = np.clip(individual, 0, 1)
            distances = pdist(individual)
            
            if len(distances) == 0 or np.max(distances) == 0:
                return -np.inf
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist <= 1e-10:
                return -np.inf
                
            # Use log transform to stabilize the ratio calculation
            return min_dist / max_dist
        
        # Create initial population
        population = [create_individual() for _ in range(population_size)]
        
        # Evolution loop with better selection and reproduction
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [fitness(individual) for individual in population]
            
            # Tournament selection with adaptive tournament size
            def select_parent():
                tournament_size = max(3, int(0.3 * population_size) + 1)
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmax(tournament_fitness)]
                return population[winner_index].copy()
            
            # Create new population
            new_population = []
            
            # Elitism - keep top 3 individuals
            elite_indices = np.argsort(fitness_scores)[-3:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring through crossover and mutation
            while len(new_population) < population_size:
                parent1 = select_parent()
                parent2 = select_parent()
                
                # Blend crossover with weighted average
                alpha = random.random()
                child = alpha * parent1 + (1 - alpha) * parent2
                
                # Add mutation with adaptive strength
                if random.random() < mutation_rate:
                    mutation_strength = 0.02 + 0.01 * (generation / generations)  # Decreasing over time
                    noise = np.random.normal(0, mutation_strength, child.shape)
                    child += noise
                    child = np.clip(child, 0, 1)
                
                new_population.append(child)
            
            population = new_population[:population_size]
        
        # Return best individual
        final_fitnesses = [fitness(individual) for individual in population]
        best_index = np.argmax(final_fitnesses)
        return population[best_index]
    
    # Enhanced simulated annealing with better cooling schedule
    def enhanced_simulated_annealing():
        """Improved simulated annealing with better cooling and neighborhood moves"""
        # Start with a good construction
        current_points = construct_algebraic_points()
        current_points += np.random.normal(0, 0.01, current_points.shape)
        current_points = np.clip(current_points, 0, 1)
        
        def calculate_ratio(points):
            distances = pdist(points)
            if len(distances) == 0 or np.max(distances) == 0:
                return 0
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            return min_dist / max_dist if max_dist > 0 else 0
        
        current_ratio = calculate_ratio(current_points)
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        # Better cooling schedule
        temperature = 1.0
        cooling_rate = 0.998  # Slower cooling for better exploration
        min_temperature = 1e-6
        steps_per_temp = 50
        
        while temperature > min_temperature:
            for _ in range(steps_per_temp):
                # Create neighbor solution with multiple strategies
                neighbor_points = current_points.copy()
                
                # Choose move type with probabilities
                move_type = random.choices(['single', 'pair', 'cluster'], weights=[0.5, 0.3, 0.2])[0]
                
                if move_type == 'single':
                    # Move single point
                    idx = random.randint(0, n - 1)
                    neighbor_points[idx] += np.random.normal(0, 0.005, 2)
                elif move_type == 'pair':
                    # Move two nearby points
                    idx1, idx2 = random.sample(range(n), 2)
                    delta = np.random.normal(0, 0.008, 2)
                    neighbor_points[idx1] += delta
                    neighbor_points[idx2] += delta
                else:  # cluster
                    # Move a group of points
                    indices = random.sample(range(n), random.randint(2, 5))
                    delta = np.random.normal(0, 0.01, 2)
                    for idx in indices:
                        neighbor_points[idx] += delta
                
                # Clip to bounds
                neighbor_points = np.clip(neighbor_points, 0, 1)
                
                neighbor_ratio = calculate_ratio(neighbor_points)
                
                # Accept or reject based on Metropolis criterion
                if neighbor_ratio > current_ratio:
                    current_points = neighbor_points
                    current_ratio = neighbor_ratio
                    if current_ratio > best_ratio:
                        best_points = neighbor_points.copy()
                        best_ratio = current_ratio
                else:
                    # Accept with probability based on temperature and difference
                    delta = neighbor_ratio - current_ratio
                    if random.random() < np.exp(delta / temperature):
                        current_points = neighbor_points
                        current_ratio = neighbor_ratio
            
            temperature *= cooling_rate
        
        return best_points
    
    # Try multiple constructive approaches first
    strategies = [
        construct_algebraic_points,
        construct_hexagonal_grid,
        construct_fibonacci_spiral
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    # Try each constructive approach
    for strategy in strategies:
        try:
            points = strategy()
            # Add small random noise to break symmetries
            points += np.random.normal(0, 0.005, points.shape)
            points = np.clip(points, 0, 1)
            
            distances = pdist(points)
            if len(distances) > 0:
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points.copy()
        except Exception as e:
            continue
    
    # If no constructive approach worked, start with random
    if best_points is None:
        best_points = np.random.rand(16, 2)
    
    # Refine using advanced evolutionary algorithm
    try:
        evolved_points = advanced_evolutionary_optimization()
        distances = pdist(evolved_points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_points = evolved_points
                    best_ratio = ratio
    except Exception:
        pass
    
    # Final refinement with enhanced simulated annealing
    try:
        sa_points = enhanced_simulated_annealing()
        distances = pdist(sa_points)
        if len(distances) > 0:
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_points = sa_points
                    best_ratio = ratio
    except Exception:
        pass
    
    # As a last resort, use differential evolution for global search
    if best_ratio < 0.08:  # If we're still not satisfied
        try:
            def objective(params):
                points = params.reshape(-1, 2)
                distances = pdist(points)
                if len(distances) == 0 or np.max(distances) == 0:
                    return 0
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                return -min_dist / max_dist if max_dist > 0 else 0
            
            bounds = [(0, 1) for _ in range(32)]
            result = differential_evolution(
                objective,
                bounds,
                maxiter=200,  # Reduced iterations to meet time limits
                popsize=15,   # Smaller population for speed
                seed=42,
                polish=True,
                strategy='best1bin'
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                distances = pdist(refined_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_points = refined_points
        except Exception:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
