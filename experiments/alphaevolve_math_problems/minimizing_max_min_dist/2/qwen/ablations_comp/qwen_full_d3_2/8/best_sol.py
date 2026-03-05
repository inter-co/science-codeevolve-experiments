# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import random


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining multiple geometric strategies, 
    robust optimization techniques, and global search methods inspired by successful approaches.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)  # For reproducibility
    
    # Constructive approach using roots of unity for good angular distribution
    def construct_from_roots_of_unity():
        """Construct points using roots of unity - excellent for angular distribution"""
        n = 16
        points = []
        
        # Create points on a circle using roots of unity
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        
        return np.array(points)
    
    # Constructive approach using finite field geometry
    def construct_from_galois_field():
        """Construct points using finite field arithmetic for uniform distribution"""
        # Use a 4x4 grid with offset pattern (similar to hexagonal tiling)
        points = []
        for i in range(4):
            for j in range(4):
                # Offset every other row for better distribution
                x_offset = 0.25 if i % 2 == 1 else 0
                x = 0.125 + j * 0.25 + x_offset
                y = 0.125 + i * 0.25
                points.append([x, y])
        
        return np.array(points)
    
    # Constructive approach using Fibonacci-like spiral
    def construct_fibonacci_spiral():
        """Construct points using Fibonacci spiral for good distribution"""
        n = 16
        points = []
        
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        for i in range(n):
            # Map to unit square [0.1, 0.9] x [0.1, 0.9]
            theta = 2 * np.pi * i / phi
            r = 0.4 * np.sqrt(i / (n - 1)) if n > 1 else 0
            x = 0.5 + 0.4 * r * np.cos(theta)
            y = 0.5 + 0.4 * r * np.sin(theta)
            points.append([x, y])
            
        return np.array(points)
    
    # Evolutionary algorithm for optimization
    def evolutionary_optimization():
        """Use genetic algorithm approach to optimize point placement"""
        n = 16
        population_size = 50
        generations = 150
        mutation_rate = 0.1
        
        # Initialize population with diverse strategies
        def create_individual():
            # Mix different construction methods
            if random.random() < 0.3:
                return construct_from_roots_of_unity() + np.random.normal(0, 0.02, (n, 2))
            elif random.random() < 0.6:
                return construct_from_galois_field() + np.random.normal(0, 0.02, (n, 2))
            else:
                return construct_fibonacci_spiral() + np.random.normal(0, 0.02, (n, 2))
        
        # Fitness function - maximize min/max ratio
        def fitness(individual):
            # Ensure points are within bounds
            individual = np.clip(individual, 0, 1)
            distances = pdist(individual)
            
            if len(distances) == 0 or np.max(distances) == 0:
                return -np.inf
                
            min_dist = np.min(distances)
            max_dist = np.max(distances)
            
            if max_dist <= 1e-12:
                return -np.inf
                
            return min_dist / max_dist
        
        # Create initial population
        population = [create_individual() for _ in range(population_size)]
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [fitness(individual) for individual in population]
            
            # Selection - tournament selection
            def select_parent():
                tournament_size = 3
                tournament_indices = random.sample(range(population_size), tournament_size)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_index = tournament_indices[np.argmax(tournament_fitness)]
                return population[winner_index].copy()
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individuals
            elite_indices = np.argsort(fitness_scores)[-5:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < population_size:
                parent1 = select_parent()
                parent2 = select_parent()
                
                # Crossover - blend parents
                alpha = random.random()
                child = alpha * parent1 + (1 - alpha) * parent2
                
                # Mutation
                if random.random() < mutation_rate:
                    mutation_strength = 0.03
                    noise = np.random.normal(0, mutation_strength, child.shape)
                    child += noise
                    child = np.clip(child, 0, 1)
                
                new_population.append(child)
            
            population = new_population[:population_size]
        
        # Return best individual
        final_fitnesses = [fitness(individual) for individual in population]
        best_index = np.argmax(final_fitnesses)
        return population[best_index]
    
    # Simulated Annealing approach
    def simulated_annealing_optimization(initial_points):
        """Use simulated annealing to refine point placement"""
        n = 16
        
        # Start with a good construction
        current_points = initial_points.copy()
        current_points += np.random.normal(0, 0.02, current_points.shape)
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
        
        # Simulated annealing parameters
        temperature = 1.0
        cooling_rate = 0.995
        min_temperature = 1e-6
        steps_per_temp = 100
        
        while temperature > min_temperature:
            for _ in range(steps_per_temp):
                # Create neighbor solution
                neighbor_points = current_points.copy()
                # Perturb one point at a time
                idx = random.randint(0, n - 1)
                neighbor_points[idx] += np.random.normal(0, 0.01, 2)
                neighbor_points[idx] = np.clip(neighbor_points[idx], 0, 1)
                
                neighbor_ratio = calculate_ratio(neighbor_points)
                
                # Accept or reject based on Metropolis criterion
                if neighbor_ratio > current_ratio:
                    current_points = neighbor_points
                    current_ratio = neighbor_ratio
                    if current_ratio > best_ratio:
                        best_points = neighbor_points.copy()
                        best_ratio = current_ratio
                else:
                    # Accept with probability based on temperature
                    delta = neighbor_ratio - current_ratio
                    if random.random() < np.exp(delta / temperature):
                        current_points = neighbor_points
                        current_ratio = neighbor_ratio
            
            temperature *= cooling_rate
        
        return best_points
    
    # Try multiple constructive approaches first
    strategies = [
        construct_from_roots_of_unity,
        construct_from_galois_field,
        construct_fibonacci_spiral
    ]
    
    best_points = None
    best_ratio = -np.inf
    
    # Try each constructive approach
    for strategy in strategies:
        try:
            points = strategy()
            # Add small random noise to break symmetries
            points += np.random.normal(0, 0.01, points.shape)
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
        except Exception:
            continue
    
    # If no constructive approach worked, start with random
    if best_points is None:
        best_points = np.random.rand(16, 2)
    
    # Refine using evolutionary algorithm
    try:
        evolved_points = evolutionary_optimization()
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
    
    # Final refinement with simulated annealing
    try:
        sa_points = simulated_annealing_optimization(best_points)
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
    if best_ratio < 0.1:  # If we're still not satisfied
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
                maxiter=500,
                popsize=30,
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
