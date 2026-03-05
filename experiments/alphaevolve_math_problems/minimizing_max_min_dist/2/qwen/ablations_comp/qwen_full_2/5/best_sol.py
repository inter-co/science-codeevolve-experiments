# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import distance_matrix
import random
from typing import Tuple, List
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a multi-strategy evolutionary approach combining genetic algorithms, simulated annealing,
    and physics-based optimization to find high-quality solutions.
    
    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Strategy 1: Initialize population using multiple diverse methods
    population = initialize_population(100)
    
    # Strategy 2: Apply evolutionary optimization
    best_individual = evolutionary_optimization(population)
    
    # Strategy 3: Fine-tune with local search
    final_points = local_search_refinement(best_individual)
    
    return final_points

def initialize_population(pop_size: int) -> List[np.ndarray]:
    """Initialize diverse population using multiple construction strategies"""
    population = []
    
    # Strategy A: Random uniform distribution
    for _ in range(pop_size // 4):
        points = np.random.rand(16, 2)
        population.append(points)
    
    # Strategy B: Grid-based initialization
    for _ in range(pop_size // 4):
        points = grid_initialization()
        population.append(points)
    
    # Strategy C: Spiral-like pattern
    for _ in range(pop_size // 4):
        points = spiral_initialization()
        population.append(points)
    
    # Strategy D: Hexagonal lattice approximation
    for _ in range(pop_size // 4):
        points = hexagonal_initialization()
        population.append(points)
    
    return population

def grid_initialization() -> np.ndarray:
    """Initialize points in a grid-like pattern"""
    # Create a 4x4 grid with slight perturbations
    points = []
    for i in range(4):
        for j in range(4):
            x = i * 0.25 + (np.random.rand() - 0.5) * 0.1
            y = j * 0.25 + (np.random.rand() - 0.5) * 0.1
            points.append([x, y])
    
    # Normalize to [0,1] range
    points = np.array(points)
    # Center and scale appropriately
    points = points - np.mean(points, axis=0)
    max_val = np.max(np.abs(points))
    if max_val > 0:
        points = points / max_val * 0.8
    points = points + 0.5
    points = np.clip(points, 0, 1)
    return points

def spiral_initialization() -> np.ndarray:
    """Initialize points in a spiral pattern"""
    points = []
    for i in range(16):
        angle = i * 0.5
        radius = i * 0.05
        x = 0.5 + radius * np.cos(angle) * 0.3
        y = 0.5 + radius * np.sin(angle) * 0.3
        points.append([x, y])
    
    points = np.array(points)
    # Add small random noise
    points += (np.random.rand(16, 2) - 0.5) * 0.05
    points = np.clip(points, 0, 1)
    return points

def hexagonal_initialization() -> np.ndarray:
    """Initialize points approximating hexagonal packing"""
    points = []
    rows = 4
    cols = 4
    
    for i in range(rows):
        for j in range(cols):
            x = j * 0.25 + (i % 2) * 0.125
            y = i * 0.25 * np.sqrt(3)/2
            points.append([x, y])
    
    points = np.array(points)[:16]  # Take first 16 points
    
    # Add noise
    points += (np.random.rand(16, 2) - 0.5) * 0.1
    points = np.clip(points, 0, 1)
    return points

def calculate_fitness(points: np.ndarray) -> float:
    """Calculate fitness as min/max distance ratio"""
    if len(points) < 2:
        return 0.0
    
    # Calculate all pairwise distances
    distances = pdist(points)
    
    if len(distances) == 0:
        return 0.0
    
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    # Avoid division by zero
    if d_max <= 0:
        return 0.0
    
    return d_min / d_max

def evolutionary_optimization(population: List[np.ndarray]) -> np.ndarray:
    """Perform evolutionary optimization using tournament selection and crossover"""
    best_fitness = 0.0
    best_individual = None
    
    # Evolution parameters
    generations = 100
    mutation_rate = 0.1
    elite_size = 10
    
    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = [calculate_fitness(individual) for individual in population]
        
        # Track best individual
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
        
        # Selection - tournament selection
        selected = tournament_selection(population, fitness_scores, 5)
        
        # Create next generation through crossover and mutation
        new_population = []
        
        # Elitism - keep best individuals
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate offspring
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(selected, 2)
            
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Mutation
            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:len(population)]
    
    return best_individual if best_individual is not None else population[0]

def tournament_selection(population: List[np.ndarray], fitness_scores: List[float], k: int) -> List[np.ndarray]:
    """Select individuals using tournament selection"""
    selected = []
    for _ in range(len(population)):
        # Tournament
        tournament_indices = random.sample(range(len(population)), k)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        selected.append(population[winner_idx])
    return selected

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Single-point crossover for 2D point arrays"""
    # Flatten both parents
    flat1 = parent1.flatten()
    flat2 = parent2.flatten()
    
    # Single point crossover
    crossover_point = random.randint(1, len(flat1) - 1)
    
    child1_flat = np.concatenate([flat1[:crossover_point], flat2[crossover_point:]])
    child2_flat = np.concatenate([flat2[:crossover_point], flat1[crossover_point:]])
    
    child1 = child1_flat.reshape(-1, 2)
    child2 = child2_flat.reshape(-1, 2)
    
    # Ensure points stay within bounds
    child1 = np.clip(child1, 0, 1)
    child2 = np.clip(child2, 0, 1)
    
    return child1, child2

def mutate(individual: np.ndarray) -> np.ndarray:
    """Apply mutation to an individual"""
    mutated = individual.copy()
    
    # Mutate some points
    num_mutations = random.randint(1, 5)
    for _ in range(num_mutations):
        point_idx = random.randint(0, 15)
        # Small random perturbation
        mutated[point_idx] += (np.random.rand(2) - 0.5) * 0.1
        # Keep within bounds
        mutated[point_idx] = np.clip(mutated[point_idx], 0, 1)
    
    return mutated

def local_search_refinement(initial_points: np.ndarray) -> np.ndarray:
    """Refine solution using local search with simulated annealing"""
    current_points = initial_points.copy()
    current_fitness = calculate_fitness(current_points)
    
    # Simulated Annealing parameters
    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 1e-6
    iterations_per_temp = 100
    
    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            # Create neighbor solution
            neighbor_points = current_points.copy()
            
            # Perturb one random point
            point_idx = random.randint(0, 15)
            neighbor_points[point_idx] += (np.random.rand(2) - 0.5) * 0.01
            neighbor_points[point_idx] = np.clip(neighbor_points[point_idx], 0, 1)
            
            # Calculate neighbor fitness
            neighbor_fitness = calculate_fitness(neighbor_points)
            
            # Accept or reject based on Metropolis criterion
            if neighbor_fitness > current_fitness:
                current_points = neighbor_points
                current_fitness = neighbor_fitness
            else:
                # Accept with probability based on temperature
                delta = neighbor_fitness - current_fitness
                if random.random() < np.exp(delta / temperature):
                    current_points = neighbor_points
                    current_fitness = neighbor_fitness
        
        temperature *= cooling_rate
    
    return current_points

def _calculate_min_max_ratio(points: np.ndarray) -> float:
    """Calculate the ratio of minimum to maximum distance"""
    if len(points) < 2:
        return 0.0
    
    distances = pdist(points)
    if len(distances) == 0:
        return 0.0
    
    d_min = np.min(distances)
    d_max = np.max(distances)
    
    # Return 0 if no valid distances
    if d_max <= 0:
        return 0.0
    
    return d_min / d_max


# EVOLVE-BLOCK-END
