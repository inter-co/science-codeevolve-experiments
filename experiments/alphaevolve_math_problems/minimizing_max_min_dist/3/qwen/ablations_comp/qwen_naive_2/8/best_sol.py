# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
from scipy.spatial import ConvexHull
import warnings
from numba import jit
from sklearn.cluster import KMeans
import random


@jit(nopython=True)
def compute_distances_fast(points):
    """Fast computation of pairwise distances using Numba"""
    n = points.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(3):
                diff = points[i, k] - points[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Implements a novel evolutionary algorithm with geometric constraints and adaptive search.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    start_time = time.time()
    
    n = 14
    d = 3
    max_time = 55  # Leave 5 seconds for cleanup
    
    # Initialize with a geometrically informed approach
    points = _initialize_geometrically_informed(n)
    
    # Track best solution
    best_points = points.copy()
    best_ratio = _calculate_min_max_ratio(points)
    
    # Evolutionary parameters
    population_size = 20
    generations = 100
    mutation_rate = 0.1
    elite_size = 2
    
    # Main evolutionary loop
    for gen in range(generations):
        if time.time() - start_time > max_time:
            break
            
        # Generate new population through evolution
        new_population = []
        
        # Keep elite solutions
        sorted_pop = sorted([(p, _calculate_min_max_ratio(p)) for p in [best_points] + [points.copy() for _ in range(population_size-1)]], 
                           key=lambda x: x[1], reverse=True)
        for i in range(elite_size):
            new_population.append(sorted_pop[i][0])
        
        # Generate offspring through crossover and mutation
        while len(new_population) < population_size:
            if time.time() - start_time > max_time:
                break
                
            # Tournament selection
            parent1 = _tournament_selection(sorted_pop, 2)
            parent2 = _tournament_selection(sorted_pop, 2)
            
            # Crossover
            child = _crossover(parent1, parent2)
            
            # Mutation
            if random.random() < mutation_rate:
                child = _mutate(child, 0.05)
            
            # Boundary correction
            child = np.clip(child, 0, 1)
            
            new_population.append(child)
        
        # Evaluate population
        evaluated_population = [(p, _calculate_min_max_ratio(p)) for p in new_population]
        evaluated_population.sort(key=lambda x: x[1], reverse=True)
        
        # Update best solution
        current_best = evaluated_population[0][0]
        current_ratio = evaluated_population[0][1]
        
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = current_best.copy()
        
        # Set next generation points
        points = current_best.copy()
        
        # Adaptive cooling of mutation rate
        mutation_rate = max(0.01, mutation_rate * 0.98)
    
    # Final local optimization with simulated annealing
    final_points = _local_optimization_sa(best_points.copy(), max_time - (time.time() - start_time))
    
    return final_points


def _initialize_geometrically_informed(n):
    """Initialize points using geometric principles for better starting configuration"""
    # Start with a regular icosahedron-like structure
    # This provides good initial spread with symmetry
    
    # Generate points on a sphere using fibonacci spiral
    points = np.zeros((n, 3))
    
    # Golden angle for Fibonacci spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    # Distribute points on sphere first
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        
        theta = golden_angle * i
        
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        
        points[i] = [x, y, z]
    
    # Scale and perturb to create more uniform distribution
    points = (points + 1) / 2  # map from [-1,1] to [0,1]
    
    # Add structured perturbations to break perfect symmetries
    for i in range(n):
        # Add small random noise but maintain geometric relationships
        noise_magnitude = 0.02
        points[i] += np.random.normal(0, noise_magnitude, 3)
    
    # Clip to ensure within bounds
    points = np.clip(points, 0, 1)
    
    return points


def _tournament_selection(population, tournament_size):
    """Select individual using tournament selection"""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x[1])[0]


def _crossover(parent1, parent2):
    """Create offspring via uniform crossover"""
    child = np.zeros_like(parent1)
    for i in range(len(parent1)):
        if random.random() < 0.5:
            child[i] = parent1[i].copy()
        else:
            child[i] = parent2[i].copy()
    return child


def _mutate(points, mutation_strength):
    """Apply Gaussian mutation to points"""
    mutated = points.copy()
    for i in range(len(mutated)):
        if random.random() < 0.3:  # Mutate about 30% of points
            mutated[i] += np.random.normal(0, mutation_strength, 3)
    return mutated


def _calculate_min_max_ratio(points):
    """Calculate the min/max distance ratio efficiently"""
    if len(points) < 2:
        return 0
    
    # Use fast distance calculation
    distances = compute_distances_fast(points)
    
    # Get upper triangular part (excluding diagonal)
    triu_indices = np.triu_indices_from(distances, k=1)
    distances_flat = distances[triu_indices]
    
    if len(distances_flat) == 0:
        return 0
    
    min_dist = np.min(distances_flat)
    max_dist = np.max(distances_flat)
    
    if max_dist <= 0:
        return 0
    
    return min_dist / max_dist


def _local_optimization_sa(points, max_time):
    """Local optimization using simulated annealing"""
    start_time = time.time()
    current_points = points.copy()
    current_ratio = _calculate_min_max_ratio(current_points)
    
    # Initial temperature and cooling schedule
    temp = 0.1
    cooling_rate = 0.995
    min_temp = 1e-6
    
    # Number of iterations per temperature
    iter_per_temp = 100
    
    while temp > min_temp and (time.time() - start_time) < max_time:
        for _ in range(iter_per_temp):
            # Create neighbor solution
            neighbor = current_points.copy()
            # Perturb one point at random
            idx = random.randint(0, len(neighbor)-1)
            neighbor[idx] += np.random.normal(0, 0.005, 3)
            neighbor[idx] = np.clip(neighbor[idx], 0, 1)
            
            # Calculate ratio for neighbor
            neighbor_ratio = _calculate_min_max_ratio(neighbor)
            
            # Accept or reject based on Metropolis criterion
            if neighbor_ratio > current_ratio:
                current_points = neighbor
                current_ratio = neighbor_ratio
            else:
                # Accept with probability based on temperature
                delta = neighbor_ratio - current_ratio
                if random.random() < np.exp(delta / temp):
                    current_points = neighbor
                    current_ratio = neighbor_ratio
        
        temp *= cooling_rate
    
    return current_points


# EVOLVE-BLOCK-END
