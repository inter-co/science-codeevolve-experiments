# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from typing import List
import random
from numba import jit
import time
from scipy.optimize import differential_evolution, minimize
import math
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform
import warnings

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: np.ndarray) -> tuple:
    """
    Fast computation of autoconvolution norms using numba-optimized loops.
    Optimized for performance and accuracy.
    """
    n = len(f_values)
    # Compute autoconvolution manually for step functions
    # g[k] = sum_{i=0}^{n-1} f[i] * f[k-i] where both indices valid
    g = np.zeros(2*n - 1)
    
    # Efficient manual convolution for step functions
    for i in range(n):
        for j in range(n):
            k = i + j
            if 0 <= k < len(g):
                g[k] += f_values[i] * f_values[j]
    
    # Take central portion for symmetric case (matching the problem setup)
    start_idx = n - 1
    end_idx = 2*n - 2
    g_centered = g[start_idx:end_idx+1]
    
    # Compute the three norms using more accurate approach
    g_squared = g_centered * g_centered
    g_abs = np.abs(g_centered)
    
    # L2 norm squared - sum of squares (more accurate than trapezoidal for discrete case)
    norm_2_sq = np.sum(g_squared)
    
    # L1 norm - sum of absolute values
    norm_1 = np.sum(g_abs)
    
    # L-infinity norm
    norm_inf = np.max(g_abs)
    
    return norm_2_sq, norm_1, norm_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    Uses efficient convolution and proper normalization.
    """
    # Convert to numpy array for easier handling
    f = np.array(f_values)
    
    # Use fast numba version
    return compute_autoconvolution_norms_fast(f)

def evaluate_c2(f_values: List[float]) -> float:
    """Evaluate C2 value for given step function."""
    try:
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero with a very small epsilon
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2
    except Exception as e:
        # In case of any numerical issues, return a small negative value
        return -1.0

def create_advanced_individual(length: int) -> List[float]:
    """Create advanced initial individuals with mathematical insights."""
    individual = []
    
    # Strategy 1: Create a pattern that balances high values with low values
    # This aims to create autoconvolution with moderate peaks and valleys
    center = length // 2
    peak_height = 1.5
    tail_width = length / 4
    
    for i in range(length):
        # Distance from center
        distance = abs(i - center) / (length / 2)
        
        # Create a pattern that's more uniform but with controlled peaks
        if distance < 0.25:
            # Central region with higher values
            value = peak_height * (1 - distance * 4) + 0.5
        elif distance < 0.5:
            # Middle region with gradual decline
            value = 0.5 * (1 - (distance - 0.25) * 2) + 0.2
        else:
            # Outer regions with low values
            value = 0.2 * math.exp(-(distance - 0.5) * 4)
        
        # Add some randomness to avoid local minima
        noise = random.gauss(0, 0.05 * max(0.1, value))
        individual.append(max(0, value + noise))
    
    return individual

def create_bimodal_individual(length: int) -> List[float]:
    """Create a bimodal distribution that might produce good autoconvolution."""
    individual = []
    
    # Create two peaks separated by a valley
    left_peak_center = length // 3
    right_peak_center = 2 * length // 3
    
    for i in range(length):
        # Left peak
        left_dist = abs(i - left_peak_center) / (length / 6)
        left_peak = 1.2 * math.exp(-left_dist**2 * 2)
        
        # Right peak  
        right_dist = abs(i - right_peak_center) / (length / 6)
        right_peak = 1.0 * math.exp(-right_dist**2 * 2)
        
        # Combine with some noise
        value = left_peak + right_peak + random.uniform(-0.1, 0.1)
        individual.append(max(0, value))
    
    return individual

def create_sine_wave_individual(length: int) -> List[float]:
    """Create sine wave pattern that promotes flat autoconvolution."""
    individual = []
    
    # Create pattern with multiple sine waves to create uniform autoconvolution
    for i in range(length):
        # Multiple frequencies to create complex but balanced pattern
        val = 0.7 + 0.2 * math.sin(i * 0.1) + 0.1 * math.sin(i * 0.25) + \
              0.05 * math.sin(i * 0.5) + random.uniform(-0.05, 0.05)
        individual.append(max(0, val))
    
    return individual

def create_uniform_individual(length: int) -> List[float]:
    """Create a uniform distribution that should give reasonable results."""
    return [1.0] * length

def create_exp_decay_individual(length: int) -> List[float]:
    """Create exponential decay pattern from center."""
    individual = []
    center = length // 2
    
    for i in range(length):
        distance = abs(i - center) / (length / 2)
        # Exponential decay pattern
        value = math.exp(-distance * 3) + random.uniform(-0.05, 0.05)
        individual.append(max(0, value))
    
    return individual

def create_multipeak_individual(length: int) -> List[float]:
    """Create multi-peak pattern for potentially better autoconvolution."""
    individual = []
    
    # Create multiple peaks with different heights and positions
    peaks = [
        (length // 4, 1.5),
        (length // 2, 1.8),
        (3 * length // 4, 1.3)
    ]
    
    for i in range(length):
        value = 0.0
        for peak_pos, peak_height in peaks:
            dist = abs(i - peak_pos) / (length / 10)
            value += peak_height * math.exp(-dist**2 * 2)
        
        # Add some noise
        value += random.uniform(-0.1, 0.1)
        individual.append(max(0, value))
    
    return individual

def create_adaptive_mutation(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Enhanced mutation with adaptive strategies based on individual characteristics."""
    mutated = individual.copy()
    
    # Calculate statistics about the current individual
    mean_val = np.mean(individual)
    std_val = np.std(individual)
    
    # Adaptive mutation based on value characteristics
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            current_val = mutated[i]
            
            # Adjust mutation strength based on context
            if current_val > mean_val + std_val:
                # High values get smaller mutations
                mutation_strength = 0.1 * std_val if std_val > 0 else 0.05
                mutated[i] = max(0, current_val + np.random.normal(0, mutation_strength))
            elif current_val < mean_val - std_val:
                # Low values get larger mutations
                mutation_strength = 0.2 * std_val if std_val > 0 else 0.1
                mutated[i] = max(0, current_val + np.random.normal(0, mutation_strength))
            else:
                # Middle values get moderate mutations
                mutation_strength = 0.15 * std_val if std_val > 0 else 0.08
                mutated[i] = max(0, current_val + np.random.normal(0, mutation_strength))
                
    return mutated

def crossover_parents(parent1: List[float], parent2: List[float]) -> List[float]:
    """Enhanced crossover with adaptive blending."""
    child = []
    min_len = min(len(parent1), len(parent2))
    
    for i in range(min_len):
        # Use adaptive crossover based on parent similarity
        diff = abs(parent1[i] - parent2[i])
        
        if random.random() < 0.7:
            # Blend with probability, weighted by difference
            alpha = random.random() * (1 - diff/10.0) if diff < 10 else random.random()
            blended = alpha * parent1[i] + (1 - alpha) * parent2[i]
            # Add noise proportional to the blended value
            noise_scale = 0.05 * blended if blended > 0 else 0.05
            noise = random.gauss(0, noise_scale)
            child.append(max(0, blended + noise))
        else:
            # Take from one parent with slight variation
            parent = parent1 if random.random() < 0.5 else parent2
            child.append(max(0, parent[i] + random.gauss(0, 0.05 * parent[i] + 0.01)))
            
    return child

def tournament_selection(population: List[List[float]], fitness_scores: List[tuple], k: int) -> List[float]:
    """Improved tournament selection with dynamic k and better pressure."""
    # Dynamic tournament size based on population diversity
    if len(population) > 50:
        k = min(10, max(3, len(population) // 20))
    else:
        k = min(7, max(3, len(population) // 5))
    
    tournament_indices = random.sample(range(len(population)), min(k, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    tournament_fitness.sort(key=lambda x: x[1], reverse=True)
    return tournament_fitness[0][0]

def optimize_with_local_search(initial_guess: List[float], max_iter: int = 100) -> List[float]:
    """Apply local search to refine promising solutions."""
    current = initial_guess.copy()
    current_c2 = evaluate_c2(current)
    
    for iteration in range(max_iter):
        # Create neighbor by small perturbations
        neighbor = create_adaptive_mutation(current, 0.1)
        neighbor_c2 = evaluate_c2(neighbor)
        
        if neighbor_c2 > current_c2:
            current = neighbor
            current_c2 = neighbor_c2
        else:
            # Occasionally accept worse solutions to escape local optima
            if random.random() < 0.05:
                current = neighbor
                current_c2 = neighbor_c2
    
    return current

def evolve_step_function() -> List[float]:
    """
    Enhanced evolution with better strategies to maximize C2.
    """
    # Parameters for enhanced genetic algorithm
    population_size = 400  # Increased population size for better exploration
    generations = 180      # More generations with early stopping
    elite_size = 40        # More elites for preservation
    mutation_rate = 0.12   # Balanced mutation rate
    
    # Initialize population with diverse strategies
    population = []
    for _ in range(population_size):
        length = random.randint(600, 1200)  # More focused range for better convergence
        strategy = random.random()
        
        if strategy < 0.25:
            # Advanced structured individuals
            population.append(create_advanced_individual(length))
        elif strategy < 0.5:
            # Bimodal patterns
            population.append(create_bimodal_individual(length))
        elif strategy < 0.75:
            # Sine wave patterns
            population.append(create_sine_wave_individual(length))
        else:
            # Multi-peak patterns
            population.append(create_multipeak_individual(length))
    
    best_fitness = 0
    best_individual = None
    start_time = time.time()
    
    for generation in range(generations):
        # Early stopping if we've been stuck for a while
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break
            
        # Evaluate fitness for all individuals
        fitness_scores = [(ind, evaluate_c2(ind)) for ind in population]
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Track best individual
        if fitness_scores[0][1] > best_fitness:
            best_fitness = fitness_scores[0][1]
            best_individual = fitness_scores[0][0].copy()
        
        # Select top individuals (elitism)
        elites = [ind for ind, _ in fitness_scores[:elite_size]]
        
        # Generate new population through crossover and mutation
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            # Tournament selection with adaptive k
            parent1 = tournament_selection(population, fitness_scores, 7)
            parent2 = tournament_selection(population, fitness_scores, 7)
            
            # Crossover
            child = crossover_parents(parent1, parent2)
            
            # Mutation with adaptive strategy
            child = flexible_mutate(child, mutation_rate)
            
            # Occasionally introduce completely new structured individual
            if random.random() < 0.1:
                child_length = random.randint(600, 1200)
                child_strategy = random.random()
                if child_strategy < 0.25:
                    child = create_advanced_individual(child_length)
                elif child_strategy < 0.5:
                    child = create_bimodal_individual(child_length)
                elif child_strategy < 0.75:
                    child = create_sine_wave_individual(child_length)
                else:
                    child = create_multipeak_individual(child_length)
            
            new_population.append(child)
        
        population = new_population
    
    # Final local search on the best solution
    if best_individual is not None:
        refined_best = optimize_with_local_search(best_individual, 75)
        refined_c2 = evaluate_c2(refined_best)
        if refined_c2 > best_fitness:
            return refined_best
    
    return best_individual if best_individual is not None else create_advanced_individual(800)

def construct_function() -> List[float]:
    """Main function to construct optimized step-function with high C2 value."""
    # Try evolutionary approach first
    evoluted = evolve_step_function()
    
    # Try local search on the evolved solution
    refined_evoluted = optimize_with_local_search(evoluted, 100)
    
    # Try different initialization strategies
    candidates = [evoluted, refined_evoluted]
    
    # Add more carefully crafted structures
    for i in range(15):
        length = random.randint(600, 1200)
        
        # Create a pattern with oscillating behavior that creates flat autoconvolution
        oscillating_pattern = []
        for j in range(length):
            # Combine multiple frequencies with different amplitudes
            val = 0.8 + 0.15 * math.sin(j * 0.05) + 0.1 * math.sin(j * 0.15) + \
                  0.05 * math.sin(j * 0.3) + random.uniform(-0.05, 0.05)
            oscillating_pattern.append(max(0, val))
        candidates.append(oscillating_pattern)
        
        # Create a multi-peak pattern with specific spacing
        multi_peak = []
        for j in range(length):
            # Multiple peaks with different heights
            peak1 = 0.6 * math.exp(-((j - length//3)**2) / (length/10)**2)
            peak2 = 0.8 * math.exp(-((j - 2*length//3)**2) / (length/10)**2)
            peak3 = 0.4 * math.exp(-((j - length//2)**2) / (length/15)**2)
            val = peak1 + peak2 + peak3 + random.gauss(0, 0.05)
            multi_peak.append(max(0, val))
        candidates.append(multi_peak)
        
        # Create a pattern with exponential decay
        exp_pattern = []
        for j in range(length):
            # Exponential decay from center
            distance = abs(j - length//2) / (length/2)
            val = 1.0 * math.exp(-distance * 5) + random.uniform(-0.1, 0.1)
            exp_pattern.append(max(0, val))
        candidates.append(exp_pattern)
    
    # Evaluate all candidates and return the best
    best_candidate = max(candidates, key=evaluate_c2)
    
    return best_candidate

def create_advanced_individual(length: int) -> List[float]:
    """Create advanced initial individuals with mathematical insights."""
    individual = []
    
    # Strategy 1: Create a pattern that balances high values with low values
    # This aims to create autoconvolution with moderate peaks and valleys
    center = length // 2
    peak_height = 1.5
    tail_width = length / 4
    
    for i in range(length):
        # Distance from center
        distance = abs(i - center) / (length / 2)
        
        # Create a pattern that's more uniform but with controlled peaks
        if distance < 0.25:
            # Central region with higher values
            value = peak_height * (1 - distance * 4) + 0.5
        elif distance < 0.5:
            # Middle region with gradual decline
            value = 0.5 * (1 - (distance - 0.25) * 2) + 0.2
        else:
            # Outer regions with low values
            value = 0.2 * math.exp(-(distance - 0.5) * 4)
        
        # Add some randomness to avoid local minima
        noise = random.gauss(0, 0.05 * max(0.1, value))
        individual.append(max(0, value + noise))
    
    return individual

def create_bimodal_individual(length: int) -> List[float]:
    """Create a bimodal distribution that might produce good autoconvolution."""
    individual = []
    
    # Create two peaks separated by a valley
    left_peak_center = length // 3
    right_peak_center = 2 * length // 3
    
    for i in range(length):
        # Left peak
        left_dist = abs(i - left_peak_center) / (length / 6)
        left_peak = 1.2 * math.exp(-left_dist**2 * 2)
        
        # Right peak  
        right_dist = abs(i - right_peak_center) / (length / 6)
        right_peak = 1.0 * math.exp(-right_dist**2 * 2)
        
        # Combine with some noise
        value = left_peak + right_peak + random.uniform(-0.1, 0.1)
        individual.append(max(0, value))
    
    return individual

def create_sine_wave_individual(length: int) -> List[float]:
    """Create sine wave pattern that promotes flat autoconvolution."""
    individual = []
    
    # Create pattern with multiple sine waves to create uniform autoconvolution
    for i in range(length):
        # Multiple frequencies to create complex but balanced pattern
        val = 0.7 + 0.2 * math.sin(i * 0.1) + 0.1 * math.sin(i * 0.25) + \
              0.05 * math.sin(i * 0.5) + random.uniform(-0.05, 0.05)
        individual.append(max(0, val))
    
    return individual

def create_flexible_mutation(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Enhanced mutation with adaptive strategies based on individual characteristics."""
    mutated = individual.copy()
    
    # Calculate statistics about the current individual
    mean_val = np.mean(individual)
    std_val = np.std(individual)
    
    # Adaptive mutation based on value characteristics
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            current_val = mutated[i]
            
            # Adjust mutation strength based on context
            if current_val > mean_val + std_val:
                # High values get smaller mutations
                mutation_strength = 0.1 * std_val if std_val > 0 else 0.05
                mutated[i] = max(0, current_val + np.random.normal(0, mutation_strength))
            elif current_val < mean_val - std_val:
                # Low values get larger mutations
                mutation_strength = 0.2 * std_val if std_val > 0 else 0.1
                mutated[i] = max(0, current_val + np.random.normal(0, mutation_strength))
            else:
                # Middle values get moderate mutations
                mutation_strength = 0.15 * std_val if std_val > 0 else 0.08
                mutated[i] = max(0, current_val + np.random.normal(0, mutation_strength))
                
    return mutated

def crossover_parents(parent1: List[float], parent2: List[float]) -> List[float]:
    """Enhanced crossover with adaptive blending."""
    child = []
    min_len = min(len(parent1), len(parent2))
    
    for i in range(min_len):
        # Use adaptive crossover based on parent similarity
        diff = abs(parent1[i] - parent2[i])
        
        if random.random() < 0.7:
            # Blend with probability, weighted by difference
            alpha = random.random() * (1 - diff/10.0) if diff < 10 else random.random()
            blended = alpha * parent1[i] + (1 - alpha) * parent2[i]
            # Add noise proportional to the blended value
            noise_scale = 0.05 * blended if blended > 0 else 0.05
            noise = random.gauss(0, noise_scale)
            child.append(max(0, blended + noise))
        else:
            # Take from one parent with slight variation
            parent = parent1 if random.random() < 0.5 else parent2
            child.append(max(0, parent[i] + random.gauss(0, 0.05 * parent[i] + 0.01)))
            
    return child

def tournament_selection(population: List[List[float]], fitness_scores: List[tuple], k: int) -> List[float]:
    """Improved tournament selection with dynamic k and better pressure."""
    # Dynamic tournament size based on population diversity
    if len(population) > 50:
        k = min(10, max(3, len(population) // 20))
    else:
        k = min(7, max(3, len(population) // 5))
    
    tournament_indices = random.sample(range(len(population)), min(k, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    tournament_fitness.sort(key=lambda x: x[1], reverse=True)
    return tournament_fitness[0][0]

def optimize_with_local_search(initial_guess: List[float], max_iter: int = 100) -> List[float]:
    """Apply local search to refine promising solutions."""
    current = initial_guess.copy()
    current_c2 = evaluate_c2(current)
    
    for iteration in range(max_iter):
        # Create neighbor by small perturbations
        neighbor = flexible_mutate(current, 0.1)
        neighbor_c2 = evaluate_c2(neighbor)
        
        if neighbor_c2 > current_c2:
            current = neighbor
            current_c2 = neighbor_c2
        else:
            # Occasionally accept worse solutions to escape local optima
            if random.random() < 0.05:
                current = neighbor
                current_c2 = neighbor_c2
    
    return current

def flexible_mutate(individual: List[float], mutation_rate: float) -> List[float]:
    """Flexible mutation with multiple strategies."""
    mutated = individual.copy()
    
    # Try different mutation strategies
    strategy = random.choice(['gaussian', 'uniform', 'adaptive'])
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            if strategy == 'gaussian':
                mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.08))
            elif strategy == 'uniform':
                mutated[i] = max(0, mutated[i] + random.uniform(-0.08, 0.08))
            else:  # adaptive
                # More sophisticated adaptive mutation
                if mutated[i] > 0.8:
                    mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.12))
                elif mutated[i] < 0.2:
                    mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.06))
                else:
                    mutated[i] = max(0, mutated[i] + np.random.normal(0, 0.1))
    
    return mutated

def evolve_step_function() -> List[float]:
    """
    Enhanced evolution with better strategies to maximize C2.
    """
    # Parameters for enhanced genetic algorithm
    population_size = 300  # Increased population size for better exploration
    generations = 150      # More generations with early stopping
    elite_size = 30        # More elites for preservation
    mutation_rate = 0.12   # Balanced mutation rate
    
    # Initialize population with diverse strategies
    population = []
    for _ in range(population_size):
        length = random.randint(400, 1200)  # More focused range
        strategy = random.random()
        
        if strategy < 0.3:
            # Advanced structured individuals
            population.append(create_advanced_individual(length))
        elif strategy < 0.6:
            # Bimodal patterns
            population.append(create_bimodal_individual(length))
        elif strategy < 0.9:
            # Sine wave patterns
            population.append(create_sine_wave_individual(length))
        else:
            # Random with better bounds
            population.append([random.uniform(0.2, 1.5) for _ in range(length)])
    
    best_fitness = 0
    best_individual = None
    start_time = time.time()
    
    for generation in range(generations):
        # Early stopping if we've been stuck for a while
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break
            
        # Evaluate fitness for all individuals
        fitness_scores = [(ind, evaluate_c2(ind)) for ind in population]
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Track best individual
        if fitness_scores[0][1] > best_fitness:
            best_fitness = fitness_scores[0][1]
            best_individual = fitness_scores[0][0].copy()
        
        # Select top individuals (elitism)
        elites = [ind for ind, _ in fitness_scores[:elite_size]]
        
        # Generate new population through crossover and mutation
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            # Tournament selection with adaptive k
            parent1 = tournament_selection(population, fitness_scores, 7)
            parent2 = tournament_selection(population, fitness_scores, 7)
            
            # Crossover
            child = crossover_parents(parent1, parent2)
            
            # Mutation with adaptive strategy
            child = flexible_mutate(child, mutation_rate)
            
            # Occasionally introduce completely new structured individual
            if random.random() < 0.1:
                child_length = random.randint(400, 1200)
                child_strategy = random.random()
                if child_strategy < 0.3:
                    child = create_advanced_individual(child_length)
                elif child_strategy < 0.6:
                    child = create_bimodal_individual(child_length)
                else:
                    child = create_sine_wave_individual(child_length)
            
            new_population.append(child)
        
        population = new_population
    
    # Final local search on the best solution
    if best_individual is not None:
        refined_best = optimize_with_local_search(best_individual, 50)
        refined_c2 = evaluate_c2(refined_best)
        if refined_c2 > best_fitness:
            return refined_best
    
    return best_individual if best_individual is not None else create_advanced_individual(500)

def construct_function() -> List[float]:
    """Main function to construct optimized step-function with high C2 value."""
    # Try evolutionary approach first
    evoluted = evolve_step_function()
    
    # Try local search on the evolved solution
    refined_evoluted = optimize_with_local_search(evoluted, 100)
    
    # Try different initialization strategies
    candidates = [evoluted, refined_evoluted]
    
    # Add more carefully crafted structures
    for i in range(12):
        length = random.randint(500, 1000)
        
        # Create a pattern with oscillating behavior that creates flat autoconvolution
        oscillating_pattern = []
        for j in range(length):
            # Combine multiple frequencies with different amplitudes
            val = 0.8 + 0.15 * math.sin(j * 0.05) + 0.1 * math.sin(j * 0.15) + \
                  0.05 * math.sin(j * 0.3) + random.uniform(-0.05, 0.05)
            oscillating_pattern.append(max(0, val))
        candidates.append(oscillating_pattern)
        
        # Create a multi-peak pattern with specific spacing
        multi_peak = []
        for j in range(length):
            # Multiple peaks with different heights
            peak1 = 0.6 * math.exp(-((j - length//3)**2) / (length/10)**2)
            peak2 = 0.8 * math.exp(-((j - 2*length//3)**2) / (length/10)**2)
            peak3 = 0.4 * math.exp(-((j - length//2)**2) / (length/15)**2)
            val = peak1 + peak2 + peak3 + random.gauss(0, 0.05)
            multi_peak.append(max(0, val))
        candidates.append(multi_peak)
        
        # Create a pattern with exponential decay
        exp_pattern = []
        for j in range(length):
            # Exponential decay from center
            distance = abs(j - length//2) / (length/2)
            val = 1.0 * math.exp(-distance * 5) + random.uniform(-0.1, 0.1)
            exp_pattern.append(max(0, val))
        candidates.append(exp_pattern)
    
    # Evaluate all candidates and return the best
    best_candidate = max(candidates, key=evaluate_c2)
    
    return best_candidate

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
