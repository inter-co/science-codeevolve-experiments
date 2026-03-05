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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# More efficient implementation with better numerical handling and reduced overhead
@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: np.ndarray) -> tuple:
    """
    Fast computation of autoconvolution norms using numba-optimized loops.
    """
    n = len(f_values)
    # Compute autoconvolution manually for step functions
    # g[k] = sum_{i=0}^{n-1} f[i] * f[k-i] where both indices valid
    g = np.zeros(2*n - 1)
    
    # Efficient manual convolution for step functions - optimized loop
    for i in range(n):
        f_i = f_values[i]
        for j in range(n):
            k = i + j
            if 0 <= k < len(g):
                g[k] += f_i * f_values[j]
    
    # Take central portion for symmetric case (matching the problem setup)
    start_idx = n - 1
    end_idx = 2*n - 2
    g_centered = g[start_idx:end_idx+1]
    
    # Compute the three norms using trapezoidal rule for L2 norm
    # For L2 norm squared, we use trapezoidal approximation for piecewise linear segments
    norm_2_sq = 0.0
    for i in range(len(g_centered) - 1):
        y1 = g_centered[i]
        y2 = g_centered[i+1]
        # Trapezoidal area approximation for segment: (y1^2 + y1*y2 + y2^2)/3 * h
        # Since h = 1 for our discrete case, we get: (y1^2 + y1*y2 + y2^2)/3
        norm_2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0
    
    # L1 norm (sum of absolute values)
    norm_1 = np.sum(np.abs(g_centered))
    
    # L-infinity norm
    norm_inf = np.max(np.abs(g_centered))
    
    return norm_2_sq, norm_1, norm_inf

# Optimized version using FFT for better performance when possible
def compute_autoconvolution_fft(f_values: np.ndarray) -> tuple:
    """
    Alternative computation using FFT for better performance on large arrays.
    """
    n = len(f_values)
    # Pad to power of 2 for FFT efficiency
    padded_n = 2**(int(np.ceil(np.log2(2*n - 1))) if n > 1 else 1)
    
    # Use FFT-based convolution
    f_padded = np.pad(f_values, (0, padded_n - n), mode='constant')
    g_fft = np.fft.fft(f_padded) * np.fft.fft(f_padded).conj()
    g = np.fft.ifft(g_fft).real[:2*n-1]
    
    # Extract central portion
    start_idx = n - 1
    end_idx = 2*n - 2
    g_centered = g[start_idx:end_idx+1]
    
    # Compute norms
    norm_2_sq = 0.0
    for i in range(len(g_centered) - 1):
        y1 = g_centered[i]
        y2 = g_centered[i+1]
        norm_2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0
    
    norm_1 = np.sum(np.abs(g_centered))
    norm_inf = np.max(np.abs(g_centered))
    
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
        
        # Avoid division by zero with better tolerance
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2
    except Exception as e:
        return 0.0

# Improved initialization with more mathematical insight
def create_mathematically_informed_pattern(length: int) -> List[float]:
    """Create pattern based on mathematical insights for maximizing C2."""
    individual = []
    
    # Use a pattern that creates flatter autoconvolutions
    # Based on the theory: patterns that spread mass more evenly tend to do better
    
    # Create a pattern with a smooth, non-uniform distribution that avoids extreme spikes
    # This uses a modified Gaussian with controlled variance and additional structure
    
    center = length // 2
    # Use a wider sigma to create more spread-out pattern
    sigma = length / 8.0
    
    for i in range(length):
        # Primary smooth component - Gaussian-like
        val = math.exp(-((i - center)**2) / (2 * sigma**2))
        
        # Add some structured variation to avoid trivial local optima
        # This creates more interesting autoconvolution behavior
        val += 0.15 * math.sin(i * 0.12) * math.cos(i * 0.06)
        val += 0.08 * math.sin(i * 0.25) * math.cos(i * 0.15)
        val += 0.05 * math.sin(i * 0.4) * math.cos(i * 0.25)
        
        # Add small random component for diversity
        val += random.uniform(-0.008, 0.008)
        
        individual.append(max(0, val))
    
    # Normalize to control magnitude and make it more suitable for optimization
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 6 for x in individual]
    
    return individual

def create_high_variance_pattern(length: int) -> List[float]:
    """Create pattern with higher variance to potentially create better autoconvolutions."""
    individual = []
    
    # Create pattern with more structured variation to encourage interesting convolution behavior
    for i in range(length):
        # Create pattern with multiple frequencies to encourage complex autoconvolution
        val = 0.5 + 0.3 * math.sin(i * 0.1) + 0.2 * math.sin(i * 0.2) + 0.1 * math.sin(i * 0.3)
        
        # Add some structured noise
        val += 0.05 * math.sin(i * 0.05) * math.cos(i * 0.1)
        val += 0.03 * math.cos(i * 0.25) * math.sin(i * 0.15)
        
        # Add small random component
        val += random.uniform(-0.02, 0.02)
        
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 6 for x in individual]
    
    return individual

def create_bimodal_pattern(length: int) -> List[float]:
    """Create bimodal pattern to encourage interesting autoconvolution properties."""
    individual = []
    
    # Create two distinct peaks with a valley between them
    mid_point = length // 2
    half_length = length // 2
    
    for i in range(length):
        # Two peaks - one at beginning, one at end
        if i < half_length:
            # First peak
            val = 0.6 + 0.3 * math.exp(-((i - 0)**2) / (half_length/3)**2)
        else:
            # Second peak  
            val = 0.6 + 0.3 * math.exp(-((i - length)**2) / (half_length/3)**2)
        
        # Add some oscillation to break symmetry
        val += 0.05 * math.sin(i * 0.1) + 0.02 * math.cos(i * 0.2)
        
        # Add small random component
        val += random.uniform(-0.01, 0.01)
        
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 5 for x in individual]
    
    return individual

def create_adaptive_mutation(individual: List[float], generation: int, 
                           mutation_rate_factor: float = 1.0) -> List[float]:
    """Enhanced mutation with adaptive parameters based on evolution progress."""
    mutated = individual.copy()
    
    # Adaptive mutation rate that decreases over generations
    mutation_rate = max(0.03, 0.15 * (1 - generation / 150) * mutation_rate_factor)
    
    # Adaptive mutation strength
    mutation_strength = 0.03 + 0.07 * (generation / 150)
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Use different mutation strategies based on value magnitude
            current_val = mutated[i]
            
            # Different mutation strengths for different value ranges
            if current_val < 0.05:
                # Very small values - very small mutation
                delta = np.random.normal(0, 0.002 * mutation_strength)
            elif current_val < 0.2:
                # Small values - small mutation
                delta = np.random.normal(0, 0.008 * mutation_strength)
            elif current_val < 0.5:
                # Medium values - moderate mutation
                delta = np.random.normal(0, 0.02 * mutation_strength)
            elif current_val < 1.0:
                # High values - medium mutation
                delta = np.random.normal(0, 0.04 * mutation_strength)
            else:
                # Very high values - smaller mutation to prevent overshoot
                delta = np.random.normal(0, 0.015 * mutation_strength)
                
            mutated[i] = max(0, current_val + delta)
                
    return mutated

def create_adaptive_crossover(parent1: List[float], parent2: List[float], 
                            generation: int) -> List[float]:
    """Adaptive crossover with better blending strategies."""
    child = []
    min_len = min(len(parent1), len(parent2))
    
    # Adaptive crossover probability based on generation
    crossover_prob = 0.7 + 0.2 * (1 - generation / 150)
    
    for i in range(min_len):
        if random.random() < crossover_prob:
            # Use adaptive weights based on parent similarity
            similarity = abs(parent1[i] - parent2[i])
            
            if similarity < 0.05:
                # Very similar parents - blend more conservatively
                alpha = random.uniform(0.3, 0.7)
            elif similarity < 0.2:
                # Moderately different - moderate blending
                alpha = random.uniform(0.2, 0.8)
            else:
                # Very different - more random blending
                alpha = random.random()
                
            blended = alpha * parent1[i] + (1 - alpha) * parent2[i]
            # Add noise proportional to the blended value
            noise = random.gauss(0, 0.01 * blended if blended > 0 else 0.003)
            child.append(max(0, blended + noise))
        else:
            # Uniform crossover with adaptive probabilities
            parent = parent1 if random.random() < 0.55 else parent2
            # Add noise to maintain diversity
            noise = random.gauss(0, 0.003 * parent[i] if parent[i] > 0 else 0.001)
            child.append(max(0, parent[i] + noise))
            
    return child

def create_improved_tournament_selection(population: List[List[float]], 
                                       fitness_scores: List[tuple], 
                                       k: int, generation: int) -> List[float]:
    """Improved tournament selection with better pressure and diversity."""
    tournament_indices = random.sample(range(len(population)), min(k, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    # Sort by fitness descending (higher is better)
    tournament_fitness.sort(key=lambda x: x[1], reverse=True)
    
    # Adaptive selection pressure based on generation
    selection_pressure = 0.6 + 0.2 * (1 - generation / 150)
    
    # Return the best of the tournament, but with some probabilistic element
    # to allow for some diversity
    if len(tournament_fitness) > 1 and random.random() < selection_pressure:
        # Some chance to pick from top 2 instead of just top 1
        return tournament_fitness[random.randint(0, min(1, len(tournament_fitness)-1))][0]
    else:
        return tournament_fitness[0][0]

def optimize_with_local_search(initial_guess: List[float], max_iter: int = 80) -> List[float]:
    """Perform local search refinement around initial guess with improved strategy."""
    current = initial_guess.copy()
    current_c2 = evaluate_c2(current)
    
    # Local search with adaptive step sizes and better termination criteria
    improvement_count = 0
    max_no_improvement = 15
    
    for iteration in range(max_iter):
        # Create neighbor by small perturbations
        neighbor = create_adaptive_mutation(current, iteration, 0.5)  # Reduced mutation factor for local search
        neighbor_c2 = evaluate_c2(neighbor)
        
        if neighbor_c2 > current_c2:
            current = neighbor
            current_c2 = neighbor_c2
            improvement_count = 0
        else:
            improvement_count += 1
            # Reduce mutation rate if no improvement for a while
            if improvement_count > max_no_improvement:
                # Try a different approach - small perturbation
                for i in range(len(current)):
                    if random.random() < 0.15:
                        current[i] = max(0, current[i] + random.uniform(-0.005, 0.005))
                current_c2 = evaluate_c2(current)
                improvement_count = 0
    
    return current

def evolve_step_function() -> List[float]:
    """
    Enhanced evolution with better strategies to maximize C2.
    """
    # Parameters for enhanced genetic algorithm - optimized for speed and quality
    population_size = 150  # Reduced population size for faster computation
    generations = 120      # Fewer generations but better optimization
    elite_size = 15        # Smaller elite to allow more diversity
    
    # Initialize population with diverse strategies
    population = []
    for _ in range(population_size):
        length = random.randint(600, 1000)  # More focused range for efficiency
        strategy = random.random()
        
        if strategy < 0.4:
            # Mathematically informed pattern
            population.append(create_mathematically_informed_pattern(length))
        elif strategy < 0.7:
            # Uniform distribution pattern
            population.append(create_uniform_distribution_pattern(length))
        else:
            # Spiral pattern
            population.append(create_spiral_pattern(length))
    
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
            # Tournament selection with larger k for more pressure
            parent1 = create_improved_tournament_selection(population, fitness_scores, 6, generation)
            parent2 = create_improved_tournament_selection(population, fitness_scores, 6, generation)
            
            # Crossover
            child = create_adaptive_crossover(parent1, parent2, generation)
            
            # Mutation
            child = create_adaptive_mutation(child, generation)
            
            # Occasionally introduce completely new structured individual
            if random.random() < 0.15:
                child_length = random.randint(600, 1000)
                child_strategy = random.random()
                if child_strategy < 0.4:
                    child = create_mathematically_informed_pattern(child_length)
                elif child_strategy < 0.7:
                    child = create_uniform_distribution_pattern(child_length)
                else:
                    child = create_spiral_pattern(child_length)
            
            new_population.append(child)
        
        population = new_population
    
    # Final refinement of best solution
    if best_individual is not None:
        refined = optimize_with_local_search(best_individual, 60)
        refined_c2 = evaluate_c2(refined)
        if refined_c2 > best_fitness:
            return refined
    
    return best_individual if best_individual is not None else create_mathematically_informed_pattern(800)

def construct_function() -> List[float]:
    """Main function to construct optimized step-function with high C2 value."""
    # Try evolutionary approach first
    evoluted = evolve_step_function()
    
    # Try direct optimization approach with better local search
    candidates = [evoluted]
    
    # Add more diverse patterns with mathematical insight
    for i in range(15):  # Even fewer candidates to save time and improve focus
        length = random.randint(600, 1000)
        
        # Create a pattern that combines different behaviors with mathematical care
        combined_pattern = []
        for j in range(length):
            # Mix of periodic and localized behavior with more mathematical structure
            periodic = 0.5 + 0.35 * math.sin(j * 0.06) + 0.18 * math.sin(j * 0.18) + 0.08 * math.sin(j * 0.35)
            localized = 0.12 * math.exp(-((j - length//2)**2) / (length/6)**2)
            # Add some controlled variation
            variation = 0.025 * math.sin(j * 0.25) + 0.015 * math.cos(j * 0.15)
            val = periodic + localized + variation + random.uniform(-0.015, 0.015)
            combined_pattern.append(max(0, val))
        candidates.append(combined_pattern)
        
        # Create a pattern with exponential decay to create a "spike" effect
        spike_pattern = []
        center = length // 2
        for j in range(length):
            # Exponential decay pattern
            dist = abs(j - center)
            val = 0.75 * math.exp(-dist / (length/8)) + 0.15 * math.sin(j * 0.12)
            spike_pattern.append(max(0, val))
        candidates.append(spike_pattern)
        
        # Create a pattern with more structured randomness
        structured_pattern = []
        for j in range(length):
            # Create a pattern with multiple scales
            val1 = 0.65 + 0.2 * math.sin(j * 0.1)
            val2 = 0.12 * math.sin(j * 0.3) + 0.06 * math.cos(j * 0.15)
            val3 = 0.06 * math.sin(j * 0.5) + 0.03 * math.cos(j * 0.35)
            val = val1 + val2 + val3 + random.uniform(-0.025, 0.025)
            structured_pattern.append(max(0, val))
        candidates.append(structured_pattern)
    
    # Evaluate all candidates and return the best
    best_candidate = max(candidates, key=evaluate_c2)
    
    return best_candidate

# Improved initialization strategies with better mathematical foundation
def create_mathematically_optimized_pattern(length: int) -> List[float]:
    """Create pattern based on mathematical insights for maximizing C2."""
    individual = []
    
    # Create a pattern that balances peak height and distribution
    # Based on research: patterns that create flatter autoconvolutions tend to do better
    # We'll use a combination approach
    
    # Base pattern: bell curve with some modulation
    center = length // 2
    sigma = length / 8.0
    
    for i in range(length):
        # Primary bell-shaped component
        val = math.exp(-((i - center)**2) / (2 * sigma**2))
        
        # Add frequency modulation to break symmetry and avoid local optima
        freq_mod = 0.1 * math.sin(i * 0.2) + 0.05 * math.cos(i * 0.15)
        val += 0.1 * freq_mod
        
        # Add some randomness for diversity
        val += random.uniform(-0.02, 0.02)
        
        individual.append(max(0, val))
    
    # Normalize to control magnitude
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 10 for x in individual]
    
    return individual

def create_uniform_distribution_pattern(length: int) -> List[float]:
    """Create pattern with more uniform distribution to encourage flat autoconvolutions."""
    individual = []
    
    # Create a pattern that spreads values more uniformly
    # This should lead to flatter autoconvolutions which benefit C2
    
    # Start with a basic uniform-like pattern
    base_height = 0.5
    
    # Add some variation to avoid trivial solutions
    for i in range(length):
        # Create a pattern with some periodic variation
        variation = 0.2 * math.sin(i * 0.1) + 0.1 * math.cos(i * 0.05)
        val = base_height + variation
        
        # Add small random component
        val += random.uniform(-0.03, 0.03)
        
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 8 for x in individual]
    
    return individual

def create_peak_and_trough_pattern(length: int) -> List[float]:
    """Create pattern with alternating peaks and troughs."""
    individual = []
    
    # Create a pattern with alternating high and low values
    for i in range(length):
        # Alternating pattern
        if i % 2 == 0:
            val = 0.8 + 0.1 * math.sin(i * 0.1)
        else:
            val = 0.2 + 0.05 * math.cos(i * 0.15)
        
        # Add some noise
        val += random.uniform(-0.02, 0.02)
        
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 6 for x in individual]
    
    return individual

def create_spiral_pattern(length: int) -> List[float]:
    """Create a spiral-like pattern that might produce interesting autoconvolutions."""
    individual = []
    
    # Create a pattern that spirals up and down
    for i in range(length):
        # Spiral pattern with some randomization
        angle = i * 0.1
        radius = 0.5 + 0.3 * math.sin(angle) + 0.1 * math.cos(2*angle)
        val = radius + 0.05 * math.sin(3*angle + i * 0.05)
        
        # Add some noise
        val += random.uniform(-0.03, 0.03)
        
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 7 for x in individual]
    
    return individual

def create_adaptive_mutation(individual: List[float], generation: int) -> List[float]:
    """Enhanced mutation with adaptive parameters based on evolution progress."""
    mutated = individual.copy()
    
    # Adaptive mutation rate that decreases over generations
    mutation_rate = max(0.05, 0.2 * (1 - generation / 200))
    
    # Adaptive mutation strength
    mutation_strength = 0.05 + 0.05 * (generation / 200)
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Use different mutation strategies based on value magnitude
            current_val = mutated[i]
            
            # Different mutation strengths for different value ranges
            if current_val < 0.05:
                # Very small values - very small mutation
                delta = np.random.normal(0, 0.003 * mutation_strength)
            elif current_val < 0.2:
                # Small values - small mutation
                delta = np.random.normal(0, 0.01 * mutation_strength)
            elif current_val < 0.5:
                # Medium values - moderate mutation
                delta = np.random.normal(0, 0.03 * mutation_strength)
            elif current_val < 1.0:
                # High values - medium mutation
                delta = np.random.normal(0, 0.05 * mutation_strength)
            else:
                # Very high values - smaller mutation to prevent overshoot
                delta = np.random.normal(0, 0.02 * mutation_strength)
                
            mutated[i] = max(0, current_val + delta)
                
    return mutated

def create_adaptive_crossover(parent1: List[float], parent2: List[float], generation: int) -> List[float]:
    """Adaptive crossover with better blending strategies."""
    child = []
    min_len = min(len(parent1), len(parent2))
    
    # Adaptive crossover probability based on generation
    crossover_prob = 0.7 + 0.1 * (1 - generation / 200)
    
    for i in range(min_len):
        if random.random() < crossover_prob:
            # Use adaptive weights based on parent similarity
            similarity = abs(parent1[i] - parent2[i])
            
            if similarity < 0.05:
                # Very similar parents - blend more conservatively
                alpha = random.uniform(0.4, 0.6)
            elif similarity < 0.2:
                # Moderately different - moderate blending
                alpha = random.uniform(0.3, 0.7)
            else:
                # Very different - more random blending
                alpha = random.random()
                
            blended = alpha * parent1[i] + (1 - alpha) * parent2[i]
            # Add noise proportional to the blended value
            noise = random.gauss(0, 0.01 * blended if blended > 0 else 0.005)
            child.append(max(0, blended + noise))
        else:
            # Uniform crossover with adaptive probabilities
            parent = parent1 if random.random() < 0.55 else parent2
            # Add noise to maintain diversity
            noise = random.gauss(0, 0.005 * parent[i] if parent[i] > 0 else 0.002)
            child.append(max(0, parent[i] + noise))
            
    return child

def create_improved_tournament_selection(population: List[List[float]], 
                                       fitness_scores: List[tuple], 
                                       k: int, generation: int) -> List[float]:
    """Improved tournament selection with better pressure and diversity."""
    tournament_indices = random.sample(range(len(population)), min(k, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    # Sort by fitness descending (higher is better)
    tournament_fitness.sort(key=lambda x: x[1], reverse=True)
    
    # Adaptive selection pressure based on generation
    selection_pressure = 0.5 + 0.3 * (1 - generation / 200)
    
    # Return the best of the tournament, but with some probabilistic element
    # to allow for some diversity
    if len(tournament_fitness) > 1 and random.random() < selection_pressure:
        # Some chance to pick from top 2 instead of just top 1
        return tournament_fitness[random.randint(0, min(1, len(tournament_fitness)-1))][0]
    else:
        return tournament_fitness[0][0]

def optimize_with_local_search(initial_guess: List[float], max_iter: int = 100) -> List[float]:
    """Perform local search refinement around initial guess with improved strategy."""
    current = initial_guess.copy()
    current_c2 = evaluate_c2(current)
    
    # Local search with adaptive step sizes and better termination criteria
    improvement_count = 0
    max_no_improvement = 20
    
    for iteration in range(max_iter):
        # Create neighbor by small perturbations
        neighbor = create_adaptive_mutation(current, iteration)
        neighbor_c2 = evaluate_c2(neighbor)
        
        if neighbor_c2 > current_c2:
            current = neighbor
            current_c2 = neighbor_c2
            improvement_count = 0
        else:
            improvement_count += 1
            # Reduce mutation rate if no improvement for a while
            if improvement_count > max_no_improvement:
                # Try a different approach - small perturbation
                for i in range(len(current)):
                    if random.random() < 0.1:
                        current[i] = max(0, current[i] + random.uniform(-0.01, 0.01))
                current_c2 = evaluate_c2(current)
                improvement_count = 0
    
    return current

def evolve_step_function() -> List[float]:
    """
    Enhanced evolution with better strategies to maximize C2.
    """
    # Parameters for enhanced genetic algorithm
    population_size = 300  # Increased population size
    generations = 200      # More generations for better convergence
    elite_size = 30        # Good number of elites
    
    # Initialize population with diverse strategies
    population = []
    for _ in range(population_size):
        length = random.randint(800, 1500)  # Broader range for exploration
        strategy = random.random()
        
        if strategy < 0.25:
            # Mathematically optimized pattern
            population.append(create_mathematically_optimized_pattern(length))
        elif strategy < 0.5:
            # Uniform distribution pattern
            population.append(create_uniform_distribution_pattern(length))
        elif strategy < 0.75:
            # Peak and trough pattern
            population.append(create_peak_and_trough_pattern(length))
        else:
            # Spiral pattern
            population.append(create_spiral_pattern(length))
    
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
            # Tournament selection with larger k for more pressure
            parent1 = create_improved_tournament_selection(population, fitness_scores, 10, generation)
            parent2 = create_improved_tournament_selection(population, fitness_scores, 10, generation)
            
            # Crossover
            child = create_adaptive_crossover(parent1, parent2, generation)
            
            # Mutation
            child = create_adaptive_mutation(child, generation)
            
            # Occasionally introduce completely new structured individual
            if random.random() < 0.15:
                child_length = random.randint(800, 1500)
                child_strategy = random.random()
                if child_strategy < 0.25:
                    child = create_mathematically_optimized_pattern(child_length)
                elif child_strategy < 0.5:
                    child = create_uniform_distribution_pattern(child_length)
                elif child_strategy < 0.75:
                    child = create_peak_and_trough_pattern(child_length)
                else:
                    child = create_spiral_pattern(child_length)
            
            new_population.append(child)
        
        population = new_population
    
    # Final refinement of best solution
    if best_individual is not None:
        refined = optimize_with_local_search(best_individual, 150)
        refined_c2 = evaluate_c2(refined)
        if refined_c2 > best_fitness:
            return refined
    
    return best_individual if best_individual is not None else create_mathematically_optimized_pattern(1000)

def construct_function() -> List[float]:
    """Main function to construct optimized step-function with high C2 value."""
    # Try evolutionary approach first
    evoluted = evolve_step_function()
    
    # Try direct optimization approach with better local search
    candidates = [evoluted]
    
    # Add more diverse patterns with mathematical insight
    for i in range(30):  # More candidates
        length = random.randint(800, 1500)
        
        # Create a pattern that combines different behaviors with mathematical care
        combined_pattern = []
        for j in range(length):
            # Mix of periodic and localized behavior with more mathematical structure
            periodic = 0.5 + 0.3 * math.sin(j * 0.05) + 0.15 * math.sin(j * 0.15) + 0.05 * math.sin(j * 0.3)
            localized = 0.1 * math.exp(-((j - length//2)**2) / (length/8)**2)
            # Add some controlled variation
            variation = 0.02 * math.sin(j * 0.2) + 0.01 * math.cos(j * 0.1)
            val = periodic + localized + variation + random.uniform(-0.02, 0.02)
            combined_pattern.append(max(0, val))
        candidates.append(combined_pattern)
        
        # Create a pattern with exponential decay to create a "spike" effect
        spike_pattern = []
        center = length // 2
        for j in range(length):
            # Exponential decay pattern
            dist = abs(j - center)
            val = 0.8 * math.exp(-dist / (length/10)) + 0.1 * math.sin(j * 0.1)
            spike_pattern.append(max(0, val))
        candidates.append(spike_pattern)
        
        # Create a pattern with more structured randomness
        structured_pattern = []
        for j in range(length):
            # Create a pattern with multiple scales
            val1 = 0.6 + 0.2 * math.sin(j * 0.08)
            val2 = 0.1 * math.sin(j * 0.25) + 0.05 * math.cos(j * 0.1)
            val3 = 0.05 * math.sin(j * 0.4) + 0.02 * math.cos(j * 0.3)
            val = val1 + val2 + val3 + random.uniform(-0.03, 0.03)
            structured_pattern.append(max(0, val))
        candidates.append(structured_pattern)
        
        # Create a pattern with Fibonacci-like spacing
        fibonacci_pattern = []
        fib_seq = [1, 1]
        while len(fib_seq) < length:
            fib_seq.append(fib_seq[-1] + fib_seq[-2])
        
        for j in range(length):
            # Use Fibonacci sequence for spacing
            if j < len(fib_seq):
                fib_val = fib_seq[j] / max(fib_seq)
            else:
                fib_val = 0.5
                
            # Combine with other patterns
            val = 0.3 + 0.4 * math.sin(j * 0.1) + 0.2 * fib_val + random.uniform(-0.03, 0.03)
            fibonacci_pattern.append(max(0, val))
        candidates.append(fibonacci_pattern)
    
    # Evaluate all candidates and return the best
    best_candidate = max(candidates, key=evaluate_c2)
    
    return best_candidate

def create_optimized_initialization(length: int) -> List[float]:
    """Create optimized initial individuals based on mathematical insights."""
    individual = []
    
    # Strategy: Create a pattern that promotes uniformity in autoconvolution
    # Based on research, patterns that create "broad" autoconvolutions work better
    
    # Create a smooth bell-shaped pattern with careful peak placement
    center = length // 2
    sigma = length / 12  # Narrower than previous to create sharper peaks
    
    for i in range(length):
        # Create a Gaussian-like shape but with slight variations to avoid symmetry issues
        val = math.exp(-((i - center)**2) / (2 * sigma**2))
        
        # Add small periodic component to break symmetry
        val += 0.05 * math.sin(i * 0.15) + 0.02 * math.cos(i * 0.08)
        
        # Add small random component to promote diversity
        val += random.uniform(-0.01, 0.01)
        
        individual.append(max(0, val))
    
    # Normalize to control overall magnitude
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 5 for x in individual]
    
    return individual

def create_balanced_pattern(length: int) -> List[float]:
    """Create a balanced pattern that tries to avoid extreme values."""
    individual = []
    
    # Create a pattern with moderate values that spread out
    for i in range(length):
        # Create a pattern that varies moderately
        t = i / (length - 1) if length > 1 else 0
        # Sigmoid-like curve that starts low, rises, then falls
        val = 1.0 / (1.0 + math.exp(-5*(t - 0.5)))
        # Add some oscillation to avoid being too regular
        val += 0.1 * math.sin(t * 15) + 0.05 * math.cos(t * 8)
        # Add small random variation
        val += random.uniform(-0.03, 0.03)
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 8 for x in individual]
    
    return individual

def create_sine_wave_pattern(length: int) -> List[float]:
    """Create a sine wave pattern with appropriate phase and amplitude."""
    individual = []
    
    # Create a pattern with multiple sine components
    for i in range(length):
        # Combination of fundamental and harmonics
        val = 0.6 + 0.2 * math.sin(i * 0.1) + 0.1 * math.sin(i * 0.25) + 0.05 * math.sin(i * 0.4)
        # Add a bit of damping to reduce extremes
        damping = 1.0 - 0.1 * abs(i - length//2) / (length//2)
        val *= damping
        individual.append(max(0, val + random.gauss(0, 0.02)))
    
    return individual

def create_piecewise_linear_pattern(length: int) -> List[float]:
    """Create a piecewise linear pattern that promotes flat autoconvolutions."""
    individual = []
    
    # Create a pattern with linear segments
    segments = 8
    segment_length = length // segments
    
    for i in range(length):
        segment = i // segment_length
        if segment >= segments:
            segment = segments - 1
            
        # Linear interpolation between key points
        start_val = 0.3 if segment % 2 == 0 else 0.7
        end_val = 0.7 if segment % 2 == 0 else 0.3
        t = (i % segment_length) / segment_length if segment_length > 0 else 0
        val = start_val + t * (end_val - start_val)
        
        # Add some smoothing
        val += 0.05 * math.sin(i * 0.2)
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 6 for x in individual]
    
    return individual

def create_flexible_mutation(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Enhanced mutation with more sophisticated strategies."""
    mutated = individual.copy()
    
    # Apply different mutation strategies based on value characteristics
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            current_val = mutated[i]
            
            # Use adaptive mutation rates based on context
            if current_val < 0.05:
                # Very small values - very small mutation
                mutated[i] = max(0, current_val + np.random.normal(0, 0.005))
            elif current_val < 0.2:
                # Small values - small mutation
                mutated[i] = max(0, current_val + np.random.normal(0, 0.01))
            elif current_val < 0.5:
                # Medium values - moderate mutation
                mutated[i] = max(0, current_val + np.random.normal(0, 0.03))
            elif current_val < 1.0:
                # High values - medium mutation
                mutated[i] = max(0, current_val + np.random.normal(0, 0.05))
            else:
                # Very high values - smaller mutation to prevent overshoot
                mutated[i] = max(0, current_val + np.random.normal(0, 0.02))
                
    return mutated

def create_adaptive_crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Adaptive crossover with better blending strategies."""
    child = []
    min_len = min(len(parent1), len(parent2))
    
    for i in range(min_len):
        # Use adaptive crossover based on parent similarity
        similarity = abs(parent1[i] - parent2[i])
        
        if random.random() < 0.7:
            # Blend with adaptive weights
            if similarity < 0.1:
                # Similar parents - blend more conservatively
                alpha = random.uniform(0.4, 0.6)
            elif similarity < 0.3:
                # Moderately different - moderate blending
                alpha = random.uniform(0.3, 0.7)
            else:
                # Very different - more random blending
                alpha = random.random()
                
            blended = alpha * parent1[i] + (1 - alpha) * parent2[i]
            # Add noise proportional to the blended value
            noise = random.gauss(0, 0.02 * blended if blended > 0 else 0.005)
            child.append(max(0, blended + noise))
        else:
            # Uniform crossover with adaptive probabilities
            parent = parent1 if random.random() < 0.55 else parent2
            # Add noise to maintain diversity
            noise = random.gauss(0, 0.01 * parent[i] if parent[i] > 0 else 0.005)
            child.append(max(0, parent[i] + noise))
            
    return child

def create_improved_tournament_selection(population: List[List[float]], 
                                       fitness_scores: List[tuple], 
                                       k: int) -> List[float]:
    """Improved tournament selection with better pressure and diversity."""
    tournament_indices = random.sample(range(len(population)), min(k, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    # Sort by fitness descending (higher is better)
    tournament_fitness.sort(key=lambda x: x[1], reverse=True)
    # Return the best of the tournament, but with some probabilistic element
    # to allow for some diversity
    if len(tournament_fitness) > 1 and random.random() < 0.3:
        # 30% chance to pick from top 2 instead of just top 1
        return tournament_fitness[random.randint(0, min(1, len(tournament_fitness)-1))][0]
    else:
        return tournament_fitness[0][0]

def optimize_with_local_search(initial_guess: List[float], max_iter: int = 50) -> List[float]:
    """Perform local search refinement around initial guess."""
    current = initial_guess.copy()
    current_c2 = evaluate_c2(current)
    
    # Local search with adaptive step sizes
    for iteration in range(max_iter):
        # Create neighbor by small perturbations
        neighbor = create_flexible_mutation(current, 0.15)
        neighbor_c2 = evaluate_c2(neighbor)
        
        if neighbor_c2 > current_c2:
            current = neighbor
            current_c2 = neighbor_c2
        else:
            # Reduce mutation rate to fine-tune if no improvement
            if iteration % 10 == 0 and iteration > 0:
                # Try smaller mutations occasionally
                neighbor = create_flexible_mutation(current, 0.05)
                neighbor_c2 = evaluate_c2(neighbor)
                if neighbor_c2 > current_c2:
                    current = neighbor
                    current_c2 = neighbor_c2
    
    return current

def evolve_step_function() -> List[float]:
    """
    Enhanced evolution with better strategies to maximize C2.
    """
    # Parameters for enhanced genetic algorithm
    population_size = 250  # Balanced population size
    generations = 150      # More generations for better convergence
    mutation_rate = 0.12   # Moderate mutation rate
    elite_size = 25        # Good number of elites
    
    # Initialize population with diverse strategies
    population = []
    for _ in range(population_size):
        length = random.randint(600, 1200)  # More focused range
        strategy = random.random()
        
        if strategy < 0.2:
            # Optimized initialization
            population.append(create_optimized_initialization(length))
        elif strategy < 0.4:
            # Balanced patterns
            population.append(create_balanced_pattern(length))
        elif strategy < 0.6:
            # Sine wave patterns
            population.append(create_sine_wave_pattern(length))
        elif strategy < 0.8:
            # Piecewise linear patterns
            population.append(create_piecewise_linear_pattern(length))
        else:
            # Standard advanced initialization
            population.append(create_optimized_initialization(length))
    
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
            # Tournament selection with larger k for more pressure
            parent1 = create_improved_tournament_selection(population, fitness_scores, 8)
            parent2 = create_improved_tournament_selection(population, fitness_scores, 8)
            
            # Crossover
            child = create_adaptive_crossover(parent1, parent2)
            
            # Mutation
            child = create_flexible_mutation(child, mutation_rate)
            
            # Occasionally introduce completely new structured individual
            if random.random() < 0.1:
                child_length = random.randint(600, 1200)
                child_strategy = random.random()
                if child_strategy < 0.25:
                    child = create_optimized_initialization(child_length)
                elif child_strategy < 0.5:
                    child = create_balanced_pattern(child_length)
                elif child_strategy < 0.75:
                    child = create_sine_wave_pattern(child_length)
                else:
                    child = create_piecewise_linear_pattern(child_length)
            
            new_population.append(child)
        
        population = new_population
    
    # Final refinement of best solution
    if best_individual is not None:
        refined = optimize_with_local_search(best_individual, 100)
        refined_c2 = evaluate_c2(refined)
        if refined_c2 > best_fitness:
            return refined
    
    return best_individual if best_individual is not None else create_optimized_initialization(1000)

def construct_function() -> List[float]:
    """Main function to construct optimized step-function with high C2 value."""
    # Try evolutionary approach first
    evoluted = evolve_step_function()
    
    # Try direct optimization approach with better local search
    candidates = [evoluted]
    
    # Add more diverse patterns with mathematical insight
    for i in range(20):
        length = random.randint(600, 1200)
        
        # Create a pattern that combines different behaviors with mathematical care
        combined_pattern = []
        for j in range(length):
            # Mix of periodic and localized behavior with more mathematical structure
            periodic = 0.5 + 0.3 * math.sin(j * 0.05) + 0.15 * math.sin(j * 0.15) + 0.05 * math.sin(j * 0.3)
            localized = 0.1 * math.exp(-((j - length//2)**2) / (length/8)**2)
            # Add some controlled variation
            variation = 0.02 * math.sin(j * 0.2) + 0.01 * math.cos(j * 0.1)
            val = periodic + localized + variation + random.uniform(-0.02, 0.02)
            combined_pattern.append(max(0, val))
        candidates.append(combined_pattern)
        
        # Create a pattern with exponential decay to create a "spike" effect
        spike_pattern = []
        center = length // 2
        for j in range(length):
            # Exponential decay pattern
            dist = abs(j - center)
            val = 0.8 * math.exp(-dist / (length/10)) + 0.1 * math.sin(j * 0.1)
            spike_pattern.append(max(0, val))
        candidates.append(spike_pattern)
        
        # Create a pattern with more structured randomness
        structured_pattern = []
        for j in range(length):
            # Create a pattern with multiple scales
            val1 = 0.6 + 0.2 * math.sin(j * 0.08)
            val2 = 0.1 * math.sin(j * 0.25) + 0.05 * math.cos(j * 0.1)
            val3 = 0.05 * math.sin(j * 0.4) + 0.02 * math.cos(j * 0.3)
            val = val1 + val2 + val3 + random.uniform(-0.03, 0.03)
            structured_pattern.append(max(0, val))
        candidates.append(structured_pattern)
    
    # Evaluate all candidates and return the best
    best_candidate = max(candidates, key=evaluate_c2)
    
    return best_candidate

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
