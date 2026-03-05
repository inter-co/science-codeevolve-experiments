# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution, minimize
import time
from numba import jit
import warnings
from functools import lru_cache
import copy
from deap import base, creator, tools, algorithms
import multiprocessing as mp
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True)
def fast_convolution_jit(a):
    """Fast convolution implementation using Numba JIT compilation."""
    n = len(a)
    result = np.zeros(2*n - 1)
    
    # Manual convolution for speed
    for i in range(n):
        for j in range(n):
            result[i + j] += a[i] * a[j]
    
    return result

def compute_c1_fft(sequence):
    """Fast computation of C1 for a given sequence using FFT-based convolution."""
    if len(sequence) == 0:
        return float('inf')
    
    # Convert to numpy array for efficient computation
    a = np.array(sequence, dtype=np.float64)
    
    # Compute convolution (auto-correlation) efficiently using FFT
    # For better numerical stability, use fftconvolve
    conv = fftconvolve(a, a, mode='full')
    
    # Maximum value in convolution (excluding the zeroth element which is sum of squares)
    max_conv = np.max(conv[1:]) if len(conv) > 1 else 0
    
    # Sum of sequence squared
    sum_sq = np.sum(a)**2
    
    # Avoid division by zero
    if sum_sq < 1e-12:
        return float('inf')
    
    # C1 = 2n * max_conv / sum_sq
    n = len(sequence)
    c1 = 2 * n * max_conv / sum_sq
    
    return c1

def compute_c1(sequence):
    """Compute C1 for a given sequence."""
    if len(sequence) == 0:
        return float('inf')
    
    # Compute convolution (auto-correlation) efficiently using FFT
    a = np.array(sequence, dtype=np.float64)
    conv = fftconvolve(a, a, mode='full')
    
    # Maximum value in convolution (excluding the zeroth element which is sum of squares)
    max_conv = np.max(conv[1:]) if len(conv) > 1 else 0
    
    # Sum of sequence squared
    sum_sq = np.sum(a)**2
    
    # Avoid division by zero
    if sum_sq < 1e-12:
        return float('inf')
    
    # C1 = 2n * max_conv / sum_sq
    n = len(sequence)
    c1 = 2 * n * max_conv / sum_sq
    
    return c1

def inv_c1_objective(sequence):
    """Objective function to maximize 1/C1 (minimize C1)."""
    # Ensure minimum sum constraint
    if np.sum(sequence) < 0.01:
        return 0  # Invalid - return very small value
    
    c1 = compute_c1(sequence)
    if c1 == float('inf') or np.isnan(c1):
        return 0  # Return very small value if invalid
    return 1.0 / c1

def generate_random_sequence(length=None, method='uniform', seed=None):
    """Generate a random sequence with specified length."""
    if seed is not None:
        random.seed(seed)
        
    if length is None:
        length = random.randint(30, 500)
    
    if method == 'gaussian':
        # Generate normally distributed heights
        sequence = [max(0, random.gauss(500, 200)) for _ in range(length)]
    else:
        # Generate uniform random heights between 0 and 1000
        sequence = [random.uniform(0, 1000) for _ in range(length)]
    
    return sequence

def optimize_with_local_search(initial_sequence, max_iter=1000):
    """Use local search optimization to improve an initial sequence."""
    current_seq = np.array(initial_sequence, dtype=float)
    current_inv_c1 = inv_c1_objective(current_seq)
    
    # Gradient-like approach with random perturbations
    for iteration in range(max_iter):
        # Make small random changes
        test_seq = current_seq.copy()
        
        # Randomly modify some elements
        num_changes = max(1, len(test_seq) // 10)
        for _ in range(num_changes):
            idx = random.randint(0, len(test_seq) - 1)
            # Small perturbation with adaptive magnitude
            perturbation = random.uniform(-20, 20)
            test_seq[idx] = max(0, test_seq[idx] + perturbation)
        
        # Accept if better
        test_inv_c1 = inv_c1_objective(test_seq)
        if test_inv_c1 > current_inv_c1:
            current_seq = test_seq
            current_inv_c1 = test_inv_c1
            
    return current_seq.tolist()

def generate_peak_sequence(n, peak_height, peak_position=None):
    """Generate a sequence with a single peak."""
    if peak_position is None:
        peak_position = n // 2
    
    sequence = []
    for i in range(n):
        # Gaussian-like decay from peak
        distance = abs(i - peak_position)
        # Use sharper decay for better optimization
        decay_factor = np.exp(-distance**2 / (2 * (n/15)**2))
        height = max(0, peak_height * decay_factor)
        sequence.append(height)
    
    return sequence

def generate_optimized_pattern(n):
    """Generate an optimized pattern based on mathematical insights."""
    # Create a sequence that balances mass concentration and spread
    # Try to create a pattern similar to what's known to work well
    sequence = []
    
    # Start with a symmetric pattern
    mid = n // 2
    
    # Create a bell-shaped curve with better control
    peak_height = random.uniform(800, 1000)
    std_dev = n / 8.0  # Standard deviation for bell shape
    
    for i in range(n):
        distance = abs(i - mid)
        # Gaussian-like decay with sharper peak
        decay = np.exp(-distance**2 / (2 * std_dev**2))
        height = max(0, peak_height * decay)
        sequence.append(height)
    
    return sequence

def generate_balanced_sequence(n):
    """Generate a sequence with balanced distribution that avoids extreme concentrations."""
    sequence = []
    # Create a sequence that avoids having too much mass concentrated at one point
    # This helps avoid high convolution peaks
    
    # Start with a base sequence and apply transformations
    base_height = random.uniform(400, 800)
    for i in range(n):
        # Add some randomness but keep it relatively smooth
        variation = random.uniform(-100, 100)
        height = max(0, base_height + variation)
        sequence.append(height)
    
    return sequence

def generate_geometric_decay_sequence(n):
    """Generate a geometric decay sequence."""
    sequence = []
    base = random.uniform(0.85, 0.95)
    start_height = random.uniform(600, 1000)
    for i in range(n):
        height = max(0, start_height * (base ** i))
        sequence.append(height)
    return sequence

def generate_sine_wave_sequence(n):
    """Generate a sine wave-like sequence."""
    sequence = []
    amplitude = random.uniform(500, 1000)
    frequency = random.uniform(0.05, 0.2)
    phase = random.uniform(0, 2*np.pi)
    for i in range(n):
        value = amplitude * np.sin(2 * np.pi * frequency * i + phase)
        sequence.append(max(0, value))
    return sequence

def generate_combination_pattern(n):
    """Generate a combination of patterns."""
    # Mix different patterns to get potentially better results
    sequence = [0.0] * n
    
    # Add some peaks
    for _ in range(random.randint(2, 5)):
        pos = random.randint(0, n-1)
        height = random.uniform(600, 1000)
        sequence[pos] = max(sequence[pos], height)
    
    # Add some decay component
    decay_base = random.uniform(0.85, 0.95)
    decay_start = random.uniform(500, 800)
    for i in range(n):
        if sequence[i] == 0:
            sequence[i] = max(0, decay_start * (decay_base ** i))
    
    # Add some randomness
    for i in range(n):
        if random.random() < 0.1:
            sequence[i] = max(0, sequence[i] + random.uniform(-100, 100))
    
    return sequence

def optimize_with_evolutionary_algorithm(pop_size=30, generations=20):
    """Use evolutionary algorithm to optimize sequence with better configuration."""
    # Define the optimization problem
    def eval_individual(individual):
        # Convert individual to sequence
        sequence = list(individual)
        # Ensure sequence has valid values
        sequence = [max(0, x) for x in sequence]
        # Compute objective
        return inv_c1_objective(sequence),
    
    # Create the evolutionary algorithm framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1000)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=100)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=100, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run evolution with better parameters
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    try:
        population, logbook = algorithms.eaSimple(
            population, toolbox, cxpb=0.8, mutpb=0.3, 
            ngen=generations, stats=stats, halloffame=hof, verbose=False
        )
    except:
        # Fallback if evolution fails
        return generate_random_sequence(100)
    
    if hof:
        best_individual = hof[0]
        return list(best_individual)
    else:
        return generate_random_sequence(100)

def generate_mathematically_optimized_sequence(n):
    """
    Generate a sequence with mathematical properties designed to minimize C1.
    Based on research that suggests sequences with certain symmetries and decay patterns
    tend to perform better.
    """
    # Create a sequence with a more structured approach
    # Using a modified Gaussian pattern that's been shown to work well
    sequence = []
    
    # Peak location near center
    peak_pos = n // 2
    peak_height = 1000.0  # Max height
    
    # Standard deviation chosen to balance concentration and spread
    sigma = n / 6.0
    
    for i in range(n):
        # Modified Gaussian decay
        distance = abs(i - peak_pos)
        # Use a slightly sharper decay to concentrate mass
        decay = np.exp(-distance**2 / (2 * sigma**2))
        # Scale to ensure good overall mass distribution
        height = peak_height * decay * (1.0 - 0.2 * (distance / (n//2)))
        sequence.append(max(0, height))
    
    # Normalize to ensure sufficient sum
    total = sum(sequence)
    if total < 0.01:
        # If sum is too small, scale up
        scale = 0.01 / max(total, 1e-10)
        sequence = [x * scale for x in sequence]
    
    return sequence

def generate_high_performance_pattern(n):
    """
    Generate a high-performance pattern inspired by known good configurations.
    These patterns often involve careful balancing of mass distribution.
    """
    sequence = []
    
    # Create a pattern that has a peak and then decays smoothly
    # but also maintains some uniformity to avoid creating spikes
    
    # Start with a base peak
    peak_height = random.uniform(700, 1000)
    peak_position = random.randint(n//4, 3*n//4)
    
    # Decay factor - adjust to create the right balance
    decay_factor = 0.95
    
    for i in range(n):
        distance_from_peak = abs(i - peak_position)
        # Exponential decay with some randomness
        decay = decay_factor ** distance_from_peak
        # Add some variance to make it less regular
        variance = 0.8 + random.uniform(0, 0.4)  # Between 0.8 and 1.2
        height = max(0, peak_height * decay * variance)
        sequence.append(height)
    
    # Normalize to ensure minimum sum
    total = sum(sequence)
    if total < 0.01:
        scale = 0.01 / max(total, 1e-10)
        sequence = [x * scale for x in sequence]
    
    return sequence

def adaptive_search():
    """Adaptive search combining multiple strategies with better optimization."""
    start_time = time.time()
    max_time = 55  # Leave some buffer for final processing
    
    best_inv_c1 = 0
    best_sequence = None
    
    # Strategy 1: Mathematically optimized patterns
    for attempt in range(150):  # Reduced to allow more time for other strategies
        if time.time() - start_time > max_time:
            break
            
        # Try different sequence lengths
        seq_length = random.randint(30, 500)
        
        # Focus on mathematically optimized patterns
        pattern_type = random.choice([
            generate_mathematically_optimized_sequence,
            generate_high_performance_pattern,
            generate_optimized_pattern,
            generate_balanced_sequence
        ])
        
        sequence = pattern_type(seq_length)
        
        # Local optimization
        optimized_seq = optimize_with_local_search(sequence)
        inv_c1 = inv_c1_objective(optimized_seq)
        
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = optimized_seq.copy()
    
    # Strategy 2: Systematic search around promising regions
    if time.time() - start_time < max_time and best_sequence is not None:
        # Try variations of the best sequence with more focused modifications
        for _ in range(75):
            if time.time() - start_time > max_time:
                break
                
            # Create a variant of the best sequence
            variant = best_sequence.copy()
            for i in range(len(variant)):
                # Apply more intelligent modifications
                if random.random() < 0.25:  # 25% chance to modify
                    # More targeted approach
                    if variant[i] > 0:
                        # Use proportional changes but limit the range
                        change = random.uniform(-variant[i]*0.2, variant[i]*0.2)
                        variant[i] = max(0, variant[i] + change)
                    else:
                        # For zero values, add some value
                        variant[i] = max(0, variant[i] + random.uniform(0, 200))
            
            variant = optimize_with_local_search(variant, max_iter=300)
            inv_c1 = inv_c1_objective(variant)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = variant.copy()
    
    # Strategy 3: Evolutionary algorithm with better parameters
    if time.time() - start_time < max_time and best_sequence is not None:
        try:
            # Run evolutionary algorithm with more generations and better population size
            ea_sequence = optimize_with_evolutionary_algorithm(pop_size=40, generations=25)
            inv_c1 = inv_c1_objective(ea_sequence)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = ea_sequence.copy()
        except:
            pass
    
    # Strategy 4: Targeted refinement of best solution with more sophisticated approach
    if time.time() - start_time < max_time and best_sequence is not None:
        # Use a more sophisticated optimization approach
        refined_seq = best_sequence.copy()
        
        # Multiple rounds of fine-tuning with varying intensities
        for round_num in range(2):
            if time.time() - start_time > max_time:
                break
                
            # Try to improve by adjusting elements with more targeted approach
            for _ in range(1500):  # More iterations per round
                if time.time() - start_time > max_time:
                    break
                    
                test_seq = refined_seq.copy()
                
                # Modify a few elements at a time - but be more selective
                num_modifications = random.randint(1, min(10, len(test_seq)//4))
                for _ in range(num_modifications):
                    idx = random.randint(0, len(test_seq)-1)
                    # More sophisticated perturbations
                    if test_seq[idx] > 0:
                        # Use a more precise change based on the value
                        change_range = min(100, test_seq[idx] * 0.15)  # Cap at 100 or 15% of value
                        change = random.uniform(-change_range, change_range)
                        test_seq[idx] = max(0, test_seq[idx] + change)
                    else:
                        # For zero values, add some value
                        test_seq[idx] = max(0, test_seq[idx] + random.uniform(0, 150))
                
                test_inv_c1 = inv_c1_objective(test_seq)
                if test_inv_c1 > inv_c1_objective(refined_seq):
                    refined_seq = test_seq
        
        inv_c1 = inv_c1_objective(refined_seq)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined_seq.copy()
    
    # Strategy 5: Hybrid approach with direct optimization
    if time.time() - start_time < max_time and best_sequence is not None:
        # Try a few different starting points near the best found
        for _ in range(15):
            if time.time() - start_time > max_time:
                break
                
            # Create a slight perturbation of the best sequence
            candidate = best_sequence.copy()
            for i in range(len(candidate)):
                if random.random() < 0.15:  # 15% chance to modify
                    # More targeted changes
                    change = random.uniform(-candidate[i]*0.15, candidate[i]*0.15) if candidate[i] > 0 else random.uniform(0, 150)
                    candidate[i] = max(0, candidate[i] + change)
            
            candidate = optimize_with_local_search(candidate, max_iter=1500)
            inv_c1 = inv_c1_objective(candidate)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = candidate.copy()
    
    # Strategy 6: Direct mathematical optimization for small sequences
    if time.time() - start_time < max_time:
        # Try optimizing smaller sequences directly with more precise methods
        for _ in range(30):
            if time.time() - start_time > max_time:
                break
                
            # Generate small sequences and optimize them directly
            n = random.randint(20, 100)
            sequence = generate_mathematically_optimized_sequence(n)
            
            # Try to optimize with a more direct approach
            optimized = optimize_with_local_search(sequence, max_iter=2000)
            inv_c1 = inv_c1_objective(optimized)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = optimized.copy()
    
    # Return best result or fallback
    return best_sequence if best_sequence is not None else generate_random_sequence(100)

def search_for_best_sequence():
    """Main search function with improved strategies."""
    start_time = time.time()
    
    # Try the adaptive search approach
    try:
        sequence = adaptive_search()
    except Exception as e:
        # Fallback to simple approach if anything goes wrong
        sequence = generate_random_sequence(100)
    
    # Final optimization pass with more thorough search
    try:
        optimized_sequence = optimize_with_local_search(sequence, max_iter=2000)
        final_inv_c1 = inv_c1_objective(optimized_sequence)
        original_inv_c1 = inv_c1_objective(sequence)
        
        if final_inv_c1 > original_inv_c1:
            sequence = optimized_sequence
    except Exception:
        pass
    
    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
