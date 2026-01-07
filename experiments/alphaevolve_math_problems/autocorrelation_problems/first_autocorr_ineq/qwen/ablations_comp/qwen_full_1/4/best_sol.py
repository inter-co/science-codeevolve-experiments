# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import convolve
from scipy.fft import fft, ifft
import time
from typing import List, Tuple
import math
from numba import jit
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def fast_autoconvolve(a):
    """Fast autoconvolution using FFT - JIT compiled for speed"""
    n = len(a)
    # Use FFT for efficient convolution
    fa = fft(a, 2*n-1)
    result = ifft(fa * fa.conj()).real
    return result[:n]

def compute_autocorrelation_constant(sequence: List[float]) -> Tuple[float, float]:
    """
    Compute C1 = 2n * max(convolution) / (sum(sequence))^2
    Returns (C1, 1/C1) where we want to maximize 1/C1
    """
    if len(sequence) == 0:
        return float('inf'), 0.0
    
    # Convert to numpy array
    a = np.array(sequence, dtype=np.float64)
    n = len(a)
    
    # Compute autoconvolution using FFT for efficiency
    try:
        # Use the fast autoconvolve function
        conv_result = fast_autoconvolve(a)
        # The maximum value in the autoconvolution
        max_conv = np.max(conv_result)
    except:
        # Fallback to standard convolution if FFT fails
        conv_result = convolve(a, a, mode='full')[:2*n-1]
        max_conv = np.max(conv_result)
    
    # Sum of the sequence
    sum_a = np.sum(a)
    
    if sum_a < 1e-10:
        return float('inf'), 0.0
    
    # Compute C1
    C1 = (2 * n * max_conv) / (sum_a ** 2)
    
    # Return both C1 and its reciprocal
    return C1, 1.0 / C1 if C1 > 0 else 0.0

def generate_random_sequence(length: int, min_height: float = 0.01, max_height: float = 1000.0) -> List[float]:
    """Generate a random sequence with specified length and height constraints"""
    # Use more sophisticated approach inspired by INSPIRATION 3
    sequence = []
    for i in range(length):
        # Use a mixture of distributions to create more diverse and potentially better sequences
        if random.random() < 0.2:
            # Heavy tail distribution for outliers
            val = random.expovariate(0.1) * random.choice([0.5, 1.0, 2.0, 5.0])
        elif random.random() < 0.4:
            # Log-normal distribution for more structured sequences
            val = np.exp(random.gauss(0, 0.5))
        else:
            # Normal distribution for bulk
            val = random.gauss(1.0, 0.7)
        sequence.append(max(min_height, min(val, max_height)))
    return sequence

def mutate_sequence(sequence: List[float], mutation_rate: float = 0.1, 
                   mutation_strength: float = 0.3) -> List[float]:
    """Create a mutated version of the sequence"""
    new_sequence = sequence.copy()
    for i in range(len(new_sequence)):
        if random.random() < mutation_rate:
            # Apply multiplicative mutation with better control - inspired by INSPIRATION 3
            # Use different strategies for better exploration
            strategy = random.random()
            if strategy < 0.5:
                # Standard multiplicative mutation
                mutation_factor = random.uniform(0.5, 2.0)
                new_value = new_sequence[i] * mutation_factor
            elif strategy < 0.8:
                # Additive mutation with larger variance
                delta = random.gauss(0, mutation_strength * new_sequence[i])
                new_value = new_sequence[i] + delta
            else:
                # Large jump mutation
                new_value = random.uniform(0.01, 1000.0)
            # Ensure non-negativity and reasonable bounds
            new_sequence[i] = max(0.01, min(1000.0, new_value))
    return new_sequence

def crossover_sequences(seq1: List[float], seq2: List[float]) -> Tuple[List[float], List[float]]:
    """Perform uniform crossover between two sequences"""
    # Use more sophisticated crossover that maintains good properties
    if len(seq1) != len(seq2):
        # If lengths differ, make them the same length by padding or truncating
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
    
    # Blend crossover with some probability
    child1, child2 = [], []
    for i in range(len(seq1)):
        if random.random() < 0.5:
            child1.append(seq1[i])
            child2.append(seq2[i])
        else:
            child1.append(seq2[i])
            child2.append(seq1[i])
    
    return child1, child2

def evaluate_sequence_fitness(sequence: List[float]) -> float:
    """
    Evaluate fitness of a sequence - we want to maximize 1/C1
    Returns fitness value (higher is better)
    """
    try:
        C1, inv_C1 = compute_autocorrelation_constant(sequence)
        # Return inverse of C1 as fitness (we want to maximize this)
        # Add penalty for very small sequences to encourage larger ones
        if len(sequence) < 10:
            inv_C1 *= 0.5
        # Additional penalty for sequences that are too uniform
        if len(sequence) > 5:
            std_dev = np.std(sequence)
            mean_val = np.mean(sequence)
            if mean_val > 0:
                cv = std_dev / mean_val  # Coefficient of variation
                if cv < 0.1:  # Very low variance - penalize
                    inv_C1 *= 0.8
        return inv_C1
    except Exception:
        return 0.0

def generate_bell_pattern(length: int) -> List[float]:
    """Generate a bell-shaped pattern that often performs well - from INSPIRATION 1"""
    sequence = []
    center = length // 2
    for i in range(length):
        # Create a Gaussian-like pattern centered in the middle
        distance_from_center = abs(i - center)
        # Gaussian decay with a slight bump at the center
        value = 100.0 * np.exp(-0.003 * distance_from_center * distance_from_center)
        # Add a peak at center
        if distance_from_center <= 2:
            value += 50.0
        # Add some randomness
        sequence.append(max(0.01, value * random.uniform(0.9, 1.1)))
    return sequence

def generate_geometric_sequence(length: int) -> List[float]:
    """Generate a geometric decay sequence"""
    sequence = []
    decay_factor = 0.93  # Slightly faster decay than previous
    for i in range(length):
        value = 100.0 * (decay_factor ** i)
        sequence.append(max(0.01, min(1000.0, value)))
    return sequence

def generate_power_law_sequence(length: int) -> List[float]:
    """Generate a power-law decay sequence"""
    sequence = []
    alpha = 1.5  # Power law exponent
    for i in range(length):
        if i == 0:
            value = 100.0
        else:
            value = 100.0 / (i ** alpha)
        sequence.append(max(0.01, min(1000.0, value)))
    return sequence

def generate_fibonacci_sequence(length: int) -> List[float]:
    """Generate a Fibonacci-like sequence"""
    sequence = []
    if length <= 0:
        return sequence
    elif length == 1:
        sequence.append(100.0)
    elif length == 2:
        sequence.extend([100.0, 100.0])
    else:
        sequence.extend([100.0, 100.0])
        for i in range(2, length):
            next_val = sequence[i-1] + sequence[i-2] * 0.8  # Slight damping
            sequence.append(max(0.01, min(1000.0, next_val)))
    return sequence

def generate_sparse_peak_sequence(length: int) -> List[float]:
    """Generate a sequence with sparse, strategically placed high peaks"""
    sequence = [0.0] * length
    # Place peaks sparsely to reduce convolution interference
    num_peaks = max(1, length // 20)  # Fewer peaks for longer sequences
    peak_positions = random.sample(range(length), min(num_peaks, length))
    for pos in peak_positions:
        # Peaks are high but not too concentrated
        sequence[pos] = 1000.0 + random.uniform(0, 500.0)
    # Smooth the transitions to reduce sharp peaks that cause high convolution
    for i in range(length):
        if sequence[i] > 0:
            # Apply smoothing to reduce sharp peaks
            smoothed = sequence[i] * 0.9
            # Apply some randomness to break perfect patterns
            sequence[i] = max(0.01, smoothed * random.uniform(0.95, 1.05))
    return sequence

def generate_spiral_sequence(length: int) -> List[float]:
    """Generate a spiral-like pattern that tends to reduce convolution peaks"""
    sequence = []
    for i in range(length):
        # Create a pattern that oscillates and decays
        oscillation = 1.0 + 0.5 * np.sin(i * 0.2)
        decay = np.exp(-i * 0.01)
        value = 100.0 * oscillation * decay
        sequence.append(max(0.01, min(1000.0, value * random.uniform(0.85, 1.15))))
    return sequence

def generate_inverse_quadratic_sequence(length: int) -> List[float]:
    """Generate inverse quadratic decay sequence - inspired by mathematical optimization"""
    sequence = []
    center = length // 2
    for i in range(length):
        # Inverse quadratic pattern centered at middle
        value = 1000.0 / (1.0 + ((i - center) / (length/4))**2)
        sequence.append(value)
    return sequence

def generate_logistic_sequence(length: int) -> List[float]:
    """Generate logistic decay sequence - inspired by mathematical optimization"""
    sequence = []
    center = length // 2
    for i in range(length):
        # Logistic decay pattern
        value = 1000.0 / (1.0 + np.exp(-(i - center) / (length/10)))
        sequence.append(value)
    return sequence

def generate_double_exponential_sequence(length: int) -> List[float]:
    """Generate double exponential pattern"""
    sequence = []
    center = length // 2
    for i in range(length):
        # Double exponential pattern
        value = 1000.0 * np.exp(-abs(i - center)/10.0) * np.exp(-(i-center)**2/100.0)
        sequence.append(value)
    return sequence

def generate_multiscale_sequence(length: int) -> List[float]:
    """Generate a multi-scale pattern with harmonic components"""
    sequence = []
    for i in range(length):
        # Combines multiple frequencies for better autocorrelation properties
        val = 1000.0 * (0.5 + 0.3 * np.sin(2 * np.pi * i / (length/3.0)) + 
                       0.2 * np.sin(2 * np.pi * i / (length/7.0)) + 
                       0.1 * np.sin(2 * np.pi * i / (length/11.0)))
        sequence.append(max(0.1, val))
    return sequence

def enhanced_genetic_algorithm_search(max_time_seconds: float = 90.0) -> List[float]:
    """
    Enhanced genetic algorithm with better strategies from inspiration programs
    """
    start_time = time.time()
    
    # Parameters optimized for better performance
    population_size = 250
    generations = 500
    elite_size = 35
    mutation_rate = 0.2
    tournament_size = 10
    min_length = 30
    max_length = 800
    
    # Initialize population with diverse approaches from INSPIRATION 1 & 2
    population = []
    
    # Add mathematical sequences (more varied)
    for _ in range(8):
        n = random.randint(50, 200)
        individual = generate_bell_pattern(n)
        population.append(individual)
    
    for _ in range(8):
        n = random.randint(50, 200)
        individual = generate_geometric_sequence(n)
        population.append(individual)
    
    for _ in range(8):
        n = random.randint(50, 200)
        individual = generate_power_law_sequence(n)
        population.append(individual)
    
    for _ in range(8):
        n = random.randint(50, 200)
        individual = generate_fibonacci_sequence(n)
        population.append(individual)
    
    for _ in range(8):
        n = random.randint(50, 200)
        individual = generate_sparse_peak_sequence(n)
        population.append(individual)
    
    # Add more advanced patterns from INSPIRATION 1
    for _ in range(5):
        n = random.randint(50, 200)
        individual = generate_spiral_sequence(n)
        population.append(individual)
    
    for _ in range(5):
        n = random.randint(50, 200)
        individual = generate_inverse_quadratic_sequence(n)
        population.append(individual)
    
    for _ in range(5):
        n = random.randint(50, 200)
        individual = generate_logistic_sequence(n)
        population.append(individual)
    
    for _ in range(5):
        n = random.randint(50, 200)
        individual = generate_double_exponential_sequence(n)
        population.append(individual)
    
    for _ in range(5):
        n = random.randint(50, 200)
        individual = generate_multiscale_sequence(n)
        population.append(individual)
    
    # Add some known good patterns as starting points
    for _ in range(10):
        # Uniform pattern
        length = random.randint(50, 200)
        individual = [1.0] * length
        population.append(individual)
    
    # Add some random sequences
    for _ in range(population_size - 60):
        # Mix of different initialization strategies
        strategy = random.random()
        if strategy < 0.15:
            # Random sequences
            length = random.randint(min_length, max_length)
            individual = generate_random_sequence(length)
        elif strategy < 0.30:
            # Pattern-based sequences
            length = random.randint(min_length, max_length)
            individual = generate_power_law_sequence(length)
        elif strategy < 0.45:
            # Fibonacci sequences
            length = random.randint(min_length, max_length)
            individual = generate_fibonacci_sequence(length)
        elif strategy < 0.60:
            # Geometric sequences
            length = random.randint(min_length, max_length)
            individual = generate_geometric_sequence(length)
        elif strategy < 0.75:
            # Spiral sequences
            length = random.randint(min_length, max_length)
            individual = generate_spiral_sequence(length)
        elif strategy < 0.90:
            # Multiscale sequences
            length = random.randint(min_length, max_length)
            individual = generate_multiscale_sequence(length)
        else:
            # Sparse peak sequences
            length = random.randint(min_length, max_length)
            individual = generate_sparse_peak_sequence(length)
        population.append(individual)
    
    best_individual = None
    best_fitness = 0.0
    
    # Evolution loop with enhanced diversity and adaptation
    for generation in range(generations):
        if time.time() - start_time > max_time_seconds * 0.95:  # Leave some buffer
            break
            
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            fitness = evaluate_sequence_fitness(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness (descending)
        fitness_scores.sort(reverse=True)
        
        # Update best individual
        current_best_fitness, current_best_individual = fitness_scores[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()
        
        # Selection using tournament selection
        new_population = []
        
        # Elitism - keep best individuals
        for i in range(elite_size):
            new_population.append(fitness_scores[i][1].copy())
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i][0] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            
            parent1 = fitness_scores[winner_idx][1]
            
            # Select second parent
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i][0] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            parent2 = fitness_scores[winner_idx][1]
            
            # Crossover
            child1, child2 = crossover_sequences(parent1, parent2)
            
            # Mutate with adaptive rate
            # Increase mutation rate slightly in later generations for exploration
            adaptive_mutation = mutation_rate * (1.0 - generation / generations)
            child1 = mutate_sequence(child1, adaptive_mutation)
            child2 = mutate_sequence(child2, adaptive_mutation)
            
            # Occasionally add new patterns for diversity
            if random.random() < 0.12:
                pattern_type = random.random()
                if pattern_type < 0.15:
                    # Add bell pattern
                    length = random.randint(min_length, max_length)
                    child1 = generate_bell_pattern(length)
                elif pattern_type < 0.30:
                    # Add geometric pattern
                    length = random.randint(min_length, max_length)
                    child1 = generate_geometric_sequence(length)
                elif pattern_type < 0.45:
                    # Add power law pattern
                    length = random.randint(min_length, max_length)
                    child1 = generate_power_law_sequence(length)
                elif pattern_type < 0.60:
                    # Add fibonacci pattern
                    length = random.randint(min_length, max_length)
                    child1 = generate_fibonacci_sequence(length)
                elif pattern_type < 0.75:
                    # Add spiral pattern
                    length = random.randint(min_length, max_length)
                    child1 = generate_spiral_sequence(length)
                elif pattern_type < 0.90:
                    # Add multiscale pattern
                    length = random.randint(min_length, max_length)
                    child1 = generate_multiscale_sequence(length)
                else:
                    # Add sparse peak pattern
                    length = random.randint(min_length, max_length)
                    child1 = generate_sparse_peak_sequence(length)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        population = new_population[:population_size]
        
        # Add diversity periodically with more aggressive strategy
        if generation % 12 == 0 and len(new_population) > 0:
            # Add some completely random individuals occasionally
            for _ in range(12):
                length = random.randint(min_length, max_length)
                individual = generate_random_sequence(length)
                if len(population) < population_size:
                    population.append(individual)
    
    # Final validation
    if best_individual is None:
        # Fallback to a good known pattern
        best_individual = [1.0] * 100
    
    return best_individual

def hybrid_local_search(initial_sequence: List[float], max_time_seconds: float = 90.0) -> List[float]:
    """
    Enhanced local search with multiple strategies from inspiration programs
    """
    start_time = time.time()
    
    current_sequence = initial_sequence.copy()
    current_fitness = evaluate_sequence_fitness(current_sequence)
    
    best_sequence = current_sequence.copy()
    best_fitness = current_fitness
    
    iteration = 0
    max_iterations = 2000  # Increased for better local search
    
    # Simulated annealing with better temperature schedule
    temp = 1.0
    cooling_rate = 0.998  # Slightly slower cooling for better exploration
    
    # Track recent improvements for adaptive cooling
    recent_improvements = []
    max_stagnation = 40
    
    while iteration < max_iterations and time.time() - start_time < max_time_seconds * 0.95:
        iteration += 1
        
        # Gradually decrease temperature for simulated annealing
        temp = max(0.001, temp * cooling_rate)
        
        # Try different types of mutations with adaptive strategy
        mutated = False
        # Adaptive number of changes based on progress
        num_changes = max(1, min(12, len(current_sequence) // 10))
        if len(recent_improvements) > 5:
            avg_improvement = sum(recent_improvements[-5:]) / 5
            if avg_improvement < 0.0005:
                # Reduce changes when improvements slow down
                num_changes = max(1, num_changes // 2)
        
        # Try changing a few elements
        for _ in range(num_changes):
            if random.random() < 0.95:  # Higher chance to change an element
                idx = random.randint(0, len(current_sequence) - 1)
                # Use different mutation strategies
                strategy = random.random()
                if strategy < 0.2:
                    # Multiplicative mutation with wide range
                    factor = random.uniform(0.1, 10.0)
                    current_sequence[idx] *= factor
                elif strategy < 0.4:
                    # Additive mutation with larger variance
                    delta = random.gauss(0, 0.5 * current_sequence[idx])
                    current_sequence[idx] += delta
                elif strategy < 0.6:
                    # Moderate additive mutation
                    delta = random.gauss(0, 0.2 * current_sequence[idx])
                    current_sequence[idx] += delta
                elif strategy < 0.8:
                    # Smaller additive mutation
                    delta = random.gauss(0, 0.05 * current_sequence[idx])
                    current_sequence[idx] += delta
                else:
                    # Large jump mutation
                    current_sequence[idx] = random.uniform(0.01, 1000.0)
                current_sequence[idx] = max(0.01, min(1000.0, current_sequence[idx]))
                mutated = True
        
        # If we made changes, evaluate and possibly accept
        if mutated:
            new_fitness = evaluate_sequence_fitness(current_sequence)
            delta = new_fitness - current_fitness
            
            if delta > 0:
                # Always accept better solutions
                current_fitness = new_fitness
                if new_fitness > best_fitness:
                    best_fitness = new_fitness
                    best_sequence = current_sequence.copy()
                    recent_improvements = []  # Reset stagnation counter
            else:
                # Accept worse solutions with probability based on temperature and delta
                if random.random() < np.exp(delta / max(temp, 1e-10)):
                    current_fitness = new_fitness
                    if new_fitness > best_fitness:
                        best_fitness = new_fitness
                        best_sequence = current_sequence.copy()
                        recent_improvements = []
            
            # Check for stagnation and take corrective action
            recent_improvements.append(delta)
            if len(recent_improvements) > max_stagnation:
                recent_improvements.pop(0)
                # If no significant improvement in recent iterations, add some noise
                if len(recent_improvements) > 1 and \
                   abs(sum(recent_improvements[-15:])) < 1e-8:
                    # Add small random perturbations to escape local minimum
                    for _ in range(4):
                        idx = random.randint(0, len(current_sequence) - 1)
                        current_sequence[idx] *= random.uniform(0.96, 1.04)
        
        # Occasionally do a complete restart with better pattern
        if random.random() < 0.04:
            # Try different pattern-based restarts
            pattern_type = random.random()
            if pattern_type < 0.15:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_bell_pattern(new_length)
            elif pattern_type < 0.30:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_geometric_sequence(new_length)
            elif pattern_type < 0.45:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_power_law_sequence(new_length)
            elif pattern_type < 0.60:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_fibonacci_sequence(new_length)
            elif pattern_type < 0.75:
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_multiscale_sequence(new_length)
            else:
                # Restart with completely random sequence but with better distribution
                new_length = max(20, min(1000, len(current_sequence) + random.randint(-40, 40)))
                current_sequence = generate_random_sequence(new_length)
            current_fitness = evaluate_sequence_fitness(current_sequence)
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_sequence = current_sequence.copy()
    
    return best_sequence

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence
    Uses hybrid approach: enhanced genetic algorithm + local improvement
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # First, run enhanced genetic algorithm to get a good starting point
    ga_solution = enhanced_genetic_algorithm_search(40.0)  # Use 40s for GA
    
    # Then refine with local search
    local_solution = hybrid_local_search(ga_solution, 35.0)  # Use 35s for local search
    
    # Finally, try differential evolution for fine-tuning (from INSPIRATION 1)
    try:
        from scipy.optimize import differential_evolution
        # Create bounds for differential evolution
        bounds = [(0.01, 1000.0) for _ in range(len(local_solution))]
        
        # Objective function for minimization (we want to maximize 1/C1)
        def objective(x):
            # Ensure minimum sum to avoid numerical issues
            if np.sum(x) < 0.01:
                return 1e10  # Large penalty
            _, inv_c1 = compute_autocorrelation_constant(x)
            return -inv_c1  # Negative because we want to maximize
        
        # Run differential evolution with more iterations and better parameters
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            mutation=(0.8, 1.0),
            recombination=0.9,
            disp=False
        )
        
        if result.success:
            _, inv_c1 = compute_autocorrelation_constant(result.x)
            # If we got a better solution, return it
            current_fitness = evaluate_sequence_fitness(local_solution)
            if inv_c1 > current_fitness:
                return list(result.x)
    except Exception:
        pass
    
    return local_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
