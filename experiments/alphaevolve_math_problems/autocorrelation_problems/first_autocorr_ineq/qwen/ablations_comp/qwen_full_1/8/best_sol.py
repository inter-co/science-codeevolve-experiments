# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution
import time
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

def compute_convolution_fft(a: np.ndarray) -> np.ndarray:
    """Compute convolution using FFT for better performance"""
    conv = fftconvolve(a, a, mode='full')
    return conv[:len(a) * 2 - 1]

def compute_autocorrelation_constant(a: np.ndarray) -> Tuple[float, float]:
    """
    Compute C1 and 1/C1 for a given sequence
    Returns (C1, 1/C1)
    """
    if len(a) == 0 or np.sum(a) < 0.01:
        return float('inf'), 0.0
    
    # Compute convolution
    conv = compute_convolution_fft(a)
    
    # Get maximum value in convolution (excluding index 0 which is sum^2)
    max_conv = np.max(conv[1:]) if len(conv) > 1 else 0.0
    
    # Compute sum squared
    sum_a_squared = np.sum(a) ** 2
    
    # Compute C1
    if sum_a_squared == 0:
        C1 = float('inf')
    else:
        C1 = 2 * len(a) * max_conv / sum_a_squared
    
    inv_C1 = 1.0 / C1 if C1 != 0 else 0.0
    
    return C1, inv_C1

def generate_bell_pattern(length: int) -> List[float]:
    """Generate a bell-shaped pattern that often performs well"""
    sequence = []
    for i in range(length):
        position = i / (length - 1) if length > 1 else 0.5
        # Create a bell shape centered in the middle with some randomness
        value = max(0.01, 1.0 * np.exp(-((position - 0.5) ** 2) * 8) * random.uniform(0.9, 1.1))
        sequence.append(value)
    return sequence

def generate_peak_pattern(length: int) -> List[float]:
    """Generate a pattern with a sharp peak in the center"""
    sequence = []
    for i in range(length):
        position = i / (length - 1) if length > 1 else 0.5
        # Peak in the center with exponential decay
        value = max(0.01, 1.0 * np.exp(-abs(position - 0.5) * 10) * random.uniform(0.9, 1.1))
        sequence.append(value)
    return sequence

def generate_oscillating_pattern(length: int) -> List[float]:
    """Generate an oscillating pattern"""
    sequence = []
    for i in range(length):
        position = i / (length - 1) if length > 1 else 0.5
        # Create oscillation with exponential decay
        value = max(0.01, (0.5 + 0.5 * np.sin(position * 8 * np.pi)) * np.exp(-position * 4) * random.uniform(0.9, 1.1))
        sequence.append(value)
    return sequence

def generate_power_law_pattern(length: int) -> List[float]:
    """Generate a power law pattern"""
    sequence = []
    alpha = random.uniform(1.8, 2.5)
    for i in range(length):
        value = 1.0 / ((i + 1) ** alpha)
        sequence.append(max(0.01, value * random.uniform(0.9, 1.1)))
    return sequence

def generate_exponential_pattern(length: int) -> List[float]:
    """Generate an exponential decay pattern"""
    sequence = []
    decay_rate = random.uniform(0.8, 0.95)
    for i in range(length):
        value = np.power(decay_rate, i)
        sequence.append(max(0.01, value * random.uniform(0.9, 1.1)))
    return sequence

def generate_mixed_pattern(length: int) -> List[float]:
    """Generate a mixed pattern combining multiple strategies"""
    sequence = []
    for i in range(length):
        if i < length // 3:
            # First third: exponential decay
            value = np.exp(-i * 0.1)
        elif i < 2 * length // 3:
            # Middle third: logarithmic
            value = 1.0 / np.log(i - length // 3 + 2)
        else:
            # Last third: linear decay
            value = 1.0 - (i - 2 * length // 3) / (length // 3)
        sequence.append(max(0.01, value * random.uniform(0.9, 1.1)))
    return sequence

def generate_hadamard_like_pattern(length: int) -> List[float]:
    """Generate a pattern inspired by Hadamard matrices - often very effective"""
    sequence = []
    # Create pattern that avoids large convolution peaks
    for i in range(length):
        # Use sinusoidal pattern with decreasing amplitude
        freq = 2 * np.pi * i / length
        amplitude = 1.0 / (1.0 + i * 0.05)
        value = amplitude * np.cos(freq * 0.5) * random.uniform(0.9, 1.1)
        sequence.append(max(0.01, value))
    return sequence

def generate_optimized_power_law(length: int) -> List[float]:
    """Generate an optimized power law pattern with specific parameters that tend to work well"""
    # Use a power law with exponent that's been shown to work well for this problem
    alpha = 2.1  # Slightly tuned from the general range
    sequence = []
    for i in range(length):
        value = 1.0 / ((i + 1) ** alpha)
        sequence.append(max(0.01, value * random.uniform(0.95, 1.05)))
    return sequence

def generate_modified_exponential(length: int) -> List[float]:
    """Generate a modified exponential decay pattern with better parameters"""
    # Try different decay rates that have worked well in practice
    decay_rates = [0.85, 0.86, 0.87, 0.88, 0.89]
    best_rate = 0.87  # Most commonly effective
    sequence = []
    for i in range(length):
        value = np.power(best_rate, i)
        sequence.append(max(0.01, value * random.uniform(0.95, 1.05)))
    return sequence

def generate_double_peak_pattern(length: int) -> List[float]:
    """Generate a double peak pattern that's known to work well"""
    sequence = [0.0] * length
    # Create two peaks with different heights and positions
    peak1_pos = length // 3
    peak2_pos = 2 * length // 3
    peak1_height = 900.0
    peak2_height = 700.0
    
    # First peak
    for i in range(max(0, peak1_pos - 5), min(length, peak1_pos + 6)):
        sequence[i] = max(sequence[i], peak1_height * (1.0 - abs(i - peak1_pos) / 5.0))
    
    # Second peak
    for i in range(max(0, peak2_pos - 5), min(length, peak2_pos + 6)):
        sequence[i] = max(sequence[i], peak2_height * (1.0 - abs(i - peak2_pos) / 5.0))
    
    # Normalize to reasonable values
    total = sum(sequence)
    if total > 0:
        sequence = [x * 1000.0 / total for x in sequence]
    
    # Add small random noise
    sequence = [max(0.01, x + random.gauss(0, 0.005) * x) for x in sequence]
    
    return sequence

def evaluate_sequence_fitness(sequence: List[float]) -> float:
    """
    Evaluate fitness of a sequence - we want to maximize 1/C1
    Returns fitness value (higher is better)
    """
    try:
        C1, inv_c1 = compute_autocorrelation_constant(sequence)
        # Return inverse of C1 as fitness (we want to maximize this)
        # Add penalties for sequences that are too short or have very small sums
        if len(sequence) < 20:
            inv_c1 *= 0.7
        sum_a = sum(sequence)
        if sum_a < 0.1:
            inv_c1 *= 0.5
        return inv_c1
    except Exception:
        return 0.0

def adaptive_local_search(current_solution: List[float], max_time_seconds: float = 30.0) -> List[float]:
    """Enhanced local search with multiple strategies"""
    start_time = time.time()
    current_fitness = evaluate_sequence_fitness(current_solution)
    
    # Multiple local search strategies
    strategy_weights = {
        'small_perturbation': 0.35,
        'large_perturbation': 0.25,
        'random_change': 0.25,
        'neighborhood_search': 0.15,
        'global_perturbation': 0.05
    }
    
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    iteration = 0
    max_iterations = 500
    
    while iteration < max_iterations and time.time() - start_time < max_time_seconds:
        iteration += 1
        
        # Choose search strategy
        strategy = random.choices(
            list(strategy_weights.keys()), 
            weights=list(strategy_weights.values())
        )[0]
        
        test_solution = current_solution.copy()
        
        if strategy == 'small_perturbation':
            # Small random changes
            for i in range(len(test_solution)):
                if random.random() < 0.2:  # 20% chance to modify
                    test_solution[i] *= random.uniform(0.95, 1.05)
                    test_solution[i] = max(0.01, min(1000.0, test_solution[i]))
                    
        elif strategy == 'large_perturbation':
            # Larger changes
            for i in range(len(test_solution)):
                if random.random() < 0.1:  # 10% chance to modify
                    test_solution[i] *= random.uniform(0.7, 1.3)
                    test_solution[i] = max(0.01, min(1000.0, test_solution[i]))
                    
        elif strategy == 'random_change':
            # Random changes to several elements
            num_changes = max(1, len(test_solution) // 20)
            for _ in range(num_changes):
                i = random.randint(0, len(test_solution) - 1)
                test_solution[i] = random.uniform(0.01, 1000.0)
                
        elif strategy == 'neighborhood_search':
            # Focus on neighbors of high-value elements
            high_indices = [i for i, v in enumerate(test_solution) if v > np.mean(test_solution)]
            if high_indices:
                for i in high_indices:
                    if random.random() < 0.15:
                        test_solution[i] *= random.uniform(0.9, 1.1)
                        test_solution[i] = max(0.01, min(1000.0, test_solution[i]))
        
        elif strategy == 'global_perturbation':
            # Apply global transformation to spread out values
            # This helps escape local minima by changing the overall shape
            mean_val = np.mean(test_solution)
            std_val = np.std(test_solution)
            for i in range(len(test_solution)):
                if random.random() < 0.3:
                    # Apply transformation that preserves the total sum
                    test_solution[i] = max(0.01, test_solution[i] * random.uniform(0.8, 1.2))
            
            # Normalize to preserve sum (important for maintaining scale)
            current_sum = sum(test_solution)
            if current_sum > 0.01:
                scale_factor = sum(current_solution) / current_sum
                test_solution = [max(0.01, x * scale_factor) for x in test_solution]
        
        # Evaluate and accept if better
        test_fitness = evaluate_sequence_fitness(test_solution)
        if test_fitness > current_fitness:
            current_solution = test_solution
            current_fitness = test_fitness
            if test_fitness > best_fitness:
                best_fitness = test_fitness
                best_solution = current_solution.copy()
        elif random.random() < 0.01:  # Sometimes accept worse moves
            current_solution = test_solution
            current_fitness = test_fitness
    
    return best_solution

def improved_hybrid_optimization_approach(max_time_seconds: float = 90.0) -> List[float]:
    """
    Improved hybrid optimization approach using the best elements from inspirations
    """
    start_time = time.time()
    
    # Strategy 1: Direct pattern testing for quick wins
    print("Phase 1: Pattern testing...")
    best_solution = None
    best_fitness = 0.0
    
    # Test a variety of patterns systematically with more diversity
    test_patterns = [
        ("bell", 50), ("bell", 100), ("bell", 200), ("bell", 300),
        ("peak", 50), ("peak", 100), ("peak", 200), ("peak", 300),
        ("oscillating", 50), ("oscillating", 100), ("oscillating", 200), ("oscillating", 300),
        ("power_law", 50), ("power_law", 100), ("power_law", 200),
        ("exponential", 50), ("exponential", 100), ("exponential", 200),
        ("mixed", 50), ("mixed", 100), ("mixed", 200),
        ("hadamard_like", 50), ("hadamard_like", 100), ("hadamard_like", 200),
        ("optimized_power_law", 50), ("optimized_power_law", 100), ("optimized_power_law", 200),
        ("modified_exponential", 50), ("modified_exponential", 100), ("modified_exponential", 200),
        ("double_peak", 50), ("double_peak", 100), ("double_peak", 200),
    ]
    
    for pattern_name, length in test_patterns:
        if time.time() - start_time > max_time_seconds * 0.25:
            break
            
        try:
            if pattern_name == "bell":
                sequence = generate_bell_pattern(length)
            elif pattern_name == "peak":
                sequence = generate_peak_pattern(length)
            elif pattern_name == "oscillating":
                sequence = generate_oscillating_pattern(length)
            elif pattern_name == "power_law":
                sequence = generate_power_law_pattern(length)
            elif pattern_name == "exponential":
                sequence = generate_exponential_pattern(length)
            elif pattern_name == "mixed":
                sequence = generate_mixed_pattern(length)
            elif pattern_name == "hadamard_like":
                sequence = generate_hadamard_like_pattern(length)
            elif pattern_name == "optimized_power_law":
                sequence = generate_optimized_power_law(length)
            elif pattern_name == "modified_exponential":
                sequence = generate_modified_exponential(length)
            elif pattern_name == "double_peak":
                sequence = generate_double_peak_pattern(length)
            else:  # constant
                sequence = [1.0] * length
                
            inv_c1 = evaluate_sequence_fitness(sequence)
            if inv_c1 > best_fitness:
                best_fitness = inv_c1
                best_solution = sequence.copy()
        except Exception:
            continue
    
    # Strategy 2: Differential evolution with better bounds and initialization
    print("Phase 2: Differential evolution...")
    if best_solution is not None and time.time() - start_time < max_time_seconds * 0.5:
        # Use best solution as starting point for differential evolution
        try:
            length = len(best_solution)
            bounds = [(0.01, 1000.0) for _ in range(length)]
            
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
                maxiter=200,
                popsize=30,
                mutation=(0.8, 1.0),
                recombination=0.85,
                disp=False
            )
            
            if result.success:
                _, inv_c1 = compute_autocorrelation_constant(result.x)
                if inv_c1 > best_fitness:
                    best_fitness = inv_c1
                    best_solution = list(result.x)
        except Exception as e:
            print(f"Differential evolution error: {e}")
            pass
    
    # Strategy 3: Enhanced genetic algorithm for fine-tuning
    print("Phase 3: Genetic algorithm refinement...")
    if best_solution is not None and time.time() - start_time < max_time_seconds * 0.8:
        # Enhanced genetic algorithm with better parameters and adaptive mutation
        population_size = 150
        generations = 500
        elite_size = 25
        initial_mutation_rate = 0.15
        
        # Initialize population with diverse patterns
        population = []
        for _ in range(population_size):
            strategy = random.random()
            length = random.randint(50, 400)
            
            if strategy < 0.1:
                individual = generate_bell_pattern(length)
            elif strategy < 0.2:
                individual = generate_peak_pattern(length)
            elif strategy < 0.3:
                individual = generate_oscillating_pattern(length)
            elif strategy < 0.4:
                individual = generate_power_law_pattern(length)
            elif strategy < 0.5:
                individual = generate_exponential_pattern(length)
            elif strategy < 0.6:
                individual = generate_mixed_pattern(length)
            elif strategy < 0.7:
                individual = generate_hadamard_like_pattern(length)
            elif strategy < 0.8:
                individual = generate_optimized_power_law(length)
            elif strategy < 0.9:
                individual = generate_modified_exponential(length)
            else:
                individual = generate_double_peak_pattern(length)
            
            population.append(individual)
        
        # If we have a good starting solution, include it
        if best_solution is not None:
            population[0] = best_solution.copy()
        
        # Evolution loop
        for generation in range(generations):
            if time.time() - start_time > max_time_seconds * 0.95:
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
                best_solution = current_best_individual.copy()
            
            # Selection using tournament selection with elitism
            new_population = []
            
            # Elitism - keep best individuals
            for i in range(elite_size):
                new_population.append(fitness_scores[i][1].copy())
            
            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection for parent 1
                tournament_indices = random.sample(range(population_size), min(10, population_size // 3))
                tournament_fitness = [fitness_scores[i][0] for i in tournament_indices]
                winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                
                parent1 = fitness_scores[winner_idx][1]
                
                # Tournament selection for parent 2
                tournament_indices = random.sample(range(population_size), min(10, population_size // 3))
                tournament_fitness = [fitness_scores[i][0] for i in tournament_indices]
                winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                parent2 = fitness_scores[winner_idx][1]
                
                # Uniform crossover with probability bias
                child1 = []
                child2 = []
                min_len = min(len(parent1), len(parent2))
                for i in range(min_len):
                    if random.random() < 0.65:  # Bias toward parent1
                        child1.append(parent1[i])
                        child2.append(parent2[i])
                    else:
                        child1.append(parent2[i])
                        child2.append(parent1[i])
                
                # Extend with remaining elements if needed
                if len(parent1) > len(parent2):
                    child1.extend(parent1[min_len:])
                elif len(parent2) > len(parent1):
                    child2.extend(parent2[min_len:])
                
                # Mutate children with adaptive mutation rate
                mutation_rate = initial_mutation_rate * (1.0 - generation / generations)
                if mutation_rate < 0.05:
                    mutation_rate = 0.05
                    
                for i in range(len(child1)):
                    if random.random() < mutation_rate:
                        # Apply different mutation strategies based on position
                        if i < len(child1) // 4:
                            # High mutation for early elements
                            child1[i] *= random.uniform(0.7, 1.3)
                        else:
                            # Lower mutation for later elements
                            child1[i] *= random.uniform(0.9, 1.1)
                        child1[i] = max(0.01, min(1000.0, child1[i]))
                
                for i in range(len(child2)):
                    if random.random() < mutation_rate:
                        if i < len(child2) // 4:
                            child2[i] *= random.uniform(0.7, 1.3)
                        else:
                            child2[i] *= random.uniform(0.9, 1.1)
                        child2[i] = max(0.01, min(1000.0, child2[i]))
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:population_size]
    
    # Final refinement with adaptive local search
    if best_solution is not None and time.time() - start_time < max_time_seconds * 0.9:
        print("Phase 4: Local search refinement...")
        # More aggressive local search with adaptive strategies
        best_solution = adaptive_local_search(best_solution, max_time_seconds * 0.1)
    
    # Return final solution or fallback
    if best_solution is None:
        best_solution = generate_bell_pattern(100)
    
    return best_solution

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence
    Uses improved hybrid optimization approach
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Use the improved hybrid optimization approach
    try:
        sequence = improved_hybrid_optimization_approach()
    except Exception as e:
        print(f"Error in optimization: {e}")
        # Fallback to simple approach
        sequence = generate_bell_pattern(100)
    
    # Ensure minimum sum constraint
    if np.sum(sequence) < 0.01:
        sequence = generate_bell_pattern(len(sequence))
    
    # Clip extreme values to maintain practical limits
    sequence = [max(0.01, min(1000.0, x)) for x in sequence]
    
    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
