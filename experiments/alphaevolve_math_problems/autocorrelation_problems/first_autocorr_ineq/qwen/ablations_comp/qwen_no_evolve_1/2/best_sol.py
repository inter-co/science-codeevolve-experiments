# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy import signal
from scipy.fft import fft, ifft
import time

def compute_autocorrelation_constant(sequence):
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    C₁ = 2n * max(convolution) / (sum(sequence))^2
    We want to maximize 1/C₁, which means minimizing C₁.
    """
    if len(sequence) == 0:
        return float('inf')
    
    # Convert to numpy array
    a = np.array(sequence, dtype=float)
    
    # Check if sum is too small
    sum_a = np.sum(a)
    if sum_a < 0.01:
        return float('inf')
    
    # Compute convolution using FFT for efficiency O(n log n)
    # Using 'full' mode then taking the middle part
    conv = signal.convolve(a, a, mode='full')
    
    # Take the middle part (the actual convolution)
    mid = len(conv) // 2
    conv = conv[mid - len(a) + 1 : mid + len(a)]
    
    # Alternative: more efficient approach using FFT
    # n_fft = 2 * len(a) - 1
    # a_padded = np.pad(a, (0, n_fft - len(a)), 'constant')
    # conv_fft = ifft(fft(a_padded) * np.conj(fft(a_padded)))
    # conv = np.real(conv_fft[:len(a)*2-1])
    
    # Get the maximum value in the convolution
    max_conv = np.max(conv)
    
    # Compute C₁
    n = len(a)
    c1 = (2 * n * max_conv) / (sum_a ** 2)
    
    return c1

def compute_inv_c1(sequence):
    """
    Compute 1/C₁ for a given sequence.
    This is what we want to maximize.
    """
    c1 = compute_autocorrelation_constant(sequence)
    if c1 == float('inf'):
        return 0.0
    return 1.0 / c1

def generate_random_sequence(length_range=(10, 200)):
    """Generate a random sequence with non-negative values."""
    n = random.randint(*length_range)
    # Generate sequence with some structure - exponential decay works well
    sequence = []
    for i in range(n):
        # Use exponential decay pattern to create good candidates
        sequence.append(random.expovariate(0.5) * random.choice([1, -1]))
    
    # Ensure non-negative values
    sequence = [max(0, x) for x in sequence]
    
    # Add some randomness
    for i in range(len(sequence)):
        if random.random() < 0.1:  # 10% chance to adjust
            sequence[i] += random.uniform(-0.1, 0.1)
    
    # Ensure at least one element is positive
    if sum(sequence) < 0.01:
        sequence[0] = max(0.01, sequence[0])
        
    return [max(0, x) for x in sequence]

def mutate_sequence(sequence, mutation_rate=0.1):
    """Create a mutated version of the sequence."""
    new_seq = sequence.copy()
    
    # Randomly change some elements
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            # Apply small random perturbation
            new_seq[i] = max(0, new_seq[i] + random.gauss(0, 0.1 * new_seq[i] if new_seq[i] > 0 else 1))
    
    # Occasionally add/remove elements to explore different lengths
    if random.random() < 0.05 and len(new_seq) > 1:
        # Remove an element
        idx = random.randint(0, len(new_seq) - 1)
        new_seq.pop(idx)
    elif random.random() < 0.05:
        # Add an element
        idx = random.randint(0, len(new_seq))
        new_seq.insert(idx, random.uniform(0, 1))
    
    # Ensure sum is meaningful
    if sum(new_seq) < 0.01:
        new_seq[random.randint(0, len(new_seq)-1)] = max(0.01, new_seq[random.randint(0, len(new_seq)-1)])
        
    return new_seq

def create_step_function_candidate():
    """Create a candidate sequence that resembles a step function."""
    n = random.randint(5, 100)
    
    # Create a step-like function with some randomness
    sequence = []
    
    # Create steps of varying heights
    num_steps = random.randint(2, min(10, n//2))
    
    # Determine step sizes
    step_heights = []
    for i in range(num_steps):
        step_heights.append(random.uniform(0.5, 2.0))
    
    # Distribute steps among positions
    for i in range(n):
        step_idx = min(i * num_steps // n, num_steps - 1)
        sequence.append(step_heights[step_idx] * random.uniform(0.8, 1.2))
    
    # Add some noise to make it more realistic
    for i in range(len(sequence)):
        sequence[i] = max(0, sequence[i] + random.gauss(0, 0.1))
    
    return sequence

def genetic_algorithm_search(max_time=60):
    """Use genetic algorithm to find optimal sequence."""
    start_time = time.time()
    
    # Initialize population
    population_size = 50
    population = []
    
    # Create initial diverse population
    for _ in range(population_size):
        if random.random() < 0.7:
            # Use step function approach
            individual = create_step_function_candidate()
        else:
            # Use random approach
            individual = generate_random_sequence()
        population.append(individual)
    
    best_individual = None
    best_fitness = 0
    
    generation = 0
    while time.time() - start_time < max_time - 1:  # Leave 1 second for final processing
        generation += 1
        
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = compute_inv_c1(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness
        fitness_scores.sort(reverse=True)
        
        # Update best
        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_individual = fitness_scores[0][1].copy()
        
        # Keep top performers
        top_performers = [ind for _, ind in fitness_scores[:population_size//3]]
        
        # Create new population through crossover and mutation
        new_population = top_performers.copy()
        
        # Fill remaining slots with offspring
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = random.choice(top_performers)
            parent2 = random.choice(top_performers)
            
            # Crossover (uniform)
            child = []
            for i in range(min(len(parent1), len(parent2))):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            
            # Extend if needed
            if len(parent1) > len(parent2):
                child.extend(parent1[len(parent2):])
            elif len(parent2) > len(parent1):
                child.extend(parent2[len(parent1):])
            
            # Mutate
            child = mutate_sequence(child)
            new_population.append(child)
        
        population = new_population
        
        # Occasionally introduce completely new individuals
        if generation % 10 == 0:
            for i in range(5):
                if random.random() < 0.5:
                    population[random.randint(0, len(population)-1)] = create_step_function_candidate()
                else:
                    population[random.randint(0, len(population)-1)] = generate_random_sequence()
    
    # Final evaluation
    final_fitness_scores = [(compute_inv_c1(ind), ind) for ind in population]
    final_fitness_scores.sort(reverse=True)
    
    if final_fitness_scores[0][0] > best_fitness:
        best_fitness = final_fitness_scores[0][0]
        best_individual = final_fitness_scores[0][1].copy()
    
    return best_individual

def search_for_best_sequence() -> list[float]:
    """Main function to search for the best coefficient sequence."""
    try:
        # Try the genetic algorithm approach
        sequence = genetic_algorithm_search(max_time=55)  # Leave 5 seconds for cleanup
        
        # If we got something reasonable, return it
        if sequence and len(sequence) > 0:
            return sequence
            
        # Fallback to simple approach
        return generate_random_sequence()
        
    except Exception as e:
        # Fallback to simple random approach
        return generate_random_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
