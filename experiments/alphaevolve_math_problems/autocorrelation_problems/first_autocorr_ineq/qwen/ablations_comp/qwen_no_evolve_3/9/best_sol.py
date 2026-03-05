# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import convolve
from scipy.optimize import differential_evolution
from deap import base, creator, tools, algorithms
import time

def compute_autocorrelation_constant(sequence):
    """
    Compute the autocorrelation constant C₁ for a given sequence.
    C₁ = 2n * max(convolution) / (sum(sequence))²
    We want to maximize 1/C₁ = (sum(sequence))² / (2n * max(convolution))
    """
    if len(sequence) == 0:
        return 0
    
    # Ensure all values are non-negative and clip to reasonable bounds
    sequence = np.array([max(0, min(1000, x)) for x in sequence])
    
    # Skip if sum is too small
    total_sum = np.sum(sequence)
    if total_sum < 0.01:
        return 0
    
    # Compute convolution (auto-correlation)
    convolution = convolve(sequence, sequence, mode='full')
    
    # Get the maximum value (excluding the zero-padding effects)
    max_conv = np.max(convolution)
    
    # Compute C₁
    n = len(sequence)
    if max_conv == 0:
        return 0
    
    c1 = (2 * n * max_conv) / (total_sum ** 2)
    
    # Return 1/C₁ (what we want to maximize)
    return 1.0 / c1 if c1 != 0 else 0

def evaluate_sequence(individual):
    """Evaluate fitness of a sequence (we maximize 1/C₁)"""
    # Convert individual to sequence with reasonable bounds
    sequence = [max(0, min(1000, x)) for x in individual]
    
    # Remove empty sequences or those with very small sum
    if sum(sequence) < 0.01:
        return (0,)
    
    inv_c1 = compute_autocorrelation_constant(sequence)
    return (inv_c1,)

def create_individual(size):
    """Create a random individual with specified size"""
    return [random.uniform(0, 100) for _ in range(size)]

def mutate_individual(individual, indpb=0.1):
    """Mutate an individual"""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.uniform(0, 1000)
    return individual,

def crossover_individuals(ind1, ind2):
    """Crossover two individuals"""
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]
    return ind1, ind2

def evolutionary_search():
    """Use evolutionary algorithm to find optimal sequence"""
    # Set up DEAP framework
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual, size=random.randint(10, 1000))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_sequence)
    toolbox.register("mate", crossover_individuals)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    pop = toolbox.population(n=50)
    
    # Run evolution
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                                         ngen=30, stats=stats, halloffame=hof, verbose=False)
    except Exception as e:
        # Fallback to simpler approach if evolutionary fails
        pass
    
    # Return the best individual found
    if hof:
        best_individual = hof[0]
        return list(best_individual)
    else:
        # Fallback: try a few different approaches
        return generate_focused_sequence()

def generate_focused_sequence():
    """Generate a focused sequence based on mathematical insights"""
    # Try to construct sequences that might work well
    # Based on theory, sequences with specific patterns often work well
    
    # Try geometric progression
    n = random.randint(100, 1000)
    # Create decreasing geometric sequence
    base_val = 1.0
    decay = 0.95
    sequence = [base_val * (decay ** i) for i in range(n)]
    
    # Normalize to have reasonable sum
    total = sum(sequence)
    if total > 0:
        sequence = [x * 100 / total for x in sequence]
    
    # Try a few different patterns
    patterns = [
        # Simple decreasing pattern
        [1.0 / (i + 1) for i in range(n)],
        # Square root pattern
        [1.0 / np.sqrt(i + 1) for i in range(n)],
        # Exponential decay
        [np.exp(-0.05 * i) for i in range(n)],
        # Random but bounded
        [random.uniform(0, 100) for _ in range(n)]
    ]
    
    best_seq = None
    best_inv_c1 = 0
    
    for pattern in patterns:
        seq = [max(0, min(1000, x)) for x in pattern]
        inv_c1 = compute_autocorrelation_constant(seq)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_seq = seq
    
    return best_seq if best_seq is not None else [1.0] * 100

def search_for_best_sequence():
    """Main function to search for the best coefficient sequence"""
    start_time = time.time()
    
    try:
        # First try evolutionary approach
        sequence = evolutionary_search()
        
        # Make sure we have a valid sequence
        if sequence is None or len(sequence) == 0:
            sequence = generate_focused_sequence()
            
        # Validate and refine
        sequence = [max(0, min(1000, x)) for x in sequence]
        if sum(sequence) < 0.01:
            sequence = [1.0] * 100
            
        # Additional refinement
        if len(sequence) < 10:
            sequence.extend([1.0] * (10 - len(sequence)))
            
        return sequence[:1000]  # Limit size
        
    except Exception as e:
        # Fallback to simple construction
        return generate_focused_sequence()

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
