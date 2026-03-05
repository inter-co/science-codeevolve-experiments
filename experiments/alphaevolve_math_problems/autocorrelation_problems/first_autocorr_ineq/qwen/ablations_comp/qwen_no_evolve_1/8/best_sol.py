# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import convolve
import random
import time

def compute_autocorrelation_constant(sequence):
    """
    Compute C₁ for a given sequence.
    C₁ = 2n * max(convolution) / (sum(sequence))²
    We want to maximize 1/C₁ = (sum(sequence))² / (2n * max(convolution))
    """
    if len(sequence) == 0:
        return 0
    
    # Ensure no empty sequences
    sum_seq = np.sum(sequence)
    if sum_seq < 0.01:
        return 0
    
    # Compute convolution (autoconvolution)
    conv = convolve(sequence, sequence, mode='full')
    
    # Take only the relevant part (the middle part where we have overlaps)
    # For autoconvolution of length n, we get 2n-1 elements
    # The maximum occurs around the center
    max_conv = np.max(conv)
    
    # Number of steps
    n = len(sequence)
    
    # Compute C₁
    if max_conv <= 0:
        return 0
    
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    
    return c1

def evaluate_sequence(sequence):
    """
    Evaluate how good a sequence is - returns 1/C₁ (we want to maximize this)
    """
    c1 = compute_autocorrelation_constant(sequence)
    if c1 <= 0:
        return 0
    return 1.0 / c1

def generate_random_sequence(length_range=(10, 100)):
    """Generate a random valid sequence"""
    n = random.randint(*length_range)
    # Generate sequence with some randomness but keep it reasonable
    sequence = [random.uniform(0.1, 10.0) for _ in range(n)]
    return sequence

def mutate_sequence(sequence, mutation_rate=0.1):
    """Create a mutated version of the sequence"""
    new_seq = sequence.copy()
    for i in range(len(new_seq)):
        if random.random() < mutation_rate:
            # Randomly change this element
            new_seq[i] = max(0.01, new_seq[i] + random.gauss(0, 0.5))
    return new_seq

def crossover_sequences(seq1, seq2):
    """Crossover two sequences"""
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return seq1 if len(seq1) > 0 else seq2
    
    # Simple uniform crossover
    crossover_point = random.randint(0, min_len)
    new_seq = seq1[:crossover_point] + seq2[crossover_point:]
    
    # Make sure we don't create empty sequences
    if len(new_seq) == 0:
        new_seq = [random.uniform(0.1, 10.0)]
        
    return new_seq

def optimize_with_evolutionary():
    """Use evolutionary algorithm to find optimal sequence"""
    best_inv_c1 = 0
    best_sequence = None
    
    # Population parameters
    population_size = 50
    generations = 100
    elite_size = 5
    
    # Initialize population
    population = [generate_random_sequence() for _ in range(population_size)]
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [(evaluate_sequence(seq), seq) for seq in population]
        fitness_scores.sort(reverse=True)  # Sort by fitness (descending)
        
        # Keep track of best
        current_best_fitness, current_best_seq = fitness_scores[0]
        if current_best_fitness > best_inv_c1:
            best_inv_c1 = current_best_fitness
            best_sequence = current_best_seq.copy()
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best individuals
        for i in range(elite_size):
            new_population.append(fitness_scores[i][1].copy())
        
        # Generate rest through selection and crossover
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(fitness_scores)
            parent2 = tournament_selection(fitness_scores)
            
            # Crossover
            child = crossover_sequences(parent1, parent2)
            
            # Mutation
            child = mutate_sequence(child)
            
            new_population.append(child)
        
        population = new_population
    
    return best_sequence, best_inv_c1

def tournament_selection(fitness_scores, tournament_size=3):
    """Select individual using tournament selection"""
    tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
    tournament.sort(reverse=True)
    return tournament[0][1]

def optimize_with_gradient_free():
    """Use gradient-free optimization methods"""
    # Try different approaches to find better solutions
    best_inv_c1 = 0
    best_sequence = None
    
    # Try different starting points
    for _ in range(20):
        # Start with a simple pattern that might work well
        n = random.randint(50, 200)
        # Try geometric progression or other structured sequences
        if random.random() < 0.5:
            # Geometric sequence
            r = random.uniform(0.8, 1.2)
            sequence = [r**i for i in range(n)]
        else:
            # Decreasing sequence
            sequence = [max(0.1, 1.0 - i/(n+1)) for i in range(n)]
        
        # Normalize to avoid very small values
        sum_seq = sum(sequence)
        if sum_seq > 0.01:
            sequence = [x/sum_seq * 10 for x in sequence]
        
        # Evaluate
        inv_c1 = evaluate_sequence(sequence)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = sequence.copy()
    
    return best_sequence, best_inv_c1

def search_for_best_sequence() -> list[float]:
    """Main search function using multiple strategies"""
    start_time = time.time()
    
    # Strategy 1: Evolutionary approach
    try:
        seq1, inv_c1_1 = optimize_with_evolutionary()
    except Exception as e:
        print(f"Evolutionary approach failed: {e}")
        seq1, inv_c1_1 = None, 0
    
    # Strategy 2: Gradient-free approach  
    try:
        seq2, inv_c1_2 = optimize_with_gradient_free()
    except Exception as e:
        print(f"Gradient-free approach failed: {e}")
        seq2, inv_c1_2 = None, 0
    
    # Strategy 3: Manual construction of promising patterns
    try:
        # Try constructing a known good pattern
        n = random.randint(100, 300)
        # Try a pattern with decreasing weights
        sequence = [max(0.01, 1.0/(i+1)**0.5) for i in range(n)]
        sequence = [x * 1000 for x in sequence]  # Scale up
        inv_c1_3 = evaluate_sequence(sequence)
    except Exception as e:
        print(f"Manual pattern failed: {e}")
        inv_c1_3 = 0
    
    # Select best result
    candidates = [
        (seq1, inv_c1_1),
        (seq2, inv_c1_2),
        (None, inv_c1_3)
    ]
    
    # Filter out None results
    valid_candidates = [(s, v) for s, v in candidates if s is not None and v > 0]
    
    if not valid_candidates:
        # Fallback to random sequence
        sequence = generate_random_sequence()
        inv_c1 = evaluate_sequence(sequence)
        return sequence
    
    # Return the best candidate
    best_seq, best_inv_c1 = max(valid_candidates, key=lambda x: x[1])
    
    # Final refinement with local optimization if time allows
    if time.time() - start_time < 30:  # If we have time left
        try:
            # Try to refine with scipy minimize
            n = len(best_seq)
            # Convert to numpy for easier handling
            initial_guess = np.array(best_seq)
            
            def objective(x):
                # Minimize negative of our target (since minimize finds minimum)
                return -evaluate_sequence(x.tolist())
            
            # Use bounds to prevent extreme values
            bounds = [(0.01, 1000.0) for _ in range(n)]
            
            result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 50})
            
            if result.success:
                refined_seq = result.x.tolist()
                refined_inv_c1 = evaluate_sequence(refined_seq)
                if refined_inv_c1 > best_inv_c1:
                    best_seq = refined_seq
                    best_inv_c1 = refined_inv_c1
                    
        except Exception as e:
            pass  # If optimization fails, keep previous best
    
    return best_seq

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
