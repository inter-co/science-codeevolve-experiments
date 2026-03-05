# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import fftconvolve
import time
from typing import List, Tuple
import warnings
import math
from collections import defaultdict
import copy
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
warnings.filterwarnings('ignore')

# Enhanced evolutionary algorithm approach with better operators and strategies

class EvolutionaryAutocorrelationOptimizer:
    def __init__(self, population_size=150, generations=1500, mutation_rate=0.35):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_individual = None
        self.best_fitness = 0
        random.seed(42)  # For reproducibility
        np.random.seed(42)
        
    def compute_c1_fft(self, sequence: List[float]) -> float:
        """Compute C₁ for a given sequence using FFT for efficient convolution."""
        if len(sequence) == 0:
            return float('inf')
        
        sum_seq = sum(sequence)
        if sum_seq < 0.01:
            return float('inf')
        
        # Use FFT-based convolution for efficiency
        seq_array = np.array(sequence)
        
        # Compute autoconvolution using FFT
        conv = fftconvolve(seq_array, seq_array, mode='full')
        
        # The convolution result has length 2*n - 1, we want the maximum over ALL positions
        max_conv = np.max(conv)
        
        n = len(sequence)
        if max_conv <= 0:
            return float('inf')
            
        c1 = 2 * n * max_conv / (sum_seq ** 2)
        return c1

    def compute_inv_c1_fft(self, sequence: List[float]) -> float:
        """Compute 1/C₁ for a given sequence using FFT."""
        c1 = self.compute_c1_fft(sequence)
        if c1 <= 0:
            return 0
        return 1.0 / c1

    def generate_individual(self, length: int) -> List[float]:
        """Generate a random individual with appropriate constraints."""
        # Use mathematical insights about optimal step functions
        individual = []
        
        # Strategy: Create sequences that balance peak concentration with global distribution
        # Inspired by known constructions that perform well in autocorrelation problems
        
        # Base approach: Start with a pattern that's known to work well
        if length <= 20:
            # Small sequences: use concentrated patterns with some variation
            # Try to create sequences with a single dominant peak plus supporting elements
            individual = [random.uniform(1, 100) for _ in range(length)]
            
            # Add one strong peak in the middle
            mid_pos = length // 2
            individual[mid_pos] = random.uniform(300, 1000)
            
            # Add some variation to surrounding elements
            for i in range(max(0, mid_pos-3), min(length, mid_pos+4)):
                if i != mid_pos:
                    individual[i] = individual[i] * random.uniform(0.5, 1.0)
                    
        elif length <= 60:
            # Medium sequences: use multi-peak patterns with strategic spacing
            individual = []
            
            # Create a few dominant peaks with good spacing
            num_peaks = min(4, length // 10 + 1)
            peak_positions = []
            
            # Place peaks with reasonable spacing
            spacing = max(5, length // (num_peaks + 1))
            for i in range(num_peaks):
                pos = min(i * spacing + spacing // 2, length - 1)
                peak_positions.append(pos)
            
            # Create peaks with varying strengths
            for i, pos in enumerate(peak_positions):
                # Vary peak heights based on position
                if i % 3 == 0:
                    peak_height = random.uniform(400, 1000)
                elif i % 3 == 1:
                    peak_height = random.uniform(200, 600)
                else:
                    peak_height = random.uniform(100, 400)
                
                individual.append(peak_height)
                
                # Add surrounding elements with decreasing strength
                for j in range(max(0, pos-3), min(length, pos+4)):
                    if j != pos:
                        # Exponential decay with distance
                        dist = abs(j - pos)
                        decay = math.exp(-dist * 0.8)
                        individual.append(random.uniform(10, 50) * decay)
            
            # Fill remaining positions with small values
            while len(individual) < length:
                individual.append(random.uniform(1, 30))
            individual = individual[:length]
            
        else:
            # Large sequences: use structured patterns with clear periodicity
            individual = []
            
            # Create a repeating pattern with strategic peaks
            pattern_length = min(10, max(4, length // 10))
            pattern = []
            
            # Build a basic pattern
            for i in range(pattern_length):
                if i % 3 == 0:
                    pattern.append(random.uniform(500, 1000))
                elif i % 3 == 1:
                    pattern.append(random.uniform(200, 600))
                else:
                    pattern.append(random.uniform(100, 400))
            
            # Repeat the pattern to fill the sequence
            for i in range(length):
                individual.append(pattern[i % pattern_length])
            
            # Add some variation to reduce regularity
            for i in range(min(20, length)):
                if random.random() < 0.15:
                    individual[i] = individual[i] * random.uniform(0.8, 1.2)
        
        # Ensure all values are within bounds
        individual = [max(0.01, min(1000.0, val)) for val in individual]
        return individual

    def fitness_function(self, individual: List[float]) -> float:
        """Evaluate fitness (1/C₁) of an individual."""
        return self.compute_inv_c1_fft(individual)

    def tournament_selection(self, population: List[List[float]], fitnesses: List[float], tournament_size: int = 3) -> List[float]:
        """Select an individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        return population[winner_index].copy()

    def crossover_operator(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Improved crossover operator for autocorrelation optimization."""
        # Use uniform crossover with strategic pattern preservation
        n = min(len(parent1), len(parent2))
        child = []
        
        # Determine key characteristics of parents
        p1_max = max(parent1) if parent1 else 0
        p2_max = max(parent2) if parent2 else 0
        
        # Create a more intelligent crossover strategy
        for i in range(n):
            # Prefer values from parent with higher max to preserve beneficial traits
            if random.random() < 0.7:
                if p1_max >= p2_max:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            else:
                # Blend values from both parents with some randomness
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
        
        # Ensure proper length
        if len(child) < n:
            # Fill with values from either parent or new random values
            for i in range(len(child), n):
                if random.random() < 0.5:
                    child.append(parent1[i] if i < len(parent1) else random.uniform(1, 100))
                else:
                    child.append(parent2[i] if i < len(parent2) else random.uniform(1, 100))
        
        return child[:n]

    def mutation_operator(self, individual: List[float], generation: int) -> List[float]:
        """Enhanced mutation operator with adaptive parameters and better structure preservation."""
        mutated = individual.copy()
        
        # Adaptive mutation rate based on generation and individual quality
        adaptive_mutation_rate = self.mutation_rate * (1 - generation / self.generations)
        
        # Track how many mutations were applied to maintain diversity
        mutations_applied = 0
        
        # Apply mutations with different strategies
        for i in range(len(mutated)):
            if random.random() < adaptive_mutation_rate:
                mutations_applied += 1
                
                # Choose mutation type based on value magnitude and position
                current_val = mutated[i]
                pos = i
                
                # Position-based mutation: preserve structure near edges
                if pos < 3 or pos >= len(mutated) - 3:
                    # Edge positions: more conservative mutations
                    if random.random() < 0.7:
                        factor = random.uniform(0.95, 1.05)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:
                        mutated[i] = random.uniform(1, 100)
                elif current_val < 10:
                    # Low values: prefer small adjustments
                    if random.random() < 0.7:
                        factor = random.uniform(0.9, 1.1)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:
                        mutated[i] = random.uniform(1, 20)
                elif current_val < 100:
                    # Medium values: balanced approach
                    mutation_type = random.choice(['small', 'medium', 'reset'])
                    if mutation_type == 'small':
                        factor = random.uniform(0.8, 1.2)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    elif mutation_type == 'medium':
                        factor = random.uniform(0.5, 2.0)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:  # reset
                        mutated[i] = random.uniform(1, 100)
                else:
                    # High values: prefer smaller adjustments to avoid instability
                    if random.random() < 0.8:
                        factor = random.uniform(0.9, 1.1)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:
                        mutated[i] = random.uniform(10, 200)
        
        # Apply bounds and ensure minimum number of elements
        mutated = [max(0.01, min(1000.0, val)) for val in mutated]
        
        # Ensure minimum diversity in case no mutations were applied
        if mutations_applied == 0 and len(mutated) > 1:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1] = max(0.01, mutated[idx1] * random.uniform(0.9, 1.1))
            mutated[idx2] = max(0.01, mutated[idx2] * random.uniform(0.9, 1.1))
        
        return mutated

    def optimize(self, max_time: float = 50.0) -> List[float]:
        """Main optimization loop using enhanced evolutionary algorithm."""
        start_time = time.time()
        
        # Initialize population with diverse strategies
        population = []
        for _ in range(self.population_size):
            # Generate individuals with varied lengths and patterns
            # Focus on ranges that have shown good performance in literature
            length = random.randint(15, 80)  
            individual = self.generate_individual(length)
            population.append(individual)
        
        # Evaluate initial population
        fitnesses = [self.fitness_function(ind) for ind in population]
        
        # Track best individual
        best_idx = fitnesses.index(max(fitnesses))
        self.best_fitness = fitnesses[best_idx]
        self.best_individual = population[best_idx].copy()
        
        # Evolutionary process with enhanced strategies
        for generation in range(self.generations):
            if time.time() - start_time > max_time - 5:
                break
                
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(self.best_individual.copy())
            
            # Generate offspring through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover
                child = self.crossover_operator(parent1, parent2)
                
                # Mutation
                child = self.mutation_operator(child, generation)
                
                new_population.append(child)
            
            # Trim to exact population size
            population = new_population[:self.population_size]
            
            # Evaluate new population
            fitnesses = [self.fitness_function(ind) for ind in population]
            
            # Update best individual
            best_idx = fitnesses.index(max(fitnesses))
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_individual = population[best_idx].copy()
        
        return self.best_individual

def direct_optimization_approach():
    """
    Use direct optimization techniques to find better solutions.
    """
    # Try different approaches for finding the best sequence
    
    # Approach 1: Use differential evolution with constraints
    def objective_function(x):
        # Convert to list and normalize
        sequence = list(x)
        # Compute 1/C1 (we want to maximize this)
        try:
            inv_c1 = compute_inv_c1_direct(sequence)
            return -inv_c1  # Negative because we're minimizing
        except:
            return 1000000  # Penalize invalid sequences
    
    # Try different sequence lengths and structures
    best_result = None
    best_inv_c1 = 0
    
    # Test various sequence configurations
    test_configs = [
        # Small sequences that might work well
        ([100]*5, "uniform_small"),
        ([500, 100, 500, 100, 500], "alternating"),
        ([1000, 100, 100, 100, 1000], "peak_at_ends"),
        ([100, 1000, 100, 1000, 100], "peak_middle"),
        
        # Medium sequences
        ([100]*10, "uniform_medium"),
        ([500, 100, 500, 100, 500, 100, 500, 100, 500, 100], "alternating_medium"),
        ([1000, 100, 100, 100, 1000, 100, 100, 100, 1000, 100], "peak_at_ends_medium"),
    ]
    
    # Try optimized sequences found in literature or with simple patterns
    for config, name in test_configs:
        try:
            inv_c1 = compute_inv_c1_direct(config)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_result = config.copy()
        except:
            continue
    
    # Also try some manually constructed patterns that are known to work well
    # These are inspired by the structure that often works well in such problems
    manual_patterns = [
        # A pattern that balances peak and distribution
        [100, 200, 300, 200, 100, 100, 200, 300, 200, 100],
        [1000, 100, 100, 100, 1000, 100, 100, 100, 1000, 100],
        [500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        [1000, 500, 250, 125, 62.5, 31.25, 15.625, 7.8125, 3.90625, 1.953125],
        [1000, 1000, 100, 100, 100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 1000, 100, 100, 100, 100],
        # Additional patterns that have shown promise
        [1000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1000],
        [100, 100, 100, 100, 1000, 100, 100, 100, 100, 100],
        [1000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1000],
        [500, 100, 100, 100, 500, 100, 100, 100, 500, 100],
        [100, 500, 100, 500, 100, 500, 100, 500, 100, 500],
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        # Patterns specifically designed to beat benchmarks
        [1000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1000],
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1000, 100],
        [1000, 1000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        [1000, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1000],
        [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1000, 100],
    ]
    
    for pattern in manual_patterns:
        try:
            inv_c1 = compute_inv_c1_direct(pattern)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_result = pattern.copy()
        except:
            continue
    
    # Try gradient-based optimization on promising candidates
    if best_result is not None:
        try:
            # Refine the best result using scipy minimize
            bounds = [(0.01, 1000.0)] * len(best_result)
            res = minimize(objective_function, best_result, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
            refined_result = list(res.x)
            refined_inv_c1 = compute_inv_c1_direct(refined_result)
            if refined_inv_c1 > best_inv_c1:
                best_inv_c1 = refined_inv_c1
                best_result = refined_result
        except:
            pass
    
    return best_result if best_result is not None else [1.0] * 20

def compute_inv_c1_direct(sequence: List[float]) -> float:
    """Direct computation of 1/C1 without class overhead"""
    if len(sequence) == 0:
        return 0.0
    
    sum_seq = sum(sequence)
    if sum_seq < 0.01:
        return 0.0
    
    # Use FFT-based convolution for efficiency
    seq_array = np.array(sequence)
    
    # Compute autoconvolution using FFT
    conv = fftconvolve(seq_array, seq_array, mode='full')
    
    # The convolution result has length 2*n - 1, we want the maximum over ALL positions
    max_conv = np.max(conv)
    
    n = len(sequence)
    if max_conv <= 0:
        return 0.0
        
    c1 = 2 * n * max_conv / (sum_seq ** 2)
    return 1.0 / c1

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence using enhanced evolutionary approach.
    """
    # First try direct optimization approach
    direct_result = direct_optimization_approach()
    
    # Then run evolutionary algorithm as backup
    optimizer = EvolutionaryAutocorrelationOptimizer(
        population_size=200,
        generations=2000,
        mutation_rate=0.4
    )
    
    try:
        evolutionary_result = optimizer.optimize(max_time=45.0)
        
        # Compare results and return the better one
        direct_inv_c1 = compute_inv_c1_direct(direct_result)
        evolutionary_inv_c1 = compute_inv_c1_direct(evolutionary_result)
        
        if direct_inv_c1 > evolutionary_inv_c1:
            return direct_result
        else:
            return evolutionary_result
            
    except Exception as e:
        print(f"Error in evolutionary optimization: {e}")
        return direct_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
