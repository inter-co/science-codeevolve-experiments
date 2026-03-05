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
warnings.filterwarnings('ignore')

# Enhanced evolutionary algorithm approach with better operators and strategies

class EvolutionaryAutocorrelationOptimizer:
    def __init__(self, population_size=150, generations=1500, mutation_rate=0.3):
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
        individual = []
        
        # Strategy: Create sequences inspired by mathematical constructions known to work well
        # Use a combination of geometric distributions and strategic peaks
        
        if length <= 30:
            # Small sequences: focus on concentrated peaks with exponential decay
            # Start with a few strong peaks and decay outward
            individual = [0.0] * length
            
            # Place a strong peak in the center
            center = length // 2
            individual[center] = random.uniform(500, 1000)
            
            # Add symmetric decay around the peak
            for i in range(length):
                dist = abs(i - center)
                if dist > 0:
                    # Exponential decay with distance
                    decay_factor = math.exp(-dist * 0.3)
                    individual[i] = random.uniform(10, 100) * decay_factor
            
            # Add a second peak if possible
            if length > 5:
                other_peak_pos = random.choice([0, length-1, center-3, center+3])
                if 0 <= other_peak_pos < length and other_peak_pos != center:
                    individual[other_peak_pos] = random.uniform(300, 700)
                    
        elif length <= 80:
            # Medium sequences: use multi-peak patterns with strategic spacing
            individual = [0.0] * length
            
            # Create peaks with strategic placement
            num_peaks = min(4, length // 10 + 1)
            peak_positions = []
            
            # Place peaks with reasonable spacing
            spacing = max(5, length // (num_peaks + 1))
            for i in range(num_peaks):
                pos = min(i * spacing + spacing // 2, length - 1)
                peak_positions.append(pos)
            
            # Create peaks with varying strengths
            for i, pos in enumerate(peak_positions):
                # Different peak types based on position
                if i % 3 == 0:
                    peak_height = random.uniform(400, 800)
                elif i % 3 == 1:
                    peak_height = random.uniform(250, 500)
                else:
                    peak_height = random.uniform(150, 350)
                
                individual[pos] = peak_height
                
                # Add surrounding elements with decreasing strength
                for j in range(max(0, pos-4), min(length, pos+5)):
                    if j != pos:
                        # Exponential decay with distance
                        dist = abs(j - pos)
                        decay = math.exp(-dist * 0.5)
                        individual[j] += random.uniform(5, 30) * decay
                        
        else:
            # Large sequences: use structured patterns with clear periodicity and strategic variations
            individual = []
            
            # Create a repeating pattern with strategic peaks
            pattern_length = min(15, max(5, length // 15))
            pattern = []
            
            # Build a basic pattern with alternating heights
            for i in range(pattern_length):
                if i % 4 == 0:
                    pattern.append(random.uniform(400, 700))
                elif i % 4 == 1:
                    pattern.append(random.uniform(200, 400))
                elif i % 4 == 2:
                    pattern.append(random.uniform(100, 300))
                else:
                    pattern.append(random.uniform(50, 150))
            
            # Repeat the pattern to fill the sequence
            for i in range(length):
                individual.append(pattern[i % pattern_length])
            
            # Add strategic variations to improve performance
            # Add some local peaks and adjustments
            for _ in range(min(10, length // 10)):
                pos = random.randint(0, length - 1)
                adjustment = random.uniform(0.8, 1.2)
                individual[pos] = max(0.01, individual[pos] * adjustment)
            
            # Add occasional stronger peaks
            for _ in range(min(5, length // 20)):
                pos = random.randint(0, length - 1)
                if individual[pos] < 300:
                    individual[pos] = random.uniform(300, 800)
        
        # Ensure all values are within bounds and at least one is significant
        individual = [max(0.01, min(1000.0, val)) for val in individual]
        
        # Ensure at least one element is substantial to prevent degenerate cases
        if sum(individual) < 10:
            max_idx = individual.index(max(individual))
            individual[max_idx] = max(100, individual[max_idx])
            
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
        # Use a blend crossover with preference for structural elements
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
                if pos < 2 or pos >= len(mutated) - 2:
                    # Edge positions: more conservative mutations
                    if random.random() < 0.8:
                        factor = random.uniform(0.9, 1.1)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:
                        mutated[i] = random.uniform(1, 200)
                elif current_val < 20:
                    # Low values: prefer small adjustments
                    if random.random() < 0.7:
                        factor = random.uniform(0.85, 1.15)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:
                        mutated[i] = random.uniform(1, 50)
                elif current_val < 100:
                    # Medium values: balanced approach
                    mutation_type = random.choice(['small', 'medium', 'reset'])
                    if mutation_type == 'small':
                        factor = random.uniform(0.8, 1.2)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    elif mutation_type == 'medium':
                        factor = random.uniform(0.6, 1.8)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:  # reset
                        mutated[i] = random.uniform(1, 200)
                else:
                    # High values: prefer smaller adjustments to avoid instability
                    if random.random() < 0.9:
                        factor = random.uniform(0.9, 1.1)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:
                        mutated[i] = random.uniform(10, 300)
        
        # Apply bounds and ensure minimum number of elements
        mutated = [max(0.01, min(1000.0, val)) for val in mutated]
        
        # Ensure minimum diversity in case no mutations were applied
        if mutations_applied == 0 and len(mutated) > 1:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1] = max(0.01, mutated[idx1] * random.uniform(0.9, 1.1))
            mutated[idx2] = max(0.01, mutated[idx2] * random.uniform(0.9, 1.1))
        
        return mutated

    def optimize(self, max_time: float = 45.0) -> List[float]:
        """Main optimization loop using enhanced evolutionary algorithm."""
        start_time = time.time()
        
        # Initialize population with diverse strategies
        population = []
        for _ in range(self.population_size):
            # Generate individuals with varied lengths and patterns
            # Focus on ranges that have shown good performance in literature
            length = random.randint(20, 100)  
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

def search_for_best_sequence() -> List[float]:
    """
    Main function to search for the best coefficient sequence using enhanced evolutionary approach.
    """
    optimizer = EvolutionaryAutocorrelationOptimizer(
        population_size=150,
        generations=1500,
        mutation_rate=0.3
    )
    
    try:
        best_sequence = optimizer.optimize(max_time=40.0)
        return best_sequence
    except Exception as e:
        print(f"Error in evolutionary optimization: {e}")
        return [1.0] * 50

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
