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
    def __init__(self, population_size=200, generations=2000, mutation_rate=0.3):
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
        
        # Focus on structures that have shown promise in mathematical literature
        if length <= 20:
            # Small sequences: create patterns inspired by known good constructions
            individual = []
            
            # Create a sequence with a strong central peak and gradual decay
            center = length // 2
            for i in range(length):
                # Central peak with exponential decay
                dist_from_center = abs(i - center)
                if dist_from_center == 0:
                    individual.append(random.uniform(700, 900))
                else:
                    # Exponential decay with some randomness
                    decay = math.exp(-dist_from_center * 0.5)
                    base_val = random.uniform(50, 150) * decay
                    individual.append(max(0.01, base_val))
                    
        elif length <= 50:
            # Medium sequences: use multi-peak patterns with strategic spacing
            individual = []
            
            # Create peaks in strategic locations
            num_peaks = max(2, min(6, length // 8))
            peak_positions = []
            
            # Place peaks with good spacing
            spacing = max(3, length // (num_peaks + 1))
            for i in range(num_peaks):
                pos = min(i * spacing + spacing // 2, length - 1)
                peak_positions.append(pos)
            
            # Create peaks with varying strengths
            for i, pos in enumerate(peak_positions):
                # Different peak types for variety
                if i % 3 == 0:
                    peak_height = random.uniform(700, 900)
                elif i % 3 == 1:
                    peak_height = random.uniform(500, 700)
                else:
                    peak_height = random.uniform(300, 500)
                
                individual.append(peak_height)
                
                # Add surrounding elements with decreasing strength
                for j in range(max(0, pos-2), min(length, pos+3)):
                    if j != pos:
                        # Exponential decay with distance
                        dist = abs(j - pos)
                        decay = math.exp(-dist * 0.8)
                        individual.append(max(0.01, random.uniform(10, 60) * decay))
            
            # Fill remaining positions with small values
            while len(individual) < length:
                individual.append(random.uniform(1, 40))
            individual = individual[:length]
            
        else:
            # Large sequences: use structured patterns with clear periodicity
            individual = []
            
            # Create a more sophisticated pattern with multiple peaks
            pattern_length = min(10, max(4, length // 10))
            pattern = []
            
            # Build a pattern with alternating high/low values
            for i in range(pattern_length):
                if i % 4 == 0:
                    pattern.append(random.uniform(600, 800))
                elif i % 4 == 1:
                    pattern.append(random.uniform(400, 600))
                elif i % 4 == 2:
                    pattern.append(random.uniform(200, 400))
                else:
                    pattern.append(random.uniform(50, 150))
            
            # Repeat and slightly vary the pattern
            for i in range(length):
                individual.append(pattern[i % pattern_length])
                
                # Add some noise to break perfect periodicity
                if random.random() < 0.1:
                    individual[-1] = max(0.01, individual[-1] * random.uniform(0.85, 1.15))
        
        # Ensure all values are within bounds
        individual = [max(0.01, min(1000.0, val)) for val in individual]
        return individual

    def fitness_function(self, individual: List[float]) -> float:
        """Evaluate fitness (1/C₁) of an individual."""
        return self.compute_inv_c1_fft(individual)

    def tournament_selection(self, population: List[List[float]], fitnesses: List[float], tournament_size: int = 4) -> List[float]:
        """Select an individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        return population[winner_index].copy()

    def crossover_operator(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Improved crossover operator for autocorrelation optimization."""
        # Use a blend crossover that preserves structural properties
        n = min(len(parent1), len(parent2))
        child = []
        
        # Determine key characteristics of parents
        p1_max = max(parent1) if parent1 else 0
        p2_max = max(parent2) if parent2 else 0
        
        # Create a more sophisticated crossover strategy
        for i in range(n):
            # Blend based on parent characteristics and local context
            if random.random() < 0.6:
                # Take from parent with higher max (better overall performance)
                if p1_max >= p2_max:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            else:
                # Blend values from both parents with weighted average
                # Weight towards the better parent's values
                if p1_max >= p2_max:
                    weight = 0.8
                else:
                    weight = 0.2
                blended = weight * parent1[i] + (1 - weight) * parent2[i]
                child.append(blended)
        
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
                    if random.random() < 0.7:
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
                        factor = random.uniform(0.85, 1.15)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    elif mutation_type == 'medium':
                        factor = random.uniform(0.7, 1.3)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:  # reset
                        mutated[i] = random.uniform(1, 200)
                else:
                    # High values: prefer smaller adjustments to avoid instability
                    if random.random() < 0.85:
                        factor = random.uniform(0.9, 1.1)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:
                        mutated[i] = random.uniform(50, 300)
        
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
        
        # Initialize population with diverse strategies and focus on promising ranges
        population = []
        # Focus on lengths that tend to produce better results
        lengths = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        for _ in range(self.population_size):
            # Generate individuals with varied lengths focusing on ranges that often work well
            length = random.choice(lengths)  
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
        population_size=200,
        generations=2000,
        mutation_rate=0.3
    )
    
    try:
        best_sequence = optimizer.optimize(max_time=45.0)
        return best_sequence
    except Exception as e:
        print(f"Error in evolutionary optimization: {e}")
        return [1.0] * 50

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
