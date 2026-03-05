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
    def __init__(self, population_size=150, generations=800, mutation_rate=0.25):
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
        # Use more informed patterns based on mathematical insights
        individual = []
        
        # Strategy 1: Create sequences with concentrated energy
        # Focus on patterns that have sharp peaks and minimal spread
        if length <= 25:
            # Small sequences: use very focused patterns
            individual = [0.0] * length
            
            # Place one dominant peak
            peak_pos = random.randint(0, length-1)
            individual[peak_pos] = random.uniform(900, 1000)
            
            # Add a few supporting values to create some spread
            support_count = min(3, length//2)
            for _ in range(support_count):
                pos = random.randint(0, length-1)
                if pos != peak_pos:
                    individual[pos] = random.uniform(100, 300)
            
            # Fill remaining with small values
            for i in range(length):
                if individual[i] == 0:
                    individual[i] = random.uniform(1, 50)
                    
        elif length <= 80:
            # Medium sequences: mix of concentrated and distributed values
            individual = []
            
            # Create several well-spaced peaks
            num_peaks = max(2, length // 15)
            peak_positions = set()
            
            # Distribute peaks more evenly
            while len(peak_positions) < num_peaks:
                pos = random.randint(0, length-1)
                peak_positions.add(pos)
            
            # Generate values with peaks having high values
            for i in range(length):
                if i in peak_positions:
                    individual.append(random.uniform(800, 1000))
                else:
                    # Create a more structured background
                    if random.random() < 0.3:
                        individual.append(random.uniform(100, 300))
                    else:
                        individual.append(random.uniform(1, 100))
            
        else:
            # Large sequences: structured approach with clear peaks
            individual = []
            # Create several strong peaks with good spacing
            num_peaks = max(3, length // 20)
            peak_positions = set()
            
            # Distribute peaks more evenly
            while len(peak_positions) < num_peaks:
                pos = random.randint(0, length-1)
                peak_positions.add(pos)
            
            # Generate values
            for i in range(length):
                if i in peak_positions:
                    individual.append(random.uniform(500, 1000))
                else:
                    # Create a more structured background
                    if random.random() < 0.2:
                        individual.append(random.uniform(100, 300))
                    else:
                        individual.append(random.uniform(1, 100))
        
        # Ensure all values are within bounds and avoid zero values
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
        # Use uniform crossover with preference for high-value elements
        n = min(len(parent1), len(parent2))
        child = []
        
        # Prefer values from parents with higher max values
        p1_max = max(parent1) if parent1 else 0
        p2_max = max(parent2) if parent2 else 0
        
        # Create a more intelligent crossover strategy
        for i in range(n):
            # With higher probability, select from the parent with better max
            if p1_max >= p2_max and random.random() < 0.75:
                child.append(parent1[i])
            elif p2_max >= p1_max and random.random() < 0.75:
                child.append(parent2[i])
            else:
                # Mix values from both parents
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
        """Enhanced mutation operator with adaptive parameters."""
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
                
                # Adjust mutation intensity based on value
                if current_val < 10:
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
            length = random.randint(20, 150)  # Broader range for better exploration
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
    # Try multiple optimization approaches to find better solutions
    best_result = None
    best_fitness = 0
    
    # Try different configurations with more emphasis on exploration
    configs = [
        (150, 800, 0.25),   # Larger population, more generations, moderate mutation
        (120, 1000, 0.2),   # Balanced approach
        (200, 600, 0.3),    # High mutation rate for exploration
    ]
    
    for pop_size, gens, mut_rate in configs:
        try:
            optimizer = EvolutionaryAutocorrelationOptimizer(
                population_size=pop_size,
                generations=gens,
                mutation_rate=mut_rate
            )
            
            result = optimizer.optimize(max_time=40.0)
            fitness = optimizer.best_fitness
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_result = result
                
        except Exception as e:
            print(f"Error with config {pop_size}, {gens}, {mut_rate}: {e}")
            continue
    
    # If no good solution found, return default
    if best_result is None:
        return [1.0] * 50
    
    return best_result

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
