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
        # Based on known constructions and mathematical intuition for minimizing C₁
        
        # Use a hybrid approach combining:
        # 1. Mathematical insight from known optimal patterns
        # 2. Randomization to explore the space
        # 3. Structure preservation to maintain good properties
        
        # For small lengths, create more structured patterns
        if length <= 15:
            # Very small sequences: focus on maximizing total mass vs peak
            individual = [random.uniform(10, 200) for _ in range(length)]
            # Add a strong peak in the center
            if length > 0:
                individual[length//2] = random.uniform(500, 1000)
            # Add a few more peaks to balance the structure
            if length > 3:
                individual[length//4] = random.uniform(300, 600)
                individual[3*length//4] = random.uniform(300, 600)
            
        elif length <= 30:
            # Small-medium sequences: balanced approach with some structure
            # Create a pattern inspired by known constructions
            individual = []
            
            # Create a symmetric pattern with peaks at key positions
            for i in range(length):
                if i == 0 or i == length-1:
                    individual.append(random.uniform(100, 300))
                elif i == length//2:
                    individual.append(random.uniform(400, 800))
                elif i == length//4 or i == 3*length//4:
                    individual.append(random.uniform(200, 500))
                else:
                    individual.append(random.uniform(50, 200))
            
            # Add some randomness to make it more robust
            for i in range(min(5, length)):
                if random.random() < 0.5:
                    individual[i] = individual[i] * random.uniform(0.7, 1.3)
                    
        elif length <= 60:
            # Medium sequences: more complex structure
            individual = []
            # Create a pattern that emphasizes both high and low values strategically
            for i in range(length):
                if i % 5 == 0:
                    individual.append(random.uniform(300, 700))
                elif i % 5 == 1:
                    individual.append(random.uniform(150, 400))
                elif i % 5 == 2:
                    individual.append(random.uniform(100, 300))
                elif i % 5 == 3:
                    individual.append(random.uniform(50, 200))
                else:
                    individual.append(random.uniform(20, 100))
            
            # Add some variation to make it more robust
            for i in range(min(10, length)):
                if random.random() < 0.3:
                    individual[i] = individual[i] * random.uniform(0.8, 1.2)
                    
        else:
            # Large sequences: use more sophisticated patterns
            # Create a base pattern that balances high and low values
            base_pattern = [random.uniform(200, 500), random.uniform(100, 300), 
                           random.uniform(50, 200), random.uniform(20, 100)]
            
            individual = []
            for i in range(length):
                individual.append(base_pattern[i % len(base_pattern)])
            
            # Add some randomness to break strict periodicity and increase diversity
            for i in range(min(15, length)):
                if random.random() < 0.25:
                    individual[i] = individual[i] * random.uniform(0.7, 1.3)
        
        # Ensure all values are within bounds and have minimal variation
        individual = [max(0.01, min(1000.0, val)) for val in individual]
        return individual

    def fitness_function(self, individual: List[float]) -> float:
        """Evaluate fitness (1/C₁) of an individual."""
        return self.compute_inv_c1_fft(individual)

    def tournament_selection(self, population: List[List[float]], fitnesses: List[float], tournament_size: int = 5) -> List[float]:
        """Select an individual using tournament selection with larger tournament size."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
        return population[winner_index].copy()

    def crossover_operator(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Improved crossover operator for autocorrelation optimization."""
        # Use uniform crossover with strategic enhancements
        n = min(len(parent1), len(parent2))
        child = []
        
        # Use a combination of uniform crossover and selective inheritance
        for i in range(n):
            # With higher probability, take from parent with better fitness
            if random.random() < 0.8:
                # Prefer parent with better overall fitness
                if i < len(parent1) and i < len(parent2):
                    # Select based on which parent performed better in the past
                    # This is a simplified version - in practice, we'd track better parents
                    child.append(parent1[i] if random.random() < 0.6 else parent2[i])
                else:
                    child.append(parent1[i] if i < len(parent1) else parent2[i])
            else:
                # Blend approach for exploration
                child.append((parent1[i] + parent2[i]) / 2.0)
        
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
                
                # Position-based strategy: mutate differently depending on where we are
                pos_factor = 1.0
                if i < len(mutated) // 4 or i > 3 * len(mutated) // 4:
                    # Boundary positions: more conservative
                    pos_factor = 0.7
                elif i < len(mutated) // 2:
                    # First half: moderate
                    pos_factor = 1.0
                else:
                    # Second half: also moderate
                    pos_factor = 1.0
                
                if current_val < 50:
                    # Low values: prefer small adjustments
                    factor = random.uniform(0.85, 1.15)
                    mutated[i] = max(0.01, mutated[i] * factor)
                elif current_val < 200:
                    # Medium values: balanced approach
                    if random.random() < 0.7:
                        factor = random.uniform(0.8, 1.2)
                        mutated[i] = max(0.01, mutated[i] * factor)
                    else:
                        mutated[i] = random.uniform(1, 300)
                else:
                    # High values: prefer smaller adjustments to avoid instability
                    factor = random.uniform(0.9, 1.1)
                    mutated[i] = max(0.01, mutated[i] * factor)
        
        # Apply bounds and ensure minimum number of elements
        mutated = [max(0.01, min(1000.0, val)) for val in mutated]
        
        # Ensure minimum diversity in case no mutations were applied
        if mutations_applied == 0 and len(mutated) > 1:
            # Introduce some diversity by perturbing two random elements
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
            length = random.randint(10, 100)  
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
