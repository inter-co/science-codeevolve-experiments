# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List
import time

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    f_values: step heights of the function on [-1/4, 1/4]
    Returns: (||g||₂², ||g||₁, ||g||∞)
    """
    # Convert to numpy array
    f = np.array(f_values)
    n = len(f)
    
    if n == 0:
        return 0.0, 1.0, 0.0
    
    # Step size for the interval [-0.25, 0.25]
    dx = 0.5 / n
    
    # Compute autoconvolution g = f * f using scipy's convolution
    g = signal.convolve(f, f, mode='full')
    
    # Compute ||g||₂² using the correct trapezoidal-like integration
    # As specified in prompt: for interval with heights y1, y2 and width h, 
    # contribution is (h/3)(y1² + y1y2 + y2²)
    if len(g) <= 1:
        norm_g2_sq = 0.0
    else:
        norm_g2_sq = 0.0
        # Each segment has width dx, connecting consecutive points in g
        for i in range(len(g) - 1):
            y1, y2 = g[i], g[i+1]
            norm_g2_sq += (dx/3) * (y1*y1 + y1*y2 + y2*y2)
    
    # ||g||₁ = sum of absolute values times dx (trapezoidal rule)
    norm_g1 = np.sum(np.abs(g)) * dx
    
    # ||g||∞ = max absolute value
    norm_ginf = np.max(np.abs(g))
    
    return norm_g2_sq, norm_g1, norm_ginf

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 = ||g||₂² / (||g||₁ · ||g||∞)"""
    norm_g2_sq, norm_g1, norm_ginf = compute_autoconvolution_norms(f_values)
    
    # Avoid division by zero
    if norm_g1 <= 1e-12 or norm_ginf <= 1e-12:
        return 0.0
    
    c2 = norm_g2_sq / (norm_g1 * norm_ginf)
    return c2

def create_sinc_pattern(n_steps: int) -> List[float]:
    """Create a sinc-based pattern that might work well for autoconvolution"""
    # Create a sinc-like pattern
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Sinc function pattern with some smoothing
    sinc_component = np.sinc(4 * x)  # Main lobe centered
    # Add some Gaussian smoothing to reduce sharp edges
    gaussian = np.exp(-x**2 * 2) 
    base = sinc_component * gaussian
    
    # Ensure non-negative and normalize
    base = np.maximum(base, 0)
    if np.sum(base) > 0:
        base = base / np.sum(base) * n_steps
    
    return base.tolist()

def create_mathematical_pattern(n_steps: int) -> List[float]:
    """Create a mathematically informed pattern that should perform well"""
    # Create a function that's likely to lead to a flat autoconvolution profile
    f_values = []
    
    # Create a smooth, symmetric function with some oscillation
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Use a combination of Gaussian and sine wave for structure
    gaussian = np.exp(-x**2 * 4)  # Narrow Gaussian
    sine_component = 0.3 * np.sin(8 * x * np.pi)  # Oscillation
    base = gaussian + sine_component
    
    # Ensure non-negative and normalize
    base = np.maximum(base, 0)
    if np.sum(base) > 0:
        base = base / np.sum(base) * n_steps
    
    return base.tolist()

def create_balanced_pattern(n_steps: int) -> List[float]:
    """Create a balanced pattern that tries to maximize C2 by creating flatter g"""
    # Create a function that has a central plateau with tapering edges
    f_values = []
    
    # Create a pattern with a central region of high values and edges tapering down
    for i in range(n_steps):
        # Position in normalized range
        pos = (i / (n_steps - 1)) * 2 - 1  # From -1 to 1
        
        # Central plateau with tapering edges
        if abs(pos) < 0.15:  # Central plateau
            height = 1.0
        elif abs(pos) < 0.3:  # Transition region
            height = 0.7 + 0.3 * (1 - abs(pos) / 0.3)
        else:  # Outer edges
            height = 0.3 * (1 - abs(pos) / 0.5)
        
        f_values.append(max(0, height))
    
    # Normalize
    total = sum(f_values)
    if total > 0:
        f_values = [x / total * n_steps for x in f_values]
    
    return f_values

def create_bell_pattern(n_steps: int) -> List[float]:
    """Create a bell-shaped pattern"""
    # Create a smooth bell-shaped function
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Gaussian-like shape
    sigma = 0.15
    bell = np.exp(-x**2 / (2 * sigma**2))
    
    # Normalize
    if np.sum(bell) > 0:
        bell = bell / np.sum(bell) * n_steps
    
    return bell.tolist()

def create_gradient_pattern(n_steps: int) -> List[float]:
    """Create a gradient pattern with structured variation"""
    # Create a smooth gradient from low to high to low
    x = np.linspace(0, 1, n_steps)
    
    # Create a smooth curve that starts low, goes up, then down
    curve = 0.5 * (1 + np.cos(np.pi * x))  # Cosine curve
    curve = np.maximum(curve, 0)
    
    if np.sum(curve) > 0:
        curve = curve / np.sum(curve) * n_steps
    
    return curve.tolist()

def create_uniform_pattern(n_steps: int) -> List[float]:
    """Create a uniform pattern"""
    return [1.0] * n_steps

def initialize_population(pop_size: int, n_steps: int) -> List[List[float]]:
    """Initialize population with diverse and strategically chosen step functions"""
    population = []
    
    # Mix of different initialization strategies - heavily weighted toward proven patterns
    for i in range(pop_size):
        if i < pop_size // 8:
            # Sinc pattern (very promising)
            individual = create_sinc_pattern(n_steps)
        elif i < pop_size // 4:
            # Mathematical pattern
            individual = create_mathematical_pattern(n_steps)
        elif i < pop_size // 3:
            # Balanced pattern
            individual = create_balanced_pattern(n_steps)
        elif i < pop_size // 2:
            # Bell pattern
            individual = create_bell_pattern(n_steps)
        elif i < 3 * pop_size // 4:
            # Gradient pattern
            individual = create_gradient_pattern(n_steps)
        elif i < 7 * pop_size // 8:
            # Uniform pattern
            individual = create_uniform_pattern(n_steps)
        else:
            # Random with some structure
            individual = []
            for j in range(n_steps):
                if random.random() < 0.15:  # 15% chance of high value
                    individual.append(random.uniform(0.8, 1.5))
                else:
                    individual.append(random.uniform(0.0, 0.5))
        
        # Normalize to keep values reasonable
        total = sum(individual)
        if total > 0:
            individual = [x / total * n_steps for x in individual]
        
        population.append(individual)
    
    return population

def mutate_individual(f: List[float], mutation_rate: float = 0.15) -> List[float]:
    """Mutate a single individual with adaptive mutation"""
    mutated = f.copy()
    
    # More aggressive mutation rate for better exploration
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Use adaptive mutation strength based on current value
            mutation_strength = 0.1 + 0.1 * (1.0 - min(1.0, mutated[i]))
            change = random.gauss(0, mutation_strength)
            mutated[i] += change
            # Ensure non-negative
            mutated[i] = max(0, mutated[i])
    
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Crossover two parents - blend crossover with preservation of structure"""
    child = []
    for i in range(max(len(parent1), len(parent2))):
        if i < len(parent1) and i < len(parent2):
            # Blend with some probability of taking from either parent
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        elif i < len(parent1):
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def tournament_selection(population: List[List[float]], 
                        fitness_scores: List[float], 
                        tournament_size: int = 3) -> List[List[float]]:
    """Select individuals using tournament selection with elitism"""
    # Keep the best individuals
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    elite_count = len(population) // 8  # Top 12.5% as elite
    
    selected = [population[i] for i in sorted_indices[:elite_count]]
    
    # Fill rest with tournament selection
    remaining_slots = len(population) - elite_count
    for _ in range(remaining_slots):
        # Tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        selected.append(population[winner_index].copy())
    
    return selected

def local_refinement(f: List[float], iterations: int = 50) -> List[float]:
    """Apply local refinement to improve the solution"""
    current = f.copy()
    current_c2 = compute_c2(current)
    
    for _ in range(iterations):
        # Try small perturbations
        candidate = current.copy()
        for i in range(len(candidate)):
            if random.random() < 0.3:  # 30% chance to modify
                # Small change
                change = random.gauss(0, 0.02)
                candidate[i] += change
                candidate[i] = max(0, candidate[i])
        
        # Normalize
        total = sum(candidate)
        if total > 0:
            candidate = [x / total * len(candidate) for x in candidate]
        
        candidate_c2 = compute_c2(candidate)
        if candidate_c2 > current_c2:
            current = candidate
            current_c2 = candidate_c2
    
    return current

def evolve_step_functions(n_steps: int = 1000) -> List[float]:
    """Evolve step functions using genetic algorithm to maximize C2"""
    # Parameters - optimized for better performance and convergence
    pop_size = 100
    generations = 100
    mutation_rate = 0.20
    
    # Initialize population
    population = initialize_population(pop_size, n_steps)
    
    best_fitness = 0.0
    best_individual = None
    
    start_time = time.time()
    
    # Track convergence to early terminate if stuck
    last_improvement_gen = 0
    patience = 20
    
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            c2 = compute_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_fitness:
                best_fitness = c2
                best_individual = individual.copy()
                last_improvement_gen = generation
        
        # Check time limit
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break
            
        # Early termination if no improvement for too long
        if generation - last_improvement_gen > patience:
            break
            
        # Sort by fitness
        sorted_pairs = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        population = [pair[0] for pair in sorted_pairs]
        fitness_scores = [pair[1] for pair in sorted_pairs]
        
        # Keep elite (top 10%)
        elite_size = pop_size // 10
        new_population = population[:elite_size]
        
        # Generate offspring through tournament selection, crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection for parents - sample from top 30% for better quality
            tournament_size = 4
            tournament_indices = random.sample(range(min(elite_size * 3, len(population))), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            parent1 = population[winner_index]
            
            # Second parent from top 50%
            parent2 = population[random.randint(0, min(elite_size * 2, len(population)) - 1)]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutate
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    # Final refinement of the best individual with multiple strategies
    if best_individual is not None:
        # Try several refinement approaches
        refined_candidates = []
        
        # Local search refinement
        refined1 = local_refinement(best_individual, 100)
        refined_candidates.append(refined1)
        
        # Simple local search
        refined2 = simple_local_search(best_individual, 100)
        refined_candidates.append(refined2)
        
        # Evaluate all candidates
        best_refined = best_individual
        best_refined_c2 = compute_c2(best_individual)
        
        for candidate in refined_candidates:
            candidate_c2 = compute_c2(candidate)
            if candidate_c2 > best_refined_c2:
                best_refined_c2 = candidate_c2
                best_refined = candidate
        
        return best_refined
    
    # Return the best individual found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback to sinc pattern
        return create_sinc_pattern(n_steps)

def simple_local_search(f: List[float], max_iterations: int = 100) -> List[float]:
    """Simple hill climbing local search with better convergence"""
    current = f.copy()
    current_c2 = compute_c2(current)
    
    for iteration in range(max_iterations):
        # Make small random changes to several elements
        candidate = current.copy()
        # Change a few random positions
        num_changes = max(1, len(candidate) // 10)  # More changes for better exploration
        for _ in range(num_changes):
            idx = random.randint(0, len(candidate) - 1)
            candidate[idx] = max(0, candidate[idx] + random.gauss(0, 0.05))
        
        # Normalize
        total = sum(candidate)
        if total > 0:
            candidate = [x / total * len(candidate) for x in candidate]
        
        candidate_c2 = compute_c2(candidate)
        if candidate_c2 > current_c2:
            current = candidate
            current_c2 = candidate_c2
            # Reset iteration counter if we made progress
            iteration = 0
    
    return current

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Use the enhanced evolutionary approach with local refinement
    return evolve_step_functions(n_steps=1000)

def create_sinc_pattern(n_steps: int) -> List[float]:
    """Create a sinc-based pattern that might work well for autoconvolution"""
    # Create a sinc-like pattern
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Sinc function pattern with some smoothing
    sinc_component = np.sinc(4 * x)  # Main lobe centered
    # Add some Gaussian smoothing to reduce sharp edges
    gaussian = np.exp(-x**2 * 2) 
    base = sinc_component * gaussian
    
    # Ensure non-negative and normalize
    base = np.maximum(base, 0)
    if np.sum(base) > 0:
        base = base / np.sum(base) * n_steps
    
    return base.tolist()

def create_mathematical_pattern(n_steps: int) -> List[float]:
    """Create a mathematically informed pattern that should perform well"""
    # Create a function that's likely to lead to a flat autoconvolution profile
    f_values = []
    
    # Create a smooth, symmetric function with some oscillation
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Use a combination of Gaussian and sine wave for structure
    gaussian = np.exp(-x**2 * 4)  # Narrow Gaussian
    sine_component = 0.3 * np.sin(8 * x * np.pi)  # Oscillation
    base = gaussian + sine_component
    
    # Ensure non-negative and normalize
    base = np.maximum(base, 0)
    if np.sum(base) > 0:
        base = base / np.sum(base) * n_steps
    
    return base.tolist()

def create_peak_pattern(n_steps: int) -> List[float]:
    """Create a pattern with distinct peaks"""
    f_values = [0.0] * n_steps
    
    # Add 3-5 peaks
    num_peaks = random.randint(3, 5)
    for _ in range(num_peaks):
        peak_pos = random.randint(0, n_steps - 1)
        peak_height = random.uniform(0.8, 1.2)
        # Spread the peak
        spread = max(1, n_steps // 20)
        for i in range(max(0, peak_pos - spread), min(n_steps, peak_pos + spread + 1)):
            f_values[i] += peak_height * (1 - abs(i - peak_pos) / spread)
    
    # Normalize
    total = sum(f_values)
    if total > 0:
        f_values = [x / total * n_steps for x in f_values]
    
    return f_values

def create_balanced_pattern(n_steps: int) -> List[float]:
    """Create a balanced pattern that tries to maximize C2 by creating flatter g"""
    # Create a function that has a central plateau with tapering edges
    f_values = []
    
    # Create a pattern with a central region of high values and edges tapering down
    for i in range(n_steps):
        # Position in normalized range
        pos = (i / (n_steps - 1)) * 2 - 1  # From -1 to 1
        
        # Central plateau with tapering edges
        if abs(pos) < 0.15:  # Central plateau
            height = 1.0
        elif abs(pos) < 0.3:  # Transition region
            height = 0.7 + 0.3 * (1 - abs(pos) / 0.3)
        else:  # Outer edges
            height = 0.3 * (1 - abs(pos) / 0.5)
        
        f_values.append(max(0, height))
    
    # Normalize
    total = sum(f_values)
    if total > 0:
        f_values = [x / total * n_steps for x in f_values]
    
    return f_values

def create_uniform_pattern(n_steps: int) -> List[float]:
    """Create a uniform pattern"""
    return [1.0] * n_steps

def create_gradient_pattern(n_steps: int) -> List[float]:
    """Create a gradient pattern with structured variation"""
    # Create a smooth gradient from low to high to low
    x = np.linspace(0, 1, n_steps)
    
    # Create a smooth curve that starts low, goes up, then down
    curve = 0.5 * (1 + np.cos(np.pi * x))  # Cosine curve
    curve = np.maximum(curve, 0)
    
    if np.sum(curve) > 0:
        curve = curve / np.sum(curve) * n_steps
    
    return curve.tolist()

def create_bell_pattern(n_steps: int) -> List[float]:
    """Create a bell-shaped pattern"""
    # Create a smooth bell-shaped function
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Gaussian-like shape
    sigma = 0.15
    bell = np.exp(-x**2 / (2 * sigma**2))
    
    # Normalize
    if np.sum(bell) > 0:
        bell = bell / np.sum(bell) * n_steps
    
    return bell.tolist()

def initialize_population(pop_size: int, n_steps: int) -> List[List[float]]:
    """Initialize population with diverse and strategically chosen step functions"""
    population = []
    
    # Mix of different initialization strategies - heavily weighted toward proven patterns
    for i in range(pop_size):
        if i < pop_size // 10:
            # Sinc pattern (very promising)
            individual = create_sinc_pattern(n_steps)
        elif i < pop_size // 5:
            # Mathematical pattern
            individual = create_mathematical_pattern(n_steps)
        elif i < pop_size // 3:
            # Peak pattern
            individual = create_peak_pattern(n_steps)
        elif i < 2 * pop_size // 5:
            # Balanced pattern
            individual = create_balanced_pattern(n_steps)
        elif i < 3 * pop_size // 5:
            # Uniform pattern
            individual = create_uniform_pattern(n_steps)
        elif i < 4 * pop_size // 5:
            # Gradient pattern
            individual = create_gradient_pattern(n_steps)
        elif i < 9 * pop_size // 10:
            # Bell pattern
            individual = create_bell_pattern(n_steps)
        else:
            # Random with some structure
            individual = []
            for j in range(n_steps):
                if random.random() < 0.15:  # 15% chance of high value
                    individual.append(random.uniform(0.8, 1.5))
                else:
                    individual.append(random.uniform(0.0, 0.5))
        
        # Normalize to keep values reasonable
        total = sum(individual)
        if total > 0:
            individual = [x / total * n_steps for x in individual]
        
        population.append(individual)
    
    return population

def mutate_individual(f: List[float], mutation_rate: float = 0.15) -> List[float]:
    """Mutate a single individual with adaptive mutation"""
    mutated = f.copy()
    
    # More aggressive mutation rate for better exploration
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Use adaptive mutation strength based on current value
            mutation_strength = 0.1 + 0.1 * (1.0 - min(1.0, mutated[i]))
            change = random.gauss(0, mutation_strength)
            mutated[i] += change
            # Ensure non-negative
            mutated[i] = max(0, mutated[i])
    
    return mutated

def crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Crossover two parents - blend crossover with preservation of structure"""
    child = []
    for i in range(max(len(parent1), len(parent2))):
        if i < len(parent1) and i < len(parent2):
            # Blend with some probability of taking from either parent
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        elif i < len(parent1):
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    return child

def tournament_selection(population: List[List[float]], 
                        fitness_scores: List[float], 
                        tournament_size: int = 3) -> List[List[float]]:
    """Select individuals using tournament selection with elitism"""
    # Keep the best individuals
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    elite_count = len(population) // 8  # Top 12.5% as elite
    
    selected = [population[i] for i in sorted_indices[:elite_count]]
    
    # Fill rest with tournament selection
    remaining_slots = len(population) - elite_count
    for _ in range(remaining_slots):
        # Tournament
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        selected.append(population[winner_index].copy())
    
    return selected

def local_refinement(f: List[float], iterations: int = 50) -> List[float]:
    """Apply local refinement to improve the solution"""
    current = f.copy()
    current_c2 = compute_c2(current)
    
    for _ in range(iterations):
        # Try small perturbations
        candidate = current.copy()
        for i in range(len(candidate)):
            if random.random() < 0.3:  # 30% chance to modify
                # Small change
                change = random.gauss(0, 0.02)
                candidate[i] += change
                candidate[i] = max(0, candidate[i])
        
        # Normalize
        total = sum(candidate)
        if total > 0:
            candidate = [x / total * len(candidate) for x in candidate]
        
        candidate_c2 = compute_c2(candidate)
        if candidate_c2 > current_c2:
            current = candidate
            current_c2 = candidate_c2
    
    return current

def evolve_step_functions(n_steps: int = 500) -> List[float]:
    """Evolve step functions using genetic algorithm to maximize C2"""
    # Parameters - optimized for better performance and convergence
    pop_size = 300
    generations = 300
    mutation_rate = 0.25
    
    # Initialize population
    population = initialize_population(pop_size, n_steps)
    
    best_fitness = 0.0
    best_individual = None
    
    start_time = time.time()
    
    # Track convergence to early terminate if stuck
    last_improvement_gen = 0
    patience = 30
    
    for generation in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        for individual in population:
            c2 = compute_c2(individual)
            fitness_scores.append(c2)
            
            if c2 > best_fitness:
                best_fitness = c2
                best_individual = individual.copy()
                last_improvement_gen = generation
        
        # Check time limit
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break
            
        # Early termination if no improvement for too long
        if generation - last_improvement_gen > patience:
            break
            
        # Sort by fitness
        sorted_pairs = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        population = [pair[0] for pair in sorted_pairs]
        fitness_scores = [pair[1] for pair in sorted_pairs]
        
        # Keep elite (top 10%)
        elite_size = pop_size // 10
        new_population = population[:elite_size]
        
        # Generate offspring through tournament selection, crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection for parents - sample from top 30% for better quality
            tournament_size = 4
            tournament_indices = random.sample(range(min(elite_size * 3, len(population))), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            parent1 = population[winner_index]
            
            # Second parent from top 50%
            parent2 = population[random.randint(0, min(elite_size * 2, len(population)) - 1)]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutate
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    # Final refinement of the best individual with multiple strategies
    if best_individual is not None:
        # Try several refinement approaches
        refined_candidates = []
        
        # Local search refinement
        refined1 = local_refinement(best_individual, 200)
        refined_candidates.append(refined1)
        
        # Simple local search
        refined2 = simple_local_search(best_individual, 150)
        refined_candidates.append(refined2)
        
        # Direct improvement - try to enhance the best areas
        direct_refined = best_individual.copy()
        # Increase values in the middle regions where they tend to matter most
        mid_start = len(direct_refined) // 3
        mid_end = 2 * len(direct_refined) // 3
        for i in range(mid_start, mid_end):
            direct_refined[i] = direct_refined[i] * 1.05  # Slight boost
        
        total = sum(direct_refined)
        if total > 0:
            direct_refined = [x / total * len(direct_refined) for x in direct_refined]
        refined_candidates.append(direct_refined)
        
        # Evaluate all candidates
        best_refined = best_individual
        best_refined_c2 = compute_c2(best_individual)
        
        for candidate in refined_candidates:
            candidate_c2 = compute_c2(candidate)
            if candidate_c2 > best_refined_c2:
                best_refined_c2 = candidate_c2
                best_refined = candidate
        
        return best_refined
    
    # Return the best individual found
    if best_individual is not None:
        return best_individual
    else:
        # Fallback to sinc pattern
        return create_sinc_pattern(n_steps)

def simple_local_search(f: List[float], max_iterations: int = 100) -> List[float]:
    """Simple hill climbing local search with better convergence"""
    current = f.copy()
    current_c2 = compute_c2(current)
    
    for iteration in range(max_iterations):
        # Make small random changes to several elements
        candidate = current.copy()
        # Change a few random positions
        num_changes = max(1, len(candidate) // 15)  # More changes for better exploration
        for _ in range(num_changes):
            idx = random.randint(0, len(candidate) - 1)
            candidate[idx] = max(0, candidate[idx] + random.gauss(0, 0.08))
        
        # Normalize
        total = sum(candidate)
        if total > 0:
            candidate = [x / total * len(candidate) for x in candidate]
        
        candidate_c2 = compute_c2(candidate)
        if candidate_c2 > current_c2:
            current = candidate
            current_c2 = candidate_c2
            # Reset iteration counter if we made progress
            iteration = 0
    
    return current

def construct_function() -> List[float]:
    """Function to construct step-function with high C2 value."""
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Use the enhanced evolutionary approach with local refinement
    return evolve_step_functions(n_steps=500)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
