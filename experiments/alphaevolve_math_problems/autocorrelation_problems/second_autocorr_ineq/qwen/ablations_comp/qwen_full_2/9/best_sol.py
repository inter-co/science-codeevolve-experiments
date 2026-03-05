# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List, Tuple
import time

def compute_autoconvolution(f: List[float]) -> List[float]:
    """Compute the autoconvolution g = f * f using scipy.signal.convolve"""
    # Convert to numpy array
    f_array = np.array(f)
    # Compute convolution with 'full' mode to get complete autoconvolution
    g = signal.convolve(f_array, f_array, mode='full')
    # Return only the valid part (center portion)
    center = len(g) // 2
    return g[center - len(f_array) + 1:center + len(f_array)]

def compute_norms(g: List[float]) -> tuple[float, float, float]:
    """Compute the three norms needed for C2 calculation - matching evaluator exactly"""
    g_array = np.array(g)
    
    # ||g||₂² (L2 norm squared) - using exact method from evaluator
    # Use piecewise linear integration with proper spacing
    if len(g_array) <= 1:
        norm_2_squared = 0.0
    else:
        # For piecewise linear integration: (dx/3)(y1² + y1*y2 + y2²) per segment
        norm_2_squared = 0.0
        dx = 0.5 / (len(g_array) - 1) if len(g_array) > 1 else 1.0
        for i in range(len(g_array) - 1):
            y1, y2 = g_array[i], g_array[i+1]
            norm_2_squared += (dx/3) * (y1*y1 + y1*y2 + y2*y2)
    
    # ||g||₁ (L1 norm) - using exact method from evaluator
    norm_1 = np.sum(np.abs(g_array)) / (len(g_array) + 1)
    
    # ||g||∞ (Infinity norm)
    norm_inf = np.max(np.abs(g_array))
    
    return norm_2_squared, norm_1, norm_inf

def compute_c2(f: List[float]) -> float:
    """Compute C2 value for given step function"""
    try:
        g = compute_autoconvolution(f)
        norm_2_squared, norm_1, norm_inf = compute_norms(g)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2
    except Exception:
        return 0.0

def create_optimized_sinc_pattern(n_steps: int) -> List[float]:
    """Create an optimized sinc-based pattern - inspired by best performing patterns"""
    # Try a more sophisticated sinc pattern with multiple components
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Multiple sinc components for richer structure - optimized for performance
    sinc1 = np.sinc(6 * x)  # Higher frequency sinc
    sinc2 = 0.7 * np.sinc(10 * x)  # Even higher frequency component
    sinc3 = 0.4 * np.sinc(14 * x)  # Very high frequency
    
    # Combine with Gaussian envelope for smoother behavior
    envelope = np.exp(-x**2 * 3)
    
    # Create combined pattern
    base = sinc1 + sinc2 + sinc3
    base = base * envelope
    
    # Ensure non-negative and normalize
    base = np.maximum(base, 0)
    if np.sum(base) > 0:
        base = base / np.sum(base) * n_steps
    
    return base.tolist()

def create_mathematical_pattern(n_steps: int) -> List[float]:
    """Create a mathematically informed pattern that should perform well"""
    # Create a function that's likely to lead to a flat autoconvolution profile
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Use a combination of Gaussian and sine wave for structure - tuned for better performance
    gaussian = np.exp(-x**2 * 6)  # Even narrower Gaussian for sharper peaks
    sine_component = 0.5 * np.sin(12 * x * np.pi)  # Even higher frequency sine
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
        
        # Central plateau with tapering edges - optimized for better performance
        if abs(pos) < 0.1:  # Central plateau (even smaller for better control)
            height = 1.3
        elif abs(pos) < 0.25:  # Transition region
            height = 0.8 + 0.5 * (1 - abs(pos) / 0.25)
        else:  # Outer edges
            height = 0.3 * (1 - abs(pos) / 0.5)
        
        f_values.append(max(0, height))
    
    # Normalize
    total = sum(f_values)
    if total > 0:
        f_values = [x / total * n_steps for x in f_values]
    
    return f_values

def create_gradient_pattern(n_steps: int) -> List[float]:
    """Create a gradient pattern with structured variation"""
    # Create a smooth gradient from low to high to low
    x = np.linspace(0, 1, n_steps)
    
    # Create a smooth curve that starts low, goes up, then down - with more variation
    curve = 0.6 * (1 + np.cos(np.pi * x)) + 0.2 * np.sin(4 * np.pi * x)  # Add some sine variation
    curve = np.maximum(curve, 0)
    
    if np.sum(curve) > 0:
        curve = curve / np.sum(curve) * n_steps
    
    return curve.tolist()

def create_peak_pattern(n_steps: int) -> List[float]:
    """Create a peak pattern with optimized distribution"""
    f_values = [0.0] * n_steps
    
    # Create more controlled peaks with specific spacing
    num_peaks = 3
    peak_positions = [n_steps // 4, n_steps // 2, 3 * n_steps // 4]  # Evenly spaced
    
    for i, pos in enumerate(peak_positions):
        # Make peaks more uniform in height
        peak_height = 1.0 if i < 2 else 0.8  # Last peak slightly smaller
        
        # Spread the peak more carefully with controlled width
        spread = max(1, n_steps // 12)  # Reduced spread for better control
        for j in range(max(0, pos - spread), min(n_steps, pos + spread + 1)):
            distance = abs(j - pos)
            # Gaussian-like spread with less variance
            f_values[j] += peak_height * np.exp(-distance**2 / (2 * spread**2))
    
    # Normalize
    total = sum(f_values)
    if total > 0:
        f_values = [x / total * n_steps for x in f_values]
    
    return f_values

def initialize_population(pop_size: int, n_steps: int) -> List[List[float]]:
    """Initialize population with diverse and strategically chosen step functions"""
    population = []
    
    # Mix of different initialization strategies - heavily weighted toward proven patterns
    for i in range(pop_size):
        if i < pop_size // 12:  # 8.3% - optimized sinc pattern (very promising)
            individual = create_optimized_sinc_pattern(n_steps)
        elif i < pop_size // 6:  # 16.7% - mathematical pattern
            individual = create_mathematical_pattern(n_steps)
        elif i < pop_size // 4:  # 25% - balanced pattern
            individual = create_balanced_pattern(n_steps)
        elif i < pop_size // 3:  # 33% - gradient pattern
            individual = create_gradient_pattern(n_steps)
        elif i < pop_size // 2:  # 50% - peak pattern
            individual = create_peak_pattern(n_steps)
        else:
            # Random with some structure - more concentrated on higher values
            individual = []
            for j in range(n_steps):
                if random.random() < 0.1:  # 10% chance of high value (more aggressive)
                    individual.append(random.uniform(1.0, 2.0))
                else:
                    individual.append(random.uniform(0.0, 0.6))
        
        # Normalize to keep values reasonable
        total = sum(individual)
        if total > 0:
            individual = [x / total * n_steps for x in individual]
        
        population.append(individual)
    
    return population

def mutate_individual(f: List[float], mutation_rate: float = 0.2) -> List[float]:
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
            # Blend with preference for parent with higher fitness (better approach)
            if random.random() < 0.6:  # 60% chance to take from first parent
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
                        tournament_size: int = 4) -> List[List[float]]:
    """Select individuals using tournament selection with elitism"""
    # Keep the best individuals
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    elite_count = len(population) // 10  # Top 10% as elite
    
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

def local_refinement(f: List[float], iterations: int = 100) -> List[float]:
    """Apply local refinement to improve the solution"""
    current = f.copy()
    current_c2 = compute_c2(current)
    
    # More aggressive local search with better step sizes
    for _ in range(iterations):
        # Try small perturbations
        candidate = current.copy()
        # Focus on modifying key positions rather than all positions
        num_modifications = max(1, len(candidate) // 10)
        for _ in range(num_modifications):
            idx = random.randint(0, len(candidate) - 1)
            # Larger changes for more aggressive exploration
            change = random.gauss(0, 0.05)
            candidate[idx] += change
            candidate[idx] = max(0, candidate[idx])
        
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
    # Parameters - optimized for better performance and convergence within time limits
    pop_size = 300  # Increased population size for better diversity
    generations = 150  # Reduced generations to save time
    mutation_rate = 0.25  # Higher mutation rate for better exploration
    
    # Initialize population with stronger emphasis on best patterns
    population = initialize_population(pop_size, n_steps)
    
    best_fitness = 0.0
    best_individual = None
    
    start_time = time.time()
    
    # Track convergence to early terminate if stuck
    last_improvement_gen = 0
    patience = 40  # More patience for convergence
    
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
            # Tournament selection for parents - sample from top 50% for better quality
            tournament_size = 5
            tournament_indices = random.sample(range(min(elite_size * 5, len(population))), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            parent1 = population[winner_index]
            
            # Second parent from top 70%
            parent2 = population[random.randint(0, min(elite_size * 4, len(population)) - 1)]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutate
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    # Final refinement of the best individual with multiple strategies
    if best_individual is not None:
        # Try several refinement approaches with more intensive searching
        refined_candidates = []
        
        # Local search refinement with more iterations
        refined1 = local_refinement(best_individual, 200)
        refined_candidates.append(refined1)
        
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
        # Fallback to optimized sinc pattern
        return create_optimized_sinc_pattern(n_steps)

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
