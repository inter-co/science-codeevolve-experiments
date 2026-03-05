# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List
import time

def compute_autoconvolution(f: List[float]) -> List[float]:
    """Compute the autoconvolution g = f * f using scipy.signal.convolve"""
    # Convert to numpy array and ensure non-negative
    f_array = np.array([max(0, val) for val in f])
    # Compute convolution with 'full' mode to get complete autoconvolution
    g = signal.convolve(f_array, f_array, mode='full')
    # Return only the valid part (center portion)
    center = len(g) // 2
    return g[center - len(f_array) + 1:center + len(f_array)]

def compute_norms(g: List[float]) -> tuple[float, float, float]:
    """Compute the three norms needed for C2 calculation - matching evaluator exactly"""
    g_array = np.array(g)
    
    # ||g||₂² (L2 norm squared) - using trapezoidal-like integration as specified
    if len(g_array) < 2:
        norm_2_squared = g_array[0]**2 if len(g_array) > 0 else 0.0
    else:
        # Use the trapezoidal-like integration formula: (dx/3)(y1² + y1*y2 + y2²)
        norm_2_squared = 0.0
        dx = 0.5 / (len(g_array) - 1)  # step size
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

def create_sinc_pattern(n_steps: int) -> List[float]:
    """Create a sinc-based pattern that might work well for autoconvolution"""
    # Create a sinc-like pattern with more careful parameter tuning
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Sinc function pattern with some smoothing - try higher frequency
    sinc_component = np.sinc(6 * x)  # Higher frequency sinc
    # Add some Gaussian smoothing to reduce sharp edges
    gaussian = np.exp(-x**2 * 3) 
    base = sinc_component * gaussian
    
    # Ensure non-negative and normalize
    base = np.maximum(base, 0)
    if np.sum(base) > 0:
        base = base / np.sum(base) * n_steps
    
    return base.tolist()

def create_better_sinc_pattern(n_steps: int) -> List[float]:
    """Create a better sinc-based pattern that's been tested to work well"""
    # This is a pattern that has shown to work well in previous experiments
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # A very precise sinc pattern with good structure - tuned for better performance
    sinc_component = np.sinc(10 * x)  # Even higher frequency sinc
    # Add Gaussian envelope to make it smoother
    gaussian = np.exp(-x**2 * 3)  # Slightly wider Gaussian envelope
    base = sinc_component * gaussian
    
    # Ensure non-negative and normalize
    base = np.maximum(base, 0)
    if np.sum(base) > 0:
        base = base / np.sum(base) * n_steps
    
    return base.tolist()

def create_oscillating_pattern(n_steps: int) -> List[float]:
    """Create an oscillating pattern that's known to work well"""
    # Create a pattern that has oscillations but maintains smoothness
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Pattern inspired by known good solutions
    # Combination of Gaussian and sinusoids
    gaussian = np.exp(-x**2 * 3)  # Moderate Gaussian
    oscillation = 0.5 * np.sin(12 * np.pi * x)  # High frequency sine
    base = gaussian + oscillation
    
    # Ensure non-negative and normalize
    base = np.maximum(base, 0)
    if np.sum(base) > 0:
        base = base / np.sum(base) * n_steps
    
    return base.tolist()

def create_mathematical_pattern(n_steps: int) -> List[float]:
    """Create a mathematically informed pattern that should perform well"""
    # Create a function that's likely to lead to a flat autoconvolution profile
    f_values = []
    
    # Create a smooth, symmetric function with some oscillation - optimized for performance
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Use a combination of Gaussian and sine wave for structure - tuned for better performance
    gaussian = np.exp(-x**2 * 3)  # Slightly wider Gaussian for better spread
    sine_component = 0.4 * np.sin(10 * x * np.pi)  # Higher frequency sine for more detail
    base = gaussian + sine_component
    
    # Ensure non-negative and normalize
    base = np.maximum(base, 0)
    if np.sum(base) > 0:
        base = base / np.sum(base) * n_steps
    
    return base.tolist()

def create_better_mathematical_pattern(n_steps: int) -> List[float]:
    """Create an even better mathematical pattern"""
    # Create a more refined mathematical pattern - optimized for C2 maximization
    x = np.linspace(-0.25, 0.25, n_steps)
    
    # Try a pattern that's been empirically shown to work well:
    # A bell-shaped with slight oscillations to create a better autoconvolution
    bell = np.exp(-x**2 * 2.5)  # Slightly narrower bell for sharper peak
    oscillation = 0.5 * np.sin(12 * np.pi * x) * np.exp(-x**2 * 1.5)  # Higher frequency modulated sine
    base = bell + oscillation
    
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
        
        # Central plateau with tapering edges - optimized for better performance
        if abs(pos) < 0.12:  # Central plateau (smaller for sharper peak)
            height = 1.0
        elif abs(pos) < 0.25:  # Transition region
            height = 0.8 + 0.2 * (1 - abs(pos) / 0.25)
        else:  # Outer edges
            height = 0.2 * (1 - abs(pos) / 0.5)
        
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
    
    # Create a smooth curve that starts low, goes up, then down - with more variation
    curve = 0.7 * (1 + np.cos(np.pi * x)) + 0.1 * np.sin(6 * np.pi * x)  # Add some sine variation
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
        if i < pop_size // 15:  # 6.7% - very specialized patterns
            # Better sinc pattern
            individual = create_better_sinc_pattern(n_steps)
        elif i < pop_size // 8:  # 12.5% 
            # Sinc pattern
            individual = create_sinc_pattern(n_steps)
        elif i < pop_size // 5:  # 20%
            # Better mathematical pattern
            individual = create_better_mathematical_pattern(n_steps)
        elif i < pop_size // 4:  # 25%
            # Mathematical pattern
            individual = create_mathematical_pattern(n_steps)
        elif i < pop_size // 3:  # 33.3%
            # Oscillating pattern
            individual = create_oscillating_pattern(n_steps)
        elif i < pop_size // 2:  # 50%
            # Peak pattern
            individual = create_peak_pattern(n_steps)
        elif i < 2 * pop_size // 3:  # 66.7%
            # Balanced pattern
            individual = create_balanced_pattern(n_steps)
        elif i < 3 * pop_size // 4:  # 75%
            # Gradient pattern
            individual = create_gradient_pattern(n_steps)
        elif i < 4 * pop_size // 5:  # 80%
            # Bell pattern
            individual = create_bell_pattern(n_steps)
        else:
            # Random with some structure - more concentrated on higher values
            individual = []
            for j in range(n_steps):
                if random.random() < 0.1:  # 10% chance of high value for better exploration
                    individual.append(random.uniform(0.8, 1.8))
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
    pop_size = 150  # Reduced for faster execution
    generations = 150  # Reduced for faster execution
    mutation_rate = 0.2  # Slightly reduced for more stable evolution
    
    # Initialize population
    population = initialize_population(pop_size, n_steps)
    
    best_fitness = 0.0
    best_individual = None
    
    start_time = time.time()
    
    # Track convergence to early terminate if stuck
    last_improvement_gen = 0
    patience = 25
    
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
        
        # Keep elite (top 25%)
        elite_size = pop_size // 4
        new_population = population[:elite_size]
        
        # Generate offspring through tournament selection, crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection for parents - sample from top 60% for better quality
            tournament_size = 4
            tournament_indices = random.sample(range(min(elite_size * 2, len(population))), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            parent1 = population[winner_index]
            
            # Second parent from top 80%
            parent2 = population[random.randint(0, min(elite_size * 2, len(population)) - 1)]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutate
            child = mutate_individual(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    # Final refinement of the best individual with multiple strategies
    if best_individual is not None:
        # Try several refinement approaches with better balance
        refined_candidates = []
        
        # Local search refinement - more aggressive
        refined1 = local_refinement(best_individual, 100)
        refined_candidates.append(refined1)
        
        # Simple local search with more iterations
        refined2 = simple_local_search(best_individual, 150)
        refined_candidates.append(refined2)
        
        # Add a gradient-based refinement approach
        gradient_refined = best_individual.copy()
        # Apply small adjustments to increase values in key regions
        for i in range(len(gradient_refined) // 4, 3 * len(gradient_refined) // 4):
            gradient_refined[i] = gradient_refined[i] * 1.02  # Slight boost
        
        total = sum(gradient_refined)
        if total > 0:
            gradient_refined = [x / total * len(gradient_refined) for x in gradient_refined]
        refined_candidates.append(gradient_refined)
        
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
        # Fallback to better sinc pattern
        return create_better_sinc_pattern(n_steps)

def simple_local_search(f: List[float], max_iterations: int = 100) -> List[float]:
    """Simple hill climbing local search with better convergence"""
    current = f.copy()
    current_c2 = compute_c2(current)
    
    for iteration in range(max_iterations):
        # Make small random changes to several elements
        candidate = current.copy()
        # Change a few random positions
        num_changes = max(1, len(candidate) // 10)  # Fewer changes for more focused search
        for _ in range(num_changes):
            idx = random.randint(0, len(candidate) - 1)
            candidate[idx] = max(0, candidate[idx] + random.gauss(0, 0.05))  # Smaller mutation
        
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
    # Use the enhanced evolutionary approach with local refinement
    return evolve_step_functions(n_steps=500)

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
