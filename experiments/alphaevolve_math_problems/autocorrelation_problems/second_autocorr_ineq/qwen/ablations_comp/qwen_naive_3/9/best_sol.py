# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from typing import List
import random
from numba import jit
import time
from scipy.optimize import differential_evolution, minimize
import math
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_autoconvolution_norms_fast(f_values: np.ndarray) -> tuple:
    """
    Fast computation of autoconvolution norms using numba-optimized loops.
    """
    n = len(f_values)
    # Compute autoconvolution manually for step functions
    # g[k] = sum_{i=0}^{n-1} f[i] * f[k-i] where both indices valid
    g = np.zeros(2*n - 1)
    
    # Efficient manual convolution for step functions
    for i in range(n):
        for j in range(n):
            k = i + j
            if 0 <= k < len(g):
                g[k] += f_values[i] * f_values[j]
    
    # Take central portion for symmetric case (matching the problem setup)
    start_idx = n - 1
    end_idx = 2*n - 2
    g_centered = g[start_idx:end_idx+1]
    
    # Compute the three norms using trapezoidal rule for L2 norm
    # For L2 norm squared, we use trapezoidal approximation for piecewise linear segments
    norm_2_sq = 0.0
    for i in range(len(g_centered) - 1):
        y1 = g_centered[i]
        y2 = g_centered[i+1]
        # Trapezoidal area approximation for segment: (y1^2 + y1*y2 + y2^2)/3 * h
        # Since h = 1 for our discrete case, we get: (y1^2 + y1*y2 + y2^2)/3
        norm_2_sq += (y1*y1 + y1*y2 + y2*y2) / 3.0
    
    # L1 norm (sum of absolute values)
    norm_1 = np.sum(np.abs(g_centered))
    
    # L-infinity norm
    norm_inf = np.max(np.abs(g_centered))
    
    return norm_2_sq, norm_1, norm_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation.
    Uses efficient convolution and proper normalization.
    """
    # Convert to numpy array for easier handling
    f = np.array(f_values)
    
    # Use fast numba version
    return compute_autoconvolution_norms_fast(f)

def evaluate_c2(f_values: List[float]) -> float:
    """Evaluate C2 value for given step function."""
    try:
        norm_2_sq, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_sq / (norm_1 * norm_inf)
        return c2
    except Exception as e:
        return 0.0

def create_advanced_initialization(length: int) -> List[float]:
    """Create advanced initial individuals with mathematical insights."""
    individual = []
    
    # Strategy: Create patterns that produce flatter autoconvolutions
    # Based on mathematical analysis, uniform or nearly uniform patterns often work well
    # But let's try to incorporate some structure that avoids sharp peaks
    
    # Create a pattern that starts flat, rises to a peak, then falls back down
    # but with sufficient spread to encourage flat autoconvolution
    center = length // 2
    peak_height = 1.0
    
    for i in range(length):
        # Distance from center normalized
        distance = abs(i - center) / (length / 2)
        
        # Create a smooth pattern that's flatter in the middle
        # Using a combination of Gaussian and polynomial decay
        gaussian_component = peak_height * math.exp(-distance**2 * 2)
        polynomial_component = 0.5 * (1 - distance**2) if distance <= 1 else 0
        
        # Combine components with a weighting factor
        base_value = 0.7 * gaussian_component + 0.3 * polynomial_component
        
        # Add some structured randomness to avoid local minima
        noise_factor = 0.1 * base_value
        noise = random.gauss(0, noise_factor)
        
        individual.append(max(0, base_value + noise))
    
    # Normalize to have reasonable magnitude
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 10 for x in individual]
    
    return individual

def create_periodic_pattern(length: int) -> List[float]:
    """Create a periodic pattern that tends to produce flatter autoconvolutions."""
    individual = []
    # Use a combination of sine waves with different frequencies
    for i in range(length):
        # Base frequency plus harmonics
        val = 0.6 + 0.2 * math.sin(i * 0.1) + 0.1 * math.sin(i * 0.3) + 0.05 * math.sin(i * 0.5)
        individual.append(max(0, val + random.gauss(0, 0.03)))
    return individual

def create_multi_peak_pattern(length: int) -> List[float]:
    """Create a multi-peak pattern designed to generate flatter autoconvolution."""
    individual = []
    # Create peaks at regular intervals
    peak_positions = [length // 4, length // 2, 3 * length // 4]
    peak_heights = [0.8, 1.2, 0.8]
    
    for i in range(length):
        # Distance to nearest peak
        min_distance = min(abs(i - pos) for pos in peak_positions)
        # Create bell-shaped peaks
        val = 0.0
        for pos, height in zip(peak_positions, peak_heights):
            distance = abs(i - pos) / (length / 8)
            val += height * math.exp(-distance**2 * 2)
        
        individual.append(max(0, val + random.gauss(0, 0.05 * val)))
    
    return individual

def create_bell_curve_pattern(length: int) -> List[float]:
    """Create a bell curve pattern that is known to work well for flat autoconvolutions."""
    individual = []
    center = length // 2
    sigma = length / 8  # Controls width of the bell curve
    
    for i in range(length):
        # Create a Gaussian-like shape
        val = math.exp(-((i - center)**2) / (2 * sigma**2))
        # Add some minor variation
        val += 0.1 * math.sin(i * 0.2)  # Add some periodicity
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 10 for x in individual]
    
    return individual

def create_smooth_gradient_pattern(length: int) -> List[float]:
    """Create a smooth gradient pattern with gradual transitions."""
    individual = []
    
    # Create a smooth transition from low to high and back to low
    for i in range(length):
        # Gradual rise and fall
        t = i / (length - 1) if length > 1 else 0
        # Smooth S-curve
        val = 3 * t**2 - 2 * t**3
        # Add some periodic variation for complexity
        val += 0.2 * math.sin(t * 10)
        # Ensure non-negative
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 10 for x in individual]
    
    return individual

def create_alternating_pattern(length: int) -> List[float]:
    """Create an alternating pattern with controlled amplitude."""
    individual = []
    
    # Create alternating high/low pattern
    for i in range(length):
        # Alternating pattern
        level = 0.8 if (i // 20) % 2 == 0 else 0.3
        # Add some smooth variation
        smooth = 0.1 * math.sin(i * 0.1)
        val = level + smooth
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 10 for x in individual]
    
    return individual

def create_optimized_pattern(length: int) -> List[float]:
    """Create an optimized pattern based on mathematical insights for high C2."""
    individual = []
    
    # Create a pattern that balances high values with sufficient spread
    # This tries to create a "flat-top" pattern that produces flatter autoconvolutions
    center = length // 2
    half_width = length // 4
    
    for i in range(length):
        # Create a flat-top pattern with smooth edges
        distance_from_center = abs(i - center)
        
        if distance_from_center <= half_width:
            # Flat region in the middle
            val = 1.0
        else:
            # Exponential decay from the flat region
            decay_distance = distance_from_center - half_width
            val = max(0, 1.0 - 0.5 * (decay_distance / (length/4)))
        
        # Add some controlled variation to avoid getting stuck in local optima
        variation = 0.1 * math.sin(i * 0.3) * math.cos(i * 0.1)
        val += variation
        
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 10 for x in individual]
    
    return individual

def create_sine_wave_pattern(length: int) -> List[float]:
    """Create a sine wave pattern with controlled amplitude."""
    individual = []
    
    # Create a sine wave with multiple harmonics
    for i in range(length):
        # Multiple sine components with decreasing amplitudes
        val = 0.7 + 0.2 * math.sin(i * 0.15) + 0.1 * math.sin(i * 0.3) + 0.05 * math.sin(i * 0.45)
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 10 for x in individual]
    
    return individual

def create_piecewise_linear_pattern(length: int) -> List[float]:
    """Create a piecewise linear pattern with strategic peaks."""
    individual = []
    
    # Create a pattern with strategic linear segments
    segments = [
        (0, length//4, 0.0),
        (length//4, length//2, 1.0),
        (length//2, 3*length//4, 0.5),
        (3*length//4, length, 0.0)
    ]
    
    for i in range(length):
        # Find which segment this point belongs to
        segment_idx = 0
        for j, (start, end, height) in enumerate(segments):
            if start <= i < end:
                segment_idx = j
                break
        
        # Linear interpolation within the segment
        start, end, height = segments[segment_idx]
        if end > start:
            ratio = (i - start) / (end - start)
            if segment_idx == 0:
                val = 0.0 + ratio * (height - 0.0)
            elif segment_idx == 1:
                val = height + ratio * (segments[segment_idx+1][2] - height)
            elif segment_idx == 2:
                val = height + ratio * (segments[segment_idx+1][2] - height)
            else:
                val = height
        else:
            val = height
        
        # Add some noise for robustness
        noise = 0.05 * random.gauss(0, 1)
        individual.append(max(0, val + noise))
    
    return individual

def create_high_variance_pattern(length: int) -> List[float]:
    """Create a pattern with high variance that may produce better autoconvolution properties."""
    individual = []
    
    # Create a pattern with significant variations that could lead to interesting autoconvolution
    for i in range(length):
        # Combination of multiple frequency components with varying amplitudes
        val = 0.5 + 0.3 * math.sin(i * 0.2) + 0.2 * math.sin(i * 0.5) + \
              0.1 * math.sin(i * 0.8) + 0.05 * math.sin(i * 1.2)
        individual.append(max(0, val + random.gauss(0, 0.03)))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 10 for x in individual]
    
    return individual

def create_spiky_pattern(length: int) -> List[float]:
    """Create a spiky pattern that might produce beneficial autoconvolution properties."""
    individual = []
    
    # Create spikes at regular intervals
    spike_positions = [i * length // 10 for i in range(10)]
    spike_heights = [1.0, 0.7, 1.2, 0.6, 1.1, 0.8, 1.3, 0.5, 1.0, 0.9]
    
    for i in range(length):
        # Find closest spike
        min_distance = min(abs(i - pos) for pos in spike_positions)
        val = 0.0
        for pos, height in zip(spike_positions, spike_heights):
            distance = abs(i - pos) / (length / 20)
            val += height * math.exp(-distance**2 * 3)
        
        individual.append(max(0, val + random.gauss(0, 0.02 * val)))
    
    return individual

def create_oscillating_pattern(length: int) -> List[float]:
    """Create an oscillating pattern that can help with autoconvolution flatness."""
    individual = []
    
    # Create an oscillating pattern with decreasing amplitude
    for i in range(length):
        # Start with high frequency, then reduce
        freq = 0.2 + 0.1 * math.sin(i * 0.05)
        amplitude = 0.8 + 0.2 * math.sin(i * 0.1)
        val = amplitude * math.sin(freq * i)
        individual.append(max(0, val + 0.5))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 10 for x in individual]
    
    return individual

def create_fat_tail_pattern(length: int) -> List[float]:
    """Create a fat tail pattern that might help with the C2 optimization."""
    individual = []
    
    # Create a pattern with fat tails - high values at ends, low in middle
    center = length // 2
    for i in range(length):
        distance = abs(i - center) / (length / 2)
        # Fat-tail distribution
        val = 1.0 / (1.0 + distance**4)
        # Add some sinusoidal modulation
        val += 0.1 * math.sin(i * 0.1)
        individual.append(max(0, val))
    
    # Normalize
    total = sum(individual)
    if total > 0:
        individual = [x / total * length / 10 for x in individual]
    
    return individual

def mutate_individual(individual: List[float], mutation_rate: float = 0.1) -> List[float]:
    """Enhanced mutation with adaptive strategies."""
    mutated = individual.copy()
    
    # Apply different mutation strategies based on value characteristics
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            current_val = mutated[i]
            
            # Adaptive mutation based on value magnitude and position
            if current_val < 0.1:
                # Very small values - small mutation
                mutated[i] = max(0, current_val + np.random.normal(0, 0.01))
            elif current_val < 0.5:
                # Medium values - moderate mutation
                mutated[i] = max(0, current_val + np.random.normal(0, 0.03))
            elif current_val < 1.0:
                # High values - medium mutation
                mutated[i] = max(0, current_val + np.random.normal(0, 0.05))
            else:
                # Very high values - smaller mutation to prevent overshoot
                mutated[i] = max(0, current_val + np.random.normal(0, 0.03))
                
    return mutated

def crossover_parents(parent1: List[float], parent2: List[float]) -> List[float]:
    """Enhanced crossover with better blending strategies."""
    child = []
    min_len = min(len(parent1), len(parent2))
    
    for i in range(min_len):
        # Use arithmetic crossover with adaptive blending
        if random.random() < 0.7:
            # Blend with weights that depend on parent values
            alpha = random.random()
            # Weight towards higher values to preserve good features
            if parent1[i] > 0.5 and parent2[i] > 0.5:
                alpha = max(0.3, min(0.7, alpha))
            elif parent1[i] < 0.3 or parent2[i] < 0.3:
                alpha = random.uniform(0.2, 0.8)
                
            blended = alpha * parent1[i] + (1 - alpha) * parent2[i]
            # Add noise proportional to the blended value
            noise = random.gauss(0, 0.02 * blended if blended > 0 else 0.01)
            child.append(max(0, blended + noise))
        else:
            # Uniform crossover with adaptive probabilities
            parent = parent1 if random.random() < 0.6 else parent2
            # Add noise to maintain diversity
            noise = random.gauss(0, 0.01 * parent[i] if parent[i] > 0 else 0.005)
            child.append(max(0, parent[i] + noise))
            
    return child

def tournament_selection(population: List[List[float]], fitness_scores: List[tuple], k: int) -> List[float]:
    """Improved tournament selection with better pressure and diversity."""
    tournament_indices = random.sample(range(len(population)), min(k, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    # Sort by fitness descending (higher is better)
    tournament_fitness.sort(key=lambda x: x[1], reverse=True)
    # Return the best of the tournament
    return tournament_fitness[0][0]

def optimize_with_direct_methods() -> List[float]:
    """Use direct optimization methods to find good starting points."""
    # Use a combination of approaches
    best_result = None
    best_c2 = 0
    
    # Try different initializations with various strategies
    for attempt in range(20):
        # Try different lengths and patterns
        length = random.randint(500, 1500)
        
        # Different initialization strategies
        strategies = [
            create_advanced_initialization,
            create_periodic_pattern,
            create_multi_peak_pattern,
            create_bell_curve_pattern,
            create_smooth_gradient_pattern,
            create_alternating_pattern,
            create_optimized_pattern,
            create_sine_wave_pattern,
            create_piecewise_linear_pattern,
            create_high_variance_pattern,
            create_spiky_pattern,
            create_oscillating_pattern,
            create_fat_tail_pattern
        ]
        
        strategy = random.choice(strategies)
        initial_guess = strategy(length)
        
        # Simple local search refinement
        try:
            current = initial_guess.copy()
            current_c2 = evaluate_c2(current)
            
            # Local search around this point with adaptive steps
            for _ in range(100):
                # Create neighbor by small perturbations
                neighbor = mutate_individual(current, 0.1)
                neighbor_c2 = evaluate_c2(neighbor)
                
                if neighbor_c2 > current_c2:
                    current = neighbor
                    current_c2 = neighbor_c2
                    
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_result = current
                
        except:
            continue
    
    return best_result if best_result is not None else create_advanced_initialization(1000)

def evolve_step_function() -> List[float]:
    """
    Enhanced evolution with better strategies to maximize C2.
    """
    # Parameters for enhanced genetic algorithm
    population_size = 500  # Larger population for better exploration
    generations = 250      # More generations for better convergence
    mutation_rate = 0.12   # Moderate mutation rate for balance
    elite_size = 50        # More elites for better preservation of good solutions
    
    # Initialize population with diverse strategies
    population = []
    for _ in range(population_size):
        length = random.randint(300, 2000)  # Wider range for diversity
        strategy = random.random()
        
        if strategy < 0.1:
            # Optimized patterns
            population.append(create_optimized_pattern(length))
        elif strategy < 0.2:
            # Sine wave patterns
            population.append(create_sine_wave_pattern(length))
        elif strategy < 0.3:
            # Piecewise linear patterns
            population.append(create_piecewise_linear_pattern(length))
        elif strategy < 0.4:
            # Advanced structured individuals
            population.append(create_advanced_initialization(length))
        elif strategy < 0.5:
            # Periodic patterns
            population.append(create_periodic_pattern(length))
        elif strategy < 0.6:
            # Multi-peak patterns
            population.append(create_multi_peak_pattern(length))
        elif strategy < 0.7:
            # Bell curve patterns
            population.append(create_bell_curve_pattern(length))
        elif strategy < 0.8:
            # Alternating patterns
            population.append(create_alternating_pattern(length))
        elif strategy < 0.9:
            # High variance patterns
            population.append(create_high_variance_pattern(length))
        else:
            # Other specialized patterns
            pattern_type = random.random()
            if pattern_type < 0.5:
                population.append(create_spiky_pattern(length))
            else:
                population.append(create_oscillating_pattern(length))
    
    best_fitness = 0
    best_individual = None
    start_time = time.time()
    
    for generation in range(generations):
        # Early stopping if we've been stuck for a while
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break
            
        # Evaluate fitness for all individuals
        fitness_scores = [(ind, evaluate_c2(ind)) for ind in population]
        
        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Track best individual
        if fitness_scores[0][1] > best_fitness:
            best_fitness = fitness_scores[0][1]
            best_individual = fitness_scores[0][0].copy()
        
        # Select top individuals (elitism)
        elites = [ind for ind, _ in fitness_scores[:elite_size]]
        
        # Generate new population through crossover and mutation
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            # Tournament selection with larger k for more pressure
            parent1 = tournament_selection(population, fitness_scores, 15)
            parent2 = tournament_selection(population, fitness_scores, 15)
            
            # Crossover
            child = crossover_parents(parent1, parent2)
            
            # Mutation
            child = mutate_individual(child, mutation_rate)
            
            # Occasionally introduce completely new structured individual
            if random.random() < 0.05:
                child_length = random.randint(300, 2000)
                child_strategy = random.random()
                if child_strategy < 0.15:
                    child = create_optimized_pattern(child_length)
                elif child_strategy < 0.3:
                    child = create_sine_wave_pattern(child_length)
                elif child_strategy < 0.45:
                    child = create_piecewise_linear_pattern(child_length)
                elif child_strategy < 0.6:
                    child = create_advanced_initialization(child_length)
                elif child_strategy < 0.75:
                    child = create_periodic_pattern(child_length)
                elif child_strategy < 0.9:
                    child = create_high_variance_pattern(child_length)
                else:
                    child = create_spiky_pattern(child_length)
            
            new_population.append(child)
        
        population = new_population
    
    return best_individual if best_individual is not None else create_advanced_initialization(1000)

def construct_function() -> List[float]:
    """Main function to construct optimized step-function with high C2 value."""
    # Try evolutionary approach first
    evoluted = evolve_step_function()
    
    # Try direct optimization approach
    direct_result = optimize_with_direct_methods()
    
    # Also try a few hand-crafted patterns known to work well
    candidates = [evoluted, direct_result]
    
    # Add some carefully crafted structures with mathematical insight
    for i in range(20):
        length = random.randint(500, 1500)
        
        # Create a pattern that combines different behaviors
        combined_pattern = []
        for j in range(length):
            # Mix of periodic and localized behavior
            periodic = 0.7 + 0.2 * math.sin(j * 0.08) + 0.1 * math.sin(j * 0.2)
            localized = 0.1 * math.exp(-((j - length//2)**2) / (length/10)**2)
            val = periodic + localized + random.uniform(-0.05, 0.05)
            combined_pattern.append(max(0, val))
        candidates.append(combined_pattern)
        
        # Create a pattern with alternating high-low regions
        alternating_pattern = []
        for j in range(length):
            # Alternating pattern with some smoothing
            level = 0.8 if (j // 50) % 2 == 0 else 0.3
            smooth = 0.1 * math.sin(j * 0.1)
            val = level + smooth + random.uniform(-0.05, 0.05)
            alternating_pattern.append(max(0, val))
        candidates.append(alternating_pattern)
        
        # Create a more complex bell curve pattern
        bell_pattern = []
        center = length // 2
        sigma = length / 10
        for j in range(length):
            val = math.exp(-((j - center)**2) / (2 * sigma**2))
            # Add some additional structure
            val += 0.1 * math.sin(j * 0.15) + 0.05 * math.cos(j * 0.05)
            bell_pattern.append(max(0, val))
        candidates.append(bell_pattern)
        
        # Create a flat-top pattern
        flat_top_pattern = []
        center = length // 2
        half_width = length // 4
        for j in range(length):
            distance_from_center = abs(j - center)
            if distance_from_center <= half_width:
                val = 1.0
            else:
                decay_distance = distance_from_center - half_width
                val = max(0, 1.0 - 0.5 * (decay_distance / (length/4)))
            flat_top_pattern.append(max(0, val))
        candidates.append(flat_top_pattern)
        
        # Create a fat tail pattern
        fat_tail_pattern = []
        center = length // 2
        for j in range(length):
            distance = abs(j - center) / (length / 2)
            val = 1.0 / (1.0 + distance**4)
            val += 0.1 * math.sin(j * 0.1)
            fat_tail_pattern.append(max(0, val))
        candidates.append(fat_tail_pattern)
        
        # Create a spiky pattern
        spiky_pattern = []
        spike_positions = [i * length // 10 for i in range(10)]
        spike_heights = [1.0, 0.7, 1.2, 0.6, 1.1, 0.8, 1.3, 0.5, 1.0, 0.9]
        for j in range(length):
            min_distance = min(abs(j - pos) for pos in spike_positions)
            val = 0.0
            for pos, height in zip(spike_positions, spike_heights):
                distance = abs(j - pos) / (length / 20)
                val += height * math.exp(-distance**2 * 3)
            spiky_pattern.append(max(0, val))
        candidates.append(spiky_pattern)
    
    # Evaluate all candidates and return the best
    best_candidate = max(candidates, key=evaluate_c2)
    
    return best_candidate

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
