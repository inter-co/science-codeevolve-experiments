# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
import random
from typing import List
import time

def compute_autoconvolution_norms(f_values: List[float]) -> tuple[float, float, float]:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    Implements correct piecewise linear integration as specified in prompt.
    """
    if not f_values:
        return 0.0, 0.0, 0.0
    
    # Convert to numpy array
    f = np.array(f_values)
    n = len(f)
    
    # Compute autoconvolution g = f * f using discrete convolution
    g = signal.convolve(f, f, mode='full')
    
    # Trim to appropriate central portion: 2*n-1 elements, keep middle n elements
    # This corresponds to the convolution result over [-1/2, 1/2] 
    # when original function is defined on [-1/4, 1/4]
    g_center_start = len(g) // 2 - (n - 1)
    g_center_end = len(g) // 2 + n
    g_centered = g[g_center_start:g_center_end]
    
    # Compute norms according to prompt specification:
    # ||g||₂² using piecewise linear integration: 
    # For each adjacent pair of values with width h, 
    # contribution is (h/3)(y1² + y1*y2 + y2²)
    if len(g_centered) < 2:
        g_norm_2_sq = 0.0
    else:
        # Width between consecutive points in original function
        step_width = 0.5 / (n - 1) if n > 1 else 0.5
        g_norm_2_sq = 0.0
        for i in range(len(g_centered) - 1):
            y1, y2 = g_centered[i], g_centered[i+1]
            # Trapezoidal-like piecewise integration: (h/3)(y1² + y1*y2 + y2²)
            g_norm_2_sq += (step_width / 3.0) * (y1**2 + y1*y2 + y2**2)
    
    # ||g||₁: sum of absolute values divided by (len(g) + 1) as per prompt
    g_norm_1 = np.sum(np.abs(g_centered)) / (len(g_centered) + 1)
    
    # ||g||∞: maximum absolute value
    g_norm_inf = np.max(np.abs(g_centered))
    
    return float(g_norm_2_sq), float(g_norm_1), float(g_norm_inf)

def compute_c2(f_values: List[float]) -> float:
    """Compute C2 value for given step function"""
    try:
        g_norm_2_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_norm_1 <= 1e-12 or g_norm_inf <= 1e-12:
            return 0.0
            
        c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
        return c2
    except Exception:
        return 0.0

def construct_function() -> List[float]:
    """
    Advanced evolutionary algorithm that combines mathematical insights from inspirations:
    1. Focus on peak structures that maximize constructive autoconvolution
    2. Use larger population and more generations for better exploration
    3. Implement specialized crossover/mutation operators
    4. Include mathematical optimizations for the specific problem structure
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Optimized parameters for better performance within 60 seconds
    n_steps = 500  # Higher resolution for better results
    max_generations = 300  # More generations for convergence
    population_size = 150  # Larger population for better exploration
    elite_size = 20        # Keep more elite individuals
    
    # Start timing
    start_time = time.time()
    
    # Initialize population with mathematically optimal strategies
    population = []
    
    # Strategy 1: Extremely sharp central peak with strategic tapering
    for _ in range(population_size // 4):
        f_values = []
        half = n_steps // 2
        # Create very sharp central peak with exponential decay
        for i in range(half):
            # Central region with very high values
            dist_from_center = abs(i - half//2)
            if dist_from_center < half//12:
                val = random.uniform(0.99, 1.0)
            elif dist_from_center < half//6:
                val = random.uniform(0.9, 0.99)
            else:
                # Exponential decay to reduce values gradually
                decay_factor = np.exp(-dist_from_center / (half//4))
                val = random.uniform(0.1, 0.5) * decay_factor
            f_values.append(val)
        # Make symmetric
        f_values.extend(reversed(f_values[:half]))
        if n_steps % 2 == 1:
            f_values.insert(half, random.uniform(0.85, 0.99))
        population.append(f_values)
    
    # Strategy 2: Multi-peak with optimal spacing and amplitude
    for _ in range(population_size // 4):
        f_values = [0.0] * n_steps
        # Place 3-5 peaks optimally
        num_peaks = 4
        peak_positions = []
        for i in range(num_peaks):
            pos = (i + 1) * n_steps // (num_peaks + 1)
            peak_positions.append(pos)
        
        # Create peaks with decreasing amplitudes to favor constructive interference
        for i, pos in enumerate(peak_positions):
            if pos < n_steps:
                # Peak amplitude decreases with distance from center
                amplitude = 0.9 * (1 - abs(pos - n_steps//2) / (n_steps//2))
                amplitude = max(0.7, amplitude)  # Minimum amplitude
                f_values[pos] = amplitude
                # Spread the peak with Gaussian-like decay
                spread = n_steps // 20
                for j in range(max(0, pos-spread), min(n_steps, pos+spread+1)):
                    dist = abs(j - pos)
                    decay = np.exp(-dist**2 / (2 * spread**2))
                    f_values[j] = max(f_values[j], amplitude * decay)
        population.append(f_values)
    
    # Strategy 3: Smooth structured function with mathematical symmetry
    for _ in range(population_size // 4):
        f_values = []
        # Create a function that balances peak concentration with smoothness
        for i in range(n_steps):
            # Use a combination of sinusoidal and polynomial shapes
            x = (i - n_steps//2) / (n_steps//2)
            # Sinusoidal component for oscillation
            sin_component = 0.5 * np.sin(4 * np.pi * x) + 0.5
            # Polynomial for shaping
            poly_component = 1.0 - 0.8 * x**2
            # Combine with random factor for diversity
            combined = 0.6 * sin_component + 0.4 * poly_component
            val = max(0.1, min(1.0, combined + random.uniform(-0.1, 0.1)))
            f_values.append(val)
        population.append(f_values)
    
    # Strategy 4: Random with controlled correlation for diversity
    for _ in range(population_size // 4):
        f_values = []
        for i in range(n_steps):
            if i == 0:
                val = random.uniform(0.3, 0.7)
            else:
                # Strong correlation but with adaptive variation
                prev_val = f_values[-1]
                # More controlled variation to maintain smoothness
                delta = (random.random() - 0.5) * 0.3
                val = max(0, min(1, prev_val + delta))
            f_values.append(val)
        population.append(f_values)
    
    # Enhanced evolutionary optimization loop
    best_fitness = -1.0
    best_solution = None
    
    # Main evolutionary loop with better operators
    for generation in range(max_generations):
        # Check time limit to prevent exceeding 60 seconds
        if time.time() - start_time > 55:  # Leave 5 seconds buffer
            break
            
        # Evaluate fitness for entire population
        fitness_scores = []
        for individual in population:
            c2 = compute_c2(individual)
            fitness_scores.append(c2)
            
        # Sort by fitness (descending)
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], reverse=True)
        
        # Track best
        current_best_idx = sorted_indices[0]
        current_best_fitness = fitness_scores[current_best_idx]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx][:]
        
        # Selection: keep top elite_size individuals
        elite_indices = sorted_indices[:elite_size]
        selected_parents = [population[i][:] for i in elite_indices]
        
        # Create new population through advanced reproduction
        new_population = selected_parents[:]  # Elitism
        
        # Generate offspring through specialized operators
        while len(new_population) < population_size:
            # Tournament selection for parents (larger tournament for better selection pressure)
            parent1_idx = random.choice(elite_indices[:elite_size//2])
            parent2_idx = random.choice(elite_indices[:elite_size//2])
            
            # Specialized crossover for step functions
            child = []
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Blend with adaptive parameter based on generation
            alpha = 0.4 + 0.2 * (1 - generation / max_generations)  # Decreasing bias over time
            for i in range(n_steps):
                blended = alpha * parent1[i] + (1 - alpha) * parent2[i]
                # Add mutation with adaptive rate
                if random.random() < 0.08:  # 8% mutation rate
                    noise = (random.random() - 0.5) * 0.15
                    blended = max(0, min(1, blended + noise))
                child.append(blended)
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # Occasionally introduce fresh diversity with mathematical inspiration
        if generation % 15 == 0 and generation > 0:
            for i in range(8):  # Add more diverse individuals
                if len(population) < population_size:
                    # Create mathematically inspired function
                    new_individual = []
                    # Create a function with strong central peak and controlled tails
                    for j in range(n_steps):
                        x = (j - n_steps//2) / (n_steps//2)
                        # Gaussian peak with exponential decay
                        peak_val = 0.9 * np.exp(-x**2 * 10)
                        tail_val = 0.2 * np.exp(-abs(x) * 2)
                        val = max(0.1, min(1.0, peak_val + tail_val + random.uniform(-0.05, 0.05)))
                        new_individual.append(val)
                    population.append(new_individual)
    
    # Final comprehensive local refinement
    if best_solution is not None:
        # Perform intensive local search on the best solution
        current_solution = best_solution[:]
        current_score = compute_c2(current_solution)
        
        # Multiple phases of local search
        for phase in range(3):
            # Phase 1: Fine-grained search
            for _ in range(500):
                neighbor = []
                for val in current_solution:
                    if val > 0:
                        # Smaller perturbation for fine-tuning
                        std_dev = 0.02 * val if val > 0.1 else 0.01
                        perturbation = random.gauss(0, std_dev)
                        new_val = max(0, val + perturbation)
                    else:
                        new_val = random.random() * 0.2
                    neighbor.append(new_val)
                
                neighbor_score = compute_c2(neighbor)
                if neighbor_score > current_score:
                    current_solution = neighbor
                    current_score = neighbor_score
            
            # Phase 2: Large perturbations to escape local optima
            for _ in range(100):
                neighbor = []
                for val in current_solution:
                    if random.random() < 0.1:  # 10% chance to make large change
                        # Large perturbation
                        new_val = max(0, min(1, val + (random.random() - 0.5) * 0.5))
                    else:
                        # Small perturbation
                        if val > 0:
                            std_dev = 0.03 * val if val > 0.1 else 0.015
                            perturbation = random.gauss(0, std_dev)
                            new_val = max(0, val + perturbation)
                        else:
                            new_val = random.random() * 0.3
                    neighbor.append(new_val)
                
                neighbor_score = compute_c2(neighbor)
                if neighbor_score > current_score:
                    current_solution = neighbor
                    current_score = neighbor_score
        
        return current_solution
    else:
        # Fallback to simple construction
        return [random.random() for _ in range(n_steps)]

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
