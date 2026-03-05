# EVOLVE-BLOCK-START

import numpy as np
from scipy import signal
from scipy.optimize import differential_evolution, minimize
import time
from typing import List
from numba import jit
import random
from scipy.spatial.distance import pdist, squareform
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_piecewise_trapezoidal_norms(g_values: np.ndarray, dx: float) -> tuple:
    """
    Efficiently compute ||g||₂² using piecewise linear integration
    For consecutive points y1, y2 with width h: contribution = (h/3)(y1² + y1*y2 + y2²)
    """
    if len(g_values) < 2:
        return 0.0, 0.0, 0.0
    
    norm_2_squared = 0.0
    norm_1 = 0.0
    norm_inf = 0.0
    
    # Compute ||g||₂²
    for i in range(len(g_values) - 1):
        y1 = g_values[i]
        y2 = g_values[i+1]
        norm_2_squared += (dx/3) * (y1*y1 + y1*y2 + y2*y2)
    
    # Compute ||g||₁
    for i in range(len(g_values)):
        norm_1 += abs(g_values[i])
    
    # Compute ||g||∞
    for i in range(len(g_values)):
        val = abs(g_values[i])
        if val > norm_inf:
            norm_inf = val
    
    return norm_2_squared, norm_1, norm_inf

def compute_autoconvolution_norms(f_values: List[float]) -> tuple:
    """
    Compute the three norms needed for C2 calculation:
    ||g||₂², ||g||₁, ||g||∞ where g = f*f
    """
    # Convert to numpy array
    f = np.array(f_values)
    
    # Compute autoconvolution g = f * f using fft for efficiency
    # This gives us the full convolution, but we need to extract the correct portion
    g = signal.convolve(f, f, mode='full')
    
    # Since f is defined on [-1/4, 1/4], the convolution will be on [-1/2, 1/2]
    # But we only care about the central part (corresponding to [-1/4, 1/4] overlap)
    # The result should be symmetric around center, and we want the overlapping region
    
    # Get the central portion that represents the meaningful convolution
    mid = len(g) // 2
    half_len = len(f) - 1
    g_centered = g[mid - half_len:mid + half_len + 1]
    
    # Compute norms using efficient numba-compiled function
    dx = 0.5 / (len(f) - 1) if len(f) > 1 else 1.0
    
    # Apply absolute value to g_centered before computing norms
    g_abs = np.abs(g_centered)
    
    norm_2_squared, norm_1, norm_inf = compute_piecewise_trapezoidal_norms(g_abs, dx)
    
    # Scale norm_1 appropriately (dx was already included in the computation)
    norm_1 *= dx
    
    return norm_2_squared, norm_1, norm_inf

def evaluate_c2(f_values: List[float]) -> float:
    """
    Calculate C₂ = ||g||₂² / (||g||₁ · ||g||∞)
    """
    try:
        norm_2_squared, norm_1, norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if norm_1 <= 1e-15 or norm_inf <= 1e-15:
            return 0.0
            
        c2 = norm_2_squared / (norm_1 * norm_inf)
        return c2
    except Exception:
        return 0.0

def create_advanced_initial_population(n_steps: int, n_pop: int) -> List[List[float]]:
    """Create more sophisticated initial population"""
    population = []
    
    # Strategy 1: Gaussian-like peak with better parameters
    for _ in range(n_pop//6):
        peak_pos = np.random.uniform(0, n_steps-1)
        std = np.random.uniform(0.05, 0.2) * (n_steps-1)
        f = [max(0, np.exp(-0.5 * ((i - peak_pos)/std)**2)) for i in range(n_steps)]
        population.append(f)
    
    # Strategy 2: Double peak with varied spacing
    for _ in range(n_pop//6):
        peak1_pos = np.random.uniform(0, n_steps-1)
        peak2_pos = np.random.uniform(0, n_steps-1)
        std = np.random.uniform(0.05, 0.25) * (n_steps-1)
        f = [max(0, np.exp(-0.5 * ((i - peak1_pos)/std)**2) + 
                 np.exp(-0.5 * ((i - peak2_pos)/std)**2)) for i in range(n_steps)]
        population.append(f)
    
    # Strategy 3: Multi-peak pattern
    for _ in range(n_pop//6):
        f = [0.0] * n_steps
        num_peaks = np.random.randint(2, 5)
        for _ in range(num_peaks):
            peak_pos = np.random.uniform(0, n_steps-1)
            std = np.random.uniform(0.05, 0.15) * (n_steps-1)
            height = np.random.uniform(0.5, 1.5)
            for i in range(n_steps):
                f[i] += height * np.exp(-0.5 * ((i - peak_pos)/std)**2)
        population.append(f)
    
    # Strategy 4: Uniform distribution with noise
    for _ in range(n_pop//6):
        f = [np.random.uniform(0, 1.0) for _ in range(n_steps)]
        # Add some smoothing
        for i in range(1, n_steps-1):
            f[i] = 0.3 * f[i] + 0.35 * f[i-1] + 0.35 * f[i+1]
        population.append(f)
    
    # Strategy 5: Binary pattern with variation
    for _ in range(n_pop//6):
        f = [1.0 if np.random.random() > 0.6 else 0.0 for _ in range(n_steps)]
        # Add some smoothing
        for i in range(1, n_steps-1):
            f[i] = 0.5 * f[i] + 0.25 * f[i-1] + 0.25 * f[i+1]
        population.append(f)
    
    # Strategy 6: Linear ramp with noise
    for _ in range(n_pop//6):
        start_val = np.random.uniform(0, 0.5)
        end_val = np.random.uniform(0.5, 1.0)
        f = [start_val + (end_val - start_val) * i / (n_steps-1) for i in range(n_steps)]
        # Add some noise
        for i in range(n_steps):
            f[i] += np.random.normal(0, 0.05)
        f = [max(0, val) for val in f]
        population.append(f)
    
    return population

def adaptive_mutation(individual: List[float], generation: int, max_generations: int) -> List[float]:
    """Apply adaptive mutation with decreasing intensity"""
    mutated = individual.copy()
    mutation_rate = 0.1 * (1.0 - generation / max_generations)
    
    for i in range(len(mutated)):
        if np.random.random() < mutation_rate:
            # Adaptive step size based on generation
            step_size = 0.1 * (1.0 - generation / max_generations)
            delta = np.random.normal(0, step_size)
            mutated[i] = max(0, mutated[i] + delta)
    return mutated

def advanced_crossover(parent1: List[float], parent2: List[float]) -> List[float]:
    """Advanced crossover with blending"""
    if len(parent1) != len(parent2):
        return parent1
    
    # Blend two parents with different weights
    child = []
    alpha = np.random.random()
    
    for i in range(len(parent1)):
        # Blend with some randomness
        blend = alpha * parent1[i] + (1 - alpha) * parent2[i]
        # Add some noise to encourage exploration
        noise = np.random.normal(0, 0.01)
        child.append(max(0, blend + noise))
    
    return child

def improved_genetic_algorithm(n_steps: int, max_iterations: int = 500) -> List[float]:
    """Improved genetic algorithm with adaptive parameters"""
    np.random.seed(42)
    random.seed(42)
    
    # Population parameters
    population_size = 60
    elite_size = 15
    mutation_rate = 0.1
    
    # Create initial population
    population = create_advanced_initial_population(n_steps, population_size)
    
    best_fitness = 0.0
    best_individual = None
    
    for generation in range(max_iterations):
        # Evaluate fitness of all individuals
        fitness_scores = []
        for individual in population:
            fitness = evaluate_c2(individual)
            fitness_scores.append((fitness, individual))
        
        # Sort by fitness (descending)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Track best solution
        current_best_fitness, current_best_individual = fitness_scores[0]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual.copy()
        
        # Print progress every 50 generations
        if generation % 50 == 0:
            print(f"Generation {generation}: Best C2 = {best_fitness:.6f}")
        
        # Selection: keep top individuals as elites
        elites = [ind for _, ind in fitness_scores[:elite_size]]
        
        # Create new population through crossover and mutation
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            # Tournament selection with varying tournament size
            tournament_size = max(3, 8 - generation // 50)  # Decrease tournament size over time
            parent1 = tournament_selection(fitness_scores, tournament_size)
            parent2 = tournament_selection(fitness_scores, tournament_size)
            
            # Crossover
            child = advanced_crossover(parent1, parent2)
            
            # Mutation with adaptive rate
            child = adaptive_mutation(child, generation, max_iterations)
            
            new_population.append(child)
        
        population = new_population
    
    return best_individual

def tournament_selection(fitness_scores, k: int):
    """Select individual using tournament selection"""
    tournament = random.sample(fitness_scores, min(k, len(fitness_scores)))
    winner = max(tournament, key=lambda x: x[0])
    return winner[1]

def construct_function() -> List[float]:
    """
    Construct optimized step function using advanced evolutionary algorithm
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Try different configurations to find best
    best_f = []
    best_c2 = 0.0
    
    # Test different step counts - we'll use larger ones for better resolution
    test_sizes = [200, 300, 400, 500]
    
    for n_steps in test_sizes:
        print(f"Testing with {n_steps} steps...")
        
        # Use improved genetic algorithm for this size
        try:
            f_candidate = improved_genetic_algorithm(n_steps, max_iterations=300)
            c2_val = evaluate_c2(f_candidate)
            
            if c2_val > best_c2:
                best_c2 = c2_val
                best_f = f_candidate.copy()
                
            print(f"Size {n_steps}: C2 = {c2_val:.6f}")
        except Exception as e:
            print(f"Error with size {n_steps}: {e}")
            continue
    
    # If we didn't find anything good, fall back to a good heuristic
    if not best_f:
        n_steps = 400
        # Create a good starting pattern - smooth bell curve
        f = [max(0, np.exp(-0.5 * ((i - (n_steps-1)/2) / ((n_steps-1)/4))**2)) for i in range(n_steps)]
        best_f = f
    
    # Final fine-tuning with gradient-based optimization
    print("Performing final fine-tuning with local optimization...")
    best_f = local_search_optimization(best_f)
    
    # Try one more optimization with scipy minimize
    try:
        best_f = scipy_local_optimization(best_f)
    except:
        pass
    
    return best_f

def local_search_optimization(initial_f: List[float], iterations: int = 500) -> List[float]:
    """Refine solution with enhanced local search"""
    current_f = initial_f.copy()
    current_c2 = evaluate_c2(current_f)
    
    # Track best so far
    best_f = current_f.copy()
    best_c2 = current_c2
    
    for i in range(iterations):
        # Try different types of perturbations
        candidate_f = current_f.copy()
        
        # Randomly choose perturbation type
        perturbation_type = np.random.choice(['small', 'medium', 'large'])
        
        if perturbation_type == 'small':
            num_perturbations = max(1, len(current_f) // 100)
            step_size = 0.01
        elif perturbation_type == 'medium':
            num_perturbations = max(1, len(current_f) // 50)
            step_size = 0.05
        else:  # large
            num_perturbations = max(1, len(current_f) // 20)
            step_size = 0.1
        
        # Perturb random points
        for _ in range(num_perturbations):
            idx = np.random.randint(len(candidate_f))
            delta = np.random.normal(0, step_size)
            candidate_f[idx] = max(0, candidate_f[idx] + delta)
        
        candidate_c2 = evaluate_c2(candidate_f)
        
        if candidate_c2 > current_c2:
            current_f = candidate_f
            current_c2 = candidate_c2
            
            # Update best if better
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_f = current_f.copy()
            
            if i % 100 == 0:
                print(f"Local search iteration {i}: C2 = {current_c2:.6f}")
    
    return best_f

def scipy_local_optimization(initial_f: List[float]) -> List[float]:
    """Use scipy optimization for final refinement"""
    def objective(f_vals):
        # We want to maximize C2, so we minimize -C2
        return -evaluate_c2(f_vals)
    
    # Create bounds (all values >= 0)
    bounds = [(0, 10.0) for _ in range(len(initial_f))]
    
    # Use L-BFGS-B optimizer which handles bounds well
    result = minimize(objective, initial_f, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 100, 'ftol': 1e-8})
    
    if result.success:
        return list(result.x)
    else:
        return initial_f

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
