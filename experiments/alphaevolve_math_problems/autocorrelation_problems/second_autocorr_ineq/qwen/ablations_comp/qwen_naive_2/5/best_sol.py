# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import pdist
import time
import random
from typing import List, Tuple
import warnings
from scipy import signal
from scipy.optimize import minimize_scalar
import numba
from scipy.optimize import dual_annealing
import optuna
import itertools

# Suppress any potential warnings
warnings.filterwarnings('ignore')

@numba.jit(nopython=True)
def compute_autoconvolution_norms_numba(f_values: np.ndarray) -> Tuple[float, float, float]:
    """
    Fast computation of autoconvolution norms using numba JIT compilation.
    """
    if len(f_values) == 0:
        return 0.0, 0.0, 0.0
    
    # Create step function on [-1/4, 1/4] with equal spacing
    n = len(f_values)
    
    # Perform convolution manually for better control
    # For two functions on [-1/4, 1/4], convolution result spans [-1/2, 1/2]
    g_len = 2 * n - 1
    g = np.zeros(g_len)
    
    # Manual convolution implementation for better accuracy
    for i in range(n):
        for j in range(n):
            idx = i + j
            if 0 <= idx < g_len:
                g[idx] += f_values[i] * f_values[j]
    
    # Properly scale the domain for the convolution result
    # When convolving two functions defined on [-1/4, 1/4], the result is defined on [-1/2, 1/2]
    domain_width = 1.0  # From -0.5 to 0.5
    step_size = domain_width / (g_len - 1)
    
    # Compute ||g||₂² using more accurate piecewise quadratic integration for ||g||₂²
    g_norm_2_sq = 0.0
    
    # Use piecewise quadratic integration for ∫g² dx: 
    # For interval with heights y1, y2 and width h, contribution is (h/3)(y1² + y1*y2 + y2²)
    for i in range(len(g)-1):
        h = step_size
        y1 = g[i]
        y2 = g[i+1]
        g_norm_2_sq += (h/3) * (y1**2 + y1*y2 + y2**2)
    
    # Compute ||g||₁ as sum of absolute values divided by number of points
    g_norm_1 = np.sum(np.abs(g)) / len(g) if len(g) > 0 else 0.0
    
    # Compute ||g||∞ as maximum absolute value
    g_norm_inf = np.max(np.abs(g))
    
    return g_norm_2_sq, g_norm_1, g_norm_inf

def compute_autoconvolution_norms(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Compute the three norms for the autoconvolution of a step function.
    
    Args:
        f_values: List of step heights
        
    Returns:
        Tuple of (||g||₂², ||g||₁, ||g||∞)
    """
    if not f_values:
        return 0.0, 0.0, 0.0
    
    # Convert to numpy array for faster processing
    f_array = np.array(f_values)
    
    # Use numba-compiled version for speed
    return compute_autoconvolution_norms_numba(f_array)

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞) for given step function values.
    
    Returns negative value since we want to maximize C₂ (minimize negative).
    """
    try:
        g_norm_2_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_norm_1 <= 1e-12 or g_norm_inf <= 1e-12:
            return -1e10  # Invalid case
            
        c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
        return c2
    except Exception:
        return -1e10  # Invalid case

def construct_function() -> List[float]:
    """
    Construct a step function using advanced optimization strategies to maximize C₂.
    Implements a hybrid approach combining multiple optimization techniques.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    start_time = time.time()
    max_time = 55  # Leave 5 seconds for final processing
    
    best_c2 = -1e10
    best_solution = []
    
    # Enhanced optimization strategies focusing on proven mathematical patterns
    strategies = [
        # Uniform distribution
        lambda size: [1.0] * size,
        # Optimized Gaussian-like peak with sharper focus
        lambda size: [np.exp(-((i - size//2)**2)/(2*(size//10)**2)) * 2.0 for i in range(size)],
        # Multi-peak pattern optimized for autoconvolution
        lambda size: [0.5 + 1.0 * np.exp(-((i - size//3)**2)/(2*(size//15)**2)) + 
                     1.0 * np.exp(-((i - 2*size//3)**2)/(2*(size//15)**2)) for i in range(size)],
        # Sinusoidal pattern with optimized frequencies
        lambda size: [1.0 + 0.6 * np.sin(4 * np.pi * i / size) + 0.4 * np.cos(8 * np.pi * i / size) + 
                     0.2 * np.sin(12 * np.pi * i / size) for i in range(size)],
        # Asymmetric multi-peak pattern
        lambda size: [1.0 + 0.8 * np.exp(-((i - size//4)**2)/(2*(size//12)**2)) + 
                     0.6 * np.exp(-((i - 3*size//4)**2)/(2*(size//12)**2)) for i in range(size)],
        # Exponential pattern with better control
        lambda size: [np.exp(-i/(size/5)) * 1.5 for i in range(size)],
        # Complex oscillating pattern
        lambda size: [1.0 + 0.5 * np.sin(2 * np.pi * i / size) + 0.3 * np.sin(6 * np.pi * i / size) + 
                     0.2 * np.cos(4 * np.pi * i / size) + 0.1 * np.cos(10 * np.pi * i / size) for i in range(size)],
        # Sharp peak pattern
        lambda size: [1.0 + 0.8 * np.exp(-((i - size//2)**2)/(2*(size//20)**2)) for i in range(size)],
        # Spiky pattern for high C2
        lambda size: [1.0 + 0.5 * (1.0 if i % 10 == 0 else 0.0) for i in range(size)],
        # Concentrated double peak with proper spacing
        lambda size: [0.7 + 0.8 * np.exp(-((i - size//3)**2)/(2*(size//15)**2)) + 
                     0.8 * np.exp(-((i - 2*size//3)**2)/(2*(size//15)**2)) for i in range(size)]
    ]
    
    # Enhanced local search with improved adaptation and more thorough exploration
    def enhanced_local_search(initial_solution: List[float], max_iterations: int = 5000) -> List[float]:
        current_solution = initial_solution.copy()
        current_c2 = evaluate_c2(current_solution)
        
        # Track best solution during local search
        best_local_solution = current_solution.copy()
        best_local_c2 = current_c2
        
        # Adaptive perturbation with better acceptance criteria
        for iteration in range(max_iterations):
            if time.time() - start_time > max_time:
                break
                
            # Create neighbor solution with varying perturbation strength
            neighbor = current_solution.copy()
            idx = np.random.randint(len(neighbor))
            
            # Use adaptive step size based on current solution and iteration
            step_size = 0.01 + 0.1 * np.random.random() * (1.0 - iteration/max_iterations)
            delta = np.random.normal(0, step_size)
            neighbor[idx] = max(0.0, neighbor[idx] + delta)
            
            # Try to improve by changing multiple elements
            if np.random.random() < 0.3:
                # Change another element
                idx2 = np.random.randint(len(neighbor))
                delta2 = np.random.normal(0, step_size * 0.7)
                neighbor[idx2] = max(0.0, neighbor[idx2] + delta2)
                
            # Also try changing a third element occasionally
            if np.random.random() < 0.15:
                idx3 = np.random.randint(len(neighbor))
                delta3 = np.random.normal(0, step_size * 0.5)
                neighbor[idx3] = max(0.0, neighbor[idx3] + delta3)
            
            # Accept or reject based on improvement with simulated annealing-like cooling
            neighbor_c2 = evaluate_c2(neighbor)
            if neighbor_c2 > current_c2:
                current_solution = neighbor
                current_c2 = neighbor_c2
                if neighbor_c2 > best_local_c2:
                    best_local_c2 = neighbor_c2
                    best_local_solution = neighbor.copy()
            else:
                # Accept worse solutions with probability based on temperature
                temp = 1.0 - iteration / max_iterations
                if np.random.random() < np.exp((neighbor_c2 - current_c2) / (temp + 1e-10)):
                    current_solution = neighbor
                    current_c2 = neighbor_c2
                
        return best_local_solution
    
    # Enhanced evolutionary algorithm with better diversity management and improved operators
    def enhanced_evolutionary_optimization():
        # Parameters for evolution - more aggressive and better tuned
        population_size = 150
        generations = 500
        mutation_rate = 0.25
        crossover_rate = 0.85
        
        # Initialize diverse population with better starting conditions
        population = []
        for i in range(population_size):
            if time.time() - start_time > max_time:
                break
                
            # Mix of strategies with better parameter diversity
            strategy_idx = i % len(strategies)
            size = 250 + (i % 20) * 15  # Different sizes for diversity
            individual = strategies[strategy_idx](size)
            # Add some random noise to increase diversity
            individual = [max(0.0, val + np.random.normal(0, 0.05 * val if val > 0 else 0.05)) for val in individual]
            population.append(individual)
        
        # Evolution loop with better termination and selection
        for generation in range(generations):
            if time.time() - start_time > max_time:
                break
                
            # Evaluate fitness
            fitness_scores = [evaluate_c2(ind) for ind in population]
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            top_individuals = [population[i] for i in sorted_indices[:population_size//3]]
            top_fitness = [fitness_scores[i] for i in sorted_indices[:population_size//3]]
            
            # Keep best individuals (elitism) - more elitism for better convergence
            new_population = top_individuals.copy()
            
            # Generate offspring with more sophisticated crossover
            while len(new_population) < population_size:
                if time.time() - start_time > max_time:
                    break
                    
                # Tournament selection with larger tournaments for better selection pressure
                tournament_size = 6
                tournament_indices = np.random.choice(len(top_individuals), tournament_size, replace=False)
                tournament_fitness = [top_fitness[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                # Another parent
                tournament_indices2 = np.random.choice(len(top_individuals), tournament_size, replace=False)
                tournament_fitness2 = [top_fitness[i] for i in tournament_indices2]
                parent2_idx = tournament_indices2[np.argmax(tournament_fitness2)]
                
                parent1 = top_individuals[parent1_idx]
                parent2 = top_individuals[parent2_idx]
                
                # Crossover with different strategies
                if np.random.random() < crossover_rate:
                    # Blend crossover for smoother transitions
                    alpha = np.random.random()
                    child = []
                    min_len = min(len(parent1), len(parent2))
                    for i in range(min_len):
                        child.append(alpha * parent1[i] + (1 - alpha) * parent2[i])
                    
                    # Extend with longer parent if needed
                    if len(parent1) > len(parent2):
                        child.extend(parent1[min_len:])
                    elif len(parent2) > len(parent1):
                        child.extend(parent2[min_len:])
                else:
                    # Clone one parent
                    child = parent1.copy() if np.random.random() < 0.5 else parent2.copy()
                
                # Mutation with adaptive rates
                for i in range(len(child)):
                    if np.random.random() < mutation_rate:
                        # Adaptive mutation based on current value and position
                        mutation_strength = 0.1 * child[i] if child[i] > 0 else 0.1
                        child[i] = max(0.0, child[i] + np.random.normal(0, mutation_strength))
                
                new_population.append(child)
            
            population = new_population[:population_size]
            
        # Return best individual
        final_fitness = [evaluate_c2(ind) for ind in population]
        if len(final_fitness) > 0:
            best_idx = np.argmax(final_fitness)
            return population[best_idx], final_fitness[best_idx]
        else:
            return [], -1e10
    
    # Mathematical pattern optimization with better understanding of what works
    def mathematical_pattern_optimization():
        best_result = []
        best_score = -1e10
        
        # Try patterns that have shown good performance mathematically
        patterns_to_try = []
        
        # Pattern 1: Optimized multi-peak pattern with mathematical precision
        for size in [350, 400, 450]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1  # Map to [-1, 1]
                # Create a mathematically optimized pattern with controlled peaks
                val = 1.0 + 0.9 * np.exp(-((x - 0.3)**2)/0.04) + 0.9 * np.exp(-((x + 0.3)**2)/0.04) + \
                      0.7 * np.exp(-((x - 0.7)**2)/0.03) + 0.7 * np.exp(-((x + 0.7)**2)/0.03)
                pattern.append(val)
            patterns_to_try.append(('optimized_multi_peak', pattern))
        
        # Pattern 2: Highly concentrated pattern with mathematical optimization
        for size in [350, 400, 450]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Very sharp peaks for maximum autoconvolution concentration
                val = 1.0 + 1.3 * np.exp(-((x - 0.25)**2)/0.02) + 1.3 * np.exp(-((x + 0.25)**2)/0.02)
                pattern.append(val)
            patterns_to_try.append(('sharp_concentrated', pattern))
            
        # Pattern 3: Optimized oscillatory pattern
        for size in [350, 400, 450]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Optimized oscillation pattern
                val = 1.0 + 0.8 * np.sin(8 * np.pi * x) + 0.6 * np.cos(16 * np.pi * x) + \
                      0.4 * np.sin(24 * np.pi * x) + 0.2 * np.cos(32 * np.pi * x)
                pattern.append(val)
            patterns_to_try.append(('oscillatory', pattern))
            
        # Pattern 4: Hybrid pattern with both smooth and sharp features
        for size in [350, 400, 450]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Combination of smooth and sharp components
                smooth_component = 0.9 * np.exp(-((x - 0.1)**2)/0.05) + 0.9 * np.exp(-((x + 0.1)**2)/0.05)
                sharp_component = 0.6 * np.exp(-((x - 0.5)**2)/0.02) + 0.6 * np.exp(-((x + 0.5)**2)/0.02)
                val = 1.0 + smooth_component + sharp_component
                pattern.append(val)
            patterns_to_try.append(('hybrid', pattern))
        
        # Evaluate all patterns
        for name, pattern in patterns_to_try:
            if time.time() - start_time > max_time:
                break
            # Fine-tune with local search
            refined = enhanced_local_search(pattern, max_iterations=1500)
            score = evaluate_c2(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        
        return best_result, best_score
    
    # Enhanced gradient-free optimization using multiple approaches
    def advanced_gradient_free_optimization():
        # Try different optimization methods with different starting points
        best_result = []
        best_score = -1e10
        
        # Method 1: Direct search with systematic grid exploration
        for size in [350, 400, 450]:
            if time.time() - start_time > max_time:
                break
                
            # Try a few key patterns systematically
            patterns = [
                [1.0 + 0.9 * np.exp(-((i - size//4)**2)/(2*(size//15)**2)) + 
                 0.9 * np.exp(-((i - 3*size//4)**2)/(2*(size//15)**2)) for i in range(size)],  # Double peak
                [1.0 + 1.0 * np.exp(-((i - size//2)**2)/(2*(size//20)**2)) for i in range(size)],  # Single sharp peak
                [1.0 + 0.7 * np.sin(8 * np.pi * i / size) + 0.5 * np.cos(16 * np.pi * i / size) for i in range(size)]  # Oscillatory
            ]
            
            for pattern in patterns:
                refined = enhanced_local_search(pattern, max_iterations=1000)
                score = evaluate_c2(refined)
                if score > best_score:
                    best_score = score
                    best_result = refined.copy()
        
        return best_result, best_score
    
    # Additional specialized optimization technique for better C2
    def specialized_optimization_approach():
        # Focus on finding patterns that produce flatter autoconvolutions
        # This is more likely to maximize C₂ = ||g||₂² / (||g||₁ · ||g||∞)
        best_result = []
        best_score = -1e10
        
        # Try patterns with more uniform autoconvolution profiles
        for size in [400, 450, 500]:
            if time.time() - start_time > max_time:
                break
                
            # Pattern: Smooth, well-distributed peaks that create flat autoconvolution
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Create a pattern that balances height and spread
                val = 1.0 + 0.8 * np.exp(-((x - 0.2)**2)/0.03) + 0.8 * np.exp(-((x + 0.2)**2)/0.03) + \
                      0.6 * np.exp(-((x - 0.6)**2)/0.04) + 0.6 * np.exp(-((x + 0.6)**2)/0.04)
                pattern.append(val)
            
            # Refine with more intensive local search
            refined = enhanced_local_search(pattern, max_iterations=2000)
            score = evaluate_c2(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
                
        return best_result, best_score
    
    # Execute strategies in order of likelihood to find best results
    try:
        # Strategy 1: Specialized optimization approach
        spec_result, spec_score = specialized_optimization_approach()
        if spec_score > best_c2:
            best_c2 = spec_score
            best_solution = spec_result.copy()
            
        # Strategy 2: Advanced gradient-free optimization 
        adv_result, adv_score = advanced_gradient_free_optimization()
        if adv_score > best_c2:
            best_c2 = adv_score
            best_solution = adv_result.copy()
            
        # Strategy 3: Enhanced evolutionary optimization  
        evol_result, evol_score = enhanced_evolutionary_optimization()
        if evol_score > best_c2:
            best_c2 = evol_score
            best_solution = evol_result.copy()
            
        # Strategy 4: Mathematical pattern optimization
        math_result, math_score = mathematical_pattern_optimization()
        if math_score > best_c2:
            best_c2 = math_score
            best_solution = math_result.copy()
            
        # Strategy 5: Local refinement of best solution found so far with more iterations
        if len(best_solution) > 0:
            # Fine-tune the best solution found
            fine_tuned = enhanced_local_search(best_solution, max_iterations=8000)
            fine_score = evaluate_c2(fine_tuned)
            if fine_score > best_c2:
                best_c2 = fine_score
                best_solution = fine_tuned.copy()
                
    except Exception as e:
        # Fallback to simple approach if something goes wrong
        print(f"Fallback due to error: {e}")
        # Simple approach: try a few different configurations
        for _ in range(100):
            if time.time() - start_time > max_time:
                break
            size = np.random.randint(300, 500)
            init = [max(0.0, np.random.normal(1.0, 0.1)) for _ in range(size)]
            score = evaluate_c2(init)
            if score > best_c2:
                best_c2 = score
                best_solution = init.copy()
    
    # Ensure we have a valid solution
    if not best_solution:
        # Fallback to optimized uniform distribution with better parameters
        best_solution = [1.0] * 400
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
