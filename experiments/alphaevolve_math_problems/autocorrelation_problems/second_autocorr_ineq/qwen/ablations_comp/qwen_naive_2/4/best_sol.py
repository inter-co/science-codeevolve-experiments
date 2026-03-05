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
        
        # Avoid division by zero with stricter thresholds
        if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
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
    
    # Enhanced optimization strategies with better mathematical patterns
    strategies = [
        # Uniform distribution
        lambda size: [1.0] * size,
        # Random with constraint enforcement - more concentrated around 1.0
        lambda size: [max(0.0, np.random.normal(1.0, 0.1)) for _ in range(size)],
        # Step pattern - alternating high-low with more variation
        lambda size: [2.0 if i % 4 == 0 else (1.0 if i % 4 == 1 else 0.5) for i in range(size)],
        # Gaussian-like peak with sharper focus - optimized for C2
        lambda size: [np.exp(-((i - size//2)**2)/(2*(size//10)**2)) * 2.0 for i in range(size)],
        # Sinusoidal pattern with more complex shape - focused on positive values
        lambda size: [1.0 + 0.5 * np.sin(2 * np.pi * i / size) + 0.3 * np.cos(4 * np.pi * i / size) + 
                     0.1 * np.sin(6 * np.pi * i / size) for i in range(size)],
        # Double peak pattern - better separated peaks with more amplitude
        lambda size: [0.5 + 1.0 * np.exp(-((i - size//4)**2)/(2*(size//12)**2)) + 
                     1.0 * np.exp(-((i - 3*size//4)**2)/(2*(size//12)**2)) for i in range(size)],
        # Exponential decay pattern - exponential tail with higher base
        lambda size: [np.exp(-i/(size/5)) * 2.0 for i in range(size)],
        # Optimized multi-peak pattern - more evenly spaced and higher amplitude
        lambda size: [1.0 + 0.8 * np.sin(4 * np.pi * i / size) + 0.6 * np.cos(8 * np.pi * i / size) + 
                     0.4 * np.sin(12 * np.pi * i / size) + 0.2 * np.cos(16 * np.pi * i / size) for i in range(size)],
        # Asymmetric pattern with higher peaks - more structured
        lambda size: [max(0.0, 1.5 + 1.0 * np.sin(3 * np.pi * i / size) + 0.5 * np.cos(6 * np.pi * i / size) + 
                         0.3 * np.sin(9 * np.pi * i / size)) for i in range(size)],
        # Multi-hump pattern - more regular spacing with higher peaks
        lambda size: [1.0 + 0.6 * np.sin(2 * np.pi * i / size) + 0.5 * np.sin(4 * np.pi * i / size) + 
                     0.4 * np.sin(6 * np.pi * i / size) + 0.3 * np.sin(8 * np.pi * i / size) for i in range(size)]
    ]
    
    # Enhanced local search with better adaptation and more thorough exploration
    def enhanced_local_search(initial_solution: List[float], max_iterations: int = 3000) -> List[float]:
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
                delta2 = np.random.normal(0, step_size * 0.5)
                neighbor[idx2] = max(0.0, neighbor[idx2] + delta2)
                
            # Also try changing a third element occasionally
            if np.random.random() < 0.15:
                idx3 = np.random.randint(len(neighbor))
                delta3 = np.random.normal(0, step_size * 0.3)
                neighbor[idx3] = max(0.0, neighbor[idx3] + delta3)
            
            # Accept or reject based on improvement
            neighbor_c2 = evaluate_c2(neighbor)
            if neighbor_c2 > current_c2:
                current_solution = neighbor
                current_c2 = neighbor_c2
                if neighbor_c2 > best_local_c2:
                    best_local_c2 = neighbor_c2
                    best_local_solution = neighbor.copy()
            elif np.random.random() < 0.05:  # Sometimes accept worse solutions for escape
                current_solution = neighbor
                current_c2 = neighbor_c2
                
        return best_local_solution
    
    # Enhanced Bayesian optimization approach using Optuna
    def enhanced_bayesian_optimization():
        def objective(trial):
            # Sample parameters for the function
            size = trial.suggest_int('size', 200, 500)
            pattern_type = trial.suggest_categorical('pattern', ['gaussian', 'sinusoidal', 'multi_peak', 'custom', 'sharp_peaks'])
            
            if pattern_type == 'gaussian':
                # Create Gaussian pattern - sharper and more focused
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1  # Map to [-1, 1]
                    val = np.exp(-x**2 / 0.03) * trial.suggest_float('amplitude', 1.0, 3.0)
                    pattern.append(val)
            elif pattern_type == 'sinusoidal':
                # Create sinusoidal pattern - more harmonics
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1
                    val = 1.0 + trial.suggest_float('amp1', 0.5, 2.0) * np.sin(2 * np.pi * x * trial.suggest_float('freq1', 1, 10)) + \
                          trial.suggest_float('amp2', 0.2, 1.0) * np.cos(4 * np.pi * x * trial.suggest_float('freq2', 1, 10)) + \
                          trial.suggest_float('amp3', 0.1, 0.5) * np.sin(6 * np.pi * x * trial.suggest_float('freq3', 1, 10))
                    pattern.append(val)
            elif pattern_type == 'multi_peak':
                # Create multi-peak pattern - well-separated peaks
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1
                    val = 1.0 + 1.0 * np.exp(-((x - 0.4)**2)/0.06) + 0.8 * np.exp(-((x + 0.4)**2)/0.06) + \
                          0.6 * np.exp(-((x - 0.7)**2)/0.03) + 0.4 * np.exp(-((x + 0.7)**2)/0.03)
                    pattern.append(val)
            elif pattern_type == 'sharp_peaks':
                # Sharp peak pattern - very concentrated
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1
                    val = 1.0 + 1.5 * np.exp(-((x - 0.3)**2)/0.02) + 1.5 * np.exp(-((x + 0.3)**2)/0.02)
                    pattern.append(val)
            else:  # custom
                # Create a more complex custom pattern
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1
                    val = 1.0 + 0.8 * np.sin(2 * np.pi * x) + 0.6 * np.sin(4 * np.pi * x) + \
                          0.4 * np.sin(6 * np.pi * x) + 0.2 * np.sin(8 * np.pi * x) + \
                          0.1 * np.cos(10 * np.pi * x)
                    pattern.append(val)
            
            # Ensure non-negative values
            pattern = [max(0.0, p) for p in pattern]
            
            return evaluate_c2(pattern)
        
        # Run optimization with more trials and better timeout handling
        study = optuna.create_study(direction='maximize')
        try:
            study.optimize(objective, n_trials=200, timeout=max_time)
        except:
            pass  # Continue if timeout or other error
        
        if len(study.trials) > 0:
            # Extract the best configuration
            best_trial = study.best_trial
            size = best_trial.params['size']
            pattern_type = best_trial.params['pattern']
            
            # Recreate the best pattern with more careful handling
            if pattern_type == 'gaussian':
                pattern = [np.exp(-((i - size//2)**2)/(2*(size//15)**2)) * best_trial.params.get('amplitude', 2.0) 
                          for i in range(size)]
            elif pattern_type == 'sinusoidal':
                pattern = [1.0 + best_trial.params.get('amp1', 1.0) * np.sin(2 * np.pi * i / size * best_trial.params.get('freq1', 4)) + 
                          best_trial.params.get('amp2', 0.5) * np.cos(4 * np.pi * i / size * best_trial.params.get('freq2', 4)) + 
                          best_trial.params.get('amp3', 0.2) * np.sin(6 * np.pi * i / size * best_trial.params.get('freq3', 4)) 
                          for i in range(size)]
            elif pattern_type == 'multi_peak':
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1
                    val = 1.0 + 1.0 * np.exp(-((x - 0.4)**2)/0.06) + 0.8 * np.exp(-((x + 0.4)**2)/0.06) + \
                          0.6 * np.exp(-((x - 0.7)**2)/0.03) + 0.4 * np.exp(-((x + 0.7)**2)/0.03)
                    pattern.append(val)
            elif pattern_type == 'sharp_peaks':
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1
                    val = 1.0 + 1.5 * np.exp(-((x - 0.3)**2)/0.02) + 1.5 * np.exp(-((x + 0.3)**2)/0.02)
                    pattern.append(val)
            else:
                pattern = [1.0 + 0.8 * np.sin(2 * np.pi * i / size) + 0.6 * np.sin(4 * np.pi * i / size) + \
                          0.4 * np.sin(6 * np.pi * i / size) + 0.2 * np.sin(8 * np.pi * i / size) + \
                          0.1 * np.cos(10 * np.pi * i / size) for i in range(size)]
            
            pattern = [max(0.0, p) for p in pattern]
            return pattern, evaluate_c2(pattern)
        else:
            return [], -1e10
    
    # Enhanced evolutionary algorithm with better diversity management
    def enhanced_evolutionary_optimization():
        # Parameters for evolution - more aggressive
        population_size = 200
        generations = 400
        mutation_rate = 0.2
        crossover_rate = 0.9
        
        # Initialize diverse population with better starting conditions
        population = []
        for i in range(population_size):
            if time.time() - start_time > max_time:
                break
                
            # Mix of strategies with better parameter diversity
            strategy_idx = i % len(strategies)
            size = 200 + (i % 20) * 15  # Different sizes for diversity
            individual = strategies[strategy_idx](size)
            # Add some random noise to increase diversity
            individual = [max(0.0, val + np.random.normal(0, 0.03 * val if val > 0 else 0.03)) for val in individual]
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
                tournament_size = 8
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
    
    # Mathematical pattern optimization with more precise construction
    def mathematical_pattern_optimization():
        best_result = []
        best_score = -1e10
        
        # Try patterns that have shown good performance mathematically
        patterns_to_try = []
        
        # Pattern 1: Multi-bell pattern with higher amplitude peaks
        for size in [300, 350, 400]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1  # Map to [-1, 1]
                # Create a pattern with multiple smooth peaks - optimized for C2
                val = 1.0 + 1.2 * np.exp(-((x - 0.3)**2)/0.05) + 1.0 * np.exp(-((x + 0.3)**2)/0.05) + \
                      0.8 * np.exp(-((x - 0.7)**2)/0.03) + 0.6 * np.exp(-((x + 0.7)**2)/0.03)
                pattern.append(val)
            patterns_to_try.append(('multi_bell', pattern))
        
        # Pattern 2: Sharp peak pattern - optimized for maximum C2
        for size in [300, 350, 400]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Sharp concentrated peaks for maximum autoconvolution peak
                val = 1.0 + 1.8 * np.exp(-((x - 0.2)**2)/0.02) + 1.8 * np.exp(-((x + 0.2)**2)/0.02)
                pattern.append(val)
            patterns_to_try.append(('sharp_peaks', pattern))
            
        # Pattern 3: Multi-sinc pattern - highly oscillating but controlled
        for size in [300, 350, 400]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Sinc-like pattern with higher amplitude
                if abs(x) < 1e-6:
                    val = 1.0
                else:
                    val = 1.0 + 1.0 * np.sin(4 * np.pi * x) / (4 * np.pi * x) + \
                          0.8 * np.sin(8 * np.pi * x) / (8 * np.pi * x) + \
                          0.6 * np.sin(12 * np.pi * x) / (12 * np.pi * x) + \
                          0.4 * np.sin(16 * np.pi * x) / (16 * np.pi * x)
                pattern.append(max(0.0, val))
            patterns_to_try.append(('multi_sinc', pattern))
            
        # Pattern 4: Optimized symmetric pattern - balanced oscillation
        for size in [300, 350, 400]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Balanced symmetric pattern
                val = 1.0 + 0.9 * np.tanh(4 * x) + 0.7 * np.tanh(8 * x) + 0.5 * np.tanh(12 * x)
                pattern.append(val)
            patterns_to_try.append(('symmetric_balanced', pattern))
        
        # Evaluate all patterns
        for name, pattern in patterns_to_try:
            if time.time() - start_time > max_time:
                break
            # Fine-tune with local search
            refined = enhanced_local_search(pattern, max_iterations=1000)
            score = evaluate_c2(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        
        return best_result, best_score
    
    # New aggressive gradient-free optimization approach
    def aggressive_gradient_free_optimization():
        # Try direct optimization with a few key strategies
        best_pattern = []
        best_c2_val = -1e10
        
        # Strategy 1: High-amplitude concentrated peaks
        for size in [350, 400, 450]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Very sharp peaks to create strong autoconvolution
                val = 1.0 + 2.0 * np.exp(-((x - 0.25)**2)/0.015) + 2.0 * np.exp(-((x + 0.25)**2)/0.015)
                pattern.append(val)
            refined = enhanced_local_search(pattern, max_iterations=1500)
            score = evaluate_c2(refined)
            if score > best_c2_val:
                best_c2_val = score
                best_pattern = refined.copy()
        
        # Strategy 2: Multi-peak with moderate amplitudes
        for size in [350, 400, 450]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Multiple moderate peaks
                val = 1.0 + 1.5 * np.exp(-((x - 0.3)**2)/0.04) + 1.5 * np.exp(-((x + 0.3)**2)/0.04) + \
                      1.0 * np.exp(-((x - 0.7)**2)/0.03) + 1.0 * np.exp(-((x + 0.7)**2)/0.03)
                pattern.append(val)
            refined = enhanced_local_search(pattern, max_iterations=1500)
            score = evaluate_c2(refined)
            if score > best_c2_val:
                best_c2_val = score
                best_pattern = refined.copy()
        
        return best_pattern, best_c2_val
    
    # Execute strategies
    try:
        # Strategy 1: Aggressive gradient-free optimization
        aggr_result, aggr_score = aggressive_gradient_free_optimization()
        if aggr_score > best_c2:
            best_c2 = aggr_score
            best_solution = aggr_result.copy()
            
        # Strategy 2: Enhanced Bayesian optimization 
        bayes_result, bayes_score = enhanced_bayesian_optimization()
        if bayes_score > best_c2:
            best_c2 = bayes_score
            best_solution = bayes_result.copy()
            
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
            fine_tuned = enhanced_local_search(best_solution, max_iterations=5000)
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
