# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import time
import random
from typing import List, Tuple
import warnings
from scipy import signal
from scipy.integrate import simpson
import math
from scipy.optimize import minimize_scalar
import numba
from numba import jit
import optuna
import matplotlib.pyplot as plt

# Suppress any potential warnings
warnings.filterwarnings('ignore')

@jit(nopython=True)
def compute_autoconvolution_norms_numba(f_values: List[float]) -> Tuple[float, float, float]:
    """
    Fast computation of autoconvolution norms using numba for speed.
    """
    if len(f_values) == 0:
        return 0.0, 0.0, 0.0
    
    # Convert to numpy array for faster operations
    f = np.array(f_values)
    n = len(f)
    
    # Compute convolution manually for better control
    # Autoconvolution: g[k] = sum_{i=0}^{n-1} f[i] * f[k-i] where indices wrap around
    g = np.zeros(2 * n - 1)
    
    # Manual computation for efficiency
    for k in range(2 * n - 1):
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                g[k] += f[i] * f[j]
    
    # Step size for convolution domain [-0.5, 0.5] with 2*n-1 points
    g_step_size = 1.0 / (len(g) - 1)
    
    # Compute ||g||₂² using piecewise quadratic integration (trapezoidal rule for ∫g² dx)
    g_norm_2_sq = 0.0
    for i in range(len(g)-1):
        h = g_step_size
        y1 = g[i]
        y2 = g[i+1]
        # For ∫g² dx: use trapezoidal rule approximation (h/3)(y1² + y1*y2 + y2²)
        g_norm_2_sq += (h/3) * (y1**2 + y1*y2 + y2**2)
    
    # Compute ||g||₁ as sum of absolute values divided by number of points
    g_norm_1 = np.sum(np.abs(g)) / len(g)
    
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
    return compute_autoconvolution_norms_numba(f_values)

def evaluate_c2(f_values: List[float]) -> float:
    """
    Evaluate C₂ = ||g||₂² / (||g||₁ · ||g||∞) for given step function values.
    
    Returns negative value since we want to maximize C₂ (minimize negative).
    """
    try:
        g_norm_2_sq, g_norm_1, g_norm_inf = compute_autoconvolution_norms(f_values)
        
        # Avoid division by zero
        if g_norm_1 <= 1e-15 or g_norm_inf <= 1e-15:
            return -1e10  # Invalid case
            
        c2 = g_norm_2_sq / (g_norm_1 * g_norm_inf)
        return c2
    except Exception as e:
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
    
    # Enhanced optimization strategies focused on maximizing C₂
    strategies = [
        # Uniform distribution
        lambda size: [1.0] * size,
        # Optimized Gaussian peak with proper normalization
        lambda size: [math.exp(-((i - size//2)**2)/(2*(size//8)**2)) * 1.5 for i in range(size)],
        # Multi-peak pattern designed to create flat convolution
        lambda size: [1.0 + 0.5 * math.sin(4 * math.pi * i / size) + 0.2 * math.cos(8 * math.pi * i / size) for i in range(size)],
        # Alternating pattern with strategic peaks
        lambda size: [1.0 if i % 4 == 0 else (0.8 if i % 4 == 1 else 0.6 if i % 4 == 2 else 0.4) for i in range(size)],
        # Optimized double peak
        lambda size: [0.7 + 0.8 * math.exp(-((i - size//3)**2)/(2*(size//10)**2)) + 
                     0.8 * math.exp(-((i - 2*size//3)**2)/(2*(size//10)**2)) for i in range(size)],
        # Sine wave pattern with amplitude variation
        lambda size: [1.0 + 0.5 * math.sin(2 * math.pi * i / size) for i in range(size)],
        # Exponential decay pattern with higher peak
        lambda size: [math.exp(-i/(size/4)) * 2.0 for i in range(size)],
        # Optimized bell-shaped pattern
        lambda size: [1.0 + 0.3 * math.exp(-((i - size//2)**2)/(2*(size//6)**2)) for i in range(size)],
        # Triangular pattern with peak at center
        lambda size: [1.0 - abs(i - size//2) / (size//2) for i in range(size)],
        # Piecewise linear pattern with strategic steps
        lambda size: [1.0 if i < size//3 else (0.5 if i < 2*size//3 else 0.2) for i in range(size)],
        # Modified sigmoid pattern
        lambda size: [1.0 / (1.0 + math.exp(-0.1 * (i - size//2))) for i in range(size)],
        # Combination of peaks with decreasing amplitude
        lambda size: [1.0 * math.exp(-i**2/(2*(size//10)**2)) + 0.5 * math.exp(-(i-size//2)**2/(2*(size//15)**2)) + 
                     0.3 * math.exp(-(i-2*size//3)**2/(2*(size//20)**2)) for i in range(size)]
    ]
    
    # Better local search with adaptive step sizes and smarter acceptance criteria
    def improved_local_search(initial_solution: List[float], max_iterations: int = 1000) -> List[float]:
        current_solution = initial_solution.copy()
        current_c2 = evaluate_c2(current_solution)
        
        # Track recent improvements for adaptive step sizing
        recent_improvements = []
        
        for iteration in range(max_iterations):
            if time.time() - start_time > max_time:
                break
                
            # Create neighbor solution with adaptive perturbation
            neighbor = current_solution.copy()
            idx = np.random.randint(len(neighbor))
            
            # Adaptive step size based on performance history and iteration
            base_step = 0.1
            if len(recent_improvements) > 0:
                avg_improvement = np.mean(recent_improvements[-10:]) if len(recent_improvements) >= 10 else np.mean(recent_improvements)
                # Adjust step size based on recent improvement rate
                step_multiplier = 1.0 + max(0, avg_improvement) * 0.5
                step_size = base_step * step_multiplier
            else:
                step_size = base_step
            
            # Apply perturbation with bounded normal distribution
            delta = np.random.normal(0, step_size)
            neighbor[idx] = max(0.0, neighbor[idx] + delta)
            
            # Occasionally try multiple changes for better exploration
            if np.random.random() < 0.2:
                idx2 = np.random.randint(len(neighbor))
                delta2 = np.random.normal(0, step_size * 0.7)
                neighbor[idx2] = max(0.0, neighbor[idx2] + delta2)
            
            # Evaluate neighbor
            neighbor_c2 = evaluate_c2(neighbor)
            
            # Smart acceptance criteria
            if neighbor_c2 > current_c2:
                current_solution = neighbor
                current_c2 = neighbor_c2
                recent_improvements.append(neighbor_c2 - current_c2)
            elif np.random.random() < 0.05:  # Sometimes accept worse solutions for escape
                current_solution = neighbor
                current_c2 = neighbor_c2
                recent_improvements.append(0)
            else:
                recent_improvements.append(0)
                
            # Keep recent improvements list bounded
            if len(recent_improvements) > 20:
                recent_improvements.pop(0)
                
        return current_solution
    
    # Improved optimization using Bayesian optimization with Optuna
    def bayesian_optimization_approach():
        def objective(trial):
            # Define the search space for the step function
            size = trial.suggest_int('size', 150, 400)
            
            # Create a pattern with varying characteristics
            pattern_type = trial.suggest_categorical('pattern', ['gaussian', 'sine', 'double_peak', 'mixed'])
            
            if pattern_type == 'gaussian':
                # Gaussian pattern with tunable parameters
                peak_center = trial.suggest_int('center', size//4, 3*size//4)
                peak_width = trial.suggest_int('width', size//20, size//5)
                peak_height = trial.suggest_float('height', 0.5, 2.0)
                f_values = [peak_height * math.exp(-((i - peak_center)**2)/(2*peak_width**2)) for i in range(size)]
                
            elif pattern_type == 'sine':
                # Sine pattern with frequency and amplitude tuning
                frequency = trial.suggest_float('freq', 0.5, 8.0)
                amplitude = trial.suggest_float('amp', 0.3, 1.5)
                baseline = trial.suggest_float('baseline', 0.5, 1.5)
                f_values = [baseline + amplitude * math.sin(frequency * 2 * math.pi * i / size) for i in range(size)]
                
            elif pattern_type == 'double_peak':
                # Double peak pattern
                peak1_center = trial.suggest_int('peak1_center', size//4, size//2)
                peak2_center = trial.suggest_int('peak2_center', size//2, 3*size//4)
                peak1_width = trial.suggest_int('peak1_width', size//20, size//10)
                peak2_width = trial.suggest_int('peak2_width', size//20, size//10)
                peak1_height = trial.suggest_float('peak1_height', 0.5, 2.0)
                peak2_height = trial.suggest_float('peak2_height', 0.5, 2.0)
                
                f_values = []
                for i in range(size):
                    val1 = peak1_height * math.exp(-((i - peak1_center)**2)/(2*peak1_width**2))
                    val2 = peak2_height * math.exp(-((i - peak2_center)**2)/(2*peak2_width**2))
                    f_values.append(val1 + val2)
                    
            else:  # mixed pattern
                # Mixed pattern with multiple components
                f_values = []
                for i in range(size):
                    val = 1.0 + 0.3 * math.sin(4 * math.pi * i / size) + \
                          0.2 * math.cos(8 * math.pi * i / size) + \
                          0.15 * math.sin(12 * math.pi * i / size)
                    f_values.append(max(0.0, val))
            
            return evaluate_c2(f_values)
        
        # Run optimization with Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, timeout=max_time)
        
        if study.best_trial:
            # Reconstruct the best solution
            best_params = study.best_trial.params
            size = best_params['size']
            pattern_type = best_params['pattern']
            
            if pattern_type == 'gaussian':
                peak_center = best_params['center']
                peak_width = best_params['width']
                peak_height = best_params['height']
                return [peak_height * math.exp(-((i - peak_center)**2)/(2*peak_width**2)) for i in range(size)]
            elif pattern_type == 'sine':
                frequency = best_params['freq']
                amplitude = best_params['amp']
                baseline = best_params['baseline']
                return [baseline + amplitude * math.sin(frequency * 2 * math.pi * i / size) for i in range(size)]
            elif pattern_type == 'double_peak':
                peak1_center = best_params['peak1_center']
                peak2_center = best_params['peak2_center']
                peak1_width = best_params['peak1_width']
                peak2_width = best_params['peak2_width']
                peak1_height = best_params['peak1_height']
                peak2_height = best_params['peak2_height']
                f_values = []
                for i in range(size):
                    val1 = peak1_height * math.exp(-((i - peak1_center)**2)/(2*peak1_width**2))
                    val2 = peak2_height * math.exp(-((i - peak2_center)**2)/(2*peak2_width**2))
                    f_values.append(val1 + val2)
                return f_values
            else:  # mixed
                return [1.0 + 0.3 * math.sin(4 * math.pi * i / size) + \
                       0.2 * math.cos(8 * math.pi * i / size) + \
                       0.15 * math.sin(12 * math.pi * i / size) for i in range(size)]
        
        return []

    # Enhanced evolutionary approach with better diversity maintenance
    def enhanced_evolutionary_optimization():
        # Initialize population with diverse strategies
        population_size = 40
        population = []
        
        # Create diverse initial individuals with better pattern selection
        for i in range(population_size):
            if time.time() - start_time > max_time:
                break
            strategy_idx = i % len(strategies)
            # Use more varied sizes to encourage exploration
            size = 100 + (i % 10) * 30 + (i // 10) * 20
            individual = strategies[strategy_idx](size)
            population.append(individual)
        
        # Evolutionary process with better selection and diversity
        for generation in range(150):  # Reduced generations but more intelligent selection
            if time.time() - start_time > max_time:
                break
                
            # Evaluate fitness
            fitness_scores = [evaluate_c2(ind) for ind in population]
            
            # Tournament selection with adaptive tournament size
            selected_indices = []
            tournament_size = max(3, min(8, len(population) // 5))  # Adaptive tournament size
            
            for _ in range(population_size // 2):
                tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected_indices.append(winner_idx)
            
            selected = [population[i] for i in selected_indices]
            
            # Create new population through crossover and mutation
            new_population = selected.copy()
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-5:]
            for i in elite_indices:
                if len(new_population) < population_size:
                    new_population.append(population[i].copy())
            
            # Generate offspring through crossover and mutation
            while len(new_population) < population_size:
                if time.time() - start_time > max_time:
                    break
                    
                # Selection with probability proportional to fitness (softmax)
                fitness_probs = np.array(fitness_scores)
                fitness_probs = np.maximum(fitness_probs, 0)  # Ensure non-negative
                if np.sum(fitness_probs) > 0:
                    # Softmax for better probability distribution
                    exp_probs = np.exp(fitness_probs - np.max(fitness_probs))
                    fitness_probs = exp_probs / np.sum(exp_probs)
                    parent1_idx, parent2_idx = np.random.choice(len(population), 2, p=fitness_probs)
                else:
                    parent1_idx, parent2_idx = np.random.choice(len(population), 2)
                
                parent1, parent2 = population[parent1_idx], population[parent2_idx]
                
                # Crossover: blend with better control
                child = []
                min_len = min(len(parent1), len(parent2))
                
                # Use more controlled blending with crossover points
                crossover_point = np.random.randint(1, min_len)
                
                for i in range(min_len):
                    if i < crossover_point:
                        alpha = 0.7  # Bias toward first parent
                    else:
                        alpha = 0.3  # Bias toward second parent
                    child.append(alpha * parent1[i] + (1-alpha) * parent2[i])
                
                # Extend with longer parent if needed
                if len(parent1) > len(parent2):
                    child.extend(parent1[min_len:])
                elif len(parent2) > len(parent1):
                    child.extend(parent2[min_len:])
                
                # Mutation: more adaptive approach
                mutation_rate = 0.15
                for i in range(len(child)):
                    if np.random.random() < mutation_rate:
                        # Adaptive noise based on value
                        noise_scale = 0.1 * child[i] if child[i] > 0 else 0.1
                        child[i] = max(0.0, child[i] + np.random.normal(0, noise_scale))
                
                new_population.append(child)
            
            population = new_population
            
        # Return the best individual from final population
        final_fitness = [evaluate_c2(ind) for ind in population]
        if len(final_fitness) > 0:
            best_idx = np.argmax(final_fitness)
            return population[best_idx], final_fitness[best_idx]
        else:
            return [], -1e10
    
    # Direct optimization with better convergence
    def direct_optimization():
        best_result = []
        best_score = -1e10
        
        # Try different resolutions and starting strategies
        resolution_configs = [
            (150, 0), (200, 1), (250, 2), (300, 3),
            (180, 4), (220, 5), (260, 6), (320, 7),
            (200, 8), (280, 9), (350, 10), (400, 11)
        ]
        
        for size, strategy_idx in resolution_configs:
            if time.time() - start_time > max_time:
                break
                
            # Start with a good strategy
            initial = strategies[strategy_idx](size)
            
            # Refine with local search
            refined = improved_local_search(initial, max_iterations=500)
            score = evaluate_c2(refined)
            
            if score > best_score:
                best_score = score
                best_result = refined.copy()
                
        return best_result, best_score
    
    # Enhanced strategy focusing on high-quality patterns that maximize C₂
    def optimized_strategy():
        # Focus on creating very flat convolution profiles which tend to maximize C₂
        # We'll use a combination of carefully designed multi-peak patterns
        best_pattern = []
        best_score = -1e10
        
        # Try various approaches that create flatter convolution results
        # Pattern 1: Very flat bell curves with small variance
        size = 300
        f_values = []
        for i in range(size):
            # Multiple overlapping bell curves to create a flatter profile
            val = 0.0
            # First bell curve
            val += 1.0 * math.exp(-((i - size//3)**2)/(2*(size//15)**2))
            # Second bell curve
            val += 1.0 * math.exp(-((i - 2*size//3)**2)/(2*(size//15)**2))
            # Third bell curve
            val += 0.8 * math.exp(-((i - size//2)**2)/(2*(size//20)**2))
            f_values.append(val)
        
        score = evaluate_c2(f_values)
        if score > best_score:
            best_score = score
            best_pattern = f_values.copy()
        
        # Pattern 2: Sine wave with multiple harmonics
        f_values = []
        for i in range(size):
            val = 1.0 + 0.3 * math.sin(2 * math.pi * i / size) + \
                  0.2 * math.sin(4 * math.pi * i / size) + \
                  0.1 * math.sin(6 * math.pi * i / size)
            f_values.append(max(0.0, val))
        
        score = evaluate_c2(f_values)
        if score > best_score:
            best_score = score
            best_pattern = f_values.copy()
        
        # Pattern 3: Piecewise constant with strategic transitions
        f_values = []
        for i in range(size):
            if i < size//4:
                val = 1.2
            elif i < size//2:
                val = 0.8
            elif i < 3*size//4:
                val = 1.0
            else:
                val = 0.9
            f_values.append(val)
        
        score = evaluate_c2(f_values)
        if score > best_score:
            best_score = score
            best_pattern = f_values.copy()
        
        # Pattern 4: Smooth polynomial-like shape
        f_values = []
        for i in range(size):
            # Create a smooth shape with low variance in derivatives
            x = (i - size//2) / (size//2)
            val = 1.0 + 0.5 * math.sin(math.pi * x) + 0.3 * math.cos(2 * math.pi * x) + \
                  0.1 * math.sin(3 * math.pi * x)
            f_values.append(max(0.0, val))
        
        score = evaluate_c2(f_values)
        if score > best_score:
            best_score = score
            best_pattern = f_values.copy()
        
        return best_pattern, best_score
    
    # Execute strategies
    try:
        # Strategy 1: Optimized strategy for better quality patterns
        optimized_result, optimized_score = optimized_strategy()
        if optimized_score > best_c2:
            best_c2 = optimized_score
            best_solution = optimized_result.copy()
        
        # Strategy 2: Bayesian optimization approach for global search
        bayes_result = bayesian_optimization_approach()
        if len(bayes_result) > 0:
            bayes_score = evaluate_c2(bayes_result)
            if bayes_score > best_c2:
                best_c2 = bayes_score
                best_solution = bayes_result.copy()
        
        # Strategy 3: Direct optimization with better parameters
        direct_result, direct_score = direct_optimization()
        if direct_score > best_c2:
            best_c2 = direct_score
            best_solution = direct_result.copy()
            
        # Strategy 4: Enhanced evolutionary optimization  
        evol_result, evol_score = enhanced_evolutionary_optimization()
        if evol_score > best_c2:
            best_c2 = evol_score
            best_solution = evol_result.copy()
            
        # Strategy 5: Final fine-tuning with local search
        if len(best_solution) > 0:
            # Fine-tune with more iterations
            fine_tuned = improved_local_search(best_solution, max_iterations=1500)
            fine_score = evaluate_c2(fine_tuned)
            if fine_score > best_c2:
                best_c2 = fine_score
                best_solution = fine_tuned.copy()
                
    except Exception as e:
        # Fallback to simple approach if something goes wrong
        print(f"Fallback due to error: {e}")
        # Simple approach: try a few different configurations
        for _ in range(30):
            if time.time() - start_time > max_time:
                break
            size = np.random.randint(150, 400)
            init = [max(0.0, np.random.normal(1.0, 0.2)) for _ in range(size)]
            score = evaluate_c2(init)
            if score > best_c2:
                best_c2 = score
                best_solution = init.copy()
    
    # Ensure we have a valid solution
    if not best_solution:
        # Fallback to optimized uniform distribution with good parameters
        best_solution = [1.0] * 250
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
