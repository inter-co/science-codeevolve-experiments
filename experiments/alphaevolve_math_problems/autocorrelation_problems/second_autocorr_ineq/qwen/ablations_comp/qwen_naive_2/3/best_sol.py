# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import differential_evolution, minimize
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import multiprocessing as mp

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
    
    # Compute ||g||₂² using trapezoidal rule for the continuous approximation
    g_norm_2_sq = 0.0
    
    # Use more accurate piecewise quadratic integration for ||g||₂²
    # This integrates g² using quadratic interpolation between points
    for i in range(len(g)-1):
        h = step_size
        y1 = g[i]
        y2 = g[i+1]
        # Trapezoidal rule for ∫g² dx: (h/3)(y1² + y1*y2 + y2²)
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
    
    # Enhanced optimization strategies with better patterns
    strategies = [
        # Uniform distribution
        lambda size: [1.0] * size,
        # Random with constraint enforcement
        lambda size: [max(0.0, np.random.normal(1.0, 0.3)) for _ in range(size)],
        # Step pattern - alternating high-low with more variation
        lambda size: [2.0 if i % 3 == 0 else (1.0 if i % 3 == 1 else 0.5) for i in range(size)],
        # Gaussian-like peak with better scaling
        lambda size: [np.exp(-((i - size//2)**2)/(2*(size//6)**2)) * 1.5 for i in range(size)],
        # Sinusoidal pattern with more complex shape
        lambda size: [1.0 + 0.5 * np.sin(2 * np.pi * i / size) + 0.2 * np.cos(4 * np.pi * i / size) for i in range(size)],
        # Double peak pattern
        lambda size: [0.5 + 0.5 * np.exp(-((i - size//3)**2)/(2*(size//12)**2)) + 
                     0.5 * np.exp(-((i - 2*size//3)**2)/(2*(size//12)**2)) for i in range(size)],
        # Exponential decay pattern
        lambda size: [np.exp(-i/(size/3)) * 1.5 for i in range(size)],
        # Optimized multi-peak pattern
        lambda size: [1.0 + 0.5 * np.sin(4 * np.pi * i / size) + 0.3 * np.cos(8 * np.pi * i / size) + 
                     0.2 * np.sin(12 * np.pi * i / size) for i in range(size)],
        # Asymmetric pattern with higher peaks
        lambda size: [max(0.0, 1.5 + 0.8 * np.sin(3 * np.pi * i / size) + 0.3 * np.cos(6 * np.pi * i / size)) for i in range(size)],
        # Multi-hump pattern
        lambda size: [1.0 + 0.3 * np.sin(2 * np.pi * i / size) + 0.2 * np.sin(4 * np.pi * i / size) + 
                     0.15 * np.sin(6 * np.pi * i / size) + 0.1 * np.sin(8 * np.pi * i / size) for i in range(size)]
    ]
    
    # Improved optimization with better convergence
    def improved_local_search(initial_solution: List[float], max_iterations: int = 1000) -> List[float]:
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
            
            # Use adaptive step size based on current solution
            step_size = 0.05 + 0.05 * np.random.random()
            delta = np.random.normal(0, step_size)
            neighbor[idx] = max(0.0, neighbor[idx] + delta)
            
            # Try to improve by changing multiple elements
            if np.random.random() < 0.3:
                # Change another element
                idx2 = np.random.randint(len(neighbor))
                delta2 = np.random.normal(0, step_size * 0.5)
                neighbor[idx2] = max(0.0, neighbor[idx2] + delta2)
            
            # Accept or reject based on improvement
            neighbor_c2 = evaluate_c2(neighbor)
            if neighbor_c2 > current_c2:
                current_solution = neighbor
                current_c2 = neighbor_c2
                if neighbor_c2 > best_local_c2:
                    best_local_c2 = neighbor_c2
                    best_local_solution = neighbor.copy()
            elif np.random.random() < 0.05:  # Sometimes accept worse solutions
                current_solution = neighbor
                current_c2 = neighbor_c2
                
        return best_local_solution
    
    # Advanced genetic algorithm approach for global optimization
    def genetic_algorithm_optimization():
        # Define the optimization problem using DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Define the gene range and individual creation
        def create_individual():
            size = np.random.randint(200, 400)
            return [max(0.0, np.random.normal(1.0, 0.5)) for _ in range(size)]
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual):
            return evaluate_c2(individual)
        
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create population and run GA
        try:
            population = toolbox.population(n=100)
            hof = tools.HallOfFame(1)
            
            # Run the evolutionary algorithm
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            population, logbook = algorithms.eaSimple(
                population, toolbox, cxpb=0.7, mutpb=0.2, 
                ngen=50, stats=stats, halloffame=hof, verbose=False
            )
            
            if len(hof) > 0:
                return hof[0], evaluate(hof[0])
        except:
            pass
            
        return [], -1e10
    
    # Advanced Bayesian optimization approach with surrogate modeling
    def bayesian_optimization_advanced():
        # Create a smarter acquisition function and use a more sophisticated GP model
        def objective(trial):
            # Sample parameters for the function
            size = trial.suggest_int('size', 150, 400)
            pattern_type = trial.suggest_categorical('pattern', ['gaussian', 'sinusoidal', 'multi_peak', 'custom'])
            
            if pattern_type == 'gaussian':
                # Create Gaussian pattern with more control
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1  # Map to [-1, 1]
                    amplitude = trial.suggest_float('amplitude', 0.5, 2.0)
                    center = trial.suggest_float('center', -0.8, 0.8)
                    width = trial.suggest_float('width', 0.05, 0.3)
                    val = amplitude * np.exp(-((x - center)**2)/(2*width**2))
                    pattern.append(val)
            elif pattern_type == 'sinusoidal':
                # Create sinusoidal pattern with more complex harmonics
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1
                    amp1 = trial.suggest_float('amp1', 0.1, 1.0)
                    freq1 = trial.suggest_float('freq1', 1, 5)
                    amp2 = trial.suggest_float('amp2', 0.05, 0.5)
                    freq2 = trial.suggest_float('freq2', 1, 5)
                    amp3 = trial.suggest_float('amp3', 0.02, 0.2)
                    freq3 = trial.suggest_float('freq3', 1, 8)
                    val = 1.0 + amp1 * np.sin(2 * np.pi * x * freq1) + \
                          amp2 * np.cos(4 * np.pi * x * freq2) + \
                          amp3 * np.sin(6 * np.pi * x * freq3)
                    pattern.append(val)
            elif pattern_type == 'multi_peak':
                # Create multi-peak pattern with better control
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1
                    val = 1.0 + 0.5 * np.exp(-((x - 0.3)**2)/0.1) + 0.3 * np.exp(-((x + 0.3)**2)/0.1) + \
                          0.2 * np.exp(-((x - 0.7)**2)/0.05) + 0.1 * np.exp(-((x + 0.7)**2)/0.05)
                    pattern.append(val)
            else:  # custom
                # Create a more complex custom pattern with better tuning
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1
                    val = 1.0 + 0.3 * np.sin(2 * np.pi * x) + 0.2 * np.sin(4 * np.pi * x) + \
                          0.15 * np.sin(6 * np.pi * x) + 0.1 * np.sin(8 * np.pi * x)
                    pattern.append(val)
            
            # Ensure non-negative values
            pattern = [max(0.0, p) for p in pattern]
            
            return evaluate_c2(pattern)
        
        # Run optimization with more trials and better timeout handling
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, timeout=max_time)
        
        if len(study.trials) > 0:
            # Extract the best configuration
            best_trial = study.best_trial
            size = best_trial.params['size']
            pattern_type = best_trial.params['pattern']
            
            # Recreate the best pattern with better parameters
            if pattern_type == 'gaussian':
                pattern = [np.exp(-((i - size//2)**2)/(2*(size//10)**2)) * best_trial.params.get('amplitude', 1.0) 
                          for i in range(size)]
            elif pattern_type == 'sinusoidal':
                pattern = [1.0 + best_trial.params.get('amp1', 0.5) * np.sin(2 * np.pi * i / size * best_trial.params.get('freq1', 2)) + 
                          best_trial.params.get('amp2', 0.2) * np.cos(4 * np.pi * i / size * best_trial.params.get('freq2', 2)) + 
                          best_trial.params.get('amp3', 0.1) * np.sin(6 * np.pi * i / size * best_trial.params.get('freq3', 3))
                          for i in range(size)]
            elif pattern_type == 'multi_peak':
                pattern = []
                for i in range(size):
                    x = i / (size - 1) * 2 - 1
                    val = 1.0 + 0.5 * np.exp(-((x - 0.3)**2)/0.1) + 0.3 * np.exp(-((x + 0.3)**2)/0.1) + \
                          0.2 * np.exp(-((x - 0.7)**2)/0.05) + 0.1 * np.exp(-((x + 0.7)**2)/0.05)
                    pattern.append(val)
            else:
                pattern = [1.0 + 0.3 * np.sin(2 * np.pi * i / size) + 0.2 * np.sin(4 * np.pi * i / size) + \
                          0.15 * np.sin(6 * np.pi * i / size) + 0.1 * np.sin(8 * np.pi * i / size) for i in range(size)]
            
            pattern = [max(0.0, p) for p in pattern]
            return pattern, evaluate_c2(pattern)
        else:
            return [], -1e10
    
    # Advanced evolutionary algorithm with better diversity management and selection pressure
    def advanced_evolutionary_optimization():
        # Parameters for evolution
        population_size = 100
        generations = 200
        mutation_rate = 0.15
        crossover_rate = 0.85
        
        # Initialize diverse population with better starting patterns
        population = []
        for i in range(population_size):
            if time.time() - start_time > max_time:
                break
                
            # Mix of strategies with better parameter diversity
            strategy_idx = i % len(strategies)
            size = 150 + (i % 10) * 20  # Different sizes for diversity
            individual = strategies[strategy_idx](size)
            
            # Add some random noise to increase diversity
            individual = [max(0.0, val + np.random.normal(0, 0.1 * val if val > 0 else 0.1)) for val in individual]
            
            # Add some structured randomness to encourage exploration
            if i < population_size // 3:
                # Add more exploration for first third
                individual = [max(0.0, val * (1 + np.random.normal(0, 0.2))) for val in individual]
            elif i < 2 * population_size // 3:
                # Mid-range diversity
                individual = [max(0.0, val * (1 + np.random.normal(0, 0.1))) for val in individual]
            else:
                # More exploitation for last third
                individual = [max(0.0, val * (1 + np.random.normal(0, 0.05))) for val in individual]
                
            population.append(individual)
        
        # Evolution loop with adaptive parameters
        for generation in range(generations):
            if time.time() - start_time > max_time:
                break
                
            # Evaluate fitness
            fitness_scores = [evaluate_c2(ind) for ind in population]
            
            # Sort by fitness
            sorted_indices = np.argsort(fitness_scores)[::-1]
            top_individuals = [population[i] for i in sorted_indices[:population_size//2]]
            top_fitness = [fitness_scores[i] for i in sorted_indices[:population_size//2]]
            
            # Keep best individuals (elitism)
            new_population = top_individuals.copy()
            
            # Generate offspring with better selection pressure
            while len(new_population) < population_size:
                if time.time() - start_time > max_time:
                    break
                    
                # Tournament selection with larger tournaments for better pressure
                tournament_size = 7
                tournament_indices = np.random.choice(len(top_individuals), tournament_size, replace=False)
                tournament_fitness = [top_fitness[i] for i in tournament_indices]
                parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
                
                # Another parent
                tournament_indices2 = np.random.choice(len(top_individuals), tournament_size, replace=False)
                tournament_fitness2 = [top_fitness[i] for i in tournament_indices2]
                parent2_idx = tournament_indices2[np.argmax(tournament_fitness2)]
                
                parent1 = top_individuals[parent1_idx]
                parent2 = top_individuals[parent2_idx]
                
                # Crossover with probability control
                if np.random.random() < crossover_rate:
                    # Uniform crossover with better mixing
                    child = []
                    min_len = min(len(parent1), len(parent2))
                    for i in range(min_len):
                        if np.random.random() < 0.5:
                            child.append(parent1[i])
                        else:
                            child.append(parent2[i])
                    
                    # Extend with longer parent if needed
                    if len(parent1) > len(parent2):
                        child.extend(parent1[min_len:])
                    elif len(parent2) > len(parent1):
                        child.extend(parent2[min_len:])
                else:
                    # Clone one parent with probability
                    child = parent1.copy() if np.random.random() < 0.7 else parent2.copy()
                
                # Mutation with adaptive rates
                for i in range(len(child)):
                    if np.random.random() < mutation_rate:
                        # Adaptive mutation based on current value and generation
                        mutation_strength = 0.1 * child[i] if child[i] > 0 else 0.1
                        # Reduce mutation strength as generation increases
                        generation_factor = 1.0 - (generation / generations) * 0.5
                        mutation_strength *= generation_factor
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
    
    # Specialized pattern optimization with mathematical insights and better refinement
    def mathematical_pattern_optimization():
        best_result = []
        best_score = -1e10
        
        # Try patterns that have shown good performance mathematically
        patterns_to_try = []
        
        # Pattern 1: Smooth bell-shaped pattern with multiple peaks (improved)
        for size in [200, 250, 300, 350]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1  # Map to [-1, 1]
                # Create a pattern with multiple smooth peaks and better amplitude control
                val = 1.0 + 0.6 * np.exp(-((x - 0.3)**2)/0.08) + 0.4 * np.exp(-((x + 0.3)**2)/0.08) + \
                      0.2 * np.exp(-((x - 0.7)**2)/0.04) + 0.1 * np.exp(-((x + 0.7)**2)/0.04) + \
                      0.05 * np.exp(-((x - 0.9)**2)/0.02) + 0.03 * np.exp(-((x + 0.9)**2)/0.02)
                pattern.append(val)
            patterns_to_try.append(('multi_bell_enhanced', pattern))
        
        # Pattern 2: Symmetric pattern with gradual transitions (improved)
        for size in [200, 250, 300, 350]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Smooth transition pattern with better shaping
                val = 1.0 + 0.5 * np.tanh(2 * x) + 0.3 * np.tanh(4 * x) + 0.2 * np.tanh(6 * x) + \
                      0.1 * np.tanh(8 * x)
                pattern.append(val)
            patterns_to_try.append(('smooth_transition_enhanced', pattern))
            
        # Pattern 3: Optimized sinc-like pattern (improved)
        for size in [200, 250, 300, 350]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Sinc-like pattern with better normalization and multiple terms
                if abs(x) < 1e-8:
                    val = 1.0
                else:
                    val = 1.0 + 0.5 * np.sin(3 * np.pi * x) / (3 * np.pi * x) + \
                          0.3 * np.sin(6 * np.pi * x) / (6 * np.pi * x) + \
                          0.2 * np.sin(9 * np.pi * x) / (9 * np.pi * x) + \
                          0.1 * np.sin(12 * np.pi * x) / (12 * np.pi * x)
                pattern.append(max(0.0, val))
            patterns_to_try.append(('sinc_like_enhanced', pattern))
        
        # Pattern 4: Optimized exponential pattern
        for size in [200, 250, 300, 350]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Better exponential pattern with multiple components
                val = 1.0 + 0.7 * np.exp(-((x - 0.4)**2)/0.1) + 0.5 * np.exp(-((x + 0.4)**2)/0.1) + \
                      0.3 * np.exp(-((x - 0.8)**2)/0.05) + 0.2 * np.exp(-((x + 0.8)**2)/0.05)
                pattern.append(val)
            patterns_to_try.append(('exponential_enhanced', pattern))
        
        # Pattern 5: New optimized pattern - spike-based with controlled spread
        for size in [250, 300, 350]:
            if time.time() - start_time > max_time:
                break
            pattern = []
            # Create a pattern with spikes at key positions to encourage flat autoconvolution
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Spike pattern with better distribution
                val = 1.0 + 0.8 * np.exp(-((x - 0.3)**2)/0.05) + 0.6 * np.exp(-((x + 0.3)**2)/0.05) + \
                      0.4 * np.exp(-((x - 0.7)**2)/0.03) + 0.3 * np.exp(-((x + 0.7)**2)/0.03) + \
                      0.2 * np.exp(-((x - 0.9)**2)/0.02) + 0.1 * np.exp(-((x + 0.9)**2)/0.02)
                pattern.append(val)
            patterns_to_try.append(('spike_pattern', pattern))
        
        # Evaluate all patterns
        for name, pattern in patterns_to_try:
            if time.time() - start_time > max_time:
                break
            # Fine-tune with local search
            refined = improved_local_search(pattern, max_iterations=500)
            score = evaluate_c2(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        
        return best_result, best_score
    
    # Advanced gradient-free optimization with adaptive strategies
    def adaptive_gradient_free_optimization():
        """Use a combination of Nelder-Mead and simulated annealing for robust optimization"""
        # Start with a good pattern from mathematical optimization
        _, best_math_pattern = mathematical_pattern_optimization()
        if len(best_math_pattern) > 0:
            initial_solution = best_math_pattern.copy()
        else:
            # Fallback to a reasonable starting point
            initial_solution = [1.0] * 300
        
        # Try different optimization approaches
        best_final = initial_solution.copy()
        best_final_score = evaluate_c2(best_final)
        
        # Try Nelder-Mead optimization on a subset
        try:
            # Select a smaller subset for local optimization
            subset_size = min(100, len(initial_solution))
            subset_indices = np.sort(np.random.choice(len(initial_solution), subset_size, replace=False))
            subset_initial = [initial_solution[i] for i in subset_indices]
            
            def subset_objective(subset_vals):
                # Reconstruct full solution
                full_solution = initial_solution.copy()
                for i, idx in enumerate(subset_indices):
                    full_solution[idx] = subset_vals[i]
                return -evaluate_c2(full_solution)  # Negative because minimize
            
            # Use differential evolution on subset
            bounds = [(0.0, 3.0) for _ in range(subset_size)]
            result = differential_evolution(subset_objective, bounds, seed=42, maxiter=50, popsize=15)
            
            if result.success:
                # Reconstruct full solution
                full_solution = initial_solution.copy()
                for i, idx in enumerate(subset_indices):
                    full_solution[idx] = result.x[i]
                score = evaluate_c2(full_solution)
                if score > best_final_score:
                    best_final_score = score
                    best_final = full_solution.copy()
        except:
            pass
        
        return best_final, best_final_score
    
    # Enhanced pattern search focusing on promising mathematical structures
    def enhanced_pattern_search():
        """Try specific mathematical constructions known to work well"""
        best_result = []
        best_score = -1e10
        
        # Try various mathematical constructions
        patterns_to_try = []
        
        # 1. Optimized Gaussian mixture
        for size in [300, 350, 400]:
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # More carefully tuned Gaussian mixture
                val = 1.0 + 0.8 * np.exp(-((x - 0.4)**2)/0.08) + 0.6 * np.exp(-((x + 0.4)**2)/0.08) + \
                      0.4 * np.exp(-((x - 0.7)**2)/0.05) + 0.3 * np.exp(-((x + 0.7)**2)/0.05) + \
                      0.2 * np.exp(-((x - 0.9)**2)/0.03) + 0.1 * np.exp(-((x + 0.9)**2)/0.03)
                pattern.append(val)
            patterns_to_try.append(('gaussian_mixture', pattern))
        
        # 2. Fourier series pattern
        for size in [300, 350, 400]:
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Construct pattern from trigonometric series
                val = 1.0 + 0.5 * np.sin(2 * np.pi * x) + 0.3 * np.sin(4 * np.pi * x) + \
                      0.2 * np.sin(6 * np.pi * x) + 0.1 * np.sin(8 * np.pi * x) + \
                      0.05 * np.cos(2 * np.pi * x) + 0.03 * np.cos(4 * np.pi * x)
                pattern.append(max(0.0, val))
            patterns_to_try.append(('fourier_series', pattern))
        
        # 3. Piecewise linear with sharp transitions
        for size in [300, 350, 400]:
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Sharp transition pattern
                if x < -0.5:
                    val = 1.0 + 0.5 * (x + 0.5) + 0.3 * (x + 0.5)**2
                elif x < 0.5:
                    val = 1.0 + 0.2 * np.sin(4 * np.pi * x)
                else:
                    val = 1.0 + 0.5 * (0.5 - x) + 0.3 * (0.5 - x)**2
                pattern.append(max(0.0, val))
            patterns_to_try.append(('piecewise_linear', pattern))
        
        # 4. Logarithmic-like pattern with peaks
        for size in [300, 350, 400]:
            pattern = []
            for i in range(size):
                x = i / (size - 1) * 2 - 1
                # Pattern with logarithmic-like peaks
                val = 1.0 + 0.6 * np.exp(-((x - 0.3)**2)/0.06) + 0.4 * np.exp(-((x + 0.3)**2)/0.06) + \
                      0.2 * np.exp(-((x - 0.7)**2)/0.04) + 0.1 * np.exp(-((x + 0.7)**2)/0.04)
                pattern.append(val)
            patterns_to_try.append(('log_peaks', pattern))
        
        # Evaluate all patterns
        for name, pattern in patterns_to_try:
            if time.time() - start_time > max_time:
                break
            # Fine-tune with local search
            refined = improved_local_search(pattern, max_iterations=1000)
            score = evaluate_c2(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        
        return best_result, best_score
    
    # Execute strategies in order of likely effectiveness
    try:
        # Strategy 1: Enhanced pattern search with mathematical constructions
        enhanced_result, enhanced_score = enhanced_pattern_search()
        if enhanced_score > best_c2:
            best_c2 = enhanced_score
            best_solution = enhanced_result.copy()
            
        # Strategy 2: Genetic algorithm optimization
        ga_result, ga_score = genetic_algorithm_optimization()
        if ga_score > best_c2:
            best_c2 = ga_score
            best_solution = ga_result.copy()
            
        # Strategy 3: Bayesian optimization
        bayes_result, bayes_score = bayesian_optimization_advanced()
        if bayes_score > best_c2:
            best_c2 = bayes_score
            best_solution = bayes_result.copy()
            
        # Strategy 4: Advanced evolutionary optimization  
        evol_result, evol_score = advanced_evolutionary_optimization()
        if evol_score > best_c2:
            best_c2 = evol_score
            best_solution = evol_result.copy()
            
        # Strategy 5: Mathematical pattern optimization
        math_result, math_score = mathematical_pattern_optimization()
        if math_score > best_c2:
            best_c2 = math_score
            best_solution = math_result.copy()
            
        # Strategy 6: Adaptive gradient-free optimization
        adaptive_result, adaptive_score = adaptive_gradient_free_optimization()
        if adaptive_score > best_c2:
            best_c2 = adaptive_score
            best_solution = adaptive_result.copy()
            
        # Strategy 7: Local refinement of best solution found so far
        if len(best_solution) > 0:
            # Fine-tune the best solution found
            fine_tuned = improved_local_search(best_solution, max_iterations=3000)
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
            init = [max(0.0, np.random.normal(1.0, 0.3)) for _ in range(size)]
            score = evaluate_c2(init)
            if score > best_c2:
                best_c2 = score
                best_solution = init.copy()
    
    # Ensure we have a valid solution
    if not best_solution:
        # Fallback to optimized uniform distribution with better parameters
        best_solution = [1.0] * 300
    
    return best_solution

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    f_values = construct_function()
    print(f"Function: {f_values}")
