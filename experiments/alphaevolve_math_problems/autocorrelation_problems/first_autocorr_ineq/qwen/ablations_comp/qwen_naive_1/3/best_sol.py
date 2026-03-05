# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import fftconvolve
import time
from numba import jit
import warnings
from deap import base, creator, tools, algorithms
import multiprocessing as mp
import os
from scipy.optimize import differential_evolution, minimize
import copy

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True)
def fast_convolution_jit(a):
    """Fast convolution implementation using Numba JIT compilation."""
    n = len(a)
    result = np.zeros(2*n - 1)
    
    # Manual convolution for speed
    for i in range(n):
        for j in range(n):
            result[i + j] += a[i] * a[j]
    
    return result

def compute_c1(sequence):
    """Compute C1 for a given sequence using FFT-based convolution."""
    if len(sequence) == 0:
        return float('inf')
    
    # Convert to numpy array for efficient computation
    a = np.array(sequence, dtype=np.float64)
    
    # Compute convolution (auto-correlation) efficiently using FFT
    conv = fftconvolve(a, a, mode='full')
    
    # Maximum value in convolution (excluding the zeroth element which is sum of squares)
    max_conv = np.max(conv[1:]) if len(conv) > 1 else 0
    
    # Sum of sequence squared
    sum_sq = np.sum(a)**2
    
    # Avoid division by zero
    if sum_sq < 1e-12:
        return float('inf')
    
    # C1 = 2n * max_conv / sum_sq
    n = len(sequence)
    c1 = 2 * n * max_conv / sum_sq
    
    return c1

def inv_c1_objective(sequence):
    """Objective function to maximize 1/C1 (minimize C1)."""
    c1 = compute_c1(sequence)
    if c1 == float('inf') or np.isnan(c1):
        return 0  # Return very small value if invalid
    return 1.0 / c1

def generate_bell_curve_sequence(n, peak_height=None):
    """Generate a bell-curve shaped sequence that tends to work well."""
    if peak_height is None:
        peak_height = random.uniform(800, 1000)
    
    sequence = []
    mid = n // 2
    # Use a sharper standard deviation to concentrate mass
    std_dev = n / 10.0
    
    for i in range(n):
        distance = abs(i - mid)
        # Sharper decay than typical gaussian
        decay = np.exp(-distance**2 / (2 * std_dev**2))
        height = max(0, peak_height * decay)
        sequence.append(height)
    
    return sequence

def generate_peak_sequence(n, peak_height=None, peak_position=None):
    """Generate a sequence with a single sharp peak."""
    if peak_height is None:
        peak_height = random.uniform(800, 1000)
    if peak_position is None:
        peak_position = n // 2
    
    sequence = []
    # Very sharp peak with exponential decay
    for i in range(n):
        distance = abs(i - peak_position)
        # Exponential decay for sharper peak
        decay = np.exp(-distance / (n/20.0)) if n > 0 else 0
        height = max(0, peak_height * decay)
        sequence.append(height)
    
    return sequence

def generate_multi_peak_sequence(n, num_peaks=2):
    """Generate a sequence with multiple peaks."""
    sequence = [0.0] * n
    
    # Place peaks with some spacing
    for i in range(num_peaks):
        pos = random.randint(n//10, 9*n//10) if n > 10 else n//2
        height = random.uniform(600, 1000)
        sequence[pos] = max(sequence[pos], height)
    
    # Apply smoothing to avoid sharp discontinuities
    smoothed = [0.0] * n
    for i in range(n):
        smoothed[i] = sequence[i]
        # Apply influence from nearby peaks
        for j in range(max(0, i-3), min(n, i+4)):
            if i != j and sequence[j] > 0:
                dist = abs(i - j)
                influence = sequence[j] * np.exp(-dist**2 / 10.0)
                smoothed[i] += influence * 0.5
    
    return smoothed

def generate_geometric_sequence(n, base=None):
    """Generate a geometric decay sequence."""
    if base is None:
        base = random.uniform(0.85, 0.95)
    start_height = random.uniform(600, 1000)
    
    sequence = []
    for i in range(n):
        height = max(0, start_height * (base ** i))
        sequence.append(height)
    return sequence

def generate_balanced_sequence(n):
    """Generate a balanced sequence with moderate variations."""
    # Create a sequence with a mix of high and low values
    sequence = []
    base_height = random.uniform(400, 800)
    
    for i in range(n):
        # Add some structured variation
        variation = 0
        if i % 10 < 5:  # Every other group
            variation = random.uniform(-200, 200)
        else:
            variation = random.uniform(-100, 100)
        height = max(0, base_height + variation)
        sequence.append(height)
    
    return sequence

def generate_optimized_peak_sequence(n):
    """Generate an optimized peak sequence focusing on high performance."""
    # Create a sequence that concentrates mass efficiently
    sequence = [0.0] * n
    
    # Place one dominant peak
    peak_pos = n // 2
    peak_height = random.uniform(900, 1000)
    sequence[peak_pos] = peak_height
    
    # Add smaller support peaks to create a more favorable convolution shape
    # Place additional peaks symmetrically
    for i in range(1, min(4, n//4)):
        left_pos = max(0, peak_pos - i * n//8)
        right_pos = min(n-1, peak_pos + i * n//8)
        
        # Smaller peaks
        left_height = peak_height * (0.7 ** i)
        right_height = peak_height * (0.7 ** i)
        
        sequence[left_pos] = max(sequence[left_pos], left_height)
        sequence[right_pos] = max(sequence[right_pos], right_height)
    
    # Apply slight smoothing to make it more numerically stable
    smoothed = [0.0] * n
    for i in range(n):
        smoothed[i] = sequence[i]
        for j in range(max(0, i-2), min(n, i+3)):
            if i != j:
                dist = abs(i - j)
                influence = sequence[j] * np.exp(-dist**2 / 8.0)
                smoothed[i] += influence * 0.1
    
    return smoothed

def generate_focused_sequence(n):
    """Generate a focused sequence that emphasizes concentration of mass."""
    # Start with a very sharp peak
    sequence = [0.0] * n
    peak_pos = n // 2
    peak_height = 1000.0
    
    # Create a very narrow peak with exponential decay
    for i in range(n):
        distance = abs(i - peak_pos)
        # Very sharp exponential decay
        decay = np.exp(-distance / (n/30.0))
        sequence[i] = max(0, peak_height * decay)
    
    # Normalize to have reasonable sum
    total_sum = sum(sequence)
    if total_sum > 0:
        sequence = [x * 1000 / total_sum for x in sequence]
    
    return sequence

def generate_explicitly_optimized_sequence(n):
    """Generate a sequence designed to be optimal by concentrating mass efficiently."""
    # Create a sequence that's optimized for convolution properties
    sequence = [0.0] * n
    
    # Place a strong central peak
    peak_pos = n // 2
    peak_height = 1000.0
    sequence[peak_pos] = peak_height
    
    # Add carefully placed supporting peaks to enhance the convolution
    # Place a few peaks around the center with decreasing amplitudes
    support_positions = [peak_pos - n//8, peak_pos + n//8, peak_pos - n//4, peak_pos + n//4]
    support_heights = [peak_height * 0.6, peak_height * 0.6, peak_height * 0.4, peak_height * 0.4]
    
    for pos, height in zip(support_positions, support_heights):
        if 0 <= pos < n:
            sequence[pos] = max(sequence[pos], height)
    
    # Apply slight smoothing to reduce numerical artifacts
    smoothed = [0.0] * n
    for i in range(n):
        smoothed[i] = sequence[i]
        for j in range(max(0, i-2), min(n, i+3)):
            if i != j:
                dist = abs(i - j)
                influence = sequence[j] * np.exp(-dist**2 / 4.0)
                smoothed[i] += influence * 0.1
    
    # Normalize to get good balance
    total_sum = sum(smoothed)
    if total_sum > 0:
        smoothed = [x * 1000 / total_sum for x in smoothed]
    
    return smoothed

def generate_symmetric_sequence(n):
    """Generate a symmetric sequence that works well for convolution."""
    # Create a symmetric pattern
    sequence = []
    mid = n // 2
    max_height = 1000.0
    
    # Create a symmetric pattern with exponential decay
    for i in range(n):
        distance_from_center = abs(i - mid)
        # Sharp decay for symmetry
        decay = np.exp(-distance_from_center / (n/20.0))
        height = max_height * decay
        sequence.append(height)
    
    # Normalize to reasonable values
    total_sum = sum(sequence)
    if total_sum > 0:
        sequence = [x * 1000 / total_sum for x in sequence]
    
    return sequence

def generate_power_sequence(n):
    """Generate a power-law decay sequence that often works well."""
    # Create a sequence with power-law decay
    sequence = []
    start_height = random.uniform(800, 1000)
    
    # Use power law with exponent slightly greater than 1 to concentrate mass
    exponent = random.uniform(1.1, 2.0)
    
    for i in range(n):
        # Ensure we don't divide by zero
        if i == 0:
            height = start_height
        else:
            # Power law decay
            height = max(0, start_height / (i ** exponent))
        sequence.append(height)
    
    # Normalize to reasonable values
    total_sum = sum(sequence)
    if total_sum > 0:
        sequence = [x * 1000 / total_sum for x in sequence]
    
    return sequence

def generate_concentrated_sequence(n):
    """Generate a highly concentrated sequence with a strong central peak."""
    # Create a very concentrated sequence
    sequence = [0.0] * n
    peak_pos = n // 2
    peak_height = 1000.0
    
    # Very sharp peak with narrow support
    for i in range(n):
        distance = abs(i - peak_pos)
        # Very sharp exponential decay
        decay = np.exp(-distance / (n/50.0)) if n > 0 else 0
        sequence[i] = max(0, peak_height * decay)
    
    # Normalize
    total_sum = sum(sequence)
    if total_sum > 0:
        sequence = [x * 1000 / total_sum for x in sequence]
    
    return sequence

def generate_sine_wave_sequence(n):
    """Generate a sine wave-like sequence that might perform well."""
    sequence = []
    amplitude = random.uniform(800, 1000)
    frequency = random.uniform(0.1, 0.5)
    
    for i in range(n):
        # Sine wave pattern
        value = amplitude * (0.5 + 0.5 * np.sin(2 * np.pi * frequency * i / n))
        sequence.append(value)
    
    # Normalize
    total_sum = sum(sequence)
    if total_sum > 0:
        sequence = [x * 1000 / total_sum for x in sequence]
    
    return sequence

def generate_comb_sequence(n):
    """Generate a comb-like sequence with spikes."""
    sequence = [0.0] * n
    
    # Place spikes at regular intervals
    num_spikes = random.randint(3, min(10, n//5))
    spike_height = random.uniform(800, 1000)
    
    for i in range(num_spikes):
        pos = random.randint(0, n-1)
        sequence[pos] = max(sequence[pos], spike_height)
    
    # Apply smoothing
    smoothed = [0.0] * n
    for i in range(n):
        smoothed[i] = sequence[i]
        for j in range(max(0, i-2), min(n, i+3)):
            if i != j and sequence[j] > 0:
                dist = abs(i - j)
                influence = sequence[j] * np.exp(-dist**2 / 4.0)
                smoothed[i] += influence * 0.2
    
    # Normalize
    total_sum = sum(smoothed)
    if total_sum > 0:
        smoothed = [x * 1000 / total_sum for x in smoothed]
    
    return smoothed

def optimize_with_local_search(initial_sequence, max_iter=2000, aggressive=False):
    """Use sophisticated local search optimization to improve an initial sequence."""
    current_seq = np.array(initial_sequence, dtype=float)
    current_inv_c1 = inv_c1_objective(current_seq)
    
    # Track best solution found
    best_seq = current_seq.copy()
    best_inv_c1 = current_inv_c1
    
    # Different types of perturbations for better exploration
    perturbation_types = ['small', 'medium', 'large', 'peak_shift', 'spread']
    
    for iteration in range(max_iter):
        # Make perturbations with varying strategies
        test_seq = current_seq.copy()
        
        # Determine how many changes to make
        num_changes = max(1, min(len(test_seq) // 10, 20))
        
        # Apply different types of perturbations
        for _ in range(num_changes):
            idx = random.randint(0, len(test_seq) - 1)
            pert_type = random.choice(perturbation_types)
            
            if pert_type == 'small':
                # Small perturbation
                change = random.uniform(-5, 5)
                test_seq[idx] = max(0, test_seq[idx] + change)
            elif pert_type == 'medium':
                # Medium perturbation
                change = random.uniform(-20, 20)
                test_seq[idx] = max(0, test_seq[idx] + change)
            elif pert_type == 'large':
                # Large perturbation
                change = random.uniform(-100, 100)
                test_seq[idx] = max(0, test_seq[idx] + change)
            elif pert_type == 'peak_shift':
                # Try shifting a peak toward center
                if test_seq[idx] > 0 and len(test_seq) > 10:
                    # Move towards center
                    center = len(test_seq) // 2
                    shift = random.randint(-5, 5)
                    new_idx = max(0, min(len(test_seq)-1, idx + shift))
                    test_seq[new_idx] = max(0, test_seq[new_idx] + test_seq[idx] * 0.3)
                    test_seq[idx] = max(0, test_seq[idx] * 0.7)
            elif pert_type == 'spread':
                # Spread mass around a peak
                if test_seq[idx] > 0:
                    # Distribute some mass to neighbors
                    spread_amount = test_seq[idx] * 0.1
                    test_seq[idx] = max(0, test_seq[idx] - spread_amount)
                    # Add to neighbors
                    neighbor_indices = [max(0, idx-1), min(len(test_seq)-1, idx+1)]
                    for neighbor in neighbor_indices:
                        test_seq[neighbor] = max(0, test_seq[neighbor] + spread_amount/2)
        
        # Accept if better
        test_inv_c1 = inv_c1_objective(test_seq)
        if test_inv_c1 > current_inv_c1:
            current_seq = test_seq
            current_inv_c1 = test_inv_c1
            
            # Update best solution
            if test_inv_c1 > best_inv_c1:
                best_seq = test_seq.copy()
                best_inv_c1 = test_inv_c1
    
    return best_seq.tolist()

def generate_initial_candidates(n_samples=50):
    """Generate diverse initial candidates for optimization."""
    candidates = []
    
    # Generate various types of sequences
    strategies = [
        lambda n: generate_bell_curve_sequence(n),
        lambda n: generate_peak_sequence(n),
        lambda n: generate_multi_peak_sequence(n),
        lambda n: generate_geometric_sequence(n),
        lambda n: generate_balanced_sequence(n),
        lambda n: generate_optimized_peak_sequence(n),
        lambda n: generate_focused_sequence(n),
        lambda n: generate_explicitly_optimized_sequence(n),
        lambda n: generate_symmetric_sequence(n),
        lambda n: generate_power_sequence(n),
        lambda n: generate_concentrated_sequence(n),
        lambda n: generate_sine_wave_sequence(n),
        lambda n: generate_comb_sequence(n)
    ]
    
    for i in range(n_samples):
        # Random length between 30 and 500
        n = random.randint(30, 500)
        
        # Choose generation strategy
        strategy = random.choice(strategies)
        
        try:
            candidate = strategy(n)
            candidates.append(candidate)
        except:
            # Fallback to random
            candidates.append([random.uniform(0, 1000) for _ in range(n)])
    
    return candidates

def optimize_with_scipy(sequence):
    """Use scipy optimization to refine a sequence."""
    # Convert to numpy array
    initial_array = np.array(sequence, dtype=float)
    
    # Define bounds (0 to 1000 for each element)
    bounds = [(0, 1000) for _ in range(len(initial_array))]
    
    # Objective function for scipy (we want to maximize 1/C1, which is equivalent to minimizing -1/C1)
    def objective(x):
        inv_c1 = inv_c1_objective(x)
        return -inv_c1 if inv_c1 > 0 else 1e10
    
    try:
        # Use differential evolution for global optimization
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=15,
            seed=42,
            disp=False
        )
        
        if result.success:
            return result.x.tolist()
    except:
        pass
    
    return sequence

def enhanced_evolutionary_search():
    """Enhanced evolutionary algorithm approach."""
    # Create a simple evolutionary algorithm for this problem
    def evaluate(individual):
        # Convert individual to sequence
        sequence = [max(0, min(1000, x)) for x in individual]  # Clip to valid range
        inv_c1 = inv_c1_objective(sequence)
        return (inv_c1,)
    
    # Create types for DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1000)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=100)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=50, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create initial population
    population = toolbox.population(n=50)
    
    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    
    # Evolution loop
    for generation in range(20):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace the old population with the new one
        population[:] = offspring
    
    # Return the best individual
    best_ind = tools.selBest(population, 1)[0]
    return [max(0, min(1000, x)) for x in best_ind]

def improved_adaptive_search():
    """Improved adaptive search with better optimization strategies."""
    start_time = time.time()
    max_time = 55  # Leave some buffer for final processing
    
    best_inv_c1 = 0
    best_sequence = None
    
    # Strategy 1: Generate diverse initial candidates with focus on promising patterns
    if time.time() - start_time < max_time:
        candidates = generate_initial_candidates(200)  # More candidates
        
        for i, candidate in enumerate(candidates):
            if time.time() - start_time > max_time:
                break
                
            # Local optimization with more iterations for promising candidates
            optimized_candidate = optimize_with_local_search(candidate, max_iter=1000)
            inv_c1 = inv_c1_objective(optimized_candidate)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = optimized_candidate.copy()
    
    # Strategy 2: Try scipy optimization on the best sequence
    if time.time() - start_time < max_time and best_sequence is not None:
        try:
            scipy_optimized = optimize_with_scipy(best_sequence)
            inv_c1 = inv_c1_objective(scipy_optimized)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = scipy_optimized.copy()
        except:
            pass
    
    # Strategy 3: Use evolutionary algorithm on top of best found
    if time.time() - start_time < max_time and best_sequence is not None:
        try:
            evol_result = enhanced_evolutionary_search()
            inv_c1 = inv_c1_objective(evol_result)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = evol_result.copy()
        except:
            pass
    
    # Strategy 4: Systematic refinement with better heuristics
    if time.time() - start_time < max_time and best_sequence is not None:
        # Try to improve the best found sequence with more targeted refinements
        refined_seq = best_sequence.copy()
        
        # Multiple refinement passes
        for pass_num in range(5):  # More passes
            if time.time() - start_time > max_time:
                break
                
            # More aggressive optimization for this pass
            for _ in range(2000):  # More iterations
                if time.time() - start_time > max_time:
                    break
                    
                test_seq = refined_seq.copy()
                
                # Apply more intelligent perturbations
                num_modifications = random.randint(5, 30)  # More modifications
                for _ in range(num_modifications):
                    idx = random.randint(0, len(test_seq)-1)
                    
                    # Adapt perturbation based on value
                    current_val = test_seq[idx]
                    if current_val > 0:
                        # Scale perturbation based on current value and position
                        scale = min(0.5, current_val / 1000.0)
                        # Make it more aggressive for better results
                        change = random.uniform(-scale * current_val * 2.0, scale * current_val * 2.0)
                        test_seq[idx] = max(0, test_seq[idx] + change)
                    else:
                        # For zero values, add some value
                        test_seq[idx] = max(0, test_seq[idx] + random.uniform(0, 500))
                
                test_inv_c1 = inv_c1_objective(test_seq)
                if test_inv_c1 > inv_c1_objective(refined_seq):
                    refined_seq = test_seq
        
        inv_c1 = inv_c1_objective(refined_seq)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined_seq.copy()
    
    # Strategy 5: Hybrid approach with better convergence
    if time.time() - start_time < max_time and best_sequence is not None:
        # Try to systematically improve by creating new candidates around best
        for _ in range(100):  # More attempts
            if time.time() - start_time > max_time:
                break
                
            # Create a candidate near the current best
            candidate = best_sequence.copy()
            
            # Apply random modifications with more structure
            for i in range(len(candidate)):
                if random.random() < 0.3:  # Higher probability to modify
                    # Perturb based on current value
                    if candidate[i] > 0:
                        change = random.uniform(-candidate[i]*0.4, candidate[i]*0.4)  # Even larger changes
                        candidate[i] = max(0, candidate[i] + change)
                    else:
                        # For zero values, add some value
                        candidate[i] = max(0, candidate[i] + random.uniform(0, 400))
            
            # Optimize this candidate
            optimized_candidate = optimize_with_local_search(candidate, max_iter=2000)
            inv_c1 = inv_c1_objective(optimized_candidate)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = optimized_candidate.copy()
    
    # Strategy 6: Direct optimization of best known candidates with aggressive local search
    if time.time() - start_time < max_time and best_sequence is not None:
        # Try even more aggressive optimization on the best sequence
        try:
            # Create a highly optimized version
            optimized_seq = best_sequence.copy()
            
            # Perform intensive optimization with more iterations
            for _ in range(5000):
                if time.time() - start_time > max_time:
                    break
                    
                test_seq = optimized_seq.copy()
                
                # Apply aggressive modifications
                num_modifications = random.randint(10, 40)
                for _ in range(num_modifications):
                    idx = random.randint(0, len(test_seq)-1)
                    # Very aggressive changes for optimization
                    change = random.uniform(-test_seq[idx]*0.7, test_seq[idx]*0.7)
                    test_seq[idx] = max(0, test_seq[idx] + change)
                
                test_inv_c1 = inv_c1_objective(test_seq)
                if test_inv_c1 > inv_c1_objective(optimized_seq):
                    optimized_seq = test_seq
            
            inv_c1 = inv_c1_objective(optimized_seq)
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = optimized_seq.copy()
        except:
            pass
    
    # Strategy 7: Focus on specific high-performing patterns
    if time.time() - start_time < max_time and best_sequence is None:
        # Try generating specific high-performance sequences
        for i in range(20):
            if time.time() - start_time > max_time:
                break
                
            # Try focused concentrated sequences
            n = random.randint(50, 300)
            seq = generate_concentrated_sequence(n)
            inv_c1 = inv_c1_objective(seq)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = seq.copy()
    
    # Strategy 8: Enhanced focused search on promising areas
    if time.time() - start_time < max_time and best_sequence is not None:
        # Try to create even better versions by exploring neighborhoods
        for _ in range(50):
            if time.time() - start_time > max_time:
                break
                
            # Create variations of the best sequence with specific improvements
            variant = best_sequence.copy()
            
            # Apply targeted changes based on the sequence characteristics
            if len(variant) > 10:
                # Add some fine-tuning
                for _ in range(5):
                    # Modify some positions
                    idx = random.randint(0, len(variant)-1)
                    change_factor = random.uniform(0.9, 1.1)  # Small multiplicative change
                    variant[idx] = max(0, variant[idx] * change_factor)
            
            # Optimize this variant
            optimized_variant = optimize_with_local_search(variant, max_iter=1000)
            inv_c1 = inv_c1_objective(optimized_variant)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = optimized_variant.copy()
    
    # Return best result or fallback
    return best_sequence if best_sequence is not None else generate_focused_sequence(100)

def search_for_best_sequence():
    """Main search function with improved strategies."""
    start_time = time.time()
    
    # Try the improved adaptive search approach
    try:
        sequence = improved_adaptive_search()
    except Exception as e:
        # Fallback to simple approach if anything goes wrong
        sequence = generate_focused_sequence(100)
    
    # Final optimization pass with focused effort
    try:
        optimized_sequence = optimize_with_local_search(sequence, max_iter=8000, aggressive=True)
        final_inv_c1 = inv_c1_objective(optimized_sequence)
        original_inv_c1 = inv_c1_objective(sequence)
        
        if final_inv_c1 > original_inv_c1:
            sequence = optimized_sequence
    except Exception:
        pass
    
    return sequence

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    sequence = search_for_best_sequence()
    print(f"Found sequence: {sequence}")
