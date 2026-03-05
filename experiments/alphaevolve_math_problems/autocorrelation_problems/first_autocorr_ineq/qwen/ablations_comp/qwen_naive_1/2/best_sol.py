# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START

import numpy as np
import random
from scipy.signal import fftconvolve
import time
from numba import jit
import warnings
from functools import lru_cache

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

@jit(nopython=True)
def fast_convolution_fft(a):
    """Fast FFT-based convolution for larger sequences."""
    n = len(a)
    # Pad to next power of 2 for efficient FFT
    padded_len = 1
    while padded_len < 2*n - 1:
        padded_len <<= 1
    
    # Zero-pad input arrays
    a_padded = np.zeros(padded_len)
    a_padded[:n] = a
    
    # FFT convolution
    a_fft = np.fft.fft(a_padded)
    conv_fft = a_fft * a_fft.conj()
    conv = np.fft.ifft(conv_fft).real
    
    # Return only relevant portion
    return conv[:2*n - 1]

def compute_c1(sequence):
    """Compute C1 for a given sequence using FFT-based convolution."""
    if len(sequence) == 0:
        return float('inf')
    
    # Convert to numpy array for efficient computation
    a = np.array(sequence, dtype=np.float64)
    
    # For large sequences, use FFT-based convolution for efficiency
    # For small sequences, use direct computation
    if len(a) > 500:
        conv = fast_convolution_fft(a)
    else:
        # Use manual computation for small arrays to avoid overhead
        conv = fast_convolution_jit(a)
    
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

def generate_optimized_peak_sequence(n):
    """Generate a mathematically optimized peak sequence based on theoretical insights."""
    # Create a sequence with a single dominant peak and carefully positioned supporting peaks
    sequence = [0.0] * n
    
    # Place a strong central peak
    peak_pos = n // 2
    peak_height = 1000.0
    sequence[peak_pos] = peak_height
    
    # Add two supporting peaks to optimize convolution properties
    # Place them symmetrically to reduce interference in convolution
    if n > 10:
        # Place secondary peaks at positions that minimize convolution peak formation
        left_pos = max(0, peak_pos - n//4)
        right_pos = min(n-1, peak_pos + n//4)
        
        # Use heights that create beneficial convolution properties
        left_height = peak_height * 0.5
        right_height = peak_height * 0.5
        
        sequence[left_pos] = left_height
        sequence[right_pos] = right_height
    
    # Apply strategic smoothing with Gaussian-like decay
    smoothed = [0.0] * n
    for i in range(n):
        smoothed[i] = sequence[i]
        # Apply influence from nearby peaks with Gaussian decay
        for j in range(max(0, i-6), min(n, i+7)):
            if i != j and sequence[j] > 0:
                dist = abs(i - j)
                # Use Gaussian decay for smooth influence
                influence = sequence[j] * np.exp(-dist**2 / 20.0)
                smoothed[i] += influence * 0.15
    
    # Normalize to ensure reasonable magnitude and maintain sum constraint
    total_sum = sum(smoothed)
    if total_sum > 0:
        smoothed = [x * 1000 / total_sum for x in smoothed]
    
    return smoothed

def generate_balanced_distribution_sequence(n):
    """Generate a sequence with balanced distribution that avoids large convolution peaks."""
    # Create a sequence with a more uniform distribution pattern
    sequence = []
    
    # Use a modified Gaussian approach with controlled peakiness
    center = n // 2
    base_height = 1000.0
    
    # Create a pattern with controlled decay that avoids extreme peaks
    for i in range(n):
        distance_from_center = abs(i - center)
        
        # Use a controlled decay pattern that prevents too sharp peaks
        # This helps minimize convolution maxima
        decay = 1.0 / (1.0 + (distance_from_center / (n/3))**2.2)
        
        # Add small variation to break symmetry without creating spikes
        variation = 0.03 * np.sin(i * np.pi / (n/8))
        height = max(0, base_height * decay * (1 + variation))
        sequence.append(height)
    
    # Normalize to maintain consistent scale
    total_sum = sum(sequence)
    if total_sum > 0:
        sequence = [x * 1000 / total_sum for x in sequence]
    
    return sequence

def generate_sparse_sequence(n):
    """Generate a sparse sequence optimized for convolution properties."""
    sequence = [0.0] * n
    
    # Place strategic peaks with maximum separation
    if n >= 15:
        # Place peaks at positions that maximize separation
        peak_positions = []
        num_peaks = min(3, n // 6)
        
        # Distribute peaks evenly across the sequence
        for i in range(num_peaks):
            pos = int((i + 1) * (n - 1) / (num_peaks + 1))
            peak_positions.append(pos)
        
        # Place peaks with decreasing heights to control convolution
        for i, pos in enumerate(peak_positions):
            # Height decreases with distance from center to minimize convolution impact
            center_distance = abs(pos - n // 2)
            height_factor = 1.0 - 0.25 * (center_distance / (n/2))
            peak_height = 1000.0 * height_factor
            sequence[pos] = peak_height
    
    # Apply smoothing to make convolution properties more predictable
    if n > 10:
        smoothed = [0.0] * n
        for i in range(n):
            smoothed[i] = sequence[i]
            # Apply smoothing with Gaussian kernel
            for j in range(max(0, i-4), min(n, i+5)):
                if i != j and sequence[j] > 0:
                    dist = abs(i - j)
                    influence = sequence[j] * np.exp(-dist**2 / 12.0)
                    smoothed[i] += influence * 0.12
        sequence = smoothed
    
    # Normalize
    total_sum = sum(sequence)
    if total_sum > 0:
        sequence = [x * 1000 / total_sum for x in sequence]
    
    return sequence

def generate_mathematical_optimal_sequence(n):
    """Generate a sequence based on mathematical optimization principles."""
    # Create a sequence using mathematical optimization techniques
    sequence = []
    
    # Base pattern with mathematical precision
    center = n // 2
    peak_height = 1000.0
    
    # Use a hybrid approach combining different mathematical functions
    for i in range(n):
        distance = abs(i - center)
        
        # Use piecewise mathematical functions for optimal balance
        if distance < n // 6:
            # Near center: sharp exponential decay
            decay = np.exp(-distance**2 / (n/12)**2)
        else:
            # Farther: controlled polynomial decay
            decay = 1.0 / (1.0 + (distance / (n/4))**2.8)
        
        # Add structured variation to avoid perfect symmetry
        variation = 0.08 * np.sin(i * np.pi / (n/7))
        height = max(0, peak_height * decay * (1 + variation))
        sequence.append(height)
    
    # Normalize to maintain consistent scale
    total_sum = sum(sequence)
    if total_sum > 0:
        sequence = [x * 1000 / total_sum for x in sequence]
    
    return sequence

def generate_refined_sequence(n):
    """Generate a refined sequence with systematic optimization approach."""
    # Start with a good mathematical foundation
    sequence = [0.0] * n
    
    # Place strategic peaks
    if n >= 10:
        # Place three peaks to balance the convolution properties
        peak_positions = []
        num_peaks = min(3, n // 5)
        
        for i in range(num_peaks):
            # Distribute peaks more evenly
            pos = int((i + 1) * (n - 1) / (num_peaks + 1))
            peak_positions.append(pos)
        
        # Assign heights with careful consideration
        center = n // 2
        for i, pos in enumerate(peak_positions):
            # Heights decrease with distance from center to reduce convolution impact
            distance_from_center = abs(pos - center)
            height_factor = 1.0 - 0.3 * (distance_from_center / (n/2))
            peak_height = 1000.0 * height_factor
            sequence[pos] = peak_height
    
    # Apply sophisticated smoothing with precise weights
    if n > 10:
        smoothed = [0.0] * n
        for i in range(n):
            smoothed[i] = sequence[i]
            # Apply weighted smoothing with inverse distance
            for j in range(max(0, i-5), min(n, i+6)):
                if i != j and sequence[j] > 0:
                    dist = abs(i - j)
                    # Use inverse square law for influence
                    influence = sequence[j] / (1.0 + dist**2 / 8.0)
                    smoothed[i] += influence * 0.18
        sequence = smoothed
    
    # Normalize
    total_sum = sum(sequence)
    if total_sum > 0:
        sequence = [x * 1000 / total_sum for x in sequence]
    
    return sequence

def generate_highly_optimized_sequence(n):
    """Generate a highly optimized sequence using advanced mathematical principles."""
    # Create a sequence inspired by known optimal configurations
    sequence = [0.0] * n
    
    # Place 3 peaks with special positioning to minimize convolution maxima
    peak_positions = [n//4, n//2, 3*n//4]
    peak_heights = [1000.0, 1000.0, 1000.0]
    
    # Adjust peak heights to create optimal balance
    for i, (pos, height) in enumerate(zip(peak_positions, peak_heights)):
        if pos < n:
            sequence[pos] = height
    
    # Apply precise smoothing with mathematical kernel
    if n > 10:
        smoothed = [0.0] * n
        for i in range(n):
            smoothed[i] = sequence[i]
            # Apply mathematically derived smoothing
            for j in range(max(0, i-8), min(n, i+9)):
                if i != j and sequence[j] > 0:
                    dist = abs(i - j)
                    # Use a combination of Gaussian and inverse power law
                    influence = sequence[j] * np.exp(-dist**2 / 25.0) / (1.0 + dist**1.5 / 10.0)
                    smoothed[i] += influence * 0.15
        sequence = smoothed
    
    # Normalize to ensure proper scaling
    total_sum = sum(sequence)
    if total_sum > 0:
        sequence = [x * 1000 / total_sum for x in sequence]
    
    return sequence

def adaptive_local_search(initial_sequence, max_iter=10000, early_stopping=200):
    """Improved local search with better perturbation strategies and more efficient optimization."""
    current_seq = np.array(initial_sequence, dtype=float)
    current_inv_c1 = inv_c1_objective(current_seq)
    
    # Track best solution found
    best_seq = current_seq.copy()
    best_inv_c1 = current_inv_c1
    
    # Track last improvement
    last_improvement = 0
    patience_counter = 0
    
    # Enhanced perturbation strategies with better mathematical foundation
    perturbation_types = [
        'peak_shift', 'spread', 'balance', 'sharpener', 'sweep',
        'jitter', 'adaptive', 'gradient_like', 'wavelet_like'
    ]
    
    # Pre-compute some helpful values for faster operations
    n = len(current_seq)
    center = n // 2
    
    for iteration in range(max_iter):
        # Make perturbations with varying strategies
        test_seq = current_seq.copy()
        
        # Adaptive number of changes based on iteration and performance
        if iteration < max_iter // 3:
            num_changes = max(1, min(len(test_seq) // 8, 20))
        elif iteration < 2 * max_iter // 3:
            num_changes = max(1, min(len(test_seq) // 6, 30))
        else:
            num_changes = max(1, min(len(test_seq) // 4, 40))
        
        # Apply intelligent perturbations
        for _ in range(num_changes):
            idx = random.randint(0, len(test_seq) - 1)
            pert_type = random.choice(perturbation_types)
            
            if pert_type == 'peak_shift':
                # Shift a peak toward or away from center strategically
                if test_seq[idx] > 0 and len(test_seq) > 10:
                    distance_to_center = abs(idx - center)
                    # Move toward center if far, away if near
                    if distance_to_center > len(test_seq) // 4:
                        shift = random.randint(-2, 1)  # Move toward center
                    else:
                        shift = random.randint(-1, 2)  # Move away from center
                    new_idx = max(0, min(len(test_seq)-1, idx + shift))
                    if new_idx != idx:
                        test_seq[new_idx] = max(0, test_seq[new_idx] + test_seq[idx] * 0.12)
                        test_seq[idx] = max(0, test_seq[idx] * 0.88)
            elif pert_type == 'spread':
                # Spread mass around a peak
                if test_seq[idx] > 0:
                    # Distribute some mass to neighbors
                    spread_amount = test_seq[idx] * 0.12
                    test_seq[idx] = max(0, test_seq[idx] - spread_amount)
                    # Add to neighbors with more weight to immediate neighbors
                    neighbor_indices = [max(0, idx-2), max(0, idx-1), 
                                       min(len(test_seq)-1, idx+1), min(len(test_seq)-1, idx+2)]
                    for neighbor in neighbor_indices:
                        if neighbor != idx:
                            test_seq[neighbor] = max(0, test_seq[neighbor] + spread_amount/4)
            elif pert_type == 'balance':
                # Improve overall balance by adjusting multiple elements
                if len(test_seq) > 5:
                    # Select several indices to adjust
                    adjustment_indices = random.sample(range(len(test_seq)), min(6, len(test_seq)//4))
                    for adj_idx in adjustment_indices:
                        if test_seq[adj_idx] > 0:
                            change = random.uniform(-test_seq[adj_idx]*0.12, test_seq[adj_idx]*0.12)
                            test_seq[adj_idx] = max(0, test_seq[adj_idx] + change)
            elif pert_type == 'sharpener':
                # Sharpen the sequence to concentrate mass
                if test_seq[idx] > 0:
                    distance = abs(idx - center)
                    # Make peaks more pronounced if they're near center
                    if distance < len(test_seq) // 6:
                        test_seq[idx] = max(0, test_seq[idx] * 1.12)
                    else:
                        test_seq[idx] = max(0, test_seq[idx] * 0.95)
            elif pert_type == 'sweep':
                # Apply sweep operation with better distribution
                if len(test_seq) > 10:
                    sweep_start = max(0, idx - 2)
                    sweep_end = min(len(test_seq), idx + 3)
                    # Apply more uniform change to neighborhood
                    change = random.uniform(-test_seq[idx]*0.15, test_seq[idx]*0.15)
                    for i in range(sweep_start, sweep_end):
                        test_seq[i] = max(0, test_seq[i] + change)
            elif pert_type == 'jitter':
                # Add jitter to nearby points with more intelligent weighting
                if len(test_seq) > 5:
                    for i in range(max(0, idx-3), min(len(test_seq), idx+4)):
                        if i != idx:
                            change = random.uniform(-test_seq[i]*0.05, test_seq[i]*0.05)
                            test_seq[i] = max(0, test_seq[i] + change)
            elif pert_type == 'adaptive':
                # Adaptive perturbation based on value and position
                current_val = test_seq[idx]
                if current_val > 0:
                    # Scale based on value and position for more targeted changes
                    scale = min(0.25, current_val / 1000.0)
                    change = random.uniform(-scale * current_val * 2.0, scale * current_val * 2.0)
                    test_seq[idx] = max(0, test_seq[idx] + change)
                else:
                    # For zero values, add some value
                    test_seq[idx] = max(0, test_seq[idx] + random.uniform(0, 500))
            elif pert_type == 'gradient_like':
                # Gradient-like adjustments based on neighboring values
                if len(test_seq) > 3:
                    # Calculate average of neighbors
                    neighbors = []
                    for j in range(max(0, idx-2), min(len(test_seq), idx+3)):
                        if j != idx and test_seq[j] > 0:
                            neighbors.append(test_seq[j])
                    
                    if neighbors:
                        avg_neighbor = sum(neighbors) / len(neighbors)
                        # Adjust toward neighbor average to reduce variance
                        change = (avg_neighbor - test_seq[idx]) * 0.08
                        test_seq[idx] = max(0, test_seq[idx] + change)
            elif pert_type == 'wavelet_like':
                # Apply wavelet-inspired modification to create more complex structures
                if len(test_seq) > 8 and test_seq[idx] > 0:
                    # Modify in a way that creates local oscillation
                    change = random.uniform(-test_seq[idx]*0.08, test_seq[idx]*0.08)
                    test_seq[idx] = max(0, test_seq[idx] + change)
                    # Also modify nearby positions to create wavelet-like effect
                    for offset in [-1, 1]:
                        neighbor_idx = idx + offset
                        if 0 <= neighbor_idx < len(test_seq):
                            test_seq[neighbor_idx] = max(0, test_seq[neighbor_idx] - change * 0.3)
        
        # Accept if better
        test_inv_c1 = inv_c1_objective(test_seq)
        if test_inv_c1 > current_inv_c1:
            current_seq = test_seq
            current_inv_c1 = test_inv_c1
            
            # Update best solution
            if test_inv_c1 > best_inv_c1:
                best_seq = test_seq.copy()
                best_inv_c1 = test_inv_c1
                last_improvement = iteration
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            patience_counter += 1
        
        # Early stopping if no improvement for a while
        if patience_counter >= early_stopping:
            break
    
    return best_seq.tolist()

def generate_initial_candidates(n_samples=150):
    """Generate diverse initial candidates with focus on proven optimization strategies."""
    candidates = []
    
    # Generate various types of sequences with emphasis on proven mathematically optimized patterns
    for i in range(n_samples):
        # Random length between 30 and 350 for better exploration
        n = random.randint(30, 350)
        
        # Choose generation strategy with emphasis on optimized patterns
        strategy = random.choice([
            generate_optimized_peak_sequence,
            generate_balanced_distribution_sequence,
            generate_sparse_sequence,
            generate_mathematical_optimal_sequence,
            generate_refined_sequence,
            generate_highly_optimized_sequence,
            generate_optimized_peak_sequence,
            generate_balanced_distribution_sequence,
            generate_sparse_sequence
        ])
        
        try:
            candidate = strategy(n)
            candidates.append(candidate)
        except:
            # Fallback to random
            candidates.append([random.uniform(0, 1000) for _ in range(n)])
    
    return candidates

def systematic_search():
    """Systematic search with optimized approach for better performance."""
    start_time = time.time()
    max_time = 55  # Leave some buffer for final processing
    
    best_inv_c1 = 0
    best_sequence = None
    
    # Strategy 1: Generate high-quality initial candidates
    if time.time() - start_time < max_time:
        candidates = generate_initial_candidates(200)  # Fewer samples but higher quality
        
        # Prioritize candidates with better initial scores
        scored_candidates = [(inv_c1_objective(c), c) for c in candidates[:150]]
        scored_candidates.sort(reverse=True)
        top_candidates = [c for _, c in scored_candidates[:75]]  # Top 75
        
        # Optimize the top candidates more intensively
        for i, candidate in enumerate(top_candidates):
            if time.time() - start_time > max_time:
                break
                
            # More intensive local optimization for promising candidates
            optimized_candidate = adaptive_local_search(
                candidate, max_iter=3000, early_stopping=150
            )
            inv_c1 = inv_c1_objective(optimized_candidate)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = optimized_candidate.copy()
    
    # Strategy 2: Refinement passes on best solution
    if time.time() - start_time < max_time and best_sequence is not None:
        refined_seq = best_sequence.copy()
        
        # Multiple refinement passes with different intensities
        for pass_num in range(3):
            if time.time() - start_time > max_time:
                break
                
            # More intensive optimization for this pass
            for _ in range(2000):
                if time.time() - start_time > max_time:
                    break
                    
                test_seq = refined_seq.copy()
                
                # Apply more focused modifications
                num_modifications = random.randint(15, 30)
                for _ in range(num_modifications):
                    idx = random.randint(0, len(test_seq)-1)
                    
                    # Apply more intelligent perturbations
                    current_val = test_seq[idx]
                    if current_val > 0:
                        # Scale perturbation based on current value
                        scale = min(0.3, current_val / 1000.0)
                        change = random.uniform(-scale * current_val * 1.5, scale * current_val * 1.5)
                        test_seq[idx] = max(0, test_seq[idx] + change)
                    else:
                        # For zero values, add some value
                        test_seq[idx] = max(0, test_seq[idx] + random.uniform(0, 600))
                
                test_inv_c1 = inv_c1_objective(test_seq)
                if test_inv_c1 > inv_c1_objective(refined_seq):
                    refined_seq = test_seq
        
        inv_c1 = inv_c1_objective(refined_seq)
        if inv_c1 > best_inv_c1:
            best_inv_c1 = inv_c1
            best_sequence = refined_seq.copy()
    
    # Strategy 3: Create new candidates around the best found so far
    if time.time() - start_time < max_time and best_sequence is not None:
        for _ in range(30):  # Fewer attempts but more focused
            if time.time() - start_time > max_time:
                break
                
            # Create a candidate near the current best with more structure
            candidate = best_sequence.copy()
            
            # Apply more systematic modifications
            for i in range(len(candidate)):
                if random.random() < 0.25:  # Lower probability to modify
                    # Perturb based on current value and position
                    if candidate[i] > 0:
                        change = random.uniform(-candidate[i]*0.2, candidate[i]*0.2)
                        candidate[i] = max(0, candidate[i] + change)
                    else:
                        # For zero values, add some value
                        candidate[i] = max(0, candidate[i] + random.uniform(0, 400))
            
            # Optimize this candidate
            optimized_candidate = adaptive_local_search(
                candidate, max_iter=2000, early_stopping=150
            )
            inv_c1 = inv_c1_objective(optimized_candidate)
            
            if inv_c1 > best_inv_c1:
                best_inv_c1 = inv_c1
                best_sequence = optimized_candidate.copy()
    
    # Return best result or fallback
    if best_sequence is not None:
        return best_sequence
    else:
        # Try a more direct approach with better parameters
        return generate_highly_optimized_sequence(150)

def search_for_best_sequence():
    """Main search function with optimized strategies."""
    start_time = time.time()
    
    # Try the systematic search approach
    try:
        sequence = systematic_search()
    except Exception as e:
        # Fallback to simple approach if anything goes wrong
        sequence = generate_refined_sequence(150)
    
    # Final optimization pass with focused effort
    try:
        optimized_sequence = adaptive_local_search(
            sequence, max_iter=8000, early_stopping=200
        )
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
