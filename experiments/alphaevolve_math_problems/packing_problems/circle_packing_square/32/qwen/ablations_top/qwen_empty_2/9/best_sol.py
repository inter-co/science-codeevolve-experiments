# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import time
from numba import jit
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

@jit(nopython=True)
def check_collision_fast(circles, i, j):
    """Fast collision checking using numba"""
    x1, y1, r1 = circles[i]
    x2, y2, r2 = circles[j]
    
    dx = x1 - x2
    dy = y1 - y2
    dist_sq = dx*dx + dy*dy
    
    return dist_sq < (r1 + r2) * (r1 + r2)

def is_valid_configuration(circles):
    """Check if configuration is valid (no overlaps, inside bounds)"""
    n = len(circles)
    
    # Check boundary constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            if check_collision_fast(circles, i, j):
                return False
    
    return True

def evaluate_fitness(circles):
    """Evaluate fitness (sum of radii)"""
    if not is_valid_configuration(circles):
        return -1000000  # Penalize invalid configurations heavily
    
    total_radius = np.sum(circles[:, 2])
    return total_radius

def generate_high_quality_initial():
    """Generate a very high-quality initial configuration based on mathematical optimization principles"""
    circles = np.zeros((32, 3))
    
    # Create a configuration inspired by the best-known circle packing solutions
    # Use a pattern that balances density with spatial distribution
    
    # 6 rows, 6 columns for 36 positions, then trim to 32
    rows = 6
    cols = 6
    
    # Precise spacing for maximum packing efficiency
    spacing_x = 0.9 / cols
    spacing_y = 0.9 / rows
    margin_x = 0.05
    margin_y = 0.05
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= 32:
                break
                
            # Hexagonal offset for even better packing
            offset = (row % 2) * spacing_x / 2
            x = margin_x + col * spacing_x + offset + spacing_x/2
            y = margin_y + row * spacing_y + spacing_y/2
            
            # Add minimal but sufficient randomness to avoid symmetries
            x += np.random.normal(0, 0.005)
            y += np.random.normal(0, 0.005)
            
            # Use a sophisticated radius distribution that promotes good packing
            # Central positions get larger radii, outer positions smaller
            center_x = 0.5
            center_y = 0.5
            distance_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Map distance to radius - inner areas get larger circles
            if distance_to_center < 0.25:
                base_radius = 0.06 + np.random.uniform(0, 0.015)
            elif distance_to_center < 0.5:
                base_radius = 0.04 + np.random.uniform(0, 0.01)
            else:
                base_radius = 0.03 + np.random.uniform(0, 0.008)
            
            r = min(base_radius, 0.35)
            
            # Clamp to valid ranges
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            r = max(0.01, min(0.4, r))
            
            circles[idx] = [x, y, r]
            idx += 1
    
    return circles

def enhanced_force_application(circles, force_strength=0.001, repulsion_power=2.0, max_iterations=150):
    """Enhanced force application with adaptive parameters and better convergence"""
    n = len(circles)
    
    # Apply multiple rounds with adaptive parameters
    for iteration in range(max_iterations):
        forces = np.zeros((n, 2))  # Force vectors (dx, dy)
        
        # Compute forces between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                dx = x1 - x2
                dy = y1 - y2
                dist = np.sqrt(dx*dx + dy*dy)
                
                # If circles overlap
                if dist < (r1 + r2) and dist > 1e-10:
                    # Normalize direction vector
                    if dist > 1e-10:
                        dx_norm = dx / dist
                        dy_norm = dy / dist
                        
                        # Enhanced force calculation with adaptive strength
                        overlap = (r1 + r2) - dist
                        # Gradually decrease force strength for stability
                        adaptive_strength = force_strength * (1.0 - iteration/max_iterations)
                        force_magnitude = adaptive_strength * overlap * (1.0 / (dist ** repulsion_power))
                        
                        # Apply force to both circles (opposite directions)
                        forces[i, 0] += force_magnitude * dx_norm
                        forces[i, 1] += force_magnitude * dy_norm
                        forces[j, 0] -= force_magnitude * dx_norm
                        forces[j, 1] -= force_magnitude * dy_norm
        
        # Apply forces to positions with adaptive damping
        new_circles = circles.copy()
        for i in range(n):
            x, y, r = new_circles[i]
            
            # Apply force with adaptive damping
            damping_factor = 0.7 + 0.2 * (1.0 - iteration/max_iterations)  # Gradually decrease damping
            new_x = x + forces[i, 0] * damping_factor
            new_y = y + forces[i, 1] * damping_factor
            
            # Keep within bounds
            new_x = max(r, min(1-r, new_x))
            new_y = max(r, min(1-r, new_y))
            
            new_circles[i] = [new_x, new_y, r]
        
        circles = new_circles
    
    return circles

def sophisticated_local_search(circles, max_iterations=300):
    """Sophisticated local search with multiple optimization strategies and adaptive mechanisms"""
    current = circles.copy()
    best_fitness = evaluate_fitness(current)
    best_config = current.copy()
    
    # Track performance for adaptive strategies
    improvement_history = []
    
    for iteration in range(max_iterations):
        # Strategy adaptation based on recent performance
        if len(improvement_history) > 15:
            recent_avg = np.mean(improvement_history[-15:])
            if recent_avg < 0.0001:
                strategy = "exploration"
            elif recent_avg < 0.001:
                strategy = "balanced"
            else:
                strategy = "exploitation"
        else:
            strategy = "balanced"
        
        candidate = current.copy()
        
        # Choose optimization strategy
        if strategy == "exploration":
            # Large-scale changes to escape local minima
            # Try several different moves
            for _ in range(5):  # Try 5 different moves for more aggressive exploration
                idx = random.randint(0, len(candidate)-1)
                x, y, r = candidate[idx]
                # Even larger perturbations for exploration
                x += np.random.normal(0, 0.03)
                y += np.random.normal(0, 0.03)
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                candidate[idx] = [x, y, r]
        elif strategy == "exploitation":
            # Fine-grained optimization for maximum improvement
            # Very small changes to fine-tune existing configuration
            for _ in range(5):  # Try 5 small moves for more precision
                idx = random.randint(0, len(candidate)-1)
                x, y, r = candidate[idx]
                # Very small changes for fine-tuning
                x += np.random.normal(0, 0.0005)
                y += np.random.normal(0, 0.0005)
                # Multiplicative change for radius with very small variance
                r *= np.exp(np.random.normal(0, 0.002))
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                r = max(0.01, min(0.4, r))
                candidate[idx] = [x, y, r]
        else:  # balanced
            # Mixed approach for general optimization
            idx = random.randint(0, len(candidate)-1)
            x, y, r = candidate[idx]
            # Mix of small and medium changes
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            # Small multiplicative change for radius
            r *= np.exp(np.random.normal(0, 0.01))
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            r = max(0.01, min(0.4, r))
            candidate[idx] = [x, y, r]
        
        # Check validity and accept if improvement
        if is_valid_configuration(candidate):
            fitness = evaluate_fitness(candidate)
            improvement = fitness - best_fitness
            improvement_history.append(improvement)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_config = candidate.copy()
                current = candidate.copy()
            elif random.random() < 0.05:  # Occasionally accept worse solutions for diversity
                current = candidate.copy()
        else:
            # Accept invalid solutions with slightly higher probability to maintain diversity
            if random.random() < 0.005:
                current = candidate.copy()
    
    return best_config

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Generate extremely high-quality initial configuration
    initial_circles = generate_high_quality_initial()
    
    # Apply comprehensive physics-based optimization to get a valid configuration
    current_circles = initial_circles.copy()
    
    # Multiple phases of physics-based optimization with varying intensities
    # Phase 1: Strong force field to resolve major overlaps
    current_circles = enhanced_force_application(current_circles, force_strength=0.02, max_iterations=120)
    
    # Phase 2: Medium force field for fine-tuning
    current_circles = enhanced_force_application(current_circles, force_strength=0.01, max_iterations=100)
    
    # Phase 3: Light force field for final refinement
    current_circles = enhanced_force_application(current_circles, force_strength=0.005, max_iterations=80)
    
    # Multi-phase optimization strategy
    best_fitness = evaluate_fitness(current_circles)
    best_solution = current_circles.copy()
    
    # Phase 1: Coarse optimization with aggressive changes
    for phase in range(3):
        for iteration in range(100):  # More iterations for better exploitation
            improved = sophisticated_local_search(current_circles, max_iterations=35)
            fitness = evaluate_fitness(improved)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = improved.copy()
                current_circles = improved.copy()
    
    # Phase 2: Balanced refinement with mixed strategies
    for iteration in range(150):
        # Alternate between physics and local search with strategic intensity
        if iteration % 4 == 0:
            # Physics-based refinement with moderate force
            current_circles = enhanced_force_application(current_circles, force_strength=0.003, max_iterations=35)
        else:
            # Local search refinement
            improved = sophisticated_local_search(current_circles, max_iterations=40)
            fitness = evaluate_fitness(improved)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = improved.copy()
                current_circles = improved.copy()
    
    # Phase 3: Intense fine-tuning with high-intensity local search
    for iteration in range(200):  # More iterations in final phase
        improved = sophisticated_local_search(current_circles, max_iterations=60)
        fitness = evaluate_fitness(improved)
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = improved.copy()
            current_circles = improved.copy()
    
    # Final validation and cleanup
    if not is_valid_configuration(best_solution):
        # Revert to initial if somehow invalid
        best_solution = generate_high_quality_initial()
    
    # Final intensive optimization pass with even more aggressive search
    final_solution = sophisticated_local_search(best_solution, max_iterations=250)
    
    # Make sure we're within time limits
    elapsed = time.time() - start_time
    if elapsed > 55:  # Leave buffer for final processing
        print(f"Time limit approaching: {elapsed:.2f}s")
    
    return final_solution


# EVOLVE-BLOCK-END
