# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import time
from typing import Tuple

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and iterative optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n_circles = 26
    max_time = 175  # seconds
    start_time = time.time()
    
    def is_valid_config(circles: np.ndarray) -> bool:
        """Check if configuration satisfies all constraints"""
        # Check containment
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlaps using efficient pairwise distance calculation
        if len(circles) < 2:
            return True
            
        positions = circles[:, :2]
        radii = circles[:, 2]
        distances = cdist(positions, positions)
        np.fill_diagonal(distances, np.inf)
        
        radii_sums = np.add.outer(radii, radii)
        
        # Check if any pair violates overlap constraint
        if np.any(distances < radii_sums):
            return False
            
        return True
    
    def calculate_sum_radii(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])
    
    # Phase 1: Create initial configuration using better hexagonal packing
    def create_initial_config() -> np.ndarray:
        # Create a more sophisticated hexagonal arrangement
        circles = np.zeros((n_circles, 3))
        
        # Try a more efficient arrangement - 6 rows, 5 columns for better coverage
        rows, cols = 6, 5
        
        # Calculate spacing with better margins
        spacing_x = 0.9 / cols  # Leave margin of 0.05 on each side
        spacing_y = 0.9 / rows
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n_circles:
                    break
                    
                # Hexagonal offset for odd rows to create better packing
                x_offset = spacing_x * 0.5 if i % 2 == 1 else 0
                x = 0.05 + (j + 0.5) * spacing_x + x_offset
                y = 0.05 + (i + 0.5) * spacing_y
                
                # Initial radius - larger to allow more aggressive optimization
                r = min(spacing_x, spacing_y) * 0.3
                
                circles[count] = [x, y, r]
                count += 1
        
        # Fill remaining positions intelligently
        while count < n_circles:
            # Try to place in a way that maximizes minimum distance to existing circles
            best_place = None
            best_min_dist = -1
            
            # Try multiple random positions and pick the one with best minimum distance
            for _ in range(100):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                
                # Find minimum distance to existing circles
                min_dist = float('inf')
                for i in range(count):
                    px, py, pr = circles[i]
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    min_dist = min(min_dist, dist - pr)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_place = (x, y)
            
            if best_place is not None:
                x, y = best_place
                # Set radius to be as large as possible while respecting bounds and overlaps
                r = min(0.1, best_min_dist, x, 1-x, y, 1-y)
                if r > 0.001:
                    circles[count] = [x, y, r]
                    count += 1
            else:
                # Fallback to random placement
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = random.uniform(0.01, 0.08)
                circles[count] = [x, y, r]
                count += 1
        
        return circles
    
    # Phase 2: Very aggressive optimization with enhanced adaptive strategies
    def optimize_config(initial_circles: np.ndarray, max_iterations: int = 50000) -> np.ndarray:
        circles = initial_circles.copy()
        best_sum = calculate_sum_radii(circles)
        best_config = circles.copy()
        
        # Track recent improvements to detect stagnation
        recent_improvements = []
        improvement_threshold = 1e-6
        
        # Even more aggressive step sizes
        max_radius_step = 0.05
        pos_step = 0.03
        
        # Add time management to prevent exceeding limits
        last_check_time = time.time()
        
        for iteration in range(max_iterations):
            # Time check every 1000 iterations
            if iteration % 1000 == 0:
                current_time = time.time()
                if current_time - start_time > max_time:
                    break
            
            improved = False
            
            # Shuffle order for better exploration
            indices = list(range(n_circles))
            random.shuffle(indices)
            
            # Try to improve each circle systematically with more aggressive steps
            for i in indices:
                # Save current state
                original_pos = circles[i, :2].copy()
                original_rad = circles[i, 2]
                
                # Try to increase radius aggressively with adaptive step size
                # Find maximum allowable radius
                max_radius = min(
                    original_pos[0], 1 - original_pos[0],
                    original_pos[1], 1 - original_pos[1]
                )
                
                # Check overlap constraints with all other circles
                for j in range(n_circles):
                    if i != j:
                        dist = np.sqrt(np.sum((original_pos - circles[j, :2])**2))
                        max_radius = min(max_radius, dist - circles[j, 2])
                
                # Adaptive radius increase with more aggressive stepping
                if max_radius > original_rad and max_radius > 0:
                    # Calculate how much we can potentially gain
                    gain_ratio = max_radius / (original_rad + 1e-8)
                    
                    # More aggressive adaptive step sizing
                    if gain_ratio > 10:
                        # Very large potential gain - take a big step
                        step_size = min(max_radius_step, max_radius * 0.3)
                    elif gain_ratio > 5:
                        # Large potential gain - take a moderate step
                        step_size = min(max_radius_step * 0.7, max_radius * 0.2)
                    elif gain_ratio > 2:
                        # Moderate potential gain - take a small step
                        step_size = min(max_radius_step * 0.5, max_radius * 0.15)
                    else:
                        # Small potential gain - take tiny step
                        step_size = min(max_radius_step * 0.2, max_radius * 0.1)
                    
                    candidate_radius = min(max_radius, original_rad + step_size)
                    # Update radius and test validity
                    circles[i, 2] = candidate_radius
                    if is_valid_config(circles):
                        improved = True
                    else:
                        # Revert if invalid
                        circles[i, 2] = original_rad
                        
                # Try position adjustments with even larger steps for exploration
                if iteration % 2 == 0:
                    # Even larger random perturbations for aggressive exploration
                    delta_x = random.uniform(-pos_step, pos_step)
                    delta_y = random.uniform(-pos_step, pos_step)
                    
                    new_x = np.clip(original_pos[0] + delta_x, original_rad, 1 - original_rad)
                    new_y = np.clip(original_pos[1] + delta_y, original_rad, 1 - original_rad)
                    
                    # Temporarily update position
                    old_pos = circles[i, :2].copy()
                    circles[i, 0] = new_x
                    circles[i, 1] = new_y
                    
                    if is_valid_config(circles):
                        improved = True
                    else:
                        # Revert position if invalid
                        circles[i, :2] = old_pos
            
            # Update best solution
            current_sum = calculate_sum_radii(circles)
            if current_sum > best_sum:
                best_sum = current_sum
                best_config = circles.copy()
                recent_improvements = []  # Reset recent improvements tracking
            else:
                recent_improvements.append(current_sum)
                # Remove oldest entry if we've tracked too many
                if len(recent_improvements) > 150:
                    recent_improvements.pop(0)
            
            # More aggressive early stopping
            if len(recent_improvements) >= 150:
                if len(recent_improvements) >= 200:
                    recent_diff = abs(recent_improvements[-1] - recent_improvements[0])
                    if recent_diff < improvement_threshold:
                        improvement_threshold *= 0.9
                        if improvement_threshold < 1e-8:
                            break
                else:
                    # For early iterations, be more lenient
                    recent_diff = abs(recent_improvements[-1] - recent_improvements[0])
                    if recent_diff < 1e-5:
                        break
                    
        return best_config
    
    # Phase 3: Multi-start optimization approach with enhanced diversity and more aggressive strategies
    best_solution = None
    best_sum = 0
    
    # Try even more diverse initialization strategies with increased attempts
    for start_iter in range(50):  # Increased from 25 to 50 for more attempts
        if time.time() - start_time > max_time:
            break
            
        # Create a variety of different starting configurations
        if start_iter < 10:  # Standard hexagonal
            initial_config = create_initial_config()
        elif start_iter < 20:  # Random with better spacing
            initial_config = np.zeros((n_circles, 3))
            for i in range(n_circles):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = random.uniform(0.01, 0.08)
                initial_config[i] = [x, y, r]
        elif start_iter < 30:  # Grid-based with different arrangement
            initial_config = np.zeros((n_circles, 3))
            # Try 5x6 grid arrangement
            rows, cols = 5, 6
            spacing_x = 0.9 / cols
            spacing_y = 0.9 / rows
            count = 0
            for i in range(rows):
                for j in range(cols):
                    if count >= n_circles:
                        break
                    x = 0.05 + (j + 0.5) * spacing_x
                    y = 0.05 + (i + 0.5) * spacing_y
                    r = min(spacing_x, spacing_y) * 0.3
                    initial_config[count] = [x, y, r]
                    count += 1
            # Fill remaining positions
            while count < n_circles:
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = random.uniform(0.005, 0.1)
                initial_config[count] = [x, y, r]
                count += 1
        elif start_iter < 40:  # Concentrated center arrangement
            initial_config = np.zeros((n_circles, 3))
            # Place some circles near center, others more spread out
            center_count = 8
            for i in range(center_count):
                x = random.uniform(0.3, 0.7)
                y = random.uniform(0.3, 0.7)
                r = random.uniform(0.02, 0.05)
                initial_config[i] = [x, y, r]
            # Fill remaining with more spread-out placement
            for i in range(center_count, n_circles):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = random.uniform(0.01, 0.06)
                initial_config[i] = [x, y, r]
        else:  # Purely random initialization
            initial_config = np.zeros((n_circles, 3))
            for i in range(n_circles):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = random.uniform(0.005, 0.1)
                initial_config[i] = [x, y, r]
        
        optimized_config = optimize_config(initial_config, 40000)  # Increased iterations
        current_sum = calculate_sum_radii(optimized_config)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_solution = optimized_config.copy()
    
    # Final intensive refinement with highest iteration count and multiple passes
    if best_solution is not None:
        # Try several rounds of very aggressive optimization
        for round_num in range(5):
            if time.time() - start_time > max_time:
                break
            final_config = optimize_config(best_solution, 50000)
            final_sum = calculate_sum_radii(final_config)
            
            # If we're improving significantly, keep going; otherwise stop
            if final_sum > best_sum + 1e-6:  # Significant improvement
                best_sum = final_sum
                best_solution = final_config.copy()
            else:
                break
        
        return best_solution
    else:
        # Fallback to a good starting configuration
        fallback_config = create_initial_config()
        return optimize_config(fallback_config, 40000)


# EVOLVE-BLOCK-END
