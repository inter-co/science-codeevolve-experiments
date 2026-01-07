# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import time

# Global constants
N_CIRCLES = 32
BENCHMARK = 2.937944526205518
MAX_TIME = 60.0

def initialize_grid_config() -> np.ndarray:
    """Initialize circles using a grid-based approach for good starting configuration"""
    circles = np.zeros((N_CIRCLES, 3))
    
    # Create a grid pattern with some randomness
    rows = int(np.ceil(np.sqrt(N_CIRCLES)))
    cols = int(np.ceil(N_CIRCLES / rows))
    
    # Ensure we have enough space for circles
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= N_CIRCLES:
                break
            # Add some randomness to positions
            x = spacing_x * (j + 1) + random.uniform(-spacing_x/4, spacing_x/4)
            y = spacing_y * (i + 1) + random.uniform(-spacing_y/4, spacing_y/4)
            
            # Initial radius - start small and let optimization increase
            r = min(spacing_x, spacing_y) / 4
            
            # Ensure within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= N_CIRCLES:
            break
    
    # Fill remaining circles with random positions
    for i in range(idx, N_CIRCLES):
        while True:
            x = random.uniform(0.01, 0.99)
            y = random.uniform(0.01, 0.99)
            r = random.uniform(0.01, 0.1)
            
            # Check containment
            if r <= x <= 1-r and r <= y <= 1-r:
                circles[i] = [x, y, r]
                break
    
    return circles

def validate_circles(circles: np.ndarray) -> bool:
    """Validate that all circles satisfy containment and non-overlap constraints"""
    n = len(circles)
    
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if not (r <= x <= 1-r and r <= y <= 1-r):
            return False
    
    # Check non-overlap using efficient pairwise comparison
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < r1 + r2:
                return False
    
    return True

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate sum of all radii"""
    return np.sum(circles[:, 2])

def apply_force_repulsion(circles: np.ndarray, strength: float = 0.05, iterations: int = 35) -> np.ndarray:
    """Apply force-based repulsion between overlapping circles with enhanced vectorized approach"""
    new_circles = circles.copy()
    
    for _ in range(iterations):
        # Vectorized force calculation for better performance
        positions = new_circles[:, :2]
        radii = new_circles[:, 2]
        
        # Compute all pairwise differences efficiently
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        # Create mask for overlapping pairs
        overlap_mask = distances < (radii[:, np.newaxis] + radii[np.newaxis, :])
        np.fill_diagonal(overlap_mask, False)  # Remove self-overlaps
        
        # Initialize forces
        forces = np.zeros_like(positions)
        
        # Apply forces only to overlapping pairs using vectorized operations
        for i in range(N_CIRCLES):
            # Get indices of overlapping circles with circle i
            overlapping_indices = np.where(overlap_mask[i])[0]
            if len(overlapping_indices) > 0:
                # Vectorized force calculation for all overlapping circles
                dx = diff[i, overlapping_indices, 0]
                dy = diff[i, overlapping_indices, 1]
                dist = distances[i, overlapping_indices]
                
                # Avoid division by zero and compute forces
                valid_mask = dist > 0
                if np.any(valid_mask):
                    force_magnitudes = strength * (radii[i] + radii[overlapping_indices] - dist[valid_mask]) / dist[valid_mask]
                    forces[i, 0] -= np.sum(force_magnitudes * dx[valid_mask])
                    forces[i, 1] -= np.sum(force_magnitudes * dy[valid_mask])
        
        # Apply forces and enforce boundaries
        for i in range(N_CIRCLES):
            new_circles[i, 0] += forces[i, 0]
            new_circles[i, 1] += forces[i, 1]
            
            # Keep within bounds with some margin
            x, y, r = new_circles[i]
            new_circles[i, 0] = max(r, min(1-r, x))
            new_circles[i, 1] = max(r, min(1-r, y))
    
    return new_circles

def optimize_with_local_search(circles: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """Use enhanced local search to improve configuration"""
    current = circles.copy()
    best = circles.copy()
    best_sum = calculate_radius_sum(best)
    
    for _ in range(max_iter):
        # Try multiple types of perturbations for better exploration
        new_circles = current.copy()
        
        # Select more circles to perturb for better exploration
        num_perturbations = min(5, N_CIRCLES // 3)
        perturbed_indices = random.sample(range(N_CIRCLES), num_perturbations)
        
        for idx in perturbed_indices:
            x, y, r = new_circles[idx]
            
            # Try different perturbation strategies with more aggressive moves
            if random.random() < 0.6:  # Position perturbation (more likely)
                new_circles[idx, 0] = max(r, min(1-r, x + random.uniform(-0.02, 0.02)))
                new_circles[idx, 1] = max(r, min(1-r, y + random.uniform(-0.02, 0.02)))
            else:  # Radius perturbation
                new_circles[idx, 2] = max(0.001, min(0.5, r + random.uniform(-0.015, 0.015)))
        
        # Ensure valid configuration
        if validate_circles(new_circles):
            new_sum = calculate_radius_sum(new_circles)
            if new_sum > best_sum:
                best = new_circles.copy()
                best_sum = new_sum
                current = new_circles.copy()
    
    return best

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining grid initialization, force-based optimization, and local search.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Phase 1: Initialize with good configuration
    circles = initialize_grid_config()
    
    # Phase 2: Apply force-based optimization with multiple iterations
    for iteration in range(250):
        if time.time() - start_time > MAX_TIME * 0.8:
            break
        circles = apply_force_repulsion(circles, 0.04, 35)
        circles = optimize_with_local_search(circles, 60)
    
    # Phase 3: Local search refinement with multiple restarts
    best_solution = circles.copy()
    best_sum = calculate_radius_sum(best_solution)
    
    # Try multiple random restarts with local search
    for attempt in range(120):
        if time.time() - start_time > MAX_TIME * 0.95:
            break
            
        # Try random restart with force-based optimization
        temp_circles = initialize_grid_config()
        temp_circles = apply_force_repulsion(temp_circles, 0.04, 90)
        temp_circles = optimize_with_local_search(temp_circles, 250)
        
        temp_sum = calculate_radius_sum(temp_circles)
        if temp_sum > best_sum:
            best_solution = temp_circles.copy()
            best_sum = temp_sum
    
    # Phase 4: Final fine-tuning with more aggressive optimization
    final_circles = best_solution.copy()
    
    # Run more rounds of focused optimization
    for round_num in range(7):
        if time.time() - start_time > MAX_TIME * 0.98:
            break
        # More aggressive force repulsion
        final_circles = apply_force_repulsion(final_circles, 0.05, 50)
        # Enhanced local search with more iterations
        final_circles = optimize_with_local_search(final_circles, 350)
    
    # Phase 5: Radius optimization - try to maximize individual radii
    # This is a greedy approach to increase radii where possible
    for _ in range(400):
        improved = False
        for i in range(N_CIRCLES):
            # Try to increase radius slightly
            old_r = final_circles[i, 2]
            x, y = final_circles[i, :2]
            
            # Maximum possible radius at this position
            max_r = min(x, 1-x, y, 1-y)
            
            # Check what we can safely increase to based on neighbors
            for j in range(N_CIRCLES):
                if i != j:
                    dist = np.sqrt((x - final_circles[j, 0])**2 + (y - final_circles[j, 1])**2)
                    max_r = min(max_r, dist - final_circles[j, 2])
            
            # Increase radius if beneficial and safe (even smaller threshold)
            if max_r > old_r and max_r > old_r * 1.015:
                final_circles[i, 2] = max_r
                improved = True
                
        if not improved:
            break
    
    # Final validation and cleanup
    if not validate_circles(final_circles):
        # If validation fails, do one final optimization pass
        final_circles = optimize_with_local_search(final_circles, 200)
    
    return final_circles


# EVOLVE-BLOCK-END
