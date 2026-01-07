# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List
import random

# Global constants for optimization
MAX_ITER = 500
CIRCLE_COUNT = 32

def _compute_distance_matrix(circles: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between circle centers efficiently"""
    centers = circles[:, :2]
    return cdist(centers, centers)

def _check_constraints(circles: np.ndarray) -> bool:
    """Check if all circles satisfy containment and non-overlap constraints"""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check non-overlap constraints using efficient matrix computation
    dist_matrix = _compute_distance_matrix(circles)
    for i in range(n):
        for j in range(i+1, n):
            distance = dist_matrix[i, j]
            radius_sum = circles[i, 2] + circles[j, 2]
            if distance < radius_sum:
                return False
    
    return True

def _initialize_hexagonal_pattern() -> np.ndarray:
    """Initialize circles in a hexagonal pattern for good starting configuration"""
    circles = np.zeros((CIRCLE_COUNT, 3))
    
    # Create a more sophisticated hexagonal packing
    rows = 6
    cols = 6
    spacing_x = 0.15
    spacing_y = 0.15
    radius = 0.04  # Slightly smaller initial radius
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= CIRCLE_COUNT:
                break
            x = 0.1 + j * spacing_x
            y = 0.1 + i * spacing_y
            
            # Apply hexagonal offset for odd rows
            if i % 2 == 1:
                x += spacing_x / 2
                
            # Ensure circles stay within bounds
            if x - radius >= 0 and x + radius <= 1 and y - radius >= 0 and y + radius <= 1:
                circles[idx] = [x, y, radius]
                idx += 1
        if idx >= CIRCLE_COUNT:
            break
    
    # Fill remaining circles with random positions but reasonable radii
    for i in range(idx, CIRCLE_COUNT):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = np.random.uniform(0.02, 0.05)
        # Ensure within bounds
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        circles[i] = [x, y, r]
        
    return circles

def _initialize_random_pattern() -> np.ndarray:
    """Initialize circles with completely random positions"""
    circles = np.zeros((CIRCLE_COUNT, 3))
    for i in range(CIRCLE_COUNT):
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        r = np.random.uniform(0.02, 0.05)
        # Ensure within bounds
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        circles[i] = [x, y, r]
    return circles

def _initialize_grid_pattern() -> np.ndarray:
    """Initialize circles in a grid pattern"""
    circles = np.zeros((CIRCLE_COUNT, 3))
    
    # Create a 6x6 grid with slight randomness
    rows = 6
    cols = 6
    spacing_x = 0.16
    spacing_y = 0.16
    radius = 0.04
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= CIRCLE_COUNT:
                break
            x = 0.1 + j * spacing_x + np.random.uniform(-0.01, 0.01)
            y = 0.1 + i * spacing_y + np.random.uniform(-0.01, 0.01)
            
            # Ensure circles stay within bounds
            if x - radius >= 0 and x + radius <= 1 and y - radius >= 0 and y + radius <= 1:
                circles[idx] = [x, y, radius]
                idx += 1
        if idx >= CIRCLE_COUNT:
            break
    
    # Fill remaining circles with small random radii
    for i in range(idx, CIRCLE_COUNT):
        circles[i] = [0.5, 0.5, 0.02]
        
    return circles

def _apply_forces(circles: np.ndarray) -> np.ndarray:
    """Apply repulsive forces between overlapping circles with enhanced physics"""
    n = len(circles)
    forces = np.zeros((n, 2))  # Forces on centers only
    
    # Compute pairwise distances and forces using vectorized operations
    dist_matrix = _compute_distance_matrix(circles)
    
    # Vectorized computation of forces - more efficient approach with stronger forces like inspiration programs
    # Pre-compute all pairwise overlaps and forces with even stronger force application
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            distance = dist_matrix[i, j]
            
            # If circles overlap, apply repulsive force
            if distance < (circles[i, 2] + circles[j, 2]):
                overlap = (circles[i, 2] + circles[j, 2]) - distance
                # Force proportional to overlap and inverse of distance with even stronger force magnitude
                force_magnitude = overlap * 3.0  # Even stronger force for better separation
                
                if distance > 1e-10:  # Avoid division by zero
                    force = force_magnitude / distance
                    forces[i, 0] += force * dx
                    forces[i, 1] += force * dy
                    forces[j, 0] -= force * dx
                    forces[j, 1] -= force * dy
    
    # Apply forces to move circles with more controlled updates
    new_circles = circles.copy()
    for i in range(n):
        # Update positions with forces but with damping - larger step size for more aggressive movement
        step_size = 0.008  # Even larger step size for more aggressive movement
        new_circles[i, 0] += forces[i, 0] * step_size
        new_circles[i, 1] += forces[i, 1] * step_size
        
        # Ensure circles don't go out of bounds with more careful clipping
        r = new_circles[i, 2]
        new_circles[i, 0] = np.clip(new_circles[i, 0], r, 1 - r)
        new_circles[i, 1] = np.clip(new_circles[i, 1], r, 1 - r)
        
    return new_circles

def _optimize_circles(circles: np.ndarray) -> np.ndarray:
    """Optimize circle positions and radii using hybrid approach"""
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Multi-phase optimization with even more aggressive refinement and better termination
    # Following the approach from inspiration programs with more phases and aggressive settings
    for phase in range(7):  # Increased to 7 phases for even better convergence
        # Phase 1: Physics-based force optimization (most iterations)
        if phase == 0:
            iterations = 400  # Even more iterations in early phases for aggressive exploration
        elif phase == 1:
            iterations = 350
        elif phase == 2:
            iterations = 300
        elif phase == 3:
            iterations = 250
        elif phase == 4:
            iterations = 200
        elif phase == 5:
            iterations = 150
        else:  # phase == 6
            iterations = 100
            
        phase_improvements = 0  # Track improvements in this phase
        last_improvement_iteration = 0  # Track when last improvement occurred
        
        for iteration in range(iterations):
            # Apply physics-based force optimization
            circles_with_forces = _apply_forces(best_circles)
            
            # Try to increase radii where beneficial - most aggressive approach like inspiration programs
            improved = False
            for i in range(len(best_circles)):
                old_radius = best_circles[i, 2]
                # Try increasing radius by up to 20% for small radii, 15% for medium, 10% for large
                if old_radius < 0.05:
                    new_radius = min(old_radius * 1.25, 0.2)  # Very aggressive for small radii
                elif old_radius < 0.1:
                    new_radius = min(old_radius * 1.2, 0.2)
                elif old_radius < 0.15:
                    new_radius = min(old_radius * 1.15, 0.2)
                else:
                    new_radius = min(old_radius * 1.1, 0.2)
                
                # Test if we can increase this radius without violating constraints
                test_circles = best_circles.copy()
                test_circles[i, 2] = new_radius
                
                # Check constraints
                if _check_constraints(test_circles):
                    test_sum = np.sum(test_circles[:, 2])
                    if test_sum > best_sum:
                        best_circles = test_circles
                        best_sum = test_sum
                        improved = True
                        phase_improvements += 1
                        last_improvement_iteration = iteration
            
            # Adaptive perturbation based on progress like in inspiration programs
            if not improved:
                # If we haven't improved in a while, increase the chance of perturbation
                if iteration - last_improvement_iteration > iterations // 4:
                    perturb_chance = 0.3  # Higher chance when stuck
                else:
                    perturb_chance = 0.2  # Normal chance
            else:
                perturb_chance = 0.2  # Reset to normal chance after improvement
                
            # If no improvement, try small random perturbations to escape local optima
            if not improved:
                # Small random perturbations to escape local optima
                for i in range(len(best_circles)):
                    if np.random.random() < perturb_chance:  # Adjustable probability
                        test_circles = best_circles.copy()
                        # Perturb position slightly with larger steps
                        test_circles[i, 0] += np.random.uniform(-0.015, 0.015)
                        test_circles[i, 1] += np.random.uniform(-0.015, 0.015)
                        # Perturb radius slightly with larger steps
                        test_circles[i, 2] += np.random.uniform(-0.006, 0.006)
                        
                        # Ensure valid radius and within bounds
                        test_circles[i, 2] = max(0.001, min(0.2, test_circles[i, 2]))
                        test_circles[i, 0] = np.clip(test_circles[i, 0], 
                                                    test_circles[i, 2], 1 - test_circles[i, 2])
                        test_circles[i, 1] = np.clip(test_circles[i, 1], 
                                                    test_circles[i, 2], 1 - test_circles[i, 2])
                        
                        if _check_constraints(test_circles):
                            test_sum = np.sum(test_circles[:, 2])
                            if test_sum > best_sum:
                                best_circles = test_circles
                                best_sum = test_sum
                                improved = True
                                phase_improvements += 1
                                last_improvement_iteration = iteration
        
        # Early termination if little improvement in this phase
        if phase_improvements < 3 and phase > 3:
            break
    
    return best_circles

def _local_search_refinement(circles: np.ndarray) -> np.ndarray:
    """Perform intensive local search refinement with enhanced strategies"""
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Enhanced local search for fine-tuning with more systematic approach
    # Following inspiration program approach with more thorough search
    max_iterations = 800  # Even more iterations for thorough search
    consecutive_failures = 0  # Track consecutive failure to improve
    
    for iteration in range(max_iterations):  # Even more iterations for more thorough search
        current_sum = np.sum(best_circles[:, 2])
        improved = False
        
        # Strategy 1: Adjust individual radii - most aggressive approach
        for i in range(CIRCLE_COUNT):
            # Try to increase radius more aggressively like in inspiration programs
            test_radius = min(best_circles[i, 2] * 1.12, 0.2)  # Even more aggressive increment
            test_circles = best_circles.copy()
            test_circles[i, 2] = test_radius
            
            # Check constraints
            if _check_constraints(test_circles):
                test_sum = np.sum(test_circles[:, 2])
                if test_sum > current_sum:
                    best_circles = test_circles
                    improved = True
                    consecutive_failures = 0  # Reset counter on success
        
        # Strategy 2: Try position adjustments with larger steps when no radius improvement
        if not improved:
            for i in range(CIRCLE_COUNT):
                if np.random.random() < 0.35:  # Increased to 35% chance to adjust position for more exploration
                    test_circles = best_circles.copy()
                    test_circles[i, 0] += np.random.uniform(-0.02, 0.02)  # Even larger step
                    test_circles[i, 1] += np.random.uniform(-0.02, 0.02)
                    
                    # Keep within bounds
                    test_circles[i, 0] = np.clip(test_circles[i, 0], 
                                                test_circles[i, 2], 1 - test_circles[i, 2])
                    test_circles[i, 1] = np.clip(test_circles[i, 1], 
                                                test_circles[i, 2], 1 - test_circles[i, 2])
                    
                    if _check_constraints(test_circles):
                        test_sum = np.sum(test_circles[:, 2])
                        if test_sum > current_sum:
                            best_circles = test_circles
                            improved = True
                            consecutive_failures = 0  # Reset counter on success
        
        # Strategy 3: Try simultaneous small adjustments to pairs of circles with higher frequency
        if not improved and iteration % 2 == 0:  # Even more frequent pair adjustments
            # Try to adjust pairs of nearby circles simultaneously
            for i in range(CIRCLE_COUNT):
                for j in range(i+1, CIRCLE_COUNT):
                    # Check if circles are close enough to potentially benefit from joint adjustment
                    dist = np.sqrt((best_circles[i, 0] - best_circles[j, 0])**2 + 
                                 (best_circles[i, 1] - best_circles[j, 1])**2)
                    if dist < (best_circles[i, 2] + best_circles[j, 2]) * 1.05:  # Even tighter proximity check
                        test_circles = best_circles.copy()
                        # Even larger adjustments to both
                        test_circles[i, 0] += np.random.uniform(-0.01, 0.01)
                        test_circles[i, 1] += np.random.uniform(-0.01, 0.01)
                        test_circles[j, 0] += np.random.uniform(-0.01, 0.01)
                        test_circles[j, 1] += np.random.uniform(-0.01, 0.01)
                        
                        # Keep within bounds
                        for k in [i, j]:
                            test_circles[k, 0] = np.clip(test_circles[k, 0], 
                                                        test_circles[k, 2], 1 - test_circles[k, 2])
                            test_circles[k, 1] = np.clip(test_circles[k, 1], 
                                                        test_circles[k, 2], 1 - test_circles[k, 2])
                        
                        if _check_constraints(test_circles):
                            test_sum = np.sum(test_circles[:, 2])
                            if test_sum > current_sum:
                                best_circles = test_circles
                                improved = True
                                consecutive_failures = 0  # Reset counter on success
        
        # Strategy 4: Periodic full re-evaluation to ensure we haven't missed better solutions
        if not improved and iteration % 8 == 0:
            # Try some random global adjustments with more thorough search
            for _ in range(12):  # Try even more random adjustments per 8 iterations
                i = np.random.randint(0, CIRCLE_COUNT)
                test_circles = best_circles.copy()
                test_circles[i, 0] += np.random.uniform(-0.02, 0.02)
                test_circles[i, 1] += np.random.uniform(-0.02, 0.02)
                test_circles[i, 2] += np.random.uniform(-0.01, 0.01)
                
                # Ensure valid radius and within bounds
                test_circles[i, 2] = max(0.001, min(0.2, test_circles[i, 2]))
                test_circles[i, 0] = np.clip(test_circles[i, 0], 
                                            test_circles[i, 2], 1 - test_circles[i, 2])
                test_circles[i, 1] = np.clip(test_circles[i, 1], 
                                            test_circles[i, 2], 1 - test_circles[i, 2])
                
                if _check_constraints(test_circles):
                    test_sum = np.sum(test_circles[:, 2])
                    if test_sum > current_sum:
                        best_circles = test_circles
                        improved = True
                        consecutive_failures = 0  # Reset counter on success
        
        # Strategy 5: Add a special mode for final refinement when stuck
        if not improved and consecutive_failures > 15 and iteration > max_iterations // 3:
            # Try a more radical approach for final optimization
            for i in range(CIRCLE_COUNT):
                # Try to make even larger adjustments
                test_circles = best_circles.copy()
                test_circles[i, 0] += np.random.uniform(-0.025, 0.025)
                test_circles[i, 1] += np.random.uniform(-0.025, 0.025)
                test_circles[i, 2] += np.random.uniform(-0.015, 0.015)
                
                # Ensure valid radius and within bounds
                test_circles[i, 2] = max(0.001, min(0.2, test_circles[i, 2]))
                test_circles[i, 0] = np.clip(test_circles[i, 0], 
                                            test_circles[i, 2], 1 - test_circles[i, 2])
                test_circles[i, 1] = np.clip(test_circles[i, 1], 
                                            test_circles[i, 2], 1 - test_circles[i, 2])
                
                if _check_constraints(test_circles):
                    test_sum = np.sum(test_circles[:, 2])
                    if test_sum > current_sum:
                        best_circles = test_circles
                        improved = True
                        consecutive_failures = 0  # Reset counter on success
        
        # Early termination if we've failed to improve for too long
        if not improved:
            consecutive_failures += 1
            if consecutive_failures > 70:  # Stop after 70 consecutive failures
                break
        else:
            consecutive_failures = 0  # Reset counter on success
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Multi-start approach with different initializations
    best_solution = None
    best_sum = 0
    
    # Try different initialization strategies with more thorough testing
    initializers = [
        _initialize_hexagonal_pattern,
        _initialize_random_pattern,
        _initialize_grid_pattern
    ]
    
    # Add deterministic initialization to ensure consistency
    np.random.seed(42)  # Set seed for reproducibility
    
    for i, initializer in enumerate(initializers):
        try:
            # Initialize with different patterns
            circles = initializer()
            
            # Apply optimization with more iterations
            optimized_circles = _optimize_circles(circles)
            
            # Final refinement with more aggressive search
            refined_circles = _local_search_refinement(optimized_circles)
            
            # Evaluate solution
            current_sum = np.sum(refined_circles[:, 2])
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution = refined_circles.copy()
                
        except Exception as e:
            # If any error occurs, continue with other initializations
            continue
    
    # If no good solution found, use default approach with extra refinement
    if best_solution is None:
        circles = _initialize_hexagonal_pattern()
        optimized_circles = _optimize_circles(circles)
        best_solution = _local_search_refinement(optimized_circles)
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Validate final solution
    if not _check_constraints(best_solution):
        print("Warning: Final configuration may have constraint violations")
    
    return best_solution


# EVOLVE-BLOCK-END
