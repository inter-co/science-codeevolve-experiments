# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
import random
from collections import defaultdict
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining simulated annealing with geometric initialization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    
    def compute_total_radius(circles):
        """Compute sum of all radii"""
        return np.sum(circles[:, 2])
    
    def compute_pairwise_distances(circles):
        """Compute pairwise distances efficiently"""
        positions = circles[:, :2]
        return cdist(positions, positions)
    
    def check_validity(circles):
        """Check if configuration is valid (no overlaps, fully contained)"""
        # Check containment
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlaps - only need to check upper triangle of distance matrix
        distances = compute_pairwise_distances(circles)
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dist = distances[i, j]
                r_i = circles[i, 2]
                r_j = circles[j, 2]
                if dist < r_i + r_j:
                    return False
        return True
    
    def compute_violation_count(circles):
        """Count number of constraint violations"""
        violation_count = 0
        # Check containment violations
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                violation_count += 1
        
        # Check overlap violations
        distances = compute_pairwise_distances(circles)
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dist = distances[i, j]
                r_i = circles[i, 2]
                r_j = circles[j, 2]
                if dist < r_i + r_j:
                    violation_count += 1
        return violation_count
    
    def generate_initial_placement():
        """Generate better initial configuration using a more sophisticated approach"""
        # Start with a dense packing approach
        circles = []
        
        # Try to fill the space with a pattern that's closer to optimal
        # Using a grid-like approach with some randomness
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        base_radius = spacing * 0.4  # Slightly smaller than spacing to allow for some optimization
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing
                y = (i + 1) * spacing
                # Add slight randomness to positions
                x += random.uniform(-spacing*0.1, spacing*0.1)
                y += random.uniform(-spacing*0.1, spacing*0.1)
                # Ensure within bounds
                x = max(base_radius, min(1-base_radius, x))
                y = max(base_radius, min(1-base_radius, y))
                circles.append([x, y, base_radius])
        
        # Fill remaining slots with random placements
        while len(circles) < n:
            r = random.uniform(0.02, 0.08)
            x = random.uniform(r, 1-r)
            y = random.uniform(r, 1-r)
            circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    def simulate_annealing(circles, max_time=55):
        """Use simulated annealing to optimize the configuration"""
        start_time = time.time()
        current_solution = circles.copy()
        current_radius_sum = compute_total_radius(current_solution)
        best_solution = current_solution.copy()
        best_radius_sum = current_radius_sum
        
        # Initial temperature and cooling schedule
        temp = 0.1
        min_temp = 1e-6
        cooling_rate = 0.9995
        
        # Iteration counters
        iterations = 0
        accepted_moves = 0
        max_iterations = 100000
        
        while temp > min_temp and iterations < max_iterations and time.time() - start_time < max_time:
            # Generate neighbor solution
            new_solution = current_solution.copy()
            
            # Select random circle to modify
            idx = random.randint(0, len(new_solution) - 1)
            
            # Perturb position and radius
            old_x, old_y, old_r = new_solution[idx]
            new_x = old_x + random.uniform(-0.02, 0.02)
            new_y = old_y + random.uniform(-0.02, 0.02)
            new_r = old_r + random.uniform(-0.01, 0.01)
            
            # Ensure bounds
            new_x = max(new_r, min(1-new_r, new_x))
            new_y = max(new_r, min(1-new_r, new_y))
            new_r = max(0.001, min(0.5, new_r))
            
            new_solution[idx] = [new_x, new_y, new_r]
            
            # Check validity
            if check_validity(new_solution):
                new_radius_sum = compute_total_radius(new_solution)
                
                # Accept or reject based on Metropolis criterion
                delta = new_radius_sum - current_radius_sum
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current_solution = new_solution
                    current_radius_sum = new_radius_sum
                    accepted_moves += 1
                    
                    # Update best solution
                    if current_radius_sum > best_radius_sum:
                        best_solution = current_solution.copy()
                        best_radius_sum = current_radius_sum
            else:
                # If invalid, try a different approach - shrink radius slightly
                new_solution[idx][2] = max(0.001, new_solution[idx][2] * 0.95)
                if check_validity(new_solution):
                    new_radius_sum = compute_total_radius(new_solution)
                    delta = new_radius_sum - current_radius_sum
                    if delta > 0 or random.random() < math.exp(delta / temp):
                        current_solution = new_solution
                        current_radius_sum = new_radius_sum
                        accepted_moves += 1
                        
                        if current_radius_sum > best_radius_sum:
                            best_solution = current_solution.copy()
                            best_radius_sum = current_radius_sum
            
            # Cool down
            temp *= cooling_rate
            iterations += 1
            
            # Occasionally reduce temperature faster if no progress
            if iterations % 1000 == 0 and accepted_moves == 0:
                temp *= 0.9
            
            # Reset acceptance counter
            if iterations % 1000 == 0:
                accepted_moves = 0
        
        return best_solution
    
    def local_improvement(circles, max_iter=1000):
        """Apply local improvement to refine solution"""
        current = circles.copy()
        best = current.copy()
        best_sum = compute_total_radius(best)
        
        for iteration in range(max_iter):
            improved = False
            # Try to improve each circle individually
            for idx in range(len(current)):
                old_circle = current[idx].copy()
                old_radius = old_circle[2]
                
                # Try several perturbations
                best_candidate = old_circle.copy()
                best_candidate_sum = best_sum
                
                # Try different moves
                for _ in range(20):
                    # Small random move
                    new_x = old_circle[0] + random.uniform(-0.005, 0.005)
                    new_y = old_circle[1] + random.uniform(-0.005, 0.005)
                    new_r = old_radius + random.uniform(-0.002, 0.002)
                    
                    # Keep within bounds
                    new_x = max(new_r, min(1-new_r, new_x))
                    new_y = max(new_r, min(1-new_r, new_y))
                    new_r = max(0.001, min(0.5, new_r))
                    
                    # Test this candidate
                    test_solution = current.copy()
                    test_solution[idx] = [new_x, new_y, new_r]
                    
                    if check_validity(test_solution):
                        test_sum = compute_total_radius(test_solution)
                        if test_sum > best_candidate_sum:
                            best_candidate = [new_x, new_y, new_r]
                            best_candidate_sum = test_sum
                            improved = True
                
                # Update if we found a better candidate
                if improved:
                    current[idx] = best_candidate
                    best_sum = best_candidate_sum
                    best = current.copy()
            
            if not improved:
                break
                
        return best
    
    # Generate initial configuration
    initial_circles = generate_initial_placement()
    
    # Improve with simulated annealing
    sa_result = simulate_annealing(initial_circles, max_time=50)
    
    # Apply local refinement
    final_result = local_improvement(sa_result, max_iter=500)
    
    # Final validation and cleanup
    if not check_validity(final_result):
        # If still invalid, do a final clean-up
        cleaned = []
        for x, y, r in final_result:
            # Clamp to bounds
            r = min(r, x, y, 1-x, 1-y)
            if r <= 0:
                r = 0.01
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            cleaned.append([x, y, r])
        final_result = np.array(cleaned)
    
    # Final check
    if not check_validity(final_result):
        # Last resort: create a simple valid configuration
        simple_circles = []
        for i in range(n):
            r = 0.05
            x = random.uniform(r, 1-r)
            y = random.uniform(r, 1-r)
            simple_circles.append([x, y, r])
        final_result = np.array(simple_circles)
    
    return final_result


# EVOLVE-BLOCK-END
