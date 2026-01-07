# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import math
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize sum of radii.
    Uses an enhanced hybrid approach with improved initialization, smarter optimization,
    and better convergence strategies to beat the benchmark.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    def validate_circles(circles):
        """Validate that all circles are within bounds and non-overlapping."""
        # Check containment constraints
        for i in range(n):
            x, y, r = circles[i]
            if x < r or x > 1 - r or y < r or y > 1 - r:
                return False
        
        # Check overlap constraints
        for i, j in combinations(range(n), 2):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False
        
        return True
    
    def calculate_radius_sum(circles):
        """Calculate the sum of all radii."""
        return np.sum(circles[:, 2])
    
    def generate_initial_placement():
        """Generate high-quality initial configuration using optimized hexagonal approach."""
        # Create a better initial configuration with improved spacing
        circles = []
        
        # Use a refined hexagonal pattern with more strategic positioning
        rows, cols = 6, 6
        spacing = 1.0 / cols
        
        # Create hexagonal pattern with tighter spacing for better coverage
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Hexagonal offset for odd rows
                x_offset = spacing * 0.5 if i % 2 == 1 else 0
                x = (j + 0.5 + x_offset) * spacing
                y = (i + 0.5) * spacing
                
                # Apply boundary constraints with tighter margins
                x = max(0.08, min(0.92, x))
                y = max(0.08, min(0.92, y))
                
                # Set initial radius with more conservative values for better convergence
                circles.append([x, y, min(0.1, spacing * 0.35)])
        
        # Fill remaining positions with carefully placed points
        while len(circles) < n:
            # Use a more sophisticated approach for remaining positions
            if len(circles) < n - 8:  # Most positions use systematic pattern
                x = random.uniform(0.1, 0.9)
                y = random.uniform(0.1, 0.9)
                r = random.uniform(0.025, 0.07)
            else:  # Strategic placement for last few circles
                # Place near edges and corners for better utilization
                edge_positions = [
                    (0.1, random.uniform(0.1, 0.9)), 
                    (0.9, random.uniform(0.1, 0.9)),
                    (random.uniform(0.1, 0.9), 0.1),
                    (random.uniform(0.1, 0.9), 0.9)
                ]
                pos = random.choice(edge_positions)
                x = pos[0] + random.uniform(-0.05, 0.05)
                y = pos[1] + random.uniform(-0.05, 0.05)
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                r = random.uniform(0.02, 0.06)
            circles.append([x, y, r])
        
        return np.array(circles[:n])
    
    def maximize_radii_precise(circles):
        """Precise radius maximization with improved numerical stability."""
        improved = True
        iterations = 0
        
        while improved and iterations < 200:
            improved = False
            iterations += 1
            
            updated_count = 0
            
            # Try to increase each radius to its maximum possible value
            for i in range(n):
                x1, y1, r1 = circles[i]
                
                # Find maximum possible radius
                max_radius = min(x1, 1-x1, y1, 1-y1)
                
                # Check overlap with all other circles more precisely
                for j in range(n):
                    if i != j:
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        # Use a small safety margin to avoid numerical issues
                        max_radius = min(max_radius, distance - r2 - 1e-8)
                
                # More conservative but precise update to avoid numerical issues
                if max_radius > r1 + 1e-8:
                    # Very conservative increase to ensure stability
                    increment = min((max_radius - r1) * 0.8, 0.01)
                    new_radius = min(max_radius, r1 + increment)
                    circles[i, 2] = new_radius
                    improved = True
                    updated_count += 1
            
            # Early stopping if no meaningful changes
            if updated_count == 0:
                break
        
        return circles
    
    def smart_local_search(circles):
        """Smart local search with more intelligent neighborhood exploration."""
        improved = False
        
        # Try to improve each circle individually with adaptive search
        for i in range(n):
            original_x, original_y, original_r = circles[i]
            
            # Try to find a better position for this circle
            best_x, best_y, best_r = original_x, original_y, original_r
            best_sum = calculate_radius_sum(circles)
            
            # Adaptive search strategy based on current configuration
            if original_r < 0.05:  # Small circles: more aggressive moves
                movements = [
                    (-0.01, -0.01), (-0.01, 0), (-0.01, 0.01),
                    (0, -0.01), (0, 0), (0, 0.01),
                    (0.01, -0.01), (0.01, 0), (0.01, 0.01),
                    (-0.02, -0.02), (-0.02, 0.02), (0.02, -0.02), (0.02, 0.02)
                ]
            else:  # Larger circles: more conservative moves
                movements = [
                    (-0.005, -0.005), (-0.005, 0), (-0.005, 0.005),
                    (0, -0.005), (0, 0), (0, 0.005),
                    (0.005, -0.005), (0.005, 0), (0.005, 0.005)
                ]
            
            # Test all movements
            for dx, dy in movements:
                test_circles = circles.copy()
                test_circles[i, 0] = max(original_r, min(1-original_r, original_x + dx))
                test_circles[i, 1] = max(original_r, min(1-original_r, original_y + dy))
                
                if validate_circles(test_circles):
                    test_sum = calculate_radius_sum(test_circles)
                    if test_sum > best_sum:
                        best_sum = test_sum
                        best_x, best_y = test_circles[i, 0], test_circles[i, 1]
            
            # Apply best improvement if found
            if best_x != original_x or best_y != original_y:
                circles[i, 0] = best_x
                circles[i, 1] = best_y
                improved = True
        
        return improved
    
    def perturb_configuration(circles, temperature):
        """Improved perturbation with adaptive step sizes and better selection."""
        new_circles = circles.copy()
        
        # Choose a random circle to perturb
        idx = random.randint(0, n - 1)
        
        # Choose perturbation type with higher probability of position changes
        if random.random() < 0.92:  # 92% chance of position perturbation
            # Perturb position with adaptive step size based on temperature
            x, y, r = new_circles[idx]
            step_size = temperature * 0.2
            
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            
            # Ensure new position stays within bounds
            new_x = max(r, min(1-r, x + dx))
            new_y = max(r, min(1-r, y + dy))
            
            new_circles[idx, 0] = new_x
            new_circles[idx, 1] = new_y
        else:  # 8% chance of radius perturbation
            # Perturb radius with adaptive step
            x, y, r = new_circles[idx]
            step_size = temperature * 0.1
            
            dr = random.uniform(-step_size, step_size)
            
            # Ensure radius stays positive and within reasonable bounds
            new_r = max(0.005, min(0.48, r + dr))
            
            new_circles[idx, 2] = new_r
        
        return new_circles
    
    # Main algorithm
    random.seed(42)  # For reproducibility
    
    # Phase 1: Generate high-quality initial configuration
    circles = generate_initial_placement()
    
    # Phase 2: Precise radius maximization
    circles = maximize_radii_precise(circles)
    
    # Phase 3: Multiple rounds of smart local search
    for round_num in range(25):
        improved = True
        iteration = 0
        while improved and iteration < 40:
            improved = smart_local_search(circles)
            iteration += 1
    
    # Phase 4: Enhanced Simulated Annealing for global optimization
    best_circles = circles.copy()
    best_sum = calculate_radius_sum(best_circles)
    
    # Optimized annealing parameters for better convergence
    temperature = 0.2
    cooling_rate = 0.99995
    min_temperature = 1e-10
    max_iterations = 15000
    
    for iteration in range(max_iterations):
        # Create perturbed configuration
        new_circles = perturb_configuration(circles, temperature)
        
        # Validate new configuration
        if validate_circles(new_circles):
            new_sum = calculate_radius_sum(new_circles)
            
            # Accept or reject based on simulated annealing criteria
            delta = new_sum - best_sum
            if delta > 0 or random.random() < math.exp(delta / temperature):
                circles = new_circles
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_circles = new_circles.copy()
        
        # Cool down temperature
        temperature = max(min_temperature, temperature * cooling_rate)
        
        # More aggressive restarts to escape local optima
        if iteration % 3000 == 0 and iteration > 0:
            circles = best_circles.copy()
            temperature = max(0.01, temperature * 0.93)
    
    # Phase 5: Final precise refinement
    best_circles = maximize_radii_precise(best_circles)
    
    # Additional smart local search for final polishing
    for _ in range(20):
        smart_local_search(best_circles)
    
    return best_circles


# EVOLVE-BLOCK-END
