# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.spatial import cKDTree
import random
from scipy.optimize import differential_evolution
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies, 
    local optimization with spatial indexing, and global optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    best_result = None
    best_sum = 0
    
    # Strategy 1: Hexagonal packing pattern (most promising)
    def hexagonal_initialization():
        circles = np.zeros((n, 3))
        
        # Arrange in a precise hexagonal pattern
        rows = 6
        cols = 6
        
        # Calculate spacing for hexagonal packing
        spacing_x = 0.8 / (cols - 1) if cols > 1 else 0.5
        spacing_y = spacing_x * math.sqrt(3) / 2
        
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Offset odd rows for hexagonal packing
                x = 0.1 + (j + 0.5 + (i % 2) * 0.5) * spacing_x
                y = 0.1 + i * spacing_y
                
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                circles[count] = [x, y, 0.05]
                count += 1
        
        # Fill remaining positions with random points if needed
        while count < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles[count] = [x, y, 0.05]
            count += 1
        
        return circles
    
    # Strategy 2: Grid-based initialization with better distribution
    def grid_initialization():
        circles = np.zeros((n, 3))
        
        # Create a 6x6 grid with some randomness
        rows, cols = 6, 6
        for i in range(n):
            row = i // cols
            col = i % cols
            
            # Grid position with more substantial randomness
            x = 0.05 + (col + 0.5 + np.random.uniform(-0.2, 0.2)) * 0.9 / cols
            y = 0.05 + (row + 0.5 + np.random.uniform(-0.2, 0.2)) * 0.9 / rows
            
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            circles[i] = [x, y, 0.05]
        
        return circles
    
    # Strategy 3: Spiral-based initialization for even better coverage
    def spiral_initialization():
        circles = np.zeros((n, 3))
        
        # Create a spiral pattern to distribute points well
        angle_step = 0.6
        radius_step = 0.1
        max_radius = 0.4
        
        for i in range(n):
            if i == 0:
                x, y = 0.5, 0.5
            else:
                angle = i * angle_step
                radius = min(i * radius_step, max_radius)
                x = 0.5 + radius * math.cos(angle)
                y = 0.5 + radius * math.sin(angle)
                
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
            
            circles[i] = [x, y, 0.05]
        
        return circles
    
    # Strategy 4: Random initialization with triangular distribution
    def random_initialization():
        circles = np.zeros((n, 3))
        for i in range(n):
            # Use triangular distribution for better point distribution
            x = np.random.triangular(0.05, 0.5, 0.95)
            y = np.random.triangular(0.05, 0.5, 0.95)
            circles[i] = [x, y, 0.05]
        return circles
    
    # Strategy 5: Voronoi-based initialization for better spatial distribution
    def voronoi_initialization():
        circles = np.zeros((n, 3))
        # Generate random points and then use Voronoi-like approach
        np.random.seed(42)  # Fixed seed for reproducibility
        points = np.random.rand(n, 2)
        for i in range(n):
            x, y = points[i]
            # Clamp to bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            circles[i] = [x, y, 0.05]
        return circles
    
    # Try multiple initialization strategies
    strategies = [
        hexagonal_initialization,
        grid_initialization,
        spiral_initialization,
        random_initialization,
        voronoi_initialization
    ]
    
    # Run optimization with different starting points
    for strategy in strategies:
        try:
            circles = strategy()
            
            # Local optimization step with improved convergence criteria
            max_improvements = 1500  # More iterations for better results
            
            for iteration in range(max_improvements):
                improved = False
                
                # Process circles in random order for better convergence
                indices = list(range(n))
                random.shuffle(indices)
                
                # Build spatial index once per iteration for efficiency
                coords = circles[:, :2]
                tree = cKDTree(coords)
                
                for i in indices:
                    x, y, old_radius = circles[i]
                    
                    # Compute maximum possible radius for this circle
                    max_radius = min(x, 1-x, y, 1-y)
                    
                    # Find neighbors efficiently using spatial indexing
                    neighbors = tree.query_ball_point([x, y], 2.0)
                    
                    # Check overlap constraints with neighbors only
                    for j in neighbors:
                        if i != j:
                            x2, y2, r2 = circles[j]
                            dx = x - x2
                            dy = y - y2
                            distance = math.sqrt(dx*dx + dy*dy)
                            
                            # Must be at least radius_i + radius_j apart
                            if distance > 0:
                                max_radius = min(max_radius, distance - r2)
                    
                    # Update to new radius (clamped to reasonable bounds)
                    new_radius = max(0.001, min(max_radius, 0.45))
                    
                    if abs(new_radius - old_radius) > 1e-6:
                        circles[i, 2] = new_radius
                        improved = True
                
                # Early stopping if no improvement
                if not improved:
                    # Stop if we haven't improved for several iterations
                    if iteration > 50:
                        break
                        
            # Final optimization pass with enhanced local search
            for _ in range(300):  # More iterations for final refinement
                improved = False
                
                # Build spatial index for this round
                coords = circles[:, :2]
                tree = cKDTree(coords)
                
                # Try to improve each circle with more extensive search
                for i in range(n):
                    best_pos = circles[i][:2].copy()
                    best_radius = circles[i][2]
                    
                    # Try more positions around current location with adaptive step size
                    step_sizes = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]
                    for dx in step_sizes:
                        for dy in step_sizes:
                            test_x = max(0.05, min(0.95, circles[i][0] + dx))
                            test_y = max(0.05, min(0.95, circles[i][1] + dy))
                            
                            # Calculate maximum possible radius at this position
                            max_radius = min(test_x, test_y, 1-test_x, 1-test_y)
                            
                            # Check overlap constraints with neighbors efficiently
                            neighbors = tree.query_ball_point([test_x, test_y], 2.0)
                            for j in neighbors:
                                if i != j:
                                    x2, y2, r2 = circles[j]
                                    dist = math.sqrt((test_x-x2)**2 + (test_y-y2)**2)
                                    if dist > 0:
                                        max_rad_for_overlap = dist - r2
                                        max_radius = min(max_radius, max_rad_for_overlap)
                            
                            # Also consider increasing radius if position allows
                            max_radius = max(0.001, min(max_radius, 0.45))
                            
                            if max_radius > best_radius:
                                best_radius = max_radius
                                best_pos = [test_x, test_y]
                    
                    # Update if we found a better configuration
                    if best_radius > circles[i][2] or \
                       abs(best_pos[0] - circles[i][0]) > 1e-4 or \
                       abs(best_pos[1] - circles[i][1]) > 1e-4:
                        circles[i] = [best_pos[0], best_pos[1], best_radius]
                        improved = True
                
                if not improved:
                    break
            
            # Final validation
            total_radius = np.sum(circles[:, 2])
            if total_radius > best_sum:
                best_sum = total_radius
                best_result = circles.copy()
                
        except Exception as e:
            continue
    
    # If no good result found, fallback to hexagonal pattern with extra optimization
    if best_result is None:
        best_result = hexagonal_initialization()
        
        # Run more intensive optimization on fallback
        for iteration in range(1500):
            improved = False
            indices = list(range(n))
            random.shuffle(indices)
            
            coords = best_result[:, :2]
            tree = cKDTree(coords)
            
            for i in indices:
                x, y, old_radius = best_result[i]
                
                max_radius = min(x, 1-x, y, 1-y)
                
                neighbors = tree.query_ball_point([x, y], 2.0)
                for j in neighbors:
                    if i != j:
                        x2, y2, r2 = best_result[j]
                        dx = x - x2
                        dy = y - y2
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance > 0:
                            max_radius = min(max_radius, distance - r2)
                
                new_radius = max(0.001, min(max_radius, 0.45))
                
                if abs(new_radius - old_radius) > 1e-6:
                    best_result[i, 2] = new_radius
                    improved = True
            
            if not improved:
                if iteration > 50:
                    break
    
    # Final boundary correction and cleanup
    for i in range(n):
        x, y, r = best_result[i]
        # Ensure containment
        r = min(r, x, 1-x, y, 1-y)
        # Ensure reasonable minimum
        r = max(0.001, r)
        best_result[i] = [x, y, r]
    
    # Apply additional global optimization if we have time
    # Use differential evolution for final refinement if we're close to the benchmark
    if best_sum > 2.8:  # Only apply if we're already quite good
        try:
            # Flatten the array for optimization
            def objective_func(params):
                circles = params.reshape(-1, 3)
                # Check constraints
                if not _validate_placement(circles):
                    return 1000000  # Large penalty for invalid solutions
                return -np.sum(circles[:, 2])  # Negative because we minimize
            
            def _validate_placement(circles):
                """Check if all circles are valid (contained and non-overlapping)"""
                n = len(circles)
                
                # Check containment constraints
                for i in range(n):
                    x, y, r = circles[i]
                    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                        return False
                
                # Check overlap constraints efficiently using distance matrix
                if n > 1:
                    positions = circles[:, :2]
                    radii = circles[:, 2]
                    dist_matrix = cKDTree(positions).query_pairs(0.001)  # Very fast check
                    
                    # More thorough check for actual overlaps
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = positions[i][0] - positions[j][0]
                            dy = positions[i][1] - positions[j][1]
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance < radii[i] + radii[j]:
                                return False
                
                return True
            
            # Prepare bounds for differential evolution
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r
            
            # Run differential evolution with reduced iterations for time limits
            result = differential_evolution(
                objective_func,
                bounds,
                maxiter=50,
                popsize=15,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            
            if result.success:
                refined_solution = result.x.reshape(-1, 3)
                if _validate_placement(refined_solution):
                    refined_sum = np.sum(refined_solution[:, 2])
                    if refined_sum > best_sum:
                        best_result = refined_solution
                        best_sum = refined_sum
                        
        except Exception:
            pass
    
    return best_result


# EVOLVE-BLOCK-END
