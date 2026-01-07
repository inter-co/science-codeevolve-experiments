# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import math
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining mathematical programming, spatial indexing, and 
    multiple initialization strategies for optimal results.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Multi-start approach with different initialization strategies
    best_result = None
    best_sum = 0
    
    # Strategy 1: Hexagonal packing pattern (inspired by INSPIRATION 2)
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
    
    # Strategy 2: Grid-based initialization with better distribution (inspired by INSPIRATION 2)
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
    
    # Strategy 3: Spiral-based initialization (inspired by INSPIRATION 2)
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
    
    # Strategy 4: Random initialization with triangular distribution (inspired by INSPIRATION 2)
    def random_initialization():
        circles = np.zeros((n, 3))
        for i in range(n):
            # Use triangular distribution for better point distribution
            x = np.random.triangular(0.05, 0.5, 0.95)
            y = np.random.triangular(0.05, 0.5, 0.95)
            circles[i] = [x, y, 0.05]
        return circles
    
    # Strategy 5: Optimized version using mathematical programming approach (inspired by INSPIRATION 1)
    def optimize_with_mathematical_programming(initial_circles):
        # Convert circles to flattened parameters for optimization
        def objective(params):
            circles = params.reshape(-1, 3)
            return -np.sum(circles[:, 2])  # Negative because we minimize
        
        def contain_constraint(params):
            circles = params.reshape(-1, 3)
            cons = []
            for i in range(len(circles)):
                x, y, r = circles[i]
                # r <= x <= 1-r
                cons.append(x - r)      # Should be >= 0
                cons.append(1 - x - r)  # Should be >= 0
                # r <= y <= 1-r  
                cons.append(y - r)      # Should be >= 0
                cons.append(1 - y - r)  # Should be >= 0
            return np.array(cons)
        
        def overlap_constraint(params):
            circles = params.reshape(-1, 3)
            cons = []
            for i in range(len(circles)):
                for j in range(i+1, len(circles)):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    # dist >= r1 + r2 (non-overlapping)
                    cons.append(dist - r1 - r2)  # Should be >= 0
            return np.array(cons)
        
        # Try optimization with SLSQP method
        initial_params = initial_circles.flatten()
        bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
        
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: contain_constraint(p)},
                    {'type': 'ineq', 'fun': lambda p: overlap_constraint(p)}
                ],
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                final_circles = result.x.reshape(-1, 3)
                return final_circles
        except Exception:
            pass
        
        return initial_circles
    
    # Try multiple initialization strategies
    initial_strategies = [
        hexagonal_initialization,
        grid_initialization,
        spiral_initialization,
        random_initialization
    ]
    
    # Run optimization with different starting points
    for strategy in initial_strategies:
        try:
            circles = strategy()
            
            # First apply iterative improvement with spatial indexing for efficiency
            for iteration in range(300):
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
            
            # Then apply mathematical programming optimization for final refinement
            circles = optimize_with_mathematical_programming(circles)
            
            # Final validation
            total_radius = np.sum(circles[:, 2])
            if total_radius > best_sum:
                best_sum = total_radius
                best_result = circles.copy()
                
        except Exception as e:
            continue
    
    # If no good result found, fallback to optimized hexagonal pattern with mathematical optimization
    if best_result is None:
        best_result = hexagonal_initialization()
        best_result = optimize_with_mathematical_programming(best_result)
    
    # Final boundary correction and cleanup
    for i in range(n):
        x, y, r = best_result[i]
        # Ensure containment
        r = min(r, x, 1-x, y, 1-y)
        # Ensure reasonable minimum
        r = max(0.001, r)
        best_result[i] = [x, y, r]
    
    return best_result


# EVOLVE-BLOCK-END
