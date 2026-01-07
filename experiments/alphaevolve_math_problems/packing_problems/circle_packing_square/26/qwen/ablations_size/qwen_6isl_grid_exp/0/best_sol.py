# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with constrained optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Enhanced hexagonal initialization with mathematical precision
    def initialize_hexagonal():
        circles = []
        
        # Use a 5x6 grid (30 positions) with appropriate spacing for 26 circles
        rows = 5
        cols = 6
        
        # For perfect hexagonal packing with circles of equal radius r:
        # Horizontal spacing = 2r, Vertical spacing = sqrt(3)r
        # We want to fit within [0,1]x[0,1] so we calculate the optimal radius
        
        # Estimate initial radius based on optimal hexagonal packing density
        # For 30 circles in 1x1 square, we can estimate a reasonable radius
        estimated_radius = 0.14  # Slightly smaller to allow room for optimization
        
        # Generate hexagonal pattern with precise positioning
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                    
                # Offset every other row for hexagonal packing
                x_offset = j + (i % 2) * 0.5
                y_offset = i * math.sqrt(3) / 2
                
                # Scale to fit in [0.1, 0.9]x[0.1, 0.9] to leave margins
                max_x = cols + 0.5
                max_y = rows * math.sqrt(3) / 2
                
                x = 0.1 + x_offset / max_x * 0.8
                y = 0.1 + y_offset / max_y * 0.8
                
                # Ensure within bounds
                x = max(estimated_radius, min(1 - estimated_radius, x))
                y = max(estimated_radius, min(1 - estimated_radius, y))
                
                circles.append([x, y, estimated_radius])
        
        # Fill remaining circles strategically with better distribution
        while len(circles) < n:
            # Add additional circles near edges and corners for better coverage
            additional_positions = [
                [0.1, 0.5, estimated_radius], [0.9, 0.5, estimated_radius],
                [0.5, 0.1, estimated_radius], [0.5, 0.9, estimated_radius],
                [0.2, 0.2, estimated_radius], [0.8, 0.2, estimated_radius],
                [0.2, 0.8, estimated_radius], [0.8, 0.8, estimated_radius],
                [0.15, 0.15, estimated_radius], [0.85, 0.15, estimated_radius],
                [0.15, 0.85, estimated_radius], [0.85, 0.85, estimated_radius],
                [0.3, 0.3, estimated_radius], [0.7, 0.3, estimated_radius],
                [0.3, 0.7, estimated_radius], [0.7, 0.7, estimated_radius]
            ]
            
            for x, y, r in additional_positions:
                if len(circles) >= n:
                    break
                x = max(estimated_radius, min(1 - estimated_radius, x))
                y = max(estimated_radius, min(1 - estimated_radius, y))
                circles.append([x, y, r])
        
        # Trim to exactly n circles
        circles = circles[:n]
        return np.array(circles)
    
    # More robust constraint checking with numerical tolerance
    def check_constraints(circles):
        """Fast constraint validation with numerical tolerance"""
        tolerance = 1e-10
        # Check containment with small tolerance
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -tolerance or x + r > 1 + tolerance or y - r < -tolerance or y + r > 1 + tolerance:
                return False
        
        # Check non-overlap using vectorized computation with tolerance
        if len(circles) > 1:
            coords = circles[:, :2]  # x, y coordinates
            radii = circles[:, 2]    # radii
            
            # Compute pairwise distances
            distances = cdist(coords, coords)
            
            # Create mask for upper triangle (avoid duplicate pairs)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            
            # Check non-overlap condition: distance >= sum of radii (with tolerance)
            distance_matrix = distances[mask]
            radius_sums = (radii[:, None] + radii[None, :])[mask]
            
            # With tolerance, allow very small overlaps (numerical precision)
            if np.any(distance_matrix < radius_sums - 1e-10):
                return False
                
        return True
    
    # Objective function to maximize sum of radii
    def objective(params):
        # Return negative because we want to maximize
        return -sum(params[2::3])  # Sum of all radii (every third element starting from index 2)
    
    # Constraint functions for scipy optimization
    def constraint_containment(params):
        circles = params.reshape(-1, 3)
        constraints = []
        for i in range(n):
            x, y, r = circles[i]
            # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
            constraints.extend([
                x - r,           # x - r >= 0
                1 - x - r,       # 1 - x - r >= 0  
                y - r,           # y - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        return np.array(constraints)
    
    def constraint_nonoverlap(params):
        circles = params.reshape(-1, 3)
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # Distance - (r1 + r2) >= 0 (non-overlap constraint)
                constraints.append(dist - (r1 + r2))
        return np.array(constraints)
    
    # Multi-start optimization approach with enhanced diversity
    best_sum = -np.inf
    best_circles = None
    
    # Strategy 1: Multiple initializations with varying approaches for diversity
    num_starts = 25  # Increase diversity further
    
    for start_iter in range(num_starts):
        # Initialize with better hexagonal layout
        circles = initialize_hexagonal()
        
        # Vary the initialization approach for each start
        if start_iter == 0:
            # First start: pure hexagonal
            pass
        elif start_iter <= 5:
            # Next 5 starts: perturbed hexagonal with moderate noise
            np.random.seed(start_iter * 42)
            for i in range(len(circles)):
                # Add moderate random perturbations to positions
                circles[i][0] += np.random.normal(0, 0.015)  # Slightly less noise
                circles[i][1] += np.random.normal(0, 0.015)
                # Keep within bounds
                circles[i][0] = np.clip(circles[i][0], 0.01, 0.99)
                circles[i][1] = np.clip(circles[i][1], 0.01, 0.99)
        elif start_iter <= 10:
            # Next 5 starts: random positions with hexagonal-like clustering
            np.random.seed(start_iter * 42)
            # Create clusters in different areas of the square
            cluster_centers = [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75], [0.5, 0.5]]
            cluster_idx = 0
            for i in range(len(circles)):
                if i < n // 2:
                    center = cluster_centers[cluster_idx % len(cluster_centers)]
                    circles[i][0] = np.clip(center[0] + np.random.normal(0, 0.12), 0.01, 0.99)
                    circles[i][1] = np.clip(center[1] + np.random.normal(0, 0.12), 0.01, 0.99)
                else:
                    circles[i][0] = np.random.uniform(0.05, 0.95)
                    circles[i][1] = np.random.uniform(0.05, 0.95)
                cluster_idx += 1
        elif start_iter <= 15:
            # Next 5 starts: random with better clustering around center
            np.random.seed(start_iter * 42)
            for i in range(len(circles)):
                if i < n // 3:  # First third: clustered around center
                    circles[i][0] = np.clip(0.5 + np.random.normal(0, 0.15), 0.01, 0.99)
                    circles[i][1] = np.clip(0.5 + np.random.normal(0, 0.15), 0.01, 0.99)
                else:  # Remaining: uniformly distributed
                    circles[i][0] = np.random.uniform(0.05, 0.95)
                    circles[i][1] = np.random.uniform(0.05, 0.95)
        else:
            # Last 10 starts: completely random with better bounds and more diversity
            np.random.seed(start_iter * 42)
            for i in range(len(circles)):
                circles[i][0] = np.random.uniform(0.03, 0.97)  # Slightly tighter bounds
                circles[i][1] = np.random.uniform(0.03, 0.97)
                circles[i][0] = np.clip(circles[i][0], 0.01, 0.99)
                circles[i][1] = np.clip(circles[i][1], 0.01, 0.99)
        
        # Flatten for optimization
        initial_params = circles.flatten()
        
        # Set up bounds for parameters (x,y in [0,1], r in [0.001, 0.499])
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Constraints for optimization
        constraints = [
            {'type': 'ineq', 'fun': constraint_containment},
            {'type': 'ineq', 'fun': constraint_nonoverlap}
        ]
        
        # Run optimization with different methods for robustness
        try:
            # Try SLSQP first with more iterations for better convergence
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6, 'disp': False}
            )
            
            if result.success:
                optimized_params = result.x
                optimized_circles = optimized_params.reshape(-1, 3)
                
                # Validate the solution
                if check_constraints(optimized_circles):
                    sum_radii = sum(optimized_circles[:, 2])
                    if sum_radii > best_sum:
                        best_sum = sum_radii
                        best_circles = optimized_circles.copy()
                        
        except Exception as e:
            # Continue with next iteration if optimization fails
            continue
    
    # If no successful optimization found, return the best initialization
    if best_circles is None:
        circles = initialize_hexagonal()
        # Final validation and cleanup
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Ensure containment constraint
            r = min(r, x, 1-x, y, 1-y)
            circles[i] = [x, y, r]
        return circles
    
    # Final validation of best result
    for i in range(len(best_circles)):
        x, y, r = best_circles[i]
        # Ensure containment constraint
        r = min(r, x, 1-x, y, 1-y)
        best_circles[i] = [x, y, r]
        
    return best_circles


# EVOLVE-BLOCK-END
