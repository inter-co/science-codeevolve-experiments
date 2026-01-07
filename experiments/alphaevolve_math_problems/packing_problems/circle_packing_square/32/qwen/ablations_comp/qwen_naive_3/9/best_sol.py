# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Generate initial configuration using improved packing strategy
    def generate_initial_config():
        # Use a more sophisticated approach based on known good packings
        # Start with a hexagonal close packing pattern, then refine
        
        # Create a better initial configuration using a combination of structured placement
        # and strategic randomness to avoid poor local minima
        
        circles = []
        
        # Strategy: place in a grid with refined spacing and slight perturbations
        # For 32 circles, try a 5x7 grid pattern (or close to it)
        rows = 5
        cols = 7
        
        # Calculate spacing
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Place circles in a grid pattern with small random perturbations
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= n:
                    break
                # Add moderate random perturbation to improve distribution
                x = (j + 0.5 + np.random.uniform(-0.2, 0.2)) * spacing_x
                y = (i + 0.5 + np.random.uniform(-0.2, 0.2)) * spacing_y
                
                # Ensure we're still within bounds with buffer
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                circles.append([x, y, 0.0])
                count += 1
            if count >= n:
                break
        
        # Adjust to exactly 32 circles if needed
        while len(circles) < n:
            # Add extra circles in strategic locations
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.0])
        
        # Set initial radii to be more realistic - use a greedy approach
        # Start with small radii and increase based on available space
        for i in range(len(circles)):
            # Initial estimate based on proximity to boundaries and other circles
            x, y = circles[i][0], circles[i][1]
            min_dist_to_boundary = min(x, 1-x, y, 1-y)
            
            # Estimate how large a radius could be
            min_dist_to_others = float('inf')
            
            # Find minimum distance to other circles
            for j in range(len(circles)):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist_to_others = min(min_dist_to_others, dist)
            
            # Conservative estimate for radius
            if min_dist_to_others < float('inf'):
                # Allow for half the minimum distance to other circles minus some safety margin
                estimated_radius = min(0.1, min_dist_to_boundary, min_dist_to_others/2.0 - 0.01)
            else:
                estimated_radius = min(0.1, min_dist_to_boundary)
            
            # Make sure it's positive and reasonable
            circles[i][2] = max(0.001, min(estimated_radius, 0.2))
        
        return np.array(circles)
    
    # Phase 2: Enhanced optimization with multiple restarts and better constraints
    def objective(radii_and_positions):
        # Extract positions and radii from flattened array
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Return negative sum of radii (since we want to maximize)
        return -np.sum(radii)
    
    def constraint_containment(radii_and_positions):
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Check containment constraints - all should be >= 0
        result = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # r <= x <= 1-r and r <= y <= 1-r
            # This means: x >= r, 1-x >= r, y >= r, 1-y >= r
            # Which means: x >= r, x <= 1-r, y >= r, y <= 1-r
            result.extend([
                x - r,           # x - r >= 0
                1 - x - r,       # 1 - x - r >= 0
                y - r,           # y - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        return np.array(result)
    
    def constraint_nonoverlap(radii_and_positions):
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Check non-overlap constraints
        result = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1 = radii[i]
                r2 = radii[j]
                
                # Distance between centers >= sum of radii
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # We want: dist_sq >= min_dist_sq
                # So: dist_sq - min_dist_sq >= 0
                result.append(dist_sq - min_dist_sq)
        return np.array(result)
    
    # Improved constraint handling with better numerical stability
    def improved_constraint_nonoverlap(radii_and_positions):
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Check non-overlap constraints with numerical tolerance
        result = []
        eps = 1e-10  # Small epsilon for numerical stability
        
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1 = radii[i]
                r2 = radii[j]
                
                # Distance between centers >= sum of radii
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                # Add a small safety margin to ensure strict inequality
                # Also add numerical tolerance to avoid constraint violations due to floating point
                result.append(dist_sq - min_dist_sq - eps)
        return np.array(result)
    
    # Better optimization with multiple restarts
    best_result = None
    best_sum = -float('inf')
    
    # Try multiple random restarts to find better solutions
    num_restarts = 5
    
    for restart in range(num_restarts):
        # Generate initial configuration
        initial_circles = generate_initial_config()
        
        # Flatten for optimization
        initial_flat = np.concatenate([
            initial_circles[:, :2].flatten(),  # positions
            initial_circles[:, 2]              # radii
        ])
        
        # Create bounds for variables (positions and radii)
        bounds = []
        # Position bounds: [0,1] for both x and y
        for _ in range(2*n):
            bounds.extend([(0, 1)])
        # Radius bounds: [0.001, 0.4] (more realistic bounds)
        for _ in range(n):
            bounds.extend([(0.001, 0.4)])
        
        # Create constraints with improved handling
        cons = [
            {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
            {'type': 'ineq', 'fun': lambda x: improved_constraint_nonoverlap(x)}
        ]
        
        # Optimize with multiple methods for better results
        try:
            # Try SLSQP first (good for constrained problems)
            result_slsqp = minimize(
                objective,
                initial_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result_slsqp.success:
                final_positions = result_slsqp.x[:2*n].reshape(-1, 2)
                final_radii = result_slsqp.x[2*n:]
                sum_radii = np.sum(final_radii)
                
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = (final_positions, final_radii)
                    
        except Exception as e:
            continue
    
    # If we found a better solution, return it; otherwise fallback to initial
    if best_result is not None:
        final_positions, final_radii = best_result
        circles = np.column_stack([final_positions, final_radii])
        return circles
    
    # Fallback to initial configuration if optimization fails
    initial_circles = generate_initial_config()
    return initial_circles


# EVOLVE-BLOCK-END
