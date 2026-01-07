# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining initial heuristic placement with optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    
    # Initial heuristic placement using hexagonal packing pattern
    def generate_initial_placement() -> np.ndarray:
        # Create a more sophisticated initial configuration
        circles = np.zeros((n, 3))
        
        # Distribute points in a grid-like pattern with some randomness
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Adjust grid to better fit the unit square
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Add slight randomness to avoid perfect grid
                x += random.uniform(-spacing_x/6, spacing_x/6)
                y += random.uniform(-spacing_y/6, spacing_y/6)
                
                # Ensure we stay within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Set initial radius to small value
                circles[idx] = [x, y, 0.02]
                idx += 1
            if idx >= n:
                break
        
        # Fill remaining positions with random valid placements
        for i in range(idx, n):
            attempts = 0
            while attempts < 100:
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                # Small initial radius
                r = 0.02
                if is_valid_placement(circles[:i], x, y, r):
                    circles[i] = [x, y, r]
                    break
                attempts += 1
        
        return circles
    
    def is_valid_placement(existing_circles: np.ndarray, x: float, y: float, r: float) -> bool:
        """Check if placing a circle at (x,y) with radius r is valid"""
        # Check boundary constraints
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
        
        # Check overlap with existing circles
        if len(existing_circles) > 0:
            for ex, ey, er in existing_circles:
                dist_sq = (x - ex)**2 + (y - ey)**2
                if dist_sq < (r + er)**2:
                    return False
        return True
    
    def compute_radius_sum(circles: np.ndarray) -> float:
        """Compute sum of all radii"""
        return np.sum(circles[:, 2])
    
    def constraint_func(circles_flat: np.ndarray) -> dict:
        """Convert flat array back to circles and apply constraints"""
        circles = circles_flat.reshape(-1, 3)
        # Boundary constraints
        constraints = []
        
        # Circle containment constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i] - c[3*i+2]})  # x >= r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i] - c[3*i+2]})  # 1-x >= r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: c[3*i+1] - c[3*i+2]})  # y >= r
            constraints.append({'type': 'ineq', 'fun': lambda c, i=i: 1 - c[3*i+1] - c[3*i+2]})  # 1-y >= r
            
        # Non-overlap constraints
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda c, i=i, j=j: np.sqrt((c[3*i] - c[3*j])**2 + (c[3*i+1] - c[3*j+1])**2) - (c[3*i+2] + c[3*j+2])
                })
        
        return constraints
    
    # Multi-start optimization approach
    best_sum = 0
    best_circles = None
    
    # Try multiple random initializations
    for attempt in range(10):
        try:
            # Generate initial configuration
            circles = generate_initial_placement()
            
            # Flatten for optimization
            flat_init = circles.flatten()
            
            # Optimization using scipy minimize with SLSQP
            def objective(x_flat):
                # Convert back to circles
                circles_local = x_flat.reshape(-1, 3)
                # We want to maximize sum of radii, so minimize negative sum
                return -np.sum(circles_local[:, 2])
            
            def constraint_func(x_flat):
                circles_local = x_flat.reshape(-1, 3)
                constraints = []
                
                # Boundary constraints
                for i in range(len(circles_local)):
                    x, y, r = circles_local[i]
                    constraints.append(x - r)  # x >= r
                    constraints.append(1 - x - r)  # 1-x >= r
                    constraints.append(y - r)  # y >= r
                    constraints.append(1 - y - r)  # 1-y >= r
                    
                # Non-overlap constraints
                for i in range(len(circles_local)):
                    for j in range(i+1, len(circles_local)):
                        dist_sq = (circles_local[i, 0] - circles_local[j, 0])**2 + \
                                 (circles_local[i, 1] - circles_local[j, 1])**2
                        min_dist = (circles_local[i, 2] + circles_local[j, 2])**2
                        constraints.append(dist_sq - min_dist)
                
                return np.array(constraints)
            
            # Define bounds (x, y, r) for each circle
            bounds = []
            for _ in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # r <= 0.5
            
            # Run optimization
            result = minimize(
                objective,
                flat_init,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = compute_radius_sum(optimized_circles)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized_circles.copy()
                    
        except Exception as e:
            continue
    
    # Final refinement with a greedy approach if needed
    if best_circles is None:
        # Fallback to simple initial placement
        best_circles = generate_initial_placement()
    
    # Apply final constraint checking and refinement
    refined_circles = best_circles.copy()
    
    # Greedy improvement step
    for _ in range(100):
        improved = False
        for i in range(n):
            # Try to increase radius of circle i
            current_r = refined_circles[i, 2]
            max_possible_r = min(
                refined_circles[i, 0], 1 - refined_circles[i, 0],
                refined_circles[i, 1], 1 - refined_circles[i, 1]
            )
            
            # Check for conflicts with other circles
            max_r = max_possible_r
            for j in range(n):
                if i != j:
                    dist_sq = (refined_circles[i, 0] - refined_circles[j, 0])**2 + \
                             (refined_circles[i, 1] - refined_circles[j, 1])**2
                    min_dist = (refined_circles[i, 2] + refined_circles[j, 2])**2
                    if dist_sq < min_dist:
                        # Need to reduce radius
                        max_r = min(max_r, np.sqrt(dist_sq) - refined_circles[j, 2] - 0.001)
            
            if max_r > current_r + 0.001:
                # Try to increase radius slightly
                test_r = min(current_r + 0.01, max_r)
                temp_circles = refined_circles.copy()
                temp_circles[i, 2] = test_r
                
                # Check validity
                valid = True
                for k in range(n):
                    if k != i:
                        dist_sq = (temp_circles[i, 0] - temp_circles[k, 0])**2 + \
                                 (temp_circles[i, 1] - temp_circles[k, 1])**2
                        if dist_sq < (temp_circles[i, 2] + temp_circles[k, 2])**2:
                            valid = False
                            break
                
                if valid:
                    refined_circles[i, 2] = test_r
                    improved = True
        
        if not improved:
            break
    
    return refined_circles


# EVOLVE-BLOCK-END
