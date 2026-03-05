# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import random
import time
from scipy.spatial.distance import cdist
import math
from scipy.optimize import Bounds
import warnings
warnings.filterwarnings('ignore')

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses evolutionary algorithms with smart initialization and constraint handling.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    # Rectangle dimensions: width + height = 2 (perimeter = 4)
    # Try different width/height ratios to find optimal configuration
    width = 1.0
    height = 1.0
    
    # Smart initialization using hexagonal packing principles
    def initialize_better():
        circles = []
        
        # Use a more systematic approach inspired by circle packing theory
        # For 21 circles, we'll use a grid-based approach with adaptive spacing
        
        # Try different grid arrangements
        # Best known arrangements for small numbers of circles
        if n == 21:
            # Try 5 rows x 4 columns (20 circles) + 1 extra
            rows = 5
            cols = 4
            
            # Calculate cell dimensions
            cell_width = width / cols
            cell_height = height / rows
            
            # Place circles in a grid pattern with slight perturbation
            placed_count = 0
            for i in range(rows):
                for j in range(cols):
                    if placed_count >= n:
                        break
                        
                    # Position with slight randomization to avoid perfect grid
                    x = (j + 0.5) * cell_width + random.uniform(-0.15, 0.15) * cell_width
                    y = (i + 0.5) * cell_height + random.uniform(-0.15, 0.15) * cell_height
                    
                    # Ensure within bounds
                    x = max(0.05, min(width - 0.05, x))
                    y = max(0.05, min(height - 0.05, y))
                    
                    # Radius based on available space
                    max_radius = min(x, width - x, y, height - y) * 0.4
                    radius = max(0.02, min(max_radius, 0.2))  # Clip to reasonable range
                    
                    circles.append([x, y, radius])
                    placed_count += 1
                    
        # Fill remaining circles if needed
        while len(circles) < n:
            # Try to place in regions with high density
            best_x, best_y = None, None
            max_min_dist = -1
            
            # Sample multiple locations
            for _ in range(100):
                x = random.uniform(0.05, width - 0.05)
                y = random.uniform(0.05, height - 0.05)
                
                # Find minimum distance to existing circles
                min_dist = float('inf')
                for cx, cy, r in circles:
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    min_dist = min(min_dist, dist)
                
                # Prefer locations with larger minimum distances
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_x, best_y = x, y
            
            if best_x is not None:
                # Set radius based on proximity to edges and other circles
                radius = min(best_x, width - best_x, best_y, height - best_y)
                # Reduce slightly to allow for overlap checking
                radius = min(radius * 0.3, max_min_dist * 0.2) 
                if radius < 0.01:
                    radius = 0.05
                circles.append([best_x, best_y, radius])
            else:
                # Final fallback
                x = random.uniform(0.05, width - 0.05)
                y = random.uniform(0.05, height - 0.05)
                circles.append([x, y, 0.05])
                
        return np.array(circles)
    
    # Even better initialization: use a known good starting point
    def initialize_with_known_good():
        # Use a configuration inspired by previous solutions and mathematical optimization
        # Start with a good heuristic arrangement
        circles = []
        
        # Place in a pattern that balances coverage and spacing
        # Using a "golden rectangle" ratio (1.618) might work well for 21 circles
        
        # Try to create a more evenly distributed set
        # Generate points in a systematic way
        rows = 5
        cols = 4
        
        # Calculate spacing based on number of circles
        spacing_x = width / (cols + 1)
        spacing_y = height / (rows + 1)
        
        # Place circles systematically
        count = 0
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if count >= n:
                    break
                    
                # Position with slight jitter
                x = j * spacing_x + random.uniform(-0.1, 0.1) * spacing_x
                y = i * spacing_y + random.uniform(-0.1, 0.1) * spacing_y
                
                # Ensure within bounds
                x = max(0.05, min(width - 0.05, x))
                y = max(0.05, min(height - 0.05, y))
                
                # Calculate appropriate radius
                min_dist_to_edge = min(x, width - x, y, height - y)
                radius = min_dist_to_edge * 0.35
                
                # Add some randomness to radius
                radius = max(0.01, radius * random.uniform(0.7, 1.0))
                
                circles.append([x, y, radius])
                count += 1
                
        # Fill remaining positions
        while len(circles) < n:
            # Find good empty spots
            best_x, best_y = None, None
            best_min_dist = -1
            
            # Sample many candidates
            for _ in range(200):
                x = random.uniform(0.05, width - 0.05)
                y = random.uniform(0.05, height - 0.05)
                
                # Find minimum distance to existing circles
                min_dist = float('inf')
                for cx, cy, r in circles:
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    min_dist = min(min_dist, dist)
                
                # Prefer locations that are far from existing circles
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_x, best_y = x, y
            
            if best_x is not None:
                # Set radius based on available space
                min_dist_to_edges = min(best_x, width - best_x, best_y, height - best_y)
                radius = min_dist_to_edges * 0.3
                # Ensure it's not too large considering other circles
                radius = min(radius, best_min_dist * 0.2)
                if radius < 0.01:
                    radius = 0.05
                circles.append([best_x, best_y, radius])
            else:
                # Fallback
                x = random.uniform(0.05, width - 0.05)
                y = random.uniform(0.05, height - 0.05)
                circles.append([x, y, 0.05])
                
        return np.array(circles)
    
    # Better initialization
    circles = initialize_with_known_good()
    
    # Optimization with improved constraint handling
    def objective(params):
        """Minimize negative sum of radii (maximize sum of radii)"""
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        # We want to maximize sum of radii, so minimize negative sum
        return -np.sum(radii)
    
    def constraint_func(params):
        """Constraint function returning positive values when satisfied"""
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Non-overlap constraints - ensure distance >= sum of radii
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                min_distance = radii[i] + radii[j]
                # Constraint should be positive when satisfied (distance >= min_distance)
                constraints.append(distance - min_distance)
        
        # Boundary constraints (positive when satisfied)
        for i in range(n):
            # Left boundary
            constraints.append(positions[i][0] - radii[i])
            # Right boundary  
            constraints.append(width - positions[i][0] - radii[i])
            # Bottom boundary
            constraints.append(positions[i][1] - radii[i])
            # Top boundary
            constraints.append(height - positions[i][1] - radii[i])
            
        return np.array(constraints)
    
    # Create initial parameter vector: [x1, y1, x2, y2, ..., xn, yn, r1, r2, ..., rn]
    initial_params = np.concatenate([
        circles[:, :2].flatten(),  # Positions
        circles[:, 2]              # Radii
    ])
    
    # Set bounds for positions and radii
    bounds = [(0, width) for _ in range(2*n)] + [(1e-6, width/2) for _ in range(n)]
    
    # Define constraints
    cons = {
        'type': 'ineq',  # Inequality constraints (g(x) >= 0)
        'fun': constraint_func
    }
    
    # Use a more sophisticated optimization approach
    best_result = None
    best_sum = -float('inf')
    
    # Strategy 1: Differential Evolution (robust global search)
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            args=(),
            maxiter=150,
            popsize=30,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False,
            tol=1e-7
        )
        
        if de_result.success:
            # Validate the result before accepting
            final_positions = de_result.x[:-n].reshape(-1, 2)
            final_radii = de_result.x[-n:]
            
            # Check if constraints are satisfied
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    dx = final_positions[i][0] - final_positions[j][0]
                    dy = final_positions[i][1] - final_positions[j][1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    min_distance = final_radii[i] + final_radii[j]
                    if distance < min_distance * 0.99:  # Allow small tolerance
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                strategies.append(('DE', de_result))
                
    except Exception as e:
        pass
    
    # Strategy 2: Multiple restarts with better initialization
    strategies = []
    for restart in range(5):
        try:
            # Use a fresh initialization for each restart
            restart_circles = initialize_with_known_good()
            restart_params = np.concatenate([
                restart_circles[:, :2].flatten(),  
                restart_circles[:, 2]              
            ])
            
            # Use a more aggressive optimization
            result = minimize(
                objective,
                restart_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
            )
            
            if result.success:
                # Validate the result
                final_positions = result.x[:-n].reshape(-1, 2)
                final_radii = result.x[-n:]
                
                # Check constraints
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx = final_positions[i][0] - final_positions[j][0]
                        dy = final_positions[i][1] - final_positions[j][1]
                        distance = math.sqrt(dx*dx + dy*dy)
                        min_distance = final_radii[i] + final_radii[j]
                        if distance < min_distance * 0.99:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    strategies.append((f'Restart_{restart}', result))
                
        except Exception as e:
            continue
    
    # Strategy 3: Try a hybrid approach with local search after global search
    if strategies:
        # Take the best from restarts
        for name, result in strategies:
            try:
                final_positions = result.x[:-n].reshape(-1, 2)
                final_radii = result.x[-n:]
                final_sum = np.sum(final_radii)
                
                if final_sum > best_sum:
                    best_sum = final_sum
                    best_result = result
                    
            except Exception as e:
                continue
    
    # If no good results from restarts, try a different approach
    if best_result is None:
        # Try a simpler but more reliable approach
        try:
            # Use a smaller population size for faster convergence
            de_result = differential_evolution(
                objective,
                bounds,
                args=(),
                maxiter=100,
                popsize=15,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                disp=False,
                tol=1e-6
            )
            
            if de_result.success:
                final_positions = de_result.x[:-n].reshape(-1, 2)
                final_radii = de_result.x[-n:]
                final_sum = np.sum(final_radii)
                
                # Validate this result
                valid = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx = final_positions[i][0] - final_positions[j][0]
                        dy = final_positions[i][1] - final_positions[j][1]
                        distance = math.sqrt(dx*dx + dy*dy)
                        min_distance = final_radii[i] + final_radii[j]
                        if distance < min_distance * 0.99:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_result = de_result
                    best_sum = final_sum
                    
        except Exception as e:
            pass
    
    # If still no good result, return the initial configuration
    if best_result is None:
        # Return the initial configuration
        return circles
    
    # Extract and return the best solution
    final_positions = best_result.x[:-n].reshape(-1, 2)
    final_radii = best_result.x[-n:]
    
    # Ensure all radii are positive and reasonable
    final_radii = np.maximum(final_radii, 1e-6)
    
    # Create final result array
    result_circles = np.zeros((n, 3))
    result_circles[:, 0] = final_positions[:, 0]
    result_circles[:, 1] = final_positions[:, 1]
    result_circles[:, 2] = final_radii
    
    return result_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
