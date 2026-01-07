# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize using an extremely refined approach based on mathematical principles
    def initialize_mathematical_layout():
        # Create a layout that attempts to achieve maximum density
        circles = []
        
        # Place circles in a pattern that balances corner, edge, and center positions
        # Corner positions (slightly offset from corners)
        corners = [(0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]
        for x, y in corners:
            circles.append([x, y, 0.0])
        
        # Edge positions (more systematically placed)
        edge_positions = [
            (0.5, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 0.5),
            (0.25, 0.1), (0.75, 0.1), (0.25, 0.9), (0.75, 0.9),
            (0.1, 0.25), (0.1, 0.75), (0.9, 0.25), (0.9, 0.75)
        ]
        
        for x, y in edge_positions:
            if len(circles) < n:
                circles.append([x, y, 0.0])
        
        # Fill remaining with a more sophisticated grid
        remaining = n - len(circles)
        if remaining > 0:
            # Create a hexagonal-like grid in the center
            rows = int(np.ceil(np.sqrt(remaining)))
            cols = int(np.ceil(remaining / rows))
            
            # Calculate spacing to fit nicely in the center area
            center_width = 0.8
            center_height = 0.8
            spacing_x = center_width / cols if cols > 0 else center_width
            spacing_y = center_height / rows if rows > 0 else center_height
            
            # Offset to center the grid
            offset_x = 0.1
            offset_y = 0.1
            
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= n:
                        break
                    x = offset_x + (j + 0.5) * spacing_x
                    y = offset_y + (i + 0.5) * spacing_y
                    circles.append([x, y, 0.0])
        
        # Ensure we have exactly n circles
        circles = circles[:n]
        return np.array(circles)
    
    # Create initial configuration
    circles = initialize_mathematical_layout()
    
    # Set initial radii to carefully tuned values
    initial_radii = np.full(n, 0.03)
    
    # Combine positions and radii into a single parameter vector
    # Format: [x0, y0, r0, x1, y1, r1, ..., x31, y31, r31]
    initial_params = np.concatenate([circles[:, :2].flatten(), initial_radii])
    
    # Define constraint functions with more efficient implementation
    def get_constraints():
        """Return constraint functions for optimization"""
        cons = []
        
        # Boundary constraints: radius <= x <= 1-radius, radius <= y <= 1-radius
        def boundary_constraint(params):
            positions = params[:-n].reshape(-1, 2)
            radii = params[-n:]
            result = []
            
            # Vectorized computation
            x = positions[:, 0]
            y = positions[:, 1]
            r = radii
            
            # x - r >= 0
            result.extend(x - r)
            # y - r >= 0  
            result.extend(y - r)
            # 1 - x - r >= 0
            result.extend(1 - x - r)
            # 1 - y - r >= 0
            result.extend(1 - y - r)
                
            return np.array(result)
        
        # Non-overlap constraints: distance >= sum of radii
        def overlap_constraint(params):
            positions = params[:-n].reshape(-1, 2)
            radii = params[-n:]
            result = []
            
            # Use a more numerically stable approach
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    distance_squared = dx*dx + dy*dy
                    # Add small epsilon to prevent numerical issues
                    distance = math.sqrt(distance_squared + 1e-12)
                    min_distance = radii[i] + radii[j]
                    # Distance - (r_i + r_j) >= 0 (positive means no overlap)
                    result.append(distance - min_distance)
                    
            return np.array(result)
            
        cons.append({'type': 'ineq', 'fun': boundary_constraint})
        cons.append({'type': 'ineq', 'fun': overlap_constraint})
        
        return cons
    
    # Objective function to maximize sum of radii
    def objective(params):
        # We want to maximize sum of radii, so we minimize negative sum
        return -np.sum(params[-n:])
    
    # Constraints
    constraints = get_constraints()
    
    # Try multiple optimization strategies with highest precision
    best_result = None
    best_sum = 0
    
    # Strategy 1: Very aggressive Differential Evolution with highest settings
    try:
        # Define bounds for parameters [x0, y0, r0, x1, y1, r1, ..., x31, y31, r31]
        bounds = []
        for i in range(n):
            # x and y bounds: [0.001, 0.999] to leave room for radius
            bounds.extend([(0.001, 0.999), (0.001, 0.999)])
        # r bounds: [0.001, 0.499] to ensure space for other circles
        for i in range(n):
            bounds.append((0.001, 0.499))
        
        de_result = differential_evolution(
            objective,
            bounds,
            constraints=constraints,
            maxiter=300,
            popsize=35,
            seed=42,
            atol=1e-10,
            rtol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.95,
            strategy='best1bin'
        )
        
        if de_result.success:
            current_sum = -de_result.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = de_result
                
    except Exception as e:
        pass
    
    # Strategy 2: Multiple local optimizations with highest precision and diversity
    if best_result is None:
        # Try several local optimizations with maximum diversity
        for attempt in range(15):
            try:
                # Slightly vary initial parameters for maximum diversity
                np.random.seed(attempt * 100 + 42)
                varied_initial = initial_params.copy()
                # Even smaller noise to fine-tune
                varied_initial[:-n:2] += np.random.normal(0, 0.001, n)  # x positions
                varied_initial[1:-n:2] += np.random.normal(0, 0.001, n)  # y positions
                varied_initial[-n:] += np.random.normal(0, 0.001, n)     # radii
                
                result = minimize(
                    objective,
                    varied_initial,
                    method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 800, 'ftol': 1e-10, 'eps': 1e-10, 'iprint': 0}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
                        
            except Exception:
                continue
    
    # Strategy 3: Final attempt with different optimization method
    if best_result is None:
        try:
            # Try L-BFGS-B with ultra-high precision
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10, 'iprint': -1}
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception:
            pass
    
    # If we found a good result, use it; otherwise fallback to initial configuration
    if best_result is not None and best_result.success:
        final_positions = best_result.x[:-n].reshape(-1, 2)
        final_radii = best_result.x[-n:]
        
        # Create final circles array
        circles = np.column_stack([final_positions, final_radii])
    else:
        # Fallback to initial configuration with tuned initial radii
        circles = initialize_mathematical_layout()
        circles[:, 2] = 0.03
    
    # Final validation and adjustment
    # Ensure all circles fit properly within the unit square
    for i in range(n):
        x, y, r = circles[i]
        # Clip radii to keep circles within bounds
        max_radius_x = min(x, 1-x)
        max_radius_y = min(y, 1-y)
        max_radius = min(max_radius_x, max_radius_y)
        if r > max_radius:
            circles[i, 2] = max_radius
        # Ensure positive radius
        if r < 0:
            circles[i, 2] = 0.001
    
    return circles


# EVOLVE-BLOCK-END
