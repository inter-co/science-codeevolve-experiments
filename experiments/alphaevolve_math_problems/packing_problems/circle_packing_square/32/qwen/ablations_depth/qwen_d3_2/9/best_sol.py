# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
from scipy.optimize import Bounds
import random
from itertools import combinations
from typing import Tuple, List

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with advanced numerical optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # More sophisticated initialization using hexagonal packing pattern for better density
    def initialize_hexagonal_layout():
        # Create a more structured initial layout based on hexagonal close packing principles
        circles = []
        
        # Place circles in a grid-like pattern with offset rows for hexagonal packing
        spacing = 0.15  # Initial spacing
        radius_guess = 0.03  # Initial radius guess
        
        # Generate a grid pattern with some randomness for diversity
        rows = 6
        cols = 6
        offset = 0.075  # Offset for hexagonal arrangement
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = 0.1 + j * spacing + (i % 2) * offset
                y = 0.1 + i * spacing
                # Keep within bounds
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, radius_guess])
        
        # Fill remaining positions with scattered points
        remaining = n - len(circles)
        if remaining > 0:
            for _ in range(remaining):
                x = random.uniform(0.1, 0.9)
                y = random.uniform(0.1, 0.9)
                circles.append([x, y, radius_guess])
        
        # Ensure we have exactly n circles
        circles = circles[:n]
        return np.array(circles)
    
    # Alternative initialization using Voronoi-inspired approach
    def initialize_voronoi_layout():
        # Start with fewer circles and expand
        circles = []
        
        # Place some key positions first
        key_positions = [
            (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),  # corners
            (0.5, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.5),  # mid-edges
            (0.5, 0.5),  # center
        ]
        
        # Add key positions with initial radii
        for x, y in key_positions:
            if len(circles) < n:
                circles.append([x, y, 0.03])
        
        # Fill remaining with scattered points
        remaining = n - len(circles)
        if remaining > 0:
            for _ in range(remaining):
                x = random.uniform(0.1, 0.9)
                y = random.uniform(0.1, 0.9)
                circles.append([x, y, 0.03])
        
        return np.array(circles)
    
    # Initialize using better approach
    circles = initialize_hexagonal_layout()
    
    # Better initialization for radii
    initial_radii = np.full(n, 0.03)
    
    # Combine positions and radii into a single parameter vector
    # Format: [x0, y0, r0, x1, y1, r1, ..., x31, y31, r31]
    initial_params = np.concatenate([circles[:, :2].flatten(), initial_radii])
    
    # More efficient constraint implementation
    def create_constraints():
        """Create optimized constraint functions for better performance"""
        # Precompute pairs for overlap constraints to avoid redundant calculations
        pairs = list(combinations(range(n), 2))
        
        def boundary_constraint(params):
            positions = params[:-n].reshape(-1, 2)
            radii = params[-n:]
            result = []
            
            # Vectorized boundary constraints
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
        
        def overlap_constraint(params):
            positions = params[:-n].reshape(-1, 2)
            radii = params[-n:]
            result = []
            
            # Optimized pairwise overlap checks
            for i, j in pairs:
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance_squared = dx*dx + dy*dy
                # Add small epsilon to prevent numerical issues
                distance = math.sqrt(distance_squared + 1e-12)
                min_distance = radii[i] + radii[j]
                # Distance - (r_i + r_j) >= 0 (positive means no overlap)
                result.append(distance - min_distance)
                
            return np.array(result)
            
        return [
            {'type': 'ineq', 'fun': boundary_constraint},
            {'type': 'ineq', 'fun': overlap_constraint}
        ]
    
    # Objective function to maximize sum of radii
    def objective(params):
        # We want to maximize sum of radii, so we minimize negative sum
        return -np.sum(params[-n:])
    
    # Constraints
    constraints = create_constraints()
    
    # Improved optimization approach with multiple strategies
    best_result = None
    best_sum = 0
    
    # Strategy 1: Global optimization with better settings
    try:
        # Define bounds for parameters [x0, y0, r0, x1, y1, r1, ..., x31, y31, r31]
        bounds = []
        for i in range(n):
            # x and y bounds: [0.001, 0.999] to leave room for radius
            bounds.extend([(0.001, 0.999), (0.001, 0.999)])
        # r bounds: [0.001, 0.499] to ensure space for other circles
        for i in range(n):
            bounds.append((0.001, 0.499))
        
        # Try different evolutionary strategies
        strategies = ['best1bin', 'best2bin', 'rand1bin']
        for strategy in strategies:
            try:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    constraints=constraints,
                    maxiter=150,
                    popsize=30,
                    seed=42,
                    atol=1e-9,
                    rtol=1e-9,
                    mutation=(0.5, 1.0),
                    recombination=0.8,
                    strategy=strategy
                )
                
                if de_result.success:
                    current_sum = -de_result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = de_result
            except Exception:
                continue
                
    except Exception as e:
        pass
    
    # Strategy 2: Local optimization with multiple restarts
    if best_result is None or best_sum < 2.8:  # Only if we haven't done well yet
        # Try several local optimizations with different starting points
        for attempt in range(10):
            try:
                # Generate diverse initial points
                np.random.seed(attempt * 100 + 42)
                varied_initial = initial_params.copy()
                
                # Add more substantial noise to positions and radii
                varied_initial[:-n:2] += np.random.normal(0, 0.01, n)  # x positions
                varied_initial[1:-n:2] += np.random.normal(0, 0.01, n)  # y positions
                varied_initial[-n:] += np.random.normal(0, 0.005, n)     # radii
                
                # Ensure bounds are respected after noise
                for i in range(n):
                    varied_initial[i*2] = np.clip(varied_initial[i*2], 0.001, 0.999)
                    varied_initial[i*2+1] = np.clip(varied_initial[i*2+1], 0.001, 0.999)
                    varied_initial[n*2+i] = np.clip(varied_initial[n*2+i], 0.001, 0.499)
                
                # Use more robust optimizer with tighter tolerances
                result = minimize(
                    objective,
                    varied_initial,
                    method='SLSQP',
                    constraints=constraints,
                    options={'maxiter': 800, 'ftol': 1e-9, 'eps': 1e-9, 'iprint': 0}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
                        
            except Exception:
                continue
    
    # Strategy 3: Try COBYLA if previous methods failed
    if best_result is None or best_sum < 2.8:
        try:
            # Try COBYLA with more iterations
            result = minimize(
                objective,
                initial_params,
                method='COBYLA',
                constraints=constraints,
                options={'maxiter': 1000, 'tol': 1e-8}
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception:
            pass
    
    # Strategy 4: Final refinement with trust-constr if available
    if best_result is None or best_sum < 2.8:
        try:
            result = minimize(
                objective,
                initial_params,
                method='trust-constr',
                constraints=constraints,
                options={'maxiter': 500, 'xtol': 1e-9, 'gtol': 1e-9}
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
        # Fallback to better initial configuration
        circles = initialize_voronoi_layout()
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
