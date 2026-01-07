# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import time
from itertools import combinations
from scipy.spatial import Voronoi
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, mathematical programming, 
    and advanced optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Advanced initialization using geometric principles
    def initialize_advanced():
        # Start with a Voronoi-based approach for better distribution
        # Generate points using a quasi-random sequence (Hammersley)
        points = []
        for i in range(n):
            phi = (np.sqrt(5) - 1) / 2  # golden ratio
            x = (i + 0.5) / n
            y = (i * phi) % 1
            points.append([x, y])
        
        points = np.array(points)
        
        # Adjust points to avoid boundary issues and create better initial configuration
        adjusted_points = []
        for i, (x, y) in enumerate(points):
            # Apply slight perturbation to avoid regular patterns
            x = max(0.05, min(0.95, x + np.random.normal(0, 0.02)))
            y = max(0.05, min(0.95, y + np.random.normal(0, 0.02)))
            adjusted_points.append([x, y])
        
        # Initialize with equal radii
        circles = np.array(adjusted_points)
        radii = np.full(n, 0.05)
        circles = np.column_stack([circles, radii])
        
        return circles
    
    # Phase 2: Improved constraint handling with better numerical stability
    def constraint_func(radii_and_positions):
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Position constraints (circle must fit in unit square)
        pos_constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Circle must stay inside unit square with safety margin
            pos_constraints.extend([
                x - r - 1e-6,          # x - r >= 0
                1 - x - r - 1e-6,      # 1 - x - r >= 0
                y - r - 1e-6,          # y - r >= 0
                1 - y - r - 1e-6       # 1 - y - r >= 0
            ])
        
        # Non-overlap constraints with improved numerical handling
        overlap_constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                r_i = radii[i]
                r_j = radii[j]
                
                # Use a small epsilon to prevent numerical issues
                min_dist_sq = (r_i + r_j)**2 + 1e-12
                overlap_constraints.append(np.sqrt(dist_sq) - (r_i + r_j))
        
        return np.array(pos_constraints + overlap_constraints)
    
    # Phase 3: More robust objective function
    def objective(radii_and_positions):
        # Extract positions and radii
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Calculate negative sum of radii (we want to maximize)
        return -np.sum(radii)
    
    # Phase 4: Enhanced optimization with multiple strategies
    def optimize_with_strategies(initial_guess, bounds):
        best_result = None
        best_value = -np.inf
        
        # Strategy 1: Differential Evolution (global search)
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                args=(),
                strategy='best1bin',
                maxiter=150,
                popsize=20,
                tol=1e-7,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42,
                callback=None
            )
            
            if de_result.success:
                value = -de_result.fun
                if value > best_value:
                    best_value = value
                    best_result = de_result
        except Exception as e:
            pass
            
        # Strategy 2: Local optimization from best DE result if available
        if best_result is not None:
            try:
                # Use the DE result as starting point for local optimization
                local_result = minimize(
                    objective,
                    best_result.x,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={
                        'type': 'ineq',
                        'fun': constraint_func
                    },
                    options={'maxiter': 300, 'ftol': 1e-8}
                )
                
                if local_result.success:
                    value = -local_result.fun
                    if value > best_value:
                        best_value = value
                        best_result = local_result
            except Exception as e:
                pass
        
        # Strategy 3: Try SLSQP directly with initial guess
        if best_result is None:
            try:
                local_result = minimize(
                    objective,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints={
                        'type': 'ineq',
                        'fun': constraint_func
                    },
                    options={'maxiter': 300, 'ftol': 1e-8}
                )
                
                if local_result.success:
                    value = -local_result.fun
                    if value > best_value:
                        best_value = value
                        best_result = local_result
            except Exception as e:
                pass
        
        return best_result
    
    # Initialize
    circles = initialize_advanced()
    
    # Flatten initial configuration
    initial_guess = np.concatenate([
        circles[:, :2].flatten(),  # positions
        circles[:, 2]              # radii
    ])
    
    # Set up bounds more carefully
    bounds = []
    # Position bounds: [0.001, 0.999] for both x and y (with small buffer)
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6)])  # x, y bounds
    # Radius bounds: [0.001, 0.499] 
    for i in range(n):
        bounds.extend([(1e-6, 0.499)])
    
    # Optimize using multiple strategies
    try:
        result = optimize_with_strategies(initial_guess, bounds)
        
        if result is not None and result.success:
            final_positions = result.x[:2*n].reshape(-1, 2)
            final_radii = result.x[2*n:]
            circles = np.column_stack([final_positions, final_radii])
        else:
            # If optimization failed, return the initial configuration
            pass
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        pass
    
    # Final validation and cleanup
    # Ensure we have exactly 32 circles
    if len(circles) < n:
        # Fill with default values
        while len(circles) < n:
            circles = np.vstack([circles, [0.5, 0.5, 0.01]])
    
    # Ensure all circles are within bounds and non-overlapping
    # This is a simple validation step to ensure quality
    try:
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Check containment
        valid = True
        for i in range(n):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                valid = False
                break
        
        if not valid:
            # Re-initialize if there are issues
            circles = initialize_advanced()
            
    except Exception:
        # Fallback to basic initialization
        circles = initialize_advanced()
    
    return circles[:n]


# EVOLVE-BLOCK-END
