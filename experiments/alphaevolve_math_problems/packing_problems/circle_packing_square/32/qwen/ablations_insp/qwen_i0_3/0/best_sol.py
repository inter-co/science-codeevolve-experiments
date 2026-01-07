# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from itertools import combinations
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, evolutionary algorithms, 
    and local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Better initialization using a more strategic approach
    def initialize_better():
        # Strategy: Use a known good starting configuration inspired by hexagonal packing
        circles = []
        
        # Create a hexagonal grid pattern in the center
        # This is a more systematic approach than random placement
        rows = 6
        cols = 6
        spacing = 0.15
        offset = 0.075
        
        # Hexagonal grid points
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Alternate rows for hexagonal pattern
                x = 0.2 + j * spacing + (i % 2) * spacing/2
                y = 0.2 + i * spacing * np.sqrt(3)/2
                if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                    circles.append([x, y, 0.05])
        
        # Fill remaining slots with scattered circles
        while len(circles) < n:
            # Random placement with some bias towards center
            x = 0.1 + 0.8 * np.random.random()
            y = 0.1 + 0.8 * np.random.random()
            # Small random radius
            r = 0.01 + 0.04 * np.random.random()
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Phase 2: Improved optimization with better constraint handling and method selection
    def objective(radii_and_positions):
        # Extract positions and radii
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Calculate negative sum of radii (we want to maximize)
        return -np.sum(radii)
    
    def constraint_func(radii_and_positions):
        positions = radii_and_positions[:2*n].reshape(-1, 2)
        radii = radii_and_positions[2*n:]
        
        # Position constraints (circle must fit in unit square)
        pos_constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Circle must stay inside unit square
            pos_constraints.extend([
                x - r,  # x - r >= 0
                1 - x - r,  # 1 - x - r >= 0
                y - r,  # y - r >= 0
                1 - y - r  # 1 - y - r >= 0
            ])
        
        # Non-overlap constraints - use more efficient approach with early termination
        overlap_constraints = []
        # Use KDTree for faster neighbor search
        tree = cKDTree(positions)
        
        for i in range(n):
            r_i = radii[i]
            # Find nearby circles using KDTree for efficiency
            neighbors = tree.query_ball_point(positions[i], 2*(r_i + 0.5), p=np.inf)
            for j in neighbors:
                if i < j:  # Avoid double counting
                    r_j = radii[j]
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx*dx + dy*dy
                    # Distance between centers must be >= sum of radii
                    overlap_constraints.append(np.sqrt(dist_sq) - r_i - r_j)
        
        return np.array(pos_constraints + overlap_constraints)
    
    # Phase 3: Enhanced optimization with multiple strategies
    def optimize_with_multiple_strategies(initial_guess, bounds, cons):
        best_result = None
        best_sum = -np.inf
        
        # Strategy 1: SLSQP with multiple restarts
        for attempt in range(5):
            try:
                # Slightly perturb initial guess for diversity
                if attempt > 0:
                    perturbed_guess = initial_guess.copy()
                    # Perturb positions slightly
                    for i in range(2*n):
                        if i % 2 == 0:  # x coordinate
                            perturbed_guess[i] += 0.01 * (np.random.random() - 0.5)
                        else:  # y coordinate
                            perturbed_guess[i] += 0.01 * (np.random.random() - 0.5)
                    # Perturb radii slightly
                    for i in range(n):
                        idx = 2*n + i
                        perturbed_guess[idx] *= (0.98 + 0.04 * np.random.random())
                    initial_guess = perturbed_guess
                
                result = minimize(
                    objective,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
                        
            except Exception as e:
                continue
        
        # Strategy 2: L-BFGS-B as backup
        if best_result is None or best_sum < 2.8:
            try:
                result = minimize(
                    objective,
                    initial_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-6}
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception as e:
                pass
        
        return best_result, best_sum
    
    # Initialize
    circles = initialize_better()
    
    # Flatten initial configuration
    initial_guess = np.concatenate([
        circles[:, :2].flatten(),  # positions
        circles[:, 2]              # radii
    ])
    
    # Set up bounds more carefully
    bounds = []
    # Position bounds: [0+r, 1-r] for both x and y (with safety margin)
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999)])  # x, y bounds
    # Radius bounds: [0.001, 0.499] 
    for i in range(n):
        bounds.extend([(0.001, 0.499)])
    
    # Create constraint dictionary
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    # Try optimization with multiple strategies
    best_result, best_sum = optimize_with_multiple_strategies(initial_guess, bounds, cons)
    
    # If we found a good result, use it; otherwise fallback to initial configuration
    if best_result is not None:
        final_positions = best_result.x[:2*n].reshape(-1, 2)
        final_radii = best_result.x[2*n:]
        circles = np.column_stack([final_positions, final_radii])
    else:
        # Fallback to initial configuration if optimization fails
        print("All optimization attempts failed, using initial configuration")
    
    return circles


# EVOLVE-BLOCK-END
