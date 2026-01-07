# EVOLVE-BLOCK-START
import numpy as np
import random
from typing import Tuple
import math
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with advanced optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 26
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    
    # Better initialization: use a more sophisticated approach
    # Start with a hexagonal packing approximation
    circles = np.zeros((n, 3))
    
    # Use a hexagonal lattice pattern for initial placement
    # This helps achieve better packing density
    rows = 5
    cols = 5
    spacing_x = 0.95 / cols
    spacing_y = 0.95 / rows
    margin = 0.025
    
    # Create a more uniform initial distribution
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Offset every other row for hexagonal packing
            x_offset = spacing_x/2 if i % 2 == 1 else 0
            x = margin + j * spacing_x + x_offset
            y = margin + i * spacing_y + spacing_y/2
            # Start with a reasonable initial radius
            r = 0.04
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining circles with strategic placement
    if idx < n:
        # Place remaining circles in a more scattered pattern
        for i in range(idx, n):
            # Use a more strategic approach than pure randomness
            # Place in regions that are likely to accommodate larger circles
            attempts = 0
            while attempts < 100:
                x = margin + random.random() * (1 - 2*margin)
                y = margin + random.random() * (1 - 2*margin)
                # Find the minimum distance to existing circles
                min_dist = float('inf')
                for k in range(i):
                    cx, cy, _ = circles[k]
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    min_dist = min(min_dist, dist)
                
                # If we're far from others, use a larger initial radius
                if min_dist > 0.1:
                    r = min(0.08, min_dist/2)
                else:
                    r = 0.03
                    
                # Check if this position works with existing circles
                valid = True
                for k in range(i):
                    cx, cy, cr = circles[k]
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < (r + cr):
                        valid = False
                        break
                
                if valid:
                    circles[i] = [x, y, r]
                    break
                attempts += 1
                
            # Fallback to random if needed
            if attempts >= 100:
                circles[i] = [
                    margin + random.random() * (1 - 2*margin),
                    margin + random.random() * (1 - 2*margin),
                    0.03
                ]
    
    # More sophisticated optimization using sequential quadratic programming
    # Create parameter vector [x1,y1,r1,x2,y2,r2,...,x26,y26,r26]
    def objective(params):
        # We want to maximize sum of radii, so return negative
        total_radius = sum(params[3*i+2] for i in range(n))
        return -total_radius
    
    def constraint_func(params):
        constraints = []
        
        # Boundary constraints: each circle must fit completely in unit square
        for i in range(n):
            x, y, r = params[3*i:3*i+3]
            constraints.extend([
                x - r,           # x - r >= 0
                y - r,           # y - r >= 0  
                1 - x - r,       # 1 - x - r >= 0
                1 - y - r        # 1 - y - r >= 0
            ])
        
        # Overlap constraints: distance >= sum of radii
        for i, j in combinations(range(n), 2):
            x1, y1, r1 = params[3*i:3*i+3]
            x2, y2, r2 = params[3*j:3*j+3]
            dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            # Constraint: dist >= r1 + r2 (so we return dist - (r1+r2) >= 0)
            constraints.append(dist - (r1 + r2))
        
        return np.array(constraints)
    
    # Set up bounds: x in [r, 1-r], y in [r, 1-r], r in [0.001, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])
    
    # Prepare initial parameters
    initial_params = []
    for i in range(n):
        x, y, r = circles[i]
        initial_params.extend([x, y, r])
    
    # Try different optimization approaches
    try:
        # First try with SLSQP which handles constraints well
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': lambda p: constraint_func(p)}],
            options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
        )
        
        if result.success:
            optimized_params = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = optimized_params[3*i:3*i+3]
        else:
            # If SLSQP fails, fall back to a more robust approach
            pass
            
    except Exception as e:
        # If optimization fails, keep the initial configuration
        pass
    
    # Post-processing with local refinement
    # Try to increase radii where possible without violating constraints
    improved = True
    max_iterations = 100
    
    for iteration in range(max_iterations):
        if not improved:
            break
        improved = False
        
        # Try to increase each radius
        for i in range(n):
            x, y, r = circles[i]
            
            # Try to increase radius slightly
            test_r = min(0.5, r + 0.001)
            
            # Check if this causes conflicts
            valid = True
            for j in range(n):
                if i != j:
                    x2, y2, r2 = circles[j]
                    dist = math.sqrt((x2-x)**2 + (y2-y)**2)
                    if dist < (test_r + r2):
                        valid = False
                        break
            
            if valid:
                circles[i][2] = test_r
                improved = True
        
        # Apply boundary corrections
        for i in range(n):
            x, y, r = circles[i]
            # Keep within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[i] = [x, y, r]
    
    # Final cleanup phase to resolve any remaining overlaps
    for _ in range(50):
        changed = False
        for i in range(n):
            x, y, r = circles[i]
            # Keep within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[i] = [x, y, r]
            
            # Resolve any overlaps with other circles
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x2 - x1
                dy = y2 - y1
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < (r1 + r2):
                    # Resolve overlap by moving both circles apart
                    if dist > 0.001:
                        overlap = (r1 + r2) - dist
                        move_x = overlap * dx / dist / 2
                        move_y = overlap * dy / dist / 2
                        
                        circles[i][0] -= move_x
                        circles[i][1] -= move_y
                        circles[j][0] += move_x
                        circles[j][1] += move_y
                    else:
                        # Very close - move randomly to break symmetry
                        angle = random.random() * 2 * math.pi
                        offset = 0.001
                        circles[i][0] -= offset * math.cos(angle)
                        circles[i][1] -= offset * math.sin(angle)
                        circles[j][0] += offset * math.cos(angle)
                        circles[j][1] += offset * math.sin(angle)
                    
                    # Keep within bounds
                    circles[i][0] = max(circles[i][2], min(1-circles[i][2], circles[i][0]))
                    circles[i][1] = max(circles[i][2], min(1-circles[i][2], circles[i][1]))
                    circles[j][0] = max(circles[j][2], min(1-circles[j][2], circles[j][0]))
                    circles[j][1] = max(circles[j][2], min(1-circles[j][2], circles[j][1]))
                    
                    changed = True
        
        if not changed:
            break
    
    return circles


# EVOLVE-BLOCK-END
