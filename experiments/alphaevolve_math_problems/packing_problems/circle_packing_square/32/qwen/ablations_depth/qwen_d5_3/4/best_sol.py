# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import math
from scipy.spatial import cKDTree
import warnings
from typing import Tuple
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a sophisticated hybrid approach combining multiple optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization using a more informed approach
    circles = initialize_better_pack(n)
    
    # Use a more targeted optimization approach
    circles = optimize_with_de_targeted(circles)
    
    # Final aggressive refinement with better local search
    circles = refine_circles_fast(circles)
    
    return circles

def initialize_better_pack(n):
    """Initialize circles with a more effective approach inspired by known good packings"""
    circles = np.zeros((n, 3))
    
    # Create a more structured initial layout
    # Use a 4x8 grid pattern with some strategic adjustments
    rows = 4
    cols = 8
    
    # Calculate spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Initial radius estimate based on hexagonal packing efficiency
    # For a hexagonal arrangement, the optimal packing density is π/(2√3) ≈ 0.9069
    max_radius = min(spacing_x, spacing_y) * 0.45
    
    # Place in grid pattern with strategic perturbations
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Add strategic perturbations to improve optimization
            if i % 2 == 1:  # Offset every other row
                x += spacing_x * 0.25
            if j % 2 == 1:  # Offset every other column
                y += spacing_y * 0.25
                
            # Add small random perturbation to break symmetry
            x += np.random.uniform(-spacing_x/8, spacing_x/8)
            y += np.random.uniform(-spacing_y/8, spacing_y/8)
            
            # Ensure within bounds
            x = np.clip(x, max_radius, 1 - max_radius)
            y = np.clip(y, max_radius, 1 - max_radius)
            
            circles[idx] = [x, y, max_radius]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with a more intelligent greedy approach
    # Start from the center and work outward to maximize space utilization
    center_positions = [(0.5, 0.5)]
    for i in range(idx, n):
        best_placement = None
        best_radius = 0
        
        # Try multiple strategies for finding good placements
        strategies = ['center', 'edge', 'random']
        
        for strategy in strategies:
            for _ in range(1000):  # More attempts for better initialization
                if strategy == 'center':
                    # Place near center with random direction
                    angle = np.random.uniform(0, 2*np.pi)
                    distance = np.random.uniform(0.1, 0.4)
                    x = 0.5 + distance * np.cos(angle)
                    y = 0.5 + distance * np.sin(angle)
                elif strategy == 'edge':
                    # Place near edges
                    side = np.random.choice(['top', 'bottom', 'left', 'right'])
                    if side == 'top':
                        y = 1 - max_radius
                        x = np.random.uniform(max_radius, 1 - max_radius)
                    elif side == 'bottom':
                        y = max_radius
                        x = np.random.uniform(max_radius, 1 - max_radius)
                    elif side == 'left':
                        x = max_radius
                        y = np.random.uniform(max_radius, 1 - max_radius)
                    else:  # right
                        x = 1 - max_radius
                        y = np.random.uniform(max_radius, 1 - max_radius)
                else:  # random
                    x = np.random.uniform(max_radius, 1 - max_radius)
                    y = np.random.uniform(max_radius, 1 - max_radius)
                
                # Calculate maximum possible radius at this position
                max_possible_radius = min(x, 1-x, y, 1-y)
                
                # Find minimum distance to existing circles
                min_dist = float('inf')
                for k in range(i):
                    cx, cy, cr = circles[k]
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    min_dist = min(min_dist, dist)
                
                # If there are existing circles, we can't go larger than half the minimum distance
                if min_dist < float('inf') and min_dist > 0:
                    max_possible_radius = min(max_possible_radius, min_dist/2)
                
                # Prefer positions that allow larger radii
                if max_possible_radius > best_radius:
                    best_radius = max_possible_radius
                    best_placement = (x, y, max_possible_radius)
        
        if best_placement is not None:
            circles[i] = best_placement
    
    return circles

def optimize_with_de_targeted(circles):
    """Use a more efficient differential evolution approach"""
    n = len(circles)
    
    # Define bounds - more carefully chosen for better convergence
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    def objective(params):
        # Convert params to circles array
        temp_circles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            temp_circles.append([x, y, r])
        
        # Maximize sum of radii (minimize negative sum)
        return -sum(circle[2] for circle in temp_circles)
    
    def constraint_func(params):
        # Check all constraints and return penalty if violated
        temp_circles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            temp_circles.append([x, y, r])
        
        # Check containment constraints
        for circle in temp_circles:
            x, y, r = circle
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return 10000  # Large penalty for containment violation
        
        # Check overlap constraints efficiently using spatial indexing
        try:
            # Build KDTree for efficient neighbor searches
            points = np.array([[c[0], c[1]] for c in temp_circles])
            tree = cKDTree(points)
            
            # Check overlaps efficiently - only check pairs that are potentially close
            for i in range(n):
                x1, y1, r1 = temp_circles[i]
                # Find nearby circles (within 2*(r1+r2) distance)
                nearby = tree.query_ball_point([x1, y1], 2*(r1 + 0.001))
                
                for j in nearby:
                    if i != j:
                        x2, y2, r2 = temp_circles[j]
                        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist < r1 + r2:
                            return 10000  # Penalty for overlap
        except:
            # Fallback to brute force if KDTree fails
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = temp_circles[i]
                    x2, y2, r2 = temp_circles[j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if dist < r1 + r2:
                        return 10000  # Penalty for overlap
        
        return 0
    
    # Single, more targeted optimization pass
    try:
        # Use fewer iterations but more sophisticated parameters for better results
        result = differential_evolution(
            objective,
            bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            seed=42,
            maxiter=1000,  # Reduced iterations for speed
            popsize=30,    # Balanced population size
            mutation=(0.8, 1),  # Slightly higher mutation rate
            recombination=0.9,   # Higher recombination rate
            disp=False
        )
        if result.success:
            optimized_params = result.x
            result_circles = np.zeros((n, 3))
            for i in range(n):
                result_circles[i] = [optimized_params[3*i], optimized_params[3*i+1], optimized_params[3*i+2]]
            circles = result_circles
    except Exception as e:
        print(f"DE optimization failed: {e}")
        pass
    
    return circles

def refine_circles_fast(circles):
    """Apply fast and effective refinement"""
    n = len(circles)
    
    # Create a copy to work with
    refined = circles.copy()
    
    # Strategy 1: Fast local search with better acceptance criteria
    for iteration in range(5000):  # Fewer iterations but more effective
        # Pick a random circle to adjust
        i = np.random.randint(0, n)
        
        # Store current state
        current_x, current_y, current_r = refined[i]
        
        # Try to improve this circle's configuration
        best_r = current_r
        best_x, best_y = current_x, current_y
        
        # Fixed step sizes for speed
        step_size = 0.01
        
        # Generate moves with different patterns
        moves = []
        
        # Primary moves: small adjustments
        for _ in range(20):
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            dr = np.random.uniform(-step_size/4, step_size/4)
            moves.append((dx, dy, dr))
        
        # Evaluate each move
        for dx, dy, dr in moves:
            new_x = current_x + dx
            new_y = current_y + dy
            new_r = current_r + dr
            
            # Ensure bounds
            new_x = np.clip(new_x, 0.001, 0.999)
            new_y = np.clip(new_y, 0.001, 0.999)
            new_r = np.clip(new_r, 0.001, 0.499)
            
            # Check constraints
            if check_constraints_single(refined, i, new_x, new_y, new_r):
                # Calculate change in objective (sum of radii)
                old_sum = sum(circle[2] for circle in refined)
                new_sum = old_sum - current_r + new_r
                
                # Accept better solutions with probability based on improvement
                if new_sum > old_sum:
                    best_r = new_r
                    best_x = new_x
                    best_y = new_y
                elif np.random.random() < 0.05:  # Higher chance to accept slightly worse moves
                    # Accept with small probability to escape local minima
                    best_r = new_r
                    best_x = new_x
                    best_y = new_y
        
        # Apply the best improvement found
        refined[i] = [best_x, best_y, best_r]
    
    # Strategy 2: Simultaneous optimization of all radii with smarter approach
    # Run a few rounds of global optimization
    for round_num in range(10):
        improved = True
        iteration_count = 0
        while improved and iteration_count < 30:
            improved = False
            iteration_count += 1
            
            # Try to improve all circles systematically
            for i in range(n):
                current_x, current_y, current_r = refined[i]
                
                # Try to increase radius while keeping constraints
                new_r = min(current_r + 0.002, 0.499)
                
                # Check if we can increase radius
                valid = True
                for j in range(n):
                    if i != j:
                        x, y, r = refined[j]
                        dist = math.sqrt((current_x - x)**2 + (current_y - y)**2)
                        if dist < new_r + r:
                            valid = False
                            break
                
                # Also check containment
                if valid and current_x - new_r >= 0 and current_x + new_r <= 1 and current_y - new_r >= 0 and current_y + new_r <= 1:
                    refined[i] = [current_x, current_y, new_r]
                    improved = True
    
    # Strategy 3: Final targeted optimization focusing on improving the sum
    # Run one final optimization pass focused on small adjustments
    for _ in range(1000):
        i = np.random.randint(0, n)
        current_x, current_y, current_r = refined[i]
        
        # Even smaller step size for final tuning
        step_size = 0.002
        dx = np.random.uniform(-step_size, step_size)
        dy = np.random.uniform(-step_size, step_size)
        dr = np.random.uniform(-step_size/8, step_size/8)
        
        new_x = current_x + dx
        new_y = current_y + dy
        new_r = current_r + dr
        
        # Ensure bounds
        new_x = np.clip(new_x, 0.001, 0.999)
        new_y = np.clip(new_y, 0.001, 0.999)
        new_r = np.clip(new_r, 0.001, 0.499)
        
        # Check constraints
        if check_constraints_single(refined, i, new_x, new_y, new_r):
            # Calculate change in objective
            old_sum = sum(circle[2] for circle in refined)
            new_sum = old_sum - current_r + new_r
            
            # Accept better solutions
            if new_sum > old_sum:
                refined[i] = [new_x, new_y, new_r]
    
    return refined

def check_constraints_single(circles, index, x, y, r):
    """Check if placing a circle at (x,y) with radius r violates any constraints"""
    # Check containment
    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
        return False
    
    # Check overlap with all other circles
    for i, circle in enumerate(circles):
        if i != index:
            cx, cy, cr = circle
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < r + cr:
                return False
    
    return True


# EVOLVE-BLOCK-END
