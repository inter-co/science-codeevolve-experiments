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
    
    # Use multiple optimization passes with increasing intensity
    circles = optimize_with_de_multiple_passes(circles)
    
    # Final aggressive refinement with better local search
    circles = refine_circles_improved(circles)
    
    return circles

def initialize_better_pack(n):
    """Initialize circles with a more sophisticated approach"""
    circles = np.zeros((n, 3))
    
    # Use a more systematic approach inspired by known optimal packings
    # Try to place circles in a pattern that maximizes density
    
    # For 32 circles, let's try a 5x7 grid pattern with strategic adjustments
    rows = 5
    cols = 7
    
    # Calculate spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Initial radius estimate
    max_radius = min(spacing_x, spacing_y) * 0.4
    
    # Place in a grid pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Add small random perturbation for better optimization
            x += np.random.uniform(-spacing_x/15, spacing_x/15)
            y += np.random.uniform(-spacing_y/15, spacing_y/15)
            
            # Ensure within bounds
            x = np.clip(x, max_radius, 1 - max_radius)
            y = np.clip(y, max_radius, 1 - max_radius)
            
            circles[idx] = [x, y, max_radius]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with more careful placement
    for i in range(idx, n):
        # Try to find good placements that maximize potential radii
        best_placement = None
        best_radius = 0
        
        # Try fewer but smarter random placements
        for _ in range(500):  # Fewer attempts but more focused
            x = np.random.uniform(max_radius, 1 - max_radius)
            y = np.random.uniform(max_radius, 1 - max_radius)
            
            # Calculate maximum possible radius at this position
            max_possible_radius = min(x, 1-x, y, 1-y)
            
            # Find minimum distance to existing circles using KDTree for efficiency
            if len(circles[:i]) > 0:
                points = np.array([[c[0], c[1]] for c in circles[:i]])
                tree = cKDTree(points)
                distances, _ = tree.query([x, y], k=1)
                min_dist = distances[0] if distances.size > 0 else float('inf')
                
                # If there are existing circles, we can't go larger than half the minimum distance
                if min_dist < float('inf') and min_dist > 0:
                    max_possible_radius = min(max_possible_radius, min_dist/2)
            
            # Prefer larger radii
            if max_possible_radius > best_radius:
                best_radius = max_possible_radius
                best_placement = (x, y, max_possible_radius)
        
        if best_placement is not None:
            circles[i] = best_placement
    
    return circles

def optimize_with_de_multiple_passes(circles):
    """Run multiple differential evolution passes with increasing intensity"""
    n = len(circles)
    
    # Define bounds for optimization - tighter bounds for better convergence
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
        
        # Check overlap constraints using spatial data structure for efficiency
        try:
            # Build KDTree for efficient neighbor searches
            points = np.array([[c[0], c[1]] for c in temp_circles])
            tree = cKDTree(points)
            
            # Check overlaps efficiently
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
    
    # Multiple passes with different settings for better exploration
    current_circles = circles.copy()
    
    # Pass 1: Coarse optimization with more iterations
    try:
        result = differential_evolution(
            objective,
            bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            seed=42,
            maxiter=300,
            popsize=20,
            mutation=(0.5, 1),
            recombination=0.7,
            disp=False
        )
        if result.success:
            optimized_params = result.x
            result_circles = np.zeros((n, 3))
            for i in range(n):
                result_circles[i] = [optimized_params[3*i], optimized_params[3*i+1], optimized_params[3*i+2]]
            current_circles = result_circles
    except Exception as e:
        print(f"DE pass 1 failed: {e}")
        pass
    
    # Pass 2: Medium optimization with higher population size
    try:
        result = differential_evolution(
            objective,
            bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            seed=42,
            maxiter=500,
            popsize=30,
            mutation=(0.8, 1),
            recombination=0.8,
            disp=False
        )
        if result.success:
            optimized_params = result.x
            result_circles = np.zeros((n, 3))
            for i in range(n):
                result_circles[i] = [optimized_params[3*i], optimized_params[3*i+1], optimized_params[3*i+2]]
            current_circles = result_circles
    except Exception as e:
        print(f"DE pass 2 failed: {e}")
        pass
    
    # Pass 3: Fine optimization with even more iterations
    try:
        result = differential_evolution(
            objective,
            bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            seed=42,
            maxiter=800,
            popsize=35,
            mutation=(0.9, 1),
            recombination=0.9,
            disp=False
        )
        if result.success:
            optimized_params = result.x
            result_circles = np.zeros((n, 3))
            for i in range(n):
                result_circles[i] = [optimized_params[3*i], optimized_params[3*i+1], optimized_params[3*i+2]]
            current_circles = result_circles
    except Exception as e:
        print(f"DE pass 3 failed: {e}")
        pass
    
    return current_circles

def refine_circles_improved(circles):
    """Apply improved refinement with better local search strategies"""
    n = len(circles)
    
    # Create a copy to work with
    refined = circles.copy()
    
    # Strategy 1: Simulated Annealing style local search with better neighborhood
    temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 0.0001
    
    # Use a more targeted approach for better results
    for iteration in range(8000):  # More iterations for better search
        # Pick a random circle to adjust
        i = np.random.randint(0, n)
        
        # Store current state
        current_x, current_y, current_r = refined[i]
        
        # Try to improve this circle's configuration
        best_r = current_r
        best_x, best_y = current_x, current_y
        
        # Adaptive step sizes based on current temperature and circle size
        step_size = max(0.0001, temperature * 0.02)
        
        # Generate moves with better distribution
        moves = []
        for _ in range(100):  # More moves per iteration
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            dr = np.random.uniform(-step_size/3, step_size/3)
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
                
                # Accept better solutions or sometimes worse ones based on temperature
                delta = new_sum - old_sum
                if delta > 0 or np.random.random() < np.exp(delta / temperature):
                    if new_sum > old_sum:
                        best_r = new_r
                        best_x = new_x
                        best_y = new_y
        
        # Apply the best improvement found
        refined[i] = [best_x, best_y, best_r]
        
        # Cool down temperature
        temperature = max(min_temperature, temperature * cooling_rate)
    
    # Strategy 2: Global optimization approach - try to improve all circles simultaneously
    # This uses a more direct approach to see if we can globally improve the configuration
    for _ in range(2000):  # Additional global optimization steps
        # Try to improve each circle by a small amount
        improved = False
        for i in range(n):
            current_x, current_y, current_r = refined[i]
            
            # Try to slightly increase radius if possible
            new_r = min(current_r + 0.0005, 0.499)
            
            # Check if we can increase radius without violating constraints
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
        
        # If no improvements were made, stop early
        if not improved:
            break
    
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
