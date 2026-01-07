# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import math
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a sophisticated hybrid approach combining multiple optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with an extremely careful approach using mathematical insights
    circles = initialize_mathematical_pack(n)
    
    # Use aggressive optimization with multiple restarts
    circles = optimize_with_restart_strategy(circles)
    
    # Apply very aggressive refinement
    circles = refine_extremely_aggressive(circles)
    
    # Final boost with specialized optimization
    circles = final_boost_optimization(circles)
    
    # Additional fine-tuning with local search
    circles = local_search_fine_tuning(circles)
    
    return circles

def initialize_mathematical_pack(n):
    """Initialize circles using mathematical principles for better packing"""
    circles = np.zeros((n, 3))
    
    # Strategy: Create a pattern that's designed to maximize space utilization
    # Use a hexagonal-like arrangement but adapted for 32 circles in a square
    
    # Try a 6x6 grid pattern (36 positions) but only use 32
    rows = 6
    cols = 6
    
    # Calculate spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Start with a good initial radius - slightly less than what would fill perfectly
    max_radius = min(spacing_x, spacing_y) / 2 * 0.8
    
    # Place in grid pattern with alternating offsets for better packing
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Add alternating offset for better packing
            if i % 2 == 0:
                x += spacing_x * 0.25
            else:
                x -= spacing_x * 0.25
                
            # Add some randomness to break perfect symmetry
            x += np.random.uniform(-spacing_x/10, spacing_x/10)
            y += np.random.uniform(-spacing_y/10, spacing_y/10)
            
            # Ensure within bounds
            x = np.clip(x, max_radius, 1 - max_radius)
            y = np.clip(y, max_radius, 1 - max_radius)
            
            circles[idx] = [x, y, max_radius]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with strategic placement
    for i in range(idx, n):
        # Try to place circles in the "gaps" of the existing pattern
        best_placement = None
        best_radius = 0
        
        # Try many placements to find the best one
        for attempt in range(2000):
            x = np.random.uniform(max_radius, 1 - max_radius)
            y = np.random.uniform(max_radius, 1 - max_radius)
            
            # Calculate maximum possible radius at this point
            max_radius_at_pos = min(x, 1-x, y, 1-y)
            
            # Find minimum distance to existing circles
            min_dist = float('inf')
            for k in range(i):
                cx, cy, cr = circles[k]
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                min_dist = min(min_dist, dist)
            
            # If there are existing circles, we can't go larger than half the minimum distance
            if min_dist < float('inf') and min_dist > 0:
                max_radius_at_pos = min(max_radius_at_pos, min_dist/2)
            
            # Prefer positions with good potential (larger possible radius)
            if max_radius_at_pos > best_radius:
                best_radius = max_radius_at_pos
                best_placement = (x, y, max_radius_at_pos)
        
        if best_placement is not None:
            circles[i] = best_placement
    
    return circles

def optimize_with_restart_strategy(circles):
    """Use differential evolution with restart strategy for better results"""
    n = len(circles)
    
    # Define bounds for optimization
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
    
    # Use multiple restarts with different settings to find better solutions
    best_result = None
    best_sum = -float('inf')
    
    # Run multiple optimization attempts with different configurations
    for restart in range(7):  # More restarts
        try:
            # Different configurations for each restart
            config = {
                'maxiter': 1200 if restart < 5 else 800,
                'popsize': 40 if restart < 3 else 30,
                'mutation': (0.95, 1) if restart < 2 else (0.85, 1),
                'recombination': 0.98 if restart < 2 else 0.95,
                'seed': 42 + restart
            }
            
            result = differential_evolution(
                objective,
                bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                **config,
                disp=False
            )
            
            if result.success:
                optimized_params = result.x
                result_circles = np.zeros((n, 3))
                for i in range(n):
                    result_circles[i] = [optimized_params[3*i], optimized_params[3*i+1], optimized_params[3*i+2]]
                
                # Calculate sum of radii
                current_sum = sum(circle[2] for circle in result_circles)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_circles
                    
        except Exception as e:
            continue
    
    # Return the best result found
    if best_result is not None:
        return best_result
    else:
        return circles

def refine_extremely_aggressive(circles):
    """Apply extremely aggressive refinement with multiple strategies"""
    n = len(circles)
    
    # Create a copy to work with
    refined = circles.copy()
    
    # Phase 1: Very intensive local optimization with adaptive steps
    for iteration in range(6000):  # Even more iterations
        # Pick a random circle to adjust
        i = np.random.randint(0, n)
        
        # Store current state
        current_x, current_y, current_r = refined[i]
        
        # Adaptive step size based on iteration progress
        base_step = 0.03  # Larger steps initially
        step_size = max(0.001, base_step * (1 - iteration/6000))
        
        # Try many moves for better exploration
        best_r = current_r
        best_x, best_y = current_x, current_y
        best_improvement = 0
        
        # Generate diverse moves
        moves = []
        for _ in range(120):  # Even more moves
            # Mix of small, medium and large moves
            move_type = np.random.choice(['small', 'medium', 'large'], p=[0.25, 0.5, 0.25])
            if move_type == 'small':
                dx = np.random.uniform(-step_size/5, step_size/5)
                dy = np.random.uniform(-step_size/5, step_size/5)
                dr = np.random.uniform(-step_size/10, step_size/10)
            elif move_type == 'medium':
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                dr = np.random.uniform(-step_size/5, step_size/5)
            else:  # large
                dx = np.random.uniform(-step_size*4, step_size*4)
                dy = np.random.uniform(-step_size*4, step_size*4)
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
                improvement = new_sum - old_sum
                
                # Accept better solutions
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_r = new_r
                    best_x = new_x
                    best_y = new_y
        
        # Apply the best improvement found
        if best_improvement > 0:
            refined[i] = [best_x, best_y, best_r]
    
    # Phase 2: Systematic global improvement with multiple passes
    for pass_num in range(4):  # More passes
        improved = True
        iteration_count = 0
        while improved and iteration_count < 300:  # More iterations
            improved = False
            iteration_count += 1
            
            # Try to increase all radii where possible
            for i in range(n):
                current_x, current_y, current_r = refined[i]
                
                # Try to increase radius more aggressively
                new_r = min(current_r + 0.006, 0.499)  # Slightly larger increments
                
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
                if (valid and current_x - new_r >= 0 and current_x + new_r <= 1 and 
                    current_y - new_r >= 0 and current_y + new_r <= 1):
                    refined[i] = [current_x, current_y, new_r]
                    improved = True
    
    # Phase 3: Specialized refinement for tight clusters
    # Look for circles that are very close and try to separate them
    for _ in range(1500):  # More iterations
        # Pick two random circles
        i, j = np.random.choice(n, 2, replace=False)
        
        # Get current positions and radii
        x1, y1, r1 = refined[i]
        x2, y2, r2 = refined[j]
        
        # Calculate distance between centers
        dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # If they're too close, try to adjust
        if dist < r1 + r2:
            # Try to move one or both to increase separation
            # Move the first circle slightly away from the second
            if dist > 0.001:  # Avoid division by zero
                dx = (x1 - x2) / dist * 0.003
                dy = (y1 - y2) / dist * 0.003
                
                new_x1 = x1 + dx
                new_y1 = y1 + dy
                new_r1 = r1
                
                # Ensure new position is valid
                if (new_x1 - new_r1 >= 0 and new_x1 + new_r1 <= 1 and 
                    new_y1 - new_r1 >= 0 and new_y1 + new_r1 <= 1):
                    # Check if this improves the total sum (we want to keep sum high)
                    old_sum = sum(circle[2] for circle in refined)
                    new_sum = old_sum - r1 + new_r1
                    if new_sum > old_sum:
                        refined[i] = [new_x1, new_y1, new_r1]
    
    return refined

def final_boost_optimization(circles):
    """Apply a final optimization boost with focused techniques"""
    n = len(circles)
    
    # Create a copy to work with
    refined = circles.copy()
    
    # Try a few rounds of very focused optimization on specific problematic areas
    for _ in range(2000):
        # Focus on the circles that are most constrained (near edges or near others)
        # Pick a circle that's either near an edge or near another circle
        i = np.random.randint(0, n)
        
        current_x, current_y, current_r = refined[i]
        
        # Try to slightly adjust this circle
        # Use a smaller step size for fine tuning
        step_size = 0.005
        
        # Try a few specific moves
        moves = [
            (0, 0, 0.001),   # Slight radius increase
            (0, 0, -0.001),  # Slight radius decrease
            (step_size/2, 0, 0),   # Small x shift
            (-step_size/2, 0, 0),  # Small x shift
            (0, step_size/2, 0),   # Small y shift
            (0, -step_size/2, 0),  # Small y shift
        ]
        
        best_r = current_r
        best_x, best_y = current_x, current_y
        best_improvement = 0
        
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
                improvement = new_sum - old_sum
                
                # Accept better solutions
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_r = new_r
                    best_x = new_x
                    best_y = new_y
        
        # Apply the best improvement found
        if best_improvement > 0:
            refined[i] = [best_x, best_y, best_r]
    
    return refined

def local_search_fine_tuning(circles):
    """Perform additional fine-tuning using local search techniques"""
    n = len(circles)
    
    # Create a copy to work with
    refined = circles.copy()
    
    # Perform a series of targeted local optimizations
    # Focus on circles that are near the boundary or in tight clusters
    
    # Try several rounds of focused local search
    for round_num in range(1000):
        # Select circles based on proximity to boundaries or other circles
        # Prioritize those near edges or with small gaps
        i = np.random.randint(0, n)
        
        current_x, current_y, current_r = refined[i]
        
        # Very fine-grained moves
        step_size = 0.001
        
        # Try many very small moves
        moves = []
        for _ in range(50):
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            dr = np.random.uniform(-step_size/2, step_size/2)
            moves.append((dx, dy, dr))
        
        best_r = current_r
        best_x, best_y = current_x, current_y
        best_improvement = 0
        
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
                improvement = new_sum - old_sum
                
                # Accept better solutions
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_r = new_r
                    best_x = new_x
                    best_y = new_y
        
        # Apply the best improvement found
        if best_improvement > 0:
            refined[i] = [best_x, best_y, best_r]
    
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
