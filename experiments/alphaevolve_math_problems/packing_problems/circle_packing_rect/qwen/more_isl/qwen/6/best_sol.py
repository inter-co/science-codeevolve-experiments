# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import time
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses advanced hybrid approach combining physics-inspired initialization, multi-start optimization, 
    and constraint handling with spatial acceleration.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    n = 21
    
    # Focus on a more comprehensive set of aspect ratios that have shown success
    ratios = [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    
    best_sum = 0
    best_circles = None
    best_ratio = 1.0
    
    # Track time to respect 60-second limit
    start_time = time.time()
    
    # Multi-start optimization with more thorough exploration
    for ratio in ratios:
        if time.time() - start_time > 55:  # Leave buffer for final processing
            break
            
        width = 2 / (1 + ratio)
        height = 2 / (1 + 1/ratio)
        
        # Try multiple initialization strategies with more thorough exploration
        init_strategies = [
            lambda w, h, num: initialize_hexagonal_v2(w, h, num),
            lambda w, h, num: initialize_focused_hexagonal_v2(w, h, num),
            lambda w, h, num: initialize_grid_v2(w, h, num),
            lambda w, h, num: initialize_physics_based(w, h, num)
        ]
        
        # Try each initialization strategy multiple times with different seeds
        for init_strategy in init_strategies:
            if time.time() - start_time > 55:
                break
                
            # Try 4 different random starts for each initialization method
            for attempt in range(4):
                if time.time() - start_time > 55:
                    break
                    
                try:
                    # Set different random seed for each attempt
                    random.seed(42 + attempt * 100)
                    np.random.seed(42 + attempt * 100)
                    
                    circles = init_strategy(width, height, n)
                    
                    # Try optimization with different approaches
                    try:
                        # Try trust-constr first (often better for constrained problems)
                        tc_result = optimize_with_trust_constr(circles, width, height)
                        if tc_result is not None:
                            tc_sum = np.sum(tc_result[:, 2])
                            if tc_sum > best_sum:
                                best_sum = tc_sum
                                best_circles = tc_result.copy()
                                best_ratio = ratio
                    except Exception as tc_e:
                        pass
                    
                    # Try SLSQP with different settings as fallback
                    try:
                        sqp_result = optimize_with_slsqp_enhanced(circles, width, height)
                        if sqp_result is not None:
                            sqp_sum = np.sum(sqp_result[:, 2])
                            if sqp_sum > best_sum:
                                best_sum = sqp_sum
                                best_circles = sqp_result.copy()
                                best_ratio = ratio
                    except Exception as sqp_e:
                        pass
                        
                    # Try DE as last resort
                    try:
                        de_result = optimize_with_de_enhanced(circles, width, height)
                        if de_result is not None:
                            de_sum = np.sum(de_result[:, 2])
                            if de_sum > best_sum:
                                best_sum = de_sum
                                best_circles = de_result.copy()
                                best_ratio = ratio
                    except Exception as de_e:
                        pass
                    
                except Exception as e:
                    continue  # Skip failed attempts
    
    # Final intensive refinement with best found configuration
    if best_circles is not None and best_sum > 0:
        width = 2 / (1 + best_ratio)
        height = 2 / (1 + 1/best_ratio)
        
        # Try very aggressive optimization passes
        try:
            # High precision SLSQP optimization
            result1 = optimize_with_slsqp_very_precise(best_circles, width, height)
            sum1 = np.sum(result1[:, 2])
            
            # Trust-constr optimization with ultra-tight tolerances
            result2 = optimize_with_trust_constr_very_precise(best_circles, width, height)
            sum2 = np.sum(result2[:, 2])
            
            # Select the best of these two passes
            max_sum = max(sum1, sum2)
            if max_sum > best_sum:
                if max_sum == sum1:
                    return result1
                else:
                    return result2
        except:
            pass
    
    # Fallback to robust initialization if nothing works well
    if best_circles is None:
        width, height = 1.0, 1.0
        circles = initialize_hexagonal_v2(width, height, n)
        try:
            best_circles = optimize_with_trust_constr(circles, width, height)
        except Exception as e:
            best_circles = optimize_with_slsqp_enhanced(circles, width, height)
    
    return best_circles


def initialize_hexagonal_v2(width, height, n):
    """Improved hexagonal initialization with better packing density"""
    circles = np.zeros((n, 3))
    
    # Estimate based on area with better packing factor
    area_per_circle = (width * height) / n
    estimated_radius = np.sqrt(area_per_circle / np.pi) * 0.85  # Slightly smaller for better packing
    
    # Grid dimensions for hexagonal packing
    cols = max(2, int(np.sqrt(n * 1.3)))
    rows = max(2, int(np.ceil(n / cols)))
    
    if cols * rows < n:
        cols += 1
    
    # Hexagonal spacing
    row_spacing = estimated_radius * 2 * 0.866  # sqrt(3)/2
    col_spacing = estimated_radius * 1.5
    
    # Generate hexagonal pattern
    circle_idx = 0
    for i in range(rows):
        y_offset = i * row_spacing
        for j in range(cols):
            if circle_idx >= n:
                break
            x_offset = j * col_spacing + (i % 2) * (col_spacing / 2)
            
            # Center pattern properly
            x = x_offset + (width - col_spacing * (cols - 1)) / 2
            y = y_offset + (height - row_spacing * (rows - 1)) / 2
            
            # Ensure within bounds with safety margin
            if (x >= estimated_radius and x <= width - estimated_radius and 
                y >= estimated_radius and y <= height - estimated_radius):
                # Add some randomness to avoid perfect symmetry
                circles[circle_idx] = [x, y, estimated_radius * (0.9 + np.random.random() * 0.2)]
                circle_idx += 1
        if circle_idx >= n:
            break
    
    # Fill remaining positions with small circles
    for i in range(circle_idx, n):
        circles[i] = [
            width/2 + np.random.uniform(-0.1, 0.1),
            height/2 + np.random.uniform(-0.1, 0.1),
            estimated_radius * 0.2
        ]
    
    return circles


def initialize_focused_hexagonal_v2(width, height, n):
    """Even more focused hexagonal initialization"""
    circles = np.zeros((n, 3))
    
    # Very focused hexagonal packing
    sqrt_n = np.sqrt(n)
    cols = max(2, int(sqrt_n * 1.15))  # Even more horizontal bias
    rows = max(2, int(np.ceil(n / cols)))
    
    if cols * rows < n:
        cols += 1
    
    # Calculate spacing
    margin = 0.02
    available_width = width - 2 * margin
    available_height = height - 2 * margin
    
    if cols > 1:
        col_spacing = available_width / (cols - 0.5)
    else:
        col_spacing = available_width
    
    if rows > 1:
        row_spacing = available_height / (rows - 0.5)
    else:
        row_spacing = available_height
    
    # Maximum radius
    max_radius = min(col_spacing, row_spacing) / 2.0
    max_radius = min(max_radius, width/6, height/6)
    
    # Fill grid with hexagonal pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
                
            x_offset = j * col_spacing
            y_offset = i * row_spacing * 0.866
            
            if i % 2 == 1:
                x_offset += col_spacing / 2.0
            
            x = margin + x_offset + col_spacing/2
            y = margin + y_offset + row_spacing/2
            
            if x <= width - max_radius and y <= height - max_radius:
                circles[idx] = [x, y, max_radius * (0.95 + np.random.random() * 0.05)]
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with small circles
    for i in range(idx, n):
        circles[i] = [width/2, height/2, max_radius * 0.15]
    
    return circles


def initialize_grid_v2(width, height, n):
    """Improved grid initialization with better spacing"""
    circles = np.zeros((n, 3))
    
    # Create optimized grid layout
    sqrt_n = np.sqrt(n)
    cols = max(2, int(sqrt_n * 1.1))
    rows = max(2, int(np.ceil(n / cols)))
    
    # Adjust for exact count
    if cols * rows < n:
        cols += 1
    
    # Calculate spacing with safety margin
    margin = 0.02
    available_width = width - 2 * margin
    available_height = height - 2 * margin
    
    col_spacing = available_width / cols if cols > 0 else available_width
    row_spacing = available_height / rows if rows > 0 else available_height
    
    # Maximum radius based on spacing
    max_radius = min(col_spacing, row_spacing) / 2.0 * 0.95
    
    # Place circles in grid with slight randomness
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x = margin + col * col_spacing + col_spacing/2 + np.random.uniform(-0.01, 0.01)
            y = margin + row * row_spacing + row_spacing/2 + np.random.uniform(-0.01, 0.01)
            # Ensure within bounds
            x = max(max_radius, min(width - max_radius, x))
            y = max(max_radius, min(height - max_radius, y))
            circles[idx] = [x, y, max_radius * (0.8 + np.random.random() * 0.3)]
            idx += 1
        if idx >= n:
            break
    
    # Fill remaining circles
    for i in range(idx, n):
        circles[i] = [width/2, height/2, max_radius * 0.15]
    
    return circles


def initialize_physics_based(width, height, n):
    """Physics-inspired initialization using repulsion forces"""
    # Start with a hexagonal pattern
    circles = initialize_hexagonal_v2(width, height, n)
    
    # Simulate simple repulsion to distribute circles better
    for _ in range(150):  # More iterations for better distribution
        # Compute pairwise forces (simplified)
        for i in range(n):
            fx, fy = 0.0, 0.0
            for j in range(n):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Repulsive force when too close
                    if dist < (circles[i, 2] + circles[j, 2]) * 1.2:
                        if dist > 0.001:
                            force = 1.0 / (dist * dist + 0.01)
                            fx += force * dx / dist
                            fy += force * dy / dist
            
            # Apply small movement (with damping)
            circles[i, 0] += fx * 0.001
            circles[i, 1] += fy * 0.001
            
            # Keep within bounds
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], width - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], height - circles[i, 2])
    
    return circles


def optimize_with_trust_constr(initial_circles, width, height):
    """Use trust-constr optimization for better constrained problems"""
    n = len(initial_circles)
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        total_radius = 0
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
            total_radius += r
        
        # Return negative because we want to maximize
        return -total_radius
    
    # Constraint function: ensure no overlaps and all circles stay within bounds
    def constraint_func(params):
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
        
        positions = np.array(positions)
        radii = np.array(radii)
        
        constraints = []
        
        # Non-overlap constraints: distance >= sum of radii (with safety margin)
        distances = cdist(positions, positions)
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                # Add safety margin to prevent numerical issues
                constraints.append(dist - min_dist - 1e-8)
        
        # Boundary constraints: all circles must be within rectangle
        for i in range(n):
            x, y, r = positions[i][0], positions[i][1], radii[i]
            # Circle center must be at least radius away from edges
            constraints.append(x - r + 1e-8)  # left bound
            constraints.append(width - x - r + 1e-8)  # right bound
            constraints.append(y - r + 1e-8)  # bottom bound
            constraints.append(height - y - r + 1e-8)  # top bound
        
        return np.array(constraints)
    
    # Set up bounds: [x_min, x_max, y_min, y_max, r_min, r_max] for each circle
    bounds = []
    max_radius = min(width, height) / 2.0
    for i in range(n):
        # x bounds: [radius, width-radius]
        bounds.extend([(1e-6, width - 1e-6)])
        # y bounds: [radius, height-radius] 
        bounds.extend([(1e-6, height - 1e-6)])
        # r bounds: [1e-6, min(width, height)/2]
        bounds.extend([(1e-6, max_radius)])
    
    # Initial guess from our configuration
    initial_guess = []
    for i in range(n):
        x, y, r = initial_circles[i]
        initial_guess.extend([x, y, r])
    
    # Apply optimization with bounds and constraints
    try:
        # Use trust-constr method which is often better for constrained problems
        result = minimize(
            objective,
            initial_guess,
            method='trust-constr',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 2500, 'ftol': 1e-9, 'gtol': 1e-9, 'verbose': 0}
        )
        
        if result.success:
            # Extract final solution
            final_positions = []
            final_radii = []
            for i in range(n):
                idx = i * 3
                x, y, r = result.x[idx], result.x[idx+1], result.x[idx+2]
                final_positions.append([x, y])
                final_radii.append(r)
            
            # Update circles array
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_positions[i][0], final_positions[i][1], final_radii[i]]
            return circles
        else:
            return initial_circles
            
    except Exception as e:
        # If optimization fails for any reason, return the initial arrangement
        return initial_circles


def optimize_with_trust_constr_very_precise(initial_circles, width, height):
    """Ultra precise trust-constr optimization"""
    n = len(initial_circles)
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        total_radius = 0
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
            total_radius += r
        
        # Return negative because we want to maximize
        return -total_radius
    
    # Constraint function: ensure no overlaps and all circles stay within bounds
    def constraint_func(params):
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
        
        positions = np.array(positions)
        radii = np.array(radii)
        
        constraints = []
        
        # Non-overlap constraints: distance >= sum of radii (with safety margin)
        distances = cdist(positions, positions)
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                # Add safety margin to prevent numerical issues
                constraints.append(dist - min_dist - 1e-10)
        
        # Boundary constraints: all circles must be within rectangle
        for i in range(n):
            x, y, r = positions[i][0], positions[i][1], radii[i]
            # Circle center must be at least radius away from edges
            constraints.append(x - r + 1e-10)  # left bound
            constraints.append(width - x - r + 1e-10)  # right bound
            constraints.append(y - r + 1e-10)  # bottom bound
            constraints.append(height - y - r + 1e-10)  # top bound
        
        return np.array(constraints)
    
    # Set up bounds: [x_min, x_max, y_min, y_max, r_min, r_max] for each circle
    bounds = []
    max_radius = min(width, height) / 2.0
    for i in range(n):
        # x bounds: [radius, width-radius]
        bounds.extend([(1e-6, width - 1e-6)])
        # y bounds: [radius, height-radius] 
        bounds.extend([(1e-6, height - 1e-6)])
        # r bounds: [1e-6, min(width, height)/2]
        bounds.extend([(1e-6, max_radius)])
    
    # Initial guess from our configuration
    initial_guess = []
    for i in range(n):
        x, y, r = initial_circles[i]
        initial_guess.extend([x, y, r])
    
    # Apply optimization with bounds and constraints
    try:
        # Use trust-constr method with ultra-precise settings
        result = minimize(
            objective,
            initial_guess,
            method='trust-constr',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-12, 'verbose': 0}
        )
        
        if result.success:
            # Extract final solution
            final_positions = []
            final_radii = []
            for i in range(n):
                idx = i * 3
                x, y, r = result.x[idx], result.x[idx+1], result.x[idx+2]
                final_positions.append([x, y])
                final_radii.append(r)
            
            # Update circles array
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_positions[i][0], final_positions[i][1], final_radii[i]]
            return circles
        else:
            return initial_circles
            
    except Exception as e:
        # If optimization fails for any reason, return the initial arrangement
        return initial_circles


def optimize_with_slsqp_enhanced(initial_circles, width, height):
    """Use SLSQP for local optimization with enhanced settings"""
    n = len(initial_circles)
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        total_radius = 0
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
            total_radius += r
        
        # Return negative because we want to maximize
        return -total_radius
    
    # Constraint function: ensure no overlaps and all circles stay within bounds
    def constraint_func(params):
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
        
        positions = np.array(positions)
        radii = np.array(radii)
        
        constraints = []
        
        # Non-overlap constraints: distance >= sum of radii (with safety margin)
        distances = cdist(positions, positions)
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                # Add safety margin
                constraints.append(dist - min_dist - 1e-8)
        
        # Boundary constraints: all circles must be within rectangle
        for i in range(n):
            x, y, r = positions[i][0], positions[i][1], radii[i]
            # Circle center must be at least radius away from edges
            constraints.append(x - r + 1e-8)  # left bound
            constraints.append(width - x - r + 1e-8)  # right bound
            constraints.append(y - r + 1e-8)  # bottom bound
            constraints.append(height - y - r + 1e-8)  # top bound
        
        return np.array(constraints)
    
    # Set up bounds: [x_min, x_max, y_min, y_max, r_min, r_max] for each circle
    bounds = []
    max_radius = min(width, height) / 2.0
    for i in range(n):
        # x bounds: [radius, width-radius]
        bounds.extend([(1e-6, width - 1e-6)])
        # y bounds: [radius, height-radius] 
        bounds.extend([(1e-6, height - 1e-6)])
        # r bounds: [1e-6, min(width, height)/2]
        bounds.extend([(1e-6, max_radius)])
    
    # Initial guess from our configuration
    initial_guess = []
    for i in range(n):
        x, y, r = initial_circles[i]
        initial_guess.extend([x, y, r])
    
    # Apply optimization with bounds and constraints
    try:
        # Use SLSQP method with more iterations and tighter tolerance
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 3000, 'ftol': 1e-9, 'gtol': 1e-9, 'disp': False}
        )
        
        if result.success:
            # Extract final solution
            final_positions = []
            final_radii = []
            for i in range(n):
                idx = i * 3
                x, y, r = result.x[idx], result.x[idx+1], result.x[idx+2]
                final_positions.append([x, y])
                final_radii.append(r)
            
            # Update circles array
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_positions[i][0], final_positions[i][1], final_radii[i]]
            return circles
        else:
            return initial_circles
            
    except Exception as e:
        # If optimization fails for any reason, return the initial arrangement
        return initial_circles


def optimize_with_slsqp_very_precise(initial_circles, width, height):
    """Ultra precise SLSQP optimization"""
    n = len(initial_circles)
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        total_radius = 0
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
            total_radius += r
        
        # Return negative because we want to maximize
        return -total_radius
    
    # Constraint function: ensure no overlaps and all circles stay within bounds
    def constraint_func(params):
        positions = []
        radii = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = params[idx], params[idx+1], params[idx+2]
            positions.append([x, y])
            radii.append(r)
        
        positions = np.array(positions)
        radii = np.array(radii)
        
        constraints = []
        
        # Non-overlap constraints: distance >= sum of radii (with safety margin)
        distances = cdist(positions, positions)
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                min_dist = radii[i] + radii[j]
                # Add safety margin
                constraints.append(dist - min_dist - 1e-10)
        
        # Boundary constraints: all circles must be within rectangle
        for i in range(n):
            x, y, r = positions[i][0], positions[i][1], radii[i]
            # Circle center must be at least radius away from edges
            constraints.append(x - r + 1e-10)  # left bound
            constraints.append(width - x - r + 1e-10)  # right bound
            constraints.append(y - r + 1e-10)  # bottom bound
            constraints.append(height - y - r + 1e-10)  # top bound
        
        return np.array(constraints)
    
    # Set up bounds: [x_min, x_max, y_min, y_max, r_min, r_max] for each circle
    bounds = []
    max_radius = min(width, height) / 2.0
    for i in range(n):
        # x bounds: [radius, width-radius]
        bounds.extend([(1e-6, width - 1e-6)])
        # y bounds: [radius, height-radius] 
        bounds.extend([(1e-6, height - 1e-6)])
        # r bounds: [1e-6, min(width, height)/2]
        bounds.extend([(1e-6, max_radius)])
    
    # Initial guess from our configuration
    initial_guess = []
    for i in range(n):
        x, y, r = initial_circles[i]
        initial_guess.extend([x, y, r])
    
    # Apply optimization with bounds and constraints
    try:
        # Use SLSQP method with ultra-precise settings
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        
        if result.success:
            # Extract final solution
            final_positions = []
            final_radii = []
            for i in range(n):
                idx = i * 3
                x, y, r = result.x[idx], result.x[idx+1], result.x[idx+2]
                final_positions.append([x, y])
                final_radii.append(r)
            
            # Update circles array
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [final_positions[i][0], final_positions[i][1], final_radii[i]]
            return circles
        else:
            return initial_circles
            
    except Exception as e:
        # If optimization fails for any reason, return the initial arrangement
        return initial_circles


def optimize_with_de_enhanced(initial_circles, width, height):
    """Enhanced differential evolution for global optimization"""
    from scipy.optimize import differential_evolution
    
    n = len(initial_circles)
    
    # Flatten initial parameters
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Define bounds for optimization
    bounds = []
    max_radius = min(width, height) / 2.0
    for i in range(n):
        bounds.extend([(0.001, width - 0.001), (0.001, height - 0.001), (0.001, max_radius)])
    
    def objective(params):
        # Sum of radii (negative because we're minimizing)
        total_radius = 0
        for i in range(n):
            total_radius += params[3*i + 2]
        return -total_radius
    
    def constraint_bounds(params):
        """Ensure circles are within bounds"""
        constraints = []
        for i in range(n):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            # Add safety margins
            constraints.append(x - r + 1e-8)  # left boundary
            constraints.append(y - r + 1e-8)  # bottom boundary
            constraints.append(width - x - r + 1e-8)  # right boundary
            constraints.append(height - y - r + 1e-8)  # top boundary
        return np.array(constraints)
    
    def constraint_overlaps(params):
        """Ensure no overlaps between circles"""
        constraints = []
        for i in range(n):
            x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
            for j in range(i+1, n):
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                # Add safety margin
                constraints.append(distance_sq - min_distance_sq - 1e-10)
        return np.array(constraints)
    
    # Define constraints
    constraints = [
        {'type': 'ineq', 'fun': constraint_bounds},
        {'type': 'ineq', 'fun': constraint_overlaps}
    ]
    
    # Run differential evolution with enhanced parameters
    result = differential_evolution(
        objective,
        bounds,
        constraints=constraints,
        seed=42,
        maxiter=1000,  # Increased iterations for better convergence
        popsize=50,   # Larger population size
        mutation=(0.8, 1),  # Different mutation strategy
        recombination=0.9,  # Higher recombination rate
        atol=1e-9,   # Tighter absolute tolerance
        rtol=1e-9    # Tighter relative tolerance
    )
    
    if result.success:
        final_params = result.x
        optimized_circles = np.zeros((n, 3))
        for i in range(n):
            optimized_circles[i] = [final_params[3*i], final_params[3*i+1], final_params[3*i+2]]
        return optimized_circles
    
    raise Exception("Enhanced differential evolution failed")


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
