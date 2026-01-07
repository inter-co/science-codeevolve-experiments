# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import random
from itertools import combinations
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-phase approach with improved initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    start_time = time.time()
    
    # Phase 1: Advanced initialization using multiple strategies
    def generate_initial_configuration():
        """Generate a better initial configuration using multiple strategies"""
        best_config = None
        best_sum = 0
        
        # Strategy 1: Hexagonal packing pattern for better density
        def hexagonal_strategy():
            positions = []
            radii = []
            
            # Try to create a hexagonal pattern with decreasing ring sizes
            max_radius = 0.15
            ring_sizes = [1, 6, 12, 18]  # Number of circles per ring
            ring_radii = [0.0, 0.1, 0.2, 0.3]  # Ring distances from center
            
            center_x, center_y = 0.5, 0.5
            ring_idx = 0
            
            for ring_size, ring_radius in zip(ring_sizes, ring_radii):
                if len(positions) >= n:
                    break
                    
                # Place circles in this ring
                if ring_idx == 0:
                    # Center circle
                    positions.append([center_x, center_y])
                    radii.append(min(max_radius, 0.12))
                else:
                    # Hexagonal ring
                    points_per_ring = min(ring_size, n - len(positions))
                    angle_step = 2 * np.pi / points_per_ring
                    
                    for i in range(points_per_ring):
                        if len(positions) >= n:
                            break
                        angle = i * angle_step
                        x = center_x + ring_radius * np.cos(angle)
                        y = center_y + ring_radius * np.sin(angle)
                        
                        # Ensure within bounds
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            positions.append([x, y])
                            # Set radius based on proximity to boundaries
                            max_r = min(x, 1-x, y, 1-y)
                            r = min(max_r, max_radius)
                            radii.append(r)
                
                ring_idx += 1
            
            # Fill remaining positions with grid if needed
            if len(positions) < n:
                grid_size = int(np.ceil(np.sqrt(n - len(positions))))
                spacing_x = 1.0 / (grid_size + 2)
                spacing_y = 1.0 / (grid_size + 2)
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        if len(positions) >= n:
                            break
                        x = spacing_x * (j + 1)
                        y = spacing_y * (i + 1)
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            positions.append([x, y])
                            max_r = min(x, 1-x, y, 1-y)
                            r = min(max_r, 0.08)
                            radii.append(r)
            
            positions = positions[:n]
            radii = radii[:n]
            
            return positions, radii
        
        # Strategy 2: Grid with adaptive spacing
        def grid_adaptive_strategy():
            positions = []
            radii = []
            
            # Create a refined grid
            grid_size = int(np.ceil(np.sqrt(n)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            # Add some randomness to avoid perfect grid artifacts
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(positions) >= n:
                        break
                    # Add small random offset
                    offset_x = (random.random() - 0.5) * spacing_x * 0.3
                    offset_y = (random.random() - 0.5) * spacing_y * 0.3
                    x = spacing_x * (j + 1) + offset_x
                    y = spacing_y * (i + 1) + offset_y
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        positions.append([x, y])
            
            positions = positions[:n]
            
            # Set initial radii with better distribution
            for i, (x, y) in enumerate(positions):
                max_r = min(x, 1-x, y, 1-y)
                # Distribute radii more evenly
                r = min(max_r, 0.12 + (i % 4) * 0.01)
                radii.append(r)
            
            return positions, radii
        
        # Strategy 3: Optimized random placement with spatial rejection
        def optimized_random_strategy():
            positions = []
            radii = []
            
            # Generate positions with better spatial distribution
            max_attempts = 5000
            attempts = 0
            
            # Start with a few well-distributed points
            initial_points = 8
            for i in range(initial_points):
                x = random.random()
                y = random.random()
                positions.append([x, y])
            
            # Then fill with more points using rejection sampling
            while len(positions) < n and attempts < max_attempts:
                x = random.random()
                y = random.random()
                
                # Check if it's far enough from existing positions
                valid = True
                for px, py in positions:
                    if np.sqrt((x - px)**2 + (y - py)**2) < 0.06:
                        valid = False
                        break
                
                if valid and 0 <= x <= 1 and 0 <= y <= 1:
                    positions.append([x, y])
                
                attempts += 1
            
            # Fill remaining positions with grid if needed
            if len(positions) < n:
                grid_size = int(np.ceil(np.sqrt(n - len(positions))))
                spacing_x = 1.0 / (grid_size + 2)
                spacing_y = 1.0 / (grid_size + 2)
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        if len(positions) >= n:
                            break
                        x = spacing_x * (j + 1)
                        y = spacing_y * (i + 1)
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            positions.append([x, y])
            
            positions = positions[:n]
            
            # Set initial radii with better distribution
            for i, (x, y) in enumerate(positions):
                max_r = min(x, 1-x, y, 1-y)
                # Distribute radii to avoid getting stuck in local minima
                r = min(max_r, 0.1 + (i % 6) * 0.015)
                radii.append(r)
            
            return positions, radii
        
        strategies = [hexagonal_strategy, grid_adaptive_strategy, optimized_random_strategy]
        
        for strategy in strategies:
            try:
                positions, radii = strategy()
                
                # Validate that configuration is feasible
                valid = True
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                     (positions[i][1] - positions[j][1])**2)
                        if dist < radii[i] + radii[j]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    current_sum = sum(radii)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_config = (positions, radii)
            except Exception as e:
                continue
        
        # Fallback to simple grid if nothing worked
        if best_config is None:
            positions = []
            radii = []
            grid_size = int(np.ceil(np.sqrt(n)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(positions) >= n:
                        break
                    x = spacing_x * (j + 1)
                    y = spacing_y * (i + 1)
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        positions.append([x, y])
            
            positions = positions[:n]
            for i, (x, y) in enumerate(positions):
                max_r = min(x, 1-x, y, 1-y)
                r = max_r * 0.12
                radii.append(r)
                
            best_config = (positions, radii)
            
        return best_config
    
    # Generate initial configuration
    initial_positions, initial_radii = generate_initial_configuration()
    
    # Phase 2: Set up optimization variables
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
    initial_vars = []
    for i in range(n):
        x, y = initial_positions[i]
        r = initial_radii[i]
        initial_vars.extend([x, y, r])
    
    # Phase 3: Define constraint functions with improved handling
    def contain_constraints(vars):
        """Ensure all circles are contained in the unit square"""
        cons = []
        for i in range(n):
            idx = 3*i
            x, y, r = vars[idx], vars[idx+1], vars[idx+2]
            # Circle must be contained in unit square
            cons.append(x - r)  # x - r >= 0
            cons.append(1 - x - r)  # 1 - x - r >= 0
            cons.append(y - r)  # y - r >= 0
            cons.append(1 - y - r)  # 1 - y - r >= 0
        return np.array(cons)
    
    def overlap_constraints(vars):
        """Ensure no overlapping circles"""
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                idx_i = 3*i
                idx_j = 3*j
                x_i, y_i, r_i = vars[idx_i], vars[idx_i+1], vars[idx_i+2]
                x_j, y_j, r_j = vars[idx_j], vars[idx_j+1], vars[idx_j+2]
                
                # Distance constraint: d >= r_i + r_j
                dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                dist = np.sqrt(dist_sq)
                # We want dist >= r_i + r_j, so we enforce: dist - r_i - r_j >= 0
                cons.append(dist - r_i - r_j)
        return np.array(cons)
    
    # Phase 4: Define objective function
    def objective(vars):
        """Maximize sum of radii (minimize negative sum)"""
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i + 2]  # radius is third component
        return -total_radius  # Negative because we minimize
    
    # Phase 5: Run optimization with enhanced approach
    try:
        # Define bounds for variables
        bounds = []
        for i in range(n):
            # x bounds: [r, 1-r] 
            bounds.append((0.001, 0.999))  # x coordinate
            bounds.append((0.001, 0.999))  # y coordinate
            bounds.append((0.001, 0.499))  # radius (limit to prevent too large circles)
        
        # Create constraint dictionaries
        constraints = [
            {'type': 'ineq', 'fun': contain_constraints},
            {'type': 'ineq', 'fun': overlap_constraints}
        ]
        
        # Try multiple optimization methods with different starting points
        methods_to_try = ['trust-constr', 'SLSQP']
        best_result = None
        best_value = float('-inf')
        
        # Try optimization with different starting points
        for method in methods_to_try:
            try:
                # First try with original initial point
                result = minimize(
                    objective,
                    initial_vars,
                    method=method,
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 3000, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                if result.success:
                    current_sum = -objective(result.x)
                    if current_sum > best_value:
                        best_value = current_sum
                        best_result = result
                        
            except Exception as e:
                continue
        
        # If we found a good result, use it; otherwise fall back to initial
        if best_result is not None and best_result.success:
            final_vars = best_result.x
        else:
            # Fall back to initial configuration with significant refinement
            final_vars = initial_vars.copy()
            
            # Perform aggressive local optimization with multiple passes
            try:
                # Multiple refinement passes
                for pass_num in range(10):  # Increased iterations
                    improved = False
                    # Try to improve each circle individually
                    for i in range(n):
                        # Save current state
                        orig_x, orig_y, orig_r = final_vars[3*i], final_vars[3*i+1], final_vars[3*i+2]
                        
                        # Try to increase radius while maintaining feasibility
                        max_r = min(orig_x, 1-orig_x, orig_y, 1-orig_y)
                        new_r = min(max_r, orig_r + 0.01)
                        
                        # Check if we can actually increase the radius
                        valid = True
                        for j in range(n):
                            if i != j:
                                x_j, y_j, r_j = final_vars[3*j], final_vars[3*j+1], final_vars[3*j+2]
                                dist = np.sqrt((orig_x - x_j)**2 + (orig_y - y_j)**2)
                                if dist < new_r + r_j:
                                    valid = False
                                    break
                        
                        if valid and new_r > orig_r:
                            final_vars[3*i + 2] = new_r
                            improved = True
                    
                    if not improved:
                        break
                        
            except:
                pass
        
        # Extract final result
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [final_vars[3*i], final_vars[3*i+1], final_vars[3*i+2]]
            
        return circles
        
    except Exception as e:
        # Fallback: return the initial configuration if optimization fails
        circles = np.zeros((n, 3))
        for i in range(n):
            x, y = initial_positions[i]
            r = initial_radii[i]
            circles[i] = [x, y, r]
        return circles


# EVOLVE-BLOCK-END
