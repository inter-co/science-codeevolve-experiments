# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import random
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary strategy with gradient-based optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Enhanced initialization using a more sophisticated approach
    def generate_initial_configuration():
        """Generate a better initial configuration using a combination of strategies"""
        best_config = None
        best_sum = 0
        
        # Strategy 1: Try multiple grid-based patterns with different parameters
        for attempt in range(8):
            positions = []
            radii = []
            
            # Different grid layouts and perturbations
            if attempt < 3:
                # Square grid with slight perturbations
                grid_size = int(np.ceil(np.sqrt(n)))
                spacing_x = 1.0 / (grid_size + 1)
                spacing_y = 1.0 / (grid_size + 1)
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        if len(positions) >= n:
                            break
                        x = spacing_x * (j + 1) + (random.random() - 0.5) * spacing_x * 0.3
                        y = spacing_y * (i + 1) + (random.random() - 0.5) * spacing_y * 0.3
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            positions.append([x, y])
            elif attempt < 6:
                # Hexagonal-inspired pattern with different spacing
                rows = 6
                cols = 6
                spacing_x = 1.0 / (cols + 1)
                spacing_y = 1.0 / (rows + 1)
                
                for i in range(rows):
                    for j in range(cols):
                        if len(positions) >= n:
                            break
                        # Offset every other row
                        x_offset = 0.5 * spacing_x if i % 2 == 1 else 0
                        x = spacing_x * (j + 1) + x_offset + (random.random() - 0.5) * spacing_x * 0.2
                        y = spacing_y * (i + 1) + (random.random() - 0.5) * spacing_y * 0.2
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            positions.append([x, y])
            else:
                # More irregular pattern with stronger perturbations
                for i in range(n):
                    x = random.random() * 0.9 + 0.05  # Keep away from edges
                    y = random.random() * 0.9 + 0.05
                    positions.append([x, y])
            
            positions = positions[:n]
            
            # Set initial radii with more careful consideration
            for i, (x, y) in enumerate(positions):
                max_r = min(x, 1-x, y, 1-y)
                # Start with a more reasonable initial radius
                r = max_r * 0.25
                radii.append(r)
            
            # Validate configuration - be more strict about overlap checking
            valid = True
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2)
                    if dist < radii[i] + radii[j] - 1e-8:  # Small tolerance
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_sum = sum(radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_config = (positions, radii)
        
        # If no good configuration found, use fallback with careful validation
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
                    x = spacing_x * (j + 1) + (random.random() - 0.5) * spacing_x * 0.2
                    y = spacing_y * (i + 1) + (random.random() - 0.5) * spacing_y * 0.2
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        positions.append([x, y])
            
            positions = positions[:n]
            for i, (x, y) in enumerate(positions):
                max_r = min(x, 1-x, y, 1-y)
                r = max_r * 0.2
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
        """Ensure no overlapping circles - vectorized for performance"""
        cons = []
        # Vectorized computation of all pairwise distances
        positions = np.array([(vars[3*i], vars[3*i+1]) for i in range(n)])
        radii = np.array([vars[3*i+2] for i in range(n)])
        
        # Compute pairwise distances
        dist_matrix = cdist(positions, positions)
        
        # For each pair, check overlap constraint (distance >= sum of radii)
        for i in range(n):
            for j in range(i+1, n):
                dist = dist_matrix[i, j]
                r_i = radii[i]
                r_j = radii[j]
                # We want: dist >= r_i + r_j, so we enforce: dist - r_i - r_j >= 0
                cons.append(dist - r_i - r_j)
        
        return np.array(cons)
    
    # Phase 4: Define objective function
    def objective(vars):
        """Maximize sum of radii (minimize negative sum)"""
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i + 2]  # radius is third component
        return -total_radius  # Negative because we minimize
    
    # Phase 5: Run optimization with enhanced strategy
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
        
        # Try multiple optimization methods with multiple restarts for robustness
        best_result = None
        best_value = float('-inf')
        
        # Try trust-constr first with multiple restarts
        for restart in range(3):
            try:
                # Slightly different initial point for each restart
                restart_vars = initial_vars.copy()
                if restart > 0:
                    # Add small perturbations to initial variables
                    for i in range(len(restart_vars)):
                        if i % 3 < 2:  # x and y coordinates
                            restart_vars[i] += (random.random() - 0.5) * 0.01
                        else:  # radius
                            restart_vars[i] += (random.random() - 0.5) * 0.005
                
                result = minimize(
                    objective,
                    restart_vars,
                    method='trust-constr',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10, 'disp': False}
                )
                
                if result.success:
                    # Calculate the actual sum of radii
                    current_sum = -objective(result.x)
                    if current_sum > best_value:
                        best_value = current_sum
                        best_result = result
            except Exception as e:
                continue
        
        # If trust-constr didn't work well, try SLSQP with multiple restarts
        if best_result is None:
            for restart in range(3):
                try:
                    # Slightly different initial point for each restart
                    restart_vars = initial_vars.copy()
                    if restart > 0:
                        # Add small perturbations to initial variables
                        for i in range(len(restart_vars)):
                            if i % 3 < 2:  # x and y coordinates
                                restart_vars[i] += (random.random() - 0.5) * 0.01
                            else:  # radius
                                restart_vars[i] += (random.random() - 0.5) * 0.005
                    
                    result = minimize(
                        objective,
                        restart_vars,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 2000, 'ftol': 1e-10, 'eps': 1e-6, 'disp': False}
                    )
                    
                    if result.success:
                        # Calculate the actual sum of radii
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
            # Fall back to initial configuration with extensive refinement
            final_vars = initial_vars.copy()
            # Apply a more aggressive refinement step
            try:
                # Multiple passes of refinement
                for pass_num in range(3):
                    improved = True
                    iteration = 0
                    max_iterations = 20
                    
                    while improved and iteration < max_iterations:
                        improved = False
                        # Try to increase radii for all circles
                        for i in range(n):
                            # Try to increase radius of circle i
                            x, y, r = final_vars[3*i], final_vars[3*i+1], final_vars[3*i+2]
                            # Find maximum possible radius
                            max_r = min(x, 1-x, y, 1-y)
                            
                            # Check how much we can increase without violating constraints
                            new_r = min(max_r, r + 0.003)
                            
                            # Check overlap constraints with neighbors
                            valid = True
                            for j in range(n):
                                if i != j:
                                    x_j, y_j, r_j = final_vars[3*j], final_vars[3*j+1], final_vars[3*j+2]
                                    dist = np.sqrt((x - x_j)**2 + (y - y_j)**2)
                                    if dist < new_r + r_j - 1e-8:  # Small tolerance
                                        valid = False
                                        break
                            
                            if valid and new_r > r:
                                final_vars[3*i + 2] = new_r
                                improved = True
                        
                        iteration += 1
                        
            except Exception as e:
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
