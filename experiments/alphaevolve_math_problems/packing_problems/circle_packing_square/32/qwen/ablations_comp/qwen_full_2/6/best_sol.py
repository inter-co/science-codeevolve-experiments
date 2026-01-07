# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 32
    circles = np.zeros((n, 3))
    
    # Phase 1: Geometric initialization using hexagonal packing pattern
    # Arrange circles in a hexagonal grid pattern, then adjust positions
    rows = 6
    cols = 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    # Initialize positions in a grid pattern
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Offset every other row for hexagonal packing
            if i % 2 == 1:
                x += spacing_x * 0.5
            positions.append([x, y])
    
    # Ensure we have exactly n positions
    while len(positions) < n:
        positions.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    
    positions = np.array(positions[:n])
    
    # Phase 2: Initialize radii based on distance to nearest neighbors
    # Start with small radii and gradually increase
    radii = np.ones(n) * 0.05
    
    # Phase 3: Optimization using scipy minimize with constraints
    def objective(radii):
        return -np.sum(radii)  # Negative because we want to maximize
    
    def constraint_containment(i, pos, rad):
        """Ensure circle i stays within bounds"""
        x, y = pos
        r = rad
        return [
            x - r,           # x >= r
            1 - x - r,       # 1 - x >= r
            y - r,           # y >= r
            1 - y - r        # 1 - y >= r
        ]
    
    def constraint_overlap(i, j, pos_i, pos_j, rad_i, rad_j):
        """Ensure circles don't overlap"""
        dist = np.sqrt(np.sum((pos_i - pos_j)**2))
        return dist - rad_i - rad_j  # Should be >= 0
    
    # Create initial guess with positions and radii
    initial_guess = np.hstack([positions.flatten(), radii])
    
    # Define constraints
    cons = []
    
    # Add containment constraints for each circle
    for i in range(n):
        def contain_constraint(x):
            pos = x[2*i:2*i+2]
            rad = x[2*n+i]
            return constraint_containment(i, pos, rad)
        
        # For simplicity, we'll use a penalty method approach
        # In practice, this would be more sophisticated
    
    # Phase 4: Physics-inspired optimization
    # Simulate repulsive forces between circles
    max_iter = 1000
    learning_rate = 0.01
    decay_factor = 0.999
    
    # Start with good initial configuration
    best_radii_sum = 0
    best_circles = None
    
    # Try multiple random starting configurations
    for trial in range(5):
        np.random.seed(trial * 1000)
        
        # Initialize with slightly different positions
        trial_positions = positions.copy()
        for i in range(n):
            trial_positions[i, 0] += np.random.normal(0, 0.02)
            trial_positions[i, 1] += np.random.normal(0, 0.02)
            trial_positions[i, 0] = np.clip(trial_positions[i, 0], 0.05, 0.95)
            trial_positions[i, 1] = np.clip(trial_positions[i, 1], 0.05, 0.95)
        
        trial_radii = np.ones(n) * 0.08
        
        # Apply optimization
        for iteration in range(max_iter):
            # Calculate distances
            pos_array = trial_positions
            radius_array = trial_radii
            
            # Compute forces
            forces = np.zeros_like(pos_array)
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dx = pos_array[i, 0] - pos_array[j, 0]
                        dy = pos_array[i, 1] - pos_array[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        # Repulsion force (inverse square law)
                        if dist > 0 and dist < (radius_array[i] + radius_array[j]) * 2:
                            force_magnitude = 1.0 / (dist * dist + 0.001)
                            forces[i, 0] += force_magnitude * dx / dist
                            forces[i, 1] += force_magnitude * dy / dist
            
            # Update positions with forces
            for i in range(n):
                # Apply force to position
                trial_positions[i, 0] += learning_rate * forces[i, 0]
                trial_positions[i, 1] += learning_rate * forces[i, 1]
                
                # Boundary checks
                trial_positions[i, 0] = np.clip(trial_positions[i, 0], 
                                               radius_array[i], 1-radius_array[i])
                trial_positions[i, 1] = np.clip(trial_positions[i, 1], 
                                               radius_array[i], 1-radius_array[i])
            
            # Update radii based on available space
            for i in range(n):
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        dx = trial_positions[i, 0] - trial_positions[j, 0]
                        dy = trial_positions[i, 1] - trial_positions[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = min(min_dist, dist)
                
                # Set radius to maximum possible without overlapping
                new_radius = min_dist / 2.0 - 0.001
                new_radius = max(0.001, min(new_radius, 0.45))  # Clamp radius
                trial_radii[i] = new_radius
            
            # Apply learning rate decay
            learning_rate *= decay_factor
            
            # Track best solution
            current_sum = np.sum(trial_radii)
            if current_sum > best_radii_sum:
                best_radii_sum = current_sum
                best_circles = np.column_stack([trial_positions, trial_radii])
    
    # Final optimization using scipy for better results
    if best_circles is not None:
        # Refine with constrained optimization
        def refine_objective(x):
            # x contains [x1,y1,r1,x2,y2,r2,...]
            positions = x[:2*n].reshape(-1, 2)
            radii = x[2*n:]
            
            # Minimize negative sum (maximize sum)
            return -np.sum(radii)
        
        def refine_constraints(x):
            positions = x[:2*n].reshape(-1, 2)
            radii = x[2*n:]
            
            # List of constraint values
            constraints = []
            
            # Containment constraints (all >= 0)
            for i in range(n):
                x_pos, y_pos = positions[i]
                r = radii[i]
                constraints.extend([
                    x_pos - r,           # x >= r
                    1 - x_pos - r,       # 1 - x >= r  
                    y_pos - r,           # y >= r
                    1 - y_pos - r        # 1 - y >= r
                ])
            
            # Non-overlap constraints (all >= 0)
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = radii[i] + radii[j]
                    constraints.append(dist - min_dist)
            
            return np.array(constraints)
        
        # Create initial guess
        initial_x = np.hstack([best_circles[:, :2].flatten(), best_circles[:, 2]])
        
        # Optimize using SLSQP
        try:
            result = minimize(
                refine_objective,
                initial_x,
                method='SLSQP',
                constraints={'type': 'ineq', 'fun': refine_constraints},
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if result.success:
                refined_positions = result.x[:2*n].reshape(-1, 2)
                refined_radii = result.x[2*n:]
                best_circles = np.column_stack([refined_positions, refined_radii])
        except:
            pass
    
    # Return final result
    if best_circles is None:
        # Fallback to simple initialization
        best_circles = np.zeros((n, 3))
        # Place in grid pattern
        grid_size = int(np.ceil(np.sqrt(n)))
        for i in range(n):
            row = i // grid_size
            col = i % grid_size
            x = 0.1 + col * 0.8 / (grid_size - 1) if grid_size > 1 else 0.5
            y = 0.1 + row * 0.8 / (grid_size - 1) if grid_size > 1 else 0.5
            r = 0.05
            best_circles[i] = [x, y, r]
    
    return best_circles


# EVOLVE-BLOCK-END
