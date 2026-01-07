# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, physics-based refinement, 
    and constrained optimization with multiple restart strategies.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 26
    
    def initialize_circles():
        """Initialize circles using a sophisticated approach inspired by evolutionary methods"""
        # Create multiple initialization strategies and pick the best
        strategies = []
        
        # Strategy 1: Hexagonal grid with slight randomness
        strategy1 = []
        rows, cols = 5, 6
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1) * np.sqrt(3)/2
        
        for i in range(rows):
            for j in range(cols):
                if len(strategy1) >= n:
                    break
                if i % 2 == 0:
                    x = (j + 1) * spacing_x + np.random.uniform(-0.01, 0.01)
                else:
                    x = (j + 1.5) * spacing_x + np.random.uniform(-0.01, 0.01)
                y = (i + 1) * spacing_y + np.random.uniform(-0.01, 0.01)
                
                if 0 <= x <= 1 and 0 <= y <= 1:
                    strategy1.append([x, y, 0.0])
        
        # Fill remaining with random
        while len(strategy1) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            strategy1.append([x, y, 0.0])
        
        # Assign radii based on available space
        for i in range(len(strategy1)):
            x, y, _ = strategy1[i]
            max_radius = min(x, 1-x, y, 1-y)
            strategy1[i][2] = max(0.01, min(max_radius * 0.3, 0.15))
        
        strategies.append(np.array(strategy1[:n]))
        
        # Strategy 2: Center-focused with random perturbations
        strategy2 = []
        # Place some in center area
        for i in range(8):
            x = 0.4 + np.random.uniform(-0.15, 0.15)
            y = 0.4 + np.random.uniform(-0.15, 0.15)
            strategy2.append([x, y, 0.0])
        
        # Fill with random
        while len(strategy2) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            strategy2.append([x, y, 0.0])
        
        # Assign radii
        for i in range(len(strategy2)):
            x, y, _ = strategy2[i]
            max_radius = min(x, 1-x, y, 1-y)
            strategy2[i][2] = max(0.01, min(max_radius * 0.25, 0.12))
        
        strategies.append(np.array(strategy2[:n]))
        
        # Strategy 3: Edge-distributed approach
        strategy3 = []
        # Place along edges and corners
        for i in range(6):
            # Top edge
            x = np.random.uniform(0.1, 0.9)
            y = 0.05
            strategy3.append([x, y, 0.0])
        
        for i in range(6):
            # Bottom edge
            x = np.random.uniform(0.1, 0.9)
            y = 0.95
            strategy3.append([x, y, 0.0])
            
        for i in range(4):
            # Left edge
            x = 0.05
            y = np.random.uniform(0.1, 0.9)
            strategy3.append([x, y, 0.0])
            
        for i in range(4):
            # Right edge
            x = 0.95
            y = np.random.uniform(0.1, 0.9)
            strategy3.append([x, y, 0.0])
            
        # Fill remaining with random
        while len(strategy3) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            strategy3.append([x, y, 0.0])
        
        # Assign radii
        for i in range(len(strategy3)):
            x, y, _ = strategy3[i]
            max_radius = min(x, 1-x, y, 1-y)
            strategy3[i][2] = max(0.01, min(max_radius * 0.2, 0.1))
        
        strategies.append(np.array(strategy3[:n]))
        
        # Choose the best initialization based on initial packing quality
        best_strategy = strategies[0]
        best_quality = 0
        
        for strategy in strategies:
            # Calculate initial packing quality (sum of radii with overlap penalty)
            total_radius = np.sum(strategy[:, 2])
            overlap_penalty = 0
            
            # Count overlaps
            for i in range(len(strategy)):
                for j in range(i+1, len(strategy)):
                    x1, y1, r1 = strategy[i]
                    x2, y2, r2 = strategy[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < (r1 + r2):
                        overlap_penalty += (r1 + r2 - distance) * 10000
            
            quality = total_radius - overlap_penalty
            if quality > best_quality:
                best_quality = quality
                best_strategy = strategy
                
        return best_strategy
    
    def apply_repulsion(circles, iterations=200):
        """Apply repulsive forces between circles to prevent overlap"""
        n_circles = len(circles)
        forces = np.zeros((n_circles, 2))  # Force vectors (dx, dy)
        
        for _ in range(iterations):
            forces.fill(0)
            
            # Calculate repulsive forces
            for i in range(n_circles):
                for j in range(i+1, n_circles):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0 and dist < (r1 + r2):
                        # Repulsive force when circles are overlapping
                        force_magnitude = (r1 + r2 - dist) / (dist + 1e-10)
                        forces[i][0] += force_magnitude * dx / dist
                        forces[i][1] += force_magnitude * dy / dist
                        forces[j][0] -= force_magnitude * dx / dist
                        forces[j][1] -= force_magnitude * dy / dist
            
            # Apply boundary repulsion with stronger force
            for i in range(n_circles):
                x, y, r = circles[i]
                # Stronger boundary repulsion
                boundary_force = 0.3
                if x - r < 0:
                    forces[i][0] += boundary_force * (r - x)
                elif x + r > 1:
                    forces[i][0] += boundary_force * (1 - r - x)
                if y - r < 0:
                    forces[i][1] += boundary_force * (r - y)
                elif y + r > 1:
                    forces[i][1] += boundary_force * (1 - r - y)
            
            # Update positions with damping
            for i in range(n_circles):
                x, y, r = circles[i]
                # Update positions with stronger movement
                new_x = max(r, min(1-r, x + 0.025 * forces[i][0]))
                new_y = max(r, min(1-r, y + 0.025 * forces[i][1]))
                circles[i] = [new_x, new_y, r]
        
        return circles
    
    def objective_function(circles_flat):
        """Objective function to minimize (negative sum of radii)"""
        return -np.sum(circles_flat[2::3])  # Sum of radii (every 3rd element starting from index 2)
    
    def constraint_containment(circles_flat):
        """Constraint function for containment within unit square"""
        constraints = []
        for i in range(n):
            x, y, r = circles_flat[3*i:3*i+3]
            # Each circle must be fully contained in [0,1]x[0,1]
            # r <= x <= 1-r and r <= y <= 1-r
            constraints.append(x - r)  # x - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            constraints.append(r)  # r >= 0
        return np.array(constraints)
    
    def constraint_nonoverlap(circles_flat):
        """Constraint function for non-overlapping circles"""
        constraints = []
        # Check all pairs of circles for overlap
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_flat[3*i:3*i+3]
                x2, y2, r2 = circles_flat[3*j:3*j+3]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                # Distance between centers must be >= sum of radii (non-overlapping)
                constraints.append(distance - (r1 + r2))  # >= 0 for non-overlapping
        return np.array(constraints)
    
    # Initialize circles with multiple strategies
    circles = initialize_circles()
    
    # Apply physics-based repulsion to get a good starting configuration
    circles = apply_repulsion(circles, iterations=200)
    
    # Flatten for optimization
    circles_flat = circles.flatten()
    
    # Set up bounds for optimization
    bounds = []
    for i in range(n):
        # Bounds for (x, y, r) triple: x, y in [0.001, 0.999], r in [0.001, 0.499]
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Create constraints dictionary
    constraints = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
    ]
    
    # Try multiple optimization strategies to get the best result
    best_result = None
    best_sum = 0
    
    # Strategy 1: Differential Evolution (global optimization) with enhanced parameters
    try:
        result_de = differential_evolution(
            objective_function,
            bounds,
            constraints=constraints,
            seed=42,
            maxiter=600,
            popsize=30,
            mutation=(0.5, 1),
            recombination=0.8,
            disp=False,
            tol=1e-9
        )
        
        if result_de.success:
            circles_opt = result_de.x.reshape(-1, 3)
            current_sum = np.sum(circles_opt[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = circles_opt
                
    except Exception as e:
        pass
    
    # Strategy 2: SLSQP with better starting point and tighter tolerances
    try:
        result_slsqp = minimize(
            objective_function,
            circles_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 600, 'ftol': 1e-12, 'eps': 1e-6}
        )
        
        if result_slsqp.success:
            circles_opt = result_slsqp.x.reshape(-1, 3)
            current_sum = np.sum(circles_opt[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = circles_opt
                
    except Exception as e:
        pass
    
    # Strategy 3: Enhanced local search with multiple restarts and better perturbation
    try:
        # Try multiple restarts with different initial configurations
        for restart in range(8):
            # Add more substantial random perturbation to current solution
            perturbed_flat = circles_flat.copy()
            # Apply different perturbations to different components
            for i in range(0, len(perturbed_flat), 3):
                # Perturb x coordinate with larger steps for exploration
                perturbed_flat[i] += np.random.normal(0, 0.03)
                # Perturb y coordinate  
                perturbed_flat[i+1] += np.random.normal(0, 0.03)
                # Perturb radius with moderate steps
                perturbed_flat[i+2] += np.random.normal(0, 0.015)
            
            # Clip to bounds
            for i in range(0, len(perturbed_flat), 3):
                perturbed_flat[i] = np.clip(perturbed_flat[i], 0.001, 0.999)      # x
                perturbed_flat[i+1] = np.clip(perturbed_flat[i+1], 0.001, 0.999)  # y
                perturbed_flat[i+2] = np.clip(perturbed_flat[i+2], 0.001, 0.499)  # r
            
            result_local = minimize(
                objective_function,
                perturbed_flat,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            if result_local.success:
                circles_opt = result_local.x.reshape(-1, 3)
                current_sum = np.sum(circles_opt[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = circles_opt
                    
    except Exception as e:
        pass
    
    # Strategy 4: Another round of local optimization from the best so far
    if best_result is not None:
        try:
            result_local2 = minimize(
                objective_function,
                best_result.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 400, 'ftol': 1e-11}
            )
            
            if result_local2.success:
                circles_opt = result_local2.x.reshape(-1, 3)
                current_sum = np.sum(circles_opt[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = circles_opt
                    
        except Exception as e:
            pass
    
    # If no optimization worked, use the repulsion result
    if best_result is None:
        best_result = circles
    
    # Final validation and adjustment
    circles_final = validate_and_adjust_circles(best_result)
    
    return circles_final

def validate_and_adjust_circles(circles):
    """Validate final configuration and make adjustments if needed"""
    # Ensure all circles are valid and within bounds
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Clamp values to valid ranges
        x = np.clip(x, 0.001, 0.999)
        y = np.clip(y, 0.001, 0.999)
        r = np.clip(r, 0.001, min(x, 1-x, y, 1-y))
        circles[i] = [x, y, r]
    
    # Recheck overlaps and adjust radii if necessary
    for _ in range(25):  # Allow more iterations to resolve conflicts
        # Simple greedy adjustment: reduce radii of overlapping circles
        overlaps_found = False
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < (r1 + r2):
                    # Overlap detected, reduce radii
                    total_reduction = (r1 + r2) - distance
                    reduction = total_reduction * 0.5
                    circles[i][2] = max(0.001, r1 - reduction)
                    circles[j][2] = max(0.001, r2 - reduction)
                    overlaps_found = True
        
        if not overlaps_found:
            break
    
    return circles


# EVOLVE-BLOCK-END
