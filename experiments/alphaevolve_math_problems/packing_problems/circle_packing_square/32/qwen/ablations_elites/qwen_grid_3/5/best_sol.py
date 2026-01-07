# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
from scipy.optimize import minimize
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach inspired by successful circle packing algorithms.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    n = 32
    
    def get_initial_config():
        """Generate initial configuration using a better systematic approach"""
        circles = np.zeros((n, 3))
        
        # Use a more strategic grid approach with better spacing
        rows = 6
        cols = 6
        
        # Calculate spacing to fit 32 circles reasonably
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows
        
        # Create a grid pattern with slight jitter for better distribution
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= 32:
                    break
                # Position with jitter for better distribution
                x = 0.05 + (j + 0.5) * spacing_x + np.random.uniform(-0.1, 0.1) * spacing_x
                y = 0.05 + (i + 0.5) * spacing_y + np.random.uniform(-0.1, 0.1) * spacing_y
                
                # Initial radius - based on spacing to allow good packing
                r = min(spacing_x, spacing_y) * 0.3
                
                # Ensure circle fits in square with margin
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                
                circles[count] = [x, y, r]
                count += 1
                
            if count >= 32:
                break
        
        # Fill remaining positions with more careful placement
        for i in range(count, 32):
            placed = False
            max_attempts = 1000
            attempts = 0
            
            while not placed and attempts < max_attempts:
                # Try to place near edges or corners to improve distribution
                if i < 4:  # Place first few at corners
                    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
                    corner_idx = i % len(corners)
                    x_base, y_base = corners[corner_idx]
                    x = x_base + np.random.uniform(-0.05, 0.05)
                    y = y_base + np.random.uniform(-0.05, 0.05)
                else:
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.uniform(0.05, 0.95)
                    
                # Initial radius
                r = np.random.uniform(0.01, 0.08)
                
                # Check boundary constraints
                if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                    # Check overlap with existing circles
                    valid = True
                    for j in range(i):
                        x_prev, y_prev, r_prev = circles[j]
                        distance = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
                        if distance < r + r_prev:
                            valid = False
                            break
                    if valid:
                        circles[i] = [x, y, r]
                        placed = True
                attempts += 1
        
        return circles
    
    def check_constraints(circles):
        """Check if all circles satisfy constraints"""
        # Check containment constraints
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlap constraints
        for i, j in combinations(range(len(circles)), 2):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if distance < r1 + r2:
                return False
        
        return True
    
    def calculate_radius_sum(circles):
        """Calculate total sum of radii"""
        return np.sum(circles[:, 2])
    
    def optimize_with_scipy(initial_circles):
        """Use scipy optimization with better constraint handling"""
        # Flatten initial configuration for optimization
        initial_vars = []
        for i in range(len(initial_circles)):
            x, y, r = initial_circles[i]
            initial_vars.extend([x, y, r])
        
        # Define objective function to maximize sum of radii (negative because we minimize)
        def objective(vars_flat):
            circles = np.array(vars_flat).reshape(-1, 3)
            return -np.sum(circles[:, 2])
        
        # Define constraints with better numerical stability
        def boundary_constraints(vars_flat):
            """Ensure all circles are within the unit square"""
            circles = np.array(vars_flat).reshape(-1, 3)
            constraints = []
            
            for i in range(len(circles)):
                x, y, r = circles[i]
                # Circle must fit entirely within square with margin for numerical stability
                constraints.append(x - r - 1e-6)      # x - r >= 1e-6
                constraints.append(1 - x - r - 1e-6)  # 1 - x - r >= 1e-6
                constraints.append(y - r - 1e-6)      # y - r >= 1e-6
                constraints.append(1 - y - r - 1e-6)  # 1 - y - r >= 1e-6
                
            return np.array(constraints)
        
        def overlap_constraints(vars_flat):
            """Ensure no two circles overlap with numerical tolerance"""
            circles = np.array(vars_flat).reshape(-1, 3)
            constraints = []
            
            for i, j in combinations(range(len(circles)), 2):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_distance_sq = (r1 + r2)**2
                
                # Add small tolerance to avoid numerical issues
                constraints.append(distance_sq - min_distance_sq - 1e-10)
                
            return np.array(constraints)
        
        # Set up bounds for variables with tighter ranges
        bounds = []
        for i in range(len(initial_circles)):
            # x bounds
            bounds.append((0.001, 0.999))
            # y bounds
            bounds.append((0.001, 0.999))
            # r bounds - slightly smaller upper bound for safety
            bounds.append((0.001, 0.49))
        
        # Set up constraints
        cons = [
            {'type': 'ineq', 'fun': lambda x: boundary_constraints(x)},
            {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
        ]
        
        # Try multiple optimization methods for robustness
        methods_to_try = ['SLSQP', 'trust-constr']
        best_result = None
        best_sum = -np.inf
        
        for method in methods_to_try:
            try:
                result = minimize(objective, initial_vars, method=method, 
                                bounds=bounds, constraints=cons, 
                                options={'maxiter': 300, 'ftol': 1e-6, 'gtol': 1e-6})
                
                if result.success:
                    # Evaluate the result
                    circles = result.x.reshape(-1, 3)
                    current_sum = np.sum(circles[:, 2])
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
                        
            except Exception as e:
                continue
        
        # If any method succeeded, return the best result
        if best_result is not None:
            optimized_circles = best_result.x.reshape(-1, 3)
            return optimized_circles
        else:
            # If all optimization failed, return initial configuration
            return initial_circles
    
    def force_based_optimization(circles, max_iter=50):
        """Apply force-based optimization to improve packing"""
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()
        
        # Parameters for force calculation
        k_repel = 1000.0
        k_containment = 1000.0
        dt = 0.001
        
        for iteration in range(max_iter):
            forces = np.zeros_like(positions)
            
            # Repulsion forces between circles
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq)
                    
                    if dist > 0 and dist < (radii[i] + radii[j]):
                        # Repulsive force with smoother decay
                        force_magnitude = k_repel * (radii[i] + radii[j] - dist) / (dist + 1e-8)
                        forces[i, 0] += force_magnitude * dx / dist
                        forces[i, 1] += force_magnitude * dy / dist
                        forces[j, 0] -= force_magnitude * dx / dist
                        forces[j, 1] -= force_magnitude * dy / dist
            
            # Containment forces (push back into bounds)
            for i in range(len(positions)):
                # Push away from boundaries with stronger force near edges
                boundary_forces = np.array([
                    max(0, radii[i] - positions[i, 0]),  # left boundary
                    max(0, radii[i] - positions[i, 1]),  # bottom boundary
                    max(0, positions[i, 0] + radii[i] - 1),  # right boundary
                    max(0, positions[i, 1] + radii[i] - 1)   # top boundary
                ])
                
                forces[i, 0] += k_containment * boundary_forces[0] - k_containment * boundary_forces[2]
                forces[i, 1] += k_containment * boundary_forces[1] - k_containment * boundary_forces[3]
            
            # Update positions
            positions += dt * forces
            
            # Keep positions within bounds
            positions[:, 0] = np.clip(positions[:, 0], radii, 1-radii)
            positions[:, 1] = np.clip(positions[:, 1], radii, 1-radii)
        
        # Create updated circles array
        updated_circles = np.column_stack([positions, radii])
        return updated_circles
    
    def enhanced_local_search(circles):
        """Enhanced local search with better compromise logic"""
        current_circles = circles.copy()
        
        # Strategy: Try to increase radii systematically
        improved = True
        iteration = 0
        max_iterations = 1000
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Sort by current radius to focus on smaller ones first
            sorted_indices = np.argsort(current_circles[:, 2])
            
            for i in sorted_indices:
                # Try to increase radius of circle i
                original_radius = current_circles[i, 2]
                test_radius = min(0.4, original_radius + 0.005)
                
                # Test if we can increase this radius
                test_circles = current_circles.copy()
                test_circles[i, 2] = test_radius
                
                # Check constraints
                if check_constraints(test_circles):
                    test_sum = calculate_radius_sum(test_circles)
                    if test_sum > calculate_radius_sum(current_circles):
                        current_circles = test_circles
                        improved = True
                else:
                    # Try to make a compromise with neighbors
                    # Try decreasing some nearby radii to make room
                    for j in range(len(current_circles)):
                        if i != j and current_circles[j, 2] > 0.02:
                            test_circles = current_circles.copy()
                            test_circles[j, 2] = max(0.01, test_circles[j, 2] - 0.002)
                            test_circles[i, 2] = test_radius
                            
                            if check_constraints(test_circles):
                                test_sum = calculate_radius_sum(test_circles)
                                if test_sum > calculate_radius_sum(current_circles):
                                    current_circles = test_circles
                                    improved = True
                                    break
        
        return current_circles
    
    # Main algorithm
    try:
        # Generate initial configuration
        initial_config = get_initial_config()
        
        # Optimize the configuration using multiple techniques
        optimized_circles = optimize_with_scipy(initial_config)
        
        # Apply force-based optimization for fine-tuning
        optimized_circles = force_based_optimization(optimized_circles, 50)
        
        # Apply enhanced local search
        optimized_circles = enhanced_local_search(optimized_circles)
        
        # Try several local search approaches with different perturbations
        best_circles = optimized_circles.copy()
        best_sum = calculate_radius_sum(best_circles)
        
        # Multiple restarts with different perturbations
        for restart in range(3):
            # Perturb the configuration differently
            perturbed = initial_config.copy()
            
            # Perturb a varying number of circles
            num_perturbed = max(1, n//4 + restart*2)  # Varying perturbation
            perturb_indices = np.random.choice(n, size=min(num_perturbed, n), replace=False)
            
            for i in perturb_indices:
                # Different perturbation sizes for different restarts
                perturbation_size = 0.02 + restart * 0.01  # Increase with restart count
                perturbed[i, 0] += np.random.uniform(-perturbation_size, perturbation_size)
                perturbed[i, 1] += np.random.uniform(-perturbation_size, perturbation_size)
                perturbed[i, 2] += np.random.uniform(-0.01, 0.01)
                
                # Ensure bounds
                perturbed[i, 0] = np.clip(perturbed[i, 0], 0.05, 0.95)
                perturbed[i, 1] = np.clip(perturbed[i, 1], 0.05, 0.95)
                perturbed[i, 2] = np.clip(perturbed[i, 2], 0.01, 0.4)
            
            # Apply optimization to perturbed version
            improved_circles = optimize_with_scipy(perturbed)
            
            # Apply force-based optimization to perturbed version
            improved_circles = force_based_optimization(improved_circles, 40)
            
            # Apply enhanced local search
            improved_circles = enhanced_local_search(improved_circles)
            
            # Keep the best result
            current_sum = calculate_radius_sum(improved_circles)
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = improved_circles
        
        # Final validation
        if not check_constraints(best_circles):
            # If constraints are violated, revert to initial config with small radii
            best_circles = get_initial_config()
            best_circles[:, 2] = 0.02  # Small equal radii
        
        circles = best_circles
        
    except Exception as e:
        # Fallback to simple configuration
        circles = np.zeros((n, 3))
        sqrt_n = int(math.ceil(math.sqrt(n)))
        spacing = 1.0 / (sqrt_n + 1)
        
        idx = 0
        for i in range(sqrt_n):
            for j in range(sqrt_n):
                if idx >= n:
                    break
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + 0.5 + offset) * spacing
                y = (i + 0.5) * spacing
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                radius = min(0.45, spacing * 0.4)
                circles[idx] = [x, y, radius]
                idx += 1
            if idx >= n:
                break
        circles = circles[:n]
    
    return circles


# EVOLVE-BLOCK-END
