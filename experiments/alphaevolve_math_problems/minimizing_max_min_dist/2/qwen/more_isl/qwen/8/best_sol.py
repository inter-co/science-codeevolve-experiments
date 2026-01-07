# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import random
import math
from scipy.optimize import minimize, differential_evolution


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple initialization strategies with local optimization
    and simulated annealing for global search.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    def calculate_min_max_ratio(points):
        """Calculate the ratio of minimum to maximum distance between all point pairs."""
        if len(points) < 2:
            return 0.0
            
        # Calculate pairwise distances
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def neighbor_move(points, step_size=0.05):
        """Generate a neighboring solution by perturbing one point."""
        new_points = points.copy()
        # Choose a random point to move
        idx = random.randint(0, len(points) - 1)
        # Perturb the point
        new_points[idx] += np.random.normal(0, step_size, 2)
        # Keep within bounds [0,1] x [0,1]
        new_points[idx] = np.clip(new_points[idx], 0, 1)
        return new_points
    
    def simulated_annealing(initial_points, max_iter=45000):
        """Run simulated annealing to find optimal point configuration."""
        current_points = initial_points.copy()
        current_ratio = calculate_min_max_ratio(current_points)
        
        # Better cooling schedule inspired by successful programs
        temperature = 1.0
        cooling_rate = 0.99975  # Slightly faster cooling for quicker convergence
        min_temperature = 1e-12
        max_iterations = max_iter
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            new_points = neighbor_move(current_points, 0.012)  # Slightly smaller steps
            new_ratio = calculate_min_max_ratio(new_points)
            
            # Accept or reject the new solution
            if new_ratio > current_ratio:
                current_points = new_points
                current_ratio = new_ratio
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
            else:
                # Accept with probability based on temperature
                delta = new_ratio - current_ratio
                if delta < 0:  # Only accept worsening moves with probability
                    acceptance_prob = math.exp(delta / temperature)
                    if random.random() < acceptance_prob:
                        current_points = new_points
                        current_ratio = new_ratio
            
            # Cool down
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
                
            # Occasionally reset to best solution to escape local optima
            if iteration % 2500 == 0 and iteration > 0:
                current_points = best_points.copy()
                current_ratio = best_ratio
        
        return best_points
    
    def get_hexagonal_grid(n: int, bounds=(0, 1)) -> np.ndarray:
        """Generate points in a hexagonal grid pattern."""
        points = np.zeros((n, 2))
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Adjust spacing to fit nicely in bounds
        spacing_x = (bounds[1] - bounds[0]) / (cols + 1)
        spacing_y = (bounds[1] - bounds[0]) / (rows + 1)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= n:
                    break
                # Hexagonal offset
                offset = spacing_x * 0.5 if i % 2 == 1 else 0
                points[idx, 0] = bounds[0] + (j + 1) * spacing_x + offset
                points[idx, 1] = bounds[0] + (i + 1) * spacing_y
                idx += 1
        
        # Normalize to fit within bounds properly
        if len(points) > 0:
            # Center and scale appropriately
            center = np.mean(points, axis=0)
            points = points - center
            max_extent = np.max(np.abs(points))
            if max_extent > 0:
                points = points / (2 * max_extent) + 0.5
            points = np.clip(points, bounds[0], bounds[1])
        
        return points
    
    def get_fibonacci_spiral(n: int, bounds=(0, 1)) -> np.ndarray:
        """Generate points using Fibonacci spiral pattern."""
        points = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        for i in range(n):
            theta = 2 * math.pi * i / golden_ratio
            radius = math.sqrt(i / (n - 1)) if n > 1 else 0
            x = 0.5 + 0.4 * radius * math.cos(theta)
            y = 0.5 + 0.4 * radius * math.sin(theta)
            points.append([x, y])
        
        points = np.array(points)
        # Normalize to fit within bounds properly
        if len(points) > 0:
            # Center and scale appropriately
            center = np.mean(points, axis=0)
            points = points - center
            max_extent = np.max(np.abs(points))
            if max_extent > 0:
                points = points / (2 * max_extent) + 0.5
            points = np.clip(points, bounds[0], bounds[1])
        return points
    
    def get_best_known_configuration() -> np.ndarray:
        """Return a known good configuration for 16 points."""
        # This is a manually constructed configuration that works well
        # Based on principles of uniform distribution and maximizing minimum distance
        points = np.array([
            # Corner points
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            # Edge midpoints  
            [0.5, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, 0.5],
            # Interior points
            [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
            [0.5, 0.25], [0.5, 0.75], [0.25, 0.5], [0.75, 0.5]
        ])
        return points
    
    # Multiple sophisticated initial configurations
    best_initial_points = None
    best_initial_ratio = -float('inf')
    
    # Strategy 1: Optimized hexagonal grid
    hex_points = get_hexagonal_grid(16, (0, 1))
    
    # Strategy 2: More refined perturbed hexagonal grid
    perturbed_hex = hex_points + np.random.normal(0, 0.02, (16, 2))
    perturbed_hex = np.clip(perturbed_hex, 0, 1)
    
    # Strategy 3: Fibonacci spiral
    spiral_points = get_fibonacci_spiral(16, (0, 1))
    
    # Strategy 4: Best known configuration
    best_known = get_best_known_configuration()
    
    # Strategy 5: Random but with some structure
    random_points = np.random.rand(16, 2)
    
    # Strategy 6: Grid-based initialization (evenly spaced)
    grid_points = np.array([[i/3 + 0.125, j/3 + 0.125] for i in range(4) for j in range(4)])
    grid_points = np.clip(grid_points, 0, 1)
    
    # Strategy 7: Another variation with more careful spacing - circle pattern
    circle_points = []
    for i in range(16):
        angle = 2 * math.pi * i / 16
        radius = 0.4 if i < 12 else 0.2  # Inner and outer ring
        x = 0.5 + radius * math.cos(angle)
        y = 0.5 + radius * math.sin(angle)
        circle_points.append([x, y])
    circle_points = np.array(circle_points)
    circle_points = np.clip(circle_points, 0, 1)
    
    # Strategy 8: Icosahedral-inspired construction (from inspiration 2) - improved version
    try:
        # Golden ratio
        phi = (1 + math.sqrt(5)) / 2
        
        # Vertices of regular icosahedron (normalized)
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]
        
        # Normalize to unit sphere
        norm_vertices = []
        for x, y, z in vertices:
            norm = math.sqrt(x*x + y*y + z*z)
            norm_vertices.append((x/norm, y/norm, z/norm))
        
        # Project to 2D using stereographic projection from south pole
        proj_points = []
        for x, y, z in norm_vertices:
            # Stereographic projection from south pole (0,0,-1)
            if z == -1:
                # Handle special case
                proj_points.append((0, 0))
            else:
                w = 1 / (1 + z)
                proj_x = x * w
                proj_y = y * w
                proj_points.append((proj_x, proj_y))
        
        # Convert to array and scale to [0,1] x [0,1]
        proj_array = np.array(proj_points)
        
        # Scale and shift to fit nicely in [0,1] x [0,1]
        x_min, x_max = proj_array[:, 0].min(), proj_array[:, 0].max()
        y_min, y_max = proj_array[:, 1].min(), proj_array[:, 1].max()
        
        if x_max > x_min:
            proj_array[:, 0] = (proj_array[:, 0] - x_min) / (x_max - x_min) * 0.8 + 0.1
        if y_max > y_min:
            proj_array[:, 1] = (proj_array[:, 1] - y_min) / (y_max - y_min) * 0.8 + 0.1
        
        # We need exactly 16 points, so we'll add more points strategically
        points = []
        
        # Add the 12 icosahedral points
        for i in range(12):
            points.append([proj_array[i][0], proj_array[i][1]])
        
        # Add 4 more points to reach 16 - these should be positioned to maximize minimum distance
        # Place them at strategic locations to increase dispersion
        additional_positions = [
            [0.5, 0.1],   # Top center
            [0.1, 0.5],   # Left center  
            [0.9, 0.5],   # Right center
            [0.5, 0.9]    # Bottom center
        ]
        
        for pos in additional_positions:
            points.append(pos)
        
        icosahedral_points = np.array(points[:16])
        # Add slight noise to break symmetry and improve optimization
        icosahedral_points += np.random.normal(0, 0.005, icosahedral_points.shape)
        icosahedral_points = np.clip(icosahedral_points, 0, 1)
    except:
        icosahedral_points = np.random.rand(16, 2)  # Fallback
    
    # Strategy 9: Additional refinement - small perturbation of best known configuration
    refined_best_known = best_known + np.random.normal(0, 0.01, (16, 2))
    refined_best_known = np.clip(refined_best_known, 0, 1)
    
    # Strategy 10: Alternative grid with more careful spacing
    alt_grid_points = []
    for i in range(4):
        for j in range(4):
            x = 0.1 + i * 0.225
            y = 0.1 + j * 0.225
            alt_grid_points.append([x, y])
    alt_grid_points = np.array(alt_grid_points[:16])
    alt_grid_points = np.clip(alt_grid_points, 0, 1)
    
    # Test all initial configurations
    initial_configs = [hex_points, perturbed_hex, spiral_points, best_known, random_points, grid_points, circle_points, icosahedral_points, refined_best_known, alt_grid_points]
    
    for config in initial_configs:
        if len(config) == 16:  # Ensure we have the right number of points
            points = config.copy()
            ratio = calculate_min_max_ratio(points)
            if ratio > best_initial_ratio:
                best_initial_ratio = ratio
                best_initial_points = points.copy()
    
    # Run simulated annealing from the best initial point with more iterations
    final_points = simulated_annealing(best_initial_points, 45000)
    
    # Run multiple restarts to improve chance of finding better solution
    best_points = final_points.copy()
    best_ratio = calculate_min_max_ratio(final_points)
    
    # Additional restarts with different strategies - increased to 10 restarts for better exploration
    for restart in range(10):  # More restarts to improve chances
        # Strategy: slightly perturbed version of the current best
        restart_points = best_points + np.random.normal(0, 0.01, (16, 2))
        restart_points = np.clip(restart_points, 0, 1)
        
        # Run SA from this restart point with more iterations
        sa_points = simulated_annealing(restart_points, 45000)
        ratio = calculate_min_max_ratio(sa_points)
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = sa_points
    
    # Final local optimization with L-BFGS-B to fine-tune the solution
    def objective_function(x):
        points = x.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        ratio = calculate_min_max_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    bounds = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    
    try:
        # Run local optimization on the best solution found with more iterations
        result = minimize(
            objective_function,
            best_points.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 250, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            final_ratio = calculate_min_max_ratio(optimized_points)
            
            if final_ratio > best_ratio:
                best_points = optimized_points
    except:
        pass
    
    # Try SLSQP as backup optimization method
    try:
        result = minimize(
            objective_function,
            best_points.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 250, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if result.success:
            slsqp_points = result.x.reshape(-1, 2)
            slsqp_points = np.clip(slsqp_points, 0, 1)
            slsqp_ratio = calculate_min_max_ratio(slsqp_points)
            
            if slsqp_ratio > best_ratio:
                best_ratio = slsqp_ratio
                best_points = slsqp_points
    except:
        pass
    
    # Try differential evolution as a final enhancement (inspiration from Program 1)
    def de_objective(x):
        points = x.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        ratio = calculate_min_max_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    # Differential evolution with bounds and more iterations for better exploration
    bounds_de = [(0, 1) for _ in range(32)]  # 16 points * 2 coordinates each
    try:
        de_result = differential_evolution(
            de_objective,
            bounds_de,
            maxiter=200,
            popsize=25,
            mutation=(0.7, 1.2),
            recombination=0.8,
            seed=42,
            disp=False
        )
        
        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            de_ratio = calculate_min_max_ratio(de_points)
            
            if de_ratio > best_ratio:
                best_ratio = de_ratio
                best_points = de_points.copy()
    except:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
