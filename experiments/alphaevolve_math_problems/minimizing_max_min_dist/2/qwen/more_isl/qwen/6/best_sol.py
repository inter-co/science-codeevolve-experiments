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
    
    def simulated_annealing(initial_points, max_iter=50000):
        """Run simulated annealing to find optimal point configuration."""
        current_points = initial_points.copy()
        current_ratio = calculate_min_max_ratio(current_points)
        
        # Optimized cooling schedule - slightly faster cooling for better convergence
        temperature = 1.0
        cooling_rate = 0.9996  # Slightly faster cooling
        min_temperature = 1e-12
        max_iterations = max_iter
        
        best_points = current_points.copy()
        best_ratio = current_ratio
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            new_points = neighbor_move(current_points, 0.012)  # Slightly smaller steps for precision
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
            # Center the points
            center = np.mean(points, axis=0)
            points = points - center
            # Scale to fit nicely within [0,1] x [0,1]
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
    
    def get_von_neumann_configuration() -> np.ndarray:
        """Return a von Neumann-like configuration that spreads points evenly."""
        # Create a 4x4 grid with some jitter
        points = []
        for i in range(4):
            for j in range(4):
                x = i * 0.3333 + 0.1667 + np.random.normal(0, 0.02)
                y = j * 0.3333 + 0.1667 + np.random.normal(0, 0.02)
                points.append([x, y])
        points = np.array(points)
        # Clip to ensure bounds
        points = np.clip(points, 0, 1)
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
    
    # Strategy 8: Von Neumann-like configuration
    von_neumann_points = get_von_neumann_configuration()
    
    # Test all initial configurations
    initial_configs = [hex_points, perturbed_hex, spiral_points, best_known, random_points, grid_points, circle_points, von_neumann_points]
    
    for config in initial_configs:
        if len(config) == 16:  # Ensure we have the right number of points
            points = config.copy()
            ratio = calculate_min_max_ratio(points)
            if ratio > best_initial_ratio:
                best_initial_ratio = ratio
                best_initial_points = points.copy()
    
    # Run simulated annealing from the best initial point with more iterations
    final_points = simulated_annealing(best_initial_points, 50000)
    
    # Run multiple restarts to improve chance of finding better solution
    best_points = final_points.copy()
    best_ratio = calculate_min_max_ratio(final_points)
    
    # Additional restarts with different strategies - increased to 6 restarts for better exploration
    for restart in range(6):  # More restarts for better exploration
        # Strategy: slightly perturbed version of the current best
        restart_points = best_points + np.random.normal(0, 0.012, (16, 2))
        restart_points = np.clip(restart_points, 0, 1)
        
        # Run SA from this restart point with more iterations
        sa_points = simulated_annealing(restart_points, 40000)
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
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9}
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
            options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-9}
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
    
    # Add differential evolution for enhanced global search
    def de_objective(x):
        points = x.reshape(-1, 2)
        points = np.clip(points, 0, 1)
        ratio = calculate_min_max_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    # Try multiple runs of differential evolution with different settings
    de_best_points = best_points.copy()
    de_best_ratio = best_ratio
    
    # Run several DE instances with different configurations
    for i in range(3):  # Run 3 different DE configurations
        try:
            # Different population sizes and mutation rates for diversity
            popsize = 15 + i * 5  # 15, 20, 25
            mutation_rate = 0.3 + i * 0.1  # 0.3, 0.4, 0.5
            
            de_result = differential_evolution(
                de_objective,
                bounds_de,
                maxiter=200,
                popsize=popsize,
                mutation=(mutation_rate, 1),
                recombination=0.7,
                seed=42 + i,
                disp=False
            )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 2)
                de_points = np.clip(de_points, 0, 1)
                de_ratio = calculate_min_max_ratio(de_points)
                
                if de_ratio > de_best_ratio:
                    de_best_ratio = de_ratio
                    de_best_points = de_points.copy()
        except:
            continue
    
    # Use the best DE result if it's better
    if de_best_ratio > best_ratio:
        best_points = de_best_points
        best_ratio = de_best_ratio
    
    return best_points


# EVOLVE-BLOCK-END
