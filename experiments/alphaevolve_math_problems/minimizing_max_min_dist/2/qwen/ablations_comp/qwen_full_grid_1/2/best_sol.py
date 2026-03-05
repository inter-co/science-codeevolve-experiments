# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import random
import math
import warnings


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, simulated annealing, and optimization
    techniques from multiple inspirations for superior convergence.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def calculate_min_max_ratio(points):
        """Calculate the min/max distance ratio for given points"""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def apply_boundary_constraints(points):
        """Ensure all points remain within [0,1] x [0,1]"""
        return np.clip(points, 0, 1)
    
    def generate_regular_grid():
        """Generate a regular 4x4 grid pattern"""
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25
                y = i * 0.25
                points.append([x, y])
        return np.array(points)
    
    def generate_hexagonal_initial():
        """Alternative initial configuration inspired by hexagonal packing"""
        points = []
        # Place points in a hexagonal-like pattern
        rows = 4
        cols = 4
        for i in range(rows):
            for j in range(cols):
                # Offset every other row
                x_offset = (i % 2) * 0.125
                x = j * 0.25 + x_offset + random.uniform(-0.03, 0.03)
                y = i * 0.25 + random.uniform(-0.03, 0.03)
                points.append([x, y])
        
        # Ensure within bounds
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    def generate_circular_initial():
        """Initial configuration inspired by circular arrangements"""
        points = []
        
        # Place 12 points in a circular pattern (like a 12-gon)
        radius = 0.4
        for i in range(12):
            angle = 2 * math.pi * i / 12
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        
        # Add 4 more points in a square pattern near center
        offset = 0.15
        points.extend([
            [0.5 - offset, 0.5 - offset],  # bottom-left
            [0.5 + offset, 0.5 - offset],  # bottom-right
            [0.5 - offset, 0.5 + offset],  # top-left
            [0.5 + offset, 0.5 + offset]   # top-right
        ])
        
        return np.array(points)
    
    def generate_perturbed_grid():
        """Generate a more sophisticated initial grid with better spacing"""
        points = []
        # Create a grid with better distribution
        for i in range(4):
            for j in range(4):
                # Add non-uniform spacing to avoid regular patterns
                x = (j + 0.5 + random.uniform(-0.08, 0.08)) / 4.0
                y = (i + 0.5 + random.uniform(-0.08, 0.08)) / 4.0
                points.append([x, y])
        
        # Ensure within bounds
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    def generate_voronoi_like():
        """Generate initial points that resemble Voronoi cell centers"""
        points = []
        # Create points that are somewhat evenly distributed
        for i in range(4):
            for j in range(4):
                x = (j + 0.5 + random.uniform(-0.1, 0.1)) / 4.0
                y = (i + 0.5 + random.uniform(-0.1, 0.1)) / 4.0
                points.append([x, y])
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    def generate_random_initial():
        """Generate completely random initial configuration"""
        np.random.seed(42)
        return np.random.rand(16, 2)
    
    def generate_better_initial():
        """Generate an even better initial configuration"""
        # Start with a regular grid and add some randomness
        points = generate_regular_grid()
        # Add small random perturbations
        for i in range(len(points)):
            points[i] += np.random.normal(0, 0.02, 2)
        points = np.clip(points, 0, 1)
        return points
    
    def perturb_point(point, delta=0.015):
        """Create a neighbor point by perturbing one coordinate"""
        new_point = point.copy()
        dim = random.randint(0, 1)
        new_point[dim] += random.uniform(-delta, delta)
        return new_point
    
    def improved_cooling_schedule(iteration, max_iter):
        """Improved cooling schedule that adapts to progress"""
        # Very aggressive cooling initially, then slower convergence
        if iteration < max_iter * 0.2:
            return 0.999
        elif iteration < max_iter * 0.5:
            return 0.9995
        else:
            return 0.9998
    
    def enhanced_simulated_annealing(initial_points, max_iter=35000):
        """Enhanced simulated annealing with multiple improvements"""
        current_points = initial_points.copy()
        current_points = apply_boundary_constraints(current_points)
        
        current_ratio = calculate_min_max_ratio(current_points)
        current_best_ratio = current_ratio
        current_best_points = current_points.copy()
        
        best_ratio = current_ratio
        best_points = current_points.copy()
        
        temperature = 1.0
        min_temperature = 1e-12
        
        # Track recent improvements for early stopping
        recent_improvements = []
        max_recent = 30  # Increased for better stability
        
        iteration = 0
        
        while temperature > min_temperature and iteration < max_iter:
            # Pick a random point to perturb
            point_idx = random.randint(0, len(current_points) - 1)
            
            # Create neighbor solution
            neighbor_points = current_points.copy()
            neighbor_points[point_idx] = perturb_point(current_points[point_idx])
            
            # Apply boundary constraints
            neighbor_points = apply_boundary_constraints(neighbor_points)
            
            # Calculate new ratio
            neighbor_ratio = calculate_min_max_ratio(neighbor_points)
            
            # Accept or reject the move
            if neighbor_ratio > current_ratio:
                # Always accept better solutions
                current_points = neighbor_points
                current_ratio = neighbor_ratio
                
                # Update current best solution if improved
                if neighbor_ratio > current_best_ratio:
                    current_best_ratio = neighbor_ratio
                    current_best_points = neighbor_points.copy()
                    
                    # Track improvements
                    recent_improvements.append(current_best_ratio)
                    if len(recent_improvements) > max_recent:
                        recent_improvements.pop(0)
                        
                    # Update global best if needed
                    if neighbor_ratio > best_ratio:
                        best_ratio = neighbor_ratio
                        best_points = neighbor_points.copy()
            else:
                # Accept worse solutions with probability based on temperature
                delta = neighbor_ratio - current_ratio
                # Avoid numerical issues with very large exponents
                if delta < -100:
                    acceptance_prob = 0.0
                else:
                    acceptance_prob = math.exp(delta / temperature)
                if random.random() < acceptance_prob:
                    current_points = neighbor_points
                    current_ratio = neighbor_ratio
            
            # Improved cooling schedule
            temperature *= improved_cooling_schedule(iteration, max_iter)
            iteration += 1
            
            # Early stopping when no significant improvement in recent iterations
            if len(recent_improvements) >= max_recent:
                if len(recent_improvements) >= 2:
                    improvement_range = max(recent_improvements) - min(recent_improvements)
                    if improvement_range < 1e-8:
                        break
        
        return best_points, best_ratio
    
    def coordinate_descent_refinement(points, max_iter=150):
        """Fine-tune the solution using coordinate descent with enhanced search"""
        current_points = points.copy()
        current_ratio = calculate_min_max_ratio(current_points)
        
        for iteration in range(max_iter):
            improved = False
            # Try improving each point individually
            for i in range(len(current_points)):
                old_point = current_points[i].copy()
                best_point = old_point.copy()
                best_ratio = current_ratio
                
                # Try multiple perturbations for this point with varied scales
                for _ in range(200):  # More thorough search
                    # Try small, medium, and large perturbations
                    scale = 0.0005 if np.random.random() < 0.2 else (0.002 if np.random.random() < 0.6 else 0.005)
                    perturbation = np.random.normal(0, scale, 2)
                    new_point = old_point + perturbation
                    new_point = np.clip(new_point, 0, 1)
                    
                    test_points = current_points.copy()
                    test_points[i] = new_point
                    new_ratio = calculate_min_max_ratio(test_points)
                    
                    if new_ratio > best_ratio:
                        best_ratio = new_ratio
                        best_point = new_point
                        improved = True
                        
                current_points[i] = best_point
            
            # If no improvement, stop early
            if not improved:
                break
                
            current_ratio = best_ratio
            
        return current_points
    
    def multi_start_optimization(initial_points):
        """Run multiple optimizations with different strategies"""
        best_points = initial_points.copy()
        best_ratio = -np.inf
        
        # Method 1: Enhanced simulated annealing with more iterations
        try:
            sa_points, sa_ratio = enhanced_simulated_annealing(best_points, max_iter=35000)
            if sa_ratio > best_ratio:
                best_ratio = sa_ratio
                best_points = sa_points.copy()
        except Exception:
            pass
        
        # Method 2: Coordinate descent refinement with more thorough search
        try:
            refined_points = coordinate_descent_refinement(best_points, max_iter=100)
            refined_ratio = calculate_min_max_ratio(refined_points)
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points.copy()
        except Exception:
            pass
        
        # Method 3: Differential evolution (more robust version)
        try:
            def objective_function(points_flat):
                points = points_flat.reshape(-1, 2)
                points = apply_boundary_constraints(points)
                distances = pdist(points)
                if len(distances) == 0:
                    return 0
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                if max_dist <= 0:
                    return -np.inf
                return -min_dist / max_dist
            
            bounds = [(0, 1) for _ in range(32)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                de_result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=25,  # More iterations
                    popsize=10,   # Larger population for better exploration
                    seed=42,
                    tol=1e-8,     # Tighter tolerance
                    mutation=(0.5, 1),
                    recombination=0.7
                )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 2)
                de_points = apply_boundary_constraints(de_points)
                de_ratio = calculate_min_max_ratio(de_points)
                if de_ratio > best_ratio:
                    best_ratio = de_ratio
                    best_points = de_points.copy()
        except Exception:
            pass
        
        # Method 4: L-BFGS-B optimization
        try:
            bounds = [(0, 1) for _ in range(32)]
            
            # Run local optimization with the current best
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective_function,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-10}
                )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = apply_boundary_constraints(optimized_points)
                distances = pdist(optimized_points)
                if len(distances) > 0:
                    min_dist = np.min(distances)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        ratio = min_dist / max_dist
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
        except Exception:
            pass
        
        return best_points
    
    def hybrid_optimization():
        """Combine multiple optimization approaches for best results"""
        best_points = None
        best_ratio = -float('inf')
        
        # Multiple initial configurations
        initial_configs = [
            generate_regular_grid(),           # Regular grid
            generate_hexagonal_initial(),     # Hexagonal pattern
            generate_circular_initial(),      # Circular arrangement
            generate_perturbed_grid(),        # Perturbed grid
            generate_voronoi_like(),          # Voronoi-like
            generate_random_initial(),        # Random
            generate_better_initial(),        # Better initial
        ]
        
        # Run enhanced simulated annealing on each initial configuration
        for i, initial_config in enumerate(initial_configs):
            sa_points, sa_ratio = enhanced_simulated_annealing(initial_config, max_iter=25000)
            
            if sa_ratio > best_ratio:
                best_ratio = sa_ratio
                best_points = sa_points.copy()
        
        # Apply coordinate descent refinement
        if best_points is not None:
            refined_points = coordinate_descent_refinement(best_points, max_iter=75)
            refined_ratio = calculate_min_max_ratio(refined_points)
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points.copy()
        
        # Final optimization with multi-start approach
        if best_points is not None:
            final_points = multi_start_optimization(best_points)
            final_ratio = calculate_min_max_ratio(final_points)
            if final_ratio > best_ratio:
                best_ratio = final_ratio
                best_points = final_points.copy()
        
        return best_points if best_points is not None else generate_regular_grid()
    
    # Run the hybrid optimization
    final_points = hybrid_optimization()
    
    return final_points


# EVOLVE-BLOCK-END
