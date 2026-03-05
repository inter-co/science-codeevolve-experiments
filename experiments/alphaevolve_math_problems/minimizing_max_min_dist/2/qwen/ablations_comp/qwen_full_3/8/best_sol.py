# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, dual_annealing, minimize
from scipy.linalg import eigh
import math
from typing import Tuple


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining mathematical constructions, spectral methods, and advanced optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points: np.ndarray) -> Tuple[float, float, float]:
        """Compute min/max distance ratio and actual values"""
        if len(points) < 2:
            return 0.0, 0.0, 0.0
            
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        ratio = d_min / d_max if d_max > 0 else 0.0
        return ratio, d_min, d_max
    
    def objective(x_flat):
        """Objective function to maximize min/max distance ratio"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio since we want to maximize
        if d_max == 0:
            return -1e10
        return -d_min / d_max
    
    def create_regular_polygon_initialization():
        """Create points on a regular 16-gon - proven good starting point"""
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Scale and center in unit square [0,1] x [0,1]
        center = np.mean(points, axis=0)
        scaled_points = (points - center) * 0.4 + 0.5
        
        # Add small random perturbations to break symmetry
        scaled_points += np.random.normal(0, 0.01, scaled_points.shape)
        scaled_points = np.clip(scaled_points, 0, 1)
        return scaled_points
    
    def create_root_of_unity_initialization():
        """Create points using 16th roots of unity in complex plane"""
        # Generate 16th roots of unity
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        roots = np.exp(1j * angles)
        
        # Convert to 2D points (real and imaginary parts)
        points = np.column_stack([roots.real, roots.imag])
        
        # Scale and center appropriately
        center = np.mean(points, axis=0)
        scaled_points = (points - center) * 0.4 + 0.5
        
        # Add small random perturbations to break symmetry
        scaled_points += np.random.normal(0, 0.01, scaled_points.shape)
        scaled_points = np.clip(scaled_points, 0, 1)
        return scaled_points
    
    def create_hexagonal_close_packing():
        """Create points arranged in hexagonal close packing pattern"""
        points = []
        # Create a hexagonal grid
        rows = 4
        cols = 4
        spacing = 0.25
        height = spacing * math.sqrt(3) / 2
        
        for i in range(rows):
            for j in range(cols):
                if len(points) < 16:
                    # Hexagonal offset for alternate rows
                    x = j * spacing + (i % 2) * spacing / 2
                    y = i * height
                    
                    # Add some randomness
                    x += np.random.normal(0, 0.02)
                    y += np.random.normal(0, 0.02)
                    
                    # Keep within bounds
                    x = max(0.01, min(0.99, x))
                    y = max(0.01, min(0.99, y))
                    points.append([x, y])
        
        return np.array(points[:16])
    
    def create_spectral_initialization():
        """Initialize using spectral graph theory approach"""
        # Start with a random configuration
        np.random.seed(42)
        points = np.random.rand(16, 2)
        
        # Spectral relaxation approach to get better initial configuration
        try:
            from scipy.spatial.distance import cdist
            # Iteratively improve using spectral relaxation
            for _ in range(30):
                # Compute distance matrix
                dist_matrix = cdist(points, points)
                # Avoid division by zero
                dist_matrix = np.maximum(dist_matrix, 1e-10)
                
                # Create weight matrix (inverse of distances)
                W = 1.0 / dist_matrix
                np.fill_diagonal(W, 0)
                
                # Compute degree matrix
                D = np.diag(np.sum(W, axis=1))
                
                # Compute Laplacian
                L = D - W
                
                # Compute first few non-trivial eigenvectors
                eigenvals, eigenvecs = eigh(L, subset_by_index=[1, 2])
                # Use second and third smallest eigenvectors for coordinates
                if eigenvecs.shape[1] >= 2:
                    points = eigenvecs[:, 1:3]
                    # Normalize to [0,1] range
                    points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0) + 1e-10)
                    points = np.clip(points, 0, 1)
        except:
            # Fallback to random if spectral method fails
            points = np.random.rand(16, 2)
        
        return points
    
    def create_symmetric_configuration():
        """Create highly symmetric configuration based on known optimal arrangements"""
        # Create points in a pattern inspired by optimal spherical codes
        # This uses a combination of regular polygons and symmetric arrangements
        
        points = []
        
        # Place 8 points in a regular octagon
        angles_oct = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for angle in angles_oct:
            points.append([0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)])
        
        # Place 8 more points in a smaller concentric circle
        angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
        for angle in angles_inner:
            points.append([0.5 + 0.15 * np.cos(angle), 0.5 + 0.15 * np.sin(angle)])
        
        points = np.array(points[:16])
        
        # Add small random perturbations
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def physics_based_approach() -> np.ndarray:
        """Enhanced physics-based approach with better force modeling"""
        # Initialize points with better distribution
        np.random.seed(42)
        points = np.random.rand(16, 2)
        
        # Physics parameters - optimized for faster convergence and better results
        max_iter = 300
        learning_rate = 0.03
        repulsion_strength = 3.0
        
        for iteration in range(max_iter):
            # Compute pairwise forces
            forces = np.zeros_like(points)
            
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    
                    if dist_sq > 0:
                        # More sophisticated repulsive force with better damping
                        force_magnitude = repulsion_strength / (dist_sq + 0.001)
                        force = force_magnitude * diff / np.sqrt(dist_sq)
                        forces[i] += force
                        forces[j] -= force
            
            # Apply forces with boundary constraints
            points += learning_rate * forces
            
            # Keep points within bounds
            points = np.clip(points, 0, 1)
            
            # Occasionally apply small random perturbations to escape local minima
            if iteration % 30 == 0 and iteration > 0:
                points += np.random.normal(0, 0.003, points.shape)
                points = np.clip(points, 0, 1)
        
        return points
    
    def grid_based_approach() -> np.ndarray:
        """Improved grid-based approach with strategic perturbations"""
        # Create a grid pattern with strategic perturbations
        points = np.zeros((16, 2))
        row_positions = np.linspace(0.1, 0.9, 4)
        col_positions = np.linspace(0.1, 0.9, 4)
        
        idx = 0
        for i, row in enumerate(row_positions):
            for j, col in enumerate(col_positions):
                if idx < 16:
                    # Add strategic jitter to avoid degenerate cases
                    jitter_x = (np.sin(i * 0.7) + np.cos(j * 0.5)) * 0.025
                    jitter_y = (np.cos(i * 0.5) + np.sin(j * 0.7)) * 0.025
                    
                    x = max(0.01, min(0.99, col + jitter_x))
                    y = max(0.01, min(0.99, row + jitter_y))
                    points[idx] = [x, y]
                    idx += 1
        return points
    
    def circular_arrangement() -> np.ndarray:
        """Enhanced circular arrangement with better spacing"""
        points = []
        radius = 0.4
        center = [0.5, 0.5]
        
        # Place points around circle with more even distribution
        for i in range(16):
            angle = 2 * math.pi * i / 16
            # Add slight perturbation for better dispersion
            angle += np.random.normal(0, 0.05)
            x = center[0] + radius * math.cos(angle) + np.random.normal(0, 0.01)
            y = center[1] + radius * math.sin(angle) + np.random.normal(0, 0.01)
            # Keep within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            points.append([x, y])
        return np.array(points)
    
    def fibonacci_spiral() -> np.ndarray:
        """Improved Fibonacci-like spiral pattern"""
        points = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        center = [0.5, 0.5]
        radius = 0.4
        
        for i in range(16):
            # Fibonacci spiral pattern with better distribution
            theta = i * 2 * math.pi / golden_ratio
            r = radius * math.sqrt(i / 15.0) if i > 0 else 0.01
            
            # Add some randomness to break symmetry
            theta += np.random.normal(0, 0.1)
            r += np.random.normal(0, 0.02)
            
            x = center[0] + r * math.cos(theta) + np.random.normal(0, 0.01)
            y = center[1] + r * math.sin(theta) + np.random.normal(0, 0.01)
            
            # Keep within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            points.append([x, y])
        return np.array(points)
    
    def hexagonal_pattern() -> np.ndarray:
        """Hexagonal pattern approach for better spacing"""
        points = []
        # Create points in a hexagonal lattice pattern
        rows = 4
        cols = 4
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Offset every other row for hexagonal packing
                offset = (i % 2) * 0.25
                x = offset + j * 0.25 + np.random.normal(0, 0.015)
                y = i * 0.25 + np.random.normal(0, 0.015)
                # Keep within bounds
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                points.append([x, y])
        return np.array(points[:16])
    
    def simulated_annealing_approach(initial_points: np.ndarray) -> np.ndarray:
        """Enhanced simulated annealing with better cooling schedule"""
        points = initial_points.copy()
        current_ratio, _, _ = compute_min_max_ratio(points)
        
        # Improved annealing parameters - faster convergence
        temperature = 1.0
        cooling_rate = 0.995
        min_temperature = 0.00001
        iterations_per_temp = 30
        
        best_points = points.copy()
        best_ratio = current_ratio
        
        while temperature > min_temperature:
            for _ in range(iterations_per_temp):
                # Make a small random change
                test_points = points.copy()
                idx = np.random.randint(0, len(test_points))
                test_points[idx] += np.random.normal(0, 0.008, 2)
                test_points = np.clip(test_points, 0, 1)
                
                # Evaluate
                test_ratio, _, _ = compute_min_max_ratio(test_points)
                
                # Accept or reject based on Metropolis criterion
                if test_ratio > current_ratio or np.random.random() < np.exp((test_ratio - current_ratio) / temperature):
                    points = test_points
                    current_ratio = test_ratio
                    
                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_points = points.copy()
            
            temperature *= cooling_rate
        
        return best_points
    
    def final_optimization_step(initial_points: np.ndarray) -> np.ndarray:
        """
        Apply final high-precision optimization to refine the best solution
        """
        # Flatten the configuration
        x0 = initial_points.flatten()
        
        # Define bounds for optimization (points in [0,1] x [0,1])
        bounds = [(0, 1) for _ in range(32)]
        
        # Try multiple optimization methods for robustness
        methods_and_settings = [
            ('L-BFGS-B', {'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-12}),
            ('TNC', {'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10}),
            ('SLSQP', {'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-10})
        ]
        
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)[0]
        
        for method, options in methods_and_settings:
            try:
                result = minimize(
                    objective,
                    x0,
                    method=method,
                    bounds=bounds,
                    options=options
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    final_points = np.clip(final_points, 0, 1)
                    final_ratio = compute_min_max_ratio(final_points)[0]
                    
                    if final_ratio > best_ratio:
                        best_ratio = final_ratio
                        best_points = final_points
            except Exception:
                continue
        
        return best_points
    
    # Try multiple initialization strategies and select the best
    initial_strategies = [
        ("root_of_unity", create_root_of_unity_initialization),
        ("regular_polygon", create_regular_polygon_initialization),
        ("symmetric", create_symmetric_configuration),
        ("spectral", create_spectral_initialization),
        ("hexagonal_close_packing", create_hexagonal_close_packing),
        ("physics", physics_based_approach),
        ("grid", grid_based_approach),
        ("circular", circular_arrangement),
        ("fibonacci", fibonacci_spiral),
        ("hexagonal", hexagonal_pattern)
    ]
    
    best_ratio = 0.0
    best_points = None
    
    for name, approach_func in initial_strategies:
        try:
            points = approach_func()
            
            # Refine with simulated annealing
            refined_points = simulated_annealing_approach(points)
            refined_ratio, _, _ = compute_min_max_ratio(refined_points)
            
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points
                
        except Exception as e:
            continue
    
    # Global optimization with dual annealing for better exploration
    if best_points is not None:
        try:
            # Multiple dual annealing runs with different seeds for better exploration
            best_global_points = best_points.copy()
            best_global_ratio = best_ratio
            
            # Run dual annealing multiple times with different seeds
            for seed_val in [42, 123, 456]:
                try:
                    bounds = [(0, 1) for _ in range(32)]
                    result = dual_annealing(
                        objective, 
                        bounds, 
                        maxiter=150,  # Reduced iterations to stay within time budget
                        seed=seed_val, 
                        no_local_search=False
                    )
                    
                    if result.success:
                        global_points = result.x.reshape(-1, 2)
                        global_points = np.clip(global_points, 0, 1)
                        global_ratio, _, _ = compute_min_max_ratio(global_points)
                        
                        if global_ratio > best_global_ratio:
                            best_global_ratio = global_ratio
                            best_global_points = global_points
                except Exception:
                    continue
            
            # Update if we found a better solution
            if best_global_ratio > best_ratio:
                best_ratio = best_global_ratio
                best_points = best_global_points
        except Exception:
            pass
    
    # Final refinement using scipy optimization if needed
    if best_points is not None:
        try:
            best_points = final_optimization_step(best_points)
            final_ratio = compute_min_max_ratio(best_points)[0]
            if final_ratio > best_ratio:
                best_ratio = final_ratio
        except Exception:
            pass
    
    # Additional fallback: Try multiple restarts with different approaches
    if best_points is None:
        # Try a few different approaches with different seeds
        for seed_val in [1, 42, 123, 456]:
            try:
                np.random.seed(seed_val)
                points = np.random.rand(16, 2)
                points = np.clip(points, 0, 1)
                refined_points = simulated_annealing_approach(points)
                refined_ratio, _, _ = compute_min_max_ratio(refined_points)
                
                if refined_ratio > best_ratio:
                    best_ratio = refined_ratio
                    best_points = refined_points
            except Exception:
                continue
    
    # Fallback to a simple good configuration if nothing worked
    if best_points is None:
        # Use a simple regular hexagon pattern
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        points = np.column_stack([
            0.5 + 0.4 * np.cos(angles),
            0.5 + 0.4 * np.sin(angles)
        ])
        best_points = points
    
    return best_points


# EVOLVE-BLOCK-END
