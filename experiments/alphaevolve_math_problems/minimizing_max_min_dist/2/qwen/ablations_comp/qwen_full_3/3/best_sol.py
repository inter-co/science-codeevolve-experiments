# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.distance import pdist
import math
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple algorithmic paradigms with optimized parameters.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points: np.ndarray) -> tuple[float, float, float]:
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
    
    def create_icosahedral_approximation():
        """Create points approximating an icosahedral arrangement"""
        # Icosahedral vertices (normalized to unit sphere) 
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        points = []
        
        # 12 vertices of icosahedron
        vertices = [
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
        ]
        
        # Normalize and project to 2D
        for x, y, z in vertices:
            norm = math.sqrt(x*x + y*y + z*z)
            x_norm, y_norm, z_norm = x/norm, y/norm, z/norm
            
            # Stereographic projection from north pole
            # This projects 3D sphere to 2D plane
            proj_x = x_norm / (1 - z_norm)
            proj_y = y_norm / (1 - z_norm)
            points.append([proj_x, proj_y])
        
        # Convert to numpy array and scale appropriately
        points_array = np.array(points)
        # Normalize to [-1,1] range then map to [0,1]  
        points_array = (points_array - np.min(points_array, axis=0)) / (np.max(points_array, axis=0) - np.min(points_array, axis=0))
        points_array = points_array * 0.8 + 0.1  # Scale to [0.1, 0.9]
        
        # Add some randomness to break symmetry
        points_array += np.random.normal(0, 0.02, points_array.shape)
        points_array = np.clip(points_array, 0, 1)
        
        # Take first 16 points (we only have 12, so duplicate some)
        extended_points = np.vstack([points_array, points_array[:4]])
        return extended_points[:16]
    
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
    
    def physics_based_approach() -> np.ndarray:
        """Physics-based approach using repulsive forces between points"""
        # Initialize points with better distribution
        np.random.seed(42)
        points = np.random.rand(16, 2)
        
        # Physics parameters - optimized for faster convergence
        max_iter = 500
        learning_rate = 0.02
        repulsion_strength = 2.0
        
        for iteration in range(max_iter):
            # Compute pairwise forces
            forces = np.zeros_like(points)
            
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff**2)
                    
                    if dist_sq > 0:
                        # Repulsive force (inverse square law)
                        force_magnitude = repulsion_strength / (dist_sq + 0.001)
                        force = force_magnitude * diff / np.sqrt(dist_sq)
                        forces[i] += force
                        forces[j] -= force
            
            # Apply forces with boundary constraints
            points += learning_rate * forces
            
            # Keep points within bounds
            points = np.clip(points, 0, 1)
            
            # Occasionally apply small random perturbations to escape local minima
            if iteration % 50 == 0 and iteration > 0:
                points += np.random.normal(0, 0.002, points.shape)
                points = np.clip(points, 0, 1)
        
        return points
    
    def grid_based_approach() -> np.ndarray:
        """Grid-based approach with strategic perturbations"""
        # Create a grid pattern with strategic perturbations
        points = np.zeros((16, 2))
        row_positions = np.linspace(0.1, 0.9, 4)
        col_positions = np.linspace(0.1, 0.9, 4)
        
        idx = 0
        for i, row in enumerate(row_positions):
            for j, col in enumerate(col_positions):
                if idx < 16:
                    # Add strategic jitter to avoid degenerate cases
                    jitter_x = (np.sin(i * 0.7) + np.cos(j * 0.5)) * 0.02
                    jitter_y = (np.cos(i * 0.5) + np.sin(j * 0.7)) * 0.02
                    
                    x = max(0.01, min(0.99, col + jitter_x))
                    y = max(0.01, min(0.99, row + jitter_y))
                    points[idx] = [x, y]
                    idx += 1
        return points
    
    def circular_arrangement() -> np.ndarray:
        """Circular arrangement with perturbations"""
        points = []
        radius = 0.4
        center = [0.5, 0.5]
        
        # Place points around circle
        for i in range(16):
            angle = 2 * np.pi * i / 16
            x = center[0] + radius * np.cos(angle) + np.random.normal(0, 0.01)
            y = center[1] + radius * np.sin(angle) + np.random.normal(0, 0.01)
            # Keep within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            points.append([x, y])
        return np.array(points)
    
    def fibonacci_spiral() -> np.ndarray:
        """Fibonacci-like spiral pattern"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        center = [0.5, 0.5]
        radius = 0.4
        
        for i in range(16):
            # Fibonacci spiral pattern
            theta = i * 2 * np.pi / golden_ratio
            r = radius * np.sqrt(i / 15.0) if i > 0 else 0.01
            
            x = center[0] + r * np.cos(theta) + np.random.normal(0, 0.01)
            y = center[1] + r * np.sin(theta) + np.random.normal(0, 0.01)
            
            # Keep within bounds
            x = max(0.01, min(0.99, x))
            y = max(0.01, min(0.99, y))
            points.append([x, y])
        return np.array(points)
    
    def simulated_annealing_approach(initial_points: np.ndarray) -> np.ndarray:
        """Refinement using simulated annealing to improve quality"""
        points = initial_points.copy()
        current_ratio, _, _ = compute_min_max_ratio(points)
        
        # Annealing parameters - faster convergence with good cooling
        temperature = 1.0
        cooling_rate = 0.99
        min_temperature = 0.0001
        iterations_per_temp = 50
        
        best_points = points.copy()
        best_ratio = current_ratio
        
        while temperature > min_temperature:
            for _ in range(iterations_per_temp):
                # Make a small random change
                test_points = points.copy()
                idx = np.random.randint(0, len(test_points))
                test_points[idx] += np.random.normal(0, 0.005, 2)
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
    
    # Try multiple approaches and select the best
    approaches = [
        ("icosahedral", create_icosahedral_approximation),
        ("regular_polygon", create_regular_polygon_initialization),
        ("physics", physics_based_approach),
        ("grid", grid_based_approach),
        ("circular", circular_arrangement),
        ("fibonacci", fibonacci_spiral)
    ]
    
    best_ratio = 0.0
    best_points = None
    
    # Run multiple initialization strategies
    for name, approach_func in approaches:
        try:
            points = approach_func()
            
            # Refine with simulated annealing
            refined_points = simulated_annealing_approach(points)
            refined_ratio, _, _ = compute_min_max_ratio(refined_points)
            
            if refined_ratio > best_ratio:
                best_ratio = refined_ratio
                best_points = refined_points
                
        except Exception:
            continue
    
    # Additional global optimization using dual annealing with better settings
    if best_points is not None:
        try:
            # Use dual annealing for global optimization with more iterations and better settings
            bounds = [(0, 1) for _ in range(32)]
            result = dual_annealing(
                objective, 
                bounds, 
                maxiter=2500,  # More iterations for better chance
                seed=42, 
                no_local_search=False,
                initial_temp=2000,  # Higher initial temp for better exploration
                restarts=15         # More restarts for better chance
            )
            
            if result.success:
                global_points = result.x.reshape(-1, 2)
                global_points = np.clip(global_points, 0, 1)
                global_ratio, _, _ = compute_min_max_ratio(global_points)
                
                if global_ratio > best_ratio:
                    best_ratio = global_ratio
                    best_points = global_points
        except Exception:
            pass
    
    # Final refinement using scipy optimization if needed
    if best_points is not None:
        try:
            # Flatten the best configuration
            x0 = best_points.flatten()
            
            # Define bounds for optimization (points in [0,1] x [0,1])
            bounds = [(0, 1) for _ in range(32)]
            
            # Try multiple optimization methods for robustness
            methods_and_settings = [
                ('L-BFGS-B', {'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15}),
                ('TNC', {'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-13}),
                ('SLSQP', {'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-13})
            ]
            
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
                        final_ratio, _, _ = compute_min_max_ratio(final_points)
                        
                        if final_ratio > best_ratio:
                            best_ratio = final_ratio
                            best_points = final_points
                except Exception:
                    continue
                    
        except Exception:
            pass
    
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
