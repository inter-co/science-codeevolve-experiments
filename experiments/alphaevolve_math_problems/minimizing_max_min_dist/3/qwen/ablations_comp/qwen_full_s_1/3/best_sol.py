# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a hybrid approach combining physics-based initialization with multiple optimization techniques
    to find a configuration that maximizes min/max distance ratio.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given points"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        distances = distances[distances > 1e-12]  # Filter out near-zero distances
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0.0
    
    def objective(x):
        """Objective function to minimize negative of min/max ratio"""
        # Reshape flat array back to 14x3 points
        points = x.reshape(-1, 3)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Calculate min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return -1e10
            
        # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
        return -min_dist / max_dist
    
    def project_to_sphere(points, radius=1.0):
        """Project points onto a sphere of given radius"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        return (points / norms) * radius
    
    def simulate_repulsion(points, steps=1000, learning_rate=0.01, damping=0.97):
        """Enhanced electrostatic repulsion with better force calculation"""
        points = points.copy()
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(points)
        
        # Track improvement for early stopping
        last_improvement_step = 0
        last_ratio = best_ratio
        
        for step in range(steps):
            # Compute forces between all pairs with better physics model
            n_points = points.shape[0]
            forces = np.zeros_like(points)
            
            # More sophisticated force computation with better repulsion model
            for i in range(n_points):
                for j in range(i+1, n_points):
                    diff = points[i] - points[j]
                    dist_sq = np.dot(diff, diff)
                    
                    # Avoid singularities
                    if dist_sq < 1e-12:
                        continue
                    
                    # Use a more stable force calculation that prevents extreme clustering
                    # and provides smoother gradients
                    dist = np.sqrt(dist_sq)
                    # Softened inverse square law with better numerical stability
                    force_magnitude = 1.0 / (dist_sq * dist + 1e-8)  # Prevent explosion at small distances
                    
                    # Direction
                    force_direction = diff / (dist + 1e-8)
                    
                    # Apply forces
                    forces[i] += force_direction * force_magnitude
                    forces[j] -= force_direction * force_magnitude
            
            # Update positions with momentum and damping
            points += learning_rate * forces
            points *= damping  # Apply damping to prevent oscillation
            
            # Project back to sphere
            points = project_to_sphere(points)
            
            # Track best configuration
            current_ratio = compute_min_max_ratio(points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()
                last_improvement_step = step
            
            # Early stopping if no improvement in last 300 steps
            if step - last_improvement_step > 300:
                break
                
            # Adaptive learning rate decay
            if step > 500 and step % 100 == 0:
                learning_rate *= 0.95
        
        return best_points
    
    def get_truncated_octahedron_config():
        """Generate points based on truncated octahedron - proven effective configuration"""
        # Truncated octahedron has 14 vertices: 6 face centers and 8 vertices
        # Face centers: (±1,0,0), (0,±1,0), (0,0,±1)  
        face_centers = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        
        # Vertices: (±1,±1,±1) but only those that form truncated octahedron
        vertices = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ])
        
        # Combine to get 14 points
        points = np.vstack([face_centers, vertices])
        
        # Scale to unit sphere and add noise for better optimization
        scale = 0.7
        points = points * scale
        
        # Add small random perturbations to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, points.shape)
        points = points + noise
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        safe_norms = np.where(norms == 0, 1, norms)
        points = points / safe_norms[:, np.newaxis]
        
        return points
    
    def get_icosahedral_config():
        """Generate points based on icosahedron vertices plus two additional points"""
        # Regular icosahedron vertices (normalized to unit sphere)
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        
        # Normalize to unit sphere
        vertices = vertices / np.linalg.norm(vertices[0])
        
        # Add two more points that are symmetrically placed
        # These will be along the principal axes
        extra_points = np.array([[0, 0, 0.85], [0, 0, -0.85]])
        
        # Combine and normalize to appropriate scale
        all_points = np.vstack([vertices, extra_points])
        
        # Randomly perturb slightly to break degeneracies
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, all_points.shape)
        all_points = all_points + noise
        
        return all_points
    
    def get_fibonacci_config():
        """Generate points using Fibonacci spiral approach"""
        n = 14
        points = []
        
        # Generate points using Fibonacci spiral with better distribution
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        
        points = np.array(points) * 0.9
        return points
    
    def get_cube_axes_config():
        """Generate points from cube plus axes structure"""
        # Cube vertices
        cube_vertices = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]) * 0.5
        
        # Axes points - more carefully placed to avoid clustering
        axes_points = np.array([
            [0, 0, 0.8], [0, 0, -0.8],
            [0.8, 0, 0], [-0.8, 0, 0],
            [0, 0.8, 0], [0, -0.8, 0]
        ])
        
        # Combine and normalize
        all_points = np.vstack([cube_vertices, axes_points])
        return all_points
    
    def get_hybrid_config():
        """Create a hybrid configuration combining multiple approaches"""
        # Start with icosahedral
        ico_points = get_icosahedral_config()
        
        # Add Fibonacci points
        fib_points = get_fibonacci_config()
        
        # Blend them together with weighted averaging
        blended = 0.6 * ico_points + 0.4 * fib_points
        
        # Add some randomness to avoid local minima
        np.random.seed(123)
        noise = np.random.normal(0, 0.03, blended.shape)
        blended = blended + noise
        
        return blended
    
    def multi_start_optimization(initial_points_list):
        """Run optimization from multiple starting points with enhanced strategy"""
        best_ratio = -np.inf
        best_points = None
        
        # Try more starting configurations
        for i, start_points in enumerate(initial_points_list):
            try:
                # Apply physics-based repulsion first to improve initial configuration
                repulsed_points = simulate_repulsion(start_points, steps=800, learning_rate=0.01, damping=0.97)
                
                # Flatten points for optimization
                x0 = repulsed_points.flatten()
                
                # Define bounds for optimization (points in [-1, 1]^3)
                bounds = [(-1, 1) for _ in range(42)]
                
                # Multi-stage optimization with different methods for robustness
                methods_to_try = ['L-BFGS-B', 'TNC', 'SLSQP']
                best_result = None
                best_result_ratio = -np.inf
                
                for method in methods_to_try:
                    try:
                        result = minimize(
                            objective,
                            x0,
                            method=method,
                            bounds=bounds,
                            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
                        )
                        
                        if result.success:
                            final_points = result.x.reshape(-1, 3)
                            final_points = np.clip(final_points, -1, 1)
                            
                            # Calculate ratio for this solution
                            ratio = compute_min_max_ratio(final_points)
                            if ratio > best_result_ratio:
                                best_result_ratio = ratio
                                best_result = result
                    except:
                        continue
                
                # If we found a valid result, use it
                if best_result is not None and best_result.success:
                    final_points = best_result.x.reshape(-1, 3)
                    final_points = np.clip(final_points, -1, 1)
                    
                    # Calculate ratio for this solution
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = final_points.copy()
                        
            except Exception as e:
                continue
                
        return best_points if best_points is not None else initial_points_list[0]
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Truncated octahedron configuration (from inspiration 2, proven effective)
    initial_configs.append(get_truncated_octahedron_config())
    
    # 2. Icosahedral-based configuration (very good starting point)
    initial_configs.append(get_icosahedral_config())
    
    # 3. Fibonacci spiral configuration
    initial_configs.append(get_fibonacci_config())
    
    # 4. Cube plus axes configuration
    initial_configs.append(get_cube_axes_config())
    
    # 5. Hybrid configuration
    initial_configs.append(get_hybrid_config())
    
    # 6. Random configuration for diversity
    np.random.seed(456)
    random_points = np.random.uniform(-0.9, 0.9, (14, 3))
    initial_configs.append(random_points)
    
    # 7. Another random configuration with different seed
    np.random.seed(789)
    random_points2 = np.random.uniform(-0.9, 0.9, (14, 3))
    initial_configs.append(random_points2)
    
    # 8. Another truncated octahedron with different seed
    np.random.seed(999)
    to_config = get_truncated_octahedron_config()
    noise = np.random.normal(0, 0.03, to_config.shape)
    to_config = to_config + noise
    initial_configs.append(to_config)
    
    # Run multi-start optimization with enhanced refinement
    try:
        final_points = multi_start_optimization(initial_configs)
    except Exception as e:
        # Fallback to the best initial configuration
        print(f"Multi-start optimization failed: {e}")
        # Use the first configuration as fallback
        final_points = initial_configs[0]
    
    # Final refinement with physics simulation
    try:
        final_points = simulate_repulsion(final_points, steps=500, learning_rate=0.005, damping=0.98)
    except Exception:
        pass
    
    # One final optimization pass with high precision
    try:
        x0 = final_points.flatten()
        bounds = [(-1, 1) for _ in range(42)]
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        if result.success:
            final_points = result.x.reshape(-1, 3)
            final_points = np.clip(final_points, -1, 1)
    except Exception:
        pass
    
    return final_points


# EVOLVE-BLOCK-END
