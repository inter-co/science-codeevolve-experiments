# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining constructive geometry, symmetry exploitation, and advanced 
    optimization with physics-inspired refinement.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        dmin = np.min(distances)
        dmax = np.max(distances)
        # Avoid division by zero and return a small value if dmax is very close to 0
        return dmin / dmax if dmax > 1e-12 else 0
    
    def compute_forces(points, softening=1e-10):
        """Compute repulsive forces between all pairs of points using a more sophisticated model"""
        n = len(points)
        forces = np.zeros_like(points)
        
        # Calculate forces between all pairs with better numerical stability
        for i in range(n):
            for j in range(i+1, n):
                diff = points[i] - points[j]
                dist_sq = np.sum(diff**2)
                
                # Avoid division by zero and very small distances
                if dist_sq > softening:
                    # Use a softened inverse square law with better numerical properties
                    # This is more physically realistic and avoids extreme force spikes
                    force_magnitude = 1.0 / (dist_sq + softening)
                    force_vector = force_magnitude * diff
                    forces[i] += force_vector
                    forces[j] -= force_vector
                    
        return forces
    
    def physics_refinement(points, max_iter=1500, learning_rate=0.02, damping=0.985):
        """Refine points using physics-based repulsion model with better parameters"""
        points = points.copy()
        best_points = points.copy()
        best_ratio = compute_min_max_ratio(points)
        
        # Track improvement for early stopping
        last_improvement = 0
        patience = 200
        
        for iteration in range(max_iter):
            # Compute forces
            forces = compute_forces(points)
            
            # Update positions with damping and momentum
            points += learning_rate * forces
            
            # Apply damping to prevent oscillation
            points *= damping
            
            # Project back to unit sphere to maintain constraint
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            points = points / norms
            
            # Check for improvement
            current_ratio = compute_min_max_ratio(points)
            if current_ratio > best_ratio:
                best_ratio = current_ratio
                best_points = points.copy()
                last_improvement = iteration
                
            # Early stopping if no improvement in last few iterations
            if iteration - last_improvement > patience:
                break
                
            # Reduce learning rate over time for fine-tuning
            if iteration > 500:
                learning_rate *= 0.995
                
        return best_points, best_ratio
    
    def objective_function(x):
        """Objective function to minimize (negative of min/max ratio)."""
        points = x.reshape(-1, 3)
        ratio = compute_min_max_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    # Generate multiple high-quality initial configurations
    initial_configs = []
    
    # Strategy 1: Enhanced icosahedral configuration
    np.random.seed(42)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    vertices = vertices / np.linalg.norm(vertices[0]) * 0.85
    
    # Add two more points along z-axis for better 14-point distribution
    extra_points = np.array([[0, 0, 0.9], [0, 0, -0.9]])
    icosahedral_points = np.vstack([vertices, extra_points])
    
    # Add random noise to break symmetries
    noise = np.random.normal(0, 0.03, icosahedral_points.shape)
    icosahedral_points = icosahedral_points + noise
    initial_configs.append(icosahedral_points)
    
    # Strategy 2: Enhanced golden spiral with better distribution
    np.random.seed(123)
    n = 14
    points = []
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)
        # Use more precise golden angle and add better variation
        theta = i * 2.399963229728653 + np.random.uniform(-0.15, 0.15)  
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    fibonacci_points = np.array(points)
    initial_configs.append(fibonacci_points)
    
    # Strategy 3: Random with better spread constraints
    np.random.seed(456)
    # Generate points with better distribution using rejection sampling
    points = []
    attempts = 0
    while len(points) < 14 and attempts < 1000:
        candidate = np.random.uniform(-0.95, 0.95, 3)
        if np.linalg.norm(candidate) <= 1.0:
            points.append(candidate)
        attempts += 1
    # Fill remaining if needed
    while len(points) < 14:
        points.append(np.random.uniform(-0.95, 0.95, 3))
    random_points = np.array(points)
    initial_configs.append(random_points)
    
    # Strategy 4: Perturbed icosahedral with rotation and better noise
    np.random.seed(789)
    base_points = icosahedral_points.copy()
    # Apply rotation with better distribution
    rotation = R.from_euler('xyz', np.random.rand(3) * 2 * np.pi).as_matrix()
    rotated_points = base_points @ rotation.T
    # Add more substantial noise for better exploration
    noise = np.random.normal(0, 0.04, rotated_points.shape)
    perturbed_points = rotated_points + noise
    initial_configs.append(perturbed_points)
    
    # Strategy 5: Spherical code approach with better parameters
    np.random.seed(321)
    points = []
    for i in range(14):
        # Modified Fibonacci approach with better spacing
        y = 1 - (i / (13)) * 2
        radius = np.sqrt(1 - y * y)
        # Use a more precise angle calculation
        theta = np.arccos(y) + np.random.uniform(-0.1, 0.1)
        phi = np.sqrt(14 * np.pi) * theta + np.random.uniform(-0.1, 0.1)
        points.append([radius * np.cos(phi), radius * np.sin(phi), y])
    spherical_points = np.array(points) * 0.85
    initial_configs.append(spherical_points)
    
    # Strategy 6: Another systematic approach with improved distribution
    np.random.seed(654)
    points = []
    for i in range(14):
        t = i / 13.0
        y = 1 - 2 * t
        radius = np.sqrt(1 - y*y)
        # Add more variation to avoid regular patterns
        theta = i * 2.399963229728653 + np.random.uniform(-0.25, 0.25)
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        points.append([x, y, z])
    systematic_points = np.array(points) * 0.9
    initial_configs.append(systematic_points)
    
    # Strategy 7: Additional configuration from known optimal patterns
    np.random.seed(999)
    # Try a different approach - start with vertices of a cube and add points
    cube_vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ]) * 0.7
    
    # Add 6 more points along coordinate axes
    axes_points = np.array([
        [0, 0, 0.8], [0, 0, -0.8],
        [0.8, 0, 0], [-0.8, 0, 0],
        [0, 0.8, 0], [0, -0.8, 0]
    ])
    
    # Combine and add noise
    cube_config = np.vstack([cube_vertices, axes_points])
    noise = np.random.normal(0, 0.03, cube_config.shape)
    cube_config = cube_config + noise
    initial_configs.append(cube_config)
    
    # Enhanced optimization with multiple restarts and physics refinement
    best_ratio = -np.inf
    best_points = None
    
    # Try each initial configuration with physics refinement first, then optimization
    for i, initial_config in enumerate(initial_configs):
        try:
            # Step 1: Physics-based refinement to get a better starting point
            refined_points, refined_ratio = physics_refinement(
                initial_config, 
                max_iter=1200, 
                learning_rate=0.02,
                damping=0.985
            )
            
            # Step 2: Scipy optimization on the refined points
            x0 = refined_points.flatten()
            bounds = [(0, 1) for _ in range(42)]
            
            # Try multiple optimization methods with aggressive parameters
            methods_and_options = [
                ('trust-constr', {'maxiter': 4000, 'ftol': 1e-17, 'gtol': 1e-17}),
                ('L-BFGS-B', {'maxiter': 4000, 'ftol': 1e-17, 'gtol': 1e-17}),
                ('SLSQP', {'maxiter': 4000, 'ftol': 1e-17, 'gtol': 1e-17})
            ]
            
            best_local_ratio = -np.inf
            best_local_points = None
            
            for method, options in methods_and_options:
                try:
                    result = minimize(
                        objective_function,
                        x0,
                        method=method,
                        bounds=bounds,
                        options=options
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 3)
                        optimized_points = np.clip(optimized_points, 0, 1)
                        ratio = compute_min_max_ratio(optimized_points)
                        
                        if ratio > best_local_ratio:
                            best_local_ratio = ratio
                            best_local_points = optimized_points.copy()
                            
                except Exception:
                    continue
            
            # Step 3: Final physics refinement on the optimized result
            if best_local_points is not None:
                final_points, final_ratio = physics_refinement(
                    best_local_points, 
                    max_iter=500, 
                    learning_rate=0.008,
                    damping=0.99
                )
                
                if final_ratio > best_local_ratio:
                    best_local_ratio = final_ratio
                    best_local_points = final_points.copy()
            
            # Keep the best result from this initialization
            if best_local_points is not None and best_local_ratio > best_ratio:
                best_ratio = best_local_ratio
                best_points = best_local_points.copy()
                
        except Exception:
            continue
    
    # If we didn't find a good solution, try a more aggressive global approach
    if best_points is None or best_ratio < 0.25:
        try:
            # Try a global optimization approach with many restarts
            np.random.seed(999)
            # Start with the best initial configuration
            best_initial = initial_configs[0]
            
            # Run physics refinement first with more iterations
            refined_points, refined_ratio = physics_refinement(
                best_initial, 
                max_iter=1500, 
                learning_rate=0.025,
                damping=0.985
            )
            
            # Aggressive optimization with trust-constr
            x0 = refined_points.flatten()
            bounds = [(0, 1) for _ in range(42)]
            
            result = minimize(
                objective_function,
                x0,
                method='trust-constr',
                bounds=bounds,
                options={'maxiter': 6000, 'ftol': 1e-18, 'gtol': 1e-18}
            )
            
            if result.success:
                optimized_points = result.x.reshape(-1, 3)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except Exception:
            pass
    
    # Fallback to the best initial configuration if nothing worked
    if best_points is None:
        # Find the best among initial configurations
        best_initial_ratio = -np.inf
        for config in initial_configs:
            ratio = compute_min_max_ratio(config)
            if ratio > best_initial_ratio:
                best_initial_ratio = ratio
                best_points = config.copy()
    
    return best_points


# EVOLVE-BLOCK-END
