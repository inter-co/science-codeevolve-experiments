# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import random
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric construction, global optimization, and adaptive refinement.
    
    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        points = np.array(points).reshape(-1, 2)
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        if max_dist == 0:
            return 0
        return min_dist / max_dist
    
    def objective_function(x_flat):
        """Objective function to maximize (negative ratio for minimization)."""
        points = x_flat.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    def create_physics_based_initialization():
        """Create initial configuration using physics-inspired approach with repulsive forces"""
        points = np.random.rand(16, 2)
        
        # Simulate repulsion using inverse power law (force ~ 1/r^2)
        for _ in range(300):
            forces = np.zeros_like(points)
            for i in range(16):
                for j in range(i+1, 16):
                    dx = points[i, 0] - points[j, 0]
                    dy = points[i, 1] - points[j, 1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq > 1e-10:
                        # Repulsive force with inverse cube law for better distribution
                        force_magnitude = 1.0 / (dist_sq * np.sqrt(dist_sq))
                        forces[i, 0] += force_magnitude * dx
                        forces[i, 1] += force_magnitude * dy
                        forces[j, 0] -= force_magnitude * dx
                        forces[j, 1] -= force_magnitude * dy
            
            # Apply forces with damping
            points += 0.005 * forces
            points = np.clip(points, 0, 1)
        
        return points
    
    def create_fibonacci_sphere_points(n):
        """Create points distributed according to Fibonacci sphere pattern"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            # Distribute points along the surface of a sphere using Fibonacci
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y*y)  # radius at y
            
            theta = np.arccos(y)  # polar angle
            phi_angle = i * 2.399963229728653  # Golden angle increment
            
            # Convert to Cartesian and project to 2D (take x,z coordinates)
            x = radius * np.cos(phi_angle)
            z = radius * np.sin(phi_angle)
            
            # Project to 2D unit square
            points.append([0.5 + 0.4 * x, 0.5 + 0.4 * y])
        
        return np.array(points)
    
    def create_regular_hexagonal_grid():
        """Create a regular hexagonal grid pattern"""
        points = []
        # Create a hexagonal pattern with 4 rows and 4 columns
        for i in range(4):
            for j in range(4):
                # Offset every other row for hexagonal packing
                x = 0.1 + 0.8 * j / 3
                y = 0.1 + 0.8 * i / 3
                if i % 2 == 1:
                    x += 0.8 / 6
                points.append([x, y])
        return np.array(points)
    
    def create_concentric_circles():
        """Create points in concentric circles for good coverage"""
        points = []
        # Add center point
        points.append([0.5, 0.5])
        
        # Add points in rings
        radii = [0.2, 0.35, 0.45]
        angles_per_ring = [6, 12, 16]  # More points in outer rings
        
        for i, (radius, num_angles) in enumerate(zip(radii, angles_per_ring)):
            for j in range(num_angles):
                angle = 2 * np.pi * j / num_angles
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        
        # If we don't have enough points, fill with random ones
        while len(points) < 16:
            points.append([np.random.rand(), np.random.rand()])
        
        return np.array(points[:16])
    
    def create_golden_spiral():
        """Create points using golden spiral for even distribution"""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(16):
            theta = i * 2 * np.pi / golden_ratio
            r = 0.4 * np.sqrt(i / 15.0)  # Radial distribution
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            points.append([x, y])
        return np.array(points)
    
    def create_perturbed_regular_polygon():
        """Create points in a regular polygon with perturbations"""
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4
        points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        # Add moderate perturbations to spread out points
        noise = np.random.normal(0, 0.04, points.shape)
        points += noise
        return points
    
    def create_voronoi_based():
        """Create points based on Voronoi cell maximization approach"""
        # Start with a regular grid and perturb
        points = []
        for i in range(4):
            for j in range(4):
                x = 0.1 + 0.8 * j / 3
                y = 0.1 + 0.8 * i / 3
                points.append([x, y])
        points = np.array(points)
        
        # Add small random perturbations
        points += np.random.normal(0, 0.03, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    def create_optimized_initial():
        """Create a highly optimized initial configuration"""
        # Start with a good known pattern and refine
        points = create_regular_hexagonal_grid()
        
        # Apply a quick physics simulation for better distribution
        for _ in range(50):
            forces = np.zeros_like(points)
            for i in range(16):
                for j in range(i+1, 16):
                    dx = points[i, 0] - points[j, 0]
                    dy = points[i, 1] - points[j, 1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq > 1e-10:
                        # Repulsive force with stronger decay
                        force_magnitude = 1.0 / (dist_sq * dist_sq)
                        forces[i, 0] += force_magnitude * dx
                        forces[i, 1] += force_magnitude * dy
                        forces[j, 0] -= force_magnitude * dx
                        forces[j, 1] -= force_magnitude * dy
            
            # Apply forces with damping
            points += 0.01 * forces
            points = np.clip(points, 0, 1)
        
        return points
    
    def create_balanced_distribution():
        """Create a balanced distribution combining multiple good patterns"""
        # Mix of different strategies for robustness
        points = []
        
        # 1. Start with hexagonal grid
        hex_points = create_regular_hexagonal_grid()
        points.extend(hex_points[:8])
        
        # 2. Add points from golden spiral
        spiral_points = create_golden_spiral()
        points.extend(spiral_points[8:])
        
        # 3. Add some random points for diversity
        for _ in range(2):
            points.append([np.random.rand(), np.random.rand()])
        
        # Trim to exactly 16 points
        points = np.array(points[:16])
        
        # Add small random perturbations
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        return points
    
    # Try multiple initialization strategies
    initial_strategies = [
        create_physics_based_initialization,
        create_fibonacci_sphere_points,
        create_regular_hexagonal_grid,
        create_concentric_circles,
        create_golden_spiral,
        create_perturbed_regular_polygon,
        create_voronoi_based,
        create_optimized_initial,
        create_balanced_distribution
    ]
    
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Multi-start with comprehensive initialization
    for i, strategy in enumerate(initial_strategies):
        try:
            initial_points = strategy(16)
            # Add some random perturbations to break symmetry
            np.random.seed(42 + i)
            initial_points += np.random.normal(0, 0.01, (16, 2))
            initial_points = np.clip(initial_points, 0, 1)
            
            # Try multiple optimization methods with different parameters
            methods_and_params = [
                ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('TNC', {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('SLSQP', {'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            ]
            
            for method, params in methods_and_params:
                try:
                    x0 = initial_points.flatten()
                    result = minimize(
                        objective_function,
                        x0,
                        method=method,
                        bounds=[(0, 1) for _ in range(32)],
                        options=params,
                        tol=1e-12
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape(-1, 2)
                        optimized_points = np.clip(optimized_points, 0, 1)
                        ratio = compute_min_max_ratio(optimized_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
                except Exception:
                    continue
                    
        except Exception:
            continue
    
    # Strategy 2: Global optimization with enhanced differential evolution
    if best_ratio < 0.01:
        try:
            bounds = [(0, 1) for _ in range(32)]
            # Use enhanced DE settings for better exploration with more time budget
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=200,  # More iterations for better search
                popsize=50,   # Even larger population
                mutation=(0.9, 1),  # Higher mutation rate
                recombination=0.95,  # Higher crossover rate
                seed=42,
                disp=False,
                atol=1e-12,
                rtol=1e-12
            )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 2)
                de_points = np.clip(de_points, 0, 1)
                de_ratio = compute_min_max_ratio(de_points)
                
                if de_ratio > best_ratio:
                    best_points = de_points
                    best_ratio = de_ratio
                    
        except Exception as e:
            warnings.warn(f"Differential evolution failed: {str(e)}")
    
    # Strategy 3: Additional restarts with better control
    if best_points is not None and best_ratio < 0.15:
        # Try several random restarts with better control
        for restart in range(25):  # More restarts for better chance
            try:
                # Create a new random initialization
                np.random.seed(3000 + restart)
                random_points = np.random.rand(16, 2)
                
                # Perturb the best solution slightly to get new starting point
                if restart < 15:  # Use best solution for more restarts
                    random_points = best_points + np.random.normal(0, 0.01, (16, 2))
                
                random_points = np.clip(random_points, 0, 1)
                
                # Optimize with tight tolerances and many iterations
                result = minimize(
                    objective_function,
                    random_points.flatten(),
                    method='SLSQP',
                    bounds=[(0, 1) for _ in range(32)],
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )
                
                if result.success:
                    refined_points = result.x.reshape(-1, 2)
                    refined_points = np.clip(refined_points, 0, 1)
                    ratio = compute_min_max_ratio(refined_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()
                        
            except Exception:
                continue
    
    # Strategy 4: Final energy-based refinement step with more iterations
    if best_points is not None:
        try:
            # Apply a more sophisticated physics-based refinement
            points = best_points.copy()
            for iteration in range(150):  # More iterations for better convergence
                forces = np.zeros_like(points)
                for i in range(16):
                    for j in range(i+1, 16):
                        dx = points[i, 0] - points[j, 0]
                        dy = points[i, 1] - points[j, 1]
                        dist_sq = dx*dx + dy*dy
                        if dist_sq > 1e-10:
                            # Repulsive force with careful scaling
                            force_magnitude = 1.0 / (dist_sq * dist_sq + 1e-12)
                            forces[i, 0] += force_magnitude * dx
                            forces[i, 1] += force_magnitude * dy
                            forces[j, 0] -= force_magnitude * dx
                            forces[j, 1] -= force_magnitude * dy
                
                # Apply forces with adaptive damping
                damping_factor = 0.002 + 0.001 * (iteration / 150)
                points += damping_factor * forces
                
                # Keep within bounds
                points = np.clip(points, 0, 1)
            
            # Final check
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_points = points
                best_ratio = ratio
                
        except Exception:
            pass
    
    # Final fallback if nothing worked well
    if best_points is None:
        # Return a reasonable configuration that should beat the baseline
        points = []
        for i in range(4):
            for j in range(4):
                x = (j + 0.5) / 4.0
                y = (i + 0.5) / 4.0
                points.append([x, y])
        best_points = np.array(points)
    
    return best_points


# EVOLVE-BLOCK-END
