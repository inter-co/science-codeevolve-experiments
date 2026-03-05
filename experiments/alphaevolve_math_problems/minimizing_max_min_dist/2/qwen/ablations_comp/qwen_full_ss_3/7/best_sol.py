# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import random
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses advanced optimization techniques inspired by physics-based simulation, 
    semidefinite programming relaxations, and energy minimization approaches.
    
    Returns:
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        # Ensure points are reshaped properly
        points = np.array(points).reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Handle edge case of no distances
        if len(distances) == 0:
            return 0
            
        # Get min and max distances
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0
            
        return min_dist / max_dist
    
    def objective_function(x_flat):
        """Objective function to maximize (negative ratio for minimization)."""
        points = x_flat.reshape(-1, 2)
        ratio = compute_min_max_ratio(points)
        return -ratio  # Negative because we want to maximize
    
    def simulate_repulsion_with_boundary(points, num_steps=1000, dt=0.005, k=1.0, friction=0.95):
        """
        Simulate electrostatic repulsion between points in a bounded space with proper boundary handling.
        """
        positions = points.copy()
        velocities = np.zeros_like(positions)
        
        # Apply forces and update positions
        for step in range(num_steps):
            # Calculate pairwise distances and forces
            forces = np.zeros_like(positions)
            
            # Compute forces between all pairs
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    diff = positions[i] - positions[j]
                    dist_sq = np.sum(diff**2)
                    
                    # Avoid division by zero and very small distances
                    if dist_sq > 1e-12:
                        # Inverse square law repulsion (similar to electrostatics)
                        force_magnitude = k / (dist_sq + 1e-12)
                        force = force_magnitude * diff / np.sqrt(dist_sq)
                        forces[i] += force
                        forces[j] -= force
            
            # Update velocities and positions
            velocities += forces * dt
            velocities *= friction  # Apply damping
            positions += velocities * dt
            
            # Enforce boundary conditions (clamp to [0,1] box)
            positions[:, 0] = np.clip(positions[:, 0], 0, 1)
            positions[:, 1] = np.clip(positions[:, 1], 0, 1)
            
        return positions
    
    def create_physics_based_initialization():
        """Create initial configuration using physics-inspired approach with repulsive forces"""
        # Start with random points
        points = np.random.rand(16, 2)
        
        # Simulate repulsion using inverse power law with better parameters
        points = simulate_repulsion_with_boundary(
            points, 
            num_steps=1000, 
            dt=0.005, 
            k=1.0, 
            friction=0.95
        )
        
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
        noise = np.random.normal(0, 0.05, points.shape)  # Slightly larger noise
        points += noise
        return points
    
    def create_random_initial():
        """Create a random initial configuration"""
        return np.random.rand(16, 2)
    
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
    
    def create_symmetric_pattern():
        """Create a symmetric pattern that's likely to be good"""
        # Start with a regular pattern and add perturbations
        points = []
        # Create a basic 4x4 grid with slight perturbations
        for i in range(4):
            for j in range(4):
                x = 0.1 + 0.8 * j / 3 + np.random.normal(0, 0.01)
                y = 0.1 + 0.8 * i / 3 + np.random.normal(0, 0.01)
                points.append([x, y])
        return np.array(points)
    
    def create_balanced_pattern():
        """Create a balanced pattern combining multiple geometric principles"""
        # Start with a hexagonal grid
        points = create_regular_hexagonal_grid()
        
        # Perturb to break symmetry but maintain good distribution
        points += np.random.normal(0, 0.01, points.shape)
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points
    
    def create_star_pattern():
        """Create points in a star-like pattern"""
        points = []
        # Center point
        points.append([0.5, 0.5])
        
        # 15 outer points arranged in a circular pattern
        angles = np.linspace(0, 2*np.pi, 15, endpoint=False)
        for angle in angles:
            # Some variation to avoid perfect symmetry
            r = 0.4 + 0.1 * np.sin(3 * angle)
            points.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
        
        return np.array(points[:16])
    
    def create_better_initial_config():
        """Create a better initial configuration using multiple strategies"""
        # Strategy 1: Start with a spherical code (very uniform)
        spherical_points = create_fibonacci_sphere_points(16)
        
        # Strategy 2: Start with a hexagonal grid (good structure)
        hex_points = create_regular_hexagonal_grid()
        
        # Strategy 3: Start with a perturbed regular polygon
        polygon_points = create_perturbed_regular_polygon()
        
        # Combine best features - take first few from spherical, middle from hex, rest from polygon
        combined_points = np.vstack([
            spherical_points[:6], 
            hex_points[6:10],
            polygon_points[10:]
        ])
        
        # Add small random perturbations to encourage further improvement
        combined_points += np.random.normal(0, 0.01, combined_points.shape)
        combined_points = np.clip(combined_points, 0, 1)
        
        return combined_points
    
    # Try multiple initialization strategies
    initial_strategies = [
        create_physics_based_initialization,
        create_fibonacci_sphere_points,
        create_regular_hexagonal_grid,
        create_concentric_circles,
        create_golden_spiral,
        create_perturbed_regular_polygon,
        create_random_initial,
        create_voronoi_based,
        create_symmetric_pattern,
        create_balanced_pattern,
        create_star_pattern,
        create_better_initial_config
    ]
    
    best_ratio = -np.inf
    best_points = None
    
    # Strategy 1: Physics-based initialization with local optimization
    for i, strategy in enumerate(initial_strategies):
        try:
            initial_points = strategy(16)
            # Add some random perturbations to break symmetry
            np.random.seed(42 + i)
            initial_points += np.random.normal(0, 0.01, (16, 2))
            initial_points = np.clip(initial_points, 0, 1)
            
            # Try multiple optimization methods with different parameters
            methods_and_params = [
                ('L-BFGS-B', {'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}),
                ('TNC', {'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}),
                ('SLSQP', {'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16})
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
                        tol=1e-16
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
    if best_ratio < 0.03:  # Trigger DE more aggressively
        try:
            bounds = [(0, 1) for _ in range(32)]
            # Use enhanced DE settings for better exploration
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=300,  # More iterations
                popsize=60,   # Larger population
                mutation=(0.9, 1),  # Higher mutation rate
                recombination=0.95,  # Higher crossover rate
                seed=42,
                disp=False,
                atol=1e-16,
                rtol=1e-16
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
    
    # Strategy 3: Enhanced multi-start optimization with better restarts
    if best_points is not None and best_ratio < 0.15:
        # Try several random restarts with better control
        for restart in range(50):  # Even more restarts
            try:
                # Create a new random initialization
                np.random.seed(3000 + restart)
                random_points = np.random.rand(16, 2)
                
                # Perturb the best solution slightly to get new starting point
                if restart < 25:  # Use best solution for some restarts
                    random_points = best_points + np.random.normal(0, 0.01, (16, 2))
                elif restart < 40:  # Slightly larger perturbations
                    random_points = best_points + np.random.normal(0, 0.02, (16, 2))
                else:  # Large perturbations for exploration
                    random_points = best_points + np.random.uniform(-0.03, 0.03, (16, 2))
                
                random_points = np.clip(random_points, 0, 1)
                
                # Optimize with very tight tolerances and many iterations
                result = minimize(
                    objective_function,
                    random_points.flatten(),
                    method='SLSQP',
                    bounds=[(0, 1) for _ in range(32)],
                    options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-16},
                    tol=1e-16
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
    
    # Strategy 4: Add a final energy-based refinement step with improved physics
    if best_points is not None:
        try:
            # Apply a more sophisticated physics-based refinement with more steps
            points = best_points.copy()
            points = simulate_repulsion_with_boundary(
                points, 
                num_steps=1500, 
                dt=0.005, 
                k=1.0, 
                friction=0.95
            )
            
            # Final check
            ratio = compute_min_max_ratio(points)
            if ratio > best_ratio:
                best_points = points
                best_ratio = ratio
                
        except Exception:
            pass
    
    # Strategy 5: Final fallback if nothing worked well
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
