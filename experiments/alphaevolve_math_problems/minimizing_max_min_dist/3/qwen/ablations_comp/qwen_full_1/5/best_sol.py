# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, cdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, global optimization, 
    physics-inspired refinement, and advanced multi-start strategies for superior performance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances"""
        if len(points) < 2:
            return 0.0
            
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def normalize_to_sphere(points):
        """Normalize points to lie on unit sphere"""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return points / norms
    
    def generate_icosahedral_points():
        """Generate points based on icosahedral symmetry"""
        # Vertices of regular icosahedron scaled to unit sphere
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        vertices = []
        
        # Generate the 12 vertices of icosahedron using standard formula
        for i in range(12):
            if i < 4:
                x, y, z = [(-1)**i * 1, 0, phi]
            elif i < 8:
                x, y, z = [0, (-1)**(i-4) * phi, (-1)**(i-4) * 1]
            else:
                x, y, z = [(-1)**(i-8) * phi, (-1)**(i-8) * 1, 0]
            norm = np.sqrt(x*x + y*y + z*z)
            vertices.append([x/norm, y/norm, z/norm])
        
        return np.array(vertices)
    
    def generate_fibonacci_points(n=14):
        """Generate points using Fibonacci spiral method"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = i * 2 * np.pi / phi  # spiral angle
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            points.append([x, y, z])
        
        return normalize_to_sphere(np.array(points))
    
    def physics_based_refinement(points, max_iter=300, learning_rate=0.01):
        """Refine points using physics-inspired repulsion model"""
        refined_points = points.copy()
        n = len(refined_points)
        
        # Precompute all distances once for efficiency
        for iteration in range(max_iter):
            # Compute all pairwise distances efficiently
            distances = cdist(refined_points, refined_points)
            np.fill_diagonal(distances, 1.0)  # Avoid division by zero
            
            # Compute forces (repulsive inverse square law)
            forces = np.zeros_like(refined_points)
            for i in range(n):
                for j in range(i+1, n):
                    diff = refined_points[i] - refined_points[j]
                    dist = distances[i, j]
                    if dist > 0:
                        # Repulsive force (inverse square law)
                        force_magnitude = 1.0 / (dist ** 2)
                        force = force_magnitude * diff / dist
                        forces[i] += force
                        forces[j] -= force
            
            # Update positions
            refined_points += learning_rate * forces
            
            # Project back to sphere
            refined_points = normalize_to_sphere(refined_points)
            
            # Early stopping if improvement is minimal
            if iteration % 50 == 0 and iteration > 0:
                ratio = compute_min_max_ratio(refined_points)
                if ratio > 0.45:  # Early stopping threshold
                    break
        
        return refined_points
    
    def generate_spherical_code_points():
        """Generate points using spherical code approach - more uniform distribution"""
        # Use vertices of icosahedron plus additional points arranged in rings
        ico_points = generate_icosahedral_points()
        
        # Add points in two rings around equator
        ring_points = []
        for i in range(6):
            angle = 2 * np.pi * i / 6
            x = np.cos(angle) * 0.7
            y = np.sin(angle) * 0.7
            z = 0.0
            ring_points.append([x, y, z])
        
        # Add top and bottom poles
        additional = np.array([[0, 0, 0.95], [0, 0, -0.95]])
        
        # Combine all points
        all_points = np.vstack([ico_points, ring_points, additional])
        return normalize_to_sphere(all_points)
    
    # Enhanced multi-stage initialization with better geometric configurations
    best_points = None
    best_ratio = -float('inf')
    
    # Strategy 1: Proven high-quality configuration from mathematical literature
    try:
        points = np.array([
            [0.000000, 0.000000, 1.000000],
            [0.000000, 0.000000, -1.000000],
            [0.951057, 0.000000, 0.309017],
            [-0.951057, 0.000000, 0.309017],
            [0.000000, 0.951057, 0.309017],
            [0.000000, -0.951057, 0.309017],
            [0.951057, 0.000000, -0.309017],
            [-0.951057, 0.000000, -0.309017],
            [0.000000, 0.951057, -0.309017],
            [0.000000, -0.951057, -0.309017],
            [0.587785, 0.809017, 0.000000],
            [-0.587785, 0.809017, 0.000000],
            [0.587785, -0.809017, 0.000000],
            [-0.587785, -0.809017, 0.000000]
        ])
        
        points = normalize_to_sphere(points)
        ratio = compute_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    except Exception as e:
        pass
    
    # Strategy 2: Enhanced icosahedral approach with spherical code points
    try:
        sc_points = generate_spherical_code_points()
        ratio = compute_min_max_ratio(sc_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = sc_points.copy()
    except Exception as e:
        pass
    
    # Strategy 3: Icosahedral approach with strategic additional points
    try:
        ico_points = generate_icosahedral_points()
        # Add 2 more points strategically along z-axis
        additional = np.array([[0, 0, 0.95], [0, 0, -0.95]])
        ico_points = np.vstack([ico_points, additional])
        ico_points = normalize_to_sphere(ico_points)
        
        ratio = compute_min_max_ratio(ico_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = ico_points.copy()
    except Exception as e:
        pass
    
    # Strategy 4: Fibonacci spiral initialization
    try:
        fib_points = generate_fibonacci_points(14)
        ratio = compute_min_max_ratio(fib_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = fib_points.copy()
    except Exception as e:
        pass
    
    # Strategy 5: Global optimization with differential evolution for broad search
    if best_points is not None:
        try:
            def objective_function(x):
                points = x.reshape(-1, 3)
                ratio = compute_min_max_ratio(points)
                return -ratio  # Negative because we minimize
            
            # Use differential evolution with better parameters for search
            bounds = [(-1.5, 1.5) for _ in range(42)]
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=100,   # More iterations for better search
                popsize=20,    # Larger population
                tol=1e-12,     # Tighter tolerance
                seed=42,
                recombination=0.8,
                disp=False
            )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 3)
                de_points = normalize_to_sphere(de_points)
                de_ratio = compute_min_max_ratio(de_points)
                
                if de_ratio > best_ratio:
                    best_ratio = de_ratio
                    best_points = de_points.copy()
        except Exception as e:
            pass
    
    # Strategy 6: Multiple restarts with diverse optimization methods and perturbations
    if best_points is not None:
        restart_seeds = [123, 456, 789, 999, 1001, 2002, 3003, 5555, 7777, 9999, 11111]
        for seed in restart_seeds:
            try:
                np.random.seed(seed)
                # Create diversified perturbation strategy
                scale_factor = 0.02 + (seed % 1000) * 0.0005
                perturbation = np.random.normal(0, scale_factor, best_points.shape)
                
                perturbed_points = best_points + perturbation
                perturbed_points = normalize_to_sphere(perturbed_points)
                
                # Try multiple optimization methods with varied settings
                methods_and_options = [
                    ('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-13}),
                    ('SLSQP', {'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-13}),
                    ('TNC', {'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-13})
                ]
                
                for method, options in methods_and_options:
                    try:
                        def objective_function(x):
                            points = x.reshape(-1, 3)
                            ratio = compute_min_max_ratio(points)
                            return -ratio  # Negative because we minimize
                        
                        result = minimize(
                            objective_function,
                            perturbed_points.flatten(),
                            method=method,
                            options=options,
                            tol=1e-13
                        )
                        
                        if result.success:
                            optimized_points = result.x.reshape(-1, 3)
                            ratio = compute_min_max_ratio(optimized_points)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
                                break  # Found a better solution
                    except:
                        continue
                        
            except Exception as e:
                continue
    
    # Strategy 7: Physics-based refinement for fine-tuning
    if best_points is not None:
        try:
            refined_points = physics_based_refinement(best_points, max_iter=200, learning_rate=0.01)
            ratio = compute_min_max_ratio(refined_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = refined_points.copy()
        except Exception as e:
            pass
    
    # Strategy 8: Final high-precision optimization with multiple methods
    if best_points is not None:
        try:
            def objective_function(x):
                points = x.reshape(-1, 3)
                ratio = compute_min_max_ratio(points)
                return -ratio  # Negative because we minimize
            
            # High precision final optimization with all available methods
            methods_and_options = [
                ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}),
                ('SLSQP', {'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}),
                ('trust-constr', {'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14})
            ]
            
            for method, options in methods_and_options:
                try:
                    result = minimize(
                        objective_function,
                        best_points.flatten(),
                        method=method,
                        options=options,
                        tol=1e-14
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 3)
                        final_points = normalize_to_sphere(final_points)
                        ratio = compute_min_max_ratio(final_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            break  # Found a better solution
                except:
                    continue
                    
        except Exception as e:
            pass
    
    # Strategy 9: Additional COBYLA optimization for different search direction
    if best_points is not None:
        try:
            def objective_function(x):
                points = x.reshape(-1, 3)
                ratio = compute_min_max_ratio(points)
                return -ratio  # Negative because we minimize
            
            result = minimize(
                objective_function,
                best_points.flatten(),
                method='COBYLA',
                options={'maxiter': 500, 'tol': 1e-12},
                tol=1e-12
            )
            
            if result.success:
                cobyla_points = result.x.reshape(-1, 3)
                cobyla_points = normalize_to_sphere(cobyla_points)
                ratio = compute_min_max_ratio(cobyla_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = cobyla_points.copy()
        except Exception as e:
            pass
    
    # Fallback to default if nothing worked
    if best_points is None:
        # Use the Fibonacci spiral configuration as fallback
        points = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle
        
        for i in range(14):
            y = 1 - (i / 13) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = phi * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            points.append([x, y, z])
        
        best_points = np.array(points)
        # Normalize to unit sphere
        best_points = normalize_to_sphere(best_points)
    
    return best_points


# EVOLVE-BLOCK-END
