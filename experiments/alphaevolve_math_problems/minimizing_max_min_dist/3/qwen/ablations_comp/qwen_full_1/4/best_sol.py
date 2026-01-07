# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, global optimization, and 
    multiple local refinement strategies inspired by advanced optimization techniques.

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
    
    def objective_function(x):
        """Objective function to minimize (negative of ratio)"""
        # Reshape x into points
        points = x.reshape(-1, 3)
        ratio = compute_min_max_ratio(points)
        # Return negative because we want to maximize
        return -ratio
    
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
        
        # Generate the 12 vertices of icosahedron
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
        
        return np.array(points)
    
    def generate_octahedral_points():
        """Generate points based on octahedral symmetry"""
        # Octahedron vertices plus additional points
        points = [
            [1, 0, 0], [-1, 0, 0],  # x-axis
            [0, 1, 0], [0, -1, 0],  # y-axis  
            [0, 0, 1], [0, 0, -1]   # z-axis
        ]
        
        # Add 8 more points for better coverage
        # These are the vertices of a cube inscribed in the sphere
        for i in range(8):
            x = 1 if i & 1 else -1
            y = 1 if i & 2 else -1
            z = 1 if i & 4 else -1
            points.append([x, y, z])
        
        # Normalize to unit sphere
        points = np.array(points)
        return normalize_to_sphere(points[:14])  # Keep only 14 points
    
    # Multi-stage initialization with diverse geometric configurations
    strategies = []
    
    # Strategy 1: Known good configuration (from inspiration 1)
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
        strategies.append(("known_good", normalize_to_sphere(points)))
    except Exception:
        pass
    
    # Strategy 2: Icosahedral approach (from inspiration 2)
    try:
        ico_points = generate_icosahedral_points()
        # Add two more points to make 14 (north/south poles)
        additional = np.array([[0, 0, 0.98], [0, 0, -0.98]])  # Slightly better spacing
        ico_points = np.vstack([ico_points, additional])
        strategies.append(("icosahedral", normalize_to_sphere(ico_points)))
    except Exception:
        pass
    
    # Strategy 3: Fibonacci spiral approach (from inspiration 2)
    try:
        fib_points = generate_fibonacci_points(14)
        strategies.append(("fibonacci", normalize_to_sphere(fib_points)))
    except Exception:
        pass
    
    # Strategy 4: Octahedral approach (from inspiration 3)
    try:
        oct_points = generate_octahedral_points()
        strategies.append(("octahedral", oct_points))
    except Exception:
        pass
    
    # Strategy 5: Random initialization (for diversity)
    try:
        np.random.seed(42)
        random_points = np.random.uniform(-1, 1, (14, 3))
        strategies.append(("random", normalize_to_sphere(random_points)))
    except Exception:
        pass
    
    best_points = None
    best_ratio = -float('inf')
    
    # Try each strategy with optimization - enhanced approach
    for strategy_name, initial_points in strategies:
        # Multiple optimization attempts with different restarts
        restart_configs = [
            (123, 0.01),   # Very small perturbation for fine-tuning
            (456, 0.02),   # Small perturbation
            (789, 0.03),   # Medium perturbation
            (999, 0.05),   # Larger perturbation for exploration
        ]
        
        for seed, perturb_scale in restart_configs:
            try:
                np.random.seed(seed)
                # Create perturbed version with adaptive scaling
                perturbation = np.random.normal(0, perturb_scale, initial_points.shape)
                perturbed_points = initial_points + perturbation
                
                # Project back to sphere
                perturbed_points = normalize_to_sphere(perturbed_points)
                
                # Try multiple optimization methods for robustness
                methods = ['L-BFGS-B', 'SLSQP']  # Try both methods
                for method in methods:
                    try:
                        result = minimize(
                            objective_function,
                            perturbed_points.flatten(),
                            method=method,
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12},
                            tol=1e-12
                        )
                        
                        if result.success:
                            optimized_points = result.x.reshape(-1, 3)
                            ratio = compute_min_max_ratio(optimized_points)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
                                # Early exit if we found a significantly good solution
                                if ratio > 0.25:  # Early stopping for very good solutions
                                    return best_points
                    except Exception:
                        continue
                        
            except Exception:
                continue
    
    # Enhanced global search with differential evolution (inspiration 1)
    if best_points is not None:
        try:
            # Use differential evolution with parameters similar to best inspirations
            bounds = [(-1.5, 1.5) for _ in range(42)]
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=75,   # Increased iterations for better search
                popsize=15,   # Moderate population size
                tol=1e-12,    # Tighter tolerance
                seed=42,
                recombination=0.8,  # Good recombination rate
                disp=False
            )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 3)
                de_points = normalize_to_sphere(de_points)
                de_ratio = compute_min_max_ratio(de_points)
                if de_ratio > best_ratio:
                    best_ratio = de_ratio
                    best_points = de_points.copy()
        except Exception:
            pass
    
    # Final comprehensive refinement with multiple methods
    if best_points is not None:
        try:
            # Try with multiple methods using aggressive settings
            methods = ['L-BFGS-B', 'SLSQP']
            for method in methods:
                try:
                    result = minimize(
                        objective_function,
                        best_points.flatten(),
                        method=method,
                        options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-14},
                        tol=1e-14
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 3)
                        final_points = normalize_to_sphere(final_points)
                        ratio = compute_min_max_ratio(final_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                except Exception:
                    continue
                    
        except Exception:
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
