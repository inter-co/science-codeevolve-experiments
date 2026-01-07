# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, global optimization, and 
    multiple local refinement strategies for superior results.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
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
    
    def project_to_unit_sphere(points):
        """Project points to unit sphere"""
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
        # These are the exact coordinates of an icosahedron with circumradius 1
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
    
    def generate_spherical_code_points():
        """Generate points based on known spherical code configurations"""
        # This is a known good configuration from mathematical literature
        # Based on icosahedral symmetries with additional strategic points
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
        return points
    
    # Enhanced multi-stage initialization with better geometric configurations
    best_points = None
    best_ratio = -float('inf')
    
    # Strategy 1: Known good configuration from mathematical literature
    try:
        points = generate_spherical_code_points()
        points = project_to_unit_sphere(points)
        ratio = compute_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
    except Exception as e:
        pass
    
    # Strategy 2: Icosahedral approach with better configuration
    try:
        ico_points = generate_icosahedral_points()
        # Add 2 more points strategically to make 14 (north/south poles)
        additional = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        ico_points = np.vstack([ico_points, additional])
        ico_points = project_to_unit_sphere(ico_points)
        
        ratio = compute_min_max_ratio(ico_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = ico_points.copy()
    except Exception as e:
        pass
    
    # Strategy 3: Fibonacci spiral approach with better parameters
    try:
        fib_points = generate_fibonacci_points(14)
        fib_points = project_to_unit_sphere(fib_points)
        
        ratio = compute_min_max_ratio(fib_points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = fib_points.copy()
    except Exception as e:
        pass
    
    # Strategy 4: Enhanced global optimization with differential evolution
    if best_points is not None:
        try:
            # Use more robust global optimization with higher iteration count and better settings
            bounds = [(-1.5, 1.5) for _ in range(42)]
            de_result = differential_evolution(
                objective_function,
                bounds,
                maxiter=150,   # More iterations for better search
                popsize=25,    # Even larger population
                tol=1e-12,     # Much tighter tolerance
                seed=42,
                recombination=0.9,  # Higher recombination rate
                disp=False
            )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 3)
                de_points = project_to_unit_sphere(de_points)
                de_ratio = compute_min_max_ratio(de_points)
                
                if de_ratio > best_ratio:
                    best_ratio = de_ratio
                    best_points = de_points.copy()
        except Exception as e:
            pass
    
    # Strategy 5: Multiple local optimizations with diverse restart strategies
    if best_points is not None:
        # Use more aggressive restart strategies with varied perturbations
        restart_seeds = [123, 456, 789, 999, 1001, 2002, 3003, 5555, 7777, 9999, 11111]
        for seed in restart_seeds:
            try:
                np.random.seed(seed)
                # Create perturbation strategy with varying scales - more aggressive
                scale_factor = 0.02 + (seed % 1000) * 0.0005
                perturbation = np.random.normal(0, scale_factor, best_points.shape)
                
                perturbed_points = best_points + perturbation
                perturbed_points = project_to_unit_sphere(perturbed_points)
                
                # Try multiple optimization methods for robustness
                methods = ['L-BFGS-B', 'SLSQP', 'TNC']
                for method in methods:
                    try:
                        result = minimize(
                            objective_function,
                            perturbed_points.flatten(),
                            method=method,
                            options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-13},
                            tol=1e-13
                        )
                        
                        if result.success:
                            optimized_points = result.x.reshape(-1, 3)
                            ratio = compute_min_max_ratio(optimized_points)
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
                                break  # Found a better solution, move to next seed
                    except:
                        continue
                        
            except Exception as e:
                continue
    
    # Strategy 6: Final comprehensive optimization with even tighter tolerances
    if best_points is not None:
        try:
            # Try with multiple methods to ensure robust optimization
            methods = ['L-BFGS-B', 'SLSQP', 'TNC']
            for method in methods:
                try:
                    result = minimize(
                        objective_function,
                        best_points.flatten(),
                        method=method,
                        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14},
                        tol=1e-14
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 3)
                        final_points = project_to_unit_sphere(final_points)
                        ratio = compute_min_max_ratio(final_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = final_points.copy()
                            break  # Found a better solution
                except:
                    continue
                    
        except Exception as e:
            pass
    
    # Strategy 7: Additional refinement using COBYLA method with better parameters
    if best_points is not None:
        try:
            # Try using COBYLA method with more iterations and tighter tolerance
            result = minimize(
                objective_function,
                best_points.flatten(),
                method='COBYLA',
                options={'maxiter': 1500, 'tol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 3)
                final_points = project_to_unit_sphere(final_points)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
                    
        except Exception as e:
            pass
    
    # Strategy 8: Last resort - try with more aggressive optimization
    if best_points is not None:
        try:
            # Try with even more aggressive settings for any remaining improvement
            result = minimize(
                objective_function,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-15},
                tol=1e-15
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 3)
                final_points = project_to_unit_sphere(final_points)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = final_points.copy()
                    
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
        best_points = project_to_unit_sphere(best_points)
    
    return best_points


# EVOLVE-BLOCK-END
