# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization, multi-start optimization, 
    and global search methods with enhanced mathematical foundations.
    
    Returns:
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
            
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    def objective_function(x_flat):
        """Objective function to minimize (negative of ratio for maximization)."""
        points = x_flat.reshape(-1, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        # Handle case where norm might be zero
        norms = np.where(norms == 0, 1.0, norms)
        points = points / norms[:, np.newaxis]
        ratio = compute_min_max_ratio(points)
        return -ratio  # Return negative because we want to maximize
    
    def generate_fibonacci_sphere(n):
        """Generate points on sphere using Fibonacci spiral."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            theta = i * 2 * np.pi / phi  # golden angle increment
            
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    def generate_icosahedral_plus_poles():
        """Generate initial configuration using icosahedral symmetry plus poles."""
        # Icosahedral vertices (normalized)
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = [
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
        ]
        
        ico_points = np.array(vertices, dtype=float)
        norms = np.linalg.norm(ico_points, axis=1)
        ico_points = ico_points / norms[:, np.newaxis]
        
        # Add 2 more points for total 14 (at poles)
        additional_points = np.array([[0, 0, 0.95], [0, 0, -0.95]])
        points = np.vstack([ico_points, additional_points])
        
        return points
    
    def generate_symmetric_config():
        """Generate highly symmetric configuration using known mathematical constructions."""
        # Start with icosahedral structure (12 vertices) plus 2 poles
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = [
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)
        ]
        
        ico_points = np.array(vertices, dtype=float)
        norms = np.linalg.norm(ico_points, axis=1)
        ico_points = ico_points / norms[:, np.newaxis]
        
        # Add 2 more points for total 14 (at poles with better spacing)
        additional_points = np.array([[0, 0, 0.95], [0, 0, -0.95]])
        points = np.vstack([ico_points, additional_points])
        
        # Apply small perturbations to break any accidental symmetries
        points += np.random.normal(0, 0.02, points.shape)
        
        return points
    
    def energy_minimization(points, max_iter=300):
        """Minimize potential energy between points on sphere to improve distribution."""
        n = points.shape[0]
        
        def energy_function(x):
            points = x.reshape((n, 3))
            distances = pdist(points)
            distances = np.maximum(distances, 1e-12)  # Avoid division by zero
            energy = np.sum(1.0 / (distances ** 2))
            return energy
        
        def constraint_sphere(x):
            points = x.reshape((n, 3))
            norms = np.linalg.norm(points, axis=1)
            return norms - 1.0
        
        x0 = points.flatten()
        bounds = [(-1.0, 1.0) for _ in range(n * 3)]
        
        try:
            result = minimize(
                energy_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': constraint_sphere},
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if result.success:
                final_points = result.x.reshape((n, 3))
                final_points = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
                return final_points
        except:
            pass
        
        return points
    
    # Enhanced initialization strategies
    initial_configs = []
    
    # Strategy 1: Highly symmetric icosahedral with optimized poles
    initial_configs.append(generate_symmetric_config())
    
    # Strategy 2: Fibonacci spiral (better distribution)
    initial_configs.append(generate_fibonacci_sphere(14))
    
    # Strategy 3: Random points with symmetry breaking
    initial_configs.append(np.random.randn(14, 3))
    
    # Strategy 4: Perturbed icosahedral with larger perturbation
    base_config = generate_symmetric_config()
    perturbed = base_config + np.random.normal(0, 0.08, base_config.shape)
    norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
    initial_configs.append(perturbed / norms)
    
    # Strategy 5: Another Fibonacci-based configuration with different parameters
    fib_points = generate_fibonacci_sphere(14)
    # Add some jitter to create variety
    fib_points += np.random.normal(0, 0.03, fib_points.shape)
    norms = np.linalg.norm(fib_points, axis=1, keepdims=True)
    initial_configs.append(fib_points / norms)
    
    best_ratio = 0
    best_points = None
    
    # Multiple optimization runs with different strategies and enhanced restarts
    for init_idx, initial_points in enumerate(initial_configs):
        # Add random perturbations to avoid degeneracy
        points = initial_points + np.random.normal(0, 0.01, initial_points.shape)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Apply energy minimization to get better initial distribution
        points = energy_minimization(points, max_iter=200)
        
        # Try multiple optimization methods with different restart strategies
        # Method 1: Local optimization (SLSQP) with multiple restarts
        for restart in range(8):  # Increased from 5 to 8 for better exploration
            if restart > 0:
                # Use varying perturbation magnitudes for better exploration
                if restart <= 3:
                    perturbation_magnitude = 0.02 + restart * 0.01
                elif restart <= 6:
                    perturbation_magnitude = 0.05 + (restart-3) * 0.02
                else:
                    perturbation_magnitude = 0.1 + (restart-6) * 0.03
                    
                perturbed_points = points + np.random.normal(0, perturbation_magnitude, points.shape)
                norms = np.linalg.norm(perturbed_points, axis=1, keepdims=True)
                points = perturbed_points / norms
            
            # Flatten for optimization
            x0 = points.flatten()
            
            # Define bounds for optimization
            bounds = [(-1.0, 1.0) for _ in range(len(x0))]
            
            # Use multiple methods for robustness
            methods = ['SLSQP', 'L-BFGS-B']  # Added L-BFGS-B for comparison
            
            method_found_success = False
            for method in methods:
                if method_found_success:
                    break
                    
                try:
                    result = minimize(
                        objective_function,
                        x0,
                        method=method,
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
                    )
                    
                    if result.success:
                        # Extract optimized points and normalize
                        optimized_points = result.x.reshape(-1, 3)
                        norms = np.linalg.norm(optimized_points, axis=1)
                        norms = np.where(norms == 0, 1.0, norms)
                        optimized_points = optimized_points / norms[:, np.newaxis]
                        
                        # Calculate ratio
                        ratio = compute_min_max_ratio(optimized_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            method_found_success = True
                except:
                    continue
        
        # Method 2: Global optimization (differential evolution) for better exploration
        try:
            # Create bounds for differential evolution
            bounds_de = [(-1.0, 1.0) for _ in range(14 * 3)]
            
            # Increase iterations for better convergence
            result_de = differential_evolution(
                objective_function,
                bounds_de,
                maxiter=100,  # Increased from 50 to 100
                popsize=15,   # Increased from 10 to 15
                mutation=(0.5, 1),
                recombination=0.7,
                seed=42 + init_idx,
                disp=False
            )
            
            if result_de.success:
                # Extract and normalize points
                optimized_points = result_de.x.reshape(-1, 3)
                norms = np.linalg.norm(optimized_points, axis=1)
                norms = np.where(norms == 0, 1.0, norms)
                optimized_points = optimized_points / norms[:, np.newaxis]
                
                # Calculate ratio
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
        except:
            continue
    
    # Final fallback: If no good solution found, use best initialization
    if best_points is None:
        # Use the symmetric configuration as final fallback
        best_points = generate_symmetric_config()
        # Add small random perturbation and normalize
        best_points += np.random.normal(0, 0.01, best_points.shape)
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
    
    # Final normalization to ensure points are on unit sphere
    norms = np.linalg.norm(best_points, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    best_points = best_points / norms[:, np.newaxis]
    
    return best_points


# EVOLVE-BLOCK-END
