# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
import warnings
from scipy.optimize import dual_annealing
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import numba

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@numba.jit(nopython=True)
def compute_distances_numba(points):
    """Fast computation of pairwise distances using numba"""
    n = points.shape[0]
    distances = np.zeros((n * (n - 1) // 2,))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            distances[idx] = np.sqrt(dx * dx + dy * dy)
            idx += 1
    return distances

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    
    def objective(points_flat):
        """Objective function to minimize (negative of min/max ratio)"""
        points = points_flat.reshape(-1, 2)
        
        # Use fast distance computation
        try:
            distances = compute_distances_numba(points)
        except:
            # Fallback to scipy version
            distances = pdist(points)
        
        if len(distances) == 0:
            return 1e10  # Large positive value for invalid configuration
            
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 1e-12:
            return 1e10  # Large positive value for invalid configuration
            
        # Return negative ratio (since we're minimizing)
        # Use a small epsilon to prevent numerical issues
        epsilon = 1e-12
        return -(d_min + epsilon) / (d_max + epsilon)
    
    # Improved initialization strategies - focused on better starting points
    def generate_golden_spiral():
        """Generate golden spiral configuration"""
        points = []
        n = 16
        golden_angle = 2.399963229728653  # ~2π(1 - 1/φ) where φ is golden ratio
        
        for i in range(n):
            r = np.sqrt(i / (n - 1)) * 0.4  # Radial scaling
            theta = i * golden_angle
            
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            
            # Add noise to avoid degeneracy
            x += np.random.normal(0, 0.005, 1)[0]
            y += np.random.normal(0, 0.005, 1)[0]
            
            # Clamp to [0,1]
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            points.append([x, y])
            
        return np.array(points)
    
    def generate_regular_grid():
        """Generate regular 4x4 grid"""
        points = []
        for i in range(4):
            for j in range(4):
                x = j / 3.0
                y = i / 3.0
                # Add small random perturbation to avoid regularity issues
                x += np.random.normal(0, 0.01, 1)[0]
                y += np.random.normal(0, 0.01, 1)[0]
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
        return np.array(points)
    
    def generate_concentric_circles():
        """Generate points in concentric circles for better spacing"""
        points = []
        n = 16
        
        # Distribute points in concentric rings
        radii = [0.15, 0.3, 0.4, 0.45]  # Different ring radii
        points_per_ring = [4, 4, 4, 4]  # Points per ring
        
        for i, (radius, count) in enumerate(zip(radii, points_per_ring)):
            angles = np.linspace(0, 2*np.pi, count, endpoint=False)
            for angle in angles:
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                # Add noise to avoid degeneracy
                x += np.random.normal(0, 0.005, 1)[0]
                y += np.random.normal(0, 0.005, 1)[0]
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
                
        return np.array(points[:16])
    
    def generate_hexagonal_grid():
        """Generate hexagonal grid configuration"""
        points = []
        rows = 4
        cols = 4
        
        # Create a staggered grid pattern (hexagonal-like)
        for i in range(rows):
            for j in range(cols):
                # Staggered rows
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + offset + 0.5) / cols
                y = (i + 0.5) / rows
                
                # Add small random perturbation
                x += np.random.normal(0, 0.01, 1)[0]
                y += np.random.normal(0, 0.01, 1)[0]
                
                # Clamp to [0,1] range
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
        
        return np.array(points)
    
    def generate_better_initialization():
        """Generate a better initial configuration using a combination of methods"""
        # Start with a regular grid and add small perturbations
        points = []
        for i in range(4):
            for j in range(4):
                x = j / 3.0 + np.random.uniform(-0.02, 0.02)
                y = i / 3.0 + np.random.uniform(-0.02, 0.02)
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
                points.append([x, y])
        return np.array(points)
    
    # Generate fewer, more strategically chosen initial configurations
    configs = [
        generate_golden_spiral(),        # Golden spiral - good for even distribution
        generate_regular_grid(),        # Regular grid - structured start
        generate_concentric_circles(),  # Concentric circles - good spread
        generate_hexagonal_grid(),      # Hexagonal grid - naturally good distribution
        generate_better_initialization(), # Better initialization
        np.random.rand(16, 2)           # Random configuration
    ]
    
    best_ratio = 0
    best_points = None
    
    # More targeted optimization approach
    for i, initial_config in enumerate(configs):
        initial_flat = initial_config.flatten()
        
        # Define bounds for each coordinate (0 to 1)
        bounds = [(0, 1) for _ in range(32)]
        
        try:
            # Strategy 1: Differential Evolution with better parameters
            # Use fewer restarts but more robust parameters
            for restart in range(2):
                try:
                    # Add small random perturbation to initial guess for diversity
                    perturbed_initial = initial_flat + np.random.normal(0, 0.01, 32)
                    perturbed_initial = np.clip(perturbed_initial, 0, 1)
                    
                    # Use differential evolution with higher quality settings
                    result = differential_evolution(
                        objective,
                        bounds,
                        maxiter=1000,  # Reduced iterations for speed
                        popsize=50,    # Moderate population size
                        seed=42+i+restart,
                        disp=False,
                        atol=1e-12,    # Tighter tolerances
                        rtol=1e-12,
                        strategy='best1bin'
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        
                        # Validate result
                        distances = pdist(final_points)
                        if len(distances) > 0:
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            if d_max > 1e-12:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = final_points.copy()
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    # If no good solution found, try global optimization with better parameters
    if best_points is None:
        try:
            # Use dual annealing for global optimization with more iterations
            bounds = [(0, 1) for _ in range(32)]
            
            result = dual_annealing(
                objective,
                bounds,
                maxiter=3000,   # Reduced iterations for speed
                initial_temp=2000,  # Reasonable initial temperature
                seed=42,
                no_local_search=True
            )
            
            final_points = result.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            
            # Validate and compare
            distances = pdist(final_points)
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 1e-12:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_points = final_points
                        
        except Exception as e:
            pass
    
    # If still no good solution, try a hybrid approach with more aggressive local search
    if best_points is None:
        try:
            # Start with golden spiral as baseline
            initial_points = generate_golden_spiral()
            initial_flat = initial_points.flatten()
            
            # Use L-BFGS-B with multiple restarts and better tolerance
            for restart in range(3):
                try:
                    # Add small random perturbation
                    perturbed_initial = initial_flat + np.random.normal(0, 0.01, 32)
                    perturbed_initial = np.clip(perturbed_initial, 0, 1)
                    
                    result = minimize(
                        objective,
                        perturbed_initial,
                        method='L-BFGS-B',
                        bounds=[(0, 1) for _ in range(32)],
                        options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12},  # More reasonable iterations
                        tol=1e-14
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        
                        # Validate and compare
                        distances = pdist(final_points)
                        if len(distances) > 0:
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            if d_max > 1e-12:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = final_points.copy()
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
    
    # Final refinement with local optimization using a more robust approach
    if best_points is not None:
        try:
            # Try to improve with local optimization using the best configuration
            initial_flat = best_points.flatten()
            bounds = [(0, 1) for _ in range(32)]
            
            # Use a single, more aggressive local optimization strategy
            strategies = [
                ('L-BFGS-B', {'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            ]
            
            for method, options in strategies:
                try:
                    result = minimize(
                        objective,
                        initial_flat,
                        method=method,
                        bounds=bounds,
                        options=options,
                        tol=1e-14
                    )
                    
                    if result.success:
                        final_points = result.x.reshape(-1, 2)
                        final_points = np.clip(final_points, 0, 1)
                        
                        # Validate and compare
                        distances = pdist(final_points)
                        if len(distances) > 0:
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            if d_max > 1e-12:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = final_points
                except Exception as e:
                    continue
                    
        except Exception as e:
            pass
    
    # Final validation check
    if best_points is None:
        # Last resort: return the golden spiral configuration
        return generate_golden_spiral()
    
    return best_points


# EVOLVE-BLOCK-END
