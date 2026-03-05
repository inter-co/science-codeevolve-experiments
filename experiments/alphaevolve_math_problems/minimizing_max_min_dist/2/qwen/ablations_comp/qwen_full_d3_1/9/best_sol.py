# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses advanced optimization techniques combining global and local search methods,
    leveraging both gradient-based and evolutionary algorithms for robust optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    d = 2
    
    # Direct objective function that computes exact min/max ratio
    def exact_objective(points_flat):
        points = points_flat.reshape(-1, 2)
        distances = squareform(pdist(points))
        # Set diagonal to large value to avoid self-distances
        np.fill_diagonal(distances, np.inf)
        min_dist = np.min(distances)
        # Set diagonal to zero to avoid self-distances for max
        np.fill_diagonal(distances, 0)
        max_dist = np.max(distances)
        # Return negative ratio to maximize ratio (minimize negative ratio)
        if max_dist <= 0:
            return float('inf')
        return -min_dist / max_dist
    
    # Energy-based objective that mimics repulsive forces between points
    def energy_objective(points_flat):
        points = points_flat.reshape(-1, 2)
        distances = squareform(pdist(points))
        # Set diagonal to large value to avoid self-distances
        np.fill_diagonal(distances, np.inf)
        # Use inverse distance squared as repulsion energy
        # This encourages points to spread out
        energies = 1.0 / (distances + 1e-10)**2
        # Sum all pairwise energies (we want to maximize minimum distance, so minimize total energy)
        return np.sum(energies) / 2  # Divide by 2 to avoid double counting
    
    # Improved smooth approximation that's more numerically stable
    def smooth_min_distance(points_flat, k=1000):
        points = points_flat.reshape(-1, 2)
        distances = squareform(pdist(points))
        # Set diagonal to large value to avoid self-distances
        np.fill_diagonal(distances, np.inf)
        # More numerically stable smooth approximation with error handling
        distances_safe = np.maximum(distances, 1e-12)  # Prevent log(0)
        try:
            smooth_min = -np.log(np.sum(np.exp(-k * distances_safe))) / k
            # Add safety check for extreme values
            if np.isnan(smooth_min) or np.isinf(smooth_min):
                return 1e10
            return smooth_min
        except:
            return 1e10
    
    # Smooth max distance with better numerical stability
    def smooth_max_distance(points_flat, k=1000):
        points = points_flat.reshape(-1, 2)
        distances = squareform(pdist(points))
        # Set diagonal to zero to avoid self-distances
        np.fill_diagonal(distances, 0)
        # Clip distances to prevent overflow in exponential and add safety
        distances_safe = np.minimum(distances, 100.0)  # Prevent overflow
        distances_safe = np.maximum(distances_safe, 1e-12)  # Prevent log(0)
        try:
            smooth_max = np.log(np.sum(np.exp(k * distances_safe))) / k
            # Add safety check for extreme values
            if np.isnan(smooth_max) or np.isinf(smooth_max):
                return 1e10
            return smooth_max
        except:
            return 1e10
    
    # Objective function with smoothing for differentiability
    def objective_with_smoothing(points_flat):
        min_dist = smooth_min_distance(points_flat, k=1000)
        max_dist = smooth_max_distance(points_flat, k=1000)
        # We want to maximize min_dist/max_dist, so we minimize -min_dist/max_dist
        # But we also want to penalize when max_dist approaches 0
        if max_dist < 1e-10:
            return 1e10
        return -min_dist / max_dist
    
    # Initialize with a better starting configuration using known good patterns
    def generate_improved_hexagonal_initial():
        # Create a more optimized hexagonal pattern
        points = []
        rows = 4
        cols = 4
        spacing = 1.0
        
        for i in range(rows):
            for j in range(cols):
                x = j * spacing + (i % 2) * spacing/2
                y = i * spacing * np.sqrt(3)/2
                points.append([x, y])
        
        # Normalize to unit square [0,1] x [0,1]
        points = np.array(points[:n])
        # Handle edge cases more carefully
        if points[:, 0].max() > points[:, 0].min():
            points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
        else:
            points[:, 0] = 0.5  # Set to middle if all same
            
        if points[:, 1].max() > points[:, 1].min():
            points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())
        else:
            points[:, 1] = 0.5  # Set to middle if all same
            
        # Apply slight random perturbations to break symmetries
        np.random.seed(42)
        for i in range(n):
            points[i, 0] += (np.random.rand() - 0.5) * 0.05
            points[i, 1] += (np.random.rand() - 0.5) * 0.05
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points
    
    # Golden spiral with better normalization
    def golden_spiral_initial():
        points = np.zeros((n, 2))
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            angle = 2 * np.pi * i / phi
            radius = np.sqrt(i / (n - 1)) if i < n - 1 else 1
            points[i] = [radius * np.cos(angle), radius * np.sin(angle)]
        # Normalize to [0,1] x [0,1] with careful edge case handling
        if points[:, 0].max() > points[:, 0].min():
            points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
        else:
            points[:, 0] = 0.5
            
        if points[:, 1].max() > points[:, 1].min():
            points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())
        else:
            points[:, 1] = 0.5
            
        # Add small random perturbations to break symmetries
        np.random.seed(42)
        for i in range(n):
            points[i, 0] += (np.random.rand() - 0.5) * 0.02
            points[i, 1] += (np.random.rand() - 0.5) * 0.02
            
        return np.clip(points, 0, 1)
    
    # Concentric circle pattern for better coverage
    def concentric_circles_initial():
        points = []
        # Create points in concentric circles with better distribution
        radii = [0.1, 0.25, 0.5, 0.75, 0.9]  # More evenly spaced radii
        points_per_ring = [1, 3, 6, 4, 2]  # Adjust for 16 points total
        
        idx = 0
        for r, count in zip(radii, points_per_ring):
            if idx + count > n:
                count = n - idx
            if count > 0:
                angles = np.linspace(0, 2*np.pi, count+1)[:-1]  # Avoid duplicate at 2pi
                for angle in angles:
                    points.append([r * np.cos(angle) + 0.5, r * np.sin(angle) + 0.5])
                    idx += 1
                    if idx >= n:
                        break
            if idx >= n:
                break
        
        # Fill remaining points randomly but with better distribution
        remaining = n - len(points)
        if remaining > 0:
            for i in range(remaining):
                # Distribute remaining points more systematically
                points.append([np.random.rand(), np.random.rand()])
            
        points_array = np.array(points[:n])
        
        # Apply small perturbations to improve distribution
        np.random.seed(42)
        for i in range(len(points_array)):
            points_array[i, 0] += (np.random.rand() - 0.5) * 0.03
            points_array[i, 1] += (np.random.rand() - 0.5) * 0.03
            
        return np.clip(points_array, 0, 1)
    
    # Fibonacci sphere-like distribution for better point spread
    def fibonacci_sphere_initial():
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            # Project Fibonacci sequence onto circle for better distribution
            y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            
            theta = np.arccos(y)  # angle from z-axis
            
            # Use Fibonacci-like angular distribution
            angle = 2 * np.pi * i / phi
            
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            
            # Map to 2D square [0,1] x [0,1]
            points.append([(x + 1) / 2, (z + 1) / 2])
        
        points_array = np.array(points)
        # Apply small random perturbations
        np.random.seed(42)
        for i in range(n):
            points_array[i, 0] += (np.random.rand() - 0.5) * 0.02
            points_array[i, 1] += (np.random.rand() - 0.5) * 0.02
            
        return np.clip(points_array, 0, 1)
    
    # Quasi-random Sobol sequence initialization for even better distribution
    def sobol_initial():
        try:
            from scipy.stats.qmc import Sobol
            # Generate quasi-random points in [0,1]^2
            sampler = Sobol(d=2, scramble=True)
            samples = sampler.random(n)
            # Scale to [0,1] x [0,1] and apply small perturbations
            points = samples.copy()
            np.random.seed(42)
            points += (np.random.rand(n, 2) - 0.5) * 0.05
            points = np.clip(points, 0, 1)
            return points
        except ImportError:
            # Fallback to random if Sobol not available
            return np.random.rand(n, d)
    
    # Multi-start optimization with diverse initializations and global search
    best_result = None
    best_ratio = -np.inf
    
    # Set up seeds for reproducibility
    np.random.seed(42)
    
    # Strategy 1: Differential Evolution for global search (very important for this problem)
    try:
        bounds = [(0, 1) for _ in range(2 * n)]
        # Use more aggressive settings for differential evolution to find good global solutions quickly
        de_result = differential_evolution(
            exact_objective,
            bounds,
            maxiter=75,   # Increased iterations for better global search
            popsize=25,   # Larger population for better exploration
            seed=42,
            disp=False,
            tol=1e-11,    # Slightly less strict tolerance to save time
            mutation=(0.5, 1.0),  # Standard mutation
            recombination=0.7     # Standard crossover rate
        )
        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            dist_matrix = squareform(pdist(de_points))
            np.fill_diagonal(dist_matrix, np.inf)
            min_dist = np.min(dist_matrix)
            np.fill_diagonal(dist_matrix, 0)
            max_dist = np.max(dist_matrix)
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = de_points.copy()
    except Exception:
        pass
    
    # Enhanced initial strategies with more diverse and higher-quality patterns
    initial_strategies = [
        lambda: generate_improved_hexagonal_initial(),   # Improved hexagonal pattern
        lambda: golden_spiral_initial(),                 # Golden spiral with perturbations
        lambda: concentric_circles_initial(),           # Enhanced concentric circles
        lambda: fibonacci_sphere_initial(),             # Fibonacci sphere-inspired
        lambda: sobol_initial(),                        # Quasi-random Sobol sequence
        lambda: np.random.rand(n, d),                   # Random uniform
        lambda: np.random.normal(0.5, 0.15, (n, d)),    # Normal distribution
        lambda: np.random.uniform(0, 1, (n, d)),         # Uniform random
        # Additional high-quality initializations
        lambda: np.random.rand(n, d) * 0.8 + 0.1,       # Centered random
        lambda: np.random.normal(0.5, 0.1, (n, d)),     # Tight normal
    ]
    
    # Run optimization with multiple starts using smooth objective first
    # Use more iterations for better convergence but stay within time limits
    num_starts = min(len(initial_strategies), 15)  # Increase number of starts to explore more
    for i in range(num_starts):
        try:
            # Generate initial points
            initial_points = initial_strategies[i]()
            initial_flat = initial_points.flatten()
            
            # Optimization with bounds using smooth objective for better convergence
            bounds = [(0, 1) for _ in range(2*n)]
            
            # Use L-BFGS-B with more aggressive settings for better convergence
            result = minimize(
                objective_with_smoothing,
                initial_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            # Extract final points
            final_points = result.x.reshape(-1, 2)
            
            # Ensure points are within bounds
            final_points = np.clip(final_points, 0, 1)
            
            # Calculate actual ratio using exact computation
            dist_matrix = squareform(pdist(final_points))
            np.fill_diagonal(dist_matrix, np.inf)
            min_dist = np.min(dist_matrix)
            np.fill_diagonal(dist_matrix, 0)
            max_dist = np.max(dist_matrix)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = final_points.copy()
                    
        except Exception as e:
            continue
    
    # If we have no valid results, fall back to improved hexagonal pattern
    if best_result is None:
        return generate_improved_hexagonal_initial()
    
    # Enhanced refinement using multiple approaches
    # First try energy-based refinement to improve distribution
    try:
        bounds = [(0, 1) for _ in range(2*n)]
        result_energy = minimize(
            energy_objective,
            best_result.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        
        if result_energy.success:
            energy_points = result_energy.x.reshape(-1, 2)
            energy_points = np.clip(energy_points, 0, 1)
            
            # Calculate ratio after energy refinement
            dist_matrix = squareform(pdist(energy_points))
            np.fill_diagonal(dist_matrix, np.inf)
            min_dist = np.min(dist_matrix)
            np.fill_diagonal(dist_matrix, 0)
            max_dist = np.max(dist_matrix)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_result = energy_points.copy()
    except Exception:
        pass
    
    # Final refinement using exact objective function on the best result
    final_flat = best_result.flatten()
    bounds = [(0, 1) for _ in range(2*n)]
    
    try:
        # Refine with exact objective function but with limited iterations
        result = minimize(
            exact_objective,
            final_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        refined_points = result.x.reshape(-1, 2)
        refined_points = np.clip(refined_points, 0, 1)
        
        # Final calculation of true ratio using exact computation
        dist_matrix = squareform(pdist(refined_points))
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix)
        np.fill_diagonal(dist_matrix, 0)
        max_dist = np.max(dist_matrix)
        
        if max_dist > 0:
            ratio = min_dist / max_dist
            if ratio > best_ratio:
                best_result = refined_points
                
    except Exception:
        pass
    
    # Additional local search refinement using simulated annealing-inspired approach
    # This helps escape local minima and potentially find better solutions
    try:
        # Try a few additional local optimizations with different random seeds
        for seed in [123, 456, 789]:
            np.random.seed(seed)
            # Start with small random perturbations of the best solution
            perturbed = best_result + (np.random.rand(n, 2) - 0.5) * 0.01
            perturbed = np.clip(perturbed, 0, 1)
            
            # Optimize this perturbed version
            result = minimize(
                exact_objective,
                perturbed.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                refined_points = result.x.reshape(-1, 2)
                refined_points = np.clip(refined_points, 0, 1)
                
                # Final calculation of true ratio using exact computation
                dist_matrix = squareform(pdist(refined_points))
                np.fill_diagonal(dist_matrix, np.inf)
                min_dist = np.min(dist_matrix)
                np.fill_diagonal(dist_matrix, 0)
                max_dist = np.max(dist_matrix)
                
                if max_dist > 0:
                    ratio = min_dist / max_dist
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_result = refined_points
                        
    except Exception:
        pass
    
    return best_result


# EVOLVE-BLOCK-END
