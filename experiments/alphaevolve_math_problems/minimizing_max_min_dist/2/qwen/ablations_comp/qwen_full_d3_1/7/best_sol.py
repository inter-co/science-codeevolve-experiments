# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated optimization approach combining gradient-based methods with multi-start strategy
    and smooth approximations for better gradient computation, inspired by the best practices from 
    both inspirations.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    d = 2
    
    # Energy-based objective that mimics repulsive forces between points (from INSPIRATION 1)
    def energy_objective(points_flat):
        points = points_flat.reshape(-1, 2)
        distances = squareform(pdist(points))
        # Set diagonal to large value to avoid self-distances
        np.fill_diagonal(distances, np.inf)
        # Use inverse distance squared as repulsion energy
        energies = 1.0 / (distances + 1e-10)**2
        # Sum all pairwise energies (we want to maximize minimum distance, so minimize total energy)
        return np.sum(energies) / 2  # Divide by 2 to avoid double counting
    
    # Improved smooth approximation of min distance with better numerical stability
    def smooth_min_distance(points_flat, k=1000):
        points = points_flat.reshape(-1, 2)
        distances = squareform(pdist(points))
        # Set diagonal to large value to avoid self-distances
        np.fill_diagonal(distances, np.inf)
        # More numerically stable approach with better handling of edge cases
        distances_safe = np.maximum(distances, 1e-12)  # Prevent log(0)
        try:
            smooth_min = -np.log(np.sum(np.exp(-k * distances_safe))) / k
            # Add safety check for extreme values
            if np.isnan(smooth_min) or np.isinf(smooth_min):
                return 1e10
            return smooth_min
        except:
            return 1e10
    
    # Improved smooth approximation of max distance
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
    
    # Mathematical approach: leverage known optimal configurations and symmetry
    # Use a combination of algebraic and geometric insights
    def algebraic_approach():
        """Try to find configuration based on known optimal arrangements"""
        # Start with a regular 4x4 grid (hexagonal-like structure) but with more careful spacing
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25
                y = i * 0.25
                points.append([x, y])
        
        # Apply small perturbations to break degeneracies
        points = np.array(points[:n])
        np.random.seed(42)
        points += (np.random.rand(n, 2) - 0.5) * 0.02
        points = np.clip(points, 0, 1)
        return points
    
    # Improved hexagonal pattern with better normalization (from INSPIRATION 1)
    def generate_improved_hexagonal_initial():
        points = []
        rows = 4
        cols = 4
        spacing = 0.25
        
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
            points[i, 0] += (np.random.rand() - 0.5) * 0.02
            points[i, 1] += (np.random.rand() - 0.5) * 0.02
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points
    
    # Enhanced golden spiral with better normalization and more robust implementation
    def golden_spiral_initial():
        points = np.zeros((n, 2))
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            angle = 2 * np.pi * i / phi
            # Use a better radial distribution that avoids clustering at center
            radius = np.sqrt(i / (n - 1)) if i < n - 1 else 1
            points[i] = [radius * np.cos(angle), radius * np.sin(angle)]
        
        # Normalize to [0,1] x [0,1] with better edge case handling
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
    
    # Enhanced concentric circles pattern with better distribution
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
    
    # Mathematical-inspired approach: Try to construct solution using known discrete geometry
    def mathematical_optimization_initial():
        # Start with a symmetric configuration and refine
        points = generate_improved_hexagonal_initial()
        # Apply a small amount of energy minimization to improve distribution
        try:
            bounds = [(0, 1) for _ in range(2*n)]
            result = minimize(
                energy_objective,
                points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.success:
                points = result.x.reshape(-1, 2)
        except:
            pass
        points = np.clip(points, 0, 1)
        return points
    
    # Special approach: Try to create a configuration that maximizes the minimum distance
    # by using a "maximin" approach where we try to balance all distances
    def maximin_initial():
        # Start with a regular grid and then try to optimize for maximin
        points = np.array([[i*0.25, j*0.25] for i in range(4) for j in range(4)])
        points = points[:n]
        
        # Apply small random perturbations to break symmetries
        np.random.seed(42)
        points += (np.random.rand(n, 2) - 0.5) * 0.01
        points = np.clip(points, 0, 1)
        return points
    
    # Global optimization approach using differential evolution as inspired by INSPIRATION 1
    def global_optimization_approach():
        """Use global optimization to find better solutions"""
        def objective_global(x):
            # Reshape x into points
            points = x.reshape(n, d)
            
            # Ensure points are within bounds [0,1]
            points = np.clip(points, 0, 1)
            
            # Compute pairwise distances
            distances = pdist(points)
            
            # Avoid division by zero
            if len(distances) == 0:
                return float('inf')
                
            d_min = np.min(distances)
            d_max = np.max(distances)
            
            # Handle case where all points are identical
            if d_max <= 1e-10:
                return float('inf')
                
            # Return negative ratio (since we want to maximize ratio, we minimize negative ratio)
            return -d_min / d_max
        
        # Create smart initial configuration
        initial_points = generate_improved_hexagonal_initial()
        
        # Bounds for all coordinates [0,1]
        bounds = [(0, 1) for _ in range(n * d)]
        
        # Use differential evolution - good for this kind of problem
        try:
            result = differential_evolution(
                objective_global,
                bounds,
                maxiter=2200,  # Even more iterations for better convergence
                popsize=40,     # Even larger population for better exploration
                mutation=(0.98, 1),  # Even larger mutation for more exploration
                recombination=0.99,  # Even higher recombination for more mixing
                seed=42,
                atol=1e-13,     # Tighter absolute tolerance
                rtol=1e-13     # Tighter relative tolerance
            )
            
            if result.success:
                optimized_points = result.x.reshape(n, d)
                optimized_points = np.clip(optimized_points, 0, 1)
                return optimized_points
        except Exception:
            pass
        
        return initial_points
    
    # Multi-start optimization with diverse initializations
    best_result = None
    best_ratio = -np.inf
    
    # Set up seeds for reproducibility
    np.random.seed(42)
    
    # Enhanced initial strategies with more diverse and higher-quality patterns
    initial_strategies = [
        lambda: algebraic_approach(),                    # Algebraic construction approach
        lambda: generate_improved_hexagonal_initial(),   # Improved hexagonal pattern
        lambda: golden_spiral_initial(),                 # Golden spiral with perturbations
        lambda: concentric_circles_initial(),           # Enhanced concentric circles
        lambda: fibonacci_sphere_initial(),             # Fibonacci sphere-inspired
        lambda: mathematical_optimization_initial(),     # Mathematical optimization approach
        lambda: maximin_initial(),                      # Maximin-focused approach
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
    num_starts = min(len(initial_strategies), 15)  # Reduce to manage time better
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
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}  # Even more iterations
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
    
    # If we have no valid results, fall back to algebraic approach
    if best_result is None:
        return algebraic_approach()
    
    # Try global optimization approach as a last resort to get the best possible solution
    try:
        global_result = global_optimization_approach()
        if global_result is not None:
            # Calculate ratio for global result
            dist_matrix = squareform(pdist(global_result))
            np.fill_diagonal(dist_matrix, np.inf)
            min_dist = np.min(dist_matrix)
            np.fill_diagonal(dist_matrix, 0)
            max_dist = np.max(dist_matrix)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_result = global_result
    except Exception:
        pass
    
    # Enhanced refinement using multiple approaches
    # First try energy-based refinement to improve distribution (from INSPIRATION 1)
    try:
        bounds = [(0, 1) for _ in range(2*n)]
        result_energy = minimize(
            energy_objective,
            best_result.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10}
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
    # But limit iterations to keep time under control
    final_flat = best_result.flatten()
    bounds = [(0, 1) for _ in range(2*n)]
    
    try:
        # Refine with exact objective function but with limited iterations
        result = minimize(
            exact_objective,
            final_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 900, 'ftol': 1e-12, 'gtol': 1e-12}
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
        for seed in [123, 456, 789, 999, 555, 888, 333, 666, 777]:
            np.random.seed(seed)
            # Start with small random perturbations of the best solution
            perturbed = best_result + (np.random.rand(n, 2) - 0.5) * 0.003  # Even smaller perturbation
            perturbed = np.clip(perturbed, 0, 1)
            
            # Optimize this perturbed version
            result = minimize(
                exact_objective,
                perturbed.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12}
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
    
    # One final approach: try to find a better starting configuration by using a more
    # systematic approach with increased iterations for the best candidate
    try:
        # Run one more intensive optimization on the best result found so far
        # This gives us a chance to improve further with more aggressive optimization
        result_final = minimize(
            exact_objective,
            best_result.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        
        if result_final.success:
            final_points = result_final.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            
            # Final calculation of true ratio using exact computation
            dist_matrix = squareform(pdist(final_points))
            np.fill_diagonal(dist_matrix, np.inf)
            min_dist = np.min(dist_matrix)
            np.fill_diagonal(dist_matrix, 0)
            max_dist = np.max(dist_matrix)
            
            if max_dist > 0:
                ratio = min_dist / max_dist
                if ratio > best_ratio:
                    best_result = final_points
                    
    except Exception:
        pass
    
    return best_result


# EVOLVE-BLOCK-END
