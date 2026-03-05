# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing, basinhopping
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and aggressive optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points: np.ndarray) -> float:
        """Computes the min/max distance ratio for given points."""
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0.0
    
    def generate_hexagonal_initialization():
        """Generate initial points using hexagonal lattice pattern."""
        # Create points arranged in a hexagonal pattern
        points = []
        
        # Arrange in 4 rows with alternating positions
        for i in range(4):
            offset = 0.5 if i % 2 == 1 else 0.0
            for j in range(4):
                x = j + offset
                y = i * 0.866  # sqrt(3)/2 spacing
                points.append([x, y])
        
        points = np.array(points[:16])
        
        # Normalize to [0,1] range
        if points[:, 0].max() > points[:, 0].min():
            points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())
        if points[:, 1].max() > points[:, 1].min():
            points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())
        
        # Scale to fit nicely in [0.05, 0.95] square and add small random noise
        points[:, 0] *= 0.9
        points[:, 1] *= 0.9
        points[:, 0] += 0.05
        points[:, 1] += 0.05
        
        # Add small random noise to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.015, points.shape)
        points += noise
        
        # Ensure all points are within [0.05, 0.95] bounds
        points[:, 0] = np.clip(points[:, 0], 0.05, 0.95)
        points[:, 1] = np.clip(points[:, 1], 0.05, 0.95)
        
        return points
    
    def generate_golden_spiral_initialization():
        """Generate initial points using golden spiral pattern for better distribution."""
        # Create points in a golden spiral pattern
        n = 16
        points = []
        
        # Use golden spiral pattern for better distribution
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n):
            radius = 0.4 * np.sqrt(i / (n - 1))  # Radial distribution
            angle = i * golden_angle  # Angular distribution
            
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        
        # Add controlled noise to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.015, points.shape)
        points += noise
        
        # Ensure all points are within [0.05, 0.95] bounds
        points[:, 0] = np.clip(points[:, 0], 0.05, 0.95)
        points[:, 1] = np.clip(points[:, 1], 0.05, 0.95)
        
        return points
    
    def generate_regular_polygon_initialization():
        """Generate points in a regular polygon configuration."""
        points = []
        n = 16
        
        # Create points around a circle
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = 0.5 + 0.4 * np.cos(angle)
            y = 0.5 + 0.4 * np.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        
        # Add small random noise to break symmetry
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        
        # Ensure all points are within [0.05, 0.95] bounds
        points[:, 0] = np.clip(points[:, 0], 0.05, 0.95)
        points[:, 1] = np.clip(points[:, 1], 0.05, 0.95)
        
        return points
    
    def generate_fibonacci_sphere_initialization():
        """Generate points using Fibonacci sphere approach for better uniformity."""
        points = []
        n = 16
        
        for i in range(n):
            # Golden angle increment
            phi = np.arccos(-1 + (2 * i) / (n - 1))
            theta = np.sqrt(n * np.pi) * phi
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            
            # Map to [0.05, 0.95] range
            x = 0.05 + 0.9 * (x + 1) / 2
            y = 0.05 + 0.9 * (y + 1) / 2
            
            points.append([x, y])
        
        return np.array(points)
    
    def generate_random_initialization():
        """Generate random points within bounds."""
        np.random.seed(42)
        points = np.random.uniform(0.05, 0.95, (16, 2))
        return points
    
    def objective(x_flat):
        """Objective function to maximize min/max distance ratio"""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Get min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Return negative ratio (since we want to maximize)
        if d_max == 0:
            return -1.0
        return -d_min / d_max
    
    # Try multiple initialization strategies and select the best one
    initial_strategies = [
        generate_hexagonal_initialization,
        generate_golden_spiral_initialization,
        generate_regular_polygon_initialization,
        generate_fibonacci_sphere_initialization,
        generate_random_initialization
    ]
    
    best_initial_ratio = -np.inf
    best_initial_points = None
    
    for strategy in initial_strategies:
        try:
            points = strategy()
            ratio = compute_min_max_ratio(points)
            if ratio > best_initial_ratio:
                best_initial_ratio = ratio
                best_initial_points = points.copy()
        except Exception:
            continue
    
    # If no good initial points found, fall back to circle initialization
    if best_initial_points is None:
        n = 16
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        initial_points = np.zeros((n, 2))
        for i, angle in enumerate(angles):
            initial_points[i, 0] = 0.5 + 0.4 * np.cos(angle)
            initial_points[i, 1] = 0.5 + 0.4 * np.sin(angle)
        np.random.seed(42)
        initial_points += np.random.normal(0, 0.02, (n, 2))
        initial_points = np.clip(initial_points, 0, 1)
        best_initial_points = initial_points
    
    # Optimization with multiple restarts for better results
    best_ratio = -np.inf
    best_points = None
    
    # Use 50 restarts for better exploration (increased from 30)
    num_restarts = 50
    for restart in range(num_restarts):
        np.random.seed(42 + restart)
        
        # Perturb initial points slightly for this restart
        x_start = best_initial_points.flatten() + np.random.normal(0, 0.05, 32)
        x_start = np.clip(x_start, 0, 1)
        
        try:
            # Use SLSQP with tighter tolerances for better convergence
            result = minimize(
                objective,
                x_start,
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-9, 'eps': 1e-9, 'gtol': 1e-9}
            )
            
            if result.success:
                points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
                    
        except Exception:
            continue
    
    # If no good solution found, return the initial configuration
    if best_points is None:
        return best_initial_points
    
    # Apply enhanced final refinement to the best solution found
    try:
        if best_points is not None:
            refined_points = best_points.copy()
            best_final_ratio = best_ratio
            
            # First refine with very tight tolerances using SLSQP
            final_x = refined_points.flatten()
            result = minimize(
                objective,
                final_x,
                method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-12, 'eps': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success:
                final_points = result.x.reshape(-1, 2)
                ratio = compute_min_max_ratio(final_points)
                if ratio > best_final_ratio:
                    best_final_ratio = ratio
                    refined_points = final_points
            
            # Second pass: L-BFGS-B optimization with extremely tight tolerances
            try:
                result = minimize(
                    objective,
                    refined_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_final_ratio:
                        best_final_ratio = ratio
                        refined_points = final_points
            except:
                pass
            
            # Third pass: TNC optimization as additional backup
            try:
                result = minimize(
                    objective,
                    refined_points.flatten(),
                    method='TNC',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_final_ratio:
                        best_final_ratio = ratio
                        refined_points = final_points
            except:
                pass
            
            # Fourth pass: Dual Annealing for global exploration
            try:
                result = dual_annealing(
                    objective,
                    bounds=[(0, 1)] * 32,
                    maxiter=500,
                    initial_temp=1000,
                    seed=42,
                    no_local_search=True
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_final_ratio:
                        best_final_ratio = ratio
                        refined_points = final_points
            except:
                pass
            
            # Fifth pass: Basin hopping for better global search
            try:
                def minimizer(x0):
                    return minimize(objective, x0, method='L-BFGS-B', 
                                 options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15})
                
                bh_result = basinhopping(
                    objective,
                    refined_points.flatten(),
                    niter=50,
                    stepsize=0.05,
                    minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15}},
                    seed=42
                )
                
                if bh_result.success:
                    final_points = bh_result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_final_ratio:
                        best_final_ratio = ratio
                        refined_points = final_points
            except:
                pass
            
            # Sixth pass: Differential Evolution for global search
            try:
                result = differential_evolution(
                    objective,
                    bounds=[(0, 1)] * 32,
                    maxiter=500,
                    popsize=20,
                    seed=42,
                    tol=1e-12,
                    recombination=0.7
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_final_ratio:
                        refined_points = final_points
            except:
                pass
            
            # Seventh pass: Final SLSQP with ultra-tight tolerances
            try:
                np.random.seed(999)  # Different seed for diversity
                perturbed_start = refined_points.flatten() + np.random.normal(0, 0.01, 32)
                perturbed_start = np.clip(perturbed_start, 0, 1)
                
                result = minimize(
                    objective,
                    perturbed_start,
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-15, 'eps': 1e-15, 'gtol': 1e-15}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_final_ratio:
                        refined_points = final_points
            except:
                pass
            
            # Eighth pass: Final L-BFGS-B refinement with ultra-tight tolerances
            try:
                result = minimize(
                    objective,
                    refined_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-20, 'gtol': 1e-20}
                )
                
                if result.success:
                    final_points = result.x.reshape(-1, 2)
                    ratio = compute_min_max_ratio(final_points)
                    if ratio > best_final_ratio:
                        refined_points = final_points
            except:
                pass
            
            best_points = refined_points
                        
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
