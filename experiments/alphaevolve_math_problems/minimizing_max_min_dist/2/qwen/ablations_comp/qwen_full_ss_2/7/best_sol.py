# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import dual_annealing, differential_evolution, minimize
import time
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining global and local optimization strategies 
    with multiple geometric initialization methods and robust restart strategies.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    d = 2
    
    # Strategy 1: Mathematical construction based on known optimal configurations
    # Using a systematic approach with better spacing and symmetry breaking
    def generate_mathematical_grid():
        points = []
        # Create a 4x4 grid with systematic perturbations
        for i in range(4):
            for j in range(4):
                x = i / 3.0 if i < 3 else 1.0
                y = j / 3.0 if j < 3 else 1.0
                points.append([x, y])
        
        points = np.array(points)
        
        # Apply systematic perturbations to break symmetry and improve distribution
        # Use trigonometric functions to create non-uniform but well-distributed perturbations
        for i in range(len(points)):
            row = i // 4
            col = i % 4
            
            # Apply different perturbation patterns based on position
            # Use a combination of sine/cosine functions to create good spacing
            pert_x = 0.005 * np.sin(row * 0.5) * np.cos(col * 0.3)
            pert_y = 0.005 * np.cos(row * 0.3) * np.sin(col * 0.5)
            
            points[i][0] += pert_x
            points[i][1] += pert_y
        
        # Ensure all points are within bounds
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 2: Golden ratio-based spiral with improved radial distribution
    def generate_golden_spiral():
        points = []
        # Use the golden angle but with better radial spacing
        golden_angle = 2.399963229728653  # 2π(1 - 1/φ)
        
        for i in range(n):
            # Better radial distribution - use square root for more uniform spacing
            r = 0.4 * np.sqrt(i / (n - 1)) if i < n - 1 else 0.4
            theta = i * golden_angle
            
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        return points
    
    # Strategy 3: Octahedral-inspired configuration with better symmetry
    def generate_octahedral():
        points = []
        
        # Create points on two circles at different heights (octahedral-like)
        # First circle (outer)
        angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        for angle in angles:
            points.append([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
        
        # Second circle (inner)
        for angle in angles:
            points.append([0.5 + 0.2 * np.cos(angle + np.pi/4), 0.5 + 0.2 * np.sin(angle + np.pi/4)])
        
        # Axis points
        axis_points = [[0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]]
        points.extend(axis_points)
        
        # Central cluster points
        center_points = [[0.4, 0.4], [0.6, 0.6], [0.4, 0.6], [0.6, 0.4]]
        points.extend(center_points)
        
        points = np.array(points[:n])
        
        # Normalize to [0,1] x [0,1] 
        if points.shape[0] > 0:
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
            
            if x_max > x_min and y_max > y_min:
                scale_x = 1.0 / (x_max - x_min)
                scale_y = 1.0 / (y_max - y_min)
                
                points[:, 0] = (points[:, 0] - x_min) * scale_x * 0.8 + 0.1
                points[:, 1] = (points[:, 1] - y_min) * scale_y * 0.8 + 0.1
        
        return points
    
    # Strategy 4: Concentric circles with optimized point distribution
    def generate_concentric_circles():
        points = []
        # Four concentric rings with optimized point counts
        radii = [0.15, 0.25, 0.35, 0.4]
        points_per_ring = [4, 4, 4, 4]  # Total 16 points
        
        for ring_idx, (radius, num_points) in enumerate(zip(radii, points_per_ring)):
            for i in range(num_points):
                # Distribute points evenly around the circle
                angle = 2 * np.pi * i / num_points + ring_idx * 0.2  # Small offset for variation
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                points.append([x, y])
        return np.array(points[:n])
    
    # Strategy 5: Regular polygon with radial variations
    def generate_radial_polygon():
        points = []
        # Create points on a circle with radial variations
        for i in range(n):
            angle = 2 * np.pi * i / n
            # Add radial variation to break uniformity and improve minimum distances
            radius_variation = 0.02 * np.sin(i * 0.5)
            radius = 0.4 + radius_variation
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    # Strategy 6: Hexagonal tiling with boundary adjustments
    def generate_hexagonal_tiling():
        points = []
        # Generate hexagonal pattern
        for i in range(4):
            for j in range(4):
                if len(points) >= n:
                    break
                # Hexagonal coordinates
                x = j + (i % 2) * 0.5
                y = i * np.sqrt(3)/2
                points.append([x, y])
        
        points = np.array(points[:n])
        
        # Normalize to [0,1] x [0,1] with better scaling
        if points.shape[0] > 0:
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
            
            if x_max > x_min and y_max > y_min:
                scale_x = 1.0 / (x_max - x_min)
                scale_y = 1.0 / (y_max - y_min)
                
                points[:, 0] = (points[:, 0] - x_min) * scale_x * 0.8 + 0.1
                points[:, 1] = (points[:, 1] - y_min) * scale_y * 0.8 + 0.1
        
        return points
    
    # Strategy 7: Fibonacci spiral with enhanced radial distribution
    def generate_enhanced_fibonacci():
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        
        # Use a more sophisticated approach with better radial distribution
        for i in range(n):
            # Better radial distribution using a modified Fibonacci sequence
            theta = i * 2 * np.pi / golden_ratio
            # Use a power law for radial distribution to avoid clustering
            r = 0.4 * np.power(i / (n - 1), 0.7) if i < n - 1 else 0.4
            x = 0.5 + r * np.cos(theta)
            y = 0.5 + r * np.sin(theta)
            points.append([x, y])
        return np.array(points)
    
    # Strategy 8: Strategic placement based on mathematical constants
    def generate_constant_based():
        points = []
        # Use mathematical constants for positioning
        # Place points using combinations of π and e
        for i in range(n):
            # Position using combinations of mathematical constants
            angle = 2 * np.pi * i / n
            # Use a more complex radial pattern
            r = 0.4 * (0.5 + 0.5 * np.sin(i * 0.3 + np.pi/4))
            x = 0.5 + r * np.cos(angle)
            y = 0.5 + r * np.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    # Strategy 9: K-means based clustering approach for better distribution
    def generate_kmeans_distribution():
        # Start with a uniform grid and apply k-means clustering to distribute points
        points = []
        for i in range(4):
            for j in range(4):
                points.append([i/3.0 if i < 3 else 1.0, j/3.0 if j < 3 else 1.0])
        
        points = np.array(points)
        
        # Add some randomness to break symmetry
        points += np.random.normal(0, 0.02, points.shape)
        points = np.clip(points, 0, 1)
        
        # Use k-means to refine distribution (but only for the first 16 points)
        kmeans = KMeans(n_clusters=n, init='random', n_init=10, random_state=42)
        labels = kmeans.fit_predict(points)
        refined_points = kmeans.cluster_centers_
        
        # Make sure we have exactly n points
        if refined_points.shape[0] > n:
            refined_points = refined_points[:n]
        elif refined_points.shape[0] < n:
            # Fill with random points if needed
            extra_points = np.random.rand(n - refined_points.shape[0], 2)
            refined_points = np.vstack([refined_points, extra_points])
        
        refined_points = np.clip(refined_points, 0, 1)
        return refined_points
    
    # Strategy 10: Optimized grid with strategic outliers (inspired by inspiration 1)
    def generate_optimized_grid():
        # Start with a basic 4x4 grid
        points = []
        for i in range(4):
            for j in range(4):
                points.append([(i + 0.5) / 4.0, (j + 0.5) / 4.0])
        
        points = np.array(points)
        
        # Add some carefully positioned outliers to improve minimum distance
        # Place some points near corners and edges
        outliers = [
            [0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
            [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]
        ]
        
        # Replace some grid points with outliers
        for i, outlier in enumerate(outliers[:8]):
            if i < len(points):
                points[i] = outlier
        
        # Add small random noise for optimization robustness
        points += np.random.normal(0, 0.005, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    # Generate multiple initial configurations
    configs = [
        generate_mathematical_grid(),      # Systematic grid approach
        generate_golden_spiral(),          # Golden spiral
        generate_octahedral(),             # Octahedral structure
        generate_concentric_circles(),     # Concentric rings
        generate_radial_polygon(),         # Radial polygon
        generate_hexagonal_tiling(),       # Hexagonal pattern
        generate_enhanced_fibonacci(),     # Enhanced fibonacci
        generate_constant_based(),         # Constant-based positioning
        generate_kmeans_distribution(),    # K-means clustering
        generate_optimized_grid()          # Optimized grid with outliers
    ]
    
    # Add some random perturbations to break symmetry and ensure diversity
    for config in configs:
        # Add noise to break symmetry
        noise = np.random.normal(0, 0.008, config.shape)
        config += noise
        # Ensure within bounds
        config[:, 0] = np.clip(config[:, 0], 0.05, 0.95)
        config[:, 1] = np.clip(config[:, 1], 0.05, 0.95)
    
    # Objective function to maximize ratio (minimize negative ratio)
    def objective(params):
        points = params.reshape(n, d)
        distances = pdist(points)
        
        if len(distances) == 0:
            return float('inf')
            
        # Avoid division by zero
        distances = np.maximum(distances, 1e-12)
        
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return float('inf')
            
        # Minimize negative of ratio (since we want to maximize ratio)
        ratio = d_min / d_max
        return -ratio  # Negative because we're minimizing
    
    # Compute ratio function for evaluation
    def compute_ratio(points_flat):
        points = points_flat.reshape(-1, 2)
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        d_min = np.min(distances)
        d_max = np.max(distances)
        if d_max == 0:
            return 0
        return d_min / d_max
    
    # Optimization bounds
    bounds = [(0, 1) for _ in range(2*n)]
    
    best_ratio = -1
    best_points = None
    
    # Time management - limit total optimization time
    start_time = time.time()
    max_time = 55  # seconds
    
    # Try multiple initialization strategies with multiple optimization approaches
    # Use a more focused approach with better optimization parameters
    for i, initial_config in enumerate(configs):
        if time.time() - start_time > max_time:
            break
            
        current_points = initial_config.copy()
        
        try:
            # Use a more balanced optimization strategy with better diversity
            # 1. Global optimization with dual annealing (high iteration count)
            if time.time() - start_time < max_time * 0.4:
                da_result = dual_annealing(
                    objective,
                    bounds,
                    maxiter=2000,  # More iterations for better convergence
                    initial_temp=2000,  # Higher temperature for better exploration
                    seed=42 + i,
                    no_local_search=False  # Allow local search for better results
                )
                
                if da_result.success:
                    optimized_points = da_result.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            
            # 2. Differential evolution for robust global search
            if time.time() - start_time < max_time * 0.7 and best_points is None:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=800,  # More iterations
                    popsize=30,   # Larger population for better exploration
                    seed=42 + i,
                    strategy='best1bin',
                    atol=1e-15,   # Tighter tolerance
                    rtol=1e-15
                )
                
                if de_result.success:
                    optimized_points = de_result.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
            
            # 3. Local refinement with multiple methods
            if time.time() - start_time < max_time * 0.9 and best_points is not None:
                # Try L-BFGS-B with very tight tolerances
                result1 = minimize(
                    objective,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18},
                    tol=1e-18
                )
                
                if result1.success:
                    optimized_points = result1.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
                # Also try SLSQP as backup
                result2 = minimize(
                    objective,
                    best_points.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
                )
                
                if result2.success:
                    optimized_points = result2.x.reshape(n, d)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = compute_ratio(optimized_points.flatten())
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
        except Exception:
            continue
    
    # If still no good solution, return the best configuration found so far
    if best_points is None:
        # Fallback to mathematical grid as it has strong mathematical foundations
        fallback = generate_mathematical_grid()
        # Add small random perturbations to break any remaining symmetries
        noise = np.random.normal(0, 0.01, fallback.shape)
        fallback += noise
        fallback[:, 0] = np.clip(fallback[:, 0], 0.05, 0.95)
        fallback[:, 1] = np.clip(fallback[:, 1], 0.05, 0.95)
        return fallback
    
    return best_points


# EVOLVE-BLOCK-END
