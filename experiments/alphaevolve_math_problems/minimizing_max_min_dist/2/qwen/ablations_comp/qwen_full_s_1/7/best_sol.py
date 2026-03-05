# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution, dual_annealing
import math
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining mathematical initialization with 
    multiple optimization strategies for robust convergence and superior results.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum pairwise distances."""
        if len(points) < 2:
            return 0.0
        
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        if max_dist <= 0:
            return 0.0
            
        return min_dist / max_dist
    
    def generate_hexagonal_grid():
        """Generate points in a hexagonal grid pattern for good distribution"""
        points = []
        # Create a 4x4 grid with offset rows for hexagonal packing
        for i in range(4):
            for j in range(4):
                offset = 0.5 if i % 2 == 1 else 0.0
                x = 0.1 + 0.8 * (j + offset) / 3.0
                y = 0.1 + 0.8 * i / 3.0
                points.append([x, y])
        return np.array(points[:16])
    
    def generate_concentric_rings():
        """Generate points in concentric rings for balanced distribution"""
        points = []
        
        # 8 points on outer circle with radius 0.4
        for i in range(8):
            angle = 2 * math.pi * i / 8
            x = 0.5 + 0.4 * math.cos(angle)
            y = 0.5 + 0.4 * math.sin(angle)
            points.append([x, y])
        
        # 8 more points on inner circle with radius 0.2 and phase shift
        for i in range(8):
            angle = 2 * math.pi * i / 8 + math.pi/8
            x = 0.5 + 0.2 * math.cos(angle)
            y = 0.5 + 0.2 * math.sin(angle)
            points.append([x, y])
            
        return np.array(points)
    
    def generate_fibonacci_spiral():
        """Generate points using Fibonacci spiral for even distribution"""
        points = []
        n = 16
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        
        for i in range(n):
            # Spiral distribution
            theta = i * 2 * math.pi / (phi * n)
            r = math.sqrt(i / (n - 1)) * 0.4 + 0.1  # Radial distribution
            
            x = 0.5 + r * math.cos(theta)
            y = 0.5 + r * math.sin(theta)
            points.append([x, y])
        
        return np.array(points)
    
    def generate_regular_polygon():
        """Generate points forming a regular polygon"""
        points = []
        for i in range(16):
            angle = 2 * math.pi * i / 16
            x = 0.5 + 0.4 * math.cos(angle)
            y = 0.5 + 0.4 * math.sin(angle)
            points.append([x, y])
        return np.array(points)
    
    def generate_varied_concentric():
        """Generate points with varied concentric rings"""
        points = np.zeros((16, 2))
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        # Use 2 rings with strategic spacing
        radii = np.concatenate([np.linspace(0.15, 0.4, 8), np.linspace(0.3, 0.6, 8)])
        for i in range(16):
            points[i, 0] = 0.5 + radii[i] * math.cos(angles[i]) * 0.4
            points[i, 1] = 0.5 + radii[i] * math.sin(angles[i]) * 0.4
                
        return points
    
    def objective(x):
        """Objective function to minimize (negative of min/max ratio)"""
        # Reshape points
        points = x.reshape(-1, 2)
        
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        dmin = np.min(distances)
        dmax = np.max(distances)
        
        # Return negative ratio to maximize (min/max ratio)
        if dmax <= 0:
            return 0.0
        return -dmin / dmax
    
    # Generate multiple initial configurations
    strategies = []
    
    # Strategy 1: Concentric rings (inspired by inspiration 2)
    points1 = generate_concentric_rings()
    # Add noise for symmetry breaking
    np.random.seed(42)
    noise = np.random.normal(0, 0.015, points1.shape)
    points1 += noise
    points1 = np.clip(points1, 0, 1)
    strategies.append(("concentric", points1))
    
    # Strategy 2: Regular 16-gon with perturbations (inspired by inspiration 2)
    points2 = generate_regular_polygon()
    # Add structured perturbations
    np.random.seed(42)
    points2 += np.random.normal(0, 0.01, (16, 2))
    points2[:, 0] = np.clip(points2[:, 0], 0, 1)
    points2[:, 1] = np.clip(points2[:, 1], 0, 1)
    strategies.append(("regular_hexagon", points2))
    
    # Strategy 3: Hexagonal grid with better distribution (inspired by inspiration 2)
    points3 = generate_hexagonal_grid()
    # Clip to [0,1] bounds and add noise
    points3[:, 0] = np.clip(points3[:, 0], 0, 1)
    points3[:, 1] = np.clip(points3[:, 1], 0, 1)
    np.random.seed(42)
    points3 += np.random.normal(0, 0.02, (16, 2))
    points3 = np.clip(points3, 0, 1)
    strategies.append(("hexagonal_grid", points3))
    
    # Strategy 4: Varied concentric rings (inspired by inspiration 2)
    points4 = generate_varied_concentric()
    # Add perturbations
    points4 += np.random.normal(0, 0.02, (16, 2))
    points4[:, 0] = np.clip(points4[:, 0], 0, 1)
    points4[:, 1] = np.clip(points4[:, 1], 0, 1)
    strategies.append(("varied_concentric", points4))
    
    # Strategy 5: Golden ratio spiral (new approach inspired by mathematical elegance)
    points5 = generate_fibonacci_spiral()
    # Add small noise for diversity
    np.random.seed(42)
    noise = np.random.normal(0, 0.01, points5.shape)
    points5 += noise
    points5 = np.clip(points5, 0, 1)
    strategies.append(("golden_spiral", points5))
    
    # Find best initial configuration
    best_initial = strategies[0][1]
    best_ratio = 0.0
    
    for name, points in strategies:
        ratio = compute_min_max_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_initial = points.copy()
    
    # Global optimization with multiple approaches
    best_points = best_initial.copy()
    best_ratio = compute_min_max_ratio(best_points)
    
    # Strategy 1: Differential Evolution with aggressive settings
    try:
        bounds = [(0, 1) for _ in range(32)]
        de_result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=2500,   # Increased iterations for better convergence
            popsize=30,     # Larger population for better exploration
            mutation=(0.8, 1),  # More aggressive mutation
            recombination=0.9,  # Higher recombination rate
            atol=1e-16,     # Tighter tolerances
            rtol=1e-16,
            strategy='best1bin'  # Better strategy selection
        )
        
        if de_result.success:
            de_points = de_result.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            de_ratio = compute_min_max_ratio(de_points)
            if de_ratio > best_ratio:
                best_ratio = de_ratio
                best_points = de_points.copy()
    except Exception as e:
        pass
    
    # Strategy 2: Dual Annealing with aggressive parameters
    try:
        bounds = [(0, 1) for _ in range(32)]
        da_result = dual_annealing(
            objective,
            bounds,
            maxiter=2500,   # Increased iterations
            initial_temp=3000,  # Higher initial temperature
            seed=42,
            no_local_search=True  # Skip local search to encourage global exploration
        )
        
        if da_result.success:
            da_points = da_result.x.reshape(-1, 2)
            da_points = np.clip(da_points, 0, 1)
            da_ratio = compute_min_max_ratio(da_points)
            if da_ratio > best_ratio:
                best_ratio = da_ratio
                best_points = da_points.copy()
    except Exception as e:
        pass
    
    # Strategy 3: Local refinement with multiple attempts (aggressive refinement)
    if best_ratio > 0.01:  # Only refine if we have a decent starting point
        # Try several local optimizations from different starting points
        for run in range(8):  # Increased number of runs for better chance of improvement
            np.random.seed(42 + run)
            # Start near the best solution with small perturbations
            x0 = best_points.flatten() + np.random.normal(0, 0.005, 32)
            x0 = np.clip(x0, 0, 1)
            
            try:
                bounds = [(0, 1) for _ in range(32)]
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 3500, 'ftol': 1e-16, 'gtol': 1e-16}  # Even tighter tolerances
                )
                
                if result.success:
                    points = result.x.reshape(-1, 2)
                    points = np.clip(points, 0, 1)
                    ratio = compute_min_max_ratio(points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points.copy()
            except Exception as e:
                continue
    
    # Final cleanup to avoid numerical issues
    final_points = best_points.copy()
    
    # Additional cleanup to prevent duplicate points
    tol = 1e-8
    for i in range(len(final_points)):
        for j in range(i+1, len(final_points)):
            dist = np.linalg.norm(final_points[i] - final_points[j])
            if dist < tol:
                # Perturb both points slightly
                perturbation = np.random.normal(0, tol/1000, 2)
                final_points[i] += perturbation
                final_points[j] -= perturbation
                final_points[i] = np.clip(final_points[i], 0, 1)
                final_points[j] = np.clip(final_points[j], 0, 1)
    
    return final_points


# EVOLVE-BLOCK-END
