# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import math
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a robust hybrid approach combining multiple optimization strategies and intelligent restarts.
    Inspired by mathematical insights and proven optimization techniques.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the min/max distance ratio for given point configuration"""
        if len(points) < 2:
            return 0
        
        distances = pdist(points)
        if len(distances) == 0:
            return 0
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        return min_dist / max_dist if max_dist > 0 else 0
    
    def objective_function(x):
        # Reshape flat array to 16x2 points
        points = x.reshape((16, 2))
        
        # Ensure points are within bounds [0,1] x [0,1]
        points = np.clip(points, 0, 1)
        
        # Calculate distances
        distances = pdist(points)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Return negative ratio since we want to maximize
        # Use a small epsilon to avoid division by zero
        epsilon = 1e-12
        if max_dist > epsilon:
            return -min_dist / max_dist
        else:
            return -1e10  # Very bad score if all points coincide
    
    # Generate multiple high-quality initial configurations
    def generate_initial_configurations():
        configs = []
        
        # Configuration 1: Hexagonal lattice with good spacing (inspired by mathematical packing)
        points = []
        for i in range(4):
            for j in range(4):
                # Hexagonal offset pattern
                x = j + (i % 2) * 0.5
                y = i * math.sqrt(3)/2
                points.append([x, y])
        
        points = np.array(points[:16])
        if np.max(points) > 0:
            points = points / np.max(points) * 0.8 + 0.1
        points += np.random.normal(0, 0.005, points.shape)
        points = np.clip(points, 0, 1)
        configs.append(points.copy())
        
        # Configuration 2: Golden ratio spiral (inspired by mathematical distribution)
        points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        for i in range(16):
            angle = 2 * math.pi * i * phi  # Golden angle
            radius = 0.4 * math.sqrt(i/15.0) if i > 0 else 0.05  # Spiral pattern
            x = 0.5 + radius * math.cos(angle)
            y = 0.5 + radius * math.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        points = np.clip(points, 0, 1)
        configs.append(points.copy())
        
        # Configuration 3: Grid with perturbations (inspired by structured optimization)
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                # Add small random perturbations
                x += np.random.normal(0, 0.015)
                y += np.random.normal(0, 0.015)
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                points.append([x, y])
        configs.append(np.array(points[:16]))
        
        # Configuration 4: Perturbed regular grid with better distribution (inspired by symmetry breaking)
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                # Add strategic perturbations to break symmetry
                if i % 2 == 0 and j % 2 == 0:
                    x += np.random.normal(0, 0.02)
                    y += np.random.normal(0, 0.02)
                elif i % 2 == 1 and j % 2 == 1:
                    x += np.random.normal(0, 0.01)
                    y += np.random.normal(0, 0.01)
                else:
                    x += np.random.normal(0, 0.015)
                    y += np.random.normal(0, 0.015)
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                points.append([x, y])
        configs.append(np.array(points[:16]))
        
        # Configuration 5: Circle arrangement with internal symmetry (inspired by spherical designs)
        points = []
        # Place 12 points on outer circle
        for i in range(12):
            angle = i * 2 * math.pi / 12
            points.append([0.5 + 0.4 * math.cos(angle), 0.5 + 0.4 * math.sin(angle)])
        
        # Add 4 inner points arranged symmetrically
        inner_angles = [0, math.pi/2, math.pi, 3*math.pi/2]
        for angle in inner_angles:
            points.append([0.5 + 0.15 * math.cos(angle), 0.5 + 0.15 * math.sin(angle)])
        
        points = np.array(points[:16])
        points = np.clip(points, 0, 1)
        configs.append(points.copy())
        
        # Configuration 6: Modified regular grid with adaptive perturbations (inspired by quasi-Monte Carlo)
        points = []
        for i in range(4):
            for j in range(4):
                x = (i + 0.5) / 4.0
                y = (j + 0.5) / 4.0
                # Apply adaptive perturbations based on position
                # Center points get less perturbation, corner points get more
                dist_from_center = math.sqrt((x - 0.5)**2 + (y - 0.5)**2)
                strength = 0.02 * (1 - dist_from_center)
                x += np.random.normal(0, strength)
                y += np.random.normal(0, strength)
                x = max(0.01, min(0.99, x))
                y = max(0.01, min(0.99, y))
                points.append([x, y])
        configs.append(np.array(points[:16]))
        
        # Configuration 7: Regular 4x4 grid (baseline)
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + 0.125
                y = i * 0.25 + 0.125
                points.append([x, y])
        configs.append(np.array(points))
        
        # Configuration 8: Random with fixed seed (robust restart strategy)
        np.random.seed(42)
        random_points = np.random.rand(16, 2)
        configs.append(random_points)
        
        return configs
    
    # Enhanced multi-stage optimization with aggressive global search
    def enhanced_optimization(initial_points, max_restarts=3):
        """Apply enhanced optimization with multiple restart strategies"""
        best_points = initial_points.copy()
        best_ratio = compute_min_max_ratio(best_points)
        
        # Strategy 1: Global optimization with differential evolution (aggressive parameters)
        try:
            bounds = [(0, 1) for _ in range(32)]
            # Use very aggressive parameters for better exploration
            result = differential_evolution(
                objective_function, 
                bounds, 
                maxiter=350,      # Even more iterations
                popsize=70,       # Even larger population
                mutation=(0.95, 1), # Even higher mutation rate
                recombination=0.98, # Even higher recombination rate
                seed=42,
                disp=False,
                atol=1e-13,
                rtol=1e-13
            )
            
            if result.success:
                de_points = result.x.reshape((16, 2))
                de_points = np.clip(de_points, 0, 1)
                ratio = compute_min_max_ratio(de_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = de_points.copy()
                    
        except Exception:
            pass
        
        # Strategy 2: Multiple local optimizations with different methods and seeds
        methods_and_seeds = [
            ('L-BFGS-B', 42),
            ('L-BFGS-B', 123),
            ('SLSQP', 42),
            ('SLSQP', 123),
            ('trust-constr', 42),
            ('trust-constr', 123)
        ]
        
        for method, seed in methods_and_seeds:
            try:
                # Add perturbation to initial points for diversity
                np.random.seed(seed)
                perturbed = best_points + np.random.normal(0, 0.01, best_points.shape)
                perturbed = np.clip(perturbed, 0, 1)
                
                bounds = [(0, 1) for _ in range(32)]
                result = minimize(
                    objective_function, 
                    perturbed.flatten(), 
                    method=method, 
                    bounds=bounds, 
                    options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-13}
                )
                
                if result.success:
                    refined_points = result.x.reshape((16, 2))
                    refined_points = np.clip(refined_points, 0, 1)
                    ratio = compute_min_max_ratio(refined_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = refined_points.copy()
                        
            except Exception:
                continue
        
        # Strategy 3: Additional refinement with multiple restarts
        for restart in range(max_restarts):
            try:
                np.random.seed(1000 + restart)
                random_start = np.random.rand(16, 2)
                bounds = [(0, 1) for _ in range(32)]
                result = minimize(
                    objective_function, 
                    random_start.flatten(), 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 600, 'ftol': 1e-13, 'gtol': 1e-13}
                )
                
                if result.success:
                    random_refined = result.x.reshape((16, 2))
                    random_refined = np.clip(random_refined, 0, 1)
                    ratio = compute_min_max_ratio(random_refined)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = random_refined.copy()
                        
            except Exception:
                continue
        
        return best_points
    
    # Main optimization process with systematic evaluation
    best_points = None
    best_ratio = 0
    
    # Generate initial configurations
    initial_configs = generate_initial_configurations()
    
    # Test each configuration with enhanced optimization - prioritize the most promising ones
    # Try first 6 configurations which tend to be better performers
    for i, config in enumerate(initial_configs[:6]):
        try:
            # Apply enhanced optimization
            optimized = enhanced_optimization(config, max_restarts=2)
            
            # Check quality
            ratio = compute_min_max_ratio(optimized)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized.copy()
                
        except Exception:
            continue
    
    # If no good solution was found, fallback to best initial configuration
    if best_points is None:
        # Evaluate all initial configurations and select the best
        best_initial_ratio = 0
        best_initial_points = None
        
        for config in initial_configs:
            try:
                ratio = compute_min_max_ratio(config)
                if ratio > best_initial_ratio:
                    best_initial_ratio = ratio
                    best_initial_points = config.copy()
            except Exception:
                continue
        
        # Apply optimization to the best initial configuration
        if best_initial_points is not None:
            try:
                best_points = enhanced_optimization(best_initial_points, max_restarts=3)
            except Exception:
                best_points = best_initial_points
    
    # Final fallback to a known good configuration
    if best_points is None:
        # Simple 4x4 grid with small perturbations
        points = []
        for i in range(4):
            for j in range(4):
                x = j * 0.25 + 0.125 + 0.005 * np.sin(i + j)
                y = i * 0.25 + 0.125 + 0.005 * np.cos(i + j)
                points.append([x, y])
        best_points = np.array(points)
        
        # Final optimization
        try:
            best_points = enhanced_optimization(best_points, max_restarts=1)
        except Exception:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
