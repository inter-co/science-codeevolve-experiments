# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, dual_annealing, minimize
import warnings
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a sophisticated hybrid approach combining mathematical initialization with global and 
    local optimization for robust convergence.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    n = 16
    d = 2
    
    # Enhanced mathematical construction inspired by optimal point distributions
    def construct_mathematical_initial():
        """Construct a highly optimized initial configuration using mathematical principles"""
        points = []
        
        # Strategy 1: 8 points on outer circle (regular octagon) + 8 points on inner circle
        # This creates a balanced distribution that's mathematically principled
        
        # 1. Outer ring: 8 points forming a regular octagon
        for i in range(8):
            angle = 2 * math.pi * i / 8
            x = 0.5 + 0.4 * math.cos(angle)
            y = 0.5 + 0.4 * math.sin(angle)
            points.append([x, y])
        
        # 2. Inner ring: 8 points forming a rotated octagon
        for i in range(8):
            angle = 2 * math.pi * i / 8 + math.pi/8  # Phase shift to break symmetry
            x = 0.5 + 0.25 * math.cos(angle)
            y = 0.5 + 0.25 * math.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        
        # Add carefully controlled noise to break symmetry while maintaining good distribution
        np.random.seed(42)
        noise = np.random.normal(0, 0.012, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        return points
    
    # Comprehensive initialization strategies
    def initialize_strategies():
        strategies = []
        
        # Strategy 1: Mathematical construction with good geometric properties
        math_points = construct_mathematical_initial()
        strategies.append(("math", math_points))
        
        # Strategy 2: Regular 4x4 grid with structured perturbations
        grid_points = np.zeros((n, d))
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx < n:
                    grid_points[idx] = [i / 3.0, j / 3.0]
                    idx += 1
        
        # Add structured perturbations to break symmetry
        np.random.seed(42)
        grid_points += np.random.normal(0, 0.01, (n, d))
        grid_points[:, 0] = np.clip(grid_points[:, 0], 0, 1)
        grid_points[:, 1] = np.clip(grid_points[:, 1], 0, 1)
        strategies.append(("grid", grid_points))
        
        # Strategy 3: Concentric rings approach
        circle_points = np.zeros((n, d))
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        # Use 2 rings with strategic spacing
        radii = np.concatenate([np.linspace(0.15, 0.4, n//2), np.linspace(0.3, 0.6, n - n//2)])
        for i in range(n):
            circle_points[i, 0] = 0.5 + radii[i] * np.cos(angles[i]) * 0.4
            circle_points[i, 1] = 0.5 + radii[i] * np.sin(angles[i]) * 0.4
            
        # Add perturbations
        circle_points += np.random.normal(0, 0.025, (n, d))
        circle_points[:, 0] = np.clip(circle_points[:, 0], 0, 1)
        circle_points[:, 1] = np.clip(circle_points[:, 1], 0, 1)
        strategies.append(("circle", circle_points))
        
        # Strategy 4: Golden spiral approach
        golden_points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            theta = 2 * math.pi * i / phi
            r = math.sqrt(i / (n - 1)) if n > 1 else 0
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            golden_points.append([x, y])
        golden_points = np.array(golden_points)
        
        # Normalize properly
        min_coords = np.min(golden_points, axis=0)
        max_coords = np.max(golden_points, axis=0)
        range_coords = max_coords - min_coords
        if np.any(range_coords == 0):
            range_coords[range_coords == 0] = 1
        golden_points = (golden_points - min_coords) / range_coords * 0.8 + 0.1
        
        strategies.append(("golden", golden_points))
        
        # Strategy 5: Hexagonal lattice approach
        hex_points = np.zeros((n, d))
        # Create hexagonal-like arrangement
        row_count = 4
        col_count = 4
        idx = 0
        for row in range(row_count):
            for col in range(col_count):
                if idx >= n:
                    break
                # Hexagonal offset
                x = col / (col_count - 1) if col_count > 1 else 0.5
                y = row / (row_count - 1) if row_count > 1 else 0.5
                if row % 2 == 1:  # Offset every other row
                    x += 0.5 / col_count
                hex_points[idx] = [x, y]
                idx += 1
                if idx >= n:
                    break
        
        # Clip to [0,1] bounds and add noise
        hex_points[:, 0] = np.clip(hex_points[:, 0], 0, 1)
        hex_points[:, 1] = np.clip(hex_points[:, 1], 0, 1)
        np.random.seed(42)
        hex_points += np.random.normal(0, 0.02, (n, d))
        hex_points = np.clip(hex_points, 0, 1)
        strategies.append(("hex", hex_points))
        
        # Strategy 6: Spiral pattern with refinement
        spiral_points = np.zeros((n, d))
        for i in range(n):
            angle = 2 * np.pi * i / n
            radius = 0.4 * (i / (n - 1)) if n > 1 else 0.4
            spiral_points[i] = [0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)]
            
        # Add small perturbations
        spiral_points += np.random.normal(0, 0.02, (n, d))
        spiral_points[:, 0] = np.clip(spiral_points[:, 0], 0, 1)
        spiral_points[:, 1] = np.clip(spiral_points[:, 1], 0, 1)
        strategies.append(("spiral", spiral_points))
        
        return strategies
    
    # Find best initial configuration
    strategies = initialize_strategies()
    best_initial = strategies[0][1]  # Default to first strategy
    best_ratio = -float('inf')
    
    for name, points in strategies:
        if len(points) >= 2:
            distances = pdist(points)
            if len(distances) > 0:
                d_min = np.min(distances)
                d_max = np.max(distances)
                if d_max > 0:
                    ratio = d_min / d_max
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_initial = points.copy()
    
    # Define objective function: minimize negative of min/max ratio
    def objective(params):
        # Reshape parameters back to points
        points = params.reshape(-1, 2)
        
        # Compute distance matrix
        distances = pdist(points)
        if len(distances) == 0:
            return float('inf')
            
        # Calculate min and max distances
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Avoid division by zero
        if d_max <= 0:
            return float('inf')
            
        # Return negative ratio to maximize (min/max ratio)
        return -d_min / d_max
    
    # Define bounds for optimization (points in [0,1] x [0,1])
    bounds = [(0, 1) for _ in range(n * d)]
    
    # Try multiple global optimization approaches for robustness
    try:
        best_points = best_initial.copy()
        best_ratio = -float('inf')
        
        # Strategy 1: Differential Evolution with aggressive settings
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=2500,  # More iterations for better convergence
                popsize=40,    # Larger population size
                mutation=(0.9, 1),  # Higher mutation rate for better exploration
                recombination=0.95,   # Higher recombination for better mixing
                atol=1e-17,
                rtol=1e-17,
                strategy='best1bin'
            )
            
            if de_result.success:
                de_points = de_result.x.reshape(-1, 2)
                de_points = np.clip(de_points, 0, 1)
                distances = pdist(de_points)
                if len(distances) > 0:
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    if d_max > 0:
                        de_ratio = d_min / d_max
                        if de_ratio > best_ratio:
                            best_ratio = de_ratio
                            best_points = de_points.copy()
        except Exception as e:
            pass
        
        # Strategy 2: Dual Annealing with aggressive parameters
        try:
            da_result = dual_annealing(
                objective,
                bounds,
                maxiter=2500,  # More iterations
                initial_temp=3500,  # Higher initial temperature
                seed=42,
                no_local_search=True  # Enable local search for better refinement
            )
            
            if da_result.success:
                da_points = da_result.x.reshape(-1, 2)
                da_points = np.clip(da_points, 0, 1)
                distances = pdist(da_points)
                if len(distances) > 0:
                    d_min = np.min(distances)
                    d_max = np.max(distances)
                    if d_max > 0:
                        da_ratio = d_min / d_max
                        if da_ratio > best_ratio:
                            best_ratio = da_ratio
                            best_points = da_points.copy()
        except Exception as e:
            pass
        
        # Strategy 3: Local optimization refinement with better settings
        # If we have a good global solution, refine it locally with high precision
        if best_ratio > -float('inf') and best_ratio > 0.01:
            # Try multiple local optimizations with different starting points
            for run in range(15):  # More runs for better chance of improvement
                np.random.seed(42 + run)
                # Start near the best solution with small perturbations
                x0 = best_points.flatten() + np.random.normal(0, 0.003, n * d)
                x0 = np.clip(x0, 0, 1)
                
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 4000, 'ftol': 1e-17, 'gtol': 1e-17}
                    )
                    
                    if result.success:
                        points = result.x.reshape(-1, 2)
                        points = np.clip(points, 0, 1)
                        distances = pdist(points)
                        if len(distances) > 0:
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            if d_max > 0:
                                ratio = d_min / d_max
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_points = points.copy()
                except Exception as e:
                    continue
        
        # Strategy 4: Additional refinement with a completely fresh approach
        # If we're still not satisfied, try one more comprehensive approach
        if best_ratio < 0.07:  # If we're not doing well yet, try extra effort
            try:
                # Try one final global optimization with even more aggressive parameters
                de_result = differential_evolution(
                    objective,
                    bounds,
                    seed=999,
                    maxiter=3500,
                    popsize=60,
                    mutation=(0.95, 1),
                    recombination=0.98,
                    atol=1e-18,
                    rtol=1e-18,
                    strategy='best1bin'
                )
                
                if de_result.success:
                    de_points = de_result.x.reshape(-1, 2)
                    de_points = np.clip(de_points, 0, 1)
                    distances = pdist(de_points)
                    if len(distances) > 0:
                        d_min = np.min(distances)
                        d_max = np.max(distances)
                        if d_max > 0:
                            de_ratio = d_min / d_max
                            if de_ratio > best_ratio:
                                return de_points
            except Exception as e:
                pass
        
        return best_points
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}. Using best initial configuration.")
        # Last resort: use the best initial configuration
        return best_initial.copy()


# EVOLVE-BLOCK-END
