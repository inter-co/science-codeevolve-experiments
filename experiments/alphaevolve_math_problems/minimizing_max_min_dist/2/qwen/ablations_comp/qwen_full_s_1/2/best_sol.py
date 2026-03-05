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
        
        # Strategy 1: Concentric rings approach with better geometric properties
        # 8 points on outer circle + 8 points on inner circle
        # Use golden ratio for better distribution
        outer_radius = 0.4
        inner_radius = 0.25
        
        # Outer ring: 8 points forming a regular octagon
        for i in range(8):
            angle = 2 * math.pi * i / 8
            x = 0.5 + outer_radius * math.cos(angle)
            y = 0.5 + outer_radius * math.sin(angle)
            points.append([x, y])
        
        # Inner ring: 8 points forming a rotated octagon (phase shifted by π/8)
        for i in range(8):
            angle = 2 * math.pi * i / 8 + math.pi/8
            x = 0.5 + inner_radius * math.cos(angle)
            y = 0.5 + inner_radius * math.sin(angle)
            points.append([x, y])
        
        points = np.array(points)
        
        # Add carefully controlled noise to break symmetry while maintaining good distribution
        np.random.seed(42)
        noise = np.random.normal(0, 0.01, points.shape)
        points += noise
        points = np.clip(points, 0, 1)
        return points
    
    # Comprehensive initialization strategies - incorporating best from all inspirations
    def initialize_strategies():
        strategies = []
        
        # Strategy 1: Mathematical construction with good geometric properties (from INSPIRATION 1 & 2)
        math_points = construct_mathematical_initial()
        strategies.append(("math", math_points))
        
        # Strategy 2: Improved concentric rings approach (from INSPIRATION 2)
        circle_points = np.zeros((n, d))
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        # Use 2 rings with strategic spacing - optimized radii
        radii = np.concatenate([np.linspace(0.15, 0.4, n//2), np.linspace(0.25, 0.55, n - n//2)])
        for i in range(n):
            circle_points[i, 0] = 0.5 + radii[i] * np.cos(angles[i]) * 0.4
            circle_points[i, 1] = 0.5 + radii[i] * np.sin(angles[i]) * 0.4
            
        # Add perturbations
        circle_points += np.random.normal(0, 0.02, (n, d))
        circle_points[:, 0] = np.clip(circle_points[:, 0], 0, 1)
        circle_points[:, 1] = np.clip(circle_points[:, 1], 0, 1)
        strategies.append(("circle", circle_points))
        
        # Strategy 3: Vortex pattern with better control (from INSPIRATION 2)
        vortex_points = []
        num_rings = 4
        points_per_ring = n // num_rings
        
        for ring_idx in range(num_rings):
            n_points = points_per_ring if ring_idx < num_rings - 1 else n - (num_rings - 1) * points_per_ring
            
            # Use logarithmic spacing for rings
            radius = 0.1 + 0.4 * (ring_idx / (num_rings - 1)) if num_rings > 1 else 0.25
            
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            
            for i, angle in enumerate(angles):
                # Add a vortex effect with sinusoidal modulation
                vortex_factor = 0.05 * math.sin(3 * angle + ring_idx * math.pi/2)
                r = radius * (1 + vortex_factor)
                
                x = 0.5 + r * math.cos(angle)
                y = 0.5 + r * math.sin(angle)
                
                x = max(0, min(1, x))
                y = max(0, min(1, y))
                
                vortex_points.append([x, y])
            
            if len(vortex_points) >= n:
                break
        
        vortex_points = np.array(vortex_points[:n])
        np.random.seed(42)
        jitter_magnitude = 0.01
        vortex_points += np.random.uniform(-jitter_magnitude, jitter_magnitude, vortex_points.shape)
        vortex_points = np.clip(vortex_points, 0, 1)
        strategies.append(("vortex", vortex_points))
        
        # Strategy 4: Golden spiral approach (from INSPIRATION 2) - improved version
        golden_points = []
        phi = (1 + math.sqrt(5)) / 2  # golden ratio
        for i in range(n):
            theta = 2 * math.pi * i / phi
            r = math.sqrt(i / (n - 1)) if n > 1 else 0
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            golden_points.append([x, y])
        golden_points = np.array(golden_points)
        
        # Normalize properly with better bounds
        min_coords = np.min(golden_points, axis=0)
        max_coords = np.max(golden_points, axis=0)
        range_coords = max_coords - min_coords
        if np.any(range_coords == 0):
            range_coords[range_coords == 0] = 1
        golden_points = (golden_points - min_coords) / range_coords * 0.8 + 0.1
        
        strategies.append(("golden", golden_points))
        
        # Strategy 5: Hexagonal lattice approach (from INSPIRATION 2) - improved version
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
        hex_points += np.random.normal(0, 0.015, (n, d))
        hex_points = np.clip(hex_points, 0, 1)
        strategies.append(("hex", hex_points))
        
        # Strategy 6: Grid with better spacing (from INSPIRATION 2)
        grid_points = np.zeros((n, d))
        idx = 0
        for i in range(4):
            for j in range(4):
                if idx < n:
                    grid_points[idx] = [i / 3.0, j / 3.0]
                    idx += 1
        
        # Add structured perturbations to break symmetry
        np.random.seed(42)
        grid_points += np.random.normal(0, 0.008, (n, d))
        grid_points[:, 0] = np.clip(grid_points[:, 0], 0, 1)
        grid_points[:, 1] = np.clip(grid_points[:, 1], 0, 1)
        strategies.append(("grid", grid_points))
        
        # Strategy 7: Optimized regular polygon approach (from INSPIRATION 1)
        poly_points = []
        # Arrange points in a 4x4 grid pattern but with optimized spacing
        for i in range(4):
            for j in range(4):
                # Add some randomness to break perfect symmetry
                x = 0.1 + 0.8 * i / 3.0 + np.random.normal(0, 0.01)
                y = 0.1 + 0.8 * j / 3.0 + np.random.normal(0, 0.01)
                poly_points.append([x, y])
        poly_points = np.array(poly_points[:n])
        strategies.append(("poly", poly_points))
        
        # Strategy 8: Spiral with better radial distribution (from INSPIRATION 2)
        spiral_points = np.zeros((n, d))
        for i in range(n):
            angle = 2 * np.pi * i / n
            radius = 0.4 * (i / (n - 1)) if n > 1 else 0.4
            spiral_points[i] = [0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)]
            
        # Add small perturbations and normalize
        spiral_points += np.random.normal(0, 0.02, (n, d))
        spiral_points[:, 0] = np.clip(spiral_points[:, 0], 0, 1)
        spiral_points[:, 1] = np.clip(spiral_points[:, 1], 0, 1)
        strategies.append(("spiral", spiral_points))
        
        # Strategy 9: Algebraic number field approach (from INSPIRATION 1) - enhanced
        # Generate 16th roots of unity with specific transformations
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        complex_roots = np.exp(1j * angles)
        points = np.column_stack([complex_roots.real, complex_roots.imag])
        
        # Apply specific transformation based on algebraic number field properties
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        transformed_points = []
        for i, (x, y) in enumerate(points):
            # Apply transformation based on index position in relation to phi
            index_mod = i % 5  # Use modulo 5 for golden ratio patterns
            if index_mod == 0:
                # Scale and rotate
                r = np.sqrt(x*x + y*y)
                theta = np.arctan2(y, x) + 0.1 * i
                new_x = r * np.cos(theta)
                new_y = r * np.sin(theta)
            elif index_mod == 1:
                # Apply conjugate transformation
                new_x = x * 0.9
                new_y = y * 1.1
            elif index_mod == 2:
                # Apply reflection
                new_x = -x * 0.8
                new_y = y * 1.2
            elif index_mod == 3:
                # Apply rotation and scaling
                angle = 0.2 * i
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                new_x = x * cos_a - y * sin_a
                new_y = x * sin_a + y * cos_a
            else:
                # Identity transformation
                new_x, new_y = x, y
            
            transformed_points.append([new_x, new_y])
        
        points = np.array(transformed_points)
        
        # Normalize and scale to unit square
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if max(x_range, y_range) > 0:
            scale = 0.9 / max(x_range, y_range)
            points = points * scale
        
        # Center in unit square
        center_offset = 0.5 - np.mean(points, axis=0)
        points = points + center_offset
        
        strategies.append(("algebraic", points))
        
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
    
    # Improved optimization approach - more aggressive and systematic (from INSPIRATION 2)
    try:
        best_points = best_initial.copy()
        best_ratio = -float('inf')
        
        # Strategy 1: Differential Evolution with aggressive settings (from INSPIRATION 2)
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=2000,  # Increased iterations to improve convergence
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
        
        # Strategy 2: Dual Annealing with aggressive parameters (from INSPIRATION 2)
        try:
            da_result = dual_annealing(
                objective,
                bounds,
                maxiter=2000,  # Increased iterations to improve convergence
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
        
        # Strategy 3: Multiple local optimizations with better restarts (from INSPIRATION 2)
        # Run local optimizations from multiple good starting points
        local_starts = []
        
        # Start with the best global solution
        if best_points is not None:
            local_starts.append(best_points.copy())
        
        # Add additional starting points from different strategies
        for i in range(5):  # More runs for better chance of improvement
            np.random.seed(100 + i)
            strategy_name, initial_points = strategies[np.random.randint(len(strategies))]
            # Add small random perturbation
            perturbed_points = initial_points.copy()
            perturbed_points += np.random.normal(0, 0.005, (16, 2))
            perturbed_points = np.clip(perturbed_points, 0, 1)
            local_starts.append(perturbed_points)
        
        # Run local optimizations with higher precision
        for i, start_points in enumerate(local_starts):
            try:
                result = minimize(
                    objective,
                    start_points.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-17}  # Higher precision
                )
                
                if result.success:
                    optimized_points = result.x.reshape(16, 2)
                    optimized_points = np.clip(optimized_points, 0, 1)
                    distances = pdist(optimized_points)
                    if len(distances) > 0:
                        d_min = np.min(distances)
                        d_max = np.max(distances)
                        if d_max > 0:
                            ratio = d_min / d_max
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_points = optimized_points.copy()
            except Exception as e:
                continue
        
        # Strategy 4: Final comprehensive refinement (from INSPIRATION 1)
        # Apply a more thorough local search approach
        if best_points is not None and best_ratio > 0.05:  # Only if we have a decent solution
            # Multi-stage refinement with decreasing step sizes
            current_points = best_points.copy()
            current_ratio = best_ratio
            
            # Stage 1: Coarse refinement
            for stage in range(3):
                for i in range(len(current_points)):
                    # Perturb each point with increasing precision
                    perturbation_scale = 0.01 / (stage + 1)
                    for _ in range(5):  # Multiple attempts per point
                        delta = np.random.uniform(-perturbation_scale, perturbation_scale, 2)
                        new_point = current_points[i] + delta
                        new_point = np.clip(new_point, 0, 1)
                        
                        # Test this move
                        test_points = current_points.copy()
                        test_points[i] = new_point
                        
                        distances = pdist(test_points)
                        if len(distances) > 0:
                            d_min = np.min(distances)
                            d_max = np.max(distances)
                            if d_max > 0:
                                ratio = d_min / d_max
                                if ratio > current_ratio:
                                    current_points[i] = new_point
                                    current_ratio = ratio
            
            if current_ratio > best_ratio:
                best_points = current_points
                best_ratio = current_ratio
        
        return best_points
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}. Using best initial configuration.")
        # Last resort: use the best initial configuration
        return best_initial.copy()


# EVOLVE-BLOCK-END
