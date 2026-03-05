# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import random
import warnings
warnings.filterwarnings('ignore')

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining multiple geometric initialization strategies and robust optimization.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    n = 16
    d = 2
    
    def calculate_ratio(points):
        """Calculate the min/max distance ratio"""
        if len(points) < 2:
            return 0.0
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
        d_min = np.min(distances)
        d_max = np.max(distances)
        return d_min / d_max if d_max > 0 else 0.0
    
    def objective(x):
        """Minimize negative of min/max distance ratio"""
        points = x.reshape(-1, 2)
        distances = pdist(points)
        
        # Handle edge cases properly
        if len(distances) == 0:
            return 1e10
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero with penalty
        if max_dist == 0:
            return 1e10
            
        # Use a penalty approach for very small distances
        if min_dist < 1e-12:
            return 1e10
            
        return -min_dist / max_dist
    
    def create_fibonacci_sphere_points():
        """Create points using Fibonacci spiral on sphere (projected to 2D)"""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(16):
            # Fibonacci-like distribution on sphere
            y = 1 - (i / 15.0) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y*y)  # radius at y
            
            # Golden angle increment
            phi_angle = i * 2.399963229728653  # Approximately 2π/φ²
            
            # Convert to Cartesian and project to 2D
            x = radius * np.cos(phi_angle)
            z = radius * np.sin(phi_angle)
            
            # Project to 2D (use x and y coordinates)
            points.append([0.5 + 0.4 * x, 0.5 + 0.4 * y])
        
        return np.array(points)
    
    def create_hexagonal_grid():
        """Create points arranged in a hexagonal grid pattern"""
        points = []
        # Create a hexagonal grid with 4 rows and 4 columns
        for i in range(4):
            for j in range(4):
                x = 0.1 + 0.8 * j / 3
                y = 0.1 + 0.8 * i / 3
                if i % 2 == 1:
                    x += 0.8 / 6
                points.append([x, y])
        return np.array(points)
    
    def create_perturbed_regular_polygon():
        """Create points in a regular polygon with significant perturbations"""
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4
        points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        # Add substantial perturbations to spread out points
        noise = np.random.normal(0, 0.05, points.shape)
        points += noise
        return points
    
    def create_symmetric_initial_config():
        """Create a highly symmetric initial configuration"""
        # Use a configuration inspired by known optimal arrangements
        # Place points in a pattern that tries to balance symmetry and uniformity
        
        # Create 4 concentric rings with different radii and angular spacing
        points = []
        
        # Ring 1: 4 points (corners of square)
        ring1_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        for angle in ring1_angles:
            points.append([0.5 + 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle)])
            
        # Ring 2: 4 points (midpoints of edges)
        ring2_angles = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 4, endpoint=False)
        for angle in ring2_angles:
            points.append([0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)])
            
        # Ring 3: 4 points (middle area)
        ring3_angles = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 4, endpoint=False)
        for angle in ring3_angles:
            points.append([0.5 + 0.6 * np.cos(angle), 0.5 + 0.6 * np.sin(angle)])
            
        # Ring 4: 4 points (outer ring)
        ring4_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        for angle in ring4_angles:
            points.append([0.5 + 0.8 * np.cos(angle), 0.5 + 0.8 * np.sin(angle)])
        
        points = np.array(points)
        # Add small random perturbations
        points += np.random.normal(0, 0.01, points.shape)
        points = np.clip(points, 0, 1)
        
        return points
    
    # Strategy 1: Global optimization with differential evolution for better exploration
    best_points = None
    best_ratio = 0.0
    
    # Try global optimization with more robust parameters
    try:
        bounds = [(0, 1) for _ in range(32)]
        # Use differential evolution with parameters that favor exploration
        de_result = differential_evolution(
            objective,
            bounds,
            maxiter=50,   # More iterations for better search
            popsize=25,   # Larger population for diversity
            mutation=(0.8, 1),  # Slightly higher mutation rate
            recombination=0.9,   # Higher recombination rate
            seed=42,
            disp=False,
            atol=1e-12,
            rtol=1e-12
        )
        
        if de_result.success:
            optimized_points = de_result.x.reshape((16, 2))
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = calculate_ratio(optimized_points)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points.copy()
    except Exception as e:
        pass
    
    # Strategy 2: Multiple initialization strategies with local optimization
    initial_strategies = [
        create_fibonacci_sphere_points,
        create_hexagonal_grid,
        create_perturbed_regular_polygon,
        create_symmetric_initial_config
    ]
    
    # Run optimization from multiple good starting points with multiple methods
    for strategy_idx, strategy in enumerate(initial_strategies):
        try:
            initial_points = strategy()
            initial_points = np.clip(initial_points, 0, 1)
            
            # Run multiple optimization methods to increase chance of finding better solution
            methods_and_options = [
                ('L-BFGS-B', {'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('SLSQP', {'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12}),
                ('TNC', {'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12})
            ]
            
            for method, options in methods_and_options:
                try:
                    x_flat = initial_points.flatten()
                    result = minimize(
                        objective,
                        x_flat,
                        method=method,
                        bounds=[(0, 1) for _ in range(32)],
                        options=options,
                        tol=1e-12
                    )
                    
                    if result.success:
                        optimized_points = result.x.reshape((16, 2))
                        optimized_points = np.clip(optimized_points, 0, 1)
                        ratio = calculate_ratio(optimized_points)
                        
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = optimized_points.copy()
                            
                except Exception:
                    continue
                    
        except Exception:
            continue
    
    # Strategy 3: Additional restarts with systematic perturbations
    if best_points is not None:
        # Try multiple restarts with different perturbation schemes
        for restart in range(8):
            try:
                # Create a perturbed version of the current best
                if restart == 0:
                    # First restart: moderate perturbation
                    perturbed = best_points + np.random.normal(0, 0.02, best_points.shape)
                elif restart < 4:
                    # Next few: larger perturbations
                    perturbed = best_points + np.random.normal(0, 0.05, best_points.shape)
                else:
                    # Later ones: smaller perturbations for fine-tuning
                    perturbed = best_points + np.random.normal(0, 0.01, best_points.shape)
                
                perturbed = np.clip(perturbed, 0, 1)
                
                # Optimize this version
                x_flat = perturbed.flatten()
                result = minimize(
                    objective,
                    x_flat,
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(32)],
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12},
                    tol=1e-12
                )
                
                if result.success:
                    optimized_points = result.x.reshape((16, 2))
                    optimized_points = np.clip(optimized_points, 0, 1)
                    ratio = calculate_ratio(optimized_points)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = optimized_points.copy()
                        
            except Exception:
                continue
    
    # Strategy 4: Final high precision optimization if we have a good candidate
    if best_points is not None:
        try:
            # Very tight optimization for final refinement
            result = minimize(
                objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1) for _ in range(32)],
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-15},
                tol=1e-15
            )
            
            if result.success:
                final_points = result.x.reshape((16, 2))
                final_points = np.clip(final_points, 0, 1)
                final_ratio = calculate_ratio(final_points)
                
                if final_ratio > best_ratio:
                    best_points = final_points
                    best_ratio = final_ratio
        except Exception:
            pass
    
    # Strategy 5: Fallback to a good known configuration if nothing worked well
    if best_points is None:
        # Use a good grid-based configuration with proper perturbations
        points = []
        for i in range(4):
            for j in range(4):
                # Use a more controlled approach to avoid degenerate cases
                x = 0.1 + 0.8 * j / 3 + (np.random.random()-0.5)*0.05
                y = 0.1 + 0.8 * i / 3 + (np.random.random()-0.5)*0.05
                points.append([x, y])
        
        best_points = np.array(points)
        best_points = np.clip(best_points, 0, 1)
    
    return best_points


# EVOLVE-BLOCK-END
