# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric initialization and advanced optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    def objective(params):
        """Minimize negative of min/max distance ratio (i.e., maximize the ratio)"""
        # Reshape parameters back to points
        points = params.reshape((16, 2))
        
        # Compute pairwise distances
        distances = pdist(points)
        
        # Avoid division by zero
        if len(distances) == 0:
            return float('inf')
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist <= 0:
            return float('inf')
            
        # Return negative ratio to minimize (since we want to maximize)
        return -min_dist / max_dist
    
    # Strategy 1: Generate high-quality initial configuration using improved 3-ring hexagonal pattern
    # Based on analysis of best practices from inspirations, especially focusing on symmetry
    np.random.seed(42)
    
    # Create a more sophisticated 3-ring structure inspired by hexagonal close packing
    points = np.zeros((16, 2))
    
    # Center point
    points[0] = [0.5, 0.5]
    
    # First ring (6 points) - arranged in a hexagon around center
    ring_radius1 = 0.28
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points[i+1] = [
            0.5 + ring_radius1 * np.cos(angle),
            0.5 + ring_radius1 * np.sin(angle)
        ]
    
    # Second ring (9 points) - arranged in a larger hexagon
    ring_radius2 = 0.48
    for i in range(9):
        angle = 2 * np.pi * i / 9 + np.pi/9  # Offset to break symmetry
        points[i+7] = [
            0.5 + ring_radius2 * np.cos(angle),
            0.5 + ring_radius2 * np.sin(angle)
        ]
    
    # Add small random perturbations to escape local minima
    # Use carefully controlled perturbations to preserve good distribution properties
    for i in range(16):
        points[i, 0] += np.random.uniform(-0.015, 0.015)
        points[i, 1] += np.random.uniform(-0.015, 0.015)
    
    # Ensure all points are within bounds [0,1]
    points = np.clip(points, 0, 1)
    
    # Strategy 2: Multi-stage optimization approach with enhanced robustness
    # Stage 1: Global optimization with differential evolution for exploration
    try:
        # Define bounds for differential evolution (more robust than simple clipping)
        bounds_de = [(0, 1)] * 32  # 16 points * 2 coordinates each
        
        # Differential evolution with parameters tuned for better exploration
        result_de = differential_evolution(
            objective,
            bounds_de,
            maxiter=150,  # Increased iterations for better convergence
            popsize=20,   # Larger population size
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42,
            disp=False
        )
        
        if result_de.success:
            de_points = result_de.x.reshape(-1, 2)
            de_points = np.clip(de_points, 0, 1)
            
            # Compare with initial solution and keep better one
            initial_ratio = -objective(points.flatten())
            de_ratio = -objective(de_points.flatten())
            
            if de_ratio > initial_ratio:
                points = de_points
    except Exception:
        pass
    
    # Stage 2: Local refinement with L-BFGS-B with more aggressive settings
    try:
        result_lbfgs = minimize(
            objective,
            points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1)] * 32,
            options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14}  # More aggressive tolerances
        )
        
        if result_lbfgs.success:
            lbfgs_points = result_lbfgs.x.reshape(-1, 2)
            lbfgs_points = np.clip(lbfgs_points, 0, 1)
            
            # Keep the better solution
            initial_ratio = -objective(points.flatten())
            lbfgs_ratio = -objective(lbfgs_points.flatten())
            
            if lbfgs_ratio > initial_ratio:
                points = lbfgs_points
    except Exception:
        pass
    
    # Stage 3: Additional refinement with SLSQP if needed
    try:
        # Check if we already have a good solution before doing expensive SLSQP
        current_ratio = -objective(points.flatten())
        
        result_slsqp = minimize(
            objective,
            points.flatten(),
            method='SLSQP',
            bounds=[(0, 1)] * 32,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}  # More aggressive tolerances
        )
        
        if result_slsqp.success:
            slsqp_points = result_slsqp.x.reshape(-1, 2)
            slsqp_points = np.clip(slsqp_points, 0, 1)
            
            # Keep the better solution
            slsqp_ratio = -objective(slsqp_points.flatten())
            
            if slsqp_ratio > current_ratio:
                points = slsqp_points
    except Exception:
        pass
    
    # Stage 4: Final aggressive refinement with even more iterations and tighter tolerances
    # This final stage pushes the boundary for better performance
    try:
        # Run one last aggressive optimization with very high iteration count
        # and extremely tight tolerances to achieve maximum possible quality
        result_final = minimize(
            objective,
            points.flatten(),
            method='L-BFGS-B',
            bounds=[(0, 1)] * 32,
            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16}
        )
        
        if result_final.success:
            final_points = result_final.x.reshape(-1, 2)
            final_points = np.clip(final_points, 0, 1)
            
            # Keep the better solution
            current_ratio = -objective(points.flatten())
            final_ratio = -objective(final_points.flatten())
            
            if final_ratio > current_ratio:
                points = final_points
    except Exception:
        pass
    
    # Strategy 3: Enhanced multi-initialization approach inspired by INSPIRATION 1
    # Try several diverse initial configurations and optimize each one
    def create_diverse_initializations():
        """Create diverse initial configurations to explore different regions"""
        configs = []
        
        # Configuration 1: Hexagonal ring structure (our main approach)
        configs.append(points.copy())
        
        # Configuration 2: Random uniform distribution
        np.random.seed(123)
        configs.append(np.random.rand(16, 2))
        
        # Configuration 3: Spiral pattern (similar to INSPIRATION 2)
        angles = np.linspace(0, 4*np.pi, 16)
        radii = np.linspace(0.1, 0.4, 16)
        x = 0.5 + radii * np.cos(angles)
        y = 0.5 + radii * np.sin(angles)
        configs.append(np.column_stack([x, y]))
        
        # Configuration 4: Grid pattern with jitter (like INSPIRATION 3)
        grid_points = np.array([[i/3.0, j/3.0] for i in range(4) for j in range(4)][:16])
        jitter = np.random.normal(0, 0.03, (16, 2))  # Slightly larger jitter
        configs.append(np.clip(grid_points + jitter, 0, 1))
        
        # Configuration 5: Another variation of hexagonal structure with different parameters
        hex_points = np.zeros((16, 2))
        hex_points[0] = [0.5, 0.5]
        # Different ring radii
        for i in range(6):
            angle = 2 * np.pi * i / 6
            hex_points[i+1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]
        for i in range(9):
            angle = 2 * np.pi * i / 9 + np.pi/9
            hex_points[i+7] = [0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle)]
        configs.append(hex_points)
        
        return configs
    
    # Try multiple initial configurations with local optimization
    best_points = points.copy()
    best_ratio = -objective(points.flatten())
    
    # Try the diverse initial configurations
    alt_configs = create_diverse_initializations()
    
    for i, config in enumerate(alt_configs):
        # Apply small random perturbation to avoid getting stuck at same local minimum
        perturbed_config = config + np.random.normal(0, 0.01, (16, 2))
        perturbed_config = np.clip(perturbed_config, 0, 1)
        
        # Local optimization on this configuration
        try:
            # Use different optimization settings for different initializations
            if i < 2:  # For hexagonal and random initializations, use more aggressive settings
                result = minimize(
                    objective,
                    perturbed_config.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-10}
                )
            else:  # For other patterns, use standard settings
                result = minimize(
                    objective,
                    perturbed_config.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-8}
                )
            
            if result.success:
                optimized_points = result.x.reshape((16, 2))
                ratio = -objective(optimized_points.flatten())
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points.copy()
                    
        except Exception:
            continue
    
    # Final refinement with the best configuration found so far
    try:
        final_result = minimize(
            objective,
            best_points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if final_result.success:
            optimized_points = final_result.x.reshape((16, 2))
            ratio = -objective(optimized_points.flatten())
            
            if ratio > best_ratio:
                best_points = optimized_points.copy()
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
