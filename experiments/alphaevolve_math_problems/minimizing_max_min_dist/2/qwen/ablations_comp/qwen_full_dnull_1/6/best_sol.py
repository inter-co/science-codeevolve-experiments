# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, dual_annealing, minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a robust multi-start optimization approach with geometric initialization and 
    carefully tuned optimization strategies.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    def compute_min_max_ratio(points):
        """Compute the ratio of minimum to maximum distance between all point pairs."""
        # Normalize points to [0,1] x [0,1] to satisfy constraints
        points = np.clip(points, 0, 1)
        
        # Compute pairwise distances
        distances = pdist(points)
        if len(distances) == 0:
            return 0.0
            
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist == 0:
            return 0.0
            
        return min_dist / max_dist
    
    def objective_function(x_flat):
        """Objective function to maximize (negative because we minimize)."""
        # Reshape flat array back to points
        points = x_flat.reshape(-1, 2)
        
        # Compute ratio (we want to maximize this)
        ratio = compute_min_max_ratio(points)
        
        # Return negative because we're minimizing in scipy.optimize
        return -ratio
    
    def initialize_points_hexagonal():
        """
        Initialize points using a hexagonal lattice pattern which often provides 
        good separation properties for point distributions.
        """
        # Create a hexagonal pattern that fits well in [0,1] x [0,1]
        points = []
        rows, cols = 4, 4
        
        # Generate points in a hexagonal pattern with offset rows
        for i in range(rows):
            for j in range(cols):
                if len(points) >= 16:
                    break
                # Hexagonal offset for alternate rows
                offset = 0.5 if i % 2 == 1 else 0.0
                x = (j + offset) / (cols - 1) if cols > 1 else 0.5
                y = i / (rows - 1) if rows > 1 else 0.5
                
                # Scale down to avoid boundary issues and add some randomness
                x = 0.1 + x * 0.8  # Scale to [0.1, 0.9]
                y = 0.1 + y * 0.8
                
                # Add controlled noise to break symmetry
                noise_scale = 0.02
                x += np.random.normal(0, noise_scale * 0.5)
                y += np.random.normal(0, noise_scale * 0.5)
                
                points.append([x, y])
        
        points_array = np.array(points[:16])  # Ensure exactly 16 points
        
        # Ensure all points are within bounds
        points_array[:, 0] = np.clip(points_array[:, 0], 0, 1)
        points_array[:, 1] = np.clip(points_array[:, 1], 0, 1)
        
        return points_array
    
    def initialize_points_grid():
        """
        Initialize points using a 4x4 grid pattern with strategic perturbations.
        """
        # Create a 4x4 grid pattern
        points = []
        spacing_x = 1.0 / 3.0
        spacing_y = 1.0 / 3.0
        
        for i in range(4):
            for j in range(4):
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                
                # Add controlled perturbation to break symmetry
                noise_x = np.random.normal(0, 0.015)
                noise_y = np.random.normal(0, 0.015)
                x += noise_x
                y += noise_y
                
                points.append([x, y])
        
        points_array = np.array(points)
        
        # Ensure all points are within bounds
        points_array[:, 0] = np.clip(points_array[:, 0], 0, 1)
        points_array[:, 1] = np.clip(points_array[:, 1], 0, 1)
        
        return points_array
    
    def initialize_points_random():
        """
        Simple random initialization.
        """
        return np.random.rand(16, 2)
    
    def initialize_points_circle():
        """
        Initialize points on a circle with random perturbations for better distribution.
        """
        points = []
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        radius = 0.4  # Adjust to fit within [0.1, 0.9] range
        
        # Distribute points around a circle, then add small random perturbations
        for i, angle in enumerate(angles):
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            
            # Add small random perturbation to break symmetry
            x += (np.random.random() - 0.5) * 0.05
            y += (np.random.random() - 0.5) * 0.05
            
            points.append([x, y])
        
        points_array = np.array(points)
        
        # Ensure all points are within bounds
        points_array[:, 0] = np.clip(points_array[:, 0], 0, 1)
        points_array[:, 1] = np.clip(points_array[:, 1], 0, 1)
        
        return points_array
    
    # Try multiple initialization strategies and select the best starting point
    initializations = [
        initialize_points_hexagonal(),
        initialize_points_grid(), 
        initialize_points_random(),
        initialize_points_circle()
    ]
    
    best_initial = initializations[0]
    best_initial_ratio = 0
    
    for init in initializations:
        ratio = compute_min_max_ratio(init)
        if ratio > best_initial_ratio:
            best_initial_ratio = ratio
            best_initial = init
    
    # Apply optimization with carefully chosen methods and parameters
    best_points = best_initial.copy()
    best_ratio = compute_min_max_ratio(best_points)
    
    # Strategy 1: Differential Evolution (global optimization) - very effective for this problem
    bounds = [(0, 1) for _ in range(32)]
    try:
        result = differential_evolution(
            objective_function,
            bounds,
            seed=42,
            maxiter=150,  # Increased for better convergence
            popsize=25,   # Larger population for better exploration
            mutation=(0.5, 1),
            recombination=0.7,
            atol=1e-8,    # Tighter tolerance
            rtol=1e-8
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = compute_min_max_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points
    except Exception:
        pass
    
    # Strategy 2: Dual Annealing (global optimization) - good backup
    try:
        result = dual_annealing(
            objective_function,
            bounds,
            maxiter=200,  # Increased iterations
            initial_temp=1000,
            seed=42
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = compute_min_max_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points
    except Exception:
        pass
    
    # Strategy 3: Local optimization from best result with SLSQP for fine-tuning
    try:
        x0 = best_points.flatten()
        # Try SLSQP first (often works well for this type of problem)
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}  # Tighter tolerances
        )
        
        if result.success:
            optimized_points = result.x.reshape(-1, 2)
            optimized_points = np.clip(optimized_points, 0, 1)
            ratio = compute_min_max_ratio(optimized_points)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = optimized_points
        else:
            # Fallback to L-BFGS-B if SLSQP fails
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if result.success:
                optimized_points = result.x.reshape(-1, 2)
                optimized_points = np.clip(optimized_points, 0, 1)
                ratio = compute_min_max_ratio(optimized_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = optimized_points
    except Exception:
        pass
    
    return best_points


# EVOLVE-BLOCK-END
