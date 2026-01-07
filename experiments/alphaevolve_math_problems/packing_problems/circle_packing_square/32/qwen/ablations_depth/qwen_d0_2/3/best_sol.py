# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a lattice-based approach combined with targeted optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 32
    
    def create_lattice_initial_config():
        """Create initial configuration using hexagonal lattice pattern"""
        # Use a hexagonal lattice for good packing density
        # Arrange in a roughly 6x6 grid (36 positions) but only take 32
        rows = 6
        cols = 6
        
        # Hexagonal packing parameters
        spacing = 0.15  # Adjust based on expected radii
        radius_guess = 0.07  # Initial guess
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                # Hexagonal offset
                x = (j + 0.5 * (i % 2)) * spacing + 0.05
                y = i * spacing * np.sqrt(3)/2 + 0.05
                
                # Keep within bounds
                x = max(radius_guess, min(1-radius_guess, x))
                y = max(radius_guess, min(1-radius_guess, y))
                positions.append([x, y])
        
        # Trim to exactly n positions
        positions = positions[:n]
        
        # Initialize radii based on spacing and boundary constraints
        radii = []
        for i, (x, y) in enumerate(positions):
            # Maximum radius based on edge constraints
            max_radius_x = min(x, 1 - x)
            max_radius_y = min(y, 1 - y)
            max_radius = min(max_radius_x, max_radius_y)
            
            # Estimate neighbor distance
            min_dist = float('inf')
            for j, (ox, oy) in enumerate(positions):
                if i != j:
                    dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                    min_dist = min(min_dist, dist)
            
            # Set radius based on neighbor spacing and boundaries
            if min_dist < 1.0 and min_dist > 0:
                # Allow for spacing around neighbors
                initial_radius = min(max_radius, min_dist * 0.45)
            else:
                initial_radius = min(max_radius, 0.1)
                
            # Add small randomness to escape local minima
            initial_radius *= (0.9 + np.random.random() * 0.2)
            radii.append(max(0.01, initial_radius))
        
        return np.column_stack([positions, radii])
    
    def create_focused_initial_config():
        """Create initial configuration with strategic focus on high-density regions"""
        positions = []
        radii = []
        
        # Create a pattern with strategic concentration in center
        # Start with a dense cluster in center
        center_density = 16  # Number of circles in central cluster
        for _ in range(center_density):
            # Place in central area with some randomness
            x = 0.5 + np.random.normal(0, 0.1)
            y = 0.5 + np.random.normal(0, 0.1)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            positions.append([x, y])
        
        # Fill remaining positions with uniform distribution
        remaining = n - center_density
        for _ in range(remaining):
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            positions.append([x, y])
        
        # Set initial radii
        for i, (x, y) in enumerate(positions):
            # Maximum radius based on edge constraints
            max_radius_x = min(x, 1 - x)
            max_radius_y = min(y, 1 - y)
            max_radius = min(max_radius_x, max_radius_y)
            
            # Find minimum distance to other circles
            min_dist = float('inf')
            for j, (ox, oy) in enumerate(positions):
                if i != j:
                    dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                    min_dist = min(min_dist, dist)
            
            # Set initial radius with emphasis on avoiding overlaps
            if min_dist < 1.0 and min_dist > 0:
                initial_radius = min(max_radius, min_dist * 0.4)
            else:
                initial_radius = min(max_radius, 0.1)
                
            # Add randomness
            initial_radius *= (0.8 + np.random.random() * 0.4)
            radii.append(max(0.01, initial_radius))
        
        return np.column_stack([positions, radii])
    
    def create_optimized_initial_config():
        """Create a very refined initial configuration"""
        # Start with a good lattice pattern
        config = create_lattice_initial_config()
        
        # Refine by adjusting positions slightly to improve packing
        positions = config[:, :2]
        radii = config[:, 2]
        
        # Apply some simple geometric adjustments
        for i in range(n):
            # Move towards better packing if possible
            # This is a simplified version of local optimization
            x, y = positions[i, 0], positions[i, 1]
            
            # If near a boundary, adjust to allow larger radius
            if x < 0.1 or x > 0.9 or y < 0.1 or y > 0.9:
                # Slightly adjust position to improve radius
                x = max(0.05, min(0.95, x + np.random.normal(0, 0.01)))
                y = max(0.05, min(0.95, y + np.random.normal(0, 0.01)))
                positions[i, 0] = x
                positions[i, 1] = y
        
        return np.column_stack([positions, radii])
    
    def objective(x):
        """Maximize sum of radii (minimize negative sum)"""
        radii = x.reshape(-1, 3)[:, 2]
        return -np.sum(radii)
    
    def constraint_containment(x):
        """Ensure all circles fit within the unit square"""
        positions = x.reshape(-1, 3)[:, :2]
        radii = x.reshape(-1, 3)[:, 2]
        
        # Vectorized constraint evaluation
        con1 = positions[:, 0] - radii  # x - r >= 0
        con2 = positions[:, 1] - radii  # y - r >= 0
        con3 = 1 - positions[:, 0] - radii  # 1 - x - r >= 0
        con4 = 1 - positions[:, 1] - radii  # 1 - y - r >= 0
        
        return np.concatenate([con1, con2, con3, con4])
    
    def constraint_overlap(x):
        """Ensure no two circles overlap"""
        positions = x.reshape(-1, 3)[:, :2]
        radii = x.reshape(-1, 3)[:, 2]
        
        # Vectorized computation of overlap constraints
        distances = cdist(positions, positions)
        constraints = []
        
        # Only compute constraints for pairs where i < j to avoid duplication
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                constraint_value = dist - radii[i] - radii[j]
                constraints.append(constraint_value)
        
        return np.array(constraints)
    
    # Try multiple initialization strategies
    initial_configs = [
        create_lattice_initial_config(),
        create_focused_initial_config(),
        create_optimized_initial_config()
    ]
    
    # Add even more diverse configurations
    for _ in range(3):
        config = create_lattice_initial_config()
        # Add more noise for diversity
        noise_scale = 0.02
        for i in range(n):
            config[i, 0] += np.random.normal(0, noise_scale)
            config[i, 1] += np.random.normal(0, noise_scale)
            config[i, 0] = max(0.05, min(0.95, config[i, 0]))
            config[i, 1] = max(0.05, min(0.95, config[i, 1]))
        initial_configs.append(config)
    
    best_result = None
    best_sum = 0
    
    # Use a more targeted optimization approach
    for initial_config in initial_configs:
        try:
            x0 = initial_config.flatten()
            
            # Define bounds
            bounds = []
            for i in range(n):
                # x coordinate bounds
                bounds.append((0.001, 0.999))
                # y coordinate bounds  
                bounds.append((0.001, 0.999))
                # radius bounds
                bounds.append((0.001, 0.49))
            
            # Define constraints
            cons = [
                {'type': 'ineq', 'fun': constraint_containment},
                {'type': 'ineq', 'fun': constraint_overlap}
            ]
            
            # Use a simpler but more reliable optimization method
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                final_radii = result.x.reshape(-1, 3)[:, 2]
                sum_radii = np.sum(final_radii)
                if sum_radii > best_sum:
                    best_sum = sum_radii
                    best_result = result
                    
        except Exception:
            continue
    
    # If optimization failed, use the best initial configuration
    if best_result is None:
        # Use the best initial configuration directly
        best_initial_config = max(initial_configs, key=lambda config: np.sum(config[:, 2]))
        final_config = best_initial_config
    else:
        final_config = best_result.x.reshape(-1, 3)
    
    # Final validation and refinement
    positions = final_config[:, :2]
    radii = final_config[:, 2]
    
    # Ensure all constraints are satisfied
    for i in range(n):
        # Boundary constraints
        max_radius_x = min(positions[i, 0], 1 - positions[i, 0])
        max_radius_y = min(positions[i, 1], 1 - positions[i, 1])
        max_radius = min(max_radius_x, max_radius_y)
        radii[i] = min(radii[i], max_radius * 0.99)
    
    # Final overlap resolution with more aggressive approach
    max_iter = 50
    for _ in range(max_iter):
        changed = False
        distances = cdist(positions, positions)
        
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < radii[i] + radii[j]:
                    overlap = (radii[i] + radii[j]) - distances[i, j]
                    # Aggressive reduction to fix overlaps
                    reduction = overlap * 0.5
                    if radii[i] > reduction and radii[j] > reduction:
                        radii[i] -= reduction
                        radii[j] -= reduction
                        changed = True
        
        if not changed:
            break
    
    # Final cleanup to ensure everything fits
    for i in range(n):
        max_radius_x = min(positions[i, 0], 1 - positions[i, 0])
        max_radius_y = min(positions[i, 1], 1 - positions[i, 1])
        max_radius = min(max_radius_x, max_radius_y)
        radii[i] = min(radii[i], max_radius * 0.99)
    
    return np.column_stack([positions, radii])


# EVOLVE-BLOCK-END
