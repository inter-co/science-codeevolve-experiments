# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses advanced optimization with improved initialization and constraint handling.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 32
    
    def create_hexagonal_initial_config():
        """Create initial configuration using hexagonal packing pattern"""
        # Create a hexagonal lattice pattern
        positions = []
        radii = []
        
        # Hexagonal packing parameters
        sqrt3 = np.sqrt(3)
        hex_radius = 0.15  # Approximate hexagon radius
        
        # Grid dimensions
        rows = int(np.ceil(np.sqrt(n)) * 1.2)
        cols = int(np.ceil(n / rows))
        
        # Create hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                    
                # Offset every other row
                x_offset = 0.0 if i % 2 == 0 else 0.5
                x = (j + x_offset) * hex_radius * 2 * 0.9
                y = i * hex_radius * sqrt3 * 0.9
                
                # Adjust to fit within unit square
                x = max(hex_radius, min(1 - hex_radius, x))
                y = max(hex_radius, min(1 - hex_radius, y))
                
                if len(positions) < n:
                    positions.append([x, y])
        
        # Fill remaining positions randomly but with good distribution
        while len(positions) < n:
            x = np.random.uniform(hex_radius, 1 - hex_radius)
            y = np.random.uniform(hex_radius, 1 - hex_radius)
            positions.append([x, y])
        
        positions = positions[:n]
        
        # Set initial radii based on local density and edge constraints
        for i, (x, y) in enumerate(positions):
            # Maximum radius based on edge constraints
            max_radius_x = min(x, 1 - x)
            max_radius_y = min(y, 1 - y)
            max_radius = min(max_radius_x, max_radius_y)
            
            # Find nearest neighbors to estimate local density
            distances = [np.sqrt((x - px)**2 + (y - py)**2) for px, py in positions]
            distances.sort()
            
            # Use first few nearest neighbors to estimate minimum spacing
            min_neighbor_dist = float('inf')
            for d in distances[1:min(6, len(distances))]:  # Check up to 5 nearest neighbors
                if d > 0:
                    min_neighbor_dist = min(min_neighbor_dist, d)
            
            # Set initial radius considering both edge and neighbor constraints
            if min_neighbor_dist < 1.0 and min_neighbor_dist > 0:
                # Allow for some spacing around neighbors
                initial_radius = min(max_radius, min_neighbor_dist * 0.3)
            else:
                initial_radius = min(max_radius, 0.1)
                
            # Add randomness to avoid getting stuck in poor local optima
            initial_radius *= (0.8 + np.random.random() * 0.4)
            radii.append(max(0.01, initial_radius))
        
        return np.column_stack([positions, radii])
    
    def create_adaptive_initial_config():
        """Create adaptive initial configuration with better distribution"""
        positions = []
        radii = []
        
        # Try to place circles in a way that maximizes space usage
        # Start with a coarse grid then refine
        
        # Coarse grid placement
        grid_size = 6
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        count = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if count >= n:
                    break
                # Position with some jitter
                x = (j + 1) * spacing_x + np.random.normal(0, 0.01)
                y = (i + 1) * spacing_y + np.random.normal(0, 0.01)
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                positions.append([x, y])
                count += 1
            if count >= n:
                break
        
        # Fill remaining positions with more strategic placement
        while len(positions) < n:
            # Prefer placing near edges if possible to maximize space
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Bias towards corners and edges for better space utilization
            if np.random.random() < 0.3:
                # Place near edge
                if np.random.random() < 0.5:
                    x = np.random.choice([0.05, 0.95])
                else:
                    y = np.random.choice([0.05, 0.95])
            positions.append([x, y])
        
        positions = positions[:n]
        
        # Set initial radii with better estimation
        for i, (x, y) in enumerate(positions):
            # Maximum radius based on edge constraints
            max_radius_x = min(x, 1 - x)
            max_radius_y = min(y, 1 - y)
            max_radius = min(max_radius_x, max_radius_y)
            
            # Find nearest neighbors
            distances = [np.sqrt((x - px)**2 + (y - py)**2) for px, py in positions]
            distances.sort()
            
            # Use first few nearest neighbors to estimate minimum spacing
            min_neighbor_dist = float('inf')
            for d in distances[1:min(6, len(distances))]:  # Check up to 5 nearest neighbors
                if d > 0:
                    min_neighbor_dist = min(min_neighbor_dist, d)
            
            # Set initial radius with more sophisticated logic
            if min_neighbor_dist < 1.0 and min_neighbor_dist > 0:
                # Allow for some spacing around neighbors
                initial_radius = min(max_radius, min_neighbor_dist * 0.35)
            else:
                initial_radius = min(max_radius, 0.12)
                
            # Add randomness but be more conservative
            initial_radius *= (0.9 + np.random.random() * 0.2)
            radii.append(max(0.01, initial_radius))
        
        return np.column_stack([positions, radii])
    
    def create_voronoi_initial_config():
        """Create Voronoi-based initial configuration"""
        # Generate points and create Voronoi diagram
        points = np.random.rand(n, 2)
        points = points * 0.8 + 0.1  # Scale to [0.1, 0.9] range
        
        # Create Voronoi cells and use centroids as circle centers
        vor = Voronoi(points)
        positions = []
        
        # Get the centroids of the finite Voronoi regions
        for region in vor.regions:
            if len(region) > 0 and -1 not in region:
                # Calculate centroid
                coords = np.array([vor.vertices[i] for i in region if i >= 0])
                if len(coords) > 0:
                    centroid = np.mean(coords, axis=0)
                    # Keep within bounds
                    centroid[0] = max(0.05, min(0.95, centroid[0]))
                    centroid[1] = max(0.05, min(0.95, centroid[1]))
                    positions.append(centroid)
        
        # Fill up if needed
        while len(positions) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            positions.append([x, y])
        
        positions = positions[:n]
        
        # Set radii based on Voronoi properties
        initial_radii = []
        for i, (x, y) in enumerate(positions):
            # Radius based on distance to nearest edge
            max_radius_x = min(x, 1 - x)
            max_radius_y = min(y, 1 - y)
            max_radius = min(max_radius_x, max_radius_y)
            
            # Find minimum distance to other circles
            min_dist_to_others = float('inf')
            for j, (ox, oy) in enumerate(positions):
                if i != j:
                    dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                    min_dist_to_others = min(min_dist_to_others, dist)
            
            # Initial radius should be constrained by both edges and neighbors
            if min_dist_to_others < 1.0 and min_dist_to_others > 0:
                initial_radius = min(max_radius, min_dist_to_others * 0.35)
            else:
                initial_radius = min(max_radius, 0.1)
                
            # Add some randomness
            initial_radius *= (0.8 + np.random.random() * 0.4)
            initial_radii.append(max(0.01, initial_radius))
        
        return np.column_stack([positions, initial_radii])
    
    # Constraint functions (optimized for performance)
    def containment_constraints(x):
        """Ensure all circles fit within the unit square"""
        positions = x.reshape(-1, 3)[:, :2]
        radii = x.reshape(-1, 3)[:, 2]
        
        # Vectorized constraint evaluation
        con1 = positions[:, 0] - radii  # x - r >= 0
        con2 = positions[:, 1] - radii  # y - r >= 0
        con3 = 1 - positions[:, 0] - radii  # 1 - x - r >= 0
        con4 = 1 - positions[:, 1] - radii  # 1 - y - r >= 0
        
        return np.concatenate([con1, con2, con3, con4])
    
    def overlap_constraints(x):
        """Ensure no two circles overlap"""
        positions = x.reshape(-1, 3)[:, :2]
        radii = x.reshape(-1, 3)[:, 2]
        
        # Vectorized computation of overlap constraints
        # Create upper triangular part of distance matrix to avoid redundancy
        distances = cdist(positions, positions)
        constraints = []
        
        # Only compute constraints for pairs where i < j to avoid duplication
        for i in range(n):
            for j in range(i+1, n):
                dist = distances[i, j]
                constraint_value = dist - radii[i] - radii[j]
                constraints.append(constraint_value)
        
        return np.array(constraints)
    
    def objective(x):
        """Maximize sum of radii (minimize negative sum)"""
        radii = x.reshape(-1, 3)[:, 2]
        return -np.sum(radii)
    
    def callback(xk):
        """Callback function for optimization progress tracking"""
        pass
    
    # Try multiple initial configurations and keep the best
    best_result = None
    best_sum = 0
    
    # Try different initialization strategies
    initial_configs = [
        create_hexagonal_initial_config(),
        create_adaptive_initial_config(), 
        create_voronoi_initial_config()
    ]
    
    # Add more diverse configurations with noise
    for i in range(3):
        config = create_adaptive_initial_config()
        # Add noise to create more diversity
        noise_scale = 0.03
        for j in range(n):
            config[j, 0] += np.random.normal(0, noise_scale)
            config[j, 1] += np.random.normal(0, noise_scale)
            config[j, 0] = max(0.05, min(0.95, config[j, 0]))
            config[j, 1] = max(0.05, min(0.95, config[j, 1]))
        initial_configs.append(config)
    
    # Run optimization with different initial configs using multiple methods
    for i, initial_config in enumerate(initial_configs):
        try:
            # Flatten for optimization
            x0 = initial_config.flatten()
            
            # Define bounds: [x, y, r] for each circle
            bounds = []
            for j in range(n):
                # x coordinate bounds
                bounds.append((0.001, 0.999))  # x must be within [0.001, 0.999] to allow for radius
                # y coordinate bounds  
                bounds.append((0.001, 0.999))  # y must be within [0.001, 0.999] to allow for radius
                # radius bounds
                bounds.append((0.001, 0.49))   # radius bounded by 0.49 to stay within bounds
            
            # Define constraints
            cons = []
            
            # Containment constraints (all must be >= 0)
            def containment_func(x):
                return containment_constraints(x)
            
            # Overlap constraints (all must be >= 0)
            def overlap_func(x):
                return overlap_constraints(x)
            
            # Add constraints
            cons.append({'type': 'ineq', 'fun': containment_func})
            cons.append({'type': 'ineq', 'fun': overlap_func})
            
            # Use more robust optimization methods
            # First try differential evolution for global search
            try:
                de_result = differential_evolution(
                    objective,
                    bounds,
                    constraints=cons,
                    seed=42,
                    maxiter=500,
                    popsize=15,
                    mutation=(0.5, 1),
                    recombination=0.7,
                    atol=1e-6,
                    rtol=1e-6
                )
                
                if de_result.success:
                    # Calculate sum of radii for this result
                    final_radii = de_result.x.reshape(-1, 3)[:, 2]
                    sum_radii = np.sum(final_radii)
                    if sum_radii > best_sum:
                        best_sum = sum_radii
                        best_result = de_result
                        
            except Exception:
                pass
            
            # If differential evolution didn't work, try local optimization
            if best_result is None:
                methods = ['trust-constr', 'SLSQP']
                method_results = []
                
                for method in methods:
                    try:
                        result = minimize(
                            objective,
                            x0,
                            method=method,
                            bounds=bounds,
                            constraints=cons,
                            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8},
                            callback=callback
                        )
                        
                        if result.success:
                            # Calculate sum of radii for this result
                            final_radii = result.x.reshape(-1, 3)[:, 2]
                            sum_radii = np.sum(final_radii)
                            method_results.append((result, sum_radii))
                            
                    except Exception:
                        continue
                
                # Pick the best result from available methods
                if method_results:
                    best_method_result = max(method_results, key=lambda x: x[1])
                    if best_method_result[1] > best_sum:
                        best_sum = best_method_result[1]
                        best_result = best_method_result[0]
                        
        except Exception:
            continue
    
    # If no optimization worked, use the best initial configuration
    if best_result is None:
        # Use the best initial configuration directly
        best_initial_config = max(initial_configs, key=lambda config: np.sum(config[:, 2]))
        final_config = best_initial_config
        # Run a quick optimization to improve it slightly
        try:
            x0 = final_config.flatten()
            bounds = []
            for j in range(n):
                bounds.append((0.001, 0.999))
                bounds.append((0.001, 0.999))
                bounds.append((0.001, 0.49))
            
            cons = []
            def containment_func(x):
                return containment_constraints(x)
            def overlap_func(x):
                return overlap_constraints(x)
            
            cons.append({'type': 'ineq', 'fun': containment_func})
            cons.append({'type': 'ineq', 'fun': overlap_func})
            
            result = minimize(
                objective,
                x0,
                method='trust-constr',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                final_config = result.x.reshape(-1, 3)
            else:
                # Fall back to just using the initial config
                pass
        except Exception:
            pass
    else:
        # Use the optimized result
        final_config = best_result.x.reshape(-1, 3)
    
    # Final validation and cleanup
    positions = final_config[:, :2]
    radii = final_config[:, 2]
    
    # Ensure containment constraints with more careful adjustment
    for i in range(n):
        max_radius_x = min(positions[i, 0], 1 - positions[i, 0])
        max_radius_y = min(positions[i, 1], 1 - positions[i, 1])
        max_radius = min(max_radius_x, max_radius_y)
        radii[i] = min(radii[i], max_radius * 0.99)
    
    # More aggressive overlap resolution with iterative improvement
    max_iter = 50
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        changed = False
        distances = cdist(positions, positions)
        
        # Check for overlaps and resolve
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < radii[i] + radii[j]:
                    # Reduce radii to prevent overlap with better strategy
                    overlap = (radii[i] + radii[j]) - distances[i, j]
                    reduction = overlap * 0.3  # Slightly more aggressive
                    if radii[i] > reduction and radii[j] > reduction:
                        radii[i] -= reduction
                        radii[j] -= reduction
                        changed = True
        
        if not changed:
            break
    
    # Final correction to ensure all constraints
    for i in range(n):
        # Make sure radii don't exceed containment limits
        max_radius_x = min(positions[i, 0], 1 - positions[i, 0])
        max_radius_y = min(positions[i, 1], 1 - positions[i, 1])
        max_radius = min(max_radius_x, max_radius_y)
        radii[i] = min(radii[i], max_radius * 0.99)
    
    # Return final configuration
    return np.column_stack([positions, radii])


# EVOLVE-BLOCK-END
