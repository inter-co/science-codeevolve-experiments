# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies, physics-inspired 
    optimization, and multiple refinement phases.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Multiple initialization strategies for better exploration (like INSPIRATION 2)
    def generate_initial_configurations():
        configs = []
        
        # Strategy 1: Hexagonal packing with better spacing (like INSPIRATION 2)
        def hexagonal_strategy():
            circles = []
            rows = 6
            cols_per_row = [5, 6, 5, 6, 5, 6]
            y_offset = 0.05
            row_radius = 0.08
            
            for i in range(rows):
                row_cols = cols_per_row[i]
                x_offset = 0.05 if i % 2 == 0 else 0.1  # Offset every other row
                for j in range(row_cols):
                    x = x_offset + j * (row_radius * 2.1)
                    y = y_offset + i * (row_radius * 1.732)  # sqrt(3) for hexagonal packing
                    
                    if x + row_radius <= 0.95 and y + row_radius <= 0.95:
                        circles.append([x, y, row_radius])
                        
            # Fill remaining with smaller circles
            while len(circles) < n:
                x = 0.05 + np.random.rand() * 0.9
                y = 0.05 + np.random.rand() * 0.9
                r = 0.01 + np.random.rand() * 0.04
                circles.append([x, y, r])
                
            return np.array(circles[:n])
        
        # Strategy 2: Fibonacci spiral for better distribution (like INSPIRATION 2)
        def fibonacci_strategy():
            circles = []
            golden_ratio = (1 + np.sqrt(5)) / 2.0
            
            for i in range(n):
                # Distribute points on a sphere then project to 2D
                theta = np.arccos(-1 + (2 * i) / (n - 1))
                phi = np.sqrt(n * np.pi) * theta
                
                # Convert to Cartesian coordinates and map to square
                x = 0.9 * (np.sin(theta) * np.cos(phi) + 1) / 2
                y = 0.9 * (np.sin(theta) * np.sin(phi) + 1) / 2
                r = 0.03
                
                circles.append([x, y, r])
                
            return np.array(circles)
        
        # Strategy 3: Grid with randomized offsets (like INSPIRATION 2)
        def grid_strategy():
            circles = []
            rows = cols = int(np.ceil(np.sqrt(n)))
            padding = 0.03
            cell_size = (1 - 2*padding) / max(rows, cols)
            
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= n:
                        break
                    x = padding + (j + 0.5) * cell_size + np.random.uniform(-0.1, 0.1) * cell_size
                    y = padding + (i + 0.5) * cell_size + np.random.uniform(-0.1, 0.1) * cell_size
                    # Clamp to valid range
                    x = np.clip(x, 0.01, 0.99)
                    y = np.clip(y, 0.01, 0.99)
                    r = 0.03
                    circles.append([x, y, r])
                    
            return np.array(circles[:n])
        
        # Strategy 4: Concentric rings (like INSPIRATION 2)
        def concentric_strategy():
            circles = []
            center = 0.5
            max_radius = 0.4
            
            # Place points in concentric rings
            ring_count = max(1, int(np.sqrt(n) / 2))
            points_per_ring = max(1, n // ring_count)
            
            for ring in range(ring_count):
                radius = max_radius * (ring + 1) / ring_count
                angle_step = 2 * np.pi / points_per_ring
                
                for point in range(points_per_ring):
                    if len(circles) >= n:
                        break
                    angle = point * angle_step
                    x = center + radius * np.cos(angle)
                    y = center + radius * np.sin(angle)
                    r = 0.03
                    circles.append([x, y, r])
                    
            # Fill remaining with random circles
            while len(circles) < n:
                x = 0.05 + np.random.rand() * 0.9
                y = 0.05 + np.random.rand() * 0.9
                r = 0.01 + np.random.rand() * 0.04
                circles.append([x, y, r])
                
            return np.array(circles)
        
        # Strategy 5: Precomputed high-quality configuration (like INSPIRATION 2)
        def precomputed_strategy():
            # Create a better starting configuration
            circles = np.zeros((n, 3))
            
            # Hexagonal pattern in center
            hex_rows, hex_cols = 4, 4
            hex_spacing_x = 0.6 / hex_cols
            hex_spacing_y = 0.6 * 0.866 / hex_rows
            hex_radius = hex_spacing_x * 0.35
            
            idx = 0
            for i in range(hex_rows):
                for j in range(hex_cols):
                    if idx >= n:
                        break
                    x = 0.2 + j * hex_spacing_x + (i % 2) * hex_spacing_x * 0.5
                    y = 0.2 + i * hex_spacing_y
                    circles[idx] = [x, y, hex_radius]
                    idx += 1
            
            # Fill remaining with more evenly distributed points
            for i in range(idx, n):
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                r = np.random.uniform(0.02, 0.06)
                circles[i] = [x, y, r]
            
            return circles
        
        # Strategy 6: Physics-inspired initialization (like INSPIRATION 1)
        def physics_init_strategy():
            # Create a configuration similar to what a physics simulation might produce
            circles = []
            
            # Start with a hexagonal pattern in the center
            hex_rows, hex_cols = 3, 3
            hex_spacing_x = 0.5 / hex_cols
            hex_spacing_y = 0.5 * 0.866 / hex_rows
            hex_radius = hex_spacing_x * 0.4
            
            for i in range(hex_rows):
                for j in range(hex_cols):
                    if len(circles) >= n:
                        break
                    x = 0.25 + j * hex_spacing_x + (i % 2) * hex_spacing_x * 0.5
                    y = 0.25 + i * hex_spacing_y
                    circles.append([x, y, hex_radius])
            
            # Add some random circles to fill gaps
            while len(circles) < n:
                x = 0.05 + np.random.rand() * 0.9
                y = 0.05 + np.random.rand() * 0.9
                r = 0.02 + np.random.rand() * 0.05
                circles.append([x, y, r])
                
            return np.array(circles[:n])
        
        configs.extend([
            hexagonal_strategy(),
            fibonacci_strategy(), 
            grid_strategy(),
            concentric_strategy(),
            precomputed_strategy(),
            physics_init_strategy()
        ])
        
        return configs
    
    # Improved constraint checking with better performance (like INSPIRATION 1)
    def check_constraints(positions_radii):
        positions = positions_radii[:, :2]
        radii = positions_radii[:, 2]
        
        # Check containment constraints
        if np.any(positions[:, 0] - radii < 0) or np.any(positions[:, 0] + radii > 1) or \
           np.any(positions[:, 1] - radii < 0) or np.any(positions[:, 1] + radii > 1):
            return False
        
        # Check overlap constraints using vectorized computation (like INSPIRATION 1)
        if len(positions) > 1:
            distances = cdist(positions, positions)
            # Create overlap matrix (excluding diagonal)
            overlap_matrix = distances < (radii.reshape(-1, 1) + radii.reshape(1, -1))
            np.fill_diagonal(overlap_matrix, False)
            
            if np.any(overlap_matrix):
                return False
                
        return True
    
    # Objective function (negative because we want to maximize)
    def objective(params):
        positions_radii = params.reshape(-1, 3)
        return -np.sum(positions_radii[:, 2])  # Negative because we want to maximize
    
    # Constraint functions with better handling (like INSPIRATION 1)
    def containment_constraint(params):
        positions_radii = params.reshape(-1, 3)
        positions = positions_radii[:, :2]
        radii = positions_radii[:, 2]
        constraints = []
        
        # Boundary constraints: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
        for i in range(len(positions)):
            x, y = positions[i]
            r = radii[i]
            constraints.extend([
                x - r,          # x - r >= 0
                1 - x - r,      # 1 - x - r >= 0  
                y - r,          # y - r >= 0
                1 - y - r       # 1 - y - r >= 0
            ])
        return np.array(constraints)
    
    def overlap_constraint(params):
        positions_radii = params.reshape(-1, 3)
        positions = positions_radii[:, :2]
        radii = positions_radii[:, 2]
        constraints = []
        
        # Overlap constraints: distance >= r1 + r2 (positive means no overlap)
        distances = cdist(positions, positions)
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = distances[i, j]
                constraints.append(dist - (radii[i] + radii[j]))
        return np.array(constraints)
    
    # Enhanced optimization approach with better restarts and bounds (like INSPIRATION 2)
    def optimize_with_restarts(initial_config):
        best_result = None
        best_sum = -np.inf
        
        # Multiple restarts with different perturbations (like INSPIRATION 2)
        for restart in range(15):  # More restarts for better exploration
            # Create perturbed version of initial configuration
            perturbed_config = initial_config.copy()
            
            # Different types of perturbations for better exploration (like INSPIRATION 2)
            if restart < 7:  # Some random noise
                noise_scale = 0.02 * (restart + 1) / 7
                perturbed_config[:, 0] += np.random.normal(0, noise_scale, n)
                perturbed_config[:, 1] += np.random.normal(0, noise_scale, n)
                perturbed_config[:, 2] += np.random.normal(0, noise_scale * 0.5, n)
            else:  # More aggressive perturbations for later restarts
                perturbed_config[:, 0] += np.random.uniform(-0.03, 0.03, n)
                perturbed_config[:, 1] += np.random.uniform(-0.03, 0.03, n)
                perturbed_config[:, 2] += np.random.uniform(-0.02, 0.02, n)
            
            # Clip to valid ranges
            perturbed_config[:, 0] = np.clip(perturbed_config[:, 0], 0.001, 0.999)
            perturbed_config[:, 1] = np.clip(perturbed_config[:, 1], 0.001, 0.999)
            perturbed_config[:, 2] = np.clip(perturbed_config[:, 2], 0.001, 0.499)
            
            # Flatten for optimization
            initial_params = perturbed_config.flatten()
            
            # Define bounds (like INSPIRATION 2)
            bounds = []
            for _ in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            # Define constraints (like INSPIRATION 2)
            cons = [
                {'type': 'ineq', 'fun': lambda x: containment_constraint(x)},
                {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
            ]
            
            try:
                # Try multiple optimization methods for robustness
                result = minimize(
                    objective,
                    initial_params,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 2000, 'ftol': 1e-7, 'gtol': 1e-7}  # Tighter tolerances
                )
                
                if result.success:
                    current_sum = -result.fun
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
                        
            except Exception as e:
                continue
        
        return best_result, best_sum
    
    # Generate initial configurations and find the best one
    initial_configs = generate_initial_configurations()
    best_final_result = None
    best_final_sum = -np.inf
    
    # Try each initial configuration (like INSPIRATION 2)
    for config in initial_configs:
        result, sum_val = optimize_with_restarts(config)
        if result is not None and sum_val > best_final_sum:
            best_final_sum = sum_val
            best_final_result = result
    
    # If we found a good solution through optimization
    if best_final_result is not None:
        final_params = best_final_result.x
        final_circles = final_params.reshape(-1, 3)
        
        # Final validation and cleanup (like INSPIRATION 1)
        positions_radii = final_circles.copy()
        positions = positions_radii[:, :2]
        radii = positions_radii[:, 2]
        
        # Make sure all circles are within bounds
        for i in range(len(positions)):
            x, y = positions[i]
            r = radii[i]
            # Adjust if out of bounds
            if x - r < 0:
                x = r
            elif x + r > 1:
                x = 1 - r
            if y - r < 0:
                y = r
            elif y + r > 1:
                y = 1 - r
            positions[i] = [x, y]
        
        # Return final result
        final_circles = np.column_stack([positions, radii])
        return final_circles
    
    # Fallback to the first configuration if optimization fails
    return initial_configs[0]


# EVOLVE-BLOCK-END
