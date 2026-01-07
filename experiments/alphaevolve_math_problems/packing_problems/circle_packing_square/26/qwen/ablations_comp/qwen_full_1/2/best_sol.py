# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric insights with advanced optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Simplified and more robust initial placement
    def generate_initial_placement(num_circles):
        # Start with a regular grid pattern that's easy to understand and implement
        rows = int(np.ceil(np.sqrt(num_circles)))
        cols = int(np.ceil(num_circles / rows))
        
        positions = []
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= num_circles:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Add small random perturbations to break symmetries
                x += np.random.normal(0, spacing_x * 0.05)
                y += np.random.normal(0, spacing_y * 0.05)
                
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                positions.append([x, y])
        
        return np.array(positions[:num_circles])
    
    # Generate initial configuration
    initial_positions = generate_initial_placement(n)
    
    # Initialize radii with better estimates
    # Based on density considerations for circles in unit square
    avg_area_per_circle = 0.8 / n  # Leave 20% space for boundary effects
    estimated_radius = np.sqrt(avg_area_per_circle / np.pi) * 0.9  # Slightly conservative
    
    initial_radii = np.full(n, max(0.01, estimated_radius))
    
    # Add some variation to help optimization
    for i in range(n):
        variation_factor = 0.8 + np.random.random() * 0.4  # Range 0.8 to 1.2
        initial_radii[i] *= variation_factor
    
    # Ensure radii are within reasonable bounds
    initial_radii = np.clip(initial_radii, 0.01, 0.4)
    
    # Combine positions and radii into a single parameter vector
    initial_params = np.concatenate([initial_positions.flatten(), initial_radii])
    
    # Define constraint functions with improved numerical stability
    def containment_constraints(params):
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        constraints = []
        
        # Each circle must be fully contained in unit square
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            # Add safety margins for numerical stability
            constraints.extend([
                x - r - 1e-8,           # x >= r + safety
                y - r - 1e-8,           # y >= r + safety
                1 - x - r - 1e-8,       # 1 - x >= r + safety
                1 - y - r - 1e-8        # 1 - y >= r + safety
            ])
        return np.array(constraints)
    
    def non_overlap_constraints(params):
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        constraints = []
        
        # Check all pairs of circles for overlap using squared distances
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                distance_squared = dx*dx + dy*dy
                min_distance_squared = (radii[i] + radii[j])**2
                # We want distance >= min_distance, so we enforce constraint: distance^2 - min_distance^2 >= 0
                # Add a small epsilon to handle numerical precision issues
                constraints.append(distance_squared - min_distance_squared - 1e-12)
        return np.array(constraints)
    
    # Objective function to maximize (negative because minimize)
    def objective(params):
        radii = params[2*n:]
        return -np.sum(radii)
    
    # Create bounds for parameters
    bounds = []
    # Position bounds [0,1] for both x and y coordinates
    for _ in range(2*n):
        bounds.extend([(0, 1)])
    # Radius bounds [0, 0.5] - reasonable upper bound
    for _ in range(n):
        bounds.extend([(0, 0.5)])
    
    # Multi-strategy optimization approach - simplified but effective
    best_result = None
    best_sum = -np.inf
    
    # Strategy 1: SLSQP with multiple restarts
    for attempt in range(5):  # Reduced restarts for efficiency
        np.random.seed(attempt * 1000 + 42)  # Fixed seed for reproducibility
        
        # Create perturbed initial parameters
        perturbed_positions = initial_positions.copy()
        perturbed_radii = initial_radii.copy()
        
        # Perturbations
        for i in range(n):
            perturbed_positions[i, 0] += np.random.normal(0, 0.01)
            perturbed_positions[i, 1] += np.random.normal(0, 0.01)
            perturbed_radii[i] *= (0.9 + np.random.random() * 0.2)  # Scale radii by factor 0.9-1.1
            
        # Keep within bounds
        perturbed_positions[:, 0] = np.clip(perturbed_positions[:, 0], 0.05, 0.95)
        perturbed_positions[:, 1] = np.clip(perturbed_positions[:, 1], 0.05, 0.95)
        perturbed_radii = np.clip(perturbed_radii, 0.01, 0.5)
        
        current_params = np.concatenate([perturbed_positions.flatten(), perturbed_radii])
        
        try:
            result = minimize(
                objective,
                current_params,
                method='SLSQP',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                    {'type': 'ineq', 'fun': lambda p: non_overlap_constraints(p)}
                ],
                options={'maxiter': 2000, 'ftol': 1e-8, 'gtol': 1e-8},
                tol=1e-8
            )
            
            if result.success:
                final_radii = result.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            continue
    
    # Strategy 2: Trust-constr for better handling of constraints
    if best_result is None or best_sum < 2.6:  # Only if we haven't found a good solution yet
        try:
            # Use a fresh initialization with better distribution
            np.random.seed(12345)
            
            # Create positions that are more spread out
            positions = []
            for i in range(n):
                # Mix of edge and interior placements
                if np.random.random() < 0.25:  # 25% chance of edge placement
                    x = np.random.uniform(0.05, 0.95)
                    y = np.random.choice([0.05, 0.95])
                else:  # 75% chance of interior
                    x = np.random.uniform(0.1, 0.9)
                    y = np.random.uniform(0.1, 0.9)
                positions.append([x, y])
            
            positions = np.array(positions)
            
            # Initialize with more varied radii
            radii = np.random.uniform(0.08, 0.25, n)
            radii = np.clip(radii, 0.01, 0.4)
            
            params = np.concatenate([positions.flatten(), radii])
            
            # Use trust-constr method
            result_trust = minimize(
                objective,
                params,
                method='trust-constr',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: containment_constraints(p)},
                    {'type': 'ineq', 'fun': lambda p: non_overlap_constraints(p)}
                ],
                options={'maxiter': 2000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if result_trust.success:
                final_radii = result_trust.x[2*n:]
                current_sum = np.sum(final_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result_trust
        except Exception as e:
            pass
    
    # Return best result or fallback to initial
    if best_result is not None and best_result.success:
        final_positions = best_result.x[:2*n].reshape(-1, 2)
        final_radii = best_result.x[2*n:]
        circles = np.column_stack([final_positions, final_radii])
        return circles
    
    # Fallback: return the initial configuration with some adjustments
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = initial_positions[i, 0]
        circles[i, 1] = initial_positions[i, 1]
        circles[i, 2] = initial_radii[i]
    return circles


# EVOLVE-BLOCK-END
