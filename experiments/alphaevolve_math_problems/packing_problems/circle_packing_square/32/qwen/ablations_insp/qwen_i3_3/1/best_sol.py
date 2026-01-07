# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, KDTree
import warnings
warnings.filterwarnings('ignore')
import random
from scipy.spatial.distance import cdist

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies, physics-based refinement,
    and robust optimization to beat the benchmark.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    benchmark = 2.937944526205518
    
    # Better hexagonal initialization with proper spacing
    def initialize_hexagonal_layout():
        circles = []
        sqrt3 = np.sqrt(3)
        
        # Determine grid dimensions for approximately 32 circles
        rows = int(np.ceil(np.sqrt(n * 2 / sqrt3)))
        cols = int(np.ceil(n / rows))
        
        # Spacing based on hexagonal packing
        radius = 0.08  # Starting radius estimate
        horizontal_spacing = 2 * radius
        vertical_spacing = sqrt3 * radius
        
        # Create hexagonal grid
        for i in range(rows):
            y = radius + i * vertical_spacing
            if y > 1 - radius:
                break
            for j in range(cols):
                x = radius + j * horizontal_spacing
                if x > 1 - radius:
                    break
                # Offset every other row
                if i % 2 == 1:
                    x += horizontal_spacing / 2
                if x <= 1 - radius and y <= 1 - radius:
                    circles.append([x, y, radius])
        
        # Fill remaining circles with random placement near grid points
        while len(circles) < n:
            # Add random placements near existing grid points
            if circles:
                base_idx = np.random.randint(len(circles))
                base_x, base_y, base_r = circles[base_idx]
                x = np.clip(base_x + np.random.normal(0, 0.03), base_r, 1-base_r)
                y = np.clip(base_y + np.random.normal(0, 0.03), base_r, 1-base_r)
                circles.append([x, y, base_r])
            else:
                # If no circles yet, place randomly
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
                circles.append([x, y, 0.05])
                
        return np.array(circles[:n])
    
    # Voronoi-based initialization
    def initialize_voronoi():
        try:
            np.random.seed(42)  # For reproducibility
            points = np.random.rand(50, 2)  # More points than needed for Voronoi
            
            vor = Voronoi(points)
            # Use Voronoi cell centroids as initial positions (but keep within bounds)
            positions = []
            for i in range(min(n, len(vor.points))):
                point = vor.points[i]
                x = np.clip(point[0], 0.05, 0.95)
                y = np.clip(point[1], 0.05, 0.95)
                positions.append([x, y])
            
            # Fill remaining positions randomly
            while len(positions) < n:
                positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
                
            return np.array(positions[:n])
            
        except Exception:
            # Fallback to grid initialization
            positions = []
            grid_size = int(np.ceil(np.sqrt(n)))
            spacing_x = 1.0 / (grid_size + 1)
            spacing_y = 1.0 / (grid_size + 1)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if len(positions) >= n:
                        break
                    x = (j + 1) * spacing_x
                    y = (i + 1) * spacing_y
                    positions.append([x, y])
                    
            # Fill remaining positions randomly
            while len(positions) < n:
                positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
                
            return np.array(positions[:n])
    
    # Grid-based initialization
    def initialize_grid_placement():
        # Create a grid layout
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Calculate spacing
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        circles = np.zeros((n, 3))
        
        for i in range(n):
            row = i // cols
            col = i % cols
            
            # Base position with some padding
            x_base = (col + 0.5) * spacing_x
            y_base = (row + 0.5) * spacing_y
            
            # Add small random perturbation
            perturbation = 0.05 * spacing_x
            x = max(perturbation, min(1.0 - perturbation, x_base + random.uniform(-perturbation, perturbation)))
            y = max(perturbation, min(1.0 - perturbation, y_base + random.uniform(-perturbation, perturbation)))
            
            # Initial radius - start with small values
            r = min(spacing_x, spacing_y) * 0.2
            
            circles[i] = [x, y, r]
        
        return circles
    
    # Physics-based refinement to improve packing quality with spatial indexing
    def apply_physics_refinement(circles, iterations=500, learning_rate=0.01):
        """Apply physics-based repulsive forces for better packing with spatial indexing"""
        circles = circles.copy()
        n = len(circles)
        
        # Precompute indices for efficient neighbor detection
        for iter_num in range(iterations):
            forces = np.zeros((n, 2))
            
            # Use KDTree for efficient neighbor searching
            positions = circles[:, :2]
            tree = KDTree(positions)
            
            # Find neighbors within a reasonable distance
            # This is more efficient than pairwise checking
            for i in range(n):
                # Find nearby circles (within 3x average radius)
                x1, y1, r1 = circles[i]
                # Query neighbors within a reasonable range
                indices = tree.query_ball_point([x1, y1], 3*(r1 if r1 > 0.01 else 0.05))
                
                for j in indices:
                    if i != j:
                        x2, y2, r2 = circles[j]
                        
                        dx = x1 - x2
                        dy = y1 - y2
                        distance = np.sqrt(dx*dx + dy*dy)
                        
                        # Only apply force if circles are overlapping or very close
                        if distance > 0 and distance < (r1 + r2) * 1.1:  # Allow some tolerance
                            force_magnitude = max(0, (r1 + r2 - distance) / (distance + 1e-8))
                            forces[i, 0] += force_magnitude * dx / distance
                            forces[i, 1] += force_magnitude * dy / distance
                            forces[j, 0] -= force_magnitude * dx / distance
                            forces[j, 1] -= force_magnitude * dy / distance
            
            # Apply forces with boundary constraints
            for i in range(n):
                # Move circle
                new_x = circles[i, 0] + learning_rate * forces[i, 0]
                new_y = circles[i, 1] + learning_rate * forces[i, 1]
                
                # Boundary constraints
                new_x = np.clip(new_x, circles[i, 2], 1 - circles[i, 2])
                new_y = np.clip(new_y, circles[i, 2], 1 - circles[i, 2])
                
                circles[i, 0] = new_x
                circles[i, 1] = new_y
                
            # Occasionally try to increase radii
            if iter_num % 100 == 0:
                for i in range(n):
                    x, y, r = circles[i]
                    # Try to increase radius if possible
                    max_radius = min(x, 1-x, y, 1-y)
                    
                    # Check distance to other circles using KDTree for efficiency
                    if n > 1:
                        tree = KDTree(circles[:, :2])
                        indices = tree.query_ball_point([x, y], 1.0)  # Find all nearby circles
                        for idx in indices:
                            if idx != i:
                                x2, y2, r2 = circles[idx]
                                distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                                max_radius = min(max_radius, distance - r2)
                    
                    if max_radius > r + 1e-6 and max_radius > 0.001:
                        circles[i, 2] = max_radius
        
        return circles
    
    # Enhanced refinement with better convergence criteria and spatial acceleration
    def refine_radii(circles_in):
        """Refine radii after positioning to maximize sum while respecting constraints"""
        circles_refined = circles_in.copy()
        n = len(circles_refined)
        improved = True
        max_iterations = 100
        
        # Precompute spatial structure for faster neighbor lookups
        tree = None
        
        for _ in range(max_iterations):
            if not improved:
                break
            improved = False
            
            # Rebuild tree for better neighbor detection
            if n > 10:
                positions = circles_refined[:, :2]
                try:
                    tree = KDTree(positions)
                except:
                    tree = None
            
            # Try to increase radii for each circle
            for i in range(n):
                x, y, r = circles_refined[i]
                old_r = r
                
                # Calculate maximum possible radius for this circle
                max_radius = min(x, 1-x, y, 1-y)
                
                # Check overlap with all other circles efficiently
                if tree is not None and n > 10:
                    # Use spatial indexing for neighbor detection
                    indices = tree.query_ball_point([x, y], 1.0)
                    for idx in indices:
                        if idx != i:
                            x2, y2, r2 = circles_refined[idx]
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            # Add small epsilon to prevent numerical issues
                            max_radius = min(max_radius, distance - r2 - 1e-8)
                else:
                    # Fallback to direct computation for small numbers
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = circles_refined[j]
                            distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                            # Add small epsilon to prevent numerical issues
                            max_radius = min(max_radius, distance - r2 - 1e-8)
                
                # Increase radius if beneficial
                if max_radius > r + 1e-6:
                    circles_refined[i, 2] = max_radius
                    improved = True
                    
        return circles_refined
    
    # Final validation
    def validate_and_correct(circles_array):
        corrected = circles_array.copy()
        for i in range(n):
            x, y, r = corrected[i]
            # Ensure circle fits in unit square
            r = min(r, x, 1-x, y, 1-y)
            # Ensure positive radius
            r = max(1e-6, r)
            corrected[i] = [x, y, r]
        return corrected
    
    # Objective function: maximize sum of radii (minimize negative sum)
    def objective(params):
        return -np.sum(params[2::3])  # Sum of all radii (every 3rd element starting from index 2)
    
    # Constraint functions
    def constraint_containment(params):
        # Ensure all circles fit inside the unit square
        constraints = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            # Circle must stay inside square: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
            constraints.extend([
                x - r,           # x >= r
                1 - x - r,       # 1 - x >= r
                y - r,           # y >= r
                1 - y - r        # 1 - y >= r
            ])
        return np.array(constraints)
    
    # Non-overlap constraints - optimized version
    def constraint_nonoverlap(params):
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                
                # Distance between centers
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                dist = np.sqrt(dist_sq)
                
                # Non-overlap constraint: distance >= r1 + r2
                constraints.append(dist - (r1 + r2))
        return np.array(constraints)
    
    # Bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])  # x, y, r
    
    # Constraints list
    cons = [
        {'type': 'ineq', 'fun': constraint_containment},
        {'type': 'ineq', 'fun': constraint_nonoverlap}
    ]
    
    # Enhanced optimization approach with multiple methods and better parameters
    def optimize_circles(initial_circles):
        # Flatten for optimization
        initial_params = initial_circles.flatten()
        
        # Try multiple optimization methods for better results
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_value = float('inf')
        
        for method in methods:
            try:
                result = minimize(
                    objective, 
                    initial_params, 
                    method=method, 
                    bounds=bounds, 
                    constraints=cons,
                    options={'maxiter': 800, 'ftol': 1e-9, 'gtol': 1e-9}
                )
                
                if result.success:
                    # Check if this is better than previous attempts
                    if result.fun < best_value:
                        best_value = result.fun
                        best_result = result
                        
            except Exception:
                continue
        
        # Return best result or fallback to initial
        if best_result is not None and best_result.success:
            return best_result.x.reshape((n, 3))
        else:
            # If optimization fails, return initial configuration with corrected radii
            corrected = initial_circles.copy()
            for i in range(n):
                x, y, r = corrected[i]
                corrected[i, 2] = min(r, x, 1-x, y, 1-y)
            return corrected
    
    # Try multiple initialization strategies and optimization attempts
    best_circles = None
    best_sum = 0
    
    # Try different initialization methods with more aggressive attempts
    init_methods = [initialize_hexagonal_layout, initialize_voronoi, initialize_grid_placement]
    
    for init_method in init_methods:
        # Try several optimization attempts with same initialization
        for attempt in range(7):  # Increased attempts for better chance of success
            try:
                # Get initial configuration
                if attempt == 0:
                    circles = init_method()
                else:
                    # Perturb previous result with more variation in later attempts
                    circles = best_circles.copy() if best_circles is not None else init_method()
                    for i in range(n):
                        # More significant perturbations in later attempts
                        strength = 0.02 if attempt < 3 else (0.05 if attempt < 5 else 0.1)
                        circles[i, 0] += np.random.normal(0, strength)
                        circles[i, 1] += np.random.normal(0, strength)
                        circles[i, 0] = np.clip(circles[i, 0], 0.01, 0.99)
                        circles[i, 1] = np.clip(circles[i, 1], 0.01, 0.99)
                
                # Apply physics-based refinement before optimization
                circles = apply_physics_refinement(circles, iterations=400, learning_rate=0.01)
                
                # Optimize with enhanced parameters
                optimized_circles = optimize_circles(circles)
                
                # Refine radii aggressively
                refined_circles = refine_radii(optimized_circles)
                
                # Validate
                final_circles = validate_and_correct(refined_circles)
                
                # Calculate sum of radii
                radii_sum = np.sum(final_circles[:, 2])
                if radii_sum > best_sum:
                    best_sum = radii_sum
                    best_circles = final_circles
                    
            except Exception:
                continue
    
    # If we still don't have a good solution, return fallback
    if best_circles is None:
        # Final fallback: hexagonal layout with refinement
        fallback = initialize_hexagonal_layout()
        best_circles = refine_radii(fallback)
    
    return best_circles


# EVOLVE-BLOCK-END
