# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import warnings
import random
warnings.filterwarnings('ignore')

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining multiple initialization strategies with advanced optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    n = 32
    
    # Enhanced initialization inspired by both inspirations
    def initialize_enhanced():
        # Strategy 1: Voronoi-inspired initialization (from INSPIRATION 1)
        def voronoi_initial():
            # Start with a grid pattern and relax using Voronoi concepts
            rows = 6
            cols = 6
            
            positions = []
            spacing_x = 0.9 / (cols + 1)
            spacing_y = 0.9 / (rows + 1)
            
            for i in range(rows):
                for j in range(cols):
                    if len(positions) >= n:
                        break
                    x = 0.05 + (j + 1) * spacing_x + random.uniform(-spacing_x*0.1, spacing_x*0.1)
                    y = 0.05 + (i + 1) * spacing_y + random.uniform(-spacing_y*0.1, spacing_y*0.1)
                    positions.append([x, y])
            
            # Relax positions using Voronoi-like technique
            for _ in range(50):
                if len(positions) >= n:
                    break
                new_positions = []
                for i, (x, y) in enumerate(positions):
                    # Move towards center of mass of neighbors
                    if len(positions) > 1:
                        distances = [(np.sqrt((x - px)**2 + (y - py)**2), (px, py)) 
                                   for px, py in positions if (px, py) != (x, y)]
                        distances.sort(key=lambda x: x[0])
                        neighbors = distances[:min(5, len(distances))]
                        
                        if neighbors:
                            avg_x = sum(px for _, (px, py) in neighbors) / len(neighbors)
                            avg_y = sum(py for _, (px, py) in neighbors) / len(neighbors)
                            # Move towards average of neighbors
                            x = x * 0.8 + avg_x * 0.2
                            y = y * 0.8 + avg_y * 0.2
                    
                    # Keep within bounds
                    x = max(0.01, min(0.99, x))
                    y = max(0.01, min(0.99, y))
                    new_positions.append([x, y])
                positions = new_positions[:n]
            
            # Fill any remaining positions
            while len(positions) < n:
                x = random.uniform(0.01, 0.99)
                y = random.uniform(0.01, 0.99)
                positions.append([x, y])
            
            # Initialize with small radii
            circles = np.array([[pos[0], pos[1], 0.02] for pos in positions[:n]])
            return circles
        
        # Strategy 2: Improved hexagonal grid (from INSPIRATION 2)
        def hexagonal_grid_initial():
            circles = []
            rows = 6
            cols = 6
            
            # Better hexagonal packing parameters
            spacing_x = 0.15
            spacing_y = 0.1299  # sqrt(3)/2 * spacing_x
            offset = 0.05
            
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= n:
                        break
                    x = offset + j * spacing_x
                    y = offset + i * spacing_y
                    if i % 2 == 1:
                        x += spacing_x / 2
                    # Ensure within bounds
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        circles.append([x, y, 0.03])
                if len(circles) >= n:
                    break
            
            # Fill remaining with random points near center
            while len(circles) < n:
                x = 0.5 + (np.random.random() - 0.5) * 0.4
                y = 0.5 + (np.random.random() - 0.5) * 0.4
                circles.append([x, y, 0.03])
            
            return np.array(circles[:n])
        
        # Strategy 3: Golden ratio spiral (from INSPIRATION 1)
        def golden_spiral_initial():
            circles = []
            golden_ratio = (1 + np.sqrt(5)) / 2
            
            for i in range(n):
                angle = 2 * np.pi * i / golden_ratio
                radius = np.sqrt(i / (n - 1)) if i < n - 1 else 0.99
                x = 0.5 + radius * np.cos(angle) * 0.45
                y = 0.5 + radius * np.sin(angle) * 0.45
                
                # Ensure we're within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                circles.append([x, y, 0.03])
            
            return np.array(circles)
        
        # Strategy 4: Hybrid approach combining multiple methods
        def hybrid_initial():
            # Start with hexagonal grid
            circles = hexagonal_grid_initial()
            
            # Perturb positions slightly to avoid perfect symmetry
            for i in range(len(circles)):
                circles[i][0] += random.uniform(-0.02, 0.02)
                circles[i][1] += random.uniform(-0.02, 0.02)
                circles[i][0] = max(0.01, min(0.99, circles[i][0]))
                circles[i][1] = max(0.01, min(0.99, circles[i][1]))
            
            return circles
        
        # Try all strategies and pick the best
        strategies = [voronoi_initial, hexagonal_grid_initial, golden_spiral_initial, hybrid_initial]
        best_circles = None
        best_sum = 0
        
        for strategy in strategies:
            try:
                circles = strategy()
                # Quick optimization to get better initial radii
                circles = optimize_local(circles, max_iter=100)
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
            except Exception as e:
                continue
        
        # Fallback to simple initialization
        if best_circles is None:
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [0.5, 0.5, 0.02]
            return circles
            
        return best_circles
    
    # Enhanced local optimization (from INSPIRATION 1 & 2)
    def optimize_local(circles, max_iter=100):
        """Perform enhanced local optimization on circle positions/radii"""
        n = len(circles)
        
        # Flatten for optimization
        initial_params = circles.flatten()
        
        def objective(params):
            positions_and_radii = params.reshape(-1, 3)
            return -np.sum(positions_and_radii[:, 2])
        
        def constraint_containment(params):
            positions_and_radii = params.reshape(-1, 3)
            constraints = []
            for i in range(n):
                x, y, r = positions_and_radii[i]
                # Add small safety margin to avoid numerical issues
                constraints.extend([
                    x - r + 1e-8,      # x - r >= 0
                    1 - x - r + 1e-8,  # 1 - x - r >= 0
                    y - r + 1e-8,      # y - r >= 0
                    1 - y - r + 1e-8   # 1 - y - r >= 0
                ])
            return np.array(constraints)
        
        def constraint_nonoverlap(params):
            positions_and_radii = params.reshape(-1, 3)
            constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = positions_and_radii[i]
                    x2, y2, r2 = positions_and_radii[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    distance = np.sqrt(dx*dx + dy*dy)
                    # Add small epsilon to avoid numerical issues
                    constraints.append(distance - (r1 + r2) + 1e-8)
            return np.array(constraints)
        
        # Set bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        try:
            # Try different optimization methods - prioritize trust-constr for better results
            methods = ['trust-constr', 'SLSQP']
            for method in methods:
                try:
                    result = minimize(
                        objective,
                        initial_params,
                        method=method,
                        bounds=bounds,
                        constraints=[
                            {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
                            {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
                        ],
                        options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6}
                    )
                    
                    if result.success:
                        optimized_circles = result.x.reshape(-1, 3)
                        # Ensure final containment and feasibility
                        for i in range(len(optimized_circles)):
                            x, y, r = optimized_circles[i]
                            optimized_circles[i] = [
                                max(0.001, min(0.999, x)),
                                max(0.001, min(0.999, y)),
                                max(0.001, min(0.499, r))
                            ]
                        return optimized_circles
                except Exception:
                    continue
        except Exception as e:
            pass
        
        return circles
    
    # Advanced optimization with multiple refinement phases
    def advanced_optimization(initial_circles):
        circles = initial_circles.copy()
        
        # Phase 1: Global optimization with trust-constr (better for constrained problems)
        try:
            n = len(circles)
            initial_params = circles.flatten()
            
            def objective(params):
                positions_and_radii = params.reshape(-1, 3)
                return -np.sum(positions_and_radii[:, 2])
            
            def constraint_containment(params):
                positions_and_radii = params.reshape(-1, 3)
                constraints = []
                for i in range(n):
                    x, y, r = positions_and_radii[i]
                    constraints.extend([
                        x - r + 1e-8,      # x - r >= 0
                        1 - x - r + 1e-8,  # 1 - x - r >= 0
                        y - r + 1e-8,      # y - r >= 0
                        1 - y - r + 1e-8   # 1 - y - r >= 0
                    ])
                return np.array(constraints)
            
            def constraint_nonoverlap(params):
                positions_and_radii = params.reshape(-1, 3)
                constraints = []
                for i in range(n):
                    for j in range(i+1, n):
                        x1, y1, r1 = positions_and_radii[i]
                        x2, y2, r2 = positions_and_radii[j]
                        dx = x1 - x2
                        dy = y1 - y2
                        distance = np.sqrt(dx*dx + dy*dy)
                        constraints.append(distance - (r1 + r2) + 1e-8)
                return np.array(constraints)
            
            bounds = []
            for i in range(n):
                bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
            
            result = minimize(
                objective,
                initial_params,
                method='trust-constr',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
                    {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
                ],
                options={'maxiter': 500, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            
            if result.success:
                circles = result.x.reshape(-1, 3)
        except Exception:
            pass
        
        # Phase 2: Iterative refinement with better constraint handling
        # Use local search with spatial indexing for better performance
        try:
            # More aggressive refinement
            for iteration in range(200):
                improved = False
                # For each circle, try to increase its radius while maintaining constraints
                for i in range(len(circles)):
                    original_pos = circles[i][:2]
                    original_radius = circles[i][2]
                    
                    # Calculate maximum possible radius at current position
                    max_r = min(original_pos[0], 1-original_pos[0], original_pos[1], 1-original_pos[1])
                    
                    # Check overlap with all others
                    for j in range(len(circles)):
                        if i != j:
                            dx = original_pos[0] - circles[j][0]
                            dy = original_pos[1] - circles[j][1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            max_r = min(max_r, dist - circles[j][2])
                    
                    max_r = max(max_r, 0.001)
                    
                    if max_r > original_radius:
                        circles[i][2] = max_r
                        improved = True
                
                if not improved:
                    break
        except Exception:
            pass
        
        return circles
    
    # Improved constraint checking using more robust methods
    def check_constraints(circles_arr):
        """Efficiently check all constraints using spatial indexing and better numerical handling"""
        # Check boundary constraints
        for i in range(len(circles_arr)):
            x, y, r = circles_arr[i]
            if not (r <= x <= 1-r and r <= y <= 1-r):
                return False
        
        # Check overlap constraints using spatial indexing for efficiency
        positions = circles_arr[:, :2]
        radii = circles_arr[:, 2]
        
        # Use more robust distance calculation with tolerance
        tree = cKDTree(positions)
        
        # Query all pairs that might be close enough to overlap
        pairs = tree.query_pairs(2 * np.max(radii), output_type='ndarray')
        
        for i, j in pairs:
            if i < j:  # Only check each pair once
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (radii[i] + radii[j])**2
                
                # Add small tolerance to handle numerical precision issues
                if dist_sq < min_dist_sq - 1e-10:
                    return False
        return True
    
    # Enhanced validation with better constraint handling
    def validate_solution(circles):
        """Ensure final solution meets all constraints with improved robustness"""
        # First check constraints
        if check_constraints(circles):
            return circles
            
        # If constraints violated, do more thorough validation
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Ensure no overlaps with iterative correction
        max_iter = 100
        for _ in range(max_iter):
            changed = False
            # Use spatial indexing for efficient neighbor lookup
            tree = cKDTree(positions)
            
            # Check all pairs efficiently
            pairs = tree.query_pairs(2 * np.max(radii), output_type='ndarray')
            
            for i, j in pairs:
                if i < j:  # Only check each pair once
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (radii[i] + radii[j])**2
                    
                    if dist_sq < min_dist_sq:
                        # Adjust radii to prevent overlap
                        overlap = min_dist_sq - dist_sq
                        # Reduce both radii proportionally with better handling
                        reduction = overlap * 0.5
                        radii[i] = max(0.001, radii[i] - reduction * 0.5)
                        radii[j] = max(0.001, radii[j] - reduction * 0.5)
                        changed = True
            
            # Boundary corrections
            for i in range(len(circles)):
                # Correct x boundaries
                if positions[i, 0] - radii[i] < 0:
                    positions[i, 0] = radii[i]
                if positions[i, 0] + radii[i] > 1:
                    positions[i, 0] = 1 - radii[i]
                
                # Correct y boundaries  
                if positions[i, 1] - radii[i] < 0:
                    positions[i, 1] = radii[i]
                if positions[i, 1] + radii[i] > 1:
                    positions[i, 1] = 1 - radii[i]
                    
            if not changed:
                break
        
        return np.column_stack([positions, radii])
    
    # Multi-start optimization approach for better results
    def multi_start_optimization():
        best_circles = None
        best_sum = 0
        
        # Try multiple random initializations with different strategies
        for start in range(8):  # Increase number of starts for better exploration
            try:
                # Initialize with enhanced strategies
                circles = initialize_enhanced()
                
                # Apply advanced optimization
                circles = advanced_optimization(circles)
                
                # Validate and compute sum
                circles = validate_solution(circles)
                current_sum = np.sum(circles[:, 2])
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
            except Exception:
                continue
        
        # If no good solution found, return the best we have
        if best_circles is not None:
            return best_circles
        else:
            # Fallback to basic initialization
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [0.5, 0.5, 0.02]
            return circles
    
    # Main optimization process
    circles = multi_start_optimization()
    return circles


# EVOLVE-BLOCK-END
