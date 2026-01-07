# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, KDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
from numba import jit
from itertools import combinations
import time
import math

@jit(nopython=True)
def compute_distance_matrix(positions):
    """Compute distance matrix efficiently"""
    n = positions.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

@jit(nopython=True)
def check_overlap_fast(positions, radii):
    """Fast overlap checking using numba"""
    n = positions.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            distance = np.sqrt(dx*dx + dy*dy)
            if distance < radii[i] + radii[j]:
                return False
    return True

def compute_voronoi_areas(positions, radii):
    """Estimate how much area each circle can potentially grow without overlap"""
    n = len(positions)
    areas = np.zeros(n)
    vor = Voronoi(positions)
    
    # For each circle, estimate available area
    for i in range(n):
        # Find neighboring cells and their areas
        areas[i] = min(0.5, 0.5 * (1.0 - radii[i]) ** 2)  # Conservative estimate
    
    return areas

def create_hexagonal_initialization(n):
    """Create a high-quality hexagonal initial configuration"""
    # Create a hexagonal grid pattern
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Ensure we have enough points
    while rows * cols < n:
        rows += 1
    
    centers = []
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    # Hexagonal pattern with alternating rows
    for i in range(rows):
        for j in range(cols):
            if len(centers) >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            # Offset odd rows
            if i % 2 == 1:
                x += spacing_x * 0.5
            centers.append([x, y])
    
    # If we don't have enough, fill with random points
    while len(centers) < n:
        x = 0.05 + np.random.random() * 0.9
        y = 0.05 + np.random.random() * 0.9
        centers.append([x, y])
    
    return np.array(centers[:n])

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a multi-phase approach with sophisticated initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    best_sum = 0
    best_circles = None
    
    # Use a more targeted approach inspired by known good configurations
    # Strategy: Start with a known good configuration and optimize from there
    
    # Phase 1: Enhanced initialization with better patterns
    # Create a more sophisticated initial pattern based on circle packing research
    
    # Create a refined hexagonal pattern with more careful spacing
    def create_better_hex_pattern():
        # Try to create a pattern that closely resembles optimal circle packings
        # We'll use a 6x6 grid with proper hexagonal offsets
        rows = 6
        cols = 6
        centers = []
        
        # Hexagonal spacing
        spacing_x = 0.9 / cols
        spacing_y = 0.9 / rows
        
        for i in range(rows):
            for j in range(cols):
                if len(centers) >= n:
                    break
                x = 0.05 + (j + 0.5) * spacing_x
                y = 0.05 + (i + 0.5) * spacing_y
                # Offset odd rows
                if i % 2 == 1:
                    x += spacing_x * 0.5
                centers.append([x, y])
        
        # Fill remaining spots with random
        while len(centers) < n:
            x = 0.05 + np.random.random() * 0.9
            y = 0.05 + np.random.random() * 0.9
            centers.append([x, y])
            
        return np.array(centers[:n])
    
    # Strategy 1: Better hexagonal pattern
    centers_hex = create_better_hex_pattern()
    initial_strategies = [('hexagon', centers_hex)]
    
    # Strategy 2: Random with spatial distribution
    np.random.seed(42)
    centers_random = np.random.rand(n, 2) * 0.9 + 0.05
    initial_strategies.append(('random', centers_random))
    
    # Strategy 3: Grid pattern with better spacing
    grid_size = 6
    centers_grid = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(centers_grid) >= n:
                break
            x = 0.1 + (j + 0.5) * 0.8 / grid_size
            y = 0.1 + (i + 0.5) * 0.8 / grid_size
            centers_grid.append([x, y])
    if len(centers_grid) < n:
        for i in range(n - len(centers_grid)):
            x = 0.05 + np.random.random() * 0.9
            y = 0.05 + np.random.random() * 0.9
            centers_grid.append([x, y])
    centers_grid = np.array(centers_grid[:n])
    initial_strategies.append(('grid', centers_grid))
    
    # Phase 2: Advanced optimization approach with better parameter tuning
    max_attempts = 20  # More attempts for better chance of finding good solution
    
    for strategy_name, initial_centers in initial_strategies:
        for attempt in range(max_attempts // len(initial_strategies)):
            # Different seeds for variety
            np.random.seed(1000 + attempt + hash(strategy_name) % 1000)
            
            # Create initial configuration
            centers = initial_centers.copy()
            
            # Add more substantial perturbation to break symmetry
            perturbation_magnitude = 0.03
            centers += np.random.uniform(-perturbation_magnitude, perturbation_magnitude, centers.shape)
            centers[:, 0] = np.clip(centers[:, 0], 0.01, 0.99)
            centers[:, 1] = np.clip(centers[:, 1], 0.01, 0.99)
            
            # Estimate initial radii based on density - but start conservatively
            total_area = 1.0
            avg_area_per_circle = total_area / n
            avg_radius = np.sqrt(avg_area_per_circle / np.pi) * 0.7  # Slightly smaller
            
            radii = np.full(n, avg_radius)
            
            # More aggressive overlap resolution
            for _ in range(50):  # Fewer iterations but more aggressive
                # Compute distance matrix
                positions = np.column_stack([centers, radii])
                distances = compute_distance_matrix(centers)
                
                # Try to reduce radii to eliminate overlaps
                any_reduced = False
                for i in range(n):
                    for j in range(i+1, n):
                        if distances[i, j] < radii[i] + radii[j]:
                            # Reduce both radii significantly
                            reduction = (radii[i] + radii[j] - distances[i, j]) * 0.7 + 0.001
                            radii[i] = max(0.001, radii[i] - reduction)
                            radii[j] = max(0.001, radii[j] - reduction)
                            any_reduced = True
                
                if not any_reduced:
                    break
            
            # Validate and fix any remaining issues
            if not check_overlap_fast(centers, radii):
                # Reset with very conservative values
                radii = np.full(n, 0.015)
            
            # Create variables vector
            initial_vars = np.column_stack([centers, radii]).flatten()
            
            # Define objective function (minimize negative sum of radii)
            def objective(vars):
                radii = vars[2::3]
                return -np.sum(radii)
            
            # Constraint functions - more efficient implementation
            def non_overlap_constraints(vars):
                positions = vars.reshape(-1, 3)[:, :2]
                radii = vars.reshape(-1, 3)[:, 2]
                
                # Compute all pairwise distances
                distances = compute_distance_matrix(positions)
                
                # Generate constraint values: distance - (r_i + r_j) >= 0
                constraints = []
                for i in range(n):
                    for j in range(i+1, n):
                        constraint_val = distances[i, j] - radii[i] - radii[j]
                        constraints.append(constraint_val)
                
                return np.array(constraints)
            
            def containment_constraints(vars):
                positions = vars.reshape(-1, 3)[:, :2]
                radii = vars.reshape(-1, 3)[:, 2]
                
                constraints = []
                for i in range(n):
                    # x - r >= 0
                    constraints.append(positions[i, 0] - radii[i])
                    # 1 - x - r >= 0  
                    constraints.append(1 - positions[i, 0] - radii[i])
                    # y - r >= 0
                    constraints.append(positions[i, 1] - radii[i])
                    # 1 - y - r >= 0
                    constraints.append(1 - positions[i, 1] - radii[i])
                
                return np.array(constraints)
            
            # Set up bounds with tighter ranges
            bounds = []
            for i in range(n):
                # Bounds for x coordinate
                bounds.append((0.001, 0.999))
                # Bounds for y coordinate  
                bounds.append((0.001, 0.999))
                # Bounds for radius - more realistic upper bound
                bounds.append((0.001, 0.45))
            
            # Define constraints
            cons = [
                {'type': 'ineq', 'fun': lambda x: non_overlap_constraints(x)},
                {'type': 'ineq', 'fun': lambda x: containment_constraints(x)}
            ]
            
            # Try multiple optimization methods with aggressive parameters
            methods_to_try = [
                ('trust-constr', {'maxiter': 300, 'ftol': 1e-7, 'gtol': 1e-7}),
                ('SLSQP', {'maxiter': 300, 'ftol': 1e-7, 'gtol': 1e-7}),
                ('L-BFGS-B', {'maxiter': 300, 'ftol': 1e-7, 'gtol': 1e-7})
            ]
            
            result = None
            for method, options in methods_to_try:
                try:
                    result = minimize(
                        objective,
                        initial_vars,
                        method=method,
                        bounds=bounds,
                        constraints=cons,
                        options=options,
                        tol=1e-7
                    )
                    
                    if result.success:
                        break
                except Exception as e:
                    continue
            
            # Process result
            if result is not None and result.success:
                final_vars = result.x
                circles = final_vars.reshape(-1, 3)
                current_sum = np.sum(circles[:, 2])
                
                # Update best solution if this one is better
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
            else:
                # Fallback to initial configuration if optimization failed
                circles = np.column_stack([centers, radii])
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
    
    # Phase 3: Enhanced local refinement with multiple strategies
    if best_circles is not None:
        # Apply multiple refinement passes
        refined_circles = best_circles.copy()
        
        # First pass: aggressive radius increases
        for iteration in range(5):
            improved = False
            temp_circles = refined_circles.copy()
            
            # Try to increase each radius as much as possible
            for i in range(n):
                # Get current configuration
                positions = temp_circles[:, :2]
                radii = temp_circles[:, 2]
                
                # Current radius
                current_radius = radii[i]
                
                # Try to increase radius - larger increments now
                max_increase = 0.02
                test_radius = min(current_radius + max_increase, 0.45)
                
                # Check if we can actually increase the radius
                valid = True
                for j in range(n):
                    if i != j:
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        if distance < test_radius + radii[j]:
                            valid = False
                            break
                
                # Check containment
                if valid:
                    if (positions[i, 0] - test_radius < 0 or 
                        positions[i, 0] + test_radius > 1 or
                        positions[i, 1] - test_radius < 0 or
                        positions[i, 1] + test_radius > 1):
                        valid = False
                
                if valid:
                    temp_circles[i, 2] = test_radius
                    improved = True
            
            if not improved:
                break
            
            refined_circles = temp_circles
        
        # Second pass: fine-grained optimization
        # Try small adjustments to positions to help with overlapping
        temp_circles = refined_circles.copy()
        for i in range(n):
            # Try small position adjustments to improve spacing
            old_pos = temp_circles[i, :2].copy()
            old_radius = temp_circles[i, 2]
            
            # Try moving in directions that might improve packing
            best_pos = old_pos.copy()
            best_radius = old_radius
            best_score = np.sum(temp_circles[:, 2])  # current sum
            
            # Test several small moves
            moves = [(0.005, 0), (-0.005, 0), (0, 0.005), (0, -0.005),
                     (0.003, 0.003), (-0.003, 0.003), (0.003, -0.003), (-0.003, -0.003)]
            
            for dx, dy in moves:
                new_x = old_pos[0] + dx
                new_y = old_pos[1] + dy
                
                # Check if move is valid
                if 0.001 <= new_x <= 0.999 and 0.001 <= new_y <= 0.999:
                    # Test if this improves the configuration
                    temp_circles[i, 0] = new_x
                    temp_circles[i, 1] = new_y
                    
                    # Check validity
                    valid = True
                    for j in range(n):
                        if i != j:
                            dx_check = temp_circles[i, 0] - temp_circles[j, 0]
                            dy_check = temp_circles[i, 1] - temp_circles[j, 1]
                            distance = np.sqrt(dx_check*dx_check + dy_check*dy_check)
                            if distance < temp_circles[i, 2] + temp_circles[j, 2]:
                                valid = False
                                break
                    
                    if valid:
                        new_sum = np.sum(temp_circles[:, 2])
                        if new_sum > best_score:
                            best_score = new_sum
                            best_pos = [new_x, new_y]
                            best_radius = temp_circles[i, 2]
                    
                    # Restore
                    temp_circles[i, 0] = old_pos[0]
                    temp_circles[i, 1] = old_pos[1]
            
            # Apply best move if it improves things
            if (best_pos[0] != old_pos[0] or best_pos[1] != old_pos[1] or 
                best_radius != old_radius):
                temp_circles[i, 0] = best_pos[0]
                temp_circles[i, 1] = best_pos[1]
                temp_circles[i, 2] = best_radius
        
        refined_circles = temp_circles
        
        # Final check of the refined solution
        final_sum = np.sum(refined_circles[:, 2])
        if final_sum > best_sum:
            best_sum = final_sum
            best_circles = refined_circles
    
    # If still no solution found, return a decent fallback
    if best_circles is None:
        centers = np.random.rand(n, 2) * 0.9 + 0.05
        radii = np.full(n, 0.02)
        best_circles = np.column_stack([centers, radii])
    
    return best_circles


# EVOLVE-BLOCK-END
