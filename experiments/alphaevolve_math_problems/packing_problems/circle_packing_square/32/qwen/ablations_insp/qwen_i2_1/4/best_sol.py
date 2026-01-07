# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
import random
import time
from typing import Tuple

# Global constants for the optimization
MAX_ITER = 1000
TOLERANCE = 1e-6
SEED = 42

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining advanced initialization, physics relaxation, and simulated annealing.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(SEED)
    random.seed(SEED)
    
    n = 32
    
    # Multi-strategy initialization with enhanced techniques
    def initialize_multiple_strategies():
        best_circles = None
        best_sum = 0
        
        # Strategy 1: Optimized hexagonal grid with tighter packing
        def hexagonal_grid_initial():
            circles = []
            # Use 6x6 grid for 36 positions, then truncate to 32
            rows = 6
            cols = 6
            
            # Even better hexagonal packing parameters - more efficient
            spacing_x = 0.15
            spacing_y = spacing_x * 0.866  # sqrt(3)/2 * spacing_x for tight hex pack
            offset_x = 0.05
            offset_y = 0.05
            
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= n:
                        break
                    x = offset_x + j * spacing_x
                    y = offset_y + i * spacing_y
                    if i % 2 == 1:
                        x += spacing_x / 2
                    # Ensure within bounds
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        circles.append([x, y, 0.03])
                if len(circles) >= n:
                    break
            
            # Fill remaining with carefully placed random points
            while len(circles) < n:
                # Place near edges for better packing utilization
                x = 0.1 + np.random.random() * 0.8
                y = 0.1 + np.random.random() * 0.8
                circles.append([x, y, 0.03])
            
            return np.array(circles[:n])
        
        # Strategy 2: Improved grid-based initialization
        def grid_initial():
            circles = []
            # Better grid size calculation
            grid_size = int(np.ceil(np.sqrt(n)))
            # Use slightly smaller spacing for better coverage
            x_coords = np.linspace(0.05, 0.95, grid_size)
            y_coords = np.linspace(0.05, 0.95, grid_size)
            xx, yy = np.meshgrid(x_coords, y_coords)
            points = np.column_stack([xx.ravel(), yy.ravel()])[:n]
            
            for i, (x, y) in enumerate(points):
                circles.append([x, y, 0.03])
            
            return np.array(circles)
        
        # Strategy 3: Spiral initialization with better distribution
        def spiral_initial():
            circles = []
            # Better spiral parameters
            angle_step = 0.7
            radius_step = 0.04
            max_radius = 0.4
            
            # Place points in spiral pattern
            for i in range(n):
                angle = i * angle_step
                radius = min(i * radius_step, max_radius)
                x = 0.5 + radius * np.cos(angle)
                y = 0.5 + radius * np.sin(angle)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    circles.append([x, y, 0.03])
                else:
                    # Fall back to random if out of bounds
                    x = 0.5 + (np.random.random() - 0.5) * 0.6
                    y = 0.5 + (np.random.random() - 0.5) * 0.6
                    circles.append([x, y, 0.03])
            
            return np.array(circles)
        
        # Strategy 4: Voronoi-inspired initialization
        def voronoi_initial():
            circles = []
            # Start with a regular grid of points
            grid_points = []
            grid_size = 6
            spacing = 1.0 / (grid_size + 1)
            
            for i in range(1, grid_size + 1):
                for j in range(1, grid_size + 1):
                    x = i * spacing
                    y = j * spacing
                    grid_points.append([x, y])
            
            # Shuffle and take first n points
            random.shuffle(grid_points)
            selected_points = grid_points[:n]
            
            for i, (x, y) in enumerate(selected_points):
                circles.append([x, y, 0.03])
            
            return np.array(circles)
        
        # Try all strategies
        strategies = [hexagonal_grid_initial, grid_initial, spiral_initial, voronoi_initial]
        
        for strategy in strategies:
            try:
                circles = strategy()
                # Quick optimization with small steps
                circles = optimize_local(circles, max_iter=50)
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
            except Exception as e:
                continue
        
        # Return best configuration found or default
        if best_circles is not None:
            return best_circles
        else:
            # Fallback to simple initialization with better starting values
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [0.5, 0.5, 0.02]
            return circles
    
    # Enhanced local optimization helper function with better constraints
    def optimize_local(circles, max_iter=100):
        """Perform quick local optimization on circle positions/radii"""
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
                # Stronger boundary constraints with safety margins
                constraints.extend([
                    x - r - 1e-6,      # x - r >= 0 (with margin)
                    1 - x - r - 1e-6,  # 1 - x - r >= 0 (with margin)
                    y - r - 1e-6,      # y - r >= 0 (with margin)
                    1 - y - r - 1e-6   # 1 - y - r >= 0 (with margin)
                ])
            return np.array(constraints)
        
        def constraint_nonoverlap(params):
            positions_and_radii = params.reshape(-1, 3)
            constraints = []
            # More efficient vectorized computation
            centers = positions_and_radii[:, :2]
            radii = positions_and_radii[:, 2]
            
            # Compute all pairwise distances once
            distances = cdist(centers, centers)
            
            # Add constraints for all pairs (only upper triangle to avoid duplication)
            for i in range(n):
                for j in range(i+1, n):
                    distance = distances[i, j]
                    r1, r2 = radii[i], radii[j]
                    # Add safety margin to prevent numerical issues
                    constraints.append(distance - (r1 + r2) - 1e-6)
            
            return np.array(constraints)
        
        # Set bounds - tighter bounds for better convergence
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        try:
            # Try multiple optimization methods for better robustness
            methods = ['SLSQP', 'L-BFGS-B']
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
                        options={'maxiter': max_iter, 'ftol': 1e-6, 'eps': 1e-4, 'disp': False}
                    )
                    
                    if result.success:
                        optimized = result.x.reshape(-1, 3)
                        # Validate the result
                        if validate_solution(optimized):
                            return optimized
                except:
                    continue
                    
        except Exception as e:
            pass
        
        return circles
    
    # Physics-based relaxation with enhanced force model and better convergence
    def physics_relaxation(positions: np.ndarray, radii: np.ndarray, iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Apply physics-based relaxation to improve packing with better force modeling"""
        n = len(positions)
        
        # Use more sophisticated force model with better convergence properties
        for iter_num in range(iterations):
            forces = np.zeros_like(positions)
            
            # Compute repulsive forces between circles
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx*dx + dy*dy
                    
                    if dist_sq < (radii[i] + radii[j])**2 and dist_sq > 1e-12:
                        dist = math.sqrt(dist_sq)
                        # Use softer repulsion force for better convergence
                        force_magnitude = max(0, (radii[i] + radii[j] - dist) / (dist + 1e-8))
                        # Add damping factor that decreases with iteration
                        damping = 0.9 * (1.0 - iter_num/iterations)
                        forces[i, 0] += damping * force_magnitude * dx / (dist + 1e-8)
                        forces[i, 1] += damping * force_magnitude * dy / (dist + 1e-8)
                        forces[j, 0] -= damping * force_magnitude * dx / (dist + 1e-8)
                        forces[j, 1] -= damping * force_magnitude * dy / (dist + 1e-8)
            
            # Apply forces with adaptive step size and better boundary handling
            for i in range(n):
                # Adaptive step size that decreases with iterations
                step_size = 0.01 * (1.0 - iter_num/iterations * 0.3)
                positions[i, 0] += step_size * forces[i, 0]
                positions[i, 1] += step_size * forces[i, 1]
                
                # Boundary constraints - keep circles within bounds with margin
                positions[i, 0] = np.clip(positions[i, 0], radii[i] + 1e-6, 1 - radii[i] - 1e-6)
                positions[i, 1] = np.clip(positions[i, 1], radii[i] + 1e-6, 1 - radii[i] - 1e-6)
        
        return positions, radii
    
    # Enhanced simulated annealing refinement with better cooling and acceptance
    def simulated_annealing_refinement(initial_circles: np.ndarray, max_time: float = 45.0) -> np.ndarray:
        """Refine solution using simulated annealing for better results"""
        start_time = time.time()
        current_circles = initial_circles.copy()
        best_circles = current_circles.copy()
        best_sum = np.sum(current_circles[:, 2])
        
        # Parameters tuned for better exploration and exploitation
        temperature = 1.0  # Higher initial temperature for more exploration
        cooling_rate = 0.9999  # Slightly slower cooling for better exploration
        min_temperature = 1e-8
        step_size = 0.02  # Larger step size for more aggressive exploration
        
        iteration = 0
        stuck_count = 0
        last_best_sum = best_sum
        
        while temperature > min_temperature and (time.time() - start_time) < max_time:
            # Create new candidate solution
            candidate_circles = current_circles.copy()
            
            # Randomly select a circle to modify
            idx = random.randint(0, len(candidate_circles)-1)
            choice = random.random()
            
            # Different modification types with better balance
            if choice < 0.4:  # Modify position (most frequent)
                candidate_circles[idx, 0] += random.uniform(-step_size, step_size)
                candidate_circles[idx, 1] += random.uniform(-step_size, step_size)
                # Keep within bounds
                candidate_circles[idx, 0] = np.clip(candidate_circles[idx, 0], 0.001, 0.999)
                candidate_circles[idx, 1] = np.clip(candidate_circles[idx, 1], 0.001, 0.999)
            elif choice < 0.7:  # Modify radius
                old_r = candidate_circles[idx, 2]
                candidate_circles[idx, 2] += random.uniform(-step_size*0.5, step_size*0.5)
                candidate_circles[idx, 2] = max(0.001, min(0.499, candidate_circles[idx, 2]))
            elif choice < 0.9:  # Move both position and radius
                candidate_circles[idx, 0] += random.uniform(-step_size, step_size)
                candidate_circles[idx, 1] += random.uniform(-step_size, step_size)
                candidate_circles[idx, 2] += random.uniform(-step_size*0.5, step_size*0.5)
                # Keep within bounds
                candidate_circles[idx, 0] = np.clip(candidate_circles[idx, 0], 0.001, 0.999)
                candidate_circles[idx, 1] = np.clip(candidate_circles[idx, 1], 0.001, 0.999)
                candidate_circles[idx, 2] = max(0.001, min(0.499, candidate_circles[idx, 2]))
            else:  # Global move - shift entire cluster for exploration
                # Shift nearby circles to explore different configurations
                shift_amount = random.uniform(-step_size*0.8, step_size*0.8)
                for i in range(len(candidate_circles)):
                    if np.linalg.norm(candidate_circles[i, :2] - candidate_circles[idx, :2]) < 0.2:
                        candidate_circles[i, 0] += shift_amount
                        candidate_circles[i, 1] += shift_amount
                        candidate_circles[i, 0] = np.clip(candidate_circles[i, 0], 0.001, 0.999)
                        candidate_circles[i, 1] = np.clip(candidate_circles[i, 1], 0.001, 0.999)
            
            # Check constraints and accept/reject
            if validate_solution(candidate_circles):
                candidate_sum = np.sum(candidate_circles[:, 2])
                
                # Improved acceptance criterion with better handling of small differences
                delta = candidate_sum - best_sum
                if delta > 0:
                    # Always accept improvements
                    current_circles = candidate_circles
                    if candidate_sum > best_sum:
                        best_circles = candidate_circles.copy()
                        best_sum = candidate_sum
                        stuck_count = 0  # Reset stuck counter
                else:
                    # Accept with probability based on temperature and magnitude of change
                    if temperature > 1e-6:  # Only apply probabilistic acceptance at higher temps
                        prob = np.exp(delta / (temperature * 0.05))  # Scale temperature effect
                        if random.random() < prob:
                            current_circles = candidate_circles
                    else:
                        # At very low temperatures, only accept improvements
                        pass
            
            # Adaptive cooling - detect stagnation and cool faster when needed
            if iteration % 20 == 0 and iteration > 0:
                current_sum = np.sum(current_circles[:, 2])
                if abs(current_sum - last_best_sum) < 1e-6:
                    stuck_count += 1
                    if stuck_count > 5:
                        # Cool faster when stuck
                        cooling_rate = min(cooling_rate * 0.999, 0.99999)
                else:
                    stuck_count = 0
                last_best_sum = current_sum
                
            temperature *= cooling_rate
            iteration += 1
        
        return best_circles
    
    # Optimized validation function with early termination
    def validate_solution(circles: np.ndarray) -> bool:
        """Validate that solution satisfies all constraints with early exit"""
        if len(circles) == 0:
            return False
            
        # Check containment with better margin - vectorized for speed
        centers = circles[:, :2]
        radii = circles[:, 2]
        
        # Vectorized containment check
        containment_check = (
            (centers[:, 0] - radii) >= 0) & \
            ((1 - centers[:, 0] - radii) >= 0) & \
            ((centers[:, 1] - radii) >= 0) & \
            ((1 - centers[:, 1] - radii) >= 0)
        
        if not np.all(containment_check):
            return False
        
        # Check overlaps more efficiently using cdist with early termination
        distances = cdist(centers, centers)
        
        # Check only upper triangle to avoid duplicates and early termination
        n = len(circles)
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] < radii[i] + radii[j] - 1e-8:
                    return False
        
        return True
    
    # Enhanced local improvement heuristic with better optimization
    def local_improvement(circles):
        """Apply aggressive local improvement heuristics"""
        n = len(circles)
        improved = True
        iterations = 0
        
        # Try multiple rounds of improvement
        while improved and iterations < 30:
            improved = False
            iterations += 1
            
            # Try to increase radii of circles that have room
            for i in range(n):
                # Save current state
                original_x, original_y, original_r = circles[i]
                
                # Try to increase radius significantly
                test_r = min(original_r * 1.05, 0.499)
                
                # Check if we can increase radius without violating constraints
                valid = True
                for j in range(n):
                    if i != j:
                        x1, y1, r1 = circles[j]
                        dist_sq = (original_x - x1)**2 + (original_y - y1)**2
                        if dist_sq < (test_r + r1)**2 - 1e-8:  # Safety margin
                            valid = False
                            break
                
                # Also check boundary constraints with margin
                if (valid and 
                    test_r <= original_x - 1e-6 and 
                    test_r <= original_y - 1e-6 and 
                    test_r <= 1-original_x - 1e-6 and 
                    test_r <= 1-original_y - 1e-6):
                    circles[i] = [original_x, original_y, test_r]
                    improved = True
        
        # Additional improvement: Try to slightly adjust positions to allow larger radii
        for i in range(n):
            original_x, original_y, original_r = circles[i]
            best_r = original_r
            
            # Try small position adjustments to see if we can increase radius
            for _ in range(5):  # Test a few small moves
                test_x = original_x + random.uniform(-0.005, 0.005)
                test_y = original_y + random.uniform(-0.005, 0.005)
                test_x = np.clip(test_x, original_r + 1e-6, 1 - original_r - 1e-6)
                test_y = np.clip(test_y, original_r + 1e-6, 1 - original_r - 1e-6)
                
                # Check if this adjustment allows larger radius
                test_r = min(original_r * 1.02, 0.499)
                valid = True
                for j in range(n):
                    if i != j:
                        x1, y1, r1 = circles[j]
                        dist_sq = (test_x - x1)**2 + (test_y - y1)**2
                        if dist_sq < (test_r + r1)**2 - 1e-8:
                            valid = False
                            break
                
                if valid and test_r > best_r:
                    best_r = test_r
                    original_x, original_y = test_x, test_y
            
            circles[i] = [original_x, original_y, best_r]
        
        return circles
    
    # Main optimization process
    # Initialize with multiple strategies
    circles = initialize_multiple_strategies()
    
    # Apply physics-based relaxation to get a better starting point
    positions = circles[:, :2]
    radii = circles[:, 2]
    relaxed_positions, relaxed_radii = physics_relaxation(positions, radii, 100)
    
    # Create refined initial solution
    refined_circles = np.column_stack([relaxed_positions, relaxed_radii])
    
    # Apply local improvement
    refined_circles = local_improvement(refined_circles)
    
    # Refine using simulated annealing
    circles = simulated_annealing_refinement(refined_circles, max_time=40.0)
    
    # Final optimization with scipy
    def advanced_objective(params):
        positions_and_radii = params.reshape(-1, 3)
        return -np.sum(positions_and_radii[:, 2])
    
    def advanced_containment(params):
        positions_and_radii = params.reshape(-1, 3)
        constraints = []
        for i in range(n):
            x, y, r = positions_and_radii[i]
            constraints.extend([
                x - r,      # x - r >= 0
                1 - x - r,  # 1 - x - r >= 0
                y - r,      # y - r >= 0
                1 - y - r   # 1 - y - r >= 0
            ])
        return np.array(constraints)
    
    def advanced_nonoverlap(params):
        positions_and_radii = params.reshape(-1, 3)
        constraints = []
        # Vectorized computation for efficiency
        centers = positions_and_radii[:, :2]
        radii = positions_and_radii[:, 2]
        distances = cdist(centers, centers)
        
        # Only consider upper triangle to avoid duplicates
        for i in range(n):
            for j in range(i+1, n):
                distance = distances[i, j]
                r1, r2 = radii[i], radii[j]
                constraints.append(distance - (r1 + r2))
        
        return np.array(constraints)
    
    # Set bounds
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    
    # Final optimization with better parameters
    initial_guess = circles.flatten()
    
    try:
        result = minimize(
            advanced_objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': lambda x: advanced_containment(x)},
                {'type': 'ineq', 'fun': lambda x: advanced_nonoverlap(x)}
            ],
            options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-4, 'disp': False}
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
            circles = optimized_circles
        else:
            # If optimization fails, try to fix any constraint violations
            pass
            
    except Exception as e:
        # If optimization fails, return the last valid solution
        pass
    
    # Final validation and recovery
    if not validate_solution(circles):
        # If still invalid, try to recover by re-initializing
        circles = initialize_multiple_strategies()
    
    # Final local improvement
    circles = local_improvement(circles)
    
    return circles


# EVOLVE-BLOCK-END
