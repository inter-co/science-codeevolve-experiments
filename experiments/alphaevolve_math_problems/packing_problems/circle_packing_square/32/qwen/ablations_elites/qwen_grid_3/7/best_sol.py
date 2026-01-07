# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import time
from itertools import combinations
import random
import math

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, physics-inspired optimization,
    and multi-start strategies to beat the benchmark.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Multi-start approach: try several different initializations
    best_circles = None
    best_sum = -np.inf
    
    # Strategy 1: Hexagonal grid initialization (like inspiration 1)
    circles1 = generate_hexagonal_initialization()
    
    # Strategy 2: Golden ratio distribution with better radius control
    circles2 = generate_golden_ratio_initialization()
    
    # Strategy 3: Improved grid-based initialization
    circles3 = generate_improved_grid_initialization()
    
    # Strategy 4: Physics-inspired force-based initialization
    circles4 = generate_force_based_initialization()
    
    # Strategy 5: Random with better constraint checking
    circles5 = generate_better_randomized_initialization()
    
    initial_strategies = [circles1, circles2, circles3, circles4, circles5]
    
    # Try all strategies with more aggressive optimization
    for i, initial_circles in enumerate(initial_strategies):
        try:
            # Apply optimization with more aggressive settings
            optimized_circles = optimize_packaging(initial_circles, maxiter=2500)
            
            # Validate and refine
            validated_circles = validate_and_refine(optimized_circles)
            
            # Evaluate
            current_sum = np.sum(validated_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = validated_circles.copy()
                
        except Exception as e:
            continue
    
    # If we still don't have a good solution, do final refinement with enhanced physics
    if best_circles is None:
        # Fallback to the best of the basic strategies
        fallback_strategy = generate_hexagonal_initialization()
        fallback_strategy = optimize_packaging(fallback_strategy, maxiter=1500)
        fallback_strategy = validate_and_refine(fallback_strategy)
        best_circles = fallback_strategy
    
    # Final enhancement with force-based optimization and local search
    if best_sum < 2.93:
        best_circles = force_based_optimization(best_circles, max_iter=150)
        best_circles = enhanced_local_search(best_circles)
    
    return best_circles

def generate_hexagonal_initialization():
    """Generate initial configuration using hexagonal packing pattern like inspiration 1"""
    n = 32
    circles = np.zeros((n, 3))
    
    # Create a hexagonal grid pattern
    rows = 6
    cols = 6
    
    # Generate positions in hexagonal pattern
    y_positions = np.linspace(0.1, 0.9, rows)
    x_positions = np.linspace(0.1, 0.9, cols)
    
    # Hexagonal offset
    x_offset = 0.0
    idx = 0
    
    for i, y in enumerate(y_positions):
        if i % 2 == 0:
            x_offset = 0.0
        else:
            x_offset = 0.5 * (x_positions[1] - x_positions[0]) if len(x_positions) > 1 else 0.0
        
        for j, x in enumerate(x_positions):
            if idx >= n:
                break
            x_pos = x + x_offset
            if x_pos <= 0.9 and x_pos >= 0.1:
                circles[idx] = [x_pos, y, 0.03]  # Start with moderate radius
                idx += 1
        if idx >= n:
            break
    
    # Fill remaining positions with random valid placements if needed
    for i in range(idx, n):
        while True:
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            # Check if position is valid (not too close to edges)
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                circles[i] = [x, y, 0.03]
                break
    
    return circles

def generate_improved_grid_initialization():
    """Generate initial configuration using improved grid-based approach"""
    circles = np.zeros((32, 3))
    
    # Create a 6x6 grid to distribute 32 circles more effectively
    rows, cols = 6, 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= 32:
                break
            # Position with better jitter for improved distribution
            x = (j + 1) * spacing_x + np.random.uniform(-0.15 * spacing_x, 0.15 * spacing_x)
            y = (i + 1) * spacing_y + np.random.uniform(-0.15 * spacing_y, 0.15 * spacing_y)
            
            # Initial radius - start with larger values to encourage better packing
            r = min(spacing_x, spacing_y) * (0.35 + 0.15 * np.random.random())
            
            # Ensure circle fits in square
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[count] = [x, y, r]
            count += 1
            
        if count >= 32:
            break
    
    return circles

def generate_golden_ratio_initialization():
    """Generate initial configuration using golden ratio distribution"""
    circles = np.zeros((32, 3))
    
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    for i in range(32):
        # Distribute points using golden ratio for better coverage
        x = ((i * phi) % 1) * 0.8 + 0.1  # Scale to [0.1, 0.9]
        y = (i / 32) * 0.8 + 0.1
        
        # Initial radius - start with more varied values
        r = 0.02 + 0.06 * np.random.random()
        
        # Ensure it fits in the square
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        
        circles[i] = [x, y, r]
    
    return circles

def generate_force_based_initialization():
    """Generate initialization inspired by physics force concepts"""
    circles = np.zeros((32, 3))
    
    # Start with a regular grid pattern
    positions = []
    for i in range(32):
        row = i // 6
        col = i % 6
        x = 0.1 + col * 0.14 + np.random.uniform(-0.02, 0.02)
        y = 0.1 + row * 0.14 + np.random.uniform(-0.02, 0.02)
        positions.append([x, y])
    
    # Initialize with radius based on proximity to others
    for i in range(32):
        x, y = positions[i]
        
        # Find closest neighbor to determine appropriate radius
        min_dist = float('inf')
        for j in range(32):
            if i != j:
                x2, y2 = positions[j]
                dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                min_dist = min(min_dist, dist)
        
        # Set radius based on neighborhood density
        r = min(0.1, min_dist * 0.25)
        r = max(0.01, r)
        
        # Ensure it fits in square
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        
        circles[i] = [x, y, r]
    
    return circles

def generate_better_randomized_initialization():
    """Generate randomized initialization with better overlap avoidance"""
    circles = np.zeros((32, 3))
    
    # Start with more evenly distributed points, prioritizing non-overlap
    for i in range(32):
        # Try to avoid overlaps by placing points carefully with more attempts
        max_attempts = 2000
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts:
            # Random position with bias towards spreading
            x = 0.05 + 0.9 * np.random.random()
            y = 0.05 + 0.9 * np.random.random()
            
            # Initial radius - start with larger values to allow better packing
            r = 0.01 + 0.05 * np.random.random()
            
            # Check if it fits within boundaries
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                # Check overlap with existing circles - more thorough check
                valid = True
                for j in range(i):
                    x_prev, y_prev, r_prev = circles[j]
                    distance = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
                    if distance < r + r_prev - 1e-8:  # Tighter overlap checking
                        valid = False
                        break
                
                if valid:
                    circles[i] = [x, y, r]
                    placed = True
            attempts += 1
        
        # If couldn't place properly, use fallback with different parameters
        if not placed:
            x = 0.1 + 0.8 * np.random.random()
            y = 0.1 + 0.8 * np.random.random()
            r = 0.02 + 0.03 * np.random.random()
            circles[i] = [x, y, r]
    
    return circles

def optimize_packaging(initial_circles, maxiter=1500):
    """Use advanced optimization to improve the packing with better settings"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_vars = []
    for i in range(n):
        x, y, r = initial_circles[i]
        initial_vars.extend([x, y, r])
    
    # Define objective function to maximize sum of radii
    def objective(vars_flat):
        # Extract variables
        circles = np.array(vars_flat).reshape(-1, 3)
        # We want to maximize sum of radii, so return negative
        return -np.sum(circles[:, 2])
    
    # Define constraints with better numerical stability
    def boundary_constraints(vars_flat):
        """Ensure all circles are within the unit square"""
        circles = np.array(vars_flat).reshape(-1, 3)
        constraints = []
        
        for i in range(n):
            x, y, r = circles[i]
            # Circle must fit entirely within square with margin for numerical stability
            constraints.append(x - r - 1e-6)      # x - r >= 1e-6
            constraints.append(1 - x - r - 1e-6)  # 1 - x - r >= 1e-6
            constraints.append(y - r - 1e-6)      # y - r >= 1e-6
            constraints.append(1 - y - r - 1e-6)  # 1 - y - r >= 1e-6
            
        return np.array(constraints)
    
    def overlap_constraints(vars_flat):
        """Ensure no two circles overlap with numerical tolerance"""
        circles = np.array(vars_flat).reshape(-1, 3)
        constraints = []
        
        for i, j in combinations(range(n), 2):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_distance_sq = (r1 + r2)**2
            
            # Add small tolerance to avoid numerical issues
            constraints.append(distance_sq - min_distance_sq - 1e-10)
            
        return np.array(constraints)
    
    # Set up bounds for variables with tighter ranges
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds
        bounds.append((0.001, 0.999))
        # r bounds - slightly smaller upper bound for safety
        bounds.append((0.001, 0.49))
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: boundary_constraints(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
    ]
    
    # Try multiple optimization methods for robustness
    methods_to_try = ['SLSQP', 'trust-constr']
    best_result = None
    best_sum = -np.inf
    
    for method in methods_to_try:
        try:
            result = minimize(objective, initial_vars, method=method, 
                             bounds=bounds, constraints=cons, 
                             options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if result.success:
                # Evaluate the result
                circles = result.x.reshape(-1, 3)
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception as e:
            continue
    
    # If any method succeeded, return the best result
    if best_result is not None:
        optimized_circles = best_result.x.reshape(-1, 3)
        return optimized_circles
    else:
        # If all optimization failed, return initial configuration
        return initial_circles

def validate_and_refine(circles):
    """Validate solution and perform final refinement with better overlap resolution"""
    # Ensure all constraints are satisfied
    n = len(circles)
    
    # Make sure no circles overlap - more thorough check
    valid = True
    for i, j in combinations(range(n), 2):
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if distance < r1 + r2 - 1e-8:  # Small tolerance for numerical errors
            valid = False
            break
    
    # If there are overlaps, perform a more sophisticated refinement
    if not valid:
        # Try a more aggressive refinement approach with physics-inspired movement
        for iteration in range(1000):  # More iterations for better refinement
            improved = False
            # Try to reduce overlaps by adjusting positions
            for i in range(n):
                x, y, r = circles[i]
                
                # Try to shrink radius if needed
                if r > 0.001:
                    # Check if reducing radius helps
                    new_r = max(0.001, r * 0.995)
                    # Check if this reduces overlaps with neighbors
                    valid_radius = True
                    for j in range(n):
                        if i != j:
                            x_j, y_j, r_j = circles[j]
                            distance = np.sqrt((x - x_j)**2 + (y - y_j)**2)
                            if distance < new_r + r_j:
                                valid_radius = False
                                break
                    if valid_radius:
                        circles[i, 2] = new_r
                        improved = True
                        
                # Try moving to reduce overlap - more systematic approach
                best_x, best_y = x, y
                best_r = r
                
                # Try several positions around current location with more granularity
                for dx in [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]:
                    for dy in [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]:
                        test_x = max(r, min(1-r, x + dx))
                        test_y = max(r, min(1-r, y + dy))
                        
                        # Check validity
                        valid_pos = True
                        for j in range(n):
                            if i != j:
                                x_j, y_j, r_j = circles[j]
                                distance = np.sqrt((test_x - x_j)**2 + (test_y - y_j)**2)
                                if distance < r + r_j:
                                    valid_pos = False
                                    break
                        if valid_pos:
                            # This position is valid, keep it
                            best_x, best_y = test_x, test_y
                            break
                    else:
                        continue
                    break
                
                if best_x != x or best_y != y:
                    circles[i, 0] = best_x
                    circles[i, 1] = best_y
                    improved = True
                    
            if not improved:
                break
    
    # Final cleanup - ensure all circles are valid
    for i in range(n):
        x, y, r = circles[i]
        # Ensure it fits within bounds
        circles[i, 0] = max(r, min(1-r, x))
        circles[i, 1] = max(r, min(1-r, y))
        circles[i, 2] = max(0.001, min(0.49, r))
    
    return circles

def force_based_optimization(circles, max_iter=100):
    """Apply force-based optimization to improve packing (like inspiration 1)"""
    n = len(circles)
    positions = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    
    # Parameters for force calculation
    k_repel = 1000.0
    k_containment = 1000.0
    dt = 0.001
    
    for iteration in range(max_iter):
        forces = np.zeros_like(positions)
        
        # Repulsion forces between circles
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                
                if dist > 0 and dist < (radii[i] + radii[j]):
                    # Repulsive force
                    force_magnitude = k_repel * (radii[i] + radii[j] - dist) / (dist + 1e-8)
                    forces[i, 0] += force_magnitude * dx / dist
                    forces[i, 1] += force_magnitude * dy / dist
                    forces[j, 0] -= force_magnitude * dx / dist
                    forces[j, 1] -= force_magnitude * dy / dist
        
        # Containment forces (push back into bounds)
        for i in range(n):
            # Push away from boundaries
            boundary_forces = np.array([
                max(0, radii[i] - positions[i, 0]),  # left boundary
                max(0, radii[i] - positions[i, 1]),  # bottom boundary
                max(0, positions[i, 0] + radii[i] - 1),  # right boundary
                max(0, positions[i, 1] + radii[i] - 1)   # top boundary
            ])
            
            forces[i, 0] += k_containment * boundary_forces[0] - k_containment * boundary_forces[2]
            forces[i, 1] += k_containment * boundary_forces[1] - k_containment * boundary_forces[3]
        
        # Update positions
        positions += dt * forces
        
        # Keep positions within bounds
        positions[:, 0] = np.clip(positions[:, 0], radii, 1-radii)
        positions[:, 1] = np.clip(positions[:, 1], radii, 1-radii)
    
    # Create updated circles array
    updated_circles = np.column_stack([positions, radii])
    return updated_circles

def enhanced_local_search(circles):
    """Enhanced local search to squeeze out additional improvement"""
    n = len(circles)
    current_circles = circles.copy()
    
    # Strategy: Try to increase radii systematically with better compromise logic
    improved = True
    iteration = 0
    max_iterations = 2000  # More iterations for better convergence
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try to improve each circle in a more systematic way
        for i in range(n):
            # Try to increase radius of circle i
            original_radius = current_circles[i, 2]
            test_radius = min(0.4, original_radius + 0.003)
            
            # Test if we can increase this radius
            test_circles = current_circles.copy()
            test_circles[i, 2] = test_radius
            
            # Check constraints
            valid, _ = check_constraints(test_circles)
            if valid:
                test_sum = np.sum(test_circles[:, 2])
                if test_sum > np.sum(current_circles[:, 2]):
                    current_circles = test_circles
                    improved = True
            else:
                # Try to make a compromise with neighbors
                # Try decreasing some nearby radii to make room
                for j in range(n):
                    if i != j and current_circles[j, 2] > 0.02:
                        test_circles = current_circles.copy()
                        test_circles[j, 2] = max(0.01, test_circles[j, 2] - 0.0015)
                        test_circles[i, 2] = test_radius
                        
                        valid, _ = check_constraints(test_circles)
                        if valid:
                            test_sum = np.sum(test_circles[:, 2])
                            if test_sum > np.sum(current_circles[:, 2]):
                                current_circles = test_circles
                                improved = True
                                break
    
    return current_circles

def check_constraints(circles):
    """Check if all circles satisfy constraints"""
    n = len(circles)
    violations = []
    
    # Check containment constraints
    x_coords = circles[:, 0]
    y_coords = circles[:, 1]
    radii = circles[:, 2]
    
    containment_violations = np.where((x_coords - radii < 0) | 
                                     (x_coords + radii > 1) | 
                                     (y_coords - radii < 0) | 
                                     (y_coords + radii > 1))[0]
    
    for i in containment_violations:
        violations.append(f"Circle {i} violates containment constraints")
    
    # Check overlap constraints
    if len(violations) == 0 and n > 1:
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Create mask for upper triangle (avoid double counting)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        distance_matrix = distances[mask]
        min_distance_matrix = (radii[:, None] + radii[None, :])[mask]
        
        # Find overlapping pairs - with small tolerance for numerical precision
        overlap_indices = np.where(distance_matrix < min_distance_matrix - 1e-10)[0]
        
        if len(overlap_indices) > 0:
            # Get the corresponding circle indices
            for idx in overlap_indices:
                i = np.triu_indices(n, k=1)[0][idx]
                j = np.triu_indices(n, k=1)[1][idx]
                violations.append(f"Circles {i} and {j} overlap")
    
    return len(violations) == 0, violations


# EVOLVE-BLOCK-END
