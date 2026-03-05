# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
from typing import Tuple
import time
import warnings

# Use fixed seed for reproducibility
np.random.seed(42)

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining proven initialization and mathematical optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Test multiple aspect ratios to find optimal configuration
    # Based on successful inspirations, focus on ratios that work well for circle packing
    aspect_ratios = [(1.5, 0.5), (1.4, 0.6), (1.3, 0.7), (1.25, 0.75), (1.2, 0.8), 
                     (1.15, 0.85), (1.1, 0.9), (1.0, 1.0)]
    
    best_sum = 0
    best_circles = None
    
    # Test multiple aspect ratios to find the optimal one
    for ratio_width, ratio_height in aspect_ratios:
        # Calculate width and height based on perimeter = 4
        width = 2 * ratio_width / (ratio_width + ratio_height)  
        height = 2 * ratio_height / (ratio_width + ratio_height)
        
        # Initialize using hexagonal grid pattern (similar to successful inspirations)
        def initialize_hexagonal():
            circles = []
            # Estimate initial radius based on area
            total_area = width * height
            circle_area = total_area / 21 * 0.8  # Leave margin for packing
            radius_estimate = math.sqrt(circle_area / math.pi)
            
            # Create hexagonal grid pattern
            rows = max(3, int(math.sqrt(21)) + 1)
            cols = max(3, int(21 / rows) + 1)
            
            # Hexagonal spacing
            spacing_x = radius_estimate * 2.0
            spacing_y = radius_estimate * math.sqrt(3)
            
            # Fill grid with circles
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= 21:
                        break
                        
                    # Offset odd rows for hexagonal packing
                    x_offset = (i % 2) * spacing_x / 2
                    x = spacing_x * j + x_offset + radius_estimate
                    y = spacing_y * i + radius_estimate
                    
                    # Check bounds
                    if (radius_estimate <= x <= width - radius_estimate and 
                        radius_estimate <= y <= height - radius_estimate):
                        circles.append([x, y, radius_estimate])
                
                if len(circles) >= 21:
                    break
            
            # Fill remaining positions with random placement
            while len(circles) < 21:
                x = np.random.uniform(radius_estimate, width - radius_estimate)
                y = np.random.uniform(radius_estimate, height - radius_estimate)
                circles.append([x, y, radius_estimate])
                
            return np.array(circles)
        
        # Try optimization approach similar to inspiration programs
        try:
            # Use a simpler approach: start with good initialization and then optimize
            initial_circles = initialize_hexagonal()
            
            # Objective function: maximize sum of radii (minimize negative sum)
            def objective(params):
                # Reshape params to circles array [x1, y1, r1, x2, y2, r2, ...]
                reshaped = params.reshape(-1, 3)
                return -np.sum(reshaped[:, 2])  # Negative because we minimize
            
            # Constraint function for scipy optimization
            def constraint_func(params):
                # params: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
                circles = params.reshape(-1, 3)
                
                constraints = []
                
                # Boundary constraints: each circle must stay within rectangle
                for i in range(21):
                    x, y, r = circles[i]
                    # Circle must be fully within bounds
                    constraints.extend([
                        x - r,                    # left boundary
                        width - x - r,           # right boundary  
                        y - r,                   # bottom boundary
                        height - y - r           # top boundary
                    ])
                
                # Overlap constraints: distance between centers >= sum of radii
                for i in range(21):
                    for j in range(i+1, 21):
                        x1, y1, r1 = circles[i]
                        x2, y2, r2 = circles[j]
                        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        # Constraint: distance >= r1 + r2 (we negate for inequality constraint)
                        constraints.append(distance - (r1 + r2))
                
                return np.array(constraints)
            
            # Bounds for optimization: (lower_bound, upper_bound) for each parameter
            bounds = []
            for i in range(21):
                # x bounds
                bounds.append((0.001, width - 0.001))
                # y bounds  
                bounds.append((0.001, height - 0.001))
                # r bounds (not too large)
                bounds.append((0.001, min(width, height)/2))
            
            # Try optimization with SLSQP
            initial_params = initial_circles.flatten()
            
            # Use SLSQP method which works well with constraints
            from scipy.optimize import minimize
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                # Extract optimized circles
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = optimized_circles
                    
        except Exception as e:
            # If optimization fails, use initial configuration
            initial_circles = initialize_hexagonal()
            current_sum = np.sum(initial_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = initial_circles
    
    # If no good solution found, return a reasonable configuration
    if best_circles is None:
        # Fallback to basic hexagonal packing with some refinement
        width, height = 1.2, 0.8  # Standard aspect ratio from successful approaches
        circles = []
        radius_estimate = 0.12  # Rough estimate
        
        # Simple hexagonal arrangement
        rows = 5
        cols = 5
        spacing_x = radius_estimate * 2.0
        spacing_y = radius_estimate * math.sqrt(3)
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= 21:
                    break
                x_offset = (i % 2) * spacing_x / 2
                x = spacing_x * j + x_offset + radius_estimate
                y = spacing_y * i + radius_estimate
                if (radius_estimate <= x <= width - radius_estimate and 
                    radius_estimate <= y <= height - radius_estimate):
                    circles.append([x, y, radius_estimate])
        
        # Fill remaining positions
        while len(circles) < 21:
            x = np.random.uniform(radius_estimate, width - radius_estimate)
            y = np.random.uniform(radius_estimate, height - radius_estimate)
            circles.append([x, y, radius_estimate])
            
        best_circles = np.array(circles[:21])
    
    return best_circles


def _create_advanced_hexagonal_initialization(width: float, height: float, n: int) -> np.ndarray:
    """Create high-quality initial configuration using advanced hexagonal packing"""
    circles = np.zeros((n, 3))
    
    # Use a 5x5 grid with proper hexagonal offsets
    rows = 5
    cols = 5
    
    # Calculate spacing with margin
    cell_width = width / (cols + 1)
    cell_height = height / (rows + 1)
    
    # Maximum possible radius for tight packing
    max_radius = min(cell_width, cell_height) / 2.0
    
    # Create hexagonal pattern with center bias
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
                
            # Standard grid position
            x_center = (j + 0.5) * cell_width
            y_center = (i + 0.5) * cell_height
            
            # Apply hexagonal offset for odd rows
            if i % 2 == 1:
                x_center += cell_width / 2
                
            # Adjust to stay within bounds
            x = max(max_radius, min(width - max_radius, x_center))
            y = max(max_radius, min(height - max_radius, y_center))
            
            # Center-biased radius distribution for better packing
            row_dist = abs(i - rows/2)
            col_dist = abs(j - cols/2)
            distance_from_center = math.sqrt(row_dist**2 + col_dist**2)
            
            # Normalize distance and create bell curve distribution
            normalized_dist = 1.0 - min(distance_from_center / (rows/2), 1.0)
            # Use sigmoid for smoother transition
            radius_factor = 0.5 + 0.5 * (1.0 / (1.0 + math.exp(-normalized_dist * 5)))
            radius = max_radius * (0.6 + 0.4 * radius_factor)
            
            circles[idx] = [x, y, radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining circles with strategic placement near edges and corners
    if idx < n:
        for i in range(idx, n):
            # Strategic placement: corners and edges
            corner_positions = [
                (0.1 * width, 0.1 * height),           # bottom-left
                (0.9 * width, 0.1 * height),           # bottom-right
                (0.1 * width, 0.9 * height),           # top-left
                (0.9 * width, 0.9 * height),           # top-right
                (width/2, 0.1 * height),               # bottom-center
                (width/2, 0.9 * height),               # top-center
                (0.1 * width, height/2),               # left-center
                (0.9 * width, height/2),               # right-center
            ]
            
            # Distribute remaining circles among corner/edge positions
            pos_idx = i % len(corner_positions)
            x, y = corner_positions[pos_idx]
            
            # Add some randomization to avoid perfect symmetry
            x += np.random.uniform(-0.05 * width, 0.05 * width)
            y += np.random.uniform(-0.05 * height, 0.05 * height)
            
            # Ensure within bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))
            
            # Smaller radius for edge circles
            radius = max_radius * 0.25
            
            circles[i] = [x, y, radius]
    
    return circles


def _compute_collision_penalty(circles: np.ndarray, width: float, height: float) -> Tuple[float, bool]:
    """Compute penalty for collisions and boundary violations"""
    penalty = 0.0
    is_valid = True
    
    # Check boundary violations
    for x, y, r in circles:
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            penalty += 10000.0  # Heavy penalty for boundary violations
            is_valid = False
    
    if not is_valid:
        return penalty, False
    
    # Check overlaps using efficient approach
    try:
        # Use spatial indexing for efficiency
        from scipy.spatial import cKDTree
        positions = circles[:, :2]
        tree = cKDTree(positions)
        
        # Query pairs within twice the maximum radius to reduce unnecessary checks
        max_radius = np.max(circles[:, 2])
        pairs = tree.query_pairs(2 * max_radius)
        
        for i, j in pairs:
            if i != j:
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    # Penalty based on how much they overlap
                    overlap = (r1 + r2 - distance)
                    penalty += 1000.0 * overlap
    
    except Exception:
        # Fallback to brute-force if spatial indexing fails
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    overlap = (r1 + r2 - distance)
                    penalty += 1000.0 * overlap
    
    return penalty, is_valid


def _physics_simulation_optimization(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Improve packing using physics-inspired relaxation"""
    # Create a copy to avoid modifying the original
    optimized = initial_circles.copy()
    
    # Physics parameters
    repulsion_strength = 1000.0
    boundary_strength = 5000.0
    max_velocity = 0.02
    damping = 0.95
    
    # Physics simulation with multiple iterations
    for iteration in range(1000):
        # Calculate forces for each circle
        forces = np.zeros_like(optimized[:, :2])
        
        # Compute repulsive forces using spatial indexing
        try:
            from scipy.spatial import cKDTree
            positions = optimized[:, :2]
            tree = cKDTree(positions)
            
            # Find nearby circles efficiently
            max_radius = np.max(optimized[:, 2])
            # Only consider circles within a reasonable distance
            for i in range(len(optimized)):
                x, y, r = optimized[i]
                
                # Get nearby circles (within 3x radius)
                nearby_indices = tree.query_ball_point([x, y], 3 * max_radius)
                
                for j in nearby_indices:
                    if i == j:
                        continue
                        
                    other_x, other_y, other_r = optimized[j]
                    dx = x - other_x
                    dy = y - other_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0 and distance < r + other_r:
                        # Strong repulsion for actual collisions
                        force_magnitude = repulsion_strength / (distance * distance + 0.01)
                        forces[i, 0] += force_magnitude * dx / distance
                        forces[i, 1] += force_magnitude * dy / distance
                    elif distance > 0 and distance < 2*(r + other_r):
                        # Moderate repulsion for near misses
                        force_magnitude = 10.0 / (distance * distance + 0.01)
                        forces[i, 0] += force_magnitude * dx / distance
                        forces[i, 1] += force_magnitude * dy / distance
                        
        except Exception:
            # Fallback to brute-force if spatial indexing fails
            for i in range(len(optimized)):
                x, y, r = optimized[i]
                
                for j in range(len(optimized)):
                    if i == j:
                        continue
                        
                    other_x, other_y, other_r = optimized[j]
                    dx = x - other_x
                    dy = y - other_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0 and distance < r + other_r:
                        force_magnitude = repulsion_strength / (distance * distance + 0.01)
                        forces[i, 0] += force_magnitude * dx / distance
                        forces[i, 1] += force_magnitude * dy / distance
                    elif distance > 0 and distance < 2*(r + other_r):
                        force_magnitude = 10.0 / (distance * distance + 0.01)
                        forces[i, 0] += force_magnitude * dx / distance
                        forces[i, 1] += force_magnitude * dy / distance
        
        # Add boundary forces
        for i in range(len(optimized)):
            x, y, r = optimized[i]
            
            # Boundary forces
            boundary_force_x, boundary_force_y = 0.0, 0.0
            
            if x < r:
                boundary_force_x = boundary_strength * (r - x)
            elif x > width - r:
                boundary_force_x = boundary_strength * (width - r - x)
                
            if y < r:
                boundary_force_y = boundary_strength * (r - y)
            elif y > height - r:
                boundary_force_y = boundary_strength * (height - r - y)
            
            forces[i, 0] += boundary_force_x
            forces[i, 1] += boundary_force_y
        
        # Apply forces to update positions
        for i in range(len(optimized)):
            x, y, r = optimized[i]
            
            # Apply force with damping
            new_x = x + forces[i, 0] * 0.01
            new_y = y + forces[i, 1] * 0.01
            
            # Apply damping
            new_x = x + (new_x - x) * damping
            new_y = y + (new_y - y) * damping
            
            # Ensure within bounds
            new_x = max(r, min(width - r, new_x))
            new_y = max(r, min(height - r, new_y))
            
            # Store updated position
            optimized[i, 0] = new_x
            optimized[i, 1] = new_y
    
    # Local refinement with simulated annealing
    return _local_improvement(optimized, width, height)


def _local_improvement(circles: np.ndarray, width: float, height: float, 
                      max_iterations: int = 500) -> np.ndarray:
    """Perform local improvement using gradient-based approach"""
    # Create a copy to avoid modifying the original
    improved = circles.copy()
    
    # Simple local search with simulated annealing-inspired cooling
    temperature = 1.0
    cooling_rate = 0.995
    
    for iteration in range(max_iterations):
        # Save current state
        current_state = improved.copy()
        current_fitness = np.sum(current_state[:, 2]) - _compute_collision_penalty(current_state, width, height)[0]
        
        # Try to improve by moving one circle at a time
        for i in range(len(improved)):
            # Save original position
            original_x, original_y, original_r = improved[i]
            
            # Try small random moves
            move_x = np.random.uniform(-0.02, 0.02)
            move_y = np.random.uniform(-0.02, 0.02)
            
            # Apply move
            new_x = original_x + move_x
            new_y = original_y + move_y
            
            # Keep within bounds
            new_x = max(original_r, min(width - original_r, new_x))
            new_y = max(original_r, min(height - original_r, new_y))
            
            # Check if this improves the solution
            improved[i, 0] = new_x
            improved[i, 1] = new_y
            
            # Compute new fitness
            new_fitness = np.sum(improved[:, 2]) - _compute_collision_penalty(improved, width, height)[0]
            
            # Accept or reject based on temperature (simulated annealing)
            if new_fitness > current_fitness or np.random.random() < math.exp((new_fitness - current_fitness) / temperature):
                # Accept the move
                pass
            else:
                # Reject the move, restore original position
                improved[i, 0] = original_x
                improved[i, 1] = original_y
        
        # Cool down
        temperature *= cooling_rate
    
    return improved


def _validate_and_fix(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Ensure all circles are within bounds and non-overlapping"""
    fixed_circles = circles.copy()
    
    # Fix boundary violations
    for i in range(len(fixed_circles)):
        x, y, r = fixed_circles[i]
        x = max(r, min(width - r, x))
        y = max(r, min(height - r, y))
        fixed_circles[i] = [x, y, r]
    
    # Resolve overlaps with a more robust iterative approach
    max_iterations = 200
    for iteration in range(max_iterations):
        any_changes = False
        
        # Check all pairs for overlaps - more efficient approach
        # Only check close pairs to reduce computation
        try:
            from scipy.spatial import cKDTree
            positions = fixed_circles[:, :2]
            tree = cKDTree(positions)
            
            # Query pairs within twice the maximum radius
            max_radius = np.max(fixed_circles[:, 2])
            pairs = tree.query_pairs(2 * max_radius)
            
            for i, j in pairs:
                if i != j:
                    x1, y1, r1 = fixed_circles[i]
                    x2, y2, r2 = fixed_circles[j]
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < (r1 + r2):
                        # Move circles apart with more careful adjustment
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = max(0.001, distance)
                        
                        # Normalize direction vector
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Push circles apart - use a more conservative approach
                        push_distance = (r1 + r2 - dist) * 0.3
                        
                        # Apply only part of the push to prevent oscillation
                        fixed_circles[i][0] -= nx * push_distance * 0.5
                        fixed_circles[i][1] -= ny * push_distance * 0.5
                        fixed_circles[j][0] += nx * push_distance * 0.5
                        fixed_circles[j][1] += ny * push_distance * 0.5
                        
                        any_changes = True
                        
        except Exception:
            # Fallback to brute-force approach
            for i in range(len(fixed_circles)):
                for j in range(i+1, len(fixed_circles)):
                    x1, y1, r1 = fixed_circles[i]
                    x2, y2, r2 = fixed_circles[j]
                    
                    distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    if distance < (r1 + r2):
                        # Move circles apart with more careful adjustment
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = max(0.001, distance)
                        
                        # Normalize direction vector
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Push circles apart - use a more conservative approach
                        push_distance = (r1 + r2 - dist) * 0.3
                        
                        # Apply only part of the push to prevent oscillation
                        fixed_circles[i][0] -= nx * push_distance * 0.5
                        fixed_circles[i][1] -= ny * push_distance * 0.5
                        fixed_circles[j][0] += nx * push_distance * 0.5
                        fixed_circles[j][1] += ny * push_distance * 0.5
                        
                        any_changes = True
        
        if not any_changes:
            break
    
    # Final boundary correction
    for i in range(len(fixed_circles)):
        x, y, r = fixed_circles[i]
        x = max(r, min(width - r, x))
        y = max(r, min(height - r, y))
        fixed_circles[i] = [x, y, r]
        
    return fixed_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
