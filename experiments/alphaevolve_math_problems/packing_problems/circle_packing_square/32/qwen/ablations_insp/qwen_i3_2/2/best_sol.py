# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import math
import random
import time

# Global constants for the problem
N_CIRCLES = 32
UNIT_SQUARE_SIZE = 1.0
BENCHMARK = 2.937944526205518

def initialize_better_starting_config():
    """Initialize with a better starting configuration based on improved patterns"""
    circles = []
    
    # Create a more strategic initial layout using a combination of grid and corner placements
    # Grid-based approach with better distribution
    
    # Generate a more balanced initial grid
    rows, cols = 6, 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    positions = []
    radii = []
    
    # Generate grid points with slight perturbations
    for i in range(rows):
        for j in range(cols):
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            
            # Add small random perturbation to avoid perfect grid
            x += random.uniform(-0.005, 0.005)
            y += random.uniform(-0.005, 0.005)
            
            # Ensure within bounds
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            
            positions.append((x, y))
    
    # Assign initial radii based on proximity to edges and other circles
    for i in range(len(positions)):
        # Initial radius based on distance to edges
        x, y = positions[i]
        r = min(x, 1-x, y, 1-y) * 0.25  # Start with smaller radii to allow optimization
        # Make some circles larger in central areas
        if 0.2 < x < 0.8 and 0.2 < y < 0.8:
            r = min(r, 0.12)  # Allow larger central circles
        else:
            r = min(r, 0.08)   # Smaller edge circles
        radii.append(max(0.01, r))  # Ensure minimum radius
    
    # Ensure we have exactly 32 circles
    while len(positions) < 32:
        # Add more strategic placements near corners and edges
        extra_positions = [
            (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
            (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5),
            (0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7),
            (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
            (0.5, 0.3), (0.5, 0.7), (0.3, 0.5), (0.7, 0.5),
            (0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85),
            (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
            (0.4, 0.4), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6)
        ]
        
        for pos in extra_positions:
            if len(positions) >= 32:
                break
            x, y = pos
            x = np.clip(x + random.uniform(-0.01, 0.01), 0.01, 0.99)
            y = np.clip(y + random.uniform(-0.01, 0.01), 0.01, 0.99)
            positions.append((x, y))
            # Determine appropriate radius
            r = min(x, 1-x, y, 1-y) * 0.15
            radii.append(max(0.01, min(r, 0.1)))
    
    # Trim to exactly 32
    positions = positions[:32]
    radii = radii[:32]
    
    # Create final circles array
    for i, (pos, r) in enumerate(zip(positions, radii)):
        x, y = pos
        # Ensure circle fits within bounds
        r = min(r, x, 1-x, y, 1-y)
        circles.append([x, y, r])
    
    return np.array(circles)

def enforce_boundary_constraints(circles):
    """Ensure all circles are within the unit square and fix any violations"""
    circles_copy = circles.copy()
    for i in range(len(circles_copy)):
        x, y, r = circles_copy[i]
        # Ensure circle fits within bounds
        r = min(r, x, 1-x, y, 1-y)
        # Keep within valid range
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles_copy[i] = [x, y, r]
    return circles_copy

def calculate_constraint_violation_penalty(circles):
    """Calculate penalty for constraint violations with improved efficiency"""
    if len(circles) < 2:
        return 0.0
    
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    # Use KDTree for efficient neighbor searches
    tree = cKDTree(positions)
    
    # Find neighbors within a reasonable distance
    max_dist = 2.0  # We only care about nearby circles
    pairs = tree.query_pairs(max_dist)
    
    penalty = 0.0
    
    # Check boundary violations first
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Check if circle is outside bounds
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            # Penalize based on how much it violates boundaries
            violation = 0
            if x - r < 0:
                violation += abs(x - r)
            if x + r > 1:
                violation += abs(x + r - 1)
            if y - r < 0:
                violation += abs(y - r)
            if y + r > 1:
                violation += abs(y + r - 1)
            penalty += violation * 100000  # Stronger penalty for boundary violations
    
    # Check overlaps among neighboring circles
    for i, j in pairs:
        if i >= j:  # Only process each pair once
            continue
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        
        # Calculate actual distance
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        min_dist = r1 + r2
        
        if dist < min_dist:
            # Overlap penalty
            overlap = min_dist - dist
            penalty += overlap * 50000  # Very strong penalty
    
    return penalty

def physics_based_optimization(initial_circles, max_iterations=1000):
    """Use physics-inspired optimization with force-based repulsion"""
    circles = initial_circles.copy()
    
    # Physics parameters - tuned for better convergence
    dt = 0.005  # Reduced time step for stability
    repulsion_strength = 500.0  # Slightly reduced for smoother optimization
    boundary_strength = 5000.0  # Reduced for better balance
    max_velocity = 0.005  # Reduced velocity limit
    
    # Track improvement for early stopping
    last_sum_radii = -float('inf')
    
    for iteration in range(max_iterations):
        # Calculate forces
        forces = np.zeros_like(circles[:, :2])
        
        # Repulsion forces between overlapping circles
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Use KDTree for efficient neighbor search
        tree = cKDTree(positions)
        pairs = tree.query_pairs(2.0)  # Look for nearby circles
        
        for i, j in pairs:
            if i >= j:  # Only process each pair once
                continue
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            
            dx = x1 - x2
            dy = y1 - y2
            distance = np.sqrt(dx*dx + dy*dy)
            
            min_distance = r1 + r2
            
            if distance < min_distance:
                # Repulsion force when circles overlap
                force_magnitude = repulsion_strength * (min_distance - distance)
                if distance > 0:
                    fx = force_magnitude * dx / distance
                    fy = force_magnitude * dy / distance
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Boundary forces
        for i in range(len(circles)):
            x, y, r = circles[i]
            fx, fy = 0.0, 0.0
            
            # Left boundary
            if x - r < 0:
                fx += boundary_strength * (r - x)
            # Right boundary
            if x + r > 1:
                fx -= boundary_strength * (x + r - 1)
            # Bottom boundary
            if y - r < 0:
                fy += boundary_strength * (r - y)
            # Top boundary
            if y + r > 1:
                fy -= boundary_strength * (y + r - 1)
            
            forces[i, 0] += fx
            forces[i, 1] += fy
        
        # Update positions with velocity
        velocities = forces * dt
        # Limit velocity
        for i in range(len(velocities)):
            vel_norm = np.sqrt(velocities[i, 0]**2 + velocities[i, 1]**2)
            if vel_norm > max_velocity:
                velocities[i] *= max_velocity / vel_norm
        
        # Apply movement
        circles[:, 0] += velocities[:, 0]
        circles[:, 1] += velocities[:, 1]
        
        # Enforce boundary constraints after each step
        circles = enforce_boundary_constraints(circles)
        
        # Early stopping based on improvement
        if iteration % 100 == 0:
            current_sum_radii = np.sum(circles[:, 2])
            if iteration > 0 and abs(current_sum_radii - last_sum_radii) < 1e-6:
                break
            last_sum_radii = current_sum_radii
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining smart initialization, physics-based optimization, and 
    mathematical optimization for refinement.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize with a better configuration
    initial_circles = initialize_better_starting_config()
    
    # Apply physics-based optimization for initial refinement with more iterations
    circles = physics_based_optimization(initial_circles, max_iterations=1000)
    
    # Refine with scipy optimization for fine-tuning
    # Extract positions and radii for optimization
    initial_positions = circles[:, :2]
    initial_radii = circles[:, 2]
    
    # Flatten for optimization
    initial_params = np.concatenate([initial_positions.flatten(), initial_radii])
    
    # Set bounds for optimization
    bounds = []
    # Position bounds [0,1] for both x and y
    for _ in range(N_CIRCLES):
        bounds.extend([(0, 1), (0, 1)])
    # Radius bounds [0, 0.5] (maximum possible radius for a single circle)
    for _ in range(N_CIRCLES):
        bounds.append((0, 0.5))
    
    # Create penalty-based objective function for scipy optimization
    def penalized_objective(params):
        positions = params[:2*N_CIRCLES].reshape(-1, 2)
        radii = params[2*N_CIRCLES:]
        
        # Objective: maximize sum of radii
        obj_value = -np.sum(radii)
        
        # Penalty for constraint violations
        penalty = 0
        
        # Containment penalties
        for i in range(N_CIRCLES):
            x, y = positions[i]
            r = radii[i]
            if x - r < 0:
                penalty += 1000 * (r - x)**2
            if x + r > 1:
                penalty += 1000 * (x + r - 1)**2
            if y - r < 0:
                penalty += 1000 * (r - y)**2
            if y + r > 1:
                penalty += 1000 * (y + r - 1)**2
        
        # Overlap penalties - use more efficient approach
        try:
            # Compute all pairwise distances efficiently
            distances = cdist(positions, positions)
            for i in range(N_CIRCLES):
                for j in range(i+1, N_CIRCLES):
                    min_distance = radii[i] + radii[j]
                    actual_distance = distances[i, j]
                    if actual_distance < min_distance:
                        penalty += 1000 * (min_distance - actual_distance)**2
        except Exception:
            # Fallback for numerical issues
            pass
            
        return obj_value + penalty
    
    # Optimize using L-BFGS-B with more iterations for better convergence
    try:
        result = minimize(
            penalized_objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9}
        )
        optimized_params = result.x
    except Exception as e:
        # If optimization fails, return current solution
        optimized_params = initial_params
    
    # Extract final result
    final_positions = optimized_params[:2*N_CIRCLES].reshape(-1, 2)
    final_radii = optimized_params[2*N_CIRCLES:]
    
    # Create final circles array
    circles = np.column_stack([final_positions, final_radii])
    
    # Final boundary enforcement
    circles = enforce_boundary_constraints(circles)
    
    # Final refinement through iterative adjustment
    max_iter = 30
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        any_changed = False
        for i in range(N_CIRCLES):
            # Fix containment first
            x, y, r = circles[i]
            # Adjust radius to keep within bounds
            new_r = min(x, 1-x, y, 1-y)
            if new_r < r:
                circles[i] = [x, y, new_r]
                any_changed = True
                
        # Then fix overlaps
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                distance = math.sqrt(dx*dx + dy*dy)
                min_distance = circles[i][2] + circles[j][2]
                
                if distance < min_distance:
                    # Reduce radii to satisfy constraint
                    reduction = (min_distance - distance) / 2
                    if circles[i][2] > reduction and circles[j][2] > reduction:
                        circles[i][2] -= reduction
                        circles[j][2] -= reduction
                        any_changed = True
        if not any_changed:
            break
    
    return circles


# EVOLVE-BLOCK-END
