# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from itertools import combinations
from sklearn.cluster import KMeans
import time

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-stage approach with improved initialization and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 32
    circles = np.zeros((n, 3))
    
    # Stage 1: Try multiple initialization strategies with enhanced optimization
    best_result = None
    best_sum = 0
    
    # Try multiple initialization strategies with different approaches
    initial_strategies = [
        lambda: initialize_hybrid_approach(n),
        lambda: initialize_from_known_good_packings(n),
        lambda: initialize_hexagonal_lattice(n),
        lambda: initialize_random_with_radius_adjustment(n),
        lambda: initialize_grid_with_perturbation(n, 0),
        lambda: initialize_grid_with_perturbation(n, 1)
    ]
    
    # Also try a physics-inspired approach
    try:
        physics_result = initialize_physics_based(n)
        if physics_result is not None:
            initial_strategies.append(lambda: physics_result)
    except Exception:
        pass
    
    # Try multiple optimization attempts per strategy
    for i, init_func in enumerate(initial_strategies):
        try:
            # Get initial configuration
            initial_config = init_func()
            
            # Run multiple optimization attempts with different starting points
            for attempt in range(3):  # Try 3 different optimization runs
                try:
                    # Add small random perturbations to get different results
                    if attempt > 0:
                        # Perturb the initial config slightly
                        perturbed = initial_config.copy()
                        noise_scale = 0.02
                        perturbed[:, :2] += np.random.normal(0, noise_scale, (n, 2))
                        # Keep within bounds
                        perturbed[:, 0] = np.clip(perturbed[:, 0], 0.01, 0.99)
                        perturbed[:, 1] = np.clip(perturbed[:, 1], 0.01, 0.99)
                        initial_config = perturbed
                    
                    # Run coarse optimization first
                    coarse_result = optimize_circles_coarse(initial_config, n)
                    
                    if coarse_result is not None:
                        # Then run fine optimization
                        fine_result = optimize_circles_fine(coarse_result, n)
                        
                        if fine_result is not None:
                            current_sum = np.sum(fine_result[:, 2])  # Sum of radii
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_result = fine_result.copy()
                                
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    # If we found a valid solution, use it; otherwise fallback to hexagonal
    if best_result is not None:
        return best_result
    else:
        # Fallback to hexagonal arrangement with better parameters
        return create_hexagonal_arrangement(n)

def initialize_hybrid_approach(n):
    """Initialize using a hybrid approach combining multiple strategies"""
    # Start with a better hexagonal packing
    positions = []
    radii = []
    
    # Create a more sophisticated hexagonal arrangement
    rows = 6
    cols = 6
    
    # Adjust spacing to fit 32 circles better
    spacing_x = 0.9 / cols
    spacing_y = 0.9 / rows
    
    # Generate points in a hexagonal pattern
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = 0.05 + (j + 0.5) * spacing_x
            y = 0.05 + (i + 0.5) * spacing_y
            
            # Offset every other row for hexagonal pattern
            if i % 2 == 1:
                x += spacing_x / 2
                
            # Ensure point is within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append([x, y])
    
    # Pad if needed with random positions
    while len(positions) < n:
        positions.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
    
    positions = np.array(positions[:n])
    
    # Initialize with reasonable radii based on local density
    radii = np.full(n, 0.05)
    
    # Set initial radii based on distance to nearest neighbors
    for i in range(n):
        distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
        distances[i] = np.inf  # Exclude self-distance
        min_distance = np.min(distances)
        # Set radius to be reasonable relative to neighbor distance
        radii[i] = min(0.1, min_distance / 3.0)
    
    # Ensure no overlaps by adjusting radii
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
            required_radius = dist - 0.001  # Small safety margin
            if required_radius > 0:
                radii[i] = min(radii[i], required_radius/2)
                radii[j] = min(radii[j], required_radius/2)
    
    return np.column_stack([positions, radii])

def initialize_from_known_good_packings(n):
    """Initialize using a configuration inspired by known good packings"""
    # Create a more structured initial layout
    # Use a pattern similar to the best known packings for 32 circles
    
    # Place circles in a structured way - start with a central cluster
    positions = []
    radii = []
    
    # Create a central dense region
    center_density = 8  # Number of circles in central region
    for i in range(center_density):
        angle = 2 * np.pi * i / center_density
        radius = 0.15 * np.random.uniform(0.7, 1.0)
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        positions.append([x, y])
        radii.append(0.08 * np.random.uniform(0.8, 1.0))
    
    # Fill remaining positions
    remaining = n - center_density
    for i in range(remaining):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        positions.append([x, y])
        radii.append(0.05)
    
    positions = np.array(positions[:n])
    radii = np.array(radii[:n])
    
    # Ensure reasonable initial radii
    for i in range(n):
        # Adjust radii to avoid immediate overlaps
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                min_dist = min(min_dist, dist)
        
        # Set radius to be reasonable relative to neighbor distance
        radii[i] = min(0.1, min_dist / 3.0)
    
    return np.column_stack([positions, radii])

def initialize_hexagonal_lattice(n):
    """Initialize with a better hexagonal lattice"""
    # Create a more efficient hexagonal packing
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Adjust for better packing efficiency
    spacing_x = 0.8 / cols
    spacing_y = 0.8 / rows
    
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            x = 0.1 + (j + 0.5) * spacing_x
            y = 0.1 + (i + 0.5) * spacing_y
            
            # Offset every other row for hexagonal pattern
            if i % 2 == 1:
                x += spacing_x / 2
                
            # Ensure point is within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
    
    # Pad if needed
    while len(points) < n:
        points.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    
    # Initialize with reasonable radii based on distance to nearest neighbors
    positions = np.array(points[:n])
    radii = np.full(n, 0.05)
    
    # Set initial radii based on local density
    for i in range(n):
        distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
        distances[i] = np.inf  # Exclude self-distance
        min_distance = np.min(distances)
        # Set radius to be half the minimum distance to nearest neighbor, but capped
        radii[i] = min(0.1, min_distance / 3.0)
    
    return np.column_stack([positions, radii])

def initialize_random_with_radius_adjustment(n):
    """Initialize with random positions and adjust radii"""
    positions = np.random.rand(n, 2) * 0.8 + 0.1  # Keep away from edges
    radii = np.full(n, 0.05)
    
    # Adjust radii to avoid overlaps
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((positions[i] - positions[j])**2))
                min_dist = min(min_dist, dist)
        
        # Set radius to be reasonable relative to neighbor distance
        radii[i] = min(0.1, min_dist / 3.0)
    
    return np.column_stack([positions, radii])

def initialize_grid_with_perturbation(n, seed_offset):
    """Initialize with grid and add perturbations"""
    np.random.seed(42 + seed_offset)  # Different seeds for different attempts
    
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Add some randomness to prevent perfect grid
            x += np.random.normal(0, spacing_x * 0.1)
            y += np.random.normal(0, spacing_y * 0.1)
            
            # Ensure within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            points.append([x, y])
    
    # Pad if needed
    while len(points) < n:
        points.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
    
    positions = np.array(points[:n])
    radii = np.full(n, 0.05)
    
    return np.column_stack([positions, radii])

def initialize_physics_based(n):
    """Initialize using a simple physics-based approach"""
    # Start with random positions
    positions = np.random.rand(n, 2)
    
    # Apply a simple repulsion force to spread them out
    for _ in range(100):  # Iterative relaxation
        forces = np.zeros_like(positions)
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        # Simple repulsion force
                        force_magnitude = 1.0 / (dist * dist + 0.01)  # Avoid division by zero
                        forces[i] += force_magnitude * diff / dist
        
        # Move positions according to forces
        step_size = 0.001
        positions += step_size * forces
        
        # Keep within bounds
        positions = np.clip(positions, 0.01, 0.99)
    
    # Now compute appropriate radii
    radii = np.full(n, 0.05)
    
    # Set radii based on distance to nearest neighbors
    for i in range(n):
        distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
        distances[i] = np.inf  # Exclude self-distance
        min_distance = np.min(distances)
        # Set radius to be reasonable relative to neighbor distance
        radii[i] = min(0.1, min_distance / 3.0)
    
    return np.column_stack([positions, radii])

def optimize_circles_coarse(initial_config, n):
    """Coarse optimization with relaxed tolerances"""
    # Extract initial positions and radii
    initial_positions = initial_config[:, :2]
    initial_radii = initial_config[:, 2]
    
    # Set up optimization variables
    x0 = np.concatenate([initial_positions.flatten(), initial_radii])
    
    # Define bounds
    bounds = []
    # Position bounds: [0, 1] for both x and y coordinates
    for _ in range(2 * n):
        bounds.extend([(0, 1)])
    # Radius bounds: [0, 0.5] 
    for _ in range(n):
        bounds.extend([(0, 0.5)])
    
    def objective(vars):
        positions = vars[:2*n].reshape(-1, 2)
        radii = vars[2*n:]
        # Minimize negative sum of radii (maximize sum)
        return -np.sum(radii)
    
    def constraint_positions(vars):
        positions = vars[:2*n].reshape(-1, 2)
        radii = vars[2*n:]
        
        constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            
            # Circle must be fully inside the unit square
            constraints.append(x - r)  # x - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        return np.array(constraints)
    
    def constraint_overlaps(vars):
        positions = vars[:2*n].reshape(-1, 2)
        radii = vars[2*n:]
        
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1 = radii[i]
                r2 = radii[j]
                
                # Distance between centers minus sum of radii must be >= 0
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                constraints.append(dist - (r1 + r2))
                
        return np.array(constraints)
    
    # Define constraint dictionaries
    pos_constraints = {
        'type': 'ineq',
        'fun': constraint_positions
    }
    
    overlap_constraints = {
        'type': 'ineq', 
        'fun': constraint_overlaps
    }
    
    # Run coarse optimization with better settings
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[pos_constraints, overlap_constraints],
            options={'maxiter': 500, 'ftol': 1e-4, 'gtol': 1e-4},
            tol=1e-4
        )
        
        if result.success:
            final_positions = result.x[:2*n].reshape(-1, 2)
            final_radii = result.x[2*n:]
            
            # Validate solution
            if validate_solution(final_positions, final_radii):
                return np.column_stack([final_positions, final_radii])
        
    except Exception as e:
        pass
    
    return None

def optimize_circles_fine(initial_config, n):
    """Fine optimization with stricter tolerances"""
    # Extract initial positions and radii
    initial_positions = initial_config[:, :2]
    initial_radii = initial_config[:, 2]
    
    # Set up optimization variables
    x0 = np.concatenate([initial_positions.flatten(), initial_radii])
    
    # Define bounds
    bounds = []
    # Position bounds: [0, 1] for both x and y coordinates
    for _ in range(2 * n):
        bounds.extend([(0, 1)])
    # Radius bounds: [0, 0.5] 
    for _ in range(n):
        bounds.extend([(0, 0.5)])
    
    def objective(vars):
        positions = vars[:2*n].reshape(-1, 2)
        radii = vars[2*n:]
        # Minimize negative sum of radii (maximize sum)
        return -np.sum(radii)
    
    def constraint_positions(vars):
        positions = vars[:2*n].reshape(-1, 2)
        radii = vars[2*n:]
        
        constraints = []
        for i in range(n):
            x, y = positions[i]
            r = radii[i]
            
            # Circle must be fully inside the unit square
            constraints.append(x - r)  # x - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
            
        return np.array(constraints)
    
    def constraint_overlaps(vars):
        positions = vars[:2*n].reshape(-1, 2)
        radii = vars[2*n:]
        
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                r1 = radii[i]
                r2 = radii[j]
                
                # Distance between centers minus sum of radii must be >= 0
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                constraints.append(dist - (r1 + r2))
                
        return np.array(constraints)
    
    # Define constraint dictionaries
    pos_constraints = {
        'type': 'ineq',
        'fun': constraint_positions
    }
    
    overlap_constraints = {
        'type': 'ineq', 
        'fun': constraint_overlaps
    }
    
    # Run fine optimization with better settings
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[pos_constraints, overlap_constraints],
            options={'maxiter': 1000, 'ftol': 1e-6, 'gtol': 1e-6},
            tol=1e-6
        )
        
        if result.success:
            final_positions = result.x[:2*n].reshape(-1, 2)
            final_radii = result.x[2*n:]
            
            # Validate solution
            if validate_solution(final_positions, final_radii):
                return np.column_stack([final_positions, final_radii])
        
    except Exception as e:
        pass
    
    return None

def validate_solution(positions, radii):
    """Validate that the solution satisfies all constraints"""
    n = len(positions)
    
    # Check containment
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlaps with more tolerance for numerical precision
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            r1 = radii[i]
            r2 = radii[j]
            
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < (r1 + r2) - 1e-6:  # Smaller tolerance for validation
                return False
    
    return True

def create_hexagonal_arrangement(n):
    """Create a hexagonal lattice arrangement as fallback"""
    # Create hexagonal grid
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    circles = np.zeros((n, 3))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
                
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Offset every other row
            if i % 2 == 1:
                x += spacing_x / 2
                
            # Ensure point is within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Set radius based on proximity to edges
            min_dist_to_edge = min(x, 1-x, y, 1-y)
            radius = min(0.05, min_dist_to_edge * 0.8)
            
            circles[idx] = [x, y, radius]
            idx += 1
            
            if idx >= n:
                break
    
    return circles


# EVOLVE-BLOCK-END
