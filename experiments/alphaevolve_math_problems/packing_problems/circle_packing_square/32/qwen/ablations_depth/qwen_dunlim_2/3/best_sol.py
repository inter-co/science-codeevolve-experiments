# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining greedy initialization, force-based physics simulation, 
    and local optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    n = 32
    
    def initialize_circles():
        """Initialize circles with a better spatial distribution"""
        positions = []
        radii = []
        
        # Create a more effective initial configuration using a spiral-like pattern
        # This helps distribute points more evenly initially
        angle_step = 2 * np.pi / 10
        radius_step = 0.4 / 10
        
        for i in range(n):
            if len(positions) >= n:
                break
            # Spiral pattern with some randomness
            angle = i * angle_step + np.random.randn() * 0.2
            radius = min(0.4, (i % 10) * radius_step + 0.1 + np.random.randn() * 0.05)
            
            # Convert to Cartesian coordinates
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            
            # Keep within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            positions.append([x, y])
            # Initial radius based on distance to nearest boundaries
            min_dist_to_boundary = min(x, 1-x, y, 1-y)
            radii.append(min(min_dist_to_boundary * 0.3, 0.1))
        
        # Fill remaining positions with random placement but ensuring good distribution
        while len(positions) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            positions.append([x, y])
            min_dist_to_boundary = min(x, 1-x, y, 1-y)
            radii.append(min(min_dist_to_boundary * 0.25, 0.1))
            
        return np.array(positions[:n]), np.array(radii[:n])
    
    def compute_forces(positions, radii):
        """Compute repulsive forces between circles with improved physics model"""
        n = len(positions)
        forces = np.zeros_like(positions)
        
        # Use KDTree for efficient neighbor search
        tree = cKDTree(positions)
        
        # For each circle, compute forces from nearby circles
        for i in range(n):
            # Find neighbors within a reasonable distance (2x max radius)
            max_radius = max(radii) if len(radii) > 0 else 0.1
            neighbors = tree.query_ball_point(positions[i], 3 * max_radius)
            
            for j in neighbors:
                if i != j:
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        min_dist = radii[i] + radii[j]
                        # Use a physically realistic force model
                        if dist < min_dist:
                            # Strong repulsion when overlapping
                            force_magnitude = (min_dist - dist) * 1000 / (dist + 1e-10)
                            forces[i][0] += force_magnitude * dx / dist
                            forces[i][1] += force_magnitude * dy / dist
                        elif dist < 2 * min_dist:
                            # Moderate repulsion for near contacts
                            force_magnitude = (min_dist - dist) * 100 / (dist + 1e-10)
                            forces[i][0] += force_magnitude * dx / dist
                            forces[i][1] += force_magnitude * dy / dist
        
        # Boundary forces - push circles back inside with very strong force
        for i in range(n):
            # Left boundary
            if positions[i][0] - radii[i] < 0:
                forces[i][0] += (0 - positions[i][0] + radii[i]) * 1000
            # Right boundary  
            if positions[i][0] + radii[i] > 1:
                forces[i][0] += (1 - positions[i][0] - radii[i]) * 1000
            # Bottom boundary
            if positions[i][1] - radii[i] < 0:
                forces[i][1] += (0 - positions[i][1] + radii[i]) * 1000
            # Top boundary
            if positions[i][1] + radii[i] > 1:
                forces[i][1] += (1 - positions[i][1] - radii[i]) * 1000
                
        return forces
    
    def update_positions(positions, radii, forces, dt=0.001):
        """Update positions based on forces with better damping"""
        new_positions = positions.copy()
        new_radii = radii.copy()
        
        # Update positions
        for i in range(len(positions)):
            # Limit force magnitude to prevent large jumps
            force_magnitude = np.sqrt(forces[i][0]**2 + forces[i][1]**2)
            if force_magnitude > 100:
                forces[i] *= 100 / (force_magnitude + 1e-10)
            
            # Apply damping
            new_positions[i] += forces[i] * dt
            
            # Keep positions within bounds
            new_positions[i][0] = np.clip(new_positions[i][0], radii[i], 1 - radii[i])
            new_positions[i][1] = np.clip(new_positions[i][1], radii[i], 1 - radii[i])
            
        return new_positions, new_radii
    
    def evaluate_fitness(positions, radii):
        """Evaluate the fitness (negative sum of radii)"""
        return -np.sum(radii)
    
    def adjust_radii(positions, radii):
        """Improve radii by trying to increase them while respecting constraints"""
        new_radii = radii.copy()
        improved = False
        
        # Try to increase each radius individually
        for i in range(len(positions)):
            old_radius = new_radii[i]
            # Maximum possible radius for this circle
            max_radius = min(positions[i][0], 1-positions[i][0], 
                           positions[i][1], 1-positions[i][1])
            
            # Find minimum distance to any other circle
            min_distance = float('inf')
            for j in range(len(positions)):
                if i != j:
                    dist = np.sqrt((positions[i][0]-positions[j][0])**2 + 
                                 (positions[i][1]-positions[j][1])**2)
                    min_distance = min(min_distance, dist)
            
            if min_distance > 0:
                # Can increase radius up to half the minimum distance minus current radius
                max_possible_radius = min(max_radius, min_distance/2 - 1e-6)
                if max_possible_radius > old_radius and max_possible_radius > 0:
                    new_radii[i] = max_possible_radius
                    improved = True
        
        return new_radii, improved
    
    # Initialize
    positions, radii = initialize_circles()
    
    # Combine into single array: [x, y, r] for each circle
    circles = np.column_stack([positions, radii])
    
    # Physics simulation loop with optimized parameters
    best_fitness = float('inf')
    best_circles = circles.copy()
    
    # Run longer simulation with better convergence control
    for iteration in range(4000):
        # Compute forces
        forces = compute_forces(positions, radii)
        
        # Update positions
        positions, radii = update_positions(positions, radii, forces, dt=0.001)
        
        # Try to adjust radii
        new_radii, improved = adjust_radii(positions, radii)
        if improved:
            radii = new_radii
        
        # Update circles array
        circles = np.column_stack([positions, radii])
        
        # Track best solution
        current_fitness = evaluate_fitness(positions, radii)
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_circles = circles.copy()
        
        # Adaptive stopping criteria - stop if we haven't improved in a while
        if iteration > 2000 and iteration % 500 == 0:
            # Check for convergence by examining recent improvements
            pass
    
    # Final optimization using scipy with enhanced settings and multiple restarts
    def objective(params):
        circles_flat = params.reshape(-1, 3)
        radii = circles_flat[:, 2]
        return -np.sum(radii)
    
    def constraint_containment(params):
        circles_flat = params.reshape(-1, 3)
        positions = circles_flat[:, :2]
        radii = circles_flat[:, 2]
        
        left_constraint = positions[:, 0] - radii
        right_constraint = 1 - positions[:, 0] - radii
        bottom_constraint = positions[:, 1] - radii
        top_constraint = 1 - positions[:, 1] - radii
        
        return np.concatenate([left_constraint, right_constraint, bottom_constraint, top_constraint])
    
    def constraint_nonoverlap(params):
        circles_flat = params.reshape(-1, 3)
        positions = circles_flat[:, :2]
        radii = circles_flat[:, 2]
        
        distances = cdist(positions, positions)
        min_distances = radii[:, None] + radii[None, :]
        violations = (distances - min_distances)
        mask = np.triu(np.ones_like(violations), k=1).astype(bool)
        violations = violations[mask]
        return violations
    
    # Use the best configuration as starting point for final optimization
    initial_guess = best_circles.flatten()
    
    # Create bounds for each parameter (x, y, r)
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x coordinate bounds
        bounds.append((0.001, 0.999))  # y coordinate bounds  
        bounds.append((0.001, 0.499))  # radius bounds
    
    # Apply constraints using scipy.optimize.minimize with more iterations and better settings
    try:
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
                {'type': 'ineq', 'fun': lambda x: constraint_nonoverlap(x)}
            ],
            options={'maxiter': 2000, 'ftol': 1e-9, 'eps': 1e-6, 'disp': False}
        )
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            return final_circles
        else:
            return best_circles
            
    except Exception as e:
        return best_circles


# EVOLVE-BLOCK-END
