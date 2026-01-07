# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
import time

# Global constants
N_CIRCLES = 32
BOUNDARY_MARGIN = 1e-6

def initialize_grid_pattern():
    """Initialize circles in a grid-like pattern"""
    # Arrange in a grid pattern that fills the space reasonably well
    rows = cols = int(math.ceil(math.sqrt(N_CIRCLES)))
    
    # Create initial grid positions
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= N_CIRCLES:
                break
            # Distribute points more evenly
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            positions.append([x, y])
    
    # If we have more positions than needed, trim down
    positions = positions[:N_CIRCLES]
    
    # Set initial radii to be small but reasonable
    radii = np.full(N_CIRCLES, 0.05)
    
    return np.array(positions), radii

def compute_forces(positions, radii):
    """Compute forces between circles based on overlap and boundary constraints"""
    n = len(positions)
    forces = np.zeros((n, 2))
    
    # Compute pairwise distances
    distances = cdist(positions, positions)
    
    # Repulsive forces between overlapping circles
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            r_i, r_j = radii[i], radii[j]
            
            # Check if circles overlap
            if dist < (r_i + r_j):
                # Stronger repulsive force when circles overlap significantly
                force_magnitude = (r_i + r_j - dist) * 1000.0
                direction = (positions[i] - positions[j]) / (dist + 1e-8)
                forces[i] += force_magnitude * direction
                forces[j] -= force_magnitude * direction
    
    # Boundary forces - stronger repulsion from edges
    for i in range(n):
        x, y, r = positions[i][0], positions[i][1], radii[i]
        # Boundary constraints - strong repulsion from edges
        fx, fy = 0.0, 0.0
        if x < r:
            fx += (r - x) * 10000
        elif x > 1 - r:
            fx -= (x - (1 - r)) * 10000
            
        if y < r:
            fy += (r - y) * 10000
        elif y > 1 - r:
            fy -= (y - (1 - r)) * 10000
            
        forces[i] += np.array([fx, fy])
    
    return forces

def optimize_circles(initial_positions, initial_radii, max_iter=1000):
    """Optimize circle positions using force-based simulation"""
    positions = initial_positions.copy()
    radii = initial_radii.copy()
    
    # Simple gradient descent with momentum
    velocity = np.zeros_like(positions)
    momentum = 0.9
    learning_rate = 0.005
    
    for _ in range(max_iter):
        forces = compute_forces(positions, radii)
        
        # Update velocities and positions
        for i in range(len(positions)):
            # Apply forces to velocity
            velocity[i] = momentum * velocity[i] + learning_rate * forces[i]
            
            # Update position
            positions[i] += velocity[i]
            
            # Enforce boundary constraints
            positions[i, 0] = np.clip(positions[i, 0], radii[i], 1 - radii[i])
            positions[i, 1] = np.clip(positions[i, 1], radii[i], 1 - radii[i])
    
    return positions, radii

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    start_time = time.time()
    
    try:
        # Initialize with a structured grid pattern
        positions, radii = initialize_grid_pattern()
        
        # Optimize using force-based simulation
        optimized_positions, optimized_radii = optimize_circles(positions, radii, max_iter=800)
        
        # Final refinement - try to increase radii while maintaining constraints
        for _ in range(200):
            improved = False
            for i in range(len(optimized_radii)):
                # Try to slightly increase radius
                test_radius = min(optimized_radii[i] * 1.03, 0.4)  # Slightly higher growth factor
                
                # Check if we can increase radius without violating constraints
                valid = True
                for j in range(len(optimized_positions)):
                    if i != j:
                        dist_sq = np.sum((optimized_positions[i] - optimized_positions[j])**2)
                        if dist_sq < (test_radius + optimized_radii[j])**2:
                            valid = False
                            break
                
                # Check boundary constraints
                if valid and test_radius <= optimized_positions[i, 0] and \
                   test_radius <= 1 - optimized_positions[i, 0] and \
                   test_radius <= optimized_positions[i, 1] and \
                   test_radius <= 1 - optimized_positions[i, 1]:
                    optimized_radii[i] = test_radius
                    improved = True
            
            # If no improvements, stop early
            if not improved:
                break
        
        # Create final result array
        circles = np.column_stack([optimized_positions, optimized_radii])
        
    except Exception as e:
        # Fallback to grid-based solution if anything fails
        print(f"Optimization failed with error: {e}")
        circles = np.zeros((N_CIRCLES, 3))
        rows = cols = int(np.ceil(np.sqrt(N_CIRCLES)))
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        radius = min(spacing_x, spacing_y) * 0.4
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N_CIRCLES:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                circles[idx] = [x, y, radius]
                idx += 1
    
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.4f} seconds")
    
    return circles


# EVOLVE-BLOCK-END
