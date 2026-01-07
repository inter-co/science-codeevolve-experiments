# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time
import math
from sklearn.cluster import KMeans
import warnings
from itertools import combinations
from scipy.spatial import Voronoi
import random
from scipy.optimize import minimize
import numba

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

@numba.jit(nopython=True)
def compute_distances_numba(circles):
    """Compute pairwise distances efficiently using numba."""
    n = len(circles)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a physics-based approach with Voronoi initialization and molecular dynamics simulation.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with Voronoi-based approach for good starting configuration
    initial_circles = voronoi_initialization(n)
    
    # Simulate physics-based optimization
    optimized_circles = physics_based_optimization(initial_circles)
    
    # Final refinement
    refined = final_refinement(optimized_circles)
    
    return refined

def voronoi_initialization(n: int) -> np.ndarray:
    """Initialize circles using Voronoi diagram approach for better distribution."""
    # Generate points using a grid-like pattern with some randomness
    points = []
    # Create a grid of points, then add some randomness
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing = 1.0 / (grid_size + 1)
    
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < n:
                x = (i + 1) * spacing + np.random.normal(0, spacing * 0.1)
                y = (j + 1) * spacing + np.random.normal(0, spacing * 0.1)
                # Keep within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                points.append([x, y])
    
    # If we don't have enough points, add random ones
    while len(points) < n:
        points.append([np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)])
    
    points = np.array(points[:n])
    
    # Initialize with small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i] = [points[i, 0], points[i, 1], 0.02]
    
    # Grow radii while respecting constraints
    circles = grow_radii_constrained(circles)
    
    return circles

def grow_radii_constrained(circles: np.ndarray) -> np.ndarray:
    """Grow radii as much as possible while maintaining constraints."""
    n = len(circles)
    max_iterations = 1000
    
    for iteration in range(max_iterations):
        improved = False
        # Try to increase each radius
        for i in range(n):
            original_r = circles[i, 2]
            # Try to increase radius by small amount
            test_r = min(original_r + 0.001, 0.45)
            
            # Check if we can increase radius
            valid = True
            for j in range(n):
                if i != j:
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < test_r + circles[j, 2]:
                        valid = False
                        break
            
            if valid:
                # Check boundary constraints
                if (circles[i, 0] - test_r >= 0 and 
                    circles[i, 0] + test_r <= 1 and
                    circles[i, 1] - test_r >= 0 and
                    circles[i, 1] + test_r <= 1):
                    circles[i, 2] = test_r
                    improved = True
        
        if not improved:
            break
    
    return circles

def physics_based_optimization(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize using physics-based molecular dynamics simulation."""
    circles = initial_circles.copy()
    n = len(circles)
    
    # Parameters for simulation
    dt = 0.001
    max_steps = 10000
    temperature = 0.1
    
    # Force calculation function
    def calculate_forces(circles):
        forces = np.zeros_like(circles)
        
        # Repulsive forces between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < circles[i, 2] + circles[j, 2] and distance > 0:
                    # Repulsion force
                    force_magnitude = 1.0 / (distance * distance + 0.001)
                    forces[i, 0] += force_magnitude * dx / distance
                    forces[i, 1] += force_magnitude * dy / distance
                    forces[j, 0] -= force_magnitude * dx / distance
                    forces[j, 1] -= force_magnitude * dy / distance
        
        # Attractive forces to boundaries (penalty for being too close to edges)
        for i in range(n):
            # Boundary repulsion
            boundary_force_x = 0
            boundary_force_y = 0
            
            # Left boundary
            if circles[i, 0] < circles[i, 2]:
                boundary_force_x += (circles[i, 2] - circles[i, 0]) * 100
            # Right boundary
            elif circles[i, 0] > 1 - circles[i, 2]:
                boundary_force_x += -(circles[i, 0] - (1 - circles[i, 2])) * 100
            
            # Bottom boundary
            if circles[i, 1] < circles[i, 2]:
                boundary_force_y += (circles[i, 2] - circles[i, 1]) * 100
            # Top boundary
            elif circles[i, 1] > 1 - circles[i, 2]:
                boundary_force_y += -(circles[i, 1] - (1 - circles[i, 2])) * 100
            
            forces[i, 0] += boundary_force_x
            forces[i, 1] += boundary_force_y
        
        return forces
    
    # Run simulation
    for step in range(max_steps):
        # Calculate forces
        forces = calculate_forces(circles)
        
        # Update positions and velocities
        for i in range(n):
            # Apply forces with temperature (random noise)
            noise_x = np.random.normal(0, temperature)
            noise_y = np.random.normal(0, temperature)
            
            # Update velocity (simple integration)
            circles[i, 0] += dt * forces[i, 0] + noise_x
            circles[i, 1] += dt * forces[i, 1] + noise_y
            
            # Enforce boundary conditions
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
        # Occasionally try to increase radii
        if step % 100 == 0:
            # Try to increase radii slightly
            for i in range(n):
                test_r = min(circles[i, 2] + 0.0005, 0.45)
                valid = True
                
                # Check overlap with other circles
                for j in range(n):
                    if i != j:
                        dx = circles[i, 0] - circles[j, 0]
                        dy = circles[i, 1] - circles[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < test_r + circles[j, 2]:
                            valid = False
                            break
                
                # Check boundary
                if valid and (circles[i, 0] - test_r >= 0 and 
                             circles[i, 0] + test_r <= 1 and
                             circles[i, 1] - test_r >= 0 and
                             circles[i, 1] + test_r <= 1):
                    circles[i, 2] = test_r
    
    return circles

def final_refinement(circles: np.ndarray) -> np.ndarray:
    """Apply final refinement to optimize the solution."""
    # Use a more sophisticated local optimization approach
    n = len(circles)
    
    # Try to improve each circle's position and radius
    improved = True
    iterations = 0
    max_iterations = 500
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to improve each circle
        for i in range(n):
            original = circles[i].copy()
            
            # Try to increase radius
            test_radius = min(original[2] + 0.001, 0.45)
            
            # Check if we can increase radius without violating constraints
            valid = True
            for j in range(n):
                if i != j:
                    dist = np.sqrt((original[0] - circles[j, 0])**2 + 
                                 (original[1] - circles[j, 1])**2)
                    if dist < test_radius + circles[j, 2]:
                        valid = False
                        break
            
            if valid:
                # Check boundary constraints
                if (original[0] - test_radius >= 0 and 
                    original[0] + test_radius <= 1 and
                    original[1] - test_radius >= 0 and
                    original[1] + test_radius <= 1):
                    circles[i] = [original[0], original[1], test_radius]
                    improved = True
                    continue
            
            # Try to slightly adjust position
            # Move towards center to avoid edge constraints
            center_x = 0.5
            center_y = 0.5
            dx = center_x - original[0]
            dy = center_y - original[1]
            
            # Normalize and scale
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 0:
                dx = dx / norm * 0.001
                dy = dy / norm * 0.001
                
                new_x = original[0] + dx
                new_y = original[1] + dy
                
                # Clip to bounds
                new_x = np.clip(new_x, original[2], 1 - original[2])
                new_y = np.clip(new_y, original[2], 1 - original[2])
                
                # Check if this improves constraints
                valid_pos = True
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((new_x - circles[j, 0])**2 + 
                                     (new_y - circles[j, 1])**2)
                        if dist < original[2] + circles[j, 2]:
                            valid_pos = False
                            break
                
                if valid_pos:
                    circles[i] = [new_x, new_y, original[2]]
                    improved = True
    
    # Final optimization using scipy
    try:
        # Flatten for optimization
        initial_vars = circles.flatten()
        
        def objective(vars):
            # Extract circles
            circles_opt = vars.reshape(-1, 3)
            # Minimize negative sum of radii (since we want to maximize)
            return -np.sum(circles_opt[:, 2])
        
        def constraint_func(vars):
            circles_opt = vars.reshape(-1, 3)
            # Check all constraints
            constraints = []
            
            # Boundary constraints
            for i in range(len(circles_opt)):
                x, y, r = circles_opt[i]
                # r <= x <= 1-r and r <= y <= 1-r
                constraints.extend([
                    x - r,           # x - r >= 0
                    1 - x - r,       # 1 - x - r >= 0
                    y - r,           # y - r >= 0
                    1 - y - r        # 1 - y - r >= 0
                ])
            
            # Overlap constraints
            for i in range(len(circles_opt)):
                for j in range(i+1, len(circles_opt)):
                    x1, y1, r1 = circles_opt[i]
                    x2, y2, r2 = circles_opt[j]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    # distance >= r1 + r2, so distance - (r1 + r2) >= 0
                    constraints.append(distance - (r1 + r2))
            
            return np.array(constraints)
        
        # Define bounds
        bounds = [(0, 1), (0, 1), (0.001, 0.5)] * n
        
        # Optimize
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Validate and clip
            for i in range(len(optimized_circles)):
                x, y, r = optimized_circles[i]
                optimized_circles[i] = [
                    np.clip(x, r, 1-r),
                    np.clip(y, r, 1-r),
                    max(r, 0.001)
                ]
            return optimized_circles
    except:
        pass
    
    return circles


# EVOLVE-BLOCK-END
