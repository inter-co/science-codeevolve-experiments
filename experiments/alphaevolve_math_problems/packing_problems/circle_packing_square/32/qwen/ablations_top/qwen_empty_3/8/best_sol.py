# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining physics simulation and optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Initialize with a good starting configuration
    circles = initialize_circles(n)
    
    # Optimize using a combination of approaches
    circles = optimize_circles(circles)
    
    return circles

def initialize_circles(n):
    """Initialize circles with a good starting configuration"""
    circles = np.zeros((n, 3))
    
    # Place circles in a grid pattern with small random perturbations
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Create initial grid layout
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = 0.1 + (0.8 * j / (cols - 1) if cols > 1 else 0.5)
            y = 0.1 + (0.8 * i / (rows - 1) if rows > 1 else 0.5)
            positions.append([x, y])
    
    # Distribute circles among positions
    for i in range(n):
        if i < len(positions):
            circles[i] = [positions[i][0], positions[i][1], 0.05]
        else:
            # Fill remaining slots with random positions
            circles[i] = [random.uniform(0.05, 0.95), random.uniform(0.05, 0.95), 0.05]
    
    # Set initial radii based on how much space is available
    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i != j:
                dist = np.sqrt((circles[i][0] - circles[j][0])**2 + (circles[i][1] - circles[j][1])**2)
                min_dist = min(min_dist, dist)
        
        # Radius is limited by minimum distance to other circles divided by 2
        max_radius = min(0.5, min_dist / 2.0) if min_dist < float('inf') else 0.5
        circles[i][2] = max(0.01, min(max_radius, 0.1))
    
    return circles

def calculate_energy(circles):
    """Calculate total energy representing constraint violations"""
    n = len(circles)
    energy = 0.0
    
    # Repulsion energy between circles
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Add repulsion energy when circles are too close
            if dist < (circles[i][2] + circles[j][2]):
                energy += 1.0 / (dist + 1e-8)  # Avoid division by zero
    
    # Boundary penalty for circles outside the unit square
    for i in range(n):
        x, y, r = circles[i]
        # Penalty for being too close to boundaries
        boundary_penalty = 0
        if x - r < 0:
            boundary_penalty += (r - x)**2
        if y - r < 0:
            boundary_penalty += (r - y)**2
        if x + r > 1:
            boundary_penalty += (x + r - 1)**2
        if y + r > 1:
            boundary_penalty += (y + r - 1)**2
        energy += boundary_penalty * 1000
    
    return energy

def optimize_circles(circles):
    """Optimize circle positions and radii using gradient-based approach"""
    n = len(circles)
    
    # Flatten the parameters: [x1, y1, r1, x2, y2, r2, ...]
    def flatten_params(circles_array):
        params = []
        for i in range(n):
            params.extend([circles_array[i][0], circles_array[i][1], circles_array[i][2]])
        return np.array(params)
    
    def unflatten_params(params):
        circles_array = np.zeros((n, 3))
        for i in range(n):
            circles_array[i] = [params[3*i], params[3*i+1], params[3*i+2]]
        return circles_array
    
    # Objective function to minimize (negative of sum of radii)
    def objective(params):
        circles_array = unflatten_params(params)
        # We want to maximize sum of radii, so minimize negative sum
        sum_radii = np.sum(circles_array[:, 2])
        return -sum_radii
    
    # Constraint function for non-overlapping circles
    def constraint_func(params):
        circles_array = unflatten_params(params)
        constraints = []
        
        # Non-overlapping constraints
        for i in range(n):
            for j in range(i+1, n):
                dx = circles_array[i][0] - circles_array[j][0]
                dy = circles_array[i][1] - circles_array[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                # Constraint: distance >= radii_sum (for non-overlap)
                constraints.append(dist - (circles_array[i][2] + circles_array[j][2]))
        
        # Boundary constraints (circles must stay inside unit square)
        for i in range(n):
            x, y, r = circles_array[i]
            constraints.extend([
                x - r,           # x >= r
                y - r,           # y >= r
                1 - x - r,       # 1 - x >= r
                1 - y - r        # 1 - y >= r
            ])
        
        return np.array(constraints)
    
    # Initial parameters
    initial_params = flatten_params(circles)
    
    # Use a simple iterative approach with local optimization
    best_circles = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Multiple optimization attempts with different starting points
    for attempt in range(5):
        # Random perturbation of initial configuration
        perturbed_params = initial_params.copy()
        # Add small random perturbations
        for i in range(len(perturbed_params)):
            if i % 3 < 2:  # Position parameters
                perturbed_params[i] += random.uniform(-0.05, 0.05)
            else:  # Radius parameter
                perturbed_params[i] += random.uniform(-0.02, 0.02)
                perturbed_params[i] = max(0.01, min(0.5, perturbed_params[i]))
        
        # Simple local optimization approach
        optimized_circles = unflatten_params(perturbed_params)
        
        # Apply constraints and adjust
        for _ in range(20):  # Iterative improvement
            # Adjust positions to avoid overlap
            for i in range(n):
                # Try to move circle away from others
                for j in range(n):
                    if i != j:
                        dx = optimized_circles[i][0] - optimized_circles[j][0]
                        dy = optimized_circles[i][1] - optimized_circles[j][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist < (optimized_circles[i][2] + optimized_circles[j][2]):
                            # Move circle away from overlapping circle
                            if dist > 1e-8:
                                move_x = dx / dist * (optimized_circles[i][2] + optimized_circles[j][2] - dist) * 0.1
                                move_y = dy / dist * (optimized_circles[i][2] + optimized_circles[j][2] - dist) * 0.1
                                
                                # Apply boundary constraints
                                new_x = optimized_circles[i][0] + move_x
                                new_y = optimized_circles[i][1] + move_y
                                
                                if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                                    optimized_circles[i][0] = new_x
                                    optimized_circles[i][1] = new_y
            
            # Adjust radii to maximize sum while maintaining constraints
            for i in range(n):
                # Calculate maximum possible radius for this circle
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        dx = optimized_circles[i][0] - optimized_circles[j][0]
                        dy = optimized_circles[i][1] - optimized_circles[j][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = min(min_dist, dist)
                
                if min_dist < float('inf'):
                    max_radius = min(0.5, min_dist / 2.0)
                    optimized_circles[i][2] = min(max_radius, optimized_circles[i][2] * 1.01)
                
                # Ensure boundary constraints
                max_radius_x = min(optimized_circles[i][0], 1 - optimized_circles[i][0])
                max_radius_y = min(optimized_circles[i][1], 1 - optimized_circles[i][1])
                max_radius = min(max_radius_x, max_radius_y)
                optimized_circles[i][2] = min(optimized_circles[i][2], max_radius)
        
        # Check if this configuration is better
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
    
    # Final refinement
    final_circles = best_circles.copy()
    
    # Apply final boundary corrections
    for i in range(n):
        x, y, r = final_circles[i]
        # Ensure circle stays within bounds
        final_circles[i][0] = max(r, min(1-r, x))
        final_circles[i][1] = max(r, min(1-r, y))
    
    return final_circles


# EVOLVE-BLOCK-END
