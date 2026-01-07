# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple

# Global constants for the problem
N_CIRCLES = 32
BENCHMARK = 2.937944526205518

def generate_hexagonal_lattice() -> np.ndarray:
    """Generate initial configuration using hexagonal lattice pattern"""
    # For 32 circles, we'll use a 6x6 grid with some adjustments
    # This provides a good starting configuration
    
    # Create a hexagonal grid pattern
    rows = 6
    cols = 6
    positions = []
    
    # Hexagonal packing parameters
    sqrt3 = np.sqrt(3)
    spacing_x = 1.0 / (cols - 1)
    spacing_y = 1.0 / (rows - 1)
    
    # Adjust for hexagonal packing
    y_offset = 0.5 * spacing_y
    
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= N_CIRCLES:
                break
            x = j * spacing_x
            y = i * spacing_y
            # Offset every other row for hexagonal packing
            if i % 2 == 1:
                x += 0.5 * spacing_x
            positions.append([x, y])
    
    # Trim to exactly N_CIRCLES
    positions = positions[:N_CIRCLES]
    
    # Initialize with small radii
    circles = np.array(positions)
    radii = np.full(N_CIRCLES, 0.02)
    circles = np.column_stack([circles, radii])
    
    return circles

def calculate_radius_bounds(circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate minimum and maximum possible radii for each circle"""
    n = len(circles)
    min_radius = np.zeros(n)
    max_radius = np.zeros(n)
    
    for i in range(n):
        x, y, _ = circles[i]
        # Maximum radius without going outside the square
        max_radius[i] = min(x, y, 1-x, 1-y)
        
        # Minimum radius is determined by non-overlap constraints
        min_radius[i] = 0.001  # Small positive value
    
    return min_radius, max_radius

def compute_objective(circles: np.ndarray) -> float:
    """Compute the objective function (negative sum of radii for minimization)"""
    return -np.sum(circles[:, 2])

def compute_constraints(circles: np.ndarray) -> dict:
    """Compute constraint violations"""
    n = len(circles)
    constraints = []
    
    # Containment constraints
    for i in range(n):
        x, y, r = circles[i]
        # r <= x, r <= y, r <= 1-x, r <= 1-y
        if r > x or r > y or r > (1-x) or r > (1-y):
            constraints.append(f"Containment violation for circle {i}")
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < (r1 + r2):
                constraints.append(f"Overlap between circles {i} and {j}")
    
    return constraints

def optimize_circles(initial_circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Optimize circle positions and radii using scipy minimize"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_flat = initial_circles.flatten()
    
    def objective_flat(params):
        # Reshape back to circles array
        circles = params.reshape((n, 3))
        return -np.sum(circles[:, 2])  # Negative because we want to maximize
    
    def constraint_func(params):
        circles = params.reshape((n, 3))
        constraints = []
        
        # Containment constraints (all must be >= 0)
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([
                x - r,      # x >= r
                y - r,      # y >= r
                1 - x - r,  # 1-x >= r
                1 - y - r   # 1-y >= r
            ])
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(distance - (r1 + r2))  # distance >= r1 + r2
        
        return np.array(constraints)
    
    # Create bounds for variables (x, y, r) for each circle
    bounds = []
    for i in range(n):
        # Bounds for x and y: [r, 1-r] to ensure containment
        # Bounds for r: [0.001, min(x, y, 1-x, 1-y)]
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.999)])
    
    # Define constraints
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    try:
        result = minimize(
            objective_flat,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-6}
        )
        
        if result.success:
            optimized_circles = result.x.reshape((n, 3))
            return optimized_circles
    except Exception as e:
        pass
    
    # Return initial if optimization fails
    return initial_circles

def perturb_configuration(circles: np.ndarray, perturbation: float = 0.01) -> np.ndarray:
    """Create a slightly perturbed version of the configuration"""
    perturbed = circles.copy()
    for i in range(len(perturbed)):
        # Slightly perturb position
        perturbed[i, 0] += random.uniform(-perturbation, perturbation)
        perturbed[i, 1] += random.uniform(-perturbation, perturbation)
        # Ensure within bounds
        perturbed[i, 0] = max(0.01, min(0.99, perturbed[i, 0]))
        perturbed[i, 1] = max(0.01, min(0.99, perturbed[i, 1]))
    return perturbed

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a multi-start optimization approach with hexagonal lattice initialization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    best_circles = None
    best_sum = 0
    
    # Try multiple starting configurations
    for attempt in range(5):
        # Generate initial configuration
        if attempt == 0:
            # First attempt: hexagonal lattice
            circles = generate_hexagonal_lattice()
        else:
            # Subsequent attempts: random perturbations of good solutions
            if best_circles is not None:
                circles = perturb_configuration(best_circles, 0.05)
            else:
                circles = generate_hexagonal_lattice()
        
        # Optimize this configuration
        optimized_circles = optimize_circles(circles, max_iter=500)
        
        # Calculate sum of radii
        sum_radii = np.sum(optimized_circles[:, 2])
        
        # Update best solution
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_circles = optimized_circles.copy()
    
    # Final refinement
    if best_circles is not None:
        final_circles = optimize_circles(best_circles, max_iter=1000)
        sum_radii = np.sum(final_circles[:, 2])
        
        # If still better, update
        if sum_radii > best_sum:
            best_circles = final_circles
    
    # Ensure we have a valid solution even if optimization failed
    if best_circles is None:
        best_circles = generate_hexagonal_lattice()
    
    return best_circles


# EVOLVE-BLOCK-END
