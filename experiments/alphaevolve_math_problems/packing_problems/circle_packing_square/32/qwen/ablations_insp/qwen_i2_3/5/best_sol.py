# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
import time
import math
from numba import jit

# Global constants
N_CIRCLES = 32
MAX_TIME = 55.0  # Leave some buffer for cleanup

@jit(nopython=True)
def fast_distance(x1, y1, x2, y2):
    """Fast computation of Euclidean distance"""
    dx = x2 - x1
    dy = y2 - y1
    return np.sqrt(dx*dx + dy*dy)

def initialize_hexagonal_grid():
    """
    Initialize circles using a hexagonal grid pattern for better packing.
    Enhanced version with better distribution and more robust initialization.
    """
    # Create a hexagonal grid pattern with better distribution
    rows = int(math.ceil(math.sqrt(N_CIRCLES)))
    cols = int(math.ceil(N_CIRCLES / rows))
    
    # Adjust for better packing
    if rows * cols < N_CIRCLES:
        rows += 1
        
    # Hexagonal spacing - better distribution
    spacing_x = 0.8 / cols  # Leave margin for boundary constraints
    spacing_y = spacing_x * math.sqrt(3) / 2
    
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= N_CIRCLES:
                break
            # Offset every other row for hexagonal packing
            offset = spacing_x / 2 if i % 2 == 1 else 0
            x = 0.1 + (j + 0.5 + offset) * spacing_x
            y = 0.1 + (i + 0.5) * spacing_y
            
            # Ensure we're within bounds
            if x >= 0.05 and x <= 0.95 and y >= 0.05 and y <= 0.95:
                # Set initial radius to be small but feasible
                r = min(x, 1-x, y, 1-y) * 0.3
                circles.append([x, y, r])
                
    # Fill remaining positions if needed with better distribution
    while len(circles) < N_CIRCLES:
        # Add positions near boundaries but not too close to edges
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = min(x, 1-x, y, 1-y) * 0.25
        circles.append([x, y, r])
        
    return np.array(circles)

def is_feasible(circles: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if the configuration is feasible"""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if not (r <= x <= 1-r and r <= y <= 1-r):
            return False
    
    # Check overlap constraints - optimized version using fast distance
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            # Use fast distance calculation
            distance = fast_distance(x1, y1, x2, y2)
            if distance < r1 + r2 - tolerance:
                return False
    
    return True

def evaluate_objective(circles: np.ndarray) -> float:
    """Evaluate the objective function (sum of radii)"""
    return np.sum(circles[:, 2])

def compute_overlap_distances(circles: np.ndarray) -> np.ndarray:
    """Compute pairwise distances between circle centers"""
    positions = circles[:, :2]
    return cdist(positions, positions)

def force_refinement(circles, max_iterations=200, time_limit=None):
    """
    Force-based refinement for better local optimization.
    Enhanced version with time management and better force computation.
    """
    circles = circles.copy()
    start_time_local = time.time()
    
    for iteration in range(max_iterations):
        if time_limit and time.time() - start_time_local > time_limit:
            break
            
        # Compute forces for all circles
        forces = np.zeros((len(circles), 2))
        
        # Repulsive forces between overlapping circles
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                
                dx = x2 - x1
                dy = y2 - y1
                distance = fast_distance(x1, y1, x2, y2)
                
                if distance > 0 and distance < (r1 + r2):
                    # Stronger repulsion when circles are very close
                    force_magnitude = 1.0 / (distance * distance + 1e-8)
                    # Scale by how much they overlap
                    overlap = (r1 + r2) - distance
                    force_magnitude *= overlap
                    
                    forces[i, 0] += force_magnitude * dx / distance
                    forces[i, 1] += force_magnitude * dy / distance
                    forces[j, 0] -= force_magnitude * dx / distance
                    forces[j, 1] -= force_magnitude * dy / distance
        
        # Attractive forces to boundaries (containment)
        for i in range(len(circles)):
            x, y, r = circles[i]
            # Attract to boundaries with strength proportional to distance
            fx, fy = 0.0, 0.0
            
            # Left boundary
            if x < r:
                fx += (r - x) * 10
            elif x > 1 - r:
                fx += (1 - r - x) * 10
                
            # Bottom boundary
            if y < r:
                fy += (r - y) * 10
            elif y > 1 - r:
                fy += (1 - r - y) * 10
                
            forces[i, 0] += fx
            forces[i, 1] += fy
        
        # Apply forces with adaptive step size
        for i in range(len(circles)):
            force_magnitude = np.sqrt(forces[i, 0]**2 + forces[i, 1]**2)
            
            # Adaptive step size
            if force_magnitude > 0:
                step_size = min(0.02, 0.1 / (force_magnitude + 1e-8))
                circles[i, 0] += forces[i, 0] * step_size
                circles[i, 1] += forces[i, 1] * step_size
                
                # Clamp to valid region with margin
                circles[i, 0] = np.clip(circles[i, 0], r, 1-r)
                circles[i, 1] = np.clip(circles[i, 1], r, 1-r)
    
    return circles

def local_improvement(circles, max_iter=50, time_limit=None):
    """
    Local improvement to maximize individual radii.
    Enhanced version with time management and better radius adjustment.
    """
    circles = circles.copy()
    start_time_local = time.time()
    
    for iteration in range(max_iter):
        if time_limit and time.time() - start_time_local > time_limit:
            break
            
        improved = False
        
        # Try to increase radius of each circle
        for i in range(len(circles)):
            if time_limit and time.time() - start_time_local > time_limit:
                break
                
            x, y, r = circles[i]
            
            # Compute maximum possible radius for this circle
            max_radius = min(x, 1-x, y, 1-y)
            
            # Check overlap constraints
            for j in range(len(circles)):
                if i != j:
                    x1, y1, r1 = circles[j]
                    distance = fast_distance(x, y, x1, y1)
                    max_radius = min(max_radius, distance - r1)
            
            # Try to increase radius if possible
            if max_radius > r:
                new_r = min(max_radius, r + 0.005)
                # Test if this change maintains feasibility
                test_circles = circles.copy()
                test_circles[i, 2] = new_r
                
                # Quick feasibility check (simple version using fast distance)
                valid = True
                for k in range(len(test_circles)):
                    if k != i:
                        x1, y1, r1 = test_circles[k]
                        x2, y2, r2 = test_circles[i]
                        distance = fast_distance(x1, y1, x2, y2)
                        if distance < r1 + r2:
                            valid = False
                            break
                
                if valid:
                    circles[i, 2] = new_r
                    improved = True
        
        if not improved:
            break
            
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a multi-strategy approach combining hexagonal initialization, force-based refinement, 
    and local optimization to beat the benchmark.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores 
        the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    start_time = time.time()
    
    best_result = None
    best_sum = -np.inf
    
    # Try several different initializations and optimization paths
    for restart in range(3):  # Multiple restarts
        if time.time() - start_time > MAX_TIME:
            break
            
        # Initialize with hexagonal grid
        circles = initialize_hexagonal_grid()
        
        # Apply force refinement with time limit
        remaining_time = MAX_TIME - (time.time() - start_time) - 10.0  # Leave 10s for final steps
        if remaining_time > 0:
            circles = force_refinement(circles, max_iterations=100, time_limit=remaining_time * 0.7)
        
        # Apply local improvement
        remaining_time = MAX_TIME - (time.time() - start_time) - 10.0  # Leave 10s for final steps
        if remaining_time > 0:
            circles = local_improvement(circles, max_iter=30, time_limit=remaining_time * 0.5)
        
        # Calculate sum of radii
        current_sum = evaluate_objective(circles)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = circles.copy()
    
    # Final optimization with scipy if we haven't timed out
    if best_result is not None and time.time() - start_time < MAX_TIME - 5.0:
        # Define constraint functions for scipy optimization
        def get_constraints():
            cons = []
            
            # Boundary constraints: each circle must fit entirely within the unit square
            def boundary_constraint(i):
                def constraint(params):
                    x, y, r = params[3*i:3*i+3]
                    # Circle must be within bounds with margin
                    return min(r, x-r, 1-x-r, y-r, 1-y-r)
                return constraint
            
            # Non-overlap constraints: distance between centers >= sum of radii
            def overlap_constraint(i, j):
                def constraint(params):
                    x1, y1, r1 = params[3*i:3*i+3]
                    x2, y2, r2 = params[3*j:3*j+3]
                    # Distance between centers minus sum of radii
                    dist = fast_distance(x1, y1, x2, y2)
                    return dist - (r1 + r2)
                return constraint
            
            # Add boundary constraints
            for i in range(N_CIRCLES):
                cons.append({'type': 'ineq', 'fun': boundary_constraint(i)})
            
            # Add overlap constraints
            for i in range(N_CIRCLES):
                for j in range(i+1, N_CIRCLES):
                    cons.append({'type': 'ineq', 'fun': overlap_constraint(i, j)})
                    
            return cons
        
        # Define objective function (negative because we want to maximize sum of radii)
        def objective(params):
            total_radius = 0
            for i in range(N_CIRCLES):
                total_radius += params[3*i+2]  # radius is third component
            return -total_radius
        
        # Flatten best result into parameter vector
        initial_params = []
        for i in range(N_CIRCLES):
            initial_params.extend([best_result[i][0], best_result[i][1], best_result[i][2]])
        
        # Get constraints
        constraints = get_constraints()
        
        # Optimize using SLSQP method which handles constraints well
        try:
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6},
                tol=1e-6
            )
            
            # Extract optimized results
            if result.success:
                optimized_circles = np.zeros((N_CIRCLES, 3))
                for i in range(N_CIRCLES):
                    optimized_circles[i] = [
                        result.x[3*i],   # x coordinate
                        result.x[3*i+1], # y coordinate
                        result.x[3*i+2]  # radius
                    ]
                # Recalculate sum
                current_sum = evaluate_objective(optimized_circles)
                if current_sum > best_sum:
                    best_result = optimized_circles
        except Exception:
            pass
    
    # Final local improvement after any optimization
    if best_result is not None:
        remaining_time = MAX_TIME - (time.time() - start_time) - 2.0  # Leave 2s for final steps
        if remaining_time > 0:
            best_result = local_improvement(best_result, max_iter=20, time_limit=remaining_time)
    
    # Final validation
    if best_result is None:
        best_result = initialize_hexagonal_grid()
    
    return best_result


# EVOLVE-BLOCK-END
