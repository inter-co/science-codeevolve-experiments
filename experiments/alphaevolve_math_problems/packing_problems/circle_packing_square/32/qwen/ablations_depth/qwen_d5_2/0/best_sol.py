# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import math
from numba import jit

@jit(nopython=True)
def check_collision_fast(circles, i, j):
    """Fast collision check using Numba"""
    x1, y1, r1 = circles[i]
    x2, y2, r2 = circles[j]
    dx = x1 - x2
    dy = y1 - y2
    dist_sq = dx*dx + dy*dy
    return dist_sq < (r1 + r2)*(r1 + r2)

@jit(nopython=True)
def compute_total_radius_fast(circles):
    """Fast computation of total radius sum"""
    total = 0.0
    for i in range(len(circles)):
        total += circles[i][2]
    return total

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining Voronoi-based initialization with simulated annealing.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Voronoi-based initialization
    # Generate points using a more sophisticated approach
    np.random.seed(42)  # For reproducibility
    
    # Start with a good distribution of points
    points = []
    # Create a grid with some randomness to avoid regular patterns
    grid_size = int(np.ceil(np.sqrt(n)))
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) >= n:
                break
            x = (j + 0.5 + np.random.uniform(-0.2, 0.2)) / grid_size
            y = (i + 0.5 + np.random.uniform(-0.2, 0.2)) / grid_size
            # Keep only points inside the unit square
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
        if len(points) >= n:
            break
    
    # Trim to exact number needed
    points = points[:n]
    
    # Phase 2: Initialize with reasonable radii
    circles = np.zeros((n, 3))
    for i, (x, y) in enumerate(points):
        # Estimate initial radius based on proximity to edges and neighbors
        min_dist_to_edge = min(x, 1-x, y, 1-y)
        # Estimate neighbor distances
        min_neighbor_dist = float('inf')
        for j, (x2, y2) in enumerate(points):
            if i != j:
                dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                min_neighbor_dist = min(min_neighbor_dist, dist)
        
        # Set radius conservatively
        if min_neighbor_dist < 0.1:
            radius = min(0.1, min_dist_to_edge * 0.5)
        else:
            radius = min(0.15, min_dist_to_edge * 0.4)
        
        circles[i] = [x, y, radius]
    
    # Phase 3: Simulated Annealing optimization
    def evaluate_fitness(circles):
        """Evaluate fitness: negative sum of radii (negative because we minimize)"""
        total_radius = 0.0
        penalty = 0.0
        
        # Check collisions and boundaries
        for i in range(n):
            x, y, r = circles[i]
            
            # Boundary penalty
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 1000.0
            
            # Collision penalties
            for j in range(i+1, n):
                x2, y2, r2 = circles[j]
                dx = x - x2
                dy = y - y2
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r + r2) * (r + r2)
                
                if dist_sq < min_dist_sq:
                    # Penalty proportional to how much they overlap
                    overlap = min_dist_sq - dist_sq
                    penalty += overlap * 100.0
        
        total_radius = sum(circles[:, 2])
        return -total_radius + penalty  # Negative because we want to minimize
    
    def perturb_solution(circles, temperature):
        """Create a small perturbation to the solution"""
        new_circles = circles.copy()
        
        # Pick a random circle to perturb
        idx = np.random.randint(n)
        
        # Randomly choose what to perturb
        choice = np.random.randint(3)
        
        if choice == 0:  # Perturb x coordinate
            new_circles[idx, 0] += np.random.normal(0, temperature * 0.01)
            new_circles[idx, 0] = np.clip(new_circles[idx, 0], 0, 1)
        elif choice == 1:  # Perturb y coordinate
            new_circles[idx, 1] += np.random.normal(0, temperature * 0.01)
            new_circles[idx, 1] = np.clip(new_circles[idx, 1], 0, 1)
        else:  # Perturb radius
            new_circles[idx, 2] += np.random.normal(0, temperature * 0.005)
            # Keep radius positive and reasonable
            new_circles[idx, 2] = max(0.001, min(0.4, new_circles[idx, 2]))
        
        return new_circles
    
    # Simulated Annealing
    current_solution = circles.copy()
    best_solution = circles.copy()
    current_fitness = evaluate_fitness(current_solution)
    best_fitness = current_fitness
    
    # Annealing parameters
    temperature = 0.1
    cooling_rate = 0.999
    min_temperature = 1e-6
    max_iterations = 10000
    
    for iteration in range(max_iterations):
        # Cool down temperature
        if temperature < min_temperature:
            break
            
        # Generate neighbor solution
        new_solution = perturb_solution(current_solution, temperature)
        new_fitness = evaluate_fitness(new_solution)
        
        # Accept or reject based on Metropolis criterion
        if new_fitness < current_fitness or np.random.rand() < np.exp(-(new_fitness - current_fitness) / temperature):
            current_solution = new_solution
            current_fitness = new_fitness
            
            # Update best solution
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        # Cool down
        temperature *= cooling_rate
    
    # Final refinement using scipy optimization for better local search
    try:
        # Convert to flat array for scipy optimization
        x0 = best_solution.flatten()
        
        # Objective function: negative sum of radii
        def objective(x_flat):
            circles_flat = x_flat.reshape(-1, 3)
            return -np.sum(circles_flat[:, 2])
        
        # Constraints
        def boundary_constraint(i):
            def con(x):
                x_c, y_c, r = x[3*i], x[3*i+1], x[3*i+2]
                return min(r, 1-r-x_c, 1-r-y_c, x_c-r, y_c-r)
            return {'type': 'ineq', 'fun': con}
        
        def overlap_constraint(i, j):
            def con(x):
                x_i, y_i, r_i = x[3*i], x[3*i+1], x[3*i+2]
                x_j, y_j, r_j = x[3*j], x[3*j+1], x[3*j+2]
                dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                return dist - (r_i + r_j)
            return {'type': 'ineq', 'fun': con}
        
        constraints = []
        for i in range(n):
            constraints.append(boundary_constraint(i))
        for i in range(n):
            for j in range(i+1, n):
                constraints.append(overlap_constraint(i, j))
        
        # Bounds: x,y in [0.01, 0.99], r in [0.001, 0.4]
        bounds = []
        for i in range(n):
            bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.001, 0.4)])
        
        # Optimize with SLSQP
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-6})
        
        if result.success:
            refined = result.x.reshape(-1, 3)
            return refined
    except:
        # If optimization fails, return the best solution found so far
        pass
    
    return best_solution


# EVOLVE-BLOCK-END
