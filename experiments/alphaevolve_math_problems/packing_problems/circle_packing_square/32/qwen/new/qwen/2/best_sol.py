# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import random

# Global constants for the optimization
MAX_ITER = 500
TOL = 1e-6

def objective(circles):
    """Objective function to maximize sum of radii"""
    return -np.sum(circles[:, 2])  # Negative because we minimize

def constraint_containment(circles):
    """Constraint function for containment: all radii must be <= distance to edges"""
    n = len(circles)
    cons = []
    for i in range(n):
        x, y, r = circles[i]
        # x >= r, 1-x >= r, y >= r, 1-y >= r
        cons.append(x - r)      # x >= r
        cons.append(1 - x - r)  # x <= 1-r
        cons.append(y - r)      # y >= r
        cons.append(1 - y - r)  # y <= 1-r
    return np.array(cons)

def constraint_overlap(circles):
    """Constraint function for non-overlap: distance >= sum of radii"""
    n = len(circles)
    cons = []
    
    # Compute all pairwise distances
    centers = circles[:, :2]
    distances = cdist(centers, centers)
    
    # Check all pairs
    for i in range(n):
        for j in range(i+1, n):
            dist = distances[i, j]
            r_i, r_j = circles[i, 2], circles[j, 2]
            # We want dist >= r_i + r_j, so we add constraint: dist - r_i - r_j >= 0
            cons.append(dist - r_i - r_j)
    
    return np.array(cons)

def generate_hexagonal_initial():
    """Generate initial configuration using hexagonal packing pattern"""
    n = 32
    circles = []
    
    # Hexagonal grid pattern with better distribution
    rows = 6
    cols = 6
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Add hexagonal offset for odd rows
            if i % 2 == 1:
                x += spacing_x / 2
                
            # Add small random jitter to avoid perfect grid
            x += (random.random() - 0.5) * spacing_x * 0.2
            y += (random.random() - 0.5) * spacing_y * 0.2
            
            # Keep within bounds
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius - small enough to fit
            r = min(spacing_x, spacing_y) * 0.3
            circles.append([x, y, r])
    
    # Fill remaining slots with random positions
    while len(circles) < n:
        x = 0.05 + random.random() * 0.9
        y = 0.05 + random.random() * 0.9
        r = 0.02 + random.random() * 0.1
        circles.append([x, y, r])
    
    return np.array(circles[:n])

def optimize_circles(circles):
    """Refine the circle configuration using optimization"""
    n = len(circles)
    
    # Flatten the circles array for optimization
    initial_vars = circles.flatten()
    
    # Define bounds for variables: x, y, r for each circle
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r
    
    # Define constraints as a single function
    def combined_constraints(vars):
        # Reconstruct circles from flattened vars
        reconstructed = vars.reshape(-1, 3)
        
        # Check containment constraints
        containment = constraint_containment(reconstructed)
        
        # Check overlap constraints  
        overlap = constraint_overlap(reconstructed)
        
        return np.concatenate([containment, overlap])
    
    # Optimization using SLSQP - much more effective than differential evolution
    try:
        result = minimize(
            lambda x: objective(x.reshape(-1, 3)),
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': combined_constraints},
            options={'maxiter': MAX_ITER, 'ftol': TOL, 'disp': False}
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            return optimized_circles
    except Exception as e:
        pass
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    
    # Try multiple initial configurations and pick the best
    best_circles = None
    best_sum = 0
    
    # Try different initialization strategies
    initial_configs = [
        generate_hexagonal_initial,
        lambda: generate_hexagonal_initial() + np.random.normal(0, 0.02, (32, 3)),
        lambda: generate_hexagonal_initial() * 0.9 + np.random.uniform(0.05, 0.1, (32, 3))
    ]
    
    for i, init_func in enumerate(initial_configs):
        try:
            circles = init_func()
            refined = optimize_circles(circles)
            current_sum = np.sum(refined[:, 2])
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = refined.copy()
        except Exception as e:
            continue
    
    # If no optimization succeeded, fall back to the hexagonal pattern
    if best_circles is None:
        best_circles = generate_hexagonal_initial()
    
    # Final local refinement
    for _ in range(3):
        # Small perturbations
        perturbed = best_circles.copy()
        for i in range(len(perturbed)):
            if np.random.rand() < 0.2:  # 20% chance to perturb
                perturbed[i, 0] += np.random.normal(0, 0.005)
                perturbed[i, 1] += np.random.normal(0, 0.005)
                perturbed[i, 2] += np.random.normal(0, 0.002)
                
                # Keep within bounds
                perturbed[i, 0] = np.clip(perturbed[i, 0], 0.001, 0.999)
                perturbed[i, 1] = np.clip(perturbed[i, 1], 0.001, 0.999)
                perturbed[i, 2] = np.clip(perturbed[i, 2], 0.001, 0.499)
        
        # Try to optimize the perturbed version
        try:
            optimized = optimize_circles(perturbed)
            current_sum = np.sum(optimized[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized
        except:
            continue
    
    print(f"Final solution sum of radii: {best_sum:.6f}")
    return best_circles


# EVOLVE-BLOCK-END
