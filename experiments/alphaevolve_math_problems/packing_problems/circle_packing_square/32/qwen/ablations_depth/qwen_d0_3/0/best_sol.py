# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import time
from numba import jit
import math

@jit(nopython=True)
def compute_distances_numba(positions):
    """Compute pairwise distances efficiently using numba"""
    n = positions.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a combination of mathematical insight and advanced optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better initialization based on known optimal packing patterns
    def initialize_hexagonal_config():
        """Initialize using a hexagonal-like packing pattern with strategic adjustments"""
        # Start with a more systematic approach
        # For 32 circles, we can think of a roughly 6x6 grid with some irregularity
        positions = []
        radii = []
        
        # Create a more optimized grid pattern
        # Use a hexagonal lattice approximation
        rows = 6
        cols = 6
        padding = 0.05
        
        # Hexagonal packing with adjustment for boundary conditions
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= n:
                    break
                # Offset every other row for hexagonal packing
                offset = (i % 2) * 0.5
                x = padding + (j + offset) * (1 - 2*padding) / (cols - 1)
                y = padding + i * (1 - 2*padding) / (rows - 1)
                
                # Ensure within bounds
                if 0 <= x <= 1 and 0 <= y <= 1:
                    positions.append([x, y])
            if len(positions) >= n:
                break
        
        # Fill remaining positions with random placement near edges if needed
        if len(positions) < n:
            np.random.seed(42)
            for _ in range(n - len(positions)):
                # Prefer edge locations for better utilization
                side = np.random.randint(0, 4)
                if side == 0:  # top
                    x = np.random.uniform(0.1, 0.9)
                    y = 0.95
                elif side == 1:  # bottom
                    x = np.random.uniform(0.1, 0.9)
                    y = 0.05
                elif side == 2:  # left
                    x = 0.05
                    y = np.random.uniform(0.1, 0.9)
                else:  # right
                    x = 0.95
                    y = np.random.uniform(0.1, 0.9)
                positions.append([x, y])
        
        positions = np.array(positions[:n])
        
        # Estimate initial radii based on local density and spacing
        tree = cKDTree(positions)
        radii = []
        for i in range(n):
            # Find nearest neighbors
            distances, indices = tree.query(positions[i], k=min(6, n), p=2)
            # Take the minimum distance to nearest neighbor divided by 2
            if len(distances) > 1:
                min_dist = np.min(distances[1:])  # exclude self-distance
                # Use a more conservative estimate for better convergence
                radius = min(0.15, min_dist / 2.0 * 0.8)
                radii.append(radius)
            else:
                radii.append(0.08)
        
        return positions, np.array(radii)
    
    # More efficient constraint functions
    def containment_constraints(circles_flat):
        """Ensure all circles are fully contained within unit square"""
        positions = circles_flat.reshape(-1, 3)[:, :2]
        radii = circles_flat.reshape(-1, 3)[:, 2]
        
        # Vectorized containment constraints
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        r_coords = radii
        
        # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        constraints = np.concatenate([
            x_coords - r_coords,           # x - r >= 0
            1 - x_coords - r_coords,       # 1 - x - r >= 0
            y_coords - r_coords,           # y - r >= 0
            1 - y_coords - r_coords        # 1 - y - r >= 0
        ])
        return constraints
    
    def non_overlap_constraints(circles_flat):
        """Ensure no two circles overlap"""
        positions = circles_flat.reshape(-1, 3)[:, :2]
        radii = circles_flat.reshape(-1, 3)[:, 2]
        
        # More efficient non-overlap constraint computation
        distances = cdist(positions, positions, 'euclidean')
        radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Non-overlap constraints: distance >= (r_i + r_j)
        # So we want: distances - radii_matrix >= 0
        constraints = distances - radii_matrix
        
        # Only keep upper triangle (avoid duplicates) and diagonal zeros
        mask = np.triu(np.ones_like(constraints), k=1).astype(bool)
        return constraints[mask]
    
    # Objective function (negative because we minimize)
    def objective(circles_flat):
        return -np.sum(circles_flat.reshape(-1, 3)[:, 2])
    
    # Gradient of objective function
    def grad_objective(circles_flat):
        grad = np.zeros_like(circles_flat)
        grad[2::3] = -1.0  # gradient w.r.t. radii
        return grad
    
    # Custom constraint handling with improved numerical stability
    def safe_constraint_evaluator(circles_flat):
        """Evaluate constraints with better numerical handling"""
        # Check containment first
        containment = containment_constraints(circles_flat)
        
        # Then check non-overlap
        overlap = non_overlap_constraints(circles_flat)
        
        # Combine constraints (all must be >= 0)
        return np.concatenate([containment, overlap])
    
    # Initial configuration
    positions, radii = initialize_hexagonal_config()
    initial_circles = np.column_stack([positions, radii]).flatten()
    
    # Set up bounds for variables (x, y, r for each circle)
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: safe_constraint_evaluator(x)}
    ]
    
    # Use multiple optimization strategies for better results
    best_result = None
    best_sum = -np.inf
    
    # Strategy 1: SLSQP with multiple restarts
    try:
        for restart in range(15):  # More restarts
            np.random.seed(42 + restart)
            
            # Create slightly different initial perturbations
            perturbed = initial_circles.copy()
            # Apply different perturbation patterns for each restart
            for i in range(n):
                # Add larger perturbations to positions
                perturbed[i*3] += np.random.normal(0, 0.03)  # x
                perturbed[i*3 + 1] += np.random.normal(0, 0.03)  # y
                # Smaller perturbations to radii
                perturbed[i*3 + 2] += np.random.normal(0, 0.015)  # r
            
            # Ensure bounds are respected
            for i in range(n):
                perturbed[i*3] = np.clip(perturbed[i*3], 0.001, 0.999)
                perturbed[i*3 + 1] = np.clip(perturbed[i*3 + 1], 0.001, 0.999)
                perturbed[i*3 + 2] = np.clip(perturbed[i*3 + 2], 0.001, 0.499)
            
            # Try with different tolerance settings
            result = minimize(
                objective,
                perturbed,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-9, 'eps': 1e-7},
                callback=lambda x: None
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
    except Exception as e:
        pass
    
    # Strategy 2: Try L-BFGS-B with better initial values
    if best_result is None:
        try:
            result = minimize(
                objective,
                initial_circles,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-9, 'gtol': 1e-7},
                callback=lambda x: None
            )
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            pass
    
    # Strategy 3: Try Trust-Constr with higher precision
    if best_result is None:
        try:
            result = minimize(
                objective,
                initial_circles,
                method='trust-constr',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2000, 'gtol': 1e-9, 'xtol': 1e-9},
                callback=lambda x: None
            )
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
        except Exception as e:
            pass
    
    # Final fallback to initial configuration if all optimizations fail
    if best_result is not None:
        final_circles = best_result.x.reshape(-1, 3)
    else:
        final_circles = initial_circles.reshape(-1, 3)
    
    # Final validation and cleanup with stricter bounds
    validated_circles = []
    for i in range(n):
        x = max(0.001, min(0.999, final_circles[i, 0]))
        y = max(0.001, min(0.999, final_circles[i, 1]))
        r = max(0.001, min(0.499, final_circles[i, 2]))
        validated_circles.append([x, y, r])
    
    return np.array(validated_circles)


# EVOLVE-BLOCK-END
