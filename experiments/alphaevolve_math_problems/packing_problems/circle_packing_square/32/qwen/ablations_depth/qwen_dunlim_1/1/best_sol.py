# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and efficient optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Initialize using a better geometric approach inspired by known circle packings
    def initialize_better_layout():
        # Use a more systematic approach that's likely to yield good results
        circles = []
        
        # Start with a hexagonal pattern in the center (more efficient packing)
        center_x, center_y = 0.5, 0.5
        radius = 0.12  # Starting radius
        
        # Place circles in a hexagonal pattern around center
        angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
        for i, angle in enumerate(angles):
            if len(circles) >= n:
                break
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Ensure we stay within bounds and have reasonable initial radius
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius based on distance from edges
            r = min(x, 1-x, y, 1-y) * 0.35
            r = max(0.01, min(0.4, r))
            
            circles.append([x, y, r])
        
        # Fill remaining positions with a grid pattern in the outer region
        grid_size = int(np.ceil(np.sqrt(n - len(circles))))
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(circles) >= n:
                    break
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                
                # Initial radius based on distance from edges
                r = min(x, 1-x, y, 1-y) * 0.25
                r = max(0.01, min(0.4, r))
                
                circles.append([x, y, r])
        
        # Fill remaining positions with random placement but with better distribution
        while len(circles) < n:
            # Place in a way that tries to avoid dense areas
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            
            # Initial radius based on distance from edges
            r = min(x, 1-x, y, 1-y) * 0.3
            r = max(0.01, min(0.4, r))
            
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Create initial configuration
    circles = initialize_better_layout()
    
    # More efficient constraint checking using vectorized operations
    def check_containment(circles):
        """Check if all circles are contained within unit square"""
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        return np.all((r <= x) & (x <= 1-r) & (r <= y) & (y <= 1-r))
    
    def check_overlaps(circles):
        """Check if any circles overlap using efficient vectorized approach"""
        if len(circles) < 2:
            return True
            
        # Vectorized distance computation
        centers = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute all pairwise distances
        distances = cdist(centers, centers)
        
        # Compute sum of radii for each pair
        sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Set diagonal to infinity to ignore self-distances
        np.fill_diagonal(distances, np.inf)
        
        # Check if any distance is less than sum of radii
        return np.all(distances >= sum_radii)
    
    # Objective function for optimization
    def objective_function(circles_flat):
        """Objective function to maximize sum of radii"""
        circles = circles_flat.reshape(-1, 3)
        return -np.sum(circles[:, 2])  # Negative because we minimize
    
    # Constraint functions using vectorized operations - optimized version
    def constraint_func(circles_flat):
        """Combined constraint function for scipy optimization"""
        circles = circles_flat.reshape(-1, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Containment constraints: r <= x <= 1-r and r <= y <= 1-r
        # This means: x >= r, x <= 1-r, y >= r, y <= 1-r
        # Return negative values for constraint violations (we want >= 0)
        containment = np.concatenate([
            x - r,                    # x >= r
            1 - x - r,               # x <= 1-r
            y - r,                    # y >= r
            1 - y - r                 # y <= 1-r
        ])
        
        # Overlap constraints using vectorized operations
        centers = circles[:, :2]
        radii = circles[:, 2]
        
        # Compute all pairwise distances efficiently
        distances = cdist(centers, centers)
        
        # Compute sum of radii for each pair
        sum_radii = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Extract upper triangular part (excluding diagonal) for overlap constraints
        # This avoids double counting constraints
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        overlap_dists = distances[mask]
        overlap_radii = sum_radii[mask]
        
        # Constraint is dist >= sum_rad, so we return dist - sum_rad
        overlap = overlap_dists - overlap_radii
        
        return np.concatenate([containment, overlap])
    
    # Bounds for variables: [x, y, r] for each circle
    bounds = []
    for i in range(n):
        # x coordinate bounds
        bounds.append((0, 1))
        # y coordinate bounds  
        bounds.append((0, 1))
        # radius bounds (must be positive and respect containment)
        bounds.append((0.001, 0.5))  # Min radius 0.001 to prevent degenerate cases
    
    # Try multiple optimization approaches with better parameters
    best_solution = None
    best_sum = float('-inf')
    
    print("Starting optimization...")
    
    # Approach 1: Direct optimization with fewer but more strategic restarts
    for restart in range(8):  # Fewer restarts to keep within time limit
        # Start with current best or random perturbation
        if restart == 0:
            # Use initial configuration
            x0 = circles.flatten()
        else:
            # Perturb previous solution more carefully
            perturbed = circles.copy()
            for i in range(n):
                # Adaptive perturbation based on iteration
                perturbation_scale = 0.025 if restart < 4 else 0.015
                perturbed[i, 0] += np.random.normal(0, perturbation_scale)
                perturbed[i, 1] += np.random.normal(0, perturbation_scale)
                # Keep within bounds
                perturbed[i, 0] = max(0.01, min(0.99, perturbed[i, 0]))
                perturbed[i, 1] = max(0.01, min(0.99, perturbed[i, 1]))
                
                # Perturb radius
                perturbed[i, 2] += np.random.normal(0, perturbation_scale)
                perturbed[i, 2] = max(0.001, min(0.49, perturbed[i, 2]))
            
            x0 = perturbed.flatten()
        
        # Optimization with bounds and constraints
        try:
            # Focus on just one reliable method for speed
            result = minimize(
                objective_function,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 600, 'ftol': 1e-6}
            )
            
            if result.success:
                current_sum = -result.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_solution = result.x
                    print(f"Improved solution found: {best_sum:.6f}")
                    
            # Early stopping if we're close to benchmark
            if best_sum > 2.93:  # Early stopping if we're very close to target
                break
                
        except Exception as e:
            continue  # Skip this restart if optimization fails
    
    # If no good solution from local optimization, use the initial one
    if best_solution is None:
        best_solution = circles.flatten()
    
    # Convert back to circles format
    final_circles = best_solution.reshape(-1, 3)
    
    # Final validation and cleanup
    if not check_containment(final_circles):
        print("Warning: Final solution violates containment constraints")
    
    if not check_overlaps(final_circles):
        print("Warning: Final solution has overlaps")
    
    # Final cleanup to ensure constraints are met
    for i in range(n):
        x, y, r = final_circles[i]
        # Ensure containment
        final_circles[i, 0] = max(r, min(1-r, x))
        final_circles[i, 1] = max(r, min(1-r, y))
        final_circles[i, 2] = max(0.001, min(0.49, r))
    
    return final_circles


# EVOLVE-BLOCK-END
