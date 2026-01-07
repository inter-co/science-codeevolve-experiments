# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
import random

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining advanced initialization, force relaxation, and 
    mathematical programming optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Phase 1: Advanced initialization with hexagonal packing
    def create_initial_configuration():
        # Create a more sophisticated initial layout
        # Use a hexagonal packing pattern with better spacing
        rows = 6
        cols = 6
        
        circles = []
        
        # Generate hexagonal lattice with slight randomness
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Hexagonal offset for even rows
                x_offset = 0.1 + (j * 0.8 / (cols - 1)) + np.random.uniform(-0.015, 0.015)
                y_offset = 0.1 + (i * 0.8 / (rows - 1)) + np.random.uniform(-0.015, 0.015)
                if i % 2 == 1:  # Offset odd rows
                    x_offset += 0.4 / (cols - 1)
                circles.append([x_offset, y_offset, 0.03])
        
        # Fill remaining slots with random positions ensuring they fit
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Set radius conservatively based on distance to boundaries
            r = min(x, y, 1-x, 1-y) * 0.8
            r = max(0.005, min(0.2, r))
            circles.append([x, y, r])
            
        return np.array(circles[:n])
    
    # Phase 2: Force relaxation with vectorized operations
    def force_relaxation(circles, max_iter=300):
        # More efficient force relaxation with vectorized operations
        prev_sum = -1
        for iteration in range(max_iter):
            positions = circles[:, :2]
            
            # Vectorized distance computation
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            
            # Initialize forces
            forces = np.zeros_like(positions)
            
            # Vectorized repulsion forces
            overlap_mask = distances < (circles[:, 2] + circles[:, 2][:, np.newaxis])
            np.fill_diagonal(overlap_mask, False)
            
            # Apply forces more efficiently
            for i in range(n):
                overlap_indices = np.where(overlap_mask[i])[0]
                for j in overlap_indices:
                    if i != j:
                        dist = distances[i, j]
                        if dist > 1e-10:
                            direction = positions[i] - positions[j]
                            direction /= dist
                            overlap = (circles[i, 2] + circles[j, 2]) - dist
                            # Use stronger force for better overlap resolution
                            forces[i] += direction * overlap * 0.5
                            forces[j] -= direction * overlap * 0.5
            
            # Boundary forces with stronger enforcement
            for i in range(n):
                # Left boundary
                if circles[i, 0] - circles[i, 2] < 0:
                    forces[i, 0] += (circles[i, 2] - circles[i, 0]) * 2.0
                # Right boundary
                if circles[i, 0] + circles[i, 2] > 1:
                    forces[i, 0] -= (circles[i, 0] + circles[i, 2] - 1) * 2.0
                # Bottom boundary
                if circles[i, 1] - circles[i, 2] < 0:
                    forces[i, 1] += (circles[i, 2] - circles[i, 1]) * 2.0
                # Top boundary
                if circles[i, 1] + circles[i, 2] > 1:
                    forces[i, 1] -= (circles[i, 1] + circles[i, 2] - 1) * 2.0
            
            # Update positions with adaptive step size
            step_size = 0.002
            circles[:, :2] += forces * step_size
            
            # Ensure circles stay within bounds
            for i in range(n):
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
            
            # Early stopping based on convergence
            current_sum = np.sum(circles[:, 2])
            if abs(current_sum - prev_sum) < 1e-6 and iteration > 50:
                break
            prev_sum = current_sum
        
        return circles
    
    # Phase 3: Mathematical optimization using scipy
    def mathematical_optimization(circles):
        try:
            # Create flat parameter vector [x1, y1, r1, x2, y2, r2, ...]
            x0 = []
            for i in range(n):
                x0.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
            
            # Objective function: minimize negative sum of radii
            def objective(params):
                total_radii = sum(params[3*i+2] for i in range(n))
                return -total_radii
            
            # Constraints for scipy optimization
            def constraint_overlap(params, i, j):
                x1, y1, r1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, r2 = params[3*j], params[3*j+1], params[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                radii_sum = r1 + r2
                return dist_sq - radii_sum**2  # Should be >= 0 for no overlap
            
            def constraint_bound(params, i):
                x, y, r = params[3*i], params[3*i+1], params[3*i+2]
                # Circle must fit in square
                left = x - r
                right = x + r
                bottom = y - r
                top = y + r
                return min(left, 1-right, bottom, 1-top)  # >= 0 for valid placement
            
            # Build constraints
            constraints = []
            # Add boundary constraints
            for i in range(n):
                constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: constraint_bound(p, idx)})
            
            # Add overlap constraints
            for i in range(n):
                for j in range(i+1, n):
                    constraints.append({'type': 'ineq', 'fun': lambda p, idx1=i, idx2=j: constraint_overlap(p, idx1, idx2)})
            
            # Bounds: x, y in [0.05, 0.95], radius in [0.01, 0.45]
            bounds = []
            for i in range(n):
                bounds.extend([(0.05, 0.95), (0.05, 0.95), (0.01, 0.45)])
            
            # Optimize using SLSQP
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'maxiter': 200, 'ftol': 1e-6, 'eps': 1e-6})
            
            if result.success:
                # Update circles with optimized values
                for i in range(n):
                    circles[i, 0] = result.x[3*i]
                    circles[i, 1] = result.x[3*i+1]
                    circles[i, 2] = result.x[3*i+2]
        except Exception as e:
            # If optimization fails, continue with current configuration
            pass
        
        return circles
    
    # Phase 4: Local refinement with greedy improvements
    def local_refinement(circles):
        # Try to improve each circle's radius greedily
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Try increasing each circle's radius one by one
            for i in range(n):
                old_radius = circles[i, 2]
                # Try different increment sizes
                increments = [0.005, 0.003, 0.001]
                for inc in increments:
                    test_radius = min(old_radius + inc, 0.5)
                    
                    # Check if we can increase this radius without violating constraints
                    valid = True
                    for j in range(n):
                        if i != j:
                            dist = np.sqrt((circles[i, 0] - circles[j, 0])**2 + 
                                         (circles[i, 1] - circles[j, 1])**2)
                            if dist < test_radius + circles[j, 2]:
                                valid = False
                                break
                    
                    # Also check boundary constraints
                    if (valid and circles[i, 0] - test_radius >= 0 and 
                        circles[i, 0] + test_radius <= 1 and 
                        circles[i, 1] - test_radius >= 0 and 
                        circles[i, 1] + test_radius <= 1):
                        circles[i, 2] = test_radius
                        improved = True
                        break
        
        return circles
    
    # Main execution
    start_time = time.time()
    
    # Create initial configuration
    circles = create_initial_configuration()
    
    # Apply force relaxation to resolve initial overlaps
    circles = force_relaxation(circles)
    
    # Apply mathematical optimization for better arrangement
    circles = mathematical_optimization(circles)
    
    # Apply local refinement for small improvements
    circles = local_refinement(circles)
    
    # Apply final force relaxation to clean up
    circles = force_relaxation(circles, max_iter=100)
    
    end_time = time.time()
    
    return circles


# EVOLVE-BLOCK-END
