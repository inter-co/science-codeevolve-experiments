# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, force relaxation, and mathematical programming.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    np.random.seed(42)  # For reproducibility
    
    # Create better initial configuration using hexagonal packing
    def create_hexagonal_initial():
        # Try 6x6 hexagonal grid which gives us 36 positions for 32 circles
        rows, cols = 6, 6
        spacing_x = 0.9 / cols  # Leave 0.1 margin on each side
        spacing_y = 0.9 / rows
        
        circles = []
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                # Hexagonal offset for odd rows
                x_offset = 0.05 + (j * spacing_x) + (0.5 if i % 2 == 1 else 0)
                y_offset = 0.05 + (i * spacing_y)
                
                # Estimate radius based on available space
                radius = min(
                    spacing_x / 2.0,
                    spacing_y / 2.0,
                    x_offset, y_offset, 1 - x_offset, 1 - y_offset
                ) * 0.9  # Slightly conservative
                
                if radius > 0.005:
                    circles.append([x_offset, y_offset, radius])
        
        # If we didn't get enough circles, fill with random positions
        while len(circles) < n:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            # Estimate reasonable radius
            radius = min(x, y, 1-x, 1-y) * 0.4
            if radius > 0.005:
                circles.append([x, y, radius])
        
        return np.array(circles[:n])
    
    # Force relaxation to resolve overlaps
    def force_relaxation(circles, iterations=100):
        # Make a copy to avoid modifying original
        config = circles.copy()
        
        for _ in range(iterations):
            # Calculate forces between all pairs
            forces = np.zeros_like(config[:, :2])
            
            # Vectorized computation of distances and forces
            positions = config[:, :2]
            radii = config[:, 2]
            
            # Compute pairwise distances
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            
            # Compute overlap forces
            overlap_mask = distances < (radii[:, np.newaxis] + radii[np.newaxis, :])
            np.fill_diagonal(overlap_mask, False)  # No self-interaction
            
            # Apply forces
            for i in range(n):
                overlap_indices = np.where(overlap_mask[i])[0]
                for j in overlap_indices:
                    if i != j:
                        dist = distances[i, j]
                        if dist > 1e-10:
                            direction = positions[i] - positions[j]
                            direction /= dist
                            overlap = (radii[i] + radii[j]) - dist
                            forces[i] += direction * overlap * 0.1
            
            # Boundary forces
            for i in range(n):
                # Left boundary
                if config[i, 0] - config[i, 2] < 0:
                    forces[i, 0] += (config[i, 2] - config[i, 0]) * 0.5
                # Right boundary
                if config[i, 0] + config[i, 2] > 1:
                    forces[i, 0] -= (config[i, 0] + config[i, 2] - 1) * 0.5
                # Bottom boundary
                if config[i, 1] - config[i, 2] < 0:
                    forces[i, 1] += (config[i, 2] - config[i, 1]) * 0.5
                # Top boundary
                if config[i, 1] + config[i, 2] > 1:
                    forces[i, 1] -= (config[i, 1] + config[i, 2] - 1) * 0.5
            
            # Apply forces
            config[:, :2] += forces * 0.001
            
            # Enforce bounds
            for i in range(n):
                config[i, 0] = np.clip(config[i, 0], config[i, 2], 1 - config[i, 2])
                config[i, 1] = np.clip(config[i, 1], config[i, 2], 1 - config[i, 2])
        
        return config
    
    # Mathematical optimization using scipy
    def optimize_with_scipy(circles):
        # Create flat parameter vector [x1, y1, r1, x2, y2, r2, ...]
        x0 = []
        for i in range(n):
            x0.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
        
        # Objective function: minimize negative sum of radii
        def objective(params):
            total_radii = sum(params[3*i+2] for i in range(n))
            return -total_radii
        
        # Constraint functions
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
        
        # Optimize
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                             options={'maxiter': 200, 'ftol': 1e-6})
            
            if result.success:
                # Update circles with optimized values
                for i in range(n):
                    circles[i, 0] = result.x[3*i]
                    circles[i, 1] = result.x[3*i+1]
                    circles[i, 2] = result.x[3*i+2]
        except Exception:
            # If optimization fails, continue with current configuration
            pass
        
        return circles
    
    # Main execution
    # Phase 1: Create initial configuration
    circles = create_hexagonal_initial()
    
    # Phase 2: Force relaxation to resolve overlaps
    circles = force_relaxation(circles)
    
    # Phase 3: Mathematical optimization
    circles = optimize_with_scipy(circles)
    
    # Phase 4: Final force relaxation for fine tuning
    circles = force_relaxation(circles, iterations=50)
    
    return circles


# EVOLVE-BLOCK-END
