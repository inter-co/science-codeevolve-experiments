# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric intuition with optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    np.random.seed(42)
    n = 16
    
    # Start with a good initial configuration - hexagonal packing pattern
    # Arrange points in a grid-like pattern but with some perturbation
    points = np.zeros((n, 2))
    
    # Create a roughly hexagonal lattice pattern
    rows = 4
    cols = 4
    spacing = 1.0 / (cols - 1) if cols > 1 else 1.0
    offset = spacing * 0.5
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx < n:
                x = j * spacing
                y = i * spacing
                # Add slight offset for odd rows to create hexagonal structure
                if i % 2 == 1:
                    x += offset
                points[idx] = [x, y]
                idx += 1
    
    # Ensure points are within bounds [0,1] x [0,1]
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    points[:, 1] = np.clip(points[:, 1], 0, 1)
    
    # Apply optimization using gradient-based method
    # We'll use a simple gradient ascent approach with constraints
    def compute_min_max_ratio(pts):
        """Compute the ratio of minimum to maximum distances"""
        if len(pts) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = pdist(pts)
        if len(distances) == 0:
            return 0.0
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        if d_max <= 0:
            return 0.0
            
        return d_min / d_max
    
    # Simple optimization loop
    best_ratio = compute_min_max_ratio(points)
    best_points = points.copy()
    
    # Gradient-based optimization with constraint handling
    for iteration in range(1000):
        # Compute current distances
        distances = pdist(points)
        if len(distances) == 0:
            break
            
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Compute gradients using finite differences
        grad_x = np.zeros(n)
        grad_y = np.zeros(n)
        
        # For each point, compute how moving it affects the min/max ratio
        for i in range(n):
            # Small perturbations
            eps = 1e-6
            orig_x, orig_y = points[i, 0], points[i, 1]
            
            # Check effect of small moves in x direction
            points[i, 0] = orig_x + eps
            new_ratio_x = compute_min_max_ratio(points)
            points[i, 0] = orig_x - eps
            new_ratio_neg_x = compute_min_max_ratio(points)
            points[i, 0] = orig_x  # restore
            
            # Check effect of small moves in y direction  
            points[i, 1] = orig_y + eps
            new_ratio_y = compute_min_max_ratio(points)
            points[i, 1] = orig_y - eps
            new_ratio_neg_y = compute_min_max_ratio(points)
            points[i, 1] = orig_y  # restore
            
            # Estimate gradients
            grad_x[i] = (new_ratio_x - new_ratio_neg_x) / (2 * eps)
            grad_y[i] = (new_ratio_y - new_ratio_neg_y) / (2 * eps)
        
        # Update points with gradient ascent (but keep within bounds)
        learning_rate = 0.01
        for i in range(n):
            # Apply gradient update
            new_x = points[i, 0] + learning_rate * grad_x[i]
            new_y = points[i, 1] + learning_rate * grad_y[i]
            
            # Keep within bounds
            points[i, 0] = np.clip(new_x, 0, 1)
            points[i, 1] = np.clip(new_y, 0, 1)
        
        # Check if we improved
        current_ratio = compute_min_max_ratio(points)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()
    
    # Final refinement with local search around the best solution
    for _ in range(100):
        # Try small random perturbations
        test_points = best_points.copy()
        for i in range(n):
            # Small random perturbation
            test_points[i, 0] += np.random.normal(0, 0.001)
            test_points[i, 1] += np.random.normal(0, 0.001)
            # Keep within bounds
            test_points[i, 0] = np.clip(test_points[i, 0], 0, 1)
            test_points[i, 1] = np.clip(test_points[i, 1], 0, 1)
        
        current_ratio = compute_min_max_ratio(test_points)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = test_points.copy()
    
    return best_points


# EVOLVE-BLOCK-END
