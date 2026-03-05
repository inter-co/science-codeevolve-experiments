# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid approach combining geometric insights with optimization.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Start with a good initial configuration based on known optimal arrangements
    # Use a hexagonal lattice pattern with some perturbation for optimization
    
    # Create a regular hexagonal grid pattern as starting point
    # This provides a good balance of uniformity and spacing
    points = []
    
    # Hexagonal packing arrangement (approximate)
    rows = 4
    cols = 4
    spacing = 1.0
    
    for i in range(rows):
        for j in range(cols):
            # Offset every other row for hexagonal packing
            x = j * spacing + (i % 2) * spacing * 0.5
            y = i * spacing * math.sqrt(3)/2
            points.append([x, y])
    
    # Normalize to unit square [0,1] x [0,1]
    points = np.array(points[:16])  # Take first 16 points
    
    # Normalize to fit in [0,1] x [0,1] 
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    if max_x > min_x:
        points[:, 0] = (points[:, 0] - min_x) / (max_x - min_x)
    if max_y > min_y:
        points[:, 1] = (points[:, 1] - min_y) / (max_y - min_y)
    
    # Ensure we have exactly 16 points
    if len(points) < 16:
        # Fill remaining points with random points
        extra_points = 16 - len(points)
        random_points = np.random.rand(extra_points, 2)
        points = np.vstack([points, random_points])
    elif len(points) > 16:
        points = points[:16]
    
    # Apply local optimization using simulated annealing
    # This improves the geometric distribution significantly
    best_points = optimize_points(points.copy())
    
    return best_points


def compute_min_max_ratio(points):
    """Compute the ratio of minimum to maximum distance"""
    if len(points) < 2:
        return 0.0
    
    # Compute pairwise distances
    distances = pdist(points)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    if max_dist == 0:
        return 0.0
    
    return min_dist / max_dist


def optimize_points(initial_points):
    """Optimize point configuration using simulated annealing approach"""
    points = initial_points.copy()
    current_ratio = compute_min_max_ratio(points)
    
    # Parameters for simulated annealing
    temperature = 1.0
    cooling_rate = 0.995
    min_temperature = 1e-6
    iterations_per_temp = 100
    
    best_points = points.copy()
    best_ratio = current_ratio
    
    # Keep track of recent improvements
    recent_improvements = []
    
    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            # Make a small random perturbation to one point
            idx = np.random.randint(len(points))
            new_points = points.copy()
            
            # Small random displacement
            delta = np.random.normal(0, 0.01, 2)
            new_points[idx] += delta
            
            # Keep points within [0,1] x [0,1]
            new_points[idx] = np.clip(new_points[idx], 0, 1)
            
            # Check if this improves the ratio
            new_ratio = compute_min_max_ratio(new_points)
            
            # Accept or reject based on Metropolis criterion
            if new_ratio > current_ratio:
                points = new_points
                current_ratio = new_ratio
                
                if new_ratio > best_ratio:
                    best_points = new_points.copy()
                    best_ratio = new_ratio
                    recent_improvements = []
                else:
                    recent_improvements.append(current_ratio)
            else:
                # Accept with probability based on temperature
                if np.random.random() < math.exp((new_ratio - current_ratio) / temperature):
                    points = new_points
                    current_ratio = new_ratio
                    recent_improvements.append(current_ratio)
                else:
                    recent_improvements.append(current_ratio)
            
            # Remove old improvements to prevent memory buildup
            if len(recent_improvements) > 1000:
                recent_improvements.pop(0)
        
        temperature *= cooling_rate
    
    return best_points


# EVOLVE-BLOCK-END
