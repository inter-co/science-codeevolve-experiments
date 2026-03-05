# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses Voronoi-based initialization combined with advanced optimization techniques.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Try to find optimal aspect ratio by testing different ratios
    best_result = None
    best_sum = 0
    
    # Test different width/height ratios - expanded range for better optimization
    ratios = [0.5, 0.6, 0.75, 0.85, 1.0, 1.15, 1.25, 1.5, 1.75, 2.0, 2.5]
    
    for ratio in ratios:
        width = 2 * ratio / (1 + ratio)  # width + height = 2, ratio = width/height
        height = 2 / (1 + ratio)
        
        result = try_optimization(width, height)
        sum_radii = np.sum(result[:, 2])
        
        if sum_radii > best_sum:
            best_sum = sum_radii
            best_result = result
    
    return best_result

def try_optimization(width: float, height: float) -> np.ndarray:
    """Try optimization with given rectangle dimensions"""
    n = 21
    
    # Step 1: Generate initial configuration using improved Voronoi-based method
    np.random.seed(42)  # For reproducibility
    
    # Use strategic initial points for better Voronoi coverage (inspired by best practices)
    initial_points = []
    
    # Add corner points to encourage better coverage
    corners = [[0, 0], [width, 0], [0, height], [width, height]]
    initial_points.extend(corners)
    
    # Add points along edges with better distribution
    edge_points = 8
    remaining_points = n - len(corners) - edge_points
    
    # Add edge points with strategic placement - more even distribution
    for i in range(edge_points):
        if i < edge_points // 2:
            # Top/bottom edges - more evenly spaced
            if edge_points // 2 > 1:
                x = 0.05 * width + 0.9 * width * (i / (edge_points // 2 - 1))
            else:
                x = width * 0.5
            y = 0.0 if i % 2 == 0 else height
        else:
            # Left/right edges - more evenly spaced
            x = 0.0 if i % 2 == 0 else width
            if edge_points // 2 > 1:
                y = 0.05 * height + 0.9 * height * ((i - edge_points // 2) / (edge_points // 2 - 1))
            else:
                y = height * 0.5
        initial_points.append([x, y])
    
    # Add random points in interior with triangular distribution for better spread
    for _ in range(remaining_points):
        # Use triangular distribution to avoid clustering in center
        x = np.random.triangular(0.05 * width, 0.5 * width, 0.95 * width)
        y = np.random.triangular(0.05 * height, 0.5 * height, 0.95 * height)
        initial_points.append([x, y])
    
    initial_points = np.array(initial_points[:n])
    
    # Create Voronoi diagram
    try:
        vor = Voronoi(initial_points)
    except Exception:
        # Fallback to simpler initialization if Voronoi fails
        centroids = np.random.rand(n, 2) * np.array([width, height])
        radii = np.full(n, min(width, height) / 10)
        return np.column_stack([centroids, radii])
    
    # Calculate centroids of Voronoi cells as initial circle positions
    centroids = []
    valid_count = 0
    
    for i in range(len(vor.points)):
        # Find vertices of the Voronoi cell for point i
        region = vor.regions[vor.point_region[i]]
        if len(region) > 0 and -1 not in region:
            # Get the vertices of this region
            vertices = np.array([vor.vertices[j] for j in region])
            # Calculate centroid
            centroid = np.mean(vertices, axis=0)
            # Ensure centroid is within bounds
            if (0 <= centroid[0] <= width) and (0 <= centroid[1] <= height):
                centroids.append(centroid)
                valid_count += 1
    
    # If we don't have enough valid centroids, fill with random points
    while len(centroids) < n:
        centroids.append(np.random.rand(2) * np.array([width, height]))
    
    centroids = np.array(centroids[:n])
    
    # Step 2: Initialize radii based on minimum distances to neighbors with better strategy
    radii = np.zeros(n)
    for i in range(n):
        # Distance to nearest neighbor
        distances = cdist([centroids[i]], centroids)[0]
        distances[i] = np.inf  # Exclude self-distance
        min_distance = np.min(distances)
        
        # Set radius to half the minimum distance (with bounds)
        max_radius = min_distance / 2.0
        # Ensure radius is positive and reasonable - use more conservative limits
        radii[i] = max(0.001, min(max_radius, min(width/5, height/5)))
    
    # Step 3: Advanced optimization using scipy minimize with improved constraints
    # Flatten parameters: [x1, y1, r1, x2, y2, r2, ...]
    initial_params = np.concatenate([centroids.flatten(), radii])
    
    def objective(params):
        # Extract positions and radii
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Maximize sum of radii (minimize negative sum)
        return -np.sum(radii)
    
    def constraint_func(params):
        positions = params[:2*n].reshape(-1, 2)
        radii = params[2*n:]
        
        # Constraint: circles must be within bounds with safety margin
        constraints = []
        
        # Circle center within rectangle bounds with safety margin
        margin = 1e-6
        for i in range(n):
            constraints.extend([
                positions[i][0] - radii[i] - margin,  # x - r - margin >= 0
                positions[i][1] - radii[i] - margin,  # y - r - margin >= 0
                width - positions[i][0] - radii[i] - margin,  # width - x - r - margin >= 0
                height - positions[i][1] - radii[i] - margin  # height - y - r - margin >= 0
            ])
        
        # Constraint: no overlapping circles with safety margin
        safety_margin = 1e-9
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                              (positions[i][1] - positions[j][1])**2)
                # Distance between centers >= sum of radii + safety margin
                constraints.append(dist - (radii[i] + radii[j]) - safety_margin)
        
        return np.array(constraints)
    
    # Define bounds for optimization - tighter bounds for better convergence
    bounds = []
    # Position bounds: [margin, width-margin] for x, [margin, height-margin] for y
    margin = 1e-6
    for i in range(n):
        bounds.extend([(margin, width - margin), (margin, height - margin)])  # x, y bounds
    # Radius bounds: [0.001, reasonable_max] 
    max_radius_allowed = min(width, height) / 2
    for i in range(n):
        bounds.extend([(0.001, max_radius_allowed)])
    
    # Optimize with enhanced parameters - even more aggressive settings for better results
    try:
        # Use SLSQP method which handles constraints well
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda x: constraint_func(x)},
            options={'maxiter': 5000, 'ftol': 1e-12, 'eps': 1e-10, 'iprint': -1}  # Very tight tolerances and quiet mode
        )
        
        if result.success:
            final_positions = result.x[:2*n].reshape(-1, 2)
            final_radii = result.x[2*n:]
            
            # Create final circles array
            circles = np.column_stack([final_positions, final_radii])
        else:
            # If optimization fails, return the Voronoi-based solution
            circles = np.column_stack([centroids, radii])
    except Exception as e:
        # Fallback to Voronoi-based solution
        circles = np.column_stack([centroids, radii])
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
