# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a Voronoi-based approach with optimization to find a high-quality solution.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    n = 32
    
    # Generate initial points using a low-discrepancy sequence for better distribution
    def generate_initial_points():
        # Use a simple grid with slight perturbation for better distribution
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n:
                    break
                # Add small random perturbation to avoid regular patterns
                x = (j + 0.5 + np.random.uniform(-0.2, 0.2)) / cols
                y = (i + 0.5 + np.random.uniform(-0.2, 0.2)) / rows
                # Keep within bounds
                x = np.clip(x, 0.01, 0.99)
                y = np.clip(y, 0.01, 0.99)
                points.append([x, y])
        
        return np.array(points[:n])
    
    # Compute Voronoi diagram and extract circle centers/radii
    def voronoi_circle_placement(points):
        # Add boundary points to ensure proper Voronoi cells near boundaries
        boundary_points = [
            [0, 0], [0, 1], [1, 0], [1, 1],
            [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5]
        ]
        all_points = np.vstack([points, boundary_points])
        
        try:
            vor = Voronoi(all_points)
        except:
            # Fallback to simple approach if Voronoi fails
            return points, np.full(len(points), 0.05)
        
        # Extract Voronoi vertices and compute radii
        radii = []
        centers = []
        
        # For each original point, compute the minimum distance to the Voronoi cell boundary
        for i in range(len(points)):
            # Get Voronoi region for this point
            region_indices = np.where(vor.point_region == i)[0]
            if len(region_indices) == 0:
                # Fallback
                radii.append(0.05)
                centers.append(points[i])
                continue
                
            # Get vertices of the Voronoi cell
            region = vor.regions[vor.point_region[i]]
            if -1 in region:
                # Infinite region, skip this cell
                radii.append(0.05)
                centers.append(points[i])
                continue
                
            # Compute Voronoi cell vertices
            vertices = vor.vertices[region]
            if len(vertices) < 3:
                # Not enough vertices, use fallback
                radii.append(0.05)
                centers.append(points[i])
                continue
            
            # Compute the minimum distance from center to any edge of the cell
            center = points[i]
            min_distance = float('inf')
            
            # Check distance to all edges of the polygon
            for j in range(len(vertices)):
                p1 = vertices[j]
                p2 = vertices[(j + 1) % len(vertices)]
                
                # Distance from point to line segment
                # Vector from p1 to p2
                line_vec = p2 - p1
                # Vector from p1 to center
                point_vec = center - p1
                
                line_len_sq = np.dot(line_vec, line_vec)
                if line_len_sq == 0:
                    # Line segment degenerates to point
                    distance = np.linalg.norm(point_vec)
                else:
                    # Project point onto line
                    t = np.dot(point_vec, line_vec) / line_len_sq
                    # Clamp t to [0,1] to stay within segment
                    t = np.clip(t, 0, 1)
                    # Closest point on segment
                    closest_point = p1 + t * line_vec
                    distance = np.linalg.norm(center - closest_point)
                
                min_distance = min(min_distance, distance)
            
            # Also check distance to boundaries of unit square
            boundary_dist = min(
                center[0], 1-center[0], center[1], 1-center[1]
            )
            
            # Take the minimum of boundary and Voronoi distances
            radius = min(min_distance, boundary_dist) * 0.9  # Safety factor
            radius = max(radius, 0.001)  # Minimum radius
            
            radii.append(radius)
            centers.append(center)
        
        return np.array(centers), np.array(radii)
    
    # Optimization function to improve the solution
    def optimize_circles(circles):
        # Convert circles to parameters: [x1, y1, r1, x2, y2, r2, ...]
        params = circles.flatten()
        
        def objective(params):
            # Reshape back to circles
            circles_current = params.reshape(-1, 3)
            
            # Compute sum of radii (we want to maximize this)
            return -np.sum(circles_current[:, 2])
        
        def constraint_func(params):
            # Reshape back to circles
            circles_current = params.reshape(-1, 3)
            
            # Constraint: all circles must be within unit square with their radii
            constraints = []
            
            # Boundary constraints (circle must fit entirely in square)
            for i in range(len(circles_current)):
                x, y, r = circles_current[i]
                constraints.extend([
                    x - r,  # x >= r
                    1 - x - r,  # 1-x >= r
                    y - r,  # y >= r
                    1 - y - r   # 1-y >= r
                ])
            
            # Non-overlap constraints
            for i in range(len(circles_current)):
                for j in range(i+1, len(circles_current)):
                    x1, y1, r1 = circles_current[i]
                    x2, y2, r2 = circles_current[j]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist_sq = (r1 + r2)**2
                    # We want dist >= r1 + r2, so we enforce dist_sq >= min_dist_sq
                    # This means we want to minimize (dist_sq - min_dist_sq) but only when it's negative
                    constraints.append(dist_sq - min_dist_sq)
            
            return np.array(constraints)
        
        # Bounds for optimization: x,y in [r, 1-r], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r
        
        # Use SLSQP for constrained optimization
        try:
            result = minimize(
                objective,
                params,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            if result.success:
                circles_opt = result.x.reshape(-1, 3)
                # Ensure all constraints are met
                for i in range(len(circles_opt)):
                    x, y, r = circles_opt[i]
                    # Enforce boundary constraints
                    r = min(r, x, 1-x, y, 1-y)
                    circles_opt[i] = [x, y, r]
                return circles_opt
        except:
            pass
        
        return circles
    
    # Main algorithm
    # Step 1: Generate initial points
    points = generate_initial_points()
    
    # Step 2: Generate Voronoi-based circles
    centers, radii = voronoi_circle_placement(points)
    circles = np.column_stack([centers, radii])
    
    # Step 3: Optimize the configuration
    circles = optimize_circles(circles)
    
    # Step 4: Final refinement
    # Make sure all constraints are strictly satisfied
    for i in range(n):
        x, y, r = circles[i]
        # Ensure circle fits in square
        r = min(r, x, 1-x, y, 1-y)
        circles[i] = [x, y, r]
    
    # Verify no overlaps
    def verify_no_overlaps(circles):
        positions = circles[:, :2]
        radii = circles[:, 2]
        distances = cdist(positions, positions)
        
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                if distances[i, j] < (radii[i] + radii[j]):
                    return False
        return True
    
    # If overlaps exist, apply local refinement
    if not verify_no_overlaps(circles):
        # Simple iterative refinement
        for _ in range(10):
            improved = False
            for i in range(n):
                # Try to slightly adjust position and radius
                x, y, r = circles[i]
                best_x, best_y, best_r = x, y, r
                
                # Try small adjustments
                for dx in [-0.01, 0, 0.01]:
                    for dy in [-0.01, 0, 0.01]:
                        for dr in [-0.005, 0, 0.005]:
                            new_x = x + dx
                            new_y = y + dy
                            new_r = r + dr
                            
                            # Check bounds
                            if (new_x >= new_r and new_x <= 1-new_r and 
                                new_y >= new_r and new_y <= 1-new_r and 
                                new_r > 0):
                                
                                # Check overlap with others
                                valid = True
                                for k in range(n):
                                    if k != i:
                                        dx_k = new_x - circles[k, 0]
                                        dy_k = new_y - circles[k, 1]
                                        dist = np.sqrt(dx_k*dx_k + dy_k*dy_k)
                                        if dist < (new_r + circles[k, 2]):
                                            valid = False
                                            break
                                
                                if valid:
                                    # Check if improvement
                                    if new_r > best_r:
                                        best_x, best_y, best_r = new_x, new_y, new_r
                                        improved = True
                            
                circles[i] = [best_x, best_y, best_r]
            
            if not improved:
                break
    
    return circles


# EVOLVE-BLOCK-END
