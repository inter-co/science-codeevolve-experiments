# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import random
from typing import Tuple
import time
from scipy.spatial.distance import cdist

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate the sum of all circle radii."""
    return np.sum(circles[:, 2])

def generate_hexagonal_initial_solution(n: int = 32) -> np.ndarray:
    """Generate initial configuration using hexagonal packing approach for better density."""
    # Create a hexagonal grid pattern
    # This helps in achieving better initial packing density
    
    # For 32 circles, determine appropriate grid dimensions
    # Using approximately sqrt(n) for both dimensions
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Ensure we have enough cells
    while rows * cols < n:
        rows += 1
    
    # Create hexagonal grid with proper spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Hexagonal offset (sqrt(3)/2) for better packing
    hex_offset = spacing_x * 0.5
    
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            # Even rows aligned, odd rows offset
            x = j * spacing_x + (i % 2) * hex_offset
            y = i * spacing_y
            
            # Ensure points are within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
    
    # If we don't have enough points, fill with random ones
    if len(points) < n:
        additional_points = np.random.rand(n - len(points), 2)
        points.extend(additional_points.tolist())
    
    points = np.array(points[:n])
    
    # Initialize with small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = points[i, 0]  # x
        circles[i, 1] = points[i, 1]  # y
        circles[i, 2] = 0.01          # initial small radius
    
    # Adjust radii to fit within bounds and avoid overlaps
    circles = adjust_radii_hex(circles)
    
    return circles

def adjust_radii_hex(circles: np.ndarray) -> np.ndarray:
    """Improved radius adjustment using more sophisticated approach."""
    n = len(circles)
    
    # First pass: set maximum possible radius for each circle
    for i in range(n):
        x, y = circles[i, 0], circles[i, 1]
        # Maximum radius without going out of bounds
        max_radius = min(x, y, 1-x, 1-y)
        circles[i, 2] = max_radius * 0.8  # Slightly smaller for safety
    
    # Second pass: iterative adjustment with better overlap resolution
    for iteration in range(300):  # More iterations for better convergence
        changed = False
        for i in range(n):
            # Calculate distances to all other circles efficiently using vectorization
            current_pos = np.array([circles[i, 0], circles[i, 1]])
            other_positions = np.array([[circles[j, 0], circles[j, 1]] for j in range(n) if j != i])
            
            if len(other_positions) > 0:
                # Vectorized distance calculation
                distances = np.sqrt(np.sum((other_positions - current_pos)**2, axis=1))
                min_dist = np.min(distances)
                
                # Maximum allowed radius based on neighbors
                max_allowed = min_dist / 2.0 if min_dist > 0 else float('inf')
            else:
                max_allowed = float('inf')
            
            # Boundary constraints
            boundary_r = min(
                circles[i, 0], 
                circles[i, 1], 
                1 - circles[i, 0], 
                1 - circles[i, 1]
            )
            
            # New radius should be minimum of all constraints
            new_radius = min(max_allowed, boundary_r, circles[i, 2] * 1.05)
            
            # Only update if there's a meaningful change
            if new_radius < circles[i, 2] and abs(new_radius - circles[i, 2]) > 1e-8:
                circles[i, 2] = new_radius
                changed = True
                
        if not changed:
            break
    
    return circles

def optimize_with_improved_scipy(circles: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Enhanced scipy optimization with better handling of constraints."""
    n = len(circles)
    
    # Flatten for scipy optimization
    initial_flat = circles.flatten()
    
    # Define bounds - tighter bounds for better convergence
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.49)])
    
    # Objective function (negative because we minimize)
    def objective(x_flat):
        circles_test = x_flat.reshape((n, 3))
        return -np.sum(circles_test[:, 2])
    
    # More efficient constraint functions using vectorization
    def containment_constraints(x_flat):
        circles_test = x_flat.reshape((n, 3))
        x = circles_test[:, 0]
        y = circles_test[:, 1]
        r = circles_test[:, 2]
        
        # Vectorized containment constraints
        # x - r >= 0, y - r >= 0, 1 - x - r >= 0, 1 - y - r >= 0
        constraints = np.concatenate([
            x - r,           # x - r >= 0
            y - r,           # y - r >= 0
            1 - x - r,       # 1 - x - r >= 0
            1 - y - r        # 1 - y - r >= 0
        ])
        return constraints
    
    def overlap_constraints(x_flat):
        circles_test = x_flat.reshape((n, 3))
        x = circles_test[:, 0]
        y = circles_test[:, 1]
        r = circles_test[:, 2]
        
        # More efficient overlap checking using vectorized operations
        # Create distance matrix for all pairs
        positions = np.column_stack([x, y])
        dist_matrix = cdist(positions, positions)
        
        # Create overlap constraints (distance - r1 - r2 >= 0)
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = dist_matrix[i, j]
                r1 = r[i]
                r2 = r[j]
                # We want dist >= r1 + r2, so we check: dist - r1 - r2 >= 0
                constraints.append(dist - r1 - r2)
        return np.array(constraints)
    
    # Create constraint dictionaries
    containment_cons = {
        'type': 'ineq',
        'fun': lambda x: containment_constraints(x)
    }
    
    overlap_cons = {
        'type': 'ineq', 
        'fun': lambda x: overlap_constraints(x)
    }
    
    try:
        # Try multiple optimization methods for better results
        methods = ['SLSQP', 'trust-constr']
        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_flat,
                    method=method,
                    bounds=bounds,
                    constraints=[containment_cons, overlap_cons],
                    options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8},
                    tol=1e-8
                )
                
                if result.success:
                    return result.x.reshape((n, 3))
            except Exception as e:
                continue
    except Exception as e:
        # If optimization fails, return original circles
        pass
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    best_solution = None
    best_sum = 0
    
    # Multi-start approach with different initialization strategies
    n_starts = 25  # Increase starts for better exploration
    
    for start_idx in range(n_starts):
        # Set different seed for each start
        np.random.seed(42 + start_idx)
        random.seed(42 + start_idx)
        
        # Try different initialization methods with weighted probabilities
        init_method = start_idx % 5
        if init_method == 0:
            # Hexagonal initialization
            initial_solution = generate_hexagonal_initial_solution(32)
        elif init_method == 1:
            # Voronoi initialization (similar to inspiration)
            initial_solution = generate_voronoi_initial_solution(32)
        elif init_method == 2:
            # Random initialization
            points = np.random.rand(32, 2)
            initial_solution = np.zeros((32, 3))
            for i in range(32):
                initial_solution[i, 0] = points[i, 0]
                initial_solution[i, 1] = points[i, 1]
                initial_solution[i, 2] = 0.01
        elif init_method == 3:
            # Mixed approach
            initial_solution = generate_mixed_initial_solution(32)
        else:
            # Alternative hexagonal approach with different parameters
            initial_solution = generate_alternative_hex_initial_solution(32)
        
        # Apply local optimization with varying iterations
        optimized_circles = optimize_with_improved_scipy(initial_solution, max_iter=800)
        
        # Apply one more round of optimization to refine further
        optimized_circles = optimize_with_improved_scipy(optimized_circles, max_iter=400)
        
        # Evaluate this solution
        current_sum = calculate_radius_sum(optimized_circles)
        
        # Keep the best solution found
        if current_sum > best_sum:
            best_sum = current_sum
            best_solution = optimized_circles.copy()
    
    # Final validation and refinement
    if best_solution is not None:
        # One final optimization pass on the best solution
        final_solution = optimize_with_improved_scipy(best_solution, max_iter=200)
        final_sum = calculate_radius_sum(final_solution)
        
        if final_sum > best_sum:
            return final_solution
    
    # Fallback to the best solution found
    return best_solution if best_solution is not None else generate_hexagonal_initial_solution(32)

def generate_alternative_hex_initial_solution(n: int = 32) -> np.ndarray:
    """Alternative hexagonal initialization with different spacing."""
    # Create a denser hexagonal grid
    rows = int(np.ceil(np.sqrt(n)) * 1.2)
    cols = int(np.ceil(n / rows))
    
    # Ensure we have enough cells
    while rows * cols < n:
        rows += 1
    
    # Create hexagonal grid with closer spacing
    spacing_x = 0.9 / cols  # Slightly less than full width for better packing
    spacing_y = 0.9 / rows  # Slightly less than full height
    
    # Hexagonal offset
    hex_offset = spacing_x * 0.5
    
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) >= n:
                break
            # Even rows aligned, odd rows offset
            x = j * spacing_x + (i % 2) * hex_offset + 0.05  # Offset to center
            y = i * spacing_y + 0.05  # Offset to center
            
            # Ensure points are within bounds
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
    
    # If we don't have enough points, fill with random ones
    if len(points) < n:
        additional_points = np.random.rand(n - len(points), 2)
        points.extend(additional_points.tolist())
    
    points = np.array(points[:n])
    
    # Initialize with small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = points[i, 0]  # x
        circles[i, 1] = points[i, 1]  # y
        circles[i, 2] = 0.01          # initial small radius
    
    # Adjust radii to fit within bounds and avoid overlaps
    circles = adjust_radii_hex(circles)
    
    return circles

def generate_mixed_initial_solution(n: int = 32) -> np.ndarray:
    """Generate mixed initial configuration combining multiple approaches."""
    # Start with hexagonal arrangement
    circles = generate_hexagonal_initial_solution(n)
    
    # Perturb some positions to escape local optima
    for i in range(n):
        if np.random.random() < 0.3:  # 30% chance to perturb
            circles[i, 0] += (np.random.random() - 0.5) * 0.05
            circles[i, 1] += (np.random.random() - 0.5) * 0.05
            
            # Clamp to valid range
            circles[i, 0] = np.clip(circles[i, 0], 0.001, 0.999)
            circles[i, 1] = np.clip(circles[i, 1], 0.001, 0.999)
    
    # Re-adjust radii after perturbation
    circles = adjust_radii_hex(circles)
    
    return circles

def generate_voronoi_initial_solution(n: int = 32) -> np.ndarray:
    """Generate initial configuration using Voronoi-based approach similar to inspiration."""
    # Generate random points first
    points = np.random.rand(n, 2)
    
    try:
        # Create Voronoi diagram
        vor = Voronoi(points)
        centroids = []
        
        # Extract centroids of bounded Voronoi regions
        for i in range(len(vor.points)):
            if i < len(vor.regions) and len(vor.regions[i]) > 0:
                region = vor.regions[i]
                if all(r != -1 for r in region):  # Skip infinite regions
                    # Compute centroid of polygon
                    polygon_points = [vor.vertices[r] for r in region]
                    if len(polygon_points) >= 3:
                        # Check if polygon is within unit square
                        if all(0 <= p[0] <= 1 and 0 <= p[1] <= 1 for p in polygon_points):
                            centroid = np.mean(polygon_points, axis=0)
                            centroids.append(centroid)
        
        # If we got good centroids, use them; otherwise fall back to random
        if len(centroids) >= n:
            points = np.array(centroids[:n])
        else:
            points = np.random.rand(n, 2)
            
    except:
        points = np.random.rand(n, 2)
    
    # Initialize with small radii
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = points[i, 0]  # x
        circles[i, 1] = points[i, 1]  # y
        circles[i, 2] = 0.01          # initial small radius
    
    # Adjust radii to fit within bounds and avoid overlaps
    circles = adjust_radii_hex(circles)
    
    return circles


# EVOLVE-BLOCK-END
