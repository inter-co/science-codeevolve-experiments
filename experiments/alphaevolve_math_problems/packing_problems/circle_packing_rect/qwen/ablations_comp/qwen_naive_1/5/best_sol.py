# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import random
from itertools import combinations
import time

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a more efficient hybrid approach with better initialization and optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    
    # Better initialization using a more systematic approach
    def initialize_better_layout():
        # Try different rectangle dimensions to find optimal aspect ratio
        best_sum = 0
        best_circles = None
        best_width = 1.0
        best_height = 1.0
        
        # Focus on aspect ratios that typically work well for circle packing
        ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        for ratio in ratios:
            width = 1.0 * ratio
            height = 2.0 - width
            
            if width <= 0 or height <= 0:
                continue
                
            circles = []
            
            # Try to place circles systematically
            # Start with a dense packing approach
            max_radius = min(width, height) * 0.15
            
            # Create a regular grid pattern for initial placement
            grid_size = int(np.ceil(np.sqrt(n)))
            spacing_x = width / (grid_size + 1)
            spacing_y = height / (grid_size + 1)
            
            # Adjust grid to fit better
            actual_grid_size = max(1, int(np.sqrt(n)))
            rows = actual_grid_size
            cols = actual_grid_size
            if rows * cols < n:
                cols = max(1, int(n / rows))
            
            # Create grid points
            for i in range(rows):
                for j in range(cols):
                    if len(circles) >= n:
                        break
                    x = spacing_x * (j + 1)
                    y = spacing_y * (i + 1)
                    
                    # Ensure we're within bounds
                    if x >= max_radius and x <= width - max_radius and \
                       y >= max_radius and y <= height - max_radius:
                        # Calculate appropriate radius based on position
                        center_dist = np.sqrt((x - width/2)**2 + (y - height/2)**2)
                        max_radius_here = min(x, width - x, y, height - y)
                        
                        # Position-based radius adjustment
                        radius_factor = 1.0 - 0.5 * (center_dist / (np.sqrt((width/2)**2 + (height/2)**2)))
                        radius_factor = max(0.3, min(1.0, radius_factor))
                        
                        # Different radius strategy for center vs edge circles
                        if center_dist < min(width, height) * 0.3:
                            radius = max_radius_here * 0.3 * radius_factor
                        elif center_dist < min(width, height) * 0.6:
                            radius = max_radius_here * 0.2 * radius_factor
                        else:
                            radius = max_radius_here * 0.15 * radius_factor
                        
                        radius = max(radius, 0.005)
                        circles.append([x, y, radius])
                
                if len(circles) >= n:
                    break
            
            # Fill remaining spots if needed
            while len(circles) < n:
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                max_radius_here = min(x, width - x, y, height - y)
                radius = max_radius_here * 0.15
                radius = max(radius, 0.005)
                circles.append([x, y, radius])
            
            current_sum = sum(circle[2] for circle in circles[:n])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles[:n]
                best_width = width
                best_height = height
        
        # Fallback to radial approach if needed
        if best_circles is None or best_sum < 1.0:
            circles = []
            width, height = 1.0, 1.0  # Square for simplicity
            
            # Place in concentric circles/rings from center
            center_x, center_y = width/2, height/2
            max_radius = min(width, height) * 0.15
            
            # Ring-based placement with decreasing radii
            ring_count = 4
            circles_per_ring = n // ring_count + 1
            
            for ring in range(ring_count):
                if len(circles) >= n:
                    break
                angle_step = 2 * np.pi / min(circles_per_ring, n - len(circles))
                ring_radius = (ring + 1) * 0.2 * min(width, height) / 2
                
                for i in range(min(circles_per_ring, n - len(circles))):
                    angle = i * angle_step
                    x = center_x + ring_radius * np.cos(angle)
                    y = center_y + ring_radius * np.sin(angle)
                    
                    # Keep within bounds
                    x = np.clip(x, max_radius, width - max_radius)
                    y = np.clip(y, max_radius, height - max_radius)
                    
                    # Set radius - smaller for outer rings
                    circle_radius = max_radius * (1.0 - ring * 0.2)
                    circle_radius = max(circle_radius, 0.005)
                    
                    if x >= circle_radius and x <= width - circle_radius and \
                       y >= circle_radius and y <= height - circle_radius:
                        circles.append([x, y, circle_radius])
            
            # Fill any remaining spots
            while len(circles) < n:
                x = random.uniform(max_radius, width - max_radius)
                y = random.uniform(max_radius, height - max_radius)
                max_radius_local = min(x, width - x, y, height - y)
                circle_radius = max_radius_local * 0.15
                circle_radius = max(circle_radius, 0.005)
                circles.append([x, y, circle_radius])
            
            best_circles = circles[:n]
            best_width = width
            best_height = height
            
        return np.array(best_circles), best_width, best_height
    
    # Initialize with better configuration
    circles, rect_width, rect_height = initialize_better_layout()
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(params):
        # Reshape params into positions and radii
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Calculate negative sum of radii (we want to maximize sum, so minimize negative)
        return -np.sum(radii)
    
    # Constraint function for non-overlapping circles
    def non_overlap_constraint(params):
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Distance matrix between circle centers
        dist_matrix = cdist(positions, positions)
        constraints = []
        
        for i in range(n):
            for j in range(i+1, n):
                # Non-overlapping constraint: distance >= sum of radii
                distance = dist_matrix[i, j]
                min_distance = radii[i] + radii[j]
                # We want distance >= min_distance, so constraint = distance - min_distance >= 0
                constraints.append(distance - min_distance)
        
        return np.array(constraints)
    
    # Boundary constraints for circles to stay within rectangle
    def boundary_constraint(params):
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        constraints = []
        # Left boundary: x - radius >= 0
        constraints.extend(positions[:, 0] - radii)
        # Right boundary: width - x - radius >= 0  
        constraints.extend(rect_width - positions[:, 0] - radii)
        # Bottom boundary: y - radius >= 0
        constraints.extend(positions[:, 1] - radii)
        # Top boundary: height - y - radius >= 0
        constraints.extend(rect_height - positions[:, 1] - radii)
        
        return np.array(constraints)
    
    # Combined constraints - all must be >= 0 for feasibility
    def combined_constraints(params):
        # Non-overlap constraints (positive means satisfied)
        overlap_violations = non_overlap_constraint(params)
        # Boundary constraints (positive means satisfied)  
        boundary_violations = boundary_constraint(params)
        # Combine constraints (positive means satisfied)
        return np.concatenate([overlap_violations, boundary_violations])
    
    # Improved optimization approach - much more focused and efficient
    # Initial parameter vector: [x1, y1, x2, y2, ..., xn, yn, r1, r2, ..., rn]
    initial_params = np.concatenate([
        circles[:, :2].flatten(),  # Positions
        circles[:, 2]              # Radii
    ])
    
    # Set bounds for positions and radii
    # Positions: [0, width] for x and y coordinates
    # Radii: [1e-6, min(width, height)/2] to prevent degenerate cases
    bounds = [(0, rect_width) for _ in range(2*n)] + [(1e-6, min(rect_width, rect_height)/2) for _ in range(n)]
    
    # Define constraints - all must be >= 0
    constraints = {
        'type': 'ineq',
        'fun': combined_constraints
    }
    
    # Simplified but effective optimization approach
    try:
        # Use a single, well-tuned optimization run
        # Reduce computational overhead by limiting iterations and using better solver
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10},
            tol=1e-10
        )
        
        # If optimization succeeded, update circles
        if result.success:
            final_positions = result.x[:-n].reshape(-1, 2)
            final_radii = result.x[-n:]
            
            # Update circles array with optimized values
            circles[:, 0] = final_positions[:, 0]
            circles[:, 1] = final_positions[:, 1]
            circles[:, 2] = final_radii
            
            # Ensure all radii are positive and reasonable
            circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
            # Make sure radii don't exceed reasonable limits
            max_radius_allowed = min(rect_width, rect_height) / 2
            circles[:, 2] = np.minimum(circles[:, 2], max_radius_allowed)
            
    except Exception as e:
        # If optimization fails, return initial configuration
        pass
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
