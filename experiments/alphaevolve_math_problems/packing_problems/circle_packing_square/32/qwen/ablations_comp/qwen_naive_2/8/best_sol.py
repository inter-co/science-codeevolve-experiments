# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.optimize import minimize
import random
from collections import defaultdict
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a Voronoi-based physics simulation approach with iterative force relaxation.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    
    n = 32
    max_iterations = 5000
    tolerance = 1e-6
    
    # Initialize circles with a better starting configuration using a more systematic approach
    def initialize_circles():
        # Use a grid-based initialization but with proper spacing
        circles = []
        
        # Create a 6x6 grid for 36 positions, then take first 32
        rows, cols = 6, 6
        spacing_x = 0.95 / cols
        spacing_y = 0.95 / rows
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                x = 0.025 + j * spacing_x
                y = 0.025 + i * spacing_y
                positions.append((x, y))
        
        # Take first 32 positions and assign small random radii
        for i, (x, y) in enumerate(positions[:n]):
            # Start with very small radius and let optimization grow them
            r = 0.02
            circles.append([x, y, r])
        
        return np.array(circles)
    
    # Physics-based force computation
    def compute_forces(circles):
        forces = np.zeros_like(circles)
        
        # Compute repulsive forces between all pairs of circles
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            for j in range(i+1, len(circles)):
                x2, y2, r2 = circles[j]
                
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist > 0:
                    # Repulsive force proportional to inverse square of distance
                    min_dist = r1 + r2
                    if dist < min_dist * 1.5:  # Only apply strong force when close
                        # Force magnitude inversely proportional to distance squared
                        force_magnitude = 1.0 / (dist * dist + 1e-10)
                        
                        # Normalize direction vector
                        fx = force_magnitude * dx / dist
                        fy = force_magnitude * dy / dist
                        
                        # Apply forces (reverse for j)
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
        
        # Compute attractive forces toward boundaries (to keep circles inside)
        boundary_force_strength = 100.0
        for i in range(len(circles)):
            x, y, r = circles[i]
            
            # Boundary forces (pull back toward center if too close to boundary)
            left_force = max(0, r - x) * boundary_force_strength
            right_force = max(0, x + r - 1) * boundary_force_strength
            bottom_force = max(0, r - y) * boundary_force_strength
            top_force = max(0, y + r - 1) * boundary_force_strength
            
            forces[i, 0] += right_force - left_force
            forces[i, 1] += top_force - bottom_force
            
        return forces
    
    # Apply forces to update circle positions and radii
    def update_circles(circles, forces, learning_rate=0.01):
        updated = circles.copy()
        
        for i in range(len(circles)):
            x, y, r = circles[i]
            fx, fy = forces[i, 0], forces[i, 1]
            
            # Update position with forces
            new_x = x + fx * learning_rate
            new_y = y + fy * learning_rate
            
            # Keep within bounds
            new_x = np.clip(new_x, r, 1 - r)
            new_y = np.clip(new_y, r, 1 - r)
            
            # Update radius based on how much force was applied
            # This encourages larger radii when there's less competition
            force_magnitude = np.sqrt(fx*fx + fy*fy)
            if force_magnitude > 1e-10:
                # Increase radius slightly when forces are low (more space available)
                delta_r = 0.001 * (1.0 / (force_magnitude + 1e-8)) * learning_rate
                new_r = min(r + delta_r, 0.45)  # Cap at reasonable size
            else:
                new_r = r
                
            # Ensure radius stays valid
            new_r = max(0.001, min(0.49, new_r))
            
            updated[i] = [new_x, new_y, new_r]
            
        return updated
    
    # Check if circles satisfy constraints
    def check_constraints(circles):
        # Check containment
        for x, y, r in circles:
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check overlap
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                if dist_sq < min_dist_sq:
                    return False
        return True
    
    # Calculate total radius
    def total_radius(circles):
        return sum(circles[:, 2])
    
    # Main optimization loop
    circles = initialize_circles()
    
    # Simulate physics-based relaxation
    prev_total = 0
    stagnation_count = 0
    
    for iteration in range(max_iterations):
        # Compute forces
        forces = compute_forces(circles)
        
        # Update circles
        circles = update_circles(circles, forces, learning_rate=0.01)
        
        # Check constraints and fix if violated
        if not check_constraints(circles):
            # Reset to previous good state if constraints violated
            circles = np.clip(circles, 0, 1)
            # Fix containment by adjusting positions and radii
            for i in range(len(circles)):
                x, y, r = circles[i]
                # Adjust to stay within bounds
                x = np.clip(x, r, 1 - r)
                y = np.clip(y, r, 1 - r)
                circles[i] = [x, y, r]
        
        # Monitor progress
        current_total = total_radius(circles)
        if abs(current_total - prev_total) < tolerance:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_total = current_total
        
        # Early stopping if no improvement for many iterations
        if stagnation_count > 100:
            break
    
    # Final optimization using scipy with better constraints
    def objective(params):
        # Reshape params into circles
        circles = params.reshape(-1, 3)
        # Minimize negative of sum of radii (maximize sum of radii)
        return -np.sum(circles[:, 2])
    
    def constraint_containment(params):
        circles = params.reshape(-1, 3)
        constraints = []
        for i in range(len(circles)):
            x, y, r = circles[i]
            # r <= x <= 1-r
            constraints.append(x - r)  # x - r >= 0
            constraints.append(1 - x - r)  # 1 - x - r >= 0
            # r <= y <= 1-r
            constraints.append(y - r)  # y - r >= 0
            constraints.append(1 - y - r)  # 1 - y - r >= 0
        return np.array(constraints)
    
    def constraint_overlap(params):
        circles = params.reshape(-1, 3)
        constraints = []
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                constraints.append(dist_sq - min_dist_sq - 1e-12)
        return np.array(constraints)
    
    # Flatten for scipy optimization
    x0 = circles.flatten()
    
    # Set bounds
    bounds = [(0.001, 0.999) for _ in range(len(x0))]
    
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_containment},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'maxiter': 500, 'ftol': 1e-8, 'eps': 1e-8, 'disp': False})
        
        if result.success:
            final_circles = result.x.reshape(-1, 3)
            return final_circles
    except:
        pass
    
    return circles


# EVOLVE-BLOCK-END
