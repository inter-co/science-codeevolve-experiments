# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import time
from scipy.spatial import cKDTree
import warnings

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with gradient-based optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    
    # Better geometric initialization based on known patterns and density considerations
    def optimized_initialization():
        circles = np.zeros((n, 3))
        
        # Use a more sophisticated approach: arrange in a pattern inspired by hexagonal close packing
        # but adapted for the constrained square domain
        
        # First, let's try a grid-based approach with some optimization
        sqrt_n = int(np.ceil(np.sqrt(n)))
        grid_rows = sqrt_n
        grid_cols = sqrt_n
        
        # Create a systematic layout with better spacing
        idx = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                if idx >= n:
                    break
                # Distribute points in a way that leaves room for optimization
                x = 0.1 + (j + 0.5) * 0.8 / grid_cols
                y = 0.1 + (i + 0.5) * 0.8 / grid_rows
                
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Start with a reasonable initial radius - higher than before to allow more growth
                circles[idx] = [x, y, 0.08]
                idx += 1
                
        # Add some randomness to avoid getting stuck in poor local optima
        for i in range(min(10, n)):  # Perturb first few circles more aggressively
            if i < n:
                circles[i, 0] += np.random.uniform(-0.05, 0.05)
                circles[i, 1] += np.random.uniform(-0.05, 0.05)
                circles[i, 0] = np.clip(circles[i, 0], 0.05, 0.95)
                circles[i, 1] = np.clip(circles[i, 1], 0.05, 0.95)
        
        return circles
    
    # More robust constraint checking
    def check_constraints(circles):
        """Check if all constraints are satisfied"""
        # Check containment
        for i in range(len(circles)):
            x, y, r = circles[i]
            if not (r <= x <= 1-r and r <= y <= 1-r):
                return False
        
        # Check non-overlap using efficient KDTree
        if len(circles) > 1:
            tree = cKDTree(circles[:, :2])
            # Query pairs with minimum distance to detect overlaps
            pairs = tree.query_pairs(r=0.0001)  # Very small threshold to catch overlaps
            for i, j in pairs:
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < (r1 + r2):
                    return False
        
        return True
    
    # Objective function for optimization
    def objective(circles_flat):
        # Return negative sum of radii (we want to maximize sum of radii)
        return -np.sum(circles_flat[2::3])
    
    # Constraint function for scipy optimization
    def constraint_func(circles_flat):
        circles = circles_flat.reshape(-1, 3)
        constraints = []
        
        # Containment constraints: r >= 0, x >= r, y >= r, 1-x >= r, 1-y >= r
        for i in range(n):
            x, y, r = circles[i]
            # All should be >= 0 for feasibility
            constraints.extend([
                r,           # r >= 0
                x - r,       # x >= r
                y - r,       # y >= r
                1 - x - r,   # 1-x >= r
                1 - y - r    # 1-y >= r
            ])
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                # distance >= r1 + r2 (so distance - r1 - r2 >= 0)
                constraints.append(distance - (r1 + r2))
                
        return np.array(constraints)
    
    # Multi-stage refinement approach
    def multi_stage_refinement(initial_circles):
        circles = initial_circles.copy()
        
        # Stage 1: Force-based optimization with high learning rate
        max_iter_stage1 = 500
        learning_rate_stage1 = 0.05
        
        for iteration in range(max_iter_stage1):
            forces = np.zeros_like(circles)
            
            # Repulsion forces (non-overlap)
            for i in range(n):
                for j in range(i+1, n):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < (circles[i, 2] + circles[j, 2]) and dist > 0.001:
                        # Overlapping - push apart
                        force_magnitude = (circles[i, 2] + circles[j, 2] - dist) / (dist + 0.001)
                        forces[i, 0] += force_magnitude * dx / dist
                        forces[i, 1] += force_magnitude * dy / dist
                        forces[j, 0] -= force_magnitude * dx / dist
                        forces[j, 1] -= force_magnitude * dy / dist
            
            # Boundary forces (push away from edges)
            for i in range(n):
                if circles[i, 0] < circles[i, 2]:
                    forces[i, 0] += (circles[i, 2] - circles[i, 0]) * 100
                elif circles[i, 0] > 1 - circles[i, 2]:
                    forces[i, 0] -= (circles[i, 0] - (1 - circles[i, 2])) * 100
                    
                if circles[i, 1] < circles[i, 2]:
                    forces[i, 1] += (circles[i, 2] - circles[i, 1]) * 100
                elif circles[i, 1] > 1 - circles[i, 2]:
                    forces[i, 1] -= (circles[i, 1] - (1 - circles[i, 2])) * 100
            
            # Update positions
            circles[:, :2] += forces[:, :2] * learning_rate_stage1
            
            # Keep within bounds
            circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], 1 - circles[:, 2])
            circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], 1 - circles[:, 2])
            
            # Periodic checks
            if iteration % 100 == 0:
                total_radius = np.sum(circles[:, 2])
                #print(f"Stage 1 Iteration {iteration}: Total radius = {total_radius:.4f}")
        
        # Stage 2: Gradual radius increase with constraint checking
        max_iter_stage2 = 500
        learning_rate_stage2 = 0.02
        
        for iteration in range(max_iter_stage2):
            # Try to increase radii gradually
            for i in range(n):
                if circles[i, 2] < 0.49:
                    # Check if we can safely increase radius
                    can_increase = True
                    for j in range(n):
                        if i != j:
                            dx = circles[i, 0] - circles[j, 0]
                            dy = circles[i, 1] - circles[j, 1]
                            dist = np.sqrt(dx*dx + dy*dy)
                            # Allow some tolerance for radius increase
                            if dist < (circles[i, 2] + circles[j, 2] + 0.005):
                                can_increase = False
                                break
                    if can_increase:
                        circles[i, 2] = min(0.49, circles[i, 2] + 0.002)
            
            # Apply force-based refinement again
            forces = np.zeros_like(circles)
            
            # Repulsion forces
            for i in range(n):
                for j in range(i+1, n):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < (circles[i, 2] + circles[j, 2]) and dist > 0.001:
                        force_magnitude = (circles[i, 2] + circles[j, 2] - dist) / (dist + 0.001)
                        forces[i, 0] += force_magnitude * dx / dist
                        forces[i, 1] += force_magnitude * dy / dist
                        forces[j, 0] -= force_magnitude * dx / dist
                        forces[j, 1] -= force_magnitude * dy / dist
            
            # Boundary forces
            for i in range(n):
                if circles[i, 0] < circles[i, 2]:
                    forces[i, 0] += (circles[i, 2] - circles[i, 0]) * 50
                elif circles[i, 0] > 1 - circles[i, 2]:
                    forces[i, 0] -= (circles[i, 0] - (1 - circles[i, 2])) * 50
                    
                if circles[i, 1] < circles[i, 2]:
                    forces[i, 1] += (circles[i, 2] - circles[i, 1]) * 50
                elif circles[i, 1] > 1 - circles[i, 2]:
                    forces[i, 1] -= (circles[i, 1] - (1 - circles[i, 2])) * 50
            
            # Update positions
            circles[:, :2] += forces[:, :2] * learning_rate_stage2
            
            # Keep within bounds
            circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], 1 - circles[:, 2])
            circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], 1 - circles[:, 2])
            circles[:, 2] = np.clip(circles[:, 2], 0.001, 0.49)
            
            # Periodic checks
            if iteration % 100 == 0:
                total_radius = np.sum(circles[:, 2])
                #print(f"Stage 2 Iteration {iteration}: Total radius = {total_radius:.4f}")
        
        return circles
    
    # Generate initial configuration
    circles = optimized_initialization()
    
    # Validate initial solution
    if not check_constraints(circles):
        # Fix constraints if needed
        for i in range(n):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
    
    # Refine using multi-stage approach
    circles = multi_stage_refinement(circles)
    
    # Final validation and cleanup
    if not check_constraints(circles):
        # Try to fix any constraint violations
        for i in range(n):
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
    
    return circles


# EVOLVE-BLOCK-END
