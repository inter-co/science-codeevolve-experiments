# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import random
from scipy.spatial import cKDTree
from itertools import combinations

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization and advanced optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Phase 1: Improved initialization using a more systematic approach
    def generate_better_initialization():
        """Generate initial configuration using a combination of grid-based and optimization-based approach"""
        # Strategy: Start with a dense grid and then refine
        circles = []
        
        # Create a denser grid pattern
        grid_size = 6
        spacing_x = 1.0 / (grid_size + 1)
        spacing_y = 1.0 / (grid_size + 1)
        
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= n:
                    break
                # Add jitter to positions for better distribution
                x = spacing_x * (j + 1) + (random.random() - 0.5) * spacing_x * 0.3
                y = spacing_y * (i + 1) + (random.random() - 0.5) * spacing_y * 0.3
                if 0 <= x <= 1 and 0 <= y <= 1:
                    positions.append([x, y])
        
        # Ensure we have exactly n positions
        positions = positions[:n]
        
        # Initialize radii based on available space
        radii = []
        for i, (x, y) in enumerate(positions):
            # Max radius is limited by distance to edges and other circles
            max_r = min(x, 1-x, y, 1-y)
            
            # Start with a small fraction of max radius
            r = max_r * 0.25
            
            # Check if we can make it larger without overlapping others
            # This is a simplified version - we'll optimize this later
            radii.append(r)
        
        # Validate initial configuration and fix overlaps
        # Start with simple greedy approach to resolve overlaps
        for _ in range(10):  # Allow a few passes to resolve overlaps
            # Check overlaps and reduce radii if needed
            changed = False
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    r1 = radii[i]
                    r2 = radii[j]
                    
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    min_dist = (r1 + r2)**2
                    
                    if dist_sq < min_dist:
                        # Overlap detected, reduce radii
                        # Reduce both radii proportionally
                        reduction = (min_dist - dist_sq) / (min_dist + 1e-10)
                        radii[i] = max(0.001, radii[i] * (1 - reduction * 0.5))
                        radii[j] = max(0.001, radii[j] * (1 - reduction * 0.5))
                        changed = True
            
            if not changed:
                break
        
        # Final validation and adjustment
        final_circles = []
        for i in range(n):
            x, y = positions[i]
            r = min(radii[i], 0.49)  # Cap radius to prevent too large circles
            # Ensure circle fits within bounds
            r = min(r, x, 1-x, y, 1-y)
            final_circles.append((x, y, r))
            
        return final_circles
    
    # Generate initial configuration
    initial_circles = generate_better_initialization()
    
    # Phase 2: Set up optimization variables
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
    initial_vars = []
    for circle in initial_circles:
        x, y, r = circle
        initial_vars.extend([x, y, r])
    
    # Phase 3: Define constraint functions with improved handling
    def contain_constraints(vars):
        """Ensure all circles are contained in the unit square"""
        cons = []
        for i in range(n):
            idx = 3*i
            x, y, r = vars[idx], vars[idx+1], vars[idx+2]
            # Circle must be contained in unit square
            cons.append(x - r)  # x - r >= 0
            cons.append(1 - x - r)  # 1 - x - r >= 0
            cons.append(y - r)  # y - r >= 0
            cons.append(1 - y - r)  # 1 - y - r >= 0
        return np.array(cons)
    
    def overlap_constraints(vars):
        """Ensure no overlapping circles - using efficient computation with early termination"""
        cons = []
        
        # For performance, only consider pairs that might actually overlap
        # We'll do this more efficiently by checking distances properly
        for i in range(n):
            idx = 3*i
            x, y, r = vars[idx], vars[idx+1], vars[idx+2]
            
            # Check against all other circles
            for j in range(i+1, n):
                idx_j = 3*j
                x_j, y_j, r_j = vars[idx_j], vars[idx_j+1], vars[idx_j+2]
                
                # Distance constraint: d >= r_i + r_j
                dist_sq = (x - x_j)**2 + (y - y_j)**2
                # We want dist >= r_i + r_j, so we enforce: dist - r_i - r_j >= 0
                # But we're computing dist^2, so we compute sqrt(dist_sq) - (r + r_j)
                dist = np.sqrt(dist_sq)
                cons.append(dist - r - r_j)
                
        return np.array(cons)
    
    # Phase 4: Define objective function
    def objective(vars):
        """Maximize sum of radii (minimize negative sum)"""
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i + 2]  # radius is third component
        return -total_radius  # Negative because we minimize
    
    # Phase 5: Run optimization with enhanced robustness
    try:
        # Define bounds for variables with tighter constraints
        bounds = []
        for i in range(n):
            # x bounds: [r, 1-r] 
            bounds.append((0.001, 0.999))  # x coordinate
            bounds.append((0.001, 0.999))  # y coordinate
            bounds.append((0.001, 0.499))  # radius (limit to prevent too large circles)
        
        # Create constraint dictionaries
        constraints = [
            {'type': 'ineq', 'fun': contain_constraints},
            {'type': 'ineq', 'fun': overlap_constraints}
        ]
        
        # Try multiple optimization methods with different starting points
        methods_to_try = ['trust-constr', 'SLSQP']
        best_result = None
        best_value = float('-inf')
        
        # Multiple random restarts for better convergence
        for restart in range(5):  # Increase restarts for better chance
            # Slightly perturb initial solution for diversity
            perturbed_vars = initial_vars.copy()
            for i in range(0, len(perturbed_vars), 3):
                # Perturb x and y slightly, and also adjust radius
                if i+0 < len(perturbed_vars):
                    perturbed_vars[i] += (random.random() - 0.5) * 0.03
                    perturbed_vars[i] = max(0.001, min(0.999, perturbed_vars[i]))
                if i+1 < len(perturbed_vars):
                    perturbed_vars[i+1] += (random.random() - 0.5) * 0.03
                    perturbed_vars[i+1] = max(0.001, min(0.999, perturbed_vars[i+1]))
                if i+2 < len(perturbed_vars):
                    perturbed_vars[i+2] += (random.random() - 0.5) * 0.02
                    perturbed_vars[i+2] = max(0.001, min(0.499, perturbed_vars[i+2]))
            
            for method in methods_to_try:
                try:
                    result = minimize(
                        objective,
                        perturbed_vars,
                        method=method,
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 2000, 'ftol': 1e-6, 'gtol': 1e-6}
                    )
                    
                    if result.success:
                        # Calculate the actual sum of radii
                        current_sum = -objective(result.x)
                        if current_sum > best_value:
                            best_value = current_sum
                            best_result = result
                except:
                    continue
        
        # If we found a good result, use it; otherwise fall back to initial
        if best_result is not None and best_result.success:
            final_vars = best_result.x
        else:
            # Fall back to initial configuration with more aggressive refinement
            final_vars = initial_vars.copy()
            
            # More sophisticated refinement using local search with better strategy
            try:
                # Perform a few iterations of local optimization
                for iter in range(30):  # More iterations
                    improved = False
                    # Try to improve each circle individually
                    for i in range(n):
                        # Save current state
                        old_x, old_y, old_r = final_vars[3*i], final_vars[3*i+1], final_vars[3*i+2]
                        
                        # Try to increase radius while maintaining constraints
                        x, y, r = old_x, old_y, old_r
                        
                        # Find maximum possible radius
                        max_r = min(x, 1-x, y, 1-y)
                        
                        # Check what we can increase to
                        new_r = min(max_r, r + 0.015)  # Smaller increment for fine-tuning
                        
                        # Check overlap constraints with neighbors
                        valid = True
                        for j in range(n):
                            if i != j:
                                x_j, y_j, r_j = final_vars[3*j], final_vars[3*j+1], final_vars[3*j+2]
                                dist = np.sqrt((x - x_j)**2 + (y - y_j)**2)
                                if dist < new_r + r_j:
                                    valid = False
                                    break
                        
                        if valid and new_r > r:
                            final_vars[3*i + 2] = new_r
                            improved = True
                    
                    if not improved:
                        break
            except:
                pass
        
        # Extract final result
        circles = np.zeros((n, 3))
        for i in range(n):
            circles[i] = [final_vars[3*i], final_vars[3*i+1], final_vars[3*i+2]]
            
        return circles
        
    except Exception as e:
        # Fallback: return the initial configuration if optimization fails
        circles = np.zeros((n, 3))
        for i in range(n):
            x, y, r = initial_circles[i]
            circles[i] = [x, y, r]
        return circles


# EVOLVE-BLOCK-END
