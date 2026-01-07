# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import random
import time
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, evolutionary algorithm-inspired 
    optimization, and constraint handling to beat the benchmark.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    max_time = 55  # Leave buffer for final processing
    start_time = time.time()
    
    # Efficient validity check using spatial indexing for overlap detection
    def check_validity(circles: np.ndarray) -> bool:
        """Check if all circles are valid (containment and non-overlap)"""
        # Check containment
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
                return False
        
        # Check overlaps efficiently using spatial data structure
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Build KDTree for efficient neighbor search
        tree = cKDTree(positions)
        
        # For each circle, find neighbors within 2*(r1+r2) distance
        for i in range(n):
            x, y, r = circles[i]
            # Find nearby circles that might overlap
            nearby = tree.query_ball_point([x, y], 2 * (r + max(radii)), p=2)
            
            # Check actual overlaps with nearby circles
            for j in nearby:
                if i != j:
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                    if dist_sq < (r1 + r2)**2:
                        return False
        return True
    
    # Objective function to maximize sum of radii
    def objective(vars):
        # vars contains [x1, y1, r1, x2, y2, r2, ...]
        return -sum(vars[2::3])  # Negative because we want to maximize
    
    # Constraint function for scipy optimization with better numerical handling
    def constraint_func(vars):
        constraints = []
        # Non-overlap constraints with safety margin
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                # Safety margin to prevent numerical issues
                overlap = np.sqrt(dist_sq) - (r1 + r2) - 1e-8
                constraints.append(overlap)  # Should be >= 0
            
            # Containment constraints  
            x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
            constraints.extend([
                x - r,      # x >= r
                y - r,      # y >= r
                1 - x - r,  # 1-x >= r
                1 - y - r   # 1-y >= r
            ])
        return constraints
    
    # Local optimization refinement function with better error handling
    def refine_solution(initial_solution):
        """Refine solution using local optimization"""
        # Convert to flat vars for optimization
        vars = []
        for i in range(n):
            vars.extend([initial_solution[i, 0], initial_solution[i, 1], initial_solution[i, 2]])
        
        # Define bounds more carefully
        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Optimize with bounds and constraints
        try:
            result = minimize(
                objective,
                vars,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 500, 'ftol': 1e-6, 'eps': 1e-6, 'disp': False}
            )
            
            if result.success:
                refined = np.zeros((n, 3))
                for i in range(n):
                    refined[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
                return refined
        except Exception as e:
            # If optimization fails, return the original solution
            pass
        return initial_solution
    
    # Enhanced initialization with fewer but better strategies
    def initialize_better_config():
        # Start with a good hexagonal pattern
        circles = []
        rows = 6
        cols = 6
        
        # Create hexagonal grid pattern
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        for i in range(rows):
            for j in range(cols):
                if len(circles) >= n:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                circles.append([x, y])
        
        # Fill remaining circles with a good distribution
        while len(circles) < n:
            # Use a more strategic placement
            x = 0.5 + np.random.normal(0, 0.15)
            y = 0.5 + np.random.normal(0, 0.15)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            circles.append([x, y])
        
        # Initialize with reasonable radii
        circles_array = np.zeros((n, 3))
        for i in range(n):
            x, y = circles[i]
            # Initial radius based on distance to nearest boundary
            r = min(x, y, 1-x, 1-y) * 0.3
            # Add some randomness but keep it reasonable
            r *= random.uniform(0.7, 1.0)
            # Ensure reasonable minimum radius
            r = max(0.001, r)
            circles_array[i] = [x, y, r]
        
        return circles_array
    
    # Single optimized initialization with better convergence
    best_sum = 0
    best_solution = None
    
    # Try just one good initialization strategy with refinement
    # This is more efficient than trying many different strategies
    try:
        # Generate initial configuration using better pattern
        initial_solution = initialize_better_config()
        
        # First, try a simple refinement with basic constraints
        refined = refine_solution(initial_solution)
        
        # Check validity and compute sum
        if check_validity(refined):
            current_sum = np.sum(refined[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution = refined.copy()
        
        # If that didn't work well, try a second round with different random seed
        if best_solution is None or best_sum < 2.8:  # Only if we're not doing well
            np.random.seed(123)  # Different seed
            initial_solution2 = initialize_better_config()
            refined2 = refine_solution(initial_solution2)
            
            if check_validity(refined2):
                current_sum2 = np.sum(refined2[:, 2])
                if current_sum2 > best_sum:
                    best_sum = current_sum2
                    best_solution = refined2.copy()
                    
    except Exception as e:
        pass
    
    # If no good solution found, fallback to a robust initialization
    if best_solution is None:
        # Use a simpler but reliable approach
        circles = np.zeros((n, 3))
        
        # Create a reasonably spaced pattern
        for i in range(n):
            row = i // 6
            col = i % 6
            x = (col + 0.5) * (1.0 / 6) + random.uniform(-0.03, 0.03)
            y = (row + 0.5) * (1.0 / 6) + random.uniform(-0.03, 0.03)
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            r = random.uniform(0.03, 0.07)  # More uniform radii
            circles[i] = [x, y, r]
        
        # Try local optimization on this
        best_solution = refine_solution(circles)
    
    # Final validation and cleanup
    if best_solution is not None:
        # Ensure all circles are valid
        if check_validity(best_solution):
            # Final adjustment to ensure boundaries are respected
            for i in range(n):
                x, y, r = best_solution[i]
                # Make sure it's within bounds
                x = max(r, min(1-r, x))
                y = max(r, min(1-r, y))
                best_solution[i] = [x, y, r]
        else:
            # If still not valid, do final correction
            for _ in range(30):  # Reduced iterations to save time
                for i in range(n):
                    x, y, r = best_solution[i]
                    # Adjust position to stay within bounds
                    x = max(r, min(1-r, x))
                    y = max(r, min(1-r, y))
                    best_solution[i] = [x, y, r]
                    
                if check_validity(best_solution):
                    break
                # Reduce radii slightly
                best_solution[:, 2] *= 0.95
    
    # Final fallback if everything fails
    if best_solution is None:
        best_solution = np.zeros((n, 3))
        for i in range(n):
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.95)
            r = random.uniform(0.02, 0.08)
            best_solution[i] = [x, y, r]
    
    return best_solution


# EVOLVE-BLOCK-END
