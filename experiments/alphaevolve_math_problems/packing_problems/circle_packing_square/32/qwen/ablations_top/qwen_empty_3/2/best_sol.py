# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import time

# Global constants
N_CIRCLES = 32
BENCHMARK = 2.937944526205518

def solve_circle_packing_mathematical_programming() -> np.ndarray:
    """
    Solve circle packing using mathematical programming approach.
    This is a fundamentally different approach from the original evolutionary/physics methods.
    """
    
    # Initialize with a structured approach
    circles = np.zeros((N_CIRCLES, 3))
    
    # Use a systematic grid-based initialization that's more mathematically sound
    # Create a grid pattern that naturally avoids overlaps
    grid_rows = 6
    grid_cols = 6
    
    # Adjust grid to fit exactly 32 circles
    if grid_rows * grid_cols < N_CIRCLES:
        grid_rows = 5
        grid_cols = 7
    
    spacing_x = 1.0 / grid_cols
    spacing_y = 1.0 / grid_rows
    
    # Use a more sophisticated approach - start with regular grid then optimize
    idx = 0
    for i in range(grid_rows):
        for j in range(grid_cols):
            if idx >= N_CIRCLES:
                break
            # Position at center of grid cell
            x = (j + 0.5) * spacing_x
            y = (i + 0.5) * spacing_y
            
            # Initial radius - start with small but reasonable values
            # This helps avoid being trapped in poor local optima early on
            r = min(spacing_x, spacing_y) * 0.35
            
            circles[idx] = [x, y, r]
            idx += 1
        if idx >= N_CIRCLES:
            break
    
    # Fill remaining positions with strategic placement
    for i in range(idx, N_CIRCLES):
        # Place in a way that tries to distribute evenly
        # Use golden ratio-like distribution for better spread
        angle = (i * 2.414213562373095) % (2 * np.pi)  # Golden angle increment
        radius = 0.4 * np.sqrt(i / N_CIRCLES)  # Radial distribution
        x = 0.5 + radius * np.cos(angle) * 0.4
        y = 0.5 + radius * np.sin(angle) * 0.4
        r = random.uniform(0.02, 0.08)
        circles[i] = [x, y, r]
    
    # Try multiple optimization approaches to get the best solution
    # Approach 1: Direct mathematical programming with trust-constr
    try:
        # Define optimization variables: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
        def objective(params):
            # Extract circles from flattened parameters
            radius_sum = 0
            for i in range(N_CIRCLES):
                radius_sum += params[3*i + 2]  # radius is third component
            return -radius_sum  # negative because we're minimizing in scipy
        
        # Create bounds for all parameters
        bounds = []
        for i in range(N_CIRCLES):
            # x bounds: [r, 1-r] 
            bounds.append((0.001, 0.999))
            # y bounds: [r, 1-r]
            bounds.append((0.001, 0.999))
            # r bounds: [0.001, 0.4] - reasonable limits
            bounds.append((0.001, 0.4))
        
        # Flatten initial configuration
        initial_params = []
        for i in range(N_CIRCLES):
            initial_params.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
        
        # Use trust-constr solver which handles constraints well
        result = minimize(
            objective,
            initial_params,
            method='trust-constr',
            bounds=bounds,
            options={'maxiter': 300, 'disp': False}
        )
        
        # Reconstruct circles from optimized parameters
        final_circles = np.zeros((N_CIRCLES, 3))
        for i in range(N_CIRCLES):
            final_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
        
        # Validate and return if successful
        if validate_solution(final_circles):
            return final_circles
            
    except Exception:
        pass
    
    # Approach 2: Try L-BFGS-B as backup
    try:
        # Define optimization variables: [x1, y1, r1, x2, y2, r2, ..., x32, y32, r32]
        def objective(params):
            # Extract circles from flattened parameters
            radius_sum = 0
            for i in range(N_CIRCLES):
                radius_sum += params[3*i + 2]  # radius is third component
            return -radius_sum  # negative because we're minimizing in scipy
        
        # Create bounds for all parameters
        bounds = []
        for i in range(N_CIRCLES):
            # x bounds: [r, 1-r] 
            bounds.append((0.001, 0.999))
            # y bounds: [r, 1-r]
            bounds.append((0.001, 0.999))
            # r bounds: [0.001, 0.4] - reasonable limits
            bounds.append((0.001, 0.4))
        
        # Flatten initial configuration
        initial_params = []
        for i in range(N_CIRCLES):
            initial_params.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
        
        # Use L-BFGS-B solver
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300}
        )
        
        # Reconstruct circles from optimized parameters
        final_circles = np.zeros((N_CIRCLES, 3))
        for i in range(N_CIRCLES):
            final_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
        
        # Validate and return if successful
        if validate_solution(final_circles):
            return final_circles
            
    except Exception:
        pass
    
    # Fallback approach: iterative improvement with careful constraint handling
    return iterative_improvement_approach(circles)

def validate_solution(circles: np.ndarray) -> bool:
    """Validate that solution satisfies all constraints"""
    # Check containment
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlaps
    positions = circles[:, :2]
    radii = circles[:, 2]
    
    if len(circles) < 2:
        return True
    
    # Vectorized overlap checking
    dist_matrix = cdist(positions, positions)
    
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            if dist_matrix[i, j] < (radii[i] + radii[j]):
                return False
    
    return True

def iterative_improvement_approach(initial_circles: np.ndarray) -> np.ndarray:
    """A different approach using systematic improvement with mathematical constraints"""
    circles = initial_circles.copy()
    
    # Use a simple but effective iterative approach
    for iteration in range(700):  # Even more iterations for better convergence
        # Store previous state
        prev_circles = circles.copy()
        
        # Simple gradient descent approach with constraint enforcement
        for i in range(N_CIRCLES):
            # Try small perturbations to find better configuration
            best_circles = circles.copy()
            best_radius_sum = np.sum(circles[:, 2])
            
            # Try several random perturbations
            for _ in range(35):  # Even more attempts for better exploration
                test_circles = circles.copy()
                
                # Small random changes to position and radius
                delta_x = random.uniform(-0.035, 0.035)
                delta_y = random.uniform(-0.035, 0.035)
                delta_r = random.uniform(-0.03, 0.03)
                
                # Apply changes
                test_circles[i, 0] = max(0.001, min(0.999, circles[i, 0] + delta_x))
                test_circles[i, 1] = max(0.001, min(0.999, circles[i, 1] + delta_y))
                test_circles[i, 2] = max(0.001, min(0.4, circles[i, 2] + delta_r))
                
                # Ensure position stays within bounds given radius
                test_circles[i, 0] = max(test_circles[i, 2], min(1-test_circles[i, 2], test_circles[i, 0]))
                test_circles[i, 1] = max(test_circles[i, 2], min(1-test_circles[i, 2], test_circles[i, 1]))
                
                # Check if this improves the solution
                if validate_solution(test_circles):
                    new_sum = np.sum(test_circles[:, 2])
                    if new_sum > best_radius_sum:
                        best_circles = test_circles
                        best_radius_sum = new_sum
            
            circles = best_circles
        
        # Check for convergence with looser tolerance for better performance
        if np.allclose(prev_circles, circles, atol=5e-6):
            break
    
    # Final cleanup
    for i in range(N_CIRCLES):
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1-circles[i, 2])
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1-circles[i, 2])
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a mathematical programming approach instead of evolutionary/physics methods.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Use the mathematical programming approach
    circles = solve_circle_packing_mathematical_programming()
    
    # Ensure final validation
    if not validate_solution(circles):
        # If somehow invalid, use a fallback approach
        circles = np.zeros((N_CIRCLES, 3))
        # Grid-based initialization with good starting points
        rows, cols = 6, 6
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N_CIRCLES:
                    break
                x = (j + 0.5) * spacing_x
                y = (i + 0.5) * spacing_y
                r = min(spacing_x, spacing_y) * 0.35
                circles[idx] = [x, y, r]
                idx += 1
            if idx >= N_CIRCLES:
                break
    
    return circles


# EVOLVE-BLOCK-END
