# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random
from typing import Tuple
import math
from itertools import combinations
import time
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric initialization, evolutionary search of aspect ratios, and mathematical optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Focus on ratios that have proven successful in circle packing literature
    best_sum = 0
    best_circles = None
    
    # Use more carefully selected aspect ratios based on successful patterns
    # Include some extreme ratios that often work well in practice
    ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    # Track timing to stay within limits
    start_time = time.time()
    
    # Try different optimization methods and strategies
    optimization_methods = ['SLSQP', 'trust-constr']
    
    # Try multiple passes with different random seeds for robustness
    for seed in [42, 123, 456, 789]:
        random.seed(seed)
        np.random.seed(seed)
        
        for ratio in ratios:
            if time.time() - start_time > 55:  # Leave some time for final processing
                break
                
            width = 2 * ratio / (1 + ratio)
            height = 2 / (1 + ratio)
            
            # Try multiple initialization strategies
            init_strategies = [
                initialize_hexagonal_pattern(width, height, 21),
                initialize_grid_pattern(width, height, 21),
                initialize_balanced_random(width, height, 21),
                initialize_focused_placement(width, height, 21),
                initialize_systematic_placement(width, height, 21)
            ]
            
            for strategy_idx, circles in enumerate(init_strategies):
                if time.time() - start_time > 55:
                    break
                    
                # Try multiple optimization approaches for robustness
                for method in optimization_methods:
                    if time.time() - start_time > 55:
                        break
                        
                    try:
                        optimized_circles = optimize_mathematical_programming(circles, width, height, method)
                        current_sum = np.sum(optimized_circles[:, 2])
                        
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_circles = optimized_circles.copy()
                    except Exception:
                        continue
    
    # Final refinement with best configuration using multiple passes
    if best_circles is not None:
        # Try one more optimization pass with better parameters
        try:
            # Use a more precise rectangle dimension
            width = best_circles[0, 0] * 2
            height = best_circles[0, 1] * 2
            final_circles = optimize_mathematical_programming(best_circles, width, height, 'trust-constr')
            final_sum = np.sum(final_circles[:, 2])
            if final_sum > best_sum:
                best_circles = final_circles
        except Exception:
            pass
    
    # If nothing worked, fall back to a decent initialization
    if best_circles is None:
        width, height = 1.0, 1.0
        circles = initialize_hexagonal_pattern(width, height, 21)
        best_circles = optimize_mathematical_programming(circles, width, height, 'SLSQP')
    
    return best_circles


def initialize_hexagonal_pattern(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circle positions using a hexagonal lattice pattern."""
    circles = np.zeros((n, 3))
    
    # For 21 circles, arrange in roughly 5 rows x 4 columns with offset rows
    rows = 5
    cols = 4
    
    # Calculate spacing based on available space
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)
    
    # Adjust for hexagonal packing efficiency
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    # Place circles in hexagonal pattern
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
                
            # Offset every other row for hexagonal packing
            offset = spacing_x * 0.5 if row % 2 == 1 else 0.0
            
            x = (col + 1) * spacing_x + offset
            y = (row + 1) * spacing_y
            
            # Ensure within bounds with safety margin
            safe_margin = spacing_x * 0.2
            x = max(safe_margin, min(width - safe_margin, x))
            y = max(safe_margin, min(height - safe_margin, y))
            
            # Reasonable initial radius
            r = min(spacing_x, spacing_y) * 0.4
            
            circles[idx] = [x, y, r]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining circles strategically
    np.random.seed(42)
    for i in range(idx, n):
        # Better random positioning
        x = random.uniform(spacing_x * 0.5, width - spacing_x * 0.5)
        y = random.uniform(spacing_y * 0.5, height - spacing_y * 0.5)
        r = random.uniform(0.01, min(spacing_x, spacing_y) * 0.3)
        circles[i] = [x, y, r]
    
    return circles


def initialize_grid_pattern(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circle positions using a regular grid pattern."""
    circles = np.zeros((n, 3))
    
    # Use a 4x5 grid for 21 circles (4 rows, 5 columns)
    rows = 4
    cols = 5
    
    # Calculate spacing
    spacing_x = width / (cols + 1)
    spacing_y = height / (rows + 1)
    
    # Calculate max radius based on spacing
    max_radius = min(spacing_x, spacing_y) * 0.3
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 1) * spacing_x
            y = (i + 1) * spacing_y
            
            # Ensure within bounds
            safe_margin = max_radius * 1.2
            x = max(safe_margin, min(width - safe_margin, x))
            y = max(safe_margin, min(height - safe_margin, y))
            
            circles[idx] = [x, y, max_radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining circles with random distribution
    for i in range(idx, n):
        x = random.uniform(max_radius * 1.2, width - max_radius * 1.2)
        y = random.uniform(max_radius * 1.2, height - max_radius * 1.2)
        radius = random.uniform(0.01, max_radius * 0.5)
        circles[i] = [x, y, radius]
    
    return circles


def initialize_balanced_random(width: float, height: float, n: int) -> np.ndarray:
    """Initialize with a balanced random approach that considers spatial distribution."""
    circles = np.zeros((n, 3))
    
    # Use a more strategic random approach that tries to avoid dense clusters
    np.random.seed(42)
    
    # First, try placing some circles in grid-like positions to create structure
    grid_rows = 4
    grid_cols = 5
    placed_count = 0
    
    # Place some circles in a structured way
    for i in range(grid_rows):
        for j in range(grid_cols):
            if placed_count >= n:
                break
            x = (j + 1) * width / (grid_cols + 1)
            y = (i + 1) * height / (grid_rows + 1)
            
            # Add small random offset to prevent perfect alignment
            offset_range = min(width, height) * 0.05
            x += random.uniform(-offset_range, offset_range)
            y += random.uniform(-offset_range, offset_range)
            
            # Clip to bounds
            x = max(0.1, min(width - 0.1, x))
            y = max(0.1, min(height - 0.1, y))
            
            # Set radius based on proximity to edges
            max_radius = min(x, width - x, y, height - y) * 0.4
            r = random.uniform(0.01, max_radius * 0.5)
            
            circles[placed_count] = [x, y, r]
            placed_count += 1
    
    # Fill remaining slots with random placement
    for i in range(placed_count, n):
        x = random.uniform(0.1, width - 0.1)
        y = random.uniform(0.1, height - 0.1)
        max_radius = min(x, width - x, y, height - y) * 0.4
        r = random.uniform(0.01, max_radius * 0.3)
        circles[i] = [x, y, r]
    
    return circles


def initialize_systematic_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circle positions using a systematic approach based on geometric packing principles."""
    circles = np.zeros((n, 3))
    
    # For 21 circles, arrange in approximately 4x5 grid with some spacing
    rows = 4
    cols = 5
    
    # Calculate grid spacing
    cell_width = width / cols
    cell_height = height / rows
    
    # Place circles in grid with slight offset to allow for optimization
    for i in range(rows):
        for j in range(cols):
            if len(circles[circles[:, 2] > 0]) >= n:
                break
            # Position in cell
            x = (j + 0.5) * cell_width
            y = (i + 0.5) * cell_height
            
            # Add small random offset to prevent perfect alignment
            x += random.uniform(-cell_width * 0.1, cell_width * 0.1)
            y += random.uniform(-cell_height * 0.1, cell_height * 0.1)
            
            # Keep within bounds
            x = max(cell_width * 0.2, min(width - cell_width * 0.2, x))
            y = max(cell_height * 0.2, min(height - cell_height * 0.2, y))
            
            # Set initial radius based on available space
            max_radius = min(x, width - x, y, height - y) * 0.4
            radius = max(0.01, max_radius * random.uniform(0.5, 0.9))
            
            # Find next empty slot
            idx = len(circles[circles[:, 2] > 0])
            if idx < n:
                circles[idx] = [x, y, radius]
    
    # Fill remaining slots with random placements
    for i in range(len(circles[circles[:, 2] > 0]), n):
        x = random.uniform(0.1, width - 0.1)
        y = random.uniform(0.1, height - 0.1)
        radius = random.uniform(0.01, min(width, height) * 0.1)
        circles[i] = [x, y, radius]
    
    return circles


def initialize_focused_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize with focused placement strategy that maximizes early circle utilization."""
    circles = np.zeros((n, 3))
    
    # Start with a dense central arrangement and expand outward
    # This helps avoid getting trapped in local optima early on
    
    # Place first few circles in a central cluster
    center_x, center_y = width / 2, height / 2
    num_center = min(6, n)  # Place up to 6 circles in center cluster
    
    # Place circles in a tight cluster near center
    for i in range(num_center):
        angle = 2 * math.pi * i / num_center
        radius = min(width, height) * 0.1 * (0.5 + 0.3 * random.random())
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        
        # Ensure within bounds
        x = max(0.1, min(width - 0.1, x))
        y = max(0.1, min(height - 0.1, y))
        
        # Set reasonable radius
        max_radius = min(x, width - x, y, height - y) * 0.3
        r = random.uniform(0.01, max_radius * 0.4)
        circles[i] = [x, y, r]
    
    # Fill remaining with strategic placement
    for i in range(num_center, n):
        # Use a more diverse sampling approach
        if random.random() < 0.3:  # 30% chance to place near edge
            # Near edge placement
            side = random.choice(['left', 'right', 'top', 'bottom'])
            if side == 'left':
                x = random.uniform(0.05, 0.15)
                y = random.uniform(0.1, height - 0.1)
            elif side == 'right':
                x = random.uniform(width - 0.15, width - 0.05)
                y = random.uniform(0.1, height - 0.1)
            elif side == 'top':
                x = random.uniform(0.1, width - 0.1)
                y = random.uniform(height - 0.15, height - 0.05)
            else:  # bottom
                x = random.uniform(0.1, width - 0.1)
                y = random.uniform(0.05, 0.15)
        else:
            # Random placement
            x = random.uniform(0.1, width - 0.1)
            y = random.uniform(0.1, height - 0.1)
        
        # Set radius based on proximity to edges
        max_radius = min(x, width - x, y, height - y) * 0.4
        r = random.uniform(0.01, max_radius * 0.3)
        circles[i] = [x, y, r]
    
    return circles


def optimize_mathematical_programming(initial_circles: np.ndarray, width: float, height: float, method: str = 'SLSQP') -> np.ndarray:
    """
    Optimize using a mathematical programming approach with explicit constraints.
    """
    # Create a structured optimization problem
    n = len(initial_circles)
    
    # Define parameter bounds: [x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    bounds = []
    for i in range(n):
        # x bounds - leave some margin for optimization
        bounds.append((initial_circles[i, 2], width - initial_circles[i, 2]))
        # y bounds  
        bounds.append((initial_circles[i, 2], height - initial_circles[i, 2]))
        # r bounds - more conservative to prevent numerical issues
        bounds.append((0.001, min(width, height) * 0.4))
    
    # Flatten initial solution
    initial_solution = initial_circles.flatten()
    
    # Define the optimization objective
    def objective(params):
        # Reshape to circles array
        circles = params.reshape(n, 3)
        # Maximize sum of radii (minimize negative sum)
        return -np.sum(circles[:, 2])
    
    # Define constraints with improved numerical stability
    def constraint_function(params):
        circles = params.reshape(n, 3)
        
        constraints = []
        
        # Boundary constraints: each circle must be fully inside rectangle
        for i in range(n):
            x, y, r = circles[i]
            constraints.extend([
                x - r,           # Left boundary
                width - x - r,   # Right boundary  
                y - r,           # Bottom boundary
                height - y - r   # Top boundary
            ])
        
        # Non-overlap constraints: distance >= sum of radii
        # Use more numerically stable approach with better error handling
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                distance_squared = dx*dx + dy*dy
                # Use squared distance to avoid sqrt computation when possible
                distance = math.sqrt(distance_squared + 1e-12)
                # Constraint: distance >= r1 + r2 (or equivalently: distance - r1 - r2 >= 0)
                # Add a small epsilon to handle numerical issues but don't overdo it
                constraints.append(distance - r1 - r2)
        
        return np.array(constraints)
    
    # Create constraint dictionary
    constraints = {'type': 'ineq', 'fun': constraint_function}
    
    # Run optimization with better parameters and robustness
    try:
        # Use different tolerances based on method
        options = {
            'maxiter': 500,  # Increased iterations for better convergence
            'ftol': 1e-10,   # Tighter tolerance for better accuracy
            'eps': 1e-6, 
            'disp': False
        }
        
        result = minimize(
            objective,
            initial_solution,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options=options
        )
        
        if result.success:
            optimized_circles = result.x.reshape(n, 3)
            # Ensure all radii are positive and within bounds
            optimized_circles[:, 2] = np.maximum(optimized_circles[:, 2], 0.001)
            optimized_circles[:, 2] = np.minimum(optimized_circles[:, 2], min(width, height) * 0.4)
            return optimized_circles
    except Exception as e:
        # Fallback to original if optimization fails, but also try a simplified approach
        try:
            # Simple local refinement as fallback
            refined = initial_circles.copy()
            for _ in range(10):  # Simple local iterations
                improved = False
                for i in range(n):
                    x, y, r = refined[i]
                    # Try to slightly increase radius if possible
                    max_radius = min(x, width - x, y, height - y)
                    # Check overlaps
                    overlap = False
                    for j in range(n):
                        if i != j:
                            dx = x - refined[j, 0]
                            dy = y - refined[j, 1]
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance < (r + refined[j, 2]):
                                overlap = True
                                break
                    if not overlap and max_radius > r:
                        # Try to increase radius slightly
                        new_r = min(max_radius, r * 1.05)  # Increase by 5%
                        if new_r > r:
                            refined[i, 2] = new_r
                            improved = True
                if not improved:
                    break
            return refined
        except Exception:
            pass
    
    return initial_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
