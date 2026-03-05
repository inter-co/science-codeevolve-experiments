# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining multiple initialization strategies with advanced mathematical optimization.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Try multiple width/height combinations to find optimal configuration
    best_width = 1.25
    best_height = 0.75
    best_circles = None
    best_sum = 0
    
    # Focus on aspect ratios that have shown promise in inspirations
    # Based on results, ratios around 1.0-1.3 seem to work best
    ratios_to_try = [0.9, 1.0, 1.1, 1.2, 1.3]
    
    for ratio in ratios_to_try:
        width = 2.0 * ratio / (1.0 + ratio)
        height = 2.0 - width
        
        # Try different initialization strategies
        strategies = []
        
        # Strategy 1: Hexagonal pattern (like INSPIRATION 1 and 3)
        circles1 = initialize_hexagonal_pattern(width, height, 21)
        optimized_circles1 = optimize_with_mathematical_approach(circles1, width, height)
        strategies.append(("hexagonal", optimized_circles1))
        
        # Strategy 2: Grid pattern (like INSPIRATION 2)
        circles2 = initialize_grid_pattern(width, height, 21)
        optimized_circles2 = optimize_with_mathematical_approach(circles2, width, height)
        strategies.append(("grid", optimized_circles2))
        
        # Strategy 3: Systematic placement (like INSPIRATION 3)
        circles3 = initialize_systematic_placement(width, height, 21)
        optimized_circles3 = optimize_with_mathematical_approach(circles3, width, height)
        strategies.append(("systematic", optimized_circles3))
        
        # Select the best strategy for this ratio
        for name, circles in strategies:
            current_sum = np.sum(circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_width = width
                best_height = height
                best_circles = circles.copy()
    
    # Apply final comprehensive refinement
    if best_circles is not None:
        # Apply enhanced refinement with better optimization parameters
        final_circles = comprehensive_refinement(best_circles, best_width, best_height)
        final_sum = np.sum(final_circles[:, 2])
        
        if final_sum > best_sum:
            best_sum = final_sum
            best_circles = final_circles.copy()
    
    # Final validation
    if best_circles is not None:
        # Ensure all circles are properly constrained
        for i in range(len(best_circles)):
            x, y, r = best_circles[i]
            # Keep within bounds
            x = np.clip(x, r, best_width - r)
            y = np.clip(y, r, best_height - r)
            best_circles[i] = [x, y, r]
    
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


def initialize_systematic_placement(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circle positions using a systematic approach."""
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


def optimize_with_mathematical_approach(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Use mathematical optimization approach with enhanced robustness."""
    try:
        n = len(initial_circles)
        
        # Flatten the initial configuration
        initial_flat = initial_circles.flatten()
        
        # Define bounds for variables (positions and radii)
        bounds = []
        for i in range(n):
            # x coordinate - ensure it's within bounds with margin for radius
            bounds.append((0.001, width - 0.001))
            # y coordinate  
            bounds.append((0.001, height - 0.001))
            # radius (must be positive and reasonably sized)
            bounds.append((0.001, min(width, height) / 2 - 0.001))
        
        # Objective function: minimize negative sum of radii (i.e., maximize sum of radii)
        def objective(x_flat):
            circles = x_flat.reshape(-1, 3)
            return -np.sum(circles[:, 2])
        
        # Constraint function: ensure all constraints are satisfied
        def constraint_func(x_flat):
            circles = x_flat.reshape(-1, 3)
            positions = circles[:, :2]
            radii = circles[:, 2]
            
            # Boundary constraints
            boundary_constraints = []
            for i in range(n):
                x, y = positions[i]
                r = radii[i]
                # Circle must be within bounds
                boundary_constraints.extend([
                    x - r,           # x >= r
                    width - x - r,   # width - x >= r
                    y - r,           # y >= r
                    height - y - r   # height - y >= r
                ])
            
            # Overlap constraints (distance between centers >= sum of radii)
            overlap_constraints = []
            for i in range(n):
                for j in range(i+1, n):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    overlap_constraints.append(distance - (radii[i] + radii[j]))
            
            return np.array(boundary_constraints + overlap_constraints)
        
        # Try multiple optimization methods for robustness
        methods = ['SLSQP', 'trust-constr']
        best_result = None
        best_value = -np.inf
        
        for method in methods:
            try:
                result = minimize(
                    objective,
                    initial_flat,
                    method=method,
                    bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func},
                    options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
                )
                
                if result.success:
                    current_sum = -result.fun  # Because we minimized negative sum
                    if current_sum > best_value:
                        best_value = current_sum
                        best_result = result
            except Exception:
                continue
        
        # If we got a good result, return it, otherwise fall back to initial
        if best_result is not None:
            optimized_circles = best_result.x.reshape(-1, 3)
            return validate_and_refine(optimized_circles, width, height)
        else:
            return validate_and_refine(initial_circles, width, height)
            
    except Exception as e:
        return validate_and_refine(initial_circles, width, height)


def validate_and_refine(circles, width, height):
    """Refine the solution to ensure all constraints are met and improve quality."""
    # Make a copy to work with
    refined = circles.copy()
    
    # Ensure all circles fit within bounds
    for i in range(len(refined)):
        x, y, r = refined[i]
        # Clip positions to ensure circles are within bounds
        x = np.clip(x, r, width - r)
        y = np.clip(y, r, height - r)
        refined[i] = [x, y, r]
    
    # Perform a few rounds of improvement through local search
    for _ in range(200):  # More iterations for better refinement
        improved = False
        for i in range(len(refined)):
            # Try to increase radius while maintaining constraints
            x, y, r = refined[i]
            
            # Calculate maximum possible radius
            max_radius = min(x, width - x, y, height - y)
            
            # Check overlap constraints with other circles
            for j in range(len(refined)):
                if i != j:
                    dx = x - refined[j, 0]
                    dy = y - refined[j, 1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    max_radius = min(max_radius, distance - refined[j, 2])
            
            # Increase radius if beneficial
            if max_radius > r and max_radius > 0:
                refined[i, 2] = max_radius
                improved = True
        
        # If no improvements were made, stop
        if not improved:
            break
    
    # Final validation
    for i in range(len(refined)):
        x, y, r = refined[i]
        refined[i] = [np.clip(x, r, width - r), np.clip(y, r, height - r), r]
    
    return refined


def comprehensive_refinement(circles, width, height):
    """Apply comprehensive refinement with multiple optimization passes."""
    # First, try mathematical optimization with higher precision
    try:
        n = len(circles)
        
        # Flatten parameters
        flat_params = circles.flatten()
        
        # Define bounds
        bounds = []
        for i in range(n):
            bounds.append((circles[i, 2], width - circles[i, 2]))
            bounds.append((circles[i, 2], height - circles[i, 2]))
            bounds.append((0.001, min(width, height) / 2))
        
        # Objective function
        def objective(params):
            circles = params.reshape(n, 3)
            return -np.sum(circles[:, 2])
        
        # Constraint functions
        def constraint_overlaps(params):
            circles = params.reshape(n, 3)
            distances = cdist(circles[:, :2], circles[:, :2])
            constraints = []
            for i in range(n):
                for j in range(i + 1, n):
                    dist = distances[i, j]
                    min_dist = circles[i, 2] + circles[j, 2]
                    constraints.append(dist - min_dist)
            return np.array(constraints)
        
        def constraint_boundaries(params):
            circles = params.reshape(n, 3)
            constraints = []
            for i in range(n):
                x, y, r = circles[i]
                constraints.extend([x - r, width - x - r, y - r, height - y - r])
            return np.array(constraints)
        
        cons = [
            {'type': 'ineq', 'fun': constraint_overlaps},
            {'type': 'ineq', 'fun': constraint_boundaries}
        ]
        
        # Try with trust-constr method which is usually most robust
        result = minimize(
            objective,
            flat_params,
            method='trust-constr',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if result.success:
            refined_circles = result.x.reshape(n, 3)
            refined_circles[:, 2] = np.maximum(refined_circles[:, 2], 0.001)
            refined_circles[:, 0] = np.clip(refined_circles[:, 0], 
                                           refined_circles[:, 2], 
                                           width - refined_circles[:, 2])
            refined_circles[:, 1] = np.clip(refined_circles[:, 1], 
                                           refined_circles[:, 2], 
                                           height - refined_circles[:, 2])
            return refined_circles
    except Exception:
        pass
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
