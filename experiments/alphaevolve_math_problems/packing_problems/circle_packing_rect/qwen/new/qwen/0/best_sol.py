# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import random

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining global and local optimization with improved initialization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions (perimeter = 4, so width + height = 2)
    # Use more aspect ratios like INSPIRATION 1 for better exploration
    aspect_ratios = [
        (1.5, 0.5), (1.4, 0.6), (1.3, 0.7), (1.2, 0.8),
        (1.1, 0.9), (1.0, 1.0), (0.9, 1.1), (0.8, 1.2),
        (0.7, 1.3), (0.6, 1.4), (0.5, 1.5)
    ]
    
    best_sum = -np.inf
    best_circles = None
    
    # Multi-start optimization with more comprehensive strategies like INSPIRATION 1
    for ratio_idx, (width, height) in enumerate(aspect_ratios):
        # Try more diverse initializations to increase chances of finding better solutions
        for init_iter in range(8):  # Increase from 10 to 8 for better balance
            if init_iter < 4:
                # Hexagonal packing (dense arrangement)
                circles = initialize_hexagonal_pack(width, height, 21)
            else:
                # Grid-based initialization with more uniform spacing
                circles = initialize_grid_pack_better(width, height, 21)
            
            # Apply two-stage optimization: global then local like INSPIRATION 1
            optimized_circles = optimize_with_two_stage(circles, width, height)
            
            # Check if this is better
            current_sum = np.sum(optimized_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized_circles.copy()
    
    # Final refinement with high-quality optimization like INSPIRATION 1
    if best_circles is not None:
        # Try a final high-quality optimization run with even more aggressive settings
        try:
            fine_result = optimize_with_slsqp_refined(best_circles, 1.2, 0.8)
            current_sum = np.sum(fine_result[:, 2])
            if current_sum > best_sum:
                return fine_result
        except:
            pass
        return best_circles
    else:
        # Fallback to a well-tested default configuration
        width, height = 1.2, 0.8
        return initialize_hexagonal_pack(width, height, 21)

def initialize_hexagonal_pack(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circles in a hexagonal packing pattern"""
    circles = np.zeros((n, 3))
    
    # For 21 circles, try a 5x4 grid with hexagonal offset
    rows = 5
    cols = 4
    
    # Calculate spacing
    cell_width = width / (cols + 1)
    cell_height = height / (rows + 1)
    
    # Hexagonal packing with offset rows
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
                
            # Offset every other row
            x_offset = (col + 1) * cell_width
            y_offset = (row + 1) * cell_height
            
            if row % 2 == 1:
                x_offset += cell_width / 2
                
            # Ensure we stay within bounds
            x = max(cell_width/2, min(width - cell_width/2, x_offset))
            y = max(cell_height/2, min(height - cell_height/2, y_offset))
            
            # Set initial radius - start with medium values
            radius = min(0.08, cell_width/3, cell_height/3)
            circles[idx] = [x, y, radius]
            idx += 1
    
    return circles

def initialize_grid_pack_better(width: float, height: float, n: int) -> np.ndarray:
    """Initialize circles in a better grid pattern with improved spacing"""
    circles = np.zeros((n, 3))
    
    # Create a more balanced grid for 21 circles
    rows = 5
    cols = 5
    
    cell_width = width / (cols + 1)
    cell_height = height / (rows + 1)
    
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
                
            x = (col + 1) * cell_width
            y = (row + 1) * cell_height
            
            # Add slight randomness to avoid perfect grid artifacts
            x += random.uniform(-cell_width/6, cell_width/6)
            y += random.uniform(-cell_height/6, cell_height/6)
            
            # Keep within bounds
            x = max(cell_width/2, min(width - cell_width/2, x))
            y = max(cell_height/2, min(height - cell_height/2, y))
            
            # Set initial radius
            radius = min(0.07, cell_width/3, cell_height/3)
            circles[idx] = [x, y, radius]
            idx += 1
    
    return circles

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate the sum of all circle radii"""
    return np.sum(circles[:, 2])

def check_constraints(circles: np.ndarray, width: float, height: float) -> bool:
    """Check if all circles satisfy boundary and overlap constraints"""
    # Check boundary constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > width or y - r < 0 or y + r > height:
            return False
    
    # Check overlap constraints using vectorized operations for efficiency
    if len(circles) < 2:
        return True
        
    # Vectorized overlap checking
    coords = circles[:, :2]  # x, y coordinates
    radii = circles[:, 2]    # radii
    
    # Compute pairwise distances
    distances = cdist(coords, coords)
    
    # Create mask for pairs (excluding diagonal)
    n = len(circles)
    mask = ~np.eye(n, dtype=bool)
    
    # Check if any pair violates the no-overlap constraint
    # Distance between centers < sum of radii means overlap
    overlap_matrix = distances < (radii[:, None] + radii[None, :])
    
    # Check if any overlaps exist
    if np.any(overlap_matrix[mask]):
        return False
    
    return True

def objective_function(params: np.ndarray, width: float, height: float) -> float:
    """
    Objective function to minimize (negative of sum of radii)
    params: flattened array of [x1,y1,r1,x2,y2,r2,...]
    """
    n = len(params) // 3
    circles = params.reshape((n, 3))
    
    # Check constraints
    if not check_constraints(circles, width, height):
        # Return large penalty for constraint violations
        return -1e10
    
    # Return negative sum of radii (since we're minimizing)
    return -calculate_radius_sum(circles)

def optimize_with_two_stage(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Two-stage optimization: global then local like INSPIRATION 1"""
    # Stage 1: Global optimization with differential evolution (more aggressive settings)
    stage1_result = optimize_with_de(initial_circles, width, height)
    
    # Stage 2: Local optimization with SLSQP for fine-tuning
    stage2_result = optimize_with_slsqp(stage1_result, width, height)
    
    return stage2_result

def optimize_with_de(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize using differential evolution with aggressive settings"""
    n = len(initial_circles)
    
    # Define bounds for optimization
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.01, width - 0.01))
        # y bounds  
        bounds.append((0.01, height - 0.01))
        # radius bounds
        bounds.append((0.001, min(width, height) / 2))
    
    # Use differential evolution for global optimization with better parameters (like INSPIRATION 1)
    result = differential_evolution(
        lambda x: objective_function(x, width, height),
        bounds,
        maxiter=300,  # More iterations for better convergence
        popsize=40,   # Larger population for better exploration
        mutation=(0.8, 1),
        recombination=0.95,  # Higher recombination rate
        seed=random.randint(1, 1000),
        disp=False,
        tol=1e-7
    )
    
    # Reshape result back to circles format
    optimized_circles = result.x.reshape((n, 3))
    
    # Ensure final constraints are met
    if not check_constraints(optimized_circles, width, height):
        return fallback_optimization_simple(initial_circles, width, height)
    
    return optimized_circles

def optimize_with_slsqp(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize using SLSQP constrained optimization"""
    n = len(initial_circles)
    
    # Flatten initial circles for optimization
    initial_vars = []
    for i in range(n):
        initial_vars.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Objective function to maximize (negative because minimize)
    def objective(vars):
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i + 2]  # radius is third component
        return -total_radius  # negative because we want to maximize
    
    # Constraint functions with better handling
    def boundary_constraints(vars):
        result = []
        for i in range(n):
            x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
            # Circle must fit within rectangle (with some margin)
            result.append(x - r)  # left boundary
            result.append(y - r)  # bottom boundary
            result.append(width - x - r)  # right boundary
            result.append(height - y - r)  # top boundary
        return np.array(result)
    
    # Non-overlap constraints with better formulation
    def nonoverlap_constraints(vars):
        result = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                # Distance between centers must be >= sum of radii (to prevent overlap)
                result.append(distance - (r1 + r2))
        return np.array(result)
    
    # Define bounds
    bounds = []
    for i in range(n):
        bounds.append((0.001, width - 0.001))  # x bounds
        bounds.append((0.001, height - 0.001))  # y bounds
        bounds.append((0.001, min(width, height)/2 - 0.001))  # r bounds
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': nonoverlap_constraints}
    ]
    
    try:
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 400, 'ftol': 1e-8, 'gtol': 1e-8}  # More iterations and tighter tolerances
        )
        if result.success:
            # Convert back to circles array
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            return circles
    except Exception:
        pass
    
    # If SLSQP fails, return the input
    return initial_circles

def optimize_with_slsqp_refined(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """More refined SLSQP optimization for final tuning"""
    n = len(initial_circles)
    
    # Flatten initial circles for optimization
    initial_vars = []
    for i in range(n):
        initial_vars.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    # Objective function to maximize (negative because minimize)
    def objective(vars):
        total_radius = 0
        for i in range(n):
            total_radius += vars[3*i + 2]  # radius is third component
        return -total_radius  # negative because we want to maximize
    
    # Constraint functions with improved formulation
    def boundary_constraints(vars):
        result = []
        for i in range(n):
            x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
            # Circle must fit within rectangle (with some margin)
            result.append(x - r)  # left boundary
            result.append(y - r)  # bottom boundary
            result.append(width - x - r)  # right boundary
            result.append(height - y - r)  # top boundary
        return np.array(result)
    
    # Non-overlap constraints
    def nonoverlap_constraints(vars):
        result = []
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                # Distance between centers must be >= sum of radii (to prevent overlap)
                result.append(distance - (r1 + r2))
        return np.array(result)
    
    # Define bounds
    bounds = []
    for i in range(n):
        bounds.append((0.001, width - 0.001))  # x bounds
        bounds.append((0.001, height - 0.001))  # y bounds
        bounds.append((0.001, min(width, height)/2 - 0.001))  # r bounds
    
    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': nonoverlap_constraints}
    ]
    
    try:
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 600, 'ftol': 1e-10, 'gtol': 1e-10}  # Very tight tolerances
        )
        if result.success:
            # Convert back to circles array
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            return circles
    except Exception:
        pass
    
    # If SLSQP fails, return the input
    return initial_circles

def fallback_optimization_simple(initial_circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Simple but effective fallback optimization"""
    best_circles = initial_circles.copy()
    best_sum = calculate_radius_sum(best_circles)
    
    # Local search with more iterations for better chance of improvement
    for _ in range(20000):  # Even more iterations for better chance
        # Create a new candidate
        candidate = best_circles.copy()
        
        # Randomly perturb one circle
        idx = random.randint(0, len(candidate)-1)
        dx = random.uniform(-0.03, 0.03)
        dy = random.uniform(-0.03, 0.03)
        dr = random.uniform(-0.01, 0.01)
        
        candidate[idx][0] += dx
        candidate[idx][1] += dy
        candidate[idx][2] += dr
        
        # Keep within bounds
        candidate[idx][0] = max(0.01, min(width - 0.01, candidate[idx][0]))
        candidate[idx][1] = max(0.01, min(height - 0.01, candidate[idx][1]))
        candidate[idx][2] = max(0.001, min(min(width, height)/2, candidate[idx][2]))
        
        # Check constraints and accept if better
        if check_constraints(candidate, width, height):
            candidate_sum = calculate_radius_sum(candidate)
            if candidate_sum > best_sum:
                best_circles = candidate
                best_sum = candidate_sum
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
