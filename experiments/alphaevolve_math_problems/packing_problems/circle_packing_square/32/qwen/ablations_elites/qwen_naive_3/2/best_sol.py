# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random

# Global constants
N_CIRCLES = 32
MAX_ITERATIONS = 1000
NUM_STARTS = 15

def initialize_better_hexagonal() -> np.ndarray:
    """Initialize circles using a better hexagonal packing approach"""
    circles = []
    
    # Create hexagonal pattern with offset rows for better packing
    rows = 6
    cols = 6
    
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= N_CIRCLES:
                break
            # Offset every other row for hexagonal packing
            offset = 0.5 if i % 2 == 0 else 0.0
            x = 0.1 + j * 0.15 + offset * 0.075
            y = 0.1 + i * 0.15
            
            # Add slight randomness to avoid perfect grid
            x += np.random.normal(0, 0.005)
            y += np.random.normal(0, 0.005)
            
            # Clamp to valid range
            x = max(0.05, min(0.95, x))
            y = max(0.05, min(0.95, y))
            
            # Initial radius - larger than before to start closer to optimal
            r = 0.05
            
            circles.append([x, y, r])
    
    # Fill remaining slots with random positions but ensuring good distribution
    while len(circles) < N_CIRCLES:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        # Slightly larger initial radius for remaining circles
        r = 0.045
        circles.append([x, y, r])
    
    return np.array(circles[:N_CIRCLES])

def enforce_boundaries(circles: np.ndarray) -> np.ndarray:
    """Ensure all circles are within the unit square with valid radii"""
    result = circles.copy()
    
    for i in range(len(result)):
        x, y, r = result[i]
        # Clamp positions to valid range
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        # Ensure positive radius
        r = max(0.001, r)
        result[i] = [x, y, r]
    
    return result

def check_validity(circles: np.ndarray) -> bool:
    """Check if circle configuration is valid"""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x < r or x > 1-r or y < r or y > 1-r:
            return False
    
    # Check overlap constraints with tolerance
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_dist_sq = (r1 + r2)**2
            if dist_sq < min_dist_sq * 0.999:  # Small tolerance
                return False
    
    return True

def local_search_refinement(circles: np.ndarray, max_iterations: int = 500) -> np.ndarray:
    """Apply local search refinement to improve solution quality"""
    best_circles = circles.copy()
    best_sum = np.sum(best_circles[:, 2])
    
    for iteration in range(max_iterations):
        # Make a small random change to one circle
        idx = random.randint(0, len(circles)-1)
        test_circles = circles.copy()
        
        # Slightly adjust position and radius
        test_circles[idx, 0] += random.uniform(-0.005, 0.005)
        test_circles[idx, 1] += random.uniform(-0.005, 0.005)
        test_circles[idx, 2] += random.uniform(-0.002, 0.002)
        
        # Enforce bounds
        test_circles[idx, 0] = max(0.001, min(0.999, test_circles[idx, 0]))
        test_circles[idx, 1] = max(0.001, min(0.999, test_circles[idx, 1]))
        test_circles[idx, 2] = max(0.001, min(0.49, test_circles[idx, 2]))
        
        # Check constraints and accept improvement
        if check_validity(test_circles):
            new_sum = np.sum(test_circles[:, 2])
            if new_sum > best_sum:
                best_sum = new_sum
                best_circles = test_circles.copy()
    
    return best_circles

def optimize_with_slsqp(initial_circles: np.ndarray) -> np.ndarray:
    """Optimize using scipy's SLSQP with proper constraint handling"""
    # Flatten circles array for optimization
    initial_flat = initial_circles.flatten()
    
    def objective(flat_circles):
        circles = flat_circles.reshape((-1, 3))
        return -np.sum(circles[:, 2])  # Negative because we maximize
    
    def constraint_func(flat_circles):
        circles = flat_circles.reshape((-1, 3))
        constraints = []
        
        # Boundary constraints (each circle must be fully contained)
        for i in range(N_CIRCLES):
            x, y, r = circles[i]
            # Circle must stay within unit square
            constraints.extend([
                x - r,          # x >= r
                y - r,          # y >= r
                1 - x - r,      # 1-x >= r
                1 - y - r       # 1-y >= r
            ])
        
        # Non-overlap constraints
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(dist - (r1 + r2) - 1e-8)  # Small tolerance
        
        return np.array(constraints)
    
    # Create bounds for variables: [x0, y0, r0, x1, y1, r1, ...]
    bounds = []
    for i in range(N_CIRCLES):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])  # x, y, r bounds
    
    # Use SLSQP method for constrained optimization
    try:
        result = minimize(
            objective,
            initial_flat,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={'maxiter': 1000, 'ftol': 1e-6, 'disp': False}
        )
        
        if result.success:
            optimized_circles = result.x.reshape((-1, 3))
            return enforce_boundaries(optimized_circles)
    except Exception as e:
        pass
    
    # If optimization fails, fall back to iterative improvement
    return iterative_improvement(initial_circles)

def iterative_improvement(circles: np.ndarray) -> np.ndarray:
    """Simple iterative improvement approach inspired by INSPIRATION 1"""
    # Use a physics-inspired approach with better constraints
    current_circles = circles.copy()
    
    for iteration in range(500):  # Limited iterations for time constraint
        # Extract current values
        current_x = current_circles[:, 0].copy()
        current_y = current_circles[:, 1].copy()
        current_r = current_circles[:, 2].copy()
        
        # Calculate pairwise distances
        pos_matrix = np.column_stack([current_x, current_y])
        distances = cdist(pos_matrix, pos_matrix)
        
        # Calculate forces and update positions/radii
        new_x = current_x.copy()
        new_y = current_y.copy()
        new_r = current_r.copy()
        
        # Apply constraints and improvements
        for i in range(N_CIRCLES):
            # Boundary constraints
            new_r[i] = max(0.001, min(0.499, new_r[i]))
            new_x[i] = max(new_r[i], min(1-new_r[i], new_x[i]))
            new_y[i] = max(new_r[i], min(1-new_r[i], new_y[i]))
            
            # Repulsion from overlapping circles
            for j in range(N_CIRCLES):
                if i != j:
                    dx = new_x[i] - new_x[j]
                    dy = new_y[i] - new_y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist < (new_r[i] + new_r[j]) and dist > 0.001:
                        # Move apart
                        force = (new_r[i] + new_r[j] - dist) / (dist + 0.001)
                        angle = np.arctan2(dy, dx)
                        new_x[i] += force * 0.01 * np.cos(angle)
                        new_y[i] += force * 0.01 * np.sin(angle)
                        
                        # Increase radius if possible
                        new_r[i] = min(0.499, new_r[i] + 0.0001)
        
        # Update positions
        current_circles[:, 0] = new_x
        current_circles[:, 1] = new_y
        current_circles[:, 2] = new_r
        
        # Ensure boundary constraints are maintained
        for i in range(N_CIRCLES):
            current_circles[i, 0] = max(current_circles[i, 2], min(1-current_circles[i, 2], current_circles[i, 0]))
            current_circles[i, 1] = max(current_circles[i, 2], min(1-current_circles[i, 2], current_circles[i, 1]))
            current_circles[i, 2] = max(0.001, min(0.499, current_circles[i, 2]))
    
    return current_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    best_sum = 0.0
    best_circles = None
    
    # Multi-start optimization to find better solutions
    for start_iter in range(NUM_STARTS):
        # Initialize with better hexagonal placement
        circles = initialize_better_hexagonal()
        
        # Apply optimization
        optimized_circles = optimize_with_slsqp(circles)
        
        # Apply local search refinement
        refined_circles = local_search_refinement(optimized_circles)
        
        # Calculate sum of radii
        radii_sum = np.sum(refined_circles[:, 2])
        
        # Keep track of best solution
        if radii_sum > best_sum:
            best_sum = radii_sum
            best_circles = refined_circles.copy()
    
    if best_circles is None:
        # Fallback to basic initialization if nothing worked
        best_circles = initialize_better_hexagonal()
    
    return best_circles


# EVOLVE-BLOCK-END
