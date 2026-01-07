# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import random
from typing import Tuple, List
import time
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization, local optimization, 
    and physics-inspired refinement to beat the benchmark.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    max_time = 55  # Leave buffer for final processing
    start_time = time.time()
    
    # Better initialization using hexagonal lattice with randomization
    def initialize_hexagonal_config():
        # Create hexagonal pattern with some randomness for better exploration
        circles = []
        
        # Arrange in hexagonal pattern
        rows = 6
        cols = 6
        
        spacing_x = 0.8 / cols
        spacing_y = 0.8 / rows
        
        # Place circles with hexagonal offset
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n:
                    break
                    
                x_offset = 0.0 if row % 2 == 0 else 0.5
                x = 0.1 + (col + x_offset) * spacing_x
                y = 0.1 + row * spacing_y
                
                # Add some randomness to avoid regular patterns that may get stuck
                x += random.uniform(-spacing_x*0.1, spacing_x*0.1)
                y += random.uniform(-spacing_y*0.1, spacing_y*0.1)
                
                # Ensure within bounds
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                
                circles.append([x, y])
                count += 1
                
        return np.array(circles[:n])
    
    # Efficient constraint checking using spatial indexing
    def check_validity(circles: np.ndarray) -> bool:
        """Fast validity check for circles using spatial indexing"""
        # Check containment
        for i in range(n):
            x, y, r = circles[i]
            if r <= 0 or x < r or x > 1-r or y < r or y > 1-r:
                return False
        
        # Check overlaps efficiently using spatial indexing
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Build KDTree for fast neighbor search
        tree = cKDTree(positions)
        
        # Find neighbors within 2*(max_radius) distance
        max_radius = np.max(radii)
        neighbors = tree.query_pairs(2*max_radius, output_type='ndarray')
        
        # Check actual overlaps
        for i, j in neighbors:
            if i >= j: continue
            dist = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
            if dist < (radii[i] + radii[j]):
                return False
                
        return True
    
    # More sophisticated objective function with better gradient handling
    def objective(vars):
        # vars contains [x1, y1, r1, x2, y2, r2, ...]
        return -sum(vars[2::3])  # Negative because we want to maximize
    
    # Constraint function for scipy optimization
    def constraint_func(vars):
        constraints = []
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = vars[3*i], vars[3*i+1], vars[3*i+2]
                x2, y2, r2 = vars[3*j], vars[3*j+1], vars[3*j+2]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                overlap = np.sqrt(dist_sq) - (r1 + r2)
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
    
    # Physics-inspired refinement function
    def refine_with_physics(initial_solution):
        """Apply physics-inspired refinement to improve solution"""
        circles = initial_solution.copy()
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Simple physics-based approach: increase radii greedily while maintaining constraints
        for iteration in range(100):
            improved = False
            # Try to increase each radius
            for i in range(n):
                # Calculate maximum possible radius for this circle
                max_radius = float('inf')
                
                # Check boundary constraints
                bound_radius = min(positions[i, 0], 1 - positions[i, 0], 
                                positions[i, 1], 1 - positions[i, 1])
                max_radius = min(max_radius, bound_radius)
                
                # Check overlap constraints with other circles
                for j in range(n):
                    if i != j:
                        dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + 
                                     (positions[i, 1] - positions[j, 1])**2)
                        if dist > 0:  # Avoid division by zero
                            max_radius = min(max_radius, dist - radii[j])
                
                # Cap at reasonable maximum
                max_radius = min(max_radius, 0.499)
                
                if max_radius > radii[i]:
                    # Increase radius
                    new_radius = min(max_radius, radii[i] + 0.005)
                    radii[i] = new_radius
                    improved = True
            
            if not improved:
                break
        
        circles[:, 2] = radii
        return circles
    
    # Local optimization refinement function with better error handling
    def refine_solution(initial_solution):
        """Refine solution using local optimization with multiple strategies"""
        # Try physics-based refinement first
        try:
            refined = refine_with_physics(initial_solution)
            if check_validity(refined):
                return refined
        except:
            pass
        
        # Convert to flat vars for optimization
        vars = []
        for i in range(n):
            vars.extend([initial_solution[i, 0], initial_solution[i, 1], initial_solution[i, 2]])
        
        # Define bounds
        bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
        
        # Optimize with bounds and constraints
        try:
            result = minimize(
                objective,
                vars,
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                options={'maxiter': 300, 'ftol': 1e-6, 'eps': 1e-6}
            )
            
            if result.success:
                refined = np.zeros((n, 3))
                for i in range(n):
                    refined[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
                if check_validity(refined):
                    return refined
        except:
            pass
            
        return initial_solution
    
    # Multi-start approach with different initialization strategies
    best_sum = 0
    best_solution = None
    
    # Try different initialization strategies with more attempts
    init_strategies = [
        initialize_hexagonal_config,  # Hexagonal grid
        lambda: np.random.rand(n, 2) * 0.9 + 0.05,  # Random uniform
        lambda: np.random.rand(n, 2) * 0.8 + 0.1,   # Slightly more centered
    ]
    
    # Multi-start with more attempts per strategy for better exploration
    for strategy_idx, init_func in enumerate(init_strategies):
        if time.time() - start_time > max_time:
            break
            
        for attempt in range(8):  # Increased from 5 to 8 attempts
            if time.time() - start_time > max_time:
                break
                
            # Generate initial configuration
            try:
                initial_centers = init_func()
                circles = np.zeros((n, 3))
                
                # Initialize with reasonable radii based on spatial distribution
                for i in range(n):
                    x, y = initial_centers[i]
                    # Initial radius based on distance to nearest boundary
                    r = min(x, y, 1-x, 1-y) * 0.3
                    # Add some randomness
                    r *= random.uniform(0.7, 1.0)
                    circles[i] = [x, y, r]
                
                # Refine using optimization
                refined = refine_solution(circles)
                
                # Check validity and compute sum
                if check_validity(refined):
                    current_sum = np.sum(refined[:, 2])
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_solution = refined.copy()
                        
            except Exception as e:
                continue
    
    # If no good solution found, fallback to robust initialization
    if best_solution is None:
        # Use a robust approach: start with hexagonal pattern
        circles = initialize_hexagonal_config()
        best_solution = np.zeros((n, 3))
        
        # Initialize with reasonable radii
        for i in range(n):
            x, y = circles[i]
            # Initial radius based on distance to nearest boundary
            r = min(x, y, 1-x, 1-y) * 0.25
            # Add some randomness
            r *= random.uniform(0.8, 1.0)
            best_solution[i] = [x, y, r]
        
        # Try local optimization on this
        best_solution = refine_solution(best_solution)
    
    # Final validation and cleanup
    if best_solution is not None:
        # Ensure all circles are valid and within bounds
        for i in range(n):
            x, y, r = best_solution[i]
            # Make sure it's within bounds
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            best_solution[i] = [x, y, r]
        
        # Final validation
        if not check_validity(best_solution):
            # Simple iterative correction if needed
            for _ in range(50):
                for i in range(n):
                    x, y, r = best_solution[i]
                    # Adjust position to stay within bounds
                    x = max(r, min(1-r, x))
                    y = max(r, min(1-r, y))
                    best_solution[i] = [x, y, r]
                    
                if check_validity(best_solution):
                    break
                # Reduce radii slightly if still invalid
                best_solution[:, 2] *= 0.98
    
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
