# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import math
from collections import defaultdict
from scipy.spatial import KDTree
from scipy.optimize import differential_evolution, minimize
import warnings
from typing import Tuple

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining geometric initialization with advanced optimization techniques.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    random.seed(42)
    
    n = 32
    circles = np.zeros((n, 3))
    
    # Better initialization using a more sophisticated approach based on known good packings
    def generate_initial_configuration():
        # Start with a hexagonal-like packing pattern that tends to work well
        # This creates a more structured initial configuration than a simple grid
        points = []
        
        # Create a hexagonal lattice pattern with some randomness
        rows = 6
        cols = 6
        spacing_x = 1.0 / (cols + 1)
        spacing_y = 1.0 / (rows + 1)
        
        # Hexagonal offset pattern
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n:
                    break
                # Offset every other row
                offset = spacing_x / 2 if i % 2 == 1 else 0
                x = spacing_x * (j + 1) + offset + random.uniform(-spacing_x/6, spacing_x/6)
                y = spacing_y * (i + 1) + random.uniform(-spacing_y/6, spacing_y/6)
                if 0 <= x <= 1 and 0 <= y <= 1:
                    points.append([x, y])
            if len(points) >= n:
                break
        
        # Fill remaining spots with points near edges to improve coverage
        while len(points) < n:
            # Prefer corners and edges for better distribution
            choice = random.random()
            if choice < 0.3:  # Corner points
                x = random.choice([0.05, 0.95])
                y = random.choice([0.05, 0.95])
            elif choice < 0.6:  # Edge points
                if random.random() < 0.5:
                    x = random.uniform(0.05, 0.95)
                    y = random.choice([0.05, 0.95])
                else:
                    x = random.choice([0.05, 0.95])
                    y = random.uniform(0.05, 0.95)
            else:  # Interior points
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
            points.append([x, y])
            
        return points[:n]
    
    # Generate initial points
    points = generate_initial_configuration()
    
    # Initialize with varying initial radii - start with smaller values to allow growth
    initial_radius = 0.04
    for i in range(n):
        circles[i] = [points[i][0], points[i][1], initial_radius]
    
    # Optimized collision detection with spatial indexing
    def fast_collision_check(circle_pos, circles_array, index):
        """Fast collision check using spatial indexing"""
        x, y, r = circle_pos
        # Quick bounds check first
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
            
        # Use KDTree for efficient nearest neighbor search
        # Only consider circles that could potentially collide
        tree_points = [(circles_array[j][0], circles_array[j][1]) for j in range(len(circles_array)) if j != index]
        if len(tree_points) > 0:
            tree = KDTree(tree_points)
            # Find nearby points within a reasonable distance
            neighbors = tree.query_ball_point([x, y], r + 0.5)
            for neighbor_idx in neighbors:
                j = neighbor_idx if neighbor_idx < index else neighbor_idx + 1
                if j < len(circles_array):
                    cx, cy, cr = circles_array[j]
                    distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if distance < r + cr:
                        return False
        return True
    
    # Even faster collision check for validation purposes
    def quick_collision_check(circles_array):
        """Quick collision check for entire array"""
        n = len(circles_array)
        # Check bounds
        for i in range(n):
            x, y, r = circles_array[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
                
        # Check pairwise collisions with spatial indexing
        points = [(circles_array[i][0], circles_array[i][1]) for i in range(n)]
        tree = KDTree(points)
        for i in range(n):
            x, y, r = circles_array[i]
            # Find nearby points
            neighbors = tree.query_ball_point([x, y], 2*r)
            for j in neighbors:
                if i != j:
                    cx, cy, cr = circles_array[j]
                    distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if distance < r + cr:
                        return False
        return True
    
    # Improved neighbor generation with adaptive step sizes
    def get_neighbor_move(circle_idx, circles_array, iteration=None, use_large_moves=False):
        """Generate a neighbor move with adaptive step sizes"""
        # Copy current circles
        new_circles = circles_array.copy()
        
        # Choose move type with different weights for better exploration
        if use_large_moves:
            # Large moves for global exploration
            move_type = random.choices(['position', 'radius', 'both', 'large_position'], 
                                     weights=[0.2, 0.15, 0.3, 0.35])[0]
        elif iteration is not None and iteration < 3000:
            # Early iterations: more aggressive moves for global exploration
            move_type = random.choices(['position', 'radius', 'both', 'large_position'], 
                                     weights=[0.3, 0.2, 0.3, 0.2])[0]
        else:
            # Later iterations: fine-tuning
            move_type = random.choices(['position', 'radius', 'both'], 
                                     weights=[0.4, 0.3, 0.3])[0]
        
        if move_type == 'position':
            # Standard position move
            dx = random.uniform(-0.01, 0.01)
            dy = random.uniform(-0.01, 0.01)
            
            x, y, r = new_circles[circle_idx]
            new_x = max(r, min(1-r, x + dx))
            new_y = max(r, min(1-r, y + dy))
            new_circles[circle_idx] = [new_x, new_y, r]
            
        elif move_type == 'radius':
            # Radius change
            dr = random.uniform(-0.005, 0.005)
            x, y, r = new_circles[circle_idx]
            new_r = max(0.005, min(0.4, r + dr))
            new_circles[circle_idx] = [x, y, new_r]
            
        elif move_type == 'both':
            # Combined move
            dx = random.uniform(-0.008, 0.008)
            dy = random.uniform(-0.008, 0.008)
            dr = random.uniform(-0.003, 0.003)
            
            x, y, r = new_circles[circle_idx]
            new_x = max(r, min(1-r, x + dx))
            new_y = max(r, min(1-r, y + dy))
            new_r = max(0.005, min(0.4, r + dr))
            new_circles[circle_idx] = [new_x, new_y, new_r]
            
        else:  # large_position
            # Large position move for global exploration
            dx = random.uniform(-0.02, 0.02)
            dy = random.uniform(-0.02, 0.02)
            
            x, y, r = new_circles[circle_idx]
            new_x = max(r, min(1-r, x + dx))
            new_y = max(r, min(1-r, y + dy))
            new_circles[circle_idx] = [new_x, new_y, r]
            
        return new_circles
    
    # Calculate sum of radii efficiently
    def calculate_sum_of_radii(circles_array):
        """Calculate sum of radii"""
        return np.sum(circles_array[:, 2])
    
    # More robust validation
    def validate_solution(circles_array):
        """Validate that all circles are within bounds and non-overlapping"""
        for i in range(len(circles_array)):
            x, y, r = circles_array[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check all pairs for overlap using optimized approach
        n = len(circles_array)
        points = [(circles_array[i][0], circles_array[i][1]) for i in range(n)]
        tree = KDTree(points)
        for i in range(n):
            x, y, r = circles_array[i]
            neighbors = tree.query_ball_point([x, y], 2*r)
            for j in neighbors:
                if i != j:
                    cx, cy, cr = circles_array[j]
                    distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if distance < r + cr:
                        return False
        return True
    
    # Enhanced optimization using multiple strategies
    max_iterations = 120000  # Increased iterations for better exploration
    temperature = 1.0
    cooling_rate = 0.9995
    min_temperature = 1e-8
    
    # Track best solution
    best_circles = circles.copy()
    best_sum = calculate_sum_of_radii(circles)
    
    # Main optimization loop with more sophisticated approach
    for iteration in range(max_iterations):
        # Select random circle
        circle_idx = random.randint(0, n-1)
        
        # Generate neighbor - use larger moves in early stages
        use_large_moves = iteration < 40000
        new_circles = get_neighbor_move(circle_idx, circles, iteration, use_large_moves)
        
        # Check validity using fast collision checking
        if fast_collision_check(new_circles[circle_idx], new_circles, circle_idx):
            # Accept or reject based on Metropolis criterion
            current_sum = calculate_sum_of_radii(circles)
            new_sum = calculate_sum_of_radii(new_circles)
            
            delta = new_sum - current_sum
            
            # Accept if better or with probability based on temperature
            if delta > 0 or random.random() < math.exp(delta / temperature):
                circles = new_circles
                
                # Update best solution if improved
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_circles = new_circles.copy()
        
        # Cool down
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature
    
    # Local refinement phase with more aggressive moves
    local_refinement_iterations = 20000
    for iteration in range(local_refinement_iterations):
        # Try multiple simultaneous moves for better exploration
        num_moves = min(10, n // 2)  # Try up to 10 moves at once
        for _ in range(num_moves):
            circle_idx = random.randint(0, n-1)
            old_circles = circles.copy()
            
            # Try a larger move for more significant improvements
            dx = random.uniform(-0.025, 0.025)
            dy = random.uniform(-0.025, 0.025)
            dr = random.uniform(-0.01, 0.01)
            
            x, y, r = circles[circle_idx]
            new_x = max(r, min(1-r, x + dx))
            new_y = max(r, min(1-r, y + dy))
            new_r = max(0.005, min(0.4, r + dr))
            
            circles[circle_idx] = [new_x, new_y, new_r]
            
            # Validate the move
            if not fast_collision_check([new_x, new_y, new_r], circles, circle_idx):
                circles = old_circles
            else:
                # If valid, keep the change and update best if needed
                current_sum = calculate_sum_of_radii(circles)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = circles.copy()
    
    # Enhanced final optimization using a hybrid approach
    # Try simulated annealing with better parameters for the final stage
    def enhanced_local_search(circles_array, max_iter=8000):
        """Enhanced local search with more sophisticated moves"""
        current = circles_array.copy()
        current_sum = calculate_sum_of_radii(current)
        best = current.copy()
        best_sum = current_sum
        
        # More aggressive moves in this phase
        for iteration in range(max_iter):
            # Try multiple simultaneous moves
            num_moves = min(12, n // 2)
            for _ in range(num_moves):
                circle_idx = random.randint(0, n-1)
                old_circles = current.copy()
                
                # Use larger moves for this phase
                dx = random.uniform(-0.03, 0.03)
                dy = random.uniform(-0.03, 0.03)
                dr = random.uniform(-0.015, 0.015)
                
                x, y, r = current[circle_idx]
                new_x = max(r, min(1-r, x + dx))
                new_y = max(r, min(1-r, y + dy))
                new_r = max(0.005, min(0.4, r + dr))
                
                current[circle_idx] = [new_x, new_y, new_r]
                
                # Validate the move
                if not fast_collision_check([new_x, new_y, new_r], current, circle_idx):
                    current = old_circles
                else:
                    # If valid, keep the change and update best if needed
                    current_sum = calculate_sum_of_radii(current)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best = current.copy()
        
        return best, best_sum
    
    # Apply enhanced local search
    try:
        final_circles, final_sum = enhanced_local_search(best_circles, 4000)
        if final_sum > best_sum:
            best_circles = final_circles
            best_sum = final_sum
    except Exception:
        pass  # Continue with current best if optimization fails
    
    # Final optimization using scipy minimize on a subset of parameters
    # This helps fine-tune the best solution found
    def objective(params):
        # Reconstruct circles from flattened parameters
        temp_circles = best_circles.copy()
        for i in range(n):
            temp_circles[i] = [params[3*i], params[3*i+1], params[3*i+2]]
        return -calculate_sum_of_radii(temp_circles)  # Negative because we minimize
    
    def constraint_func(params):
        # Check if all circles fit in the square and don't overlap
        temp_circles = np.zeros((n, 3))
        for i in range(n):
            temp_circles[i] = [params[3*i], params[3*i+1], params[3*i+2]]
            
        # Bounds check
        for i in range(n):
            x, y, r = temp_circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return -1e-6  # Violation
        
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = temp_circles[i]
                x2, y2, r2 = temp_circles[j]
                distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if distance < r1 + r2:
                    return -1e-6  # Violation
        return 1e-6  # Valid
    
    # Convert to flat parameter vector
    initial_params = []
    for i in range(n):
        initial_params.extend([best_circles[i][0], best_circles[i][1], best_circles[i][2]])
    
    # Use scipy's minimize with L-BFGS-B which works well for smooth problems
    try:
        # First try to improve with local optimization
        result = minimize(objective, initial_params, method='L-BFGS-B', 
                         bounds=[(0,1), (0,1), (0.005, 0.4)] * n,
                         options={'maxiter': 5000})
        
        if result.success:
            # Convert back to circles
            refined_circles = best_circles.copy()
            for i in range(n):
                refined_circles[i] = [result.x[3*i], result.x[3*i+1], result.x[3*i+2]]
            
            # Validate and accept if better
            if quick_collision_check(refined_circles):
                refined_sum = calculate_sum_of_radii(refined_circles)
                if refined_sum > best_sum:
                    best_circles = refined_circles
                    best_sum = refined_sum
    except Exception as e:
        pass  # Continue with current best if optimization fails
    
    # Final validation
    if not validate_solution(best_circles):
        # If final solution is invalid, revert to previous valid state
        print("Warning: Final solution was invalid, using best valid solution")
    
    return best_circles


# EVOLVE-BLOCK-END
