# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')
import random
from itertools import combinations
import time

# Import evolutionary optimization library
try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False

# For better optimization, let's also import some additional libraries
try:
    import nevergrad as ng
    HAS_NEVERGRAD = True
except ImportError:
    HAS_NEVERGRAD = False

# More efficient constraint handling
def compute_distances(positions):
    """Compute pairwise distances between circle centers efficiently"""
    return cdist(positions, positions)

def check_feasibility_fast(positions, radii, width, height):
    """Fast feasibility check using vectorized operations"""
    # Check boundary constraints
    if np.any(positions[:, 0] - radii < 0) or \
       np.any(positions[:, 0] + radii > width) or \
       np.any(positions[:, 1] - radii < 0) or \
       np.any(positions[:, 1] + radii > height):
        return False
    
    # Check overlap constraints using vectorized operations
    if len(positions) > 1:
        dist_matrix = compute_distances(positions)
        # Create mask for upper triangle (avoid double counting)
        upper_triangle = np.triu(np.ones((len(positions), len(positions)), dtype=bool), k=1)
        # Check if any pair violates non-overlap constraint
        min_distances = radii[:, None] + radii[None, :]
        actual_distances = dist_matrix[upper_triangle]
        min_distances = min_distances[upper_triangle]
        
        if np.any(actual_distances < min_distances):
            return False
    
    return True

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses an improved evolutionary algorithm approach with better initialization and constraint handling.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    
    # Better initialization using more principled approaches
    def initialize_better_layout():
        best_sum = 0
        best_circles = None
        best_width = 1.0
        best_height = 1.0
        
        # Try several rectangle aspect ratios that might be optimal
        ratios_to_try = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5]
        
        for ratio in ratios_to_try:
            width = 1.0 * ratio
            height = 2.0 - width
            
            if width <= 0 or height <= 0:
                continue
                
            # Initialize with a simple grid-based approach first
            circles = []
            
            # Grid-based initialization with adaptive spacing
            grid_rows = max(1, int(np.ceil(np.sqrt(n))))
            grid_cols = max(1, int(np.ceil(n / grid_rows)))
            
            cell_width = width / (grid_cols + 1)
            cell_height = height / (grid_rows + 1)
            
            # Place circles in a grid pattern
            placed_count = 0
            for row in range(grid_rows):
                if placed_count >= n:
                    break
                for col in range(grid_cols):
                    if placed_count >= n:
                        break
                        
                    # Position in grid
                    x = (col + 1) * cell_width
                    y = (row + 1) * cell_height
                    
                    # Keep within bounds
                    x = np.clip(x, 0, width)
                    y = np.clip(y, 0, height)
                    
                    # Calculate max possible radius
                    max_radius_at_pos = min(x, width - x, y, height - y)
                    
                    # Use a more uniform distribution of radii
                    # Circles closer to center get larger radii
                    center_dist = np.sqrt((x - width/2)**2 + (y - height/2)**2)
                    center_factor = 1.0 - 0.7 * (center_dist / (np.sqrt((width/2)**2 + (height/2)**2)))
                    center_factor = max(0.1, min(1.0, center_factor))
                    
                    # Radius decreases with position index to avoid clustering at edges
                    radius_factor = 1.0 - 0.1 * (placed_count / n)
                    radius = max_radius_at_pos * 0.2 * center_factor * radius_factor
                    
                    radius = max(radius, 0.005)
                    
                    # Ensure valid placement
                    if x >= radius and x <= width - radius and \
                       y >= radius and y <= height - radius and \
                       radius > 0:
                        circles.append([x, y, radius])
                        placed_count += 1
            
            # If we don't have enough circles, fill with random placements
            if len(circles) < n:
                for i in range(len(circles), n):
                    x = random.uniform(0.05, width - 0.05)
                    y = random.uniform(0.05, height - 0.05)
                    max_radius = min(x, width - x, y, height - y)
                    radius = max_radius * 0.15  # Smaller initial radius
                    radius = max(radius, 0.005)
                    circles.append([x, y, radius])
            
            current_sum = sum(circle[2] for circle in circles[:n])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles[:n]
                best_width = width
                best_height = height
        
        # If we still haven't found a good configuration, use a better systematic approach
        if best_circles is None or best_sum < 1.0:
            circles = []
            width, height = 1.0, 1.0  # Default square
            
            # Try a concentric arrangement with different radii
            center_x, center_y = width/2, height/2
            max_radius = min(width, height) * 0.4
            
            # Arrange circles in concentric rings from center outward
            ring_count = 4
            rings = []
            
            # Distribute circles among rings
            circles_per_ring = [0] * ring_count
            remaining = n
            for i in range(ring_count):
                circles_per_ring[i] = max(1, remaining // (ring_count - i))
                remaining -= circles_per_ring[i]
            
            # Generate circles in rings
            total_placed = 0
            for ring_idx, num_circles in enumerate(circles_per_ring):
                if total_placed >= n:
                    break
                    
                # Ring radius and angular spacing
                ring_radius = (ring_idx + 1) * (max_radius / ring_count) * 0.8
                angle_step = 2 * np.pi / num_circles
                
                for i in range(min(num_circles, n - total_placed)):
                    if total_placed >= n:
                        break
                        
                    angle = i * angle_step
                    x = center_x + ring_radius * np.cos(angle)
                    y = center_y + ring_radius * np.sin(angle)
                    
                    # Keep within bounds
                    x = np.clip(x, 0, width)
                    y = np.clip(y, 0, height)
                    
                    # Calculate max radius at this position
                    max_radius_at_pos = min(x, width - x, y, height - y)
                    radius = max_radius_at_pos * 0.15
                    
                    # Ensure valid placement
                    if x >= radius and x <= width - radius and \
                       y >= radius and y <= height - radius and \
                       radius > 0:
                        circles.append([x, y, radius])
                        total_placed += 1
            
            # Fill any remaining spots with random placement
            while len(circles) < n:
                x = random.uniform(0.05, width - 0.05)
                y = random.uniform(0.05, height - 0.05)
                max_radius = min(x, width - x, y, height - y)
                radius = max_radius * 0.12
                radius = max(radius, 0.005)
                circles.append([x, y, radius])
            
            best_circles = circles[:n]
            best_width = width
            best_height = height
            
        return np.array(best_circles), best_width, best_height
    
    # Initialize with better configuration
    circles, rect_width, rect_height = initialize_better_layout()
    
    # Objective function to maximize sum of radii (minimize negative sum)
    def objective(params):
        # Reshape params into positions and radii
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Calculate negative sum of radii (we want to maximize sum, so minimize negative)
        return -np.sum(radii)
    
    # Vectorized constraint functions for better performance
    def non_overlap_constraint(params):
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Compute distance matrix efficiently
        dist_matrix = compute_distances(positions)
        
        # Get all constraint violations using vectorized operations
        # Create upper triangular matrix of all pairs
        upper_triangle = np.triu(np.ones((n, n), dtype=bool), k=1)
        actual_distances = dist_matrix[upper_triangle]
        required_distances = (radii[:, None] + radii[None, :])[upper_triangle]
        
        # Violations are negative values (distance < required distance)
        violations = actual_distances - required_distances
        
        return violations
    
    # Boundary constraints for circles to stay within rectangle
    def boundary_constraint(params):
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        # Vectorized boundary constraints
        left_bound = positions[:, 0] - radii
        right_bound = rect_width - positions[:, 0] - radii
        bottom_bound = positions[:, 1] - radii
        top_bound = rect_height - positions[:, 1] - radii
        
        # Return all boundary violations (should be positive for feasibility)
        return np.concatenate([left_bound, right_bound, bottom_bound, top_bound])
    
    # Combined constraints - all must be >= 0 for feasibility
    def combined_constraints(params):
        # Non-overlap constraints (positive means satisfied)
        overlap_violations = non_overlap_constraint(params)
        # Boundary constraints (positive means satisfied)  
        boundary_violations = boundary_constraint(params)
        # Combine constraints (positive means satisfied)
        return np.concatenate([overlap_violations, boundary_violations])
    
    # Enhanced constraint checking with vectorized operations
    def check_feasibility(params):
        positions = params[:-n].reshape(-1, 2)
        radii = params[-n:]
        
        return check_feasibility_fast(positions, radii, rect_width, rect_height)
    
    # Improved optimization using better metaheuristic approach
    # Initial parameter vector: [x1, y1, x2, y2, ..., xn, yn, r1, r2, ..., rn]
    initial_params = np.concatenate([
        circles[:, :2].flatten(),  # Positions
        circles[:, 2]              # Radii
    ])
    
    # Set bounds for positions and radii
    # Positions: [0, width] for x and y coordinates
    # Radii: [1e-6, min(width, height)/2] to prevent degenerate cases
    bounds = [(0, rect_width) for _ in range(2*n)] + [(1e-6, min(rect_width, rect_height)/2) for _ in range(n)]
    
    # Define constraints - all must be >= 0
    constraints = {
        'type': 'ineq',
        'fun': combined_constraints
    }
    
    # Use a more robust optimization approach with better strategies
    try:
        # Strategy 1: Try global optimization with better parameters
        best_result = None
        best_value = float('-inf')
        
        # Try differential evolution with high-quality settings
        try:
            # Use more iterations and better parameters for this problem
            de_result = differential_evolution(
                objective,
                bounds,
                constraints=constraints,
                seed=42,
                maxiter=2000,  # Reduced iterations to save time
                popsize=50,    # Reduced population size for speed
                mutation=(0.8, 1.0),  # Better mutation strategy
                recombination=0.9,    # Higher recombination rate
                atol=1e-10,
                rtol=1e-10,
                disp=False
            )
            
            if de_result.success:
                current_value = -de_result.fun
                if current_value > best_value:
                    best_value = current_value
                    best_result = de_result
        except:
            pass
        
        # Strategy 2: Use Nevergrad for potentially better optimization
        if HAS_NEVERGRAD and best_result is None:
            try:
                # Create optimizer with better settings for this problem
                instrum = ng.p.Array(shape=(2*n + n,))
                optimizer = ng.optimizers.NGOpt(instrumentation=instrum, budget=500)
                
                def nevergrad_objective(x):
                    # Convert to parameters
                    params = np.array(x)
                    return objective(params)
                
                # Run optimization
                recommendation = optimizer.minimize(nevergrad_objective)
                if recommendation is not None:
                    current_value = -nevergrad_objective(recommendation.value)
                    if current_value > best_value:
                        best_value = current_value
                        # Convert back to result format for consistency
                        final_params = np.array(recommendation.value)
                        best_result = type('obj', (object,), {'x': final_params, 'fun': -current_value, 'success': True})()
            except:
                pass
        
        # Strategy 3: Multi-start local optimization with better diversity
        if best_result is None:
            # Multi-start local optimization with better diversity
            for restart in range(20):  # Reduced restarts for speed
                # Perturb the initial solution with more substantial changes
                perturbed_params = initial_params.copy()
                # Add more significant noise for exploration
                noise_scale = 0.2  # Slightly reduced noise
                perturbed_params += np.random.normal(0, noise_scale, len(initial_params))
                
                # Clip to bounds
                for i, (bound, param) in enumerate(zip(bounds, perturbed_params)):
                    perturbed_params[i] = np.clip(param, bound[0], bound[1])
                
                try:
                    local_result = minimize(
                        objective,
                        perturbed_params,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10},  # Tighter tolerances
                        tol=1e-10
                    )
                    
                    if local_result.success:
                        current_value = -local_result.fun
                        if current_value > best_value:
                            best_value = current_value
                            best_result = local_result
                except:
                    continue
        
        # Extract results if we have a successful optimization
        if best_result is not None:
            final_positions = best_result.x[:-n].reshape(-1, 2)
            final_radii = best_result.x[-n:]
            
            # Update circles array with optimized values
            circles[:, 0] = final_positions[:, 0]
            circles[:, 1] = final_positions[:, 1]
            circles[:, 2] = final_radii
            
            # Ensure all radii are positive and reasonable
            circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
            # Make sure radii don't exceed reasonable limits
            max_radius_allowed = min(rect_width, rect_height) / 2
            circles[:, 2] = np.minimum(circles[:, 2], max_radius_allowed)
            
    except Exception as e:
        # If optimization fails, return initial configuration
        pass
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
