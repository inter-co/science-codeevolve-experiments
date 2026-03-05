# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
import math
from itertools import combinations
import time
from scipy.spatial.distance import cdist
import random
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses advanced optimization techniques to achieve better results than the benchmark.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    
    # Rectangle dimensions: perimeter = 4, so width + height = 2
    best_sum_radii = 0
    best_circles = None
    best_width = 1.0
    best_height = 1.0
    
    # Try different aspect ratios - optimized for circle packing
    ratios = [0.5, 0.7, 0.8, 1.0, 1.2, 1.4, 1.618, 1.8, 2.0, 2.5]  # More refined ratios
    
    # Improved initialization with better hexagonal packing
    def create_improved_hexagonal_initialization(width, height, n=21):
        """Create initial configuration using improved hexagonal packing"""
        # For 21 circles, use 5 rows with pattern: 4, 5, 4, 4, 4
        rows = 5
        cols_per_row = [4, 5, 4, 4, 4]
        
        # Calculate cell dimensions based on available space
        max_cols = max(cols_per_row)
        cell_width = width / max_cols
        cell_height = height / rows
        
        # Make sure cells are small enough for good packing
        cell_width = min(cell_width, cell_height * 0.8)
        cell_height = min(cell_width * 0.8, cell_height)
        
        circles = []
        idx = 0
        
        for row in range(rows):
            cols = cols_per_row[row]
            row_y = (row + 0.5) * cell_height
            
            # Offset every other row for hexagonal packing
            x_offset = (row % 2) * cell_width * 0.5
            
            for col in range(cols):
                if idx >= n:
                    break
                row_x = (col + 0.5) * cell_width + x_offset
                
                # Ensure we're within bounds with some margin
                row_x = max(cell_width/2, min(width - cell_width/2, row_x))
                row_y = max(cell_height/2, min(height - cell_height/2, row_y))
                
                # Initial radius - based on cell size but leave room for optimization
                radius = min(cell_width, cell_height) * 0.35
                
                circles.append([row_x, row_y, radius])
                idx += 1
                
            if idx >= n:
                break
        
        # Fill remaining slots if needed with better random positions
        while len(circles) < n:
            # Use more strategic placement near edges for better optimization
            edge_prob = np.random.random()
            if edge_prob < 0.3:  # Place near edges
                x = np.random.uniform(0.05, width - 0.05)
                y = np.random.choice([0.05, height - 0.05]) if np.random.random() < 0.5 else np.random.uniform(0.05, height - 0.05)
            else:  # Place in center
                x = np.random.uniform(0.1, width - 0.1)
                y = np.random.uniform(0.1, height - 0.1)
            
            radius = np.random.uniform(0.02, min(width, height) * 0.15)
            circles.append([x, y, radius])
            
        return np.array(circles)
    
    # Optimized evaluation function with more efficient overlap checking
    def evaluate_individual(individual):
        # Decode individual into circles: [x1, y1, r1, x2, y2, r2, ...]
        circles = np.array(individual).reshape(-1, 3)
        n = len(circles)
        
        # Extract parameters
        width = 1.0
        height = 1.0
        
        # Check if any circles are outside boundaries
        for i in range(n):
            x, y, r = circles[i]
            if x < r or x > width - r or y < r or y > height - r:
                return -1e12  # Severe penalty for invalid solutions
        
        # Compute total radii (negative because we want to maximize)
        total_radii = np.sum(circles[:, 2])
        
        # Vectorized overlap penalty calculation - more efficient
        penalty = 0
        if n > 1:
            # Create coordinate arrays
            coords = circles[:, :2]
            radii = circles[:, 2]
            
            # Vectorized distance calculation using scipy for better performance
            dist_matrix = distance_matrix(coords, coords)
            
            # Create overlap matrix
            sum_radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
            overlaps = np.maximum(0, sum_radii_matrix - dist_matrix)
            
            # Apply penalty for overlaps (only count unique pairs once)
            penalty = np.sum(overlaps * (1 - np.eye(n))) * 1000
            
        return total_radii - penalty
    
    # Enhanced local optimization approach with better constraint handling
    def run_local_optimization(width, height, initial_circles):
        """Use more robust local optimization with better constraint handling"""
        n = len(initial_circles)
        
        def objective(params):
            circles_flat = params.reshape(-1, 3)
            radii = circles_flat[:, 2]
            return -np.sum(radii)  # Negative because we want to maximize
        
        def constraint_func(params):
            circles_flat = params.reshape(-1, 3)
            constraints = []
            
            # Pairwise distance constraints (no overlaps) - more efficient version
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles_flat[i]
                    x2, y2, r2 = circles_flat[j]
                    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    constraints.append(dist - (r1 + r2))  # Should be >= 0
            
            # Boundary constraints
            for i in range(n):
                x, y, r = circles_flat[i]
                constraints.extend([
                    x - r,              # left boundary
                    width - x - r,      # right boundary
                    y - r,              # bottom boundary
                    height - y - r      # top boundary
                ])
            
            return np.array(constraints)
        
        # Set bounds for optimization
        bounds = []
        for i in range(n):
            bounds.extend([(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)])
        
        # Try multiple optimization strategies for better results
        try:
            # First try differential evolution with better parameters
            result = differential_evolution(
                objective,
                bounds,
                constraints={'type': 'ineq', 'fun': constraint_func},
                maxiter=150,
                popsize=20,
                strategy='best1bin',
                seed=42,
                polish=True,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                return current_sum, optimized_circles
        except Exception:
            pass
            
        # Fallback to simpler optimization approach
        try:
            # Simple gradient-based approach for refinement with better tolerances
            initial_params = initial_circles.flatten()
            cons = {'type': 'ineq', 'fun': constraint_func}
            bounds = [(1e-6, width - 1e-6), (1e-6, height - 1e-6), (1e-6, min(width, height)/2)] * n
            
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10},
                tol=1e-10
            )
            
            if result.success:
                optimized_circles = result.x.reshape(-1, 3)
                current_sum = np.sum(optimized_circles[:, 2])
                return current_sum, optimized_circles
        except Exception:
            pass
        
        return np.sum(initial_circles[:, 2]), initial_circles
    
    # Try different aspect ratios and optimization approaches
    for ratio in ratios:
        width = 2.0 / (1 + ratio)  # width + height = 2, and width/height = ratio
        height = 2.0 / (1 + 1/ratio)
        
        # Start with better initialization
        initial_circles = create_improved_hexagonal_initialization(width, height, 21)
        
        # Local optimization
        local_sum, local_circles = run_local_optimization(width, height, initial_circles)
        
        if local_sum > best_sum_radii:
            best_sum_radii = local_sum
            best_circles = local_circles.copy()
            best_width = width
            best_height = height
    
    # Additional refinement: try multiple starting configurations
    if best_circles is None or best_sum_radii < 2.2:
        # Try different initialization strategies
        width, height = 1.0, 1.0
        strategies = [
            create_improved_hexagonal_initialization,
            lambda w, h, n: np.random.rand(n, 3) * np.array([w, h, 0.5]),
        ]
        
        for strategy in strategies:
            try:
                initial_circles = strategy(width, height, 21)
                local_sum, local_circles = run_local_optimization(width, height, initial_circles)
                if local_sum > best_sum_radii:
                    best_sum_radii = local_sum
                    best_circles = local_circles.copy()
                    best_width = width
                    best_height = height
            except Exception:
                continue
    
    # Final refinement with focused search around best found solution
    if best_circles is not None and best_sum_radii > 2.0:
        # Try small perturbations around the best solution
        for _ in range(5):
            perturbed = best_circles.copy()
            # Add small random perturbations
            for i in range(len(perturbed)):
                perturbed[i][0] += np.random.normal(0, 0.02)
                perturbed[i][1] += np.random.normal(0, 0.02)
                perturbed[i][2] += np.random.normal(0, 0.005)
                # Keep radii positive and reasonable
                perturbed[i][2] = max(0.01, min(0.5, perturbed[i][2]))
            
            # Reoptimize this perturbed version
            local_sum, local_circles = run_local_optimization(best_width, best_height, perturbed)
            if local_sum > best_sum_radii:
                best_sum_radii = local_sum
                best_circles = local_circles.copy()
    
    # Final fallback: use best heuristic
    if best_circles is None:
        # Use the best hexagonal packing approach
        width, height = 1.0, 1.0
        circles = create_improved_hexagonal_initialization(width, height, 21)
        return circles
    
    return best_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
