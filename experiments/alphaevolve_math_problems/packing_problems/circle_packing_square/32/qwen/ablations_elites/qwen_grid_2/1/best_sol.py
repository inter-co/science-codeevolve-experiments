# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
import math
import random
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Global constants for optimization
INITIAL_RADIUS = 0.05
MAX_ITERATIONS = 50

def validate_circles(circles: np.ndarray) -> bool:
    """Validate that all circles are within bounds and non-overlapping."""
    n = len(circles)
    
    # Check containment constraints
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlap constraints using efficient spatial indexing
    if n > 1:
        coords = circles[:, :2]
        radii = circles[:, 2]
        
        # Use KDTree for efficient neighbor search
        tree = cKDTree(coords)
        # Find all pairs within distance of (r1 + r2)
        pairs = tree.query_pairs(radii.max() * 2)
        
        # Check actual overlaps
        for i, j in pairs:
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            if distance < (r1 + r2):
                return False
    
    return True

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate sum of all radii."""
    return np.sum(circles[:, 2])

def initialize_circles_hexagonal(n: int) -> np.ndarray:
    """Initialize circles using a hexagonal packing pattern for better initial placement."""
    circles = np.zeros((n, 3))
    
    # Create a hexagonal grid pattern
    rows = 6
    cols = 6
    spacing_x = 0.15
    spacing_y = 0.15 * np.sqrt(3)/2
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            x = 0.1 + j * spacing_x
            y = 0.1 + i * spacing_y
            # Adjust for odd rows
            if i % 2 == 1:
                x += spacing_x / 2
            
            # Ensure we're within bounds
            if x <= 0.9 and y <= 0.9:
                circles[count] = [x, y, 0.05]
                count += 1
        
        if count >= n:
            break
    
    # Fill remaining slots with random positions
    while count < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        circles[count] = [x, y, 0.05]
        count += 1
        
    return circles

def initialize_circles_grid(n: int) -> np.ndarray:
    """Initialize circles in a grid pattern."""
    circles = np.zeros((n, 3))
    sqrt_n = int(math.ceil(math.sqrt(n)))
    
    # Place in a grid pattern
    count = 0
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if count >= n:
                break
            x = (j + 0.5) / sqrt_n
            y = (i + 0.5) / sqrt_n
            r = INITIAL_RADIUS
            # Adjust for boundaries
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            circles[count] = [x, y, r]
            count += 1
        if count >= n:
            break
    return circles

def initialize_circles_random(n: int) -> np.ndarray:
    """Initialize circles with random positions and small radii."""
    circles = np.zeros((n, 3))
    for i in range(n):
        # Random position within unit square
        x = np.random.uniform(INITIAL_RADIUS, 1 - INITIAL_RADIUS)
        y = np.random.uniform(INITIAL_RADIUS, 1 - INITIAL_RADIUS)
        # Small initial radius
        r = INITIAL_RADIUS
        circles[i] = [x, y, r]
    return circles

def compute_max_radius(circles: np.ndarray, i: int) -> float:
    """Compute maximum possible radius for circle i without violating constraints."""
    x, y, r = circles[i]
    
    # Maximum radius based on boundaries
    max_r = min(x, 1-x, y, 1-y)
    
    # Check constraints with all other circles
    for j in range(len(circles)):
        if i != j:
            x2, y2, r2 = circles[j]
            distance = math.sqrt((x-x2)**2 + (y-y2)**2)
            if distance > 0:
                max_r = min(max_r, distance - r2)
    
    return max_r

def local_radius_increase(circles: np.ndarray) -> np.ndarray:
    """Try to locally increase radii while maintaining constraints."""
    new_circles = circles.copy()
    n = len(new_circles)
    improved = False
    
    # Try to increase each radius aggressively
    for i in range(n):
        x, y, r = new_circles[i]
        max_r = compute_max_radius(new_circles, i)
        
        # Increase radius more aggressively - use larger multiplier
        if max_r > r * 1.05:  # More aggressive threshold
            # Aggressive increase to push towards optimal
            new_r = min(max_r, r * 1.3)  # Larger multiplier
            new_circles[i, 2] = new_r
            improved = True
    
    return new_circles, improved

def local_position_improvement(circles: np.ndarray) -> np.ndarray:
    """Try to improve positions slightly with more extensive search."""
    new_circles = circles.copy()
    n = len(new_circles)
    improved = False
    
    # Try more extensive position adjustments for each circle
    for i in range(n):
        x, y, r = new_circles[i]
        old_x, old_y = x, y
        
        # Try several nearby positions with wider search range
        best_x, best_y = x, y
        best_radius = r
        best_score = r
        
        # Try larger perturbations for better exploration
        search_range = [-0.01, -0.005, 0, 0.005, 0.01]
        for dx in search_range:
            for dy in search_range:
                new_x = x + dx
                new_y = y + dy
                
                # Keep within bounds
                if (new_x - r >= 0 and new_x + r <= 1 and
                    new_y - r >= 0 and new_y + r <= 1):
                    
                    # Check constraints with neighbors
                    valid = True
                    for j in range(n):
                        if i != j:
                            x2, y2, r2 = new_circles[j]
                            distance = math.sqrt((new_x-x2)**2 + (new_y-y2)**2)
                            if distance < (r + r2):
                                valid = False
                                break
                    
                    if valid:
                        # Try to increase radius at this new position
                        max_r = min(new_x, 1-new_x, new_y, 1-new_y)
                        for j in range(n):
                            if i != j:
                                x2, y2, r2 = new_circles[j]
                                distance = math.sqrt((new_x-x2)**2 + (new_y-y2)**2)
                                if distance > 0:
                                    max_r = min(max_r, distance - r2)
                        
                        if max_r > best_radius:
                            best_x, best_y = new_x, new_y
                            best_radius = max_r
                            best_score = max_r
            
        # Apply the best change found
        if (abs(best_x - old_x) > 1e-6 or abs(best_y - old_y) > 1e-6 or 
            abs(best_radius - r) > 1e-6):
            new_circles[i, 0] = best_x
            new_circles[i, 1] = best_y
            new_circles[i, 2] = best_radius
            improved = True
    
    return new_circles, improved

def optimize_with_differential_evolution(circles: np.ndarray) -> np.ndarray:
    """Use differential evolution optimization on the entire configuration."""
    n = len(circles)
    
    # Objective function for optimization
    def objective(params):
        # Reshape params into circles array
        reconstructed = params.reshape(-1, 3)
        
        # Check constraints
        if not validate_circles(reconstructed):
            # Return very large negative value for infeasible solutions
            return 1e10
        
        # Return negative sum of radii (since we minimize)
        return -np.sum(reconstructed[:, 2])
    
    # Flatten initial circles for optimization
    initial_params = circles.flatten()
    
    # Set up bounds for differential evolution
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])  # x, y, r bounds
    
    try:
        # Run differential evolution optimization with more aggressive settings
        result = differential_evolution(
            objective,
            bounds,
            maxiter=100,  # More iterations
            popsize=25,   # Larger population
            mutation=(0.8, 1),  # More aggressive mutation
            recombination=0.9,  # Higher recombination rate
            seed=42,
            disp=False,
            atol=1e-6
        )
        
        # Extract optimized solution
        optimized_circles = result.x.reshape(-1, 3)
        
        # Verify final constraints
        if validate_circles(optimized_circles):
            return optimized_circles
        else:
            return circles
            
    except Exception:
        # If optimization fails, return original circles
        return circles

def multi_start_optimization(initial_circles: np.ndarray) -> np.ndarray:
    """Multi-start optimization with different initialization strategies."""
    best_circles = initial_circles.copy()
    best_sum = calculate_radius_sum(best_circles)
    
    # Try several different approaches with more diversity
    strategies = [
        lambda: initialize_circles_hexagonal(32),
        lambda: initialize_circles_grid(32),
        lambda: initial_circles.copy(),  # Current solution as baseline
    ]
    
    # Add more randomness to exploration
    for _ in range(5):  # More random starts
        # Random initialization with better distribution
        rand_circles = initialize_circles_random(32)
        strategies.append(lambda c=rand_circles: c.copy())
    
    # Add a perturbed version of hexagonal
    def perturbed_hexagonal():
        circles = initialize_circles_hexagonal(32)
        # Add small random perturbations to get better diversity
        for i in range(len(circles)):
            circles[i, 0] += np.random.normal(0, 0.01)
            circles[i, 1] += np.random.normal(0, 0.01)
            # Keep within bounds
            circles[i, 0] = np.clip(circles[i, 0], 0.01, 0.99)
            circles[i, 1] = np.clip(circles[i, 1], 0.01, 0.99)
        return circles
    
    strategies.append(perturbed_hexagonal)
    
    for i, strategy in enumerate(strategies):
        try:
            current_circles = strategy()
            
            # Apply local optimizations with more iterations
            for iteration in range(50):  # More iterations
                # Local radius increase
                current_circles, improved1 = local_radius_increase(current_circles)
                
                # Local position improvement
                current_circles, improved2 = local_position_improvement(current_circles)
                
                if not improved1 and not improved2:
                    break
                
                # Periodically run DE optimization
                if iteration % 15 == 0:
                    current_circles = optimize_with_differential_evolution(current_circles)
                
                current_sum = calculate_radius_sum(current_circles)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_circles = current_circles.copy()
                    
                # Early stopping if no improvement
                if current_sum <= best_sum * 1.001:
                    break
                    
        except Exception:
            continue
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
        of the i-th circle of radius r.
    """
    # Start with hexagonal initialization for better coverage
    circles = initialize_circles_hexagonal(32)
    
    # Apply multi-start optimization with more thorough search
    circles = multi_start_optimization(circles)
    
    # Additional refinement steps with more aggressive local search
    for _ in range(30):  # More iterations for better convergence
        # Local improvements
        circles, improved1 = local_radius_increase(circles)
        circles, improved2 = local_position_improvement(circles)
        
        if not improved1 and not improved2:
            break
            
        # Occasionally run global optimization with more aggressive parameters
        if _ % 10 == 0:
            circles = optimize_with_differential_evolution(circles)
    
    # Final validation and cleanup
    if not validate_circles(circles):
        # Fallback to a simple valid configuration
        circles = initialize_circles_random(32)
        circles = multi_start_optimization(circles)
    
    # Ensure final bounds compliance
    for i in range(len(circles)):
        x, y, r = circles[i]
        # Clamp to valid bounds
        circles[i, 0] = np.clip(x, r, 1-r)
        circles[i, 1] = np.clip(y, r, 1-r)
    
    return circles


# EVOLVE-BLOCK-END
