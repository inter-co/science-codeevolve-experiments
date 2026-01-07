# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import Voronoi
import time
from itertools import combinations
import random
import math

# Set seed for reproducibility
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining golden ratio initialization with aggressive local 
    optimization and multi-start strategies to beat the benchmark.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Multi-start approach: try several different initializations
    best_circles = None
    best_sum = -np.inf
    
    # Strategy 1: Golden ratio approach (inspiration 1)
    circles1 = generate_golden_ratio_config()
    
    # Strategy 2: Voronoi-based initialization (inspiration 2)
    circles2 = generate_enhanced_voronoi_initialization()
    
    # Strategy 3: Hexagonal pattern (inspiration 1 approach)
    circles3 = generate_hexagonal_initialization()
    
    # Strategy 4: Grid-based with random perturbations
    circles4 = generate_grid_initialization()
    
    # Strategy 5: Randomized with overlap-aware initialization
    circles5 = generate_randomized_initialization()
    
    initial_strategies = [circles1, circles2, circles3, circles4, circles5]
    
    # Use a more aggressive multi-start strategy
    for i, initial_circles in enumerate(initial_strategies):
        try:
            # Apply optimization to each initial configuration
            optimized_circles = optimize_packaging(initial_circles)
            
            # Validate and refine
            validated_circles = validate_and_refine(optimized_circles)
            
            # Evaluate
            current_sum = np.sum(validated_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = validated_circles.copy()
                
        except Exception as e:
            continue
    
    # If we still don't have a good solution, do final refinement with golden ratio approach
    if best_circles is None:
        # Fallback to golden ratio approach with aggressive local optimization
        best_circles = generate_golden_ratio_config()
        best_circles = apply_aggressive_local_optimization(best_circles)
    
    # Additional fine-tuning with multiple restarts
    best_circles = apply_aggressive_local_optimization(best_circles)
    
    return best_circles

def generate_golden_ratio_config() -> np.ndarray:
    """
    Initialize circles using golden ratio distribution for better coverage.
    This provides a good starting configuration that's less likely to have overlaps.
    """
    # Use golden ratio distribution for better coverage
    phi = (1 + math.sqrt(5)) / 2
    circles = []
    for i in range(32):
        # Distribute points using golden ratio for better uniformity
        x = ((i * phi) % 1) * 0.8 + 0.1  # Scale to [0.1, 0.9]
        y = (i / 32) * 0.8 + 0.1
        # Add some randomness to avoid grid artifacts
        x += random.uniform(-0.02, 0.02)
        y += random.uniform(-0.02, 0.02)
        # Keep within bounds
        x = max(0.05, min(0.95, x))
        y = max(0.05, min(0.95, y))
        # Initial small radius
        r = 0.03
        circles.append([x, y, r])
    return np.array(circles)

def generate_hexagonal_initialization():
    """Generate initial configuration inspired by hexagonal packing"""
    circles = np.zeros((32, 3))
    
    # Create a hexagonal pattern with 5 rows and 7 columns
    rows = 5
    cols = 7
    
    # Calculate spacing based on desired packing efficiency
    spacing_x = 0.8 / cols  # Leave 0.1 margin on each side
    spacing_y = 0.8 / rows
    
    # Create pattern with alternating offset for better hexagonal packing
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= 32:
                break
            # Offset every other row for hexagonal packing
            offset = 0.5 * spacing_x if i % 2 == 1 else 0
            x = 0.1 + offset + (j + 0.5) * spacing_x
            y = 0.1 + (i + 0.5) * spacing_y
            
            # Initial radius - based on spacing but not too large
            r = min(spacing_x, spacing_y) * 0.35
            
            # Ensure circle fits in square
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[count] = [x, y, r]
            count += 1
            
        if count >= 32:
            break
    
    # For remaining circles, place them strategically in corners and edges
    corner_positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    
    for i in range(count, 32):
        # Try placing in corners first, then distribute more evenly
        if i < count + len(corner_positions):
            corner_idx = (i - count) % len(corner_positions)
            x_base, y_base = corner_positions[corner_idx]
            # Add small random variation to avoid perfect symmetries
            x = x_base + np.random.uniform(-0.03, 0.03)
            y = y_base + np.random.uniform(-0.03, 0.03)
        else:
            # Distribute remaining circles more evenly
            x = 0.1 + np.random.uniform(0, 0.8)
            y = 0.1 + np.random.uniform(0, 0.8)
        
        # Set reasonable initial radius
        r = np.random.uniform(0.02, 0.08)
        
        # Ensure it fits within bounds
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        
        circles[i] = [x, y, r]
    
    return circles

def generate_enhanced_voronoi_initialization():
    """Generate initial configuration using enhanced Voronoi diagram approach"""
    # Generate random points for Voronoi
    np.random.seed(42)
    points = np.random.rand(32, 2) * 0.8 + 0.1  # Keep away from edges
    
    # Create Voronoi diagram
    vor = Voronoi(points)
    
    circles = np.zeros((32, 3))
    
    # For each Voronoi cell, place a circle at the centroid with appropriate radius
    for i in range(32):
        if i < len(vor.points):
            x, y = vor.points[i]
            
            # Compute radius based on Voronoi cell characteristics
            # Use more sophisticated approach: minimum distance to neighbors plus cell area considerations
            min_dist = float('inf')
            for j in range(32):
                if i != j:
                    dist = np.sqrt((x - vor.points[j, 0])**2 + (y - vor.points[j, 1])**2)
                    min_dist = min(min_dist, dist)
            
            # Combined radius calculation - more conservative to allow for optimization
            r = min(0.15, min_dist * 0.25)  # Base on neighbor distance
            r = max(0.01, r)  # Minimum radius
            
            # Ensure it fits in the square
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[i] = [x, y, r]
    
    return circles

def generate_grid_initialization():
    """Generate initial configuration using grid-based approach"""
    circles = np.zeros((32, 3))
    
    # Create a 6x6 grid to distribute 32 circles
    rows, cols = 6, 6
    spacing_x = 1.0 / (cols + 1)
    spacing_y = 1.0 / (rows + 1)
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= 32:
                break
            # Position with jitter for better distribution
            x = (j + 1) * spacing_x + np.random.uniform(-0.05 * spacing_x, 0.05 * spacing_x)
            y = (i + 1) * spacing_y + np.random.uniform(-0.05 * spacing_y, 0.05 * spacing_y)
            
            # Initial radius - vary to encourage better packing
            r = min(spacing_x, spacing_y) * (0.25 + 0.2 * np.random.random())
            
            # Ensure circle fits in square
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[count] = [x, y, r]
            count += 1
            
        if count >= 32:
            break
    
    return circles

def generate_randomized_initialization():
    """Generate randomized initialization with overlap awareness"""
    circles = np.zeros((32, 3))
    
    # Start with more evenly distributed points
    for i in range(32):
        # Try to avoid overlaps by placing points carefully
        max_attempts = 1000
        placed = False
        attempts = 0
        
        while not placed and attempts < max_attempts:
            # Random position with some bias towards spreading
            x = 0.1 + 0.8 * np.random.random()
            y = 0.1 + 0.8 * np.random.random()
            
            # Initial radius - start small to allow for expansion
            r = 0.01 + 0.04 * np.random.random()
            
            # Check if it fits within boundaries
            if x - r >= 0 and x + r <= 1 and y - r >= 0 and y + r <= 1:
                # Check overlap with existing circles
                valid = True
                for j in range(i):
                    x_prev, y_prev, r_prev = circles[j]
                    distance = np.sqrt((x - x_prev)**2 + (y - y_prev)**2)
                    if distance < r + r_prev:
                        valid = False
                        break
                
                if valid:
                    circles[i] = [x, y, r]
                    placed = True
            attempts += 1
        
        # If couldn't place properly, use fallback
        if not placed:
            x = 0.1 + 0.8 * np.random.random()
            y = 0.1 + 0.8 * np.random.random()
            r = 0.03 + 0.02 * np.random.random()
            circles[i] = [x, y, r]
    
    return circles

def calculate_radius_sum(circles: np.ndarray) -> float:
    """Calculate sum of all radii"""
    return np.sum(circles[:, 2])

def is_valid_configuration(circles: np.ndarray) -> bool:
    """Check if configuration satisfies all constraints"""
    # Check containment
    for x, y, r in circles:
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check overlaps - optimized version using distance comparison
    for i in range(len(circles)):
        x1, y1, r1 = circles[i]
        for j in range(i + 1, len(circles)):
            x2, y2, r2 = circles[j]
            distance_sq = (x1 - x2)**2 + (y1 - y2)**2
            if distance_sq < (r1 + r2)**2:
                return False
    return True

def compute_max_radius_at_position(circles: np.ndarray, x: float, y: float, index: int) -> float:
    """Calculate maximum possible radius at given position without overlapping others"""
    # Maximum radius without violating containment
    max_radius = min(x, 1-x, y, 1-y)
    
    # Check overlaps with existing circles
    for i in range(len(circles)):
        if i != index:
            distance_sq = (x - circles[i, 0])**2 + (y - circles[i, 1])**2
            if distance_sq > 0:  # Avoid division by zero
                distance = math.sqrt(distance_sq)
                max_radius = min(max_radius, distance - circles[i, 2])
    
    return max(0.001, max_radius)

def apply_aggressive_local_optimization(circles):
    """Apply aggressive local optimization like inspiration 1 approach"""
    n = len(circles)
    
    # Run multiple passes of aggressive local improvement with increased search scope
    for iteration in range(15):  # More iterations
        improved = True
        iterations = 0
        while improved and iterations < 300:  # More iterations allowed
            improved = False
            iterations += 1
            
            # Try improving each circle systematically
            for i in range(n):
                orig_x, orig_y, orig_r = circles[i, 0], circles[i, 1], circles[i, 2]
                best_x, best_y, best_r = orig_x, orig_y, orig_r
                best_score = -orig_r  # Negative because we're minimizing negative sum
                
                # Even more comprehensive search around current position
                # Expand search range for better exploration
                search_dx = [-0.12, -0.08, -0.06, -0.04, -0.02, -0.01, 0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12]
                search_dy = [-0.12, -0.08, -0.06, -0.04, -0.02, -0.01, 0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12]
                
                # Try multiple positions in neighborhood
                for dx in search_dx:
                    for dy in search_dy:
                        if abs(dx) + abs(dy) == 0:
                            continue  # Skip center
                        
                        test_x = max(0.05, min(0.95, orig_x + dx))
                        test_y = max(0.05, min(0.95, orig_y + dy))
                        
                        # Calculate max possible radius at new position
                        max_radius = min(test_x, 1 - test_x, test_y, 1 - test_y)
                        for j in range(n):
                            if i != j:
                                distance = math.sqrt((test_x - circles[j, 0])**2 + (test_y - circles[j, 1])**2)
                                max_radius = min(max_radius, distance - circles[j, 2])
                        
                        test_r = max(0.001, max_radius)
                        
                        # Score is negative sum of radii (we want to maximize sum)
                        score = -test_r
                        
                        if score < best_score:
                            best_score = score
                            best_x, best_y, best_r = test_x, test_y, test_r
                            improved = True
                
                # Apply the best change
                circles[i, 0], circles[i, 1], circles[i, 2] = best_x, best_y, best_r
    
    return circles

def optimize_packaging(initial_circles):
    """Use advanced optimization to improve the packing"""
    n = len(initial_circles)
    
    # Flatten initial configuration for optimization
    initial_vars = []
    for i in range(n):
        x, y, r = initial_circles[i]
        initial_vars.extend([x, y, r])
    
    # Define objective function to maximize sum of radii
    def objective(vars_flat):
        # Extract variables
        circles = np.array(vars_flat).reshape(-1, 3)
        # We want to maximize sum of radii, so return negative
        return -np.sum(circles[:, 2])
    
    # Define constraints with better numerical stability
    def boundary_constraints(vars_flat):
        """Ensure all circles are within the unit square"""
        circles = np.array(vars_flat).reshape(-1, 3)
        constraints = []
        
        for i in range(n):
            x, y, r = circles[i]
            # Circle must fit entirely within square with margin for numerical stability
            constraints.append(x - r - 1e-6)      # x - r >= 1e-6
            constraints.append(1 - x - r - 1e-6)  # 1 - x - r >= 1e-6
            constraints.append(y - r - 1e-6)      # y - r >= 1e-6
            constraints.append(1 - y - r - 1e-6)  # 1 - y - r >= 1e-6
            
        return np.array(constraints)
    
    def overlap_constraints(vars_flat):
        """Ensure no two circles overlap with numerical tolerance"""
        circles = np.array(vars_flat).reshape(-1, 3)
        constraints = []
        
        for i, j in combinations(range(n), 2):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            distance_sq = (x1 - x2)**2 + (y1 - y2)**2
            min_distance_sq = (r1 + r2)**2
            
            # Add small tolerance to avoid numerical issues
            constraints.append(distance_sq - min_distance_sq - 1e-10)
            
        return np.array(constraints)
    
    # Set up bounds for variables with tighter ranges
    bounds = []
    for i in range(n):
        # x bounds
        bounds.append((0.001, 0.999))
        # y bounds
        bounds.append((0.001, 0.999))
        # r bounds - slightly smaller upper bound for safety
        bounds.append((0.001, 0.49))
    
    # Set up constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: boundary_constraints(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
    ]
    
    # Try multiple optimization methods for robustness
    methods_to_try = ['SLSQP', 'trust-constr']
    best_result = None
    best_sum = -np.inf
    
    for method in methods_to_try:
        try:
            result = minimize(objective, initial_vars, method=method, 
                             bounds=bounds, constraints=cons, 
                             options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10})
            
            if result.success:
                # Evaluate the result
                circles = result.x.reshape(-1, 3)
                current_sum = np.sum(circles[:, 2])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = result
                    
        except Exception as e:
            continue
    
    # If any method succeeded, return the best result
    if best_result is not None:
        optimized_circles = best_result.x.reshape(-1, 3)
        return optimized_circles
    else:
        # If all optimization failed, return initial configuration
        return initial_circles

def validate_and_refine(circles):
    """Validate solution and perform final refinement with better overlap resolution"""
    # Ensure all constraints are satisfied
    n = len(circles)
    
    # Make sure no circles overlap - more thorough check
    valid = True
    for i, j in combinations(range(n), 2):
        x1, y1, r1 = circles[i]
        x2, y2, r2 = circles[j]
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        if distance < r1 + r2 - 1e-8:  # Small tolerance for numerical errors
            valid = False
            break
    
    # If there are overlaps, perform a more sophisticated refinement
    if not valid:
        # Try a more aggressive refinement approach
        for iteration in range(500):  # More iterations for better refinement
            improved = False
            # Try to reduce overlaps by adjusting positions
            for i in range(n):
                x, y, r = circles[i]
                
                # Try to shrink radius if needed
                if r > 0.001:
                    # Check if reducing radius helps
                    new_r = max(0.001, r * 0.995)
                    # Check if this reduces overlaps with neighbors
                    valid_radius = True
                    for j in range(n):
                        if i != j:
                            x_j, y_j, r_j = circles[j]
                            distance = np.sqrt((x - x_j)**2 + (y - y_j)**2)
                            if distance < new_r + r_j:
                                valid_radius = False
                                break
                    if valid_radius:
                        circles[i, 2] = new_r
                        improved = True
                        
                # Try moving to reduce overlap
                best_x, best_y = x, y
                best_r = r
                
                # Try several positions around current location
                for dx in [-0.01, -0.005, 0, 0.005, 0.01]:
                    for dy in [-0.01, -0.005, 0, 0.005, 0.01]:
                        test_x = max(r, min(1-r, x + dx))
                        test_y = max(r, min(1-r, y + dy))
                        
                        # Check validity
                        valid_pos = True
                        for j in range(n):
                            if i != j:
                                x_j, y_j, r_j = circles[j]
                                distance = np.sqrt((test_x - x_j)**2 + (test_y - y_j)**2)
                                if distance < r + r_j:
                                    valid_pos = False
                                    break
                        if valid_pos:
                            # This position is valid, keep it
                            best_x, best_y = test_x, test_y
                            break
                    else:
                        continue
                    break
                
                if best_x != x or best_y != y:
                    circles[i, 0] = best_x
                    circles[i, 1] = best_y
                    improved = True
                    
            if not improved:
                break
    
    # Final cleanup - ensure all circles are valid
    for i in range(n):
        x, y, r = circles[i]
        # Ensure it fits within bounds
        circles[i, 0] = max(r, min(1-r, x))
        circles[i, 1] = max(r, min(1-r, y))
        circles[i, 2] = max(0.001, min(0.49, r))
    
    return circles


# EVOLVE-BLOCK-END
