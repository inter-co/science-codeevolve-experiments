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
    Uses a hybrid approach combining geometric initialization, physics-inspired optimization,
    and multi-start strategies to beat the benchmark.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Multi-start approach: try several different initializations
    best_circles = None
    best_sum = -np.inf
    
    # Strategy 1: Voronoi-based initialization (enhanced)
    circles1 = generate_enhanced_voronoi_initialization()
    
    # Strategy 2: Golden ratio distribution with local refinement
    circles2 = generate_golden_ratio_initialization()
    
    # Strategy 3: Grid-based with random perturbations
    circles3 = generate_grid_initialization()
    
    # Strategy 4: Hybrid approach with center concentration
    circles4 = generate_hybrid_initialization()
    
    # Strategy 5: Randomized with overlap-aware initialization
    circles5 = generate_randomized_initialization()
    
    initial_strategies = [circles1, circles2, circles3, circles4, circles5]
    
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
    
    # If we still don't have a good solution, do final refinement
    if best_circles is None:
        # Fallback to a simple but effective initialization
        best_circles = generate_simple_initialization()
        best_circles = optimize_packaging(best_circles)
        best_circles = validate_and_refine(best_circles)
    
    return best_circles

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
            
            # Also consider the cell area for better density estimation
            # Use a more accurate Voronoi cell area approximation
            cell_area = 0.01  # Placeholder - for now use a reasonable estimate
            
            # Combined radius calculation
            r = min(0.15, min_dist * 0.25)  # Base on neighbor distance
            r = max(0.01, r)  # Minimum radius
            
            # Adjust based on area considerations
            if cell_area > 0:
                area_based_radius = np.sqrt(cell_area / np.pi) * 0.5
                r = min(r, area_based_radius)
            
            # Ensure it fits in the square
            x = max(r, min(1-r, x))
            y = max(r, min(1-r, y))
            
            circles[i] = [x, y, r]
    
    return circles

def generate_golden_ratio_initialization():
    """Generate initial configuration using golden ratio distribution"""
    circles = np.zeros((32, 3))
    
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    for i in range(32):
        # Distribute points using golden ratio for better coverage
        x = ((i * phi) % 1) * 0.8 + 0.1  # Scale to [0.1, 0.9]
        y = (i / 32) * 0.8 + 0.1
        
        # Initial radius - vary more to allow for better packing
        r = 0.03 + 0.05 * np.random.random()
        
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

def generate_hybrid_initialization():
    """Generate hybrid initial configuration combining center and edge placement"""
    circles = np.zeros((32, 3))
    
    # Place some circles near the center (more concentrated)
    center_count = 12
    for i in range(center_count):
        # Place in a circular pattern around center with some randomness
        angle = 2 * np.pi * i / center_count
        radius = 0.1 + 0.3 * np.random.random()  # Spread out from center
        x = 0.5 + radius * np.cos(angle)
        y = 0.5 + radius * np.sin(angle)
        r = 0.03 + 0.03 * np.random.random()
        
        # Ensure it fits in square
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        
        circles[i] = [x, y, r]
    
    # Place remaining circles near edges and corners
    remaining = 32 - center_count
    for i in range(remaining):
        # Place near edges or corners for better spread
        edge_type = i % 4
        if edge_type == 0:  # Near bottom
            x = 0.1 + 0.8 * np.random.random()
            y = 0.1 + 0.05 * np.random.random()
        elif edge_type == 1:  # Near top
            x = 0.1 + 0.8 * np.random.random()
            y = 0.9 - 0.05 * np.random.random()
        elif edge_type == 2:  # Near left
            x = 0.1 + 0.05 * np.random.random()
            y = 0.1 + 0.8 * np.random.random()
        else:  # Near right
            x = 0.9 - 0.05 * np.random.random()
            y = 0.1 + 0.8 * np.random.random()
            
        r = 0.02 + 0.04 * np.random.random()
        
        # Ensure it fits in square
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        
        circles[center_count + i] = [x, y, r]
    
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

def generate_simple_initialization():
    """Simple but effective initialization"""
    circles = np.zeros((32, 3))
    
    # Create a good starting pattern
    for i in range(32):
        # Distribute more evenly with some randomness
        x = 0.1 + 0.8 * (i % 8) / 7.0 + np.random.uniform(-0.02, 0.02)
        y = 0.1 + 0.8 * (i // 8) / 3.0 + np.random.uniform(-0.02, 0.02)
        
        # Initial radius - vary more to allow better packing
        r = 0.03 + 0.03 * np.random.random()
        
        # Ensure it fits in square
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        
        circles[i] = [x, y, r]
    
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
                             options={'maxiter': 1500, 'ftol': 1e-9, 'gtol': 1e-9})
            
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
                    new_r = max(0.001, r * 0.99)
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
