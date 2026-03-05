# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon, Point
import math
import time
from itertools import combinations

def get_hexagon_vertices(center, radius=1, rotation=0):
    """Get vertices of a unit hexagon at given position and rotation."""
    # Unit hexagon has side length 1, so circumradius is 1
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    points = np.column_stack([center[0] + radius * np.cos(angles),
                             center[1] + radius * np.sin(angles)])
    return points[:-1]

def check_containment(hexagon_vertices, outer_hex_center, outer_hex_radius):
    """Check if all vertices of hexagon are inside the outer hexagon."""
    outer_hex_points = get_hexagon_vertices(outer_hex_center, outer_hex_radius, 0)
    outer_polygon = Polygon(outer_hex_points)
    
    # Use buffer to handle floating point precision issues
    epsilon = 1e-10
    for vertex in hexagon_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.buffer(epsilon).contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely with epsilon handling."""
    try:
        poly1 = Polygon(hex1_vertices)
        poly2 = Polygon(hex2_vertices)
        # Use buffer with epsilon to handle floating point precision issues
        epsilon = 1e-10
        return poly1.buffer(epsilon).intersects(poly2.buffer(epsilon))
    except:
        # Fallback to a simple distance check if polygon creation fails
        centers1 = np.mean(hex1_vertices, axis=0)
        centers2 = np.mean(hex2_vertices, axis=0)
        distance = np.linalg.norm(centers1 - centers2)
        # Two unit hexagons overlap if their centers are less than 2 units apart
        return distance < 2.0

def calculate_outer_hex_side_length(inner_hex_data, outer_hex_center=(0, 0)):
    """Calculate minimum outer hexagon side length needed to contain all inner hexagons."""
    max_distance = 0
    
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        hex_vertices = get_hexagon_vertices(center, 1, rotation)
        
        # Find maximum distance from center to any vertex
        distances = np.sqrt(np.sum((hex_vertices - center)**2, axis=1))
        max_dist_from_center = np.max(distances)
        
        # Add this to the distance from outer center to inner center
        center_distance = np.sqrt(np.sum((np.array(center) - np.array(outer_hex_center))**2))
        total_distance = center_distance + max_dist_from_center
        
        max_distance = max(max_distance, total_distance)
    
    # Convert to side length of outer hexagon (circumradius = side length for regular hexagon)
    return max_distance

def is_valid_configuration(inner_hex_data, outer_hex_center=(0, 0), outer_hex_radius=10):
    """Check if a configuration is valid (no overlaps, all contained)."""
    # Check containment first
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i, 0], inner_hex_data[i, 1])
        rotation = inner_hex_data[i, 2]
        hex_vertices = get_hexagon_vertices(center, 1, rotation)
        
        if not check_containment(hex_vertices, outer_hex_center, outer_hex_radius):
            return False
    
    # Check overlaps - use more efficient approach with early termination
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center1 = (inner_hex_data[i, 0], inner_hex_data[i, 1])
            rotation1 = inner_hex_data[i, 2]
            center2 = (inner_hex_data[j, 0], inner_hex_data[j, 1])
            rotation2 = inner_hex_data[j, 2]
            
            hex1_vertices = get_hexagon_vertices(center1, 1, rotation1)
            hex2_vertices = get_hexagon_vertices(center2, 1, rotation2)
            
            if check_overlap(hex1_vertices, hex2_vertices):
                return False
    
    return True

def objective_with_constraints(params, outer_hex_center=(0, 0)):
    """
    Objective function that penalizes constraint violations.
    Returns negative reciprocal of outer hexagon side length for maximization.
    """
    # Reshape params into 11 hexagons with (x, y, rotation) each
    inner_hex_data = params.reshape(-1, 3)
    
    # Calculate required outer hexagon size
    outer_radius = calculate_outer_hex_side_length(inner_hex_data, outer_hex_center)
    
    # Check validity and penalize invalid configurations heavily
    if not is_valid_configuration(inner_hex_data, outer_hex_center, outer_radius):
        # Large penalty for constraint violations - make it extremely negative
        return 1e10  # We minimize this, so a large positive number means bad
    
    # Return negative because we want to maximize 1/outer_radius (minimize outer_radius)
    # But since we're minimizing, return the negative of 1/outer_radius
    return -1.0 / outer_radius

def generate_best_initial_patterns():
    """Generate the best initial patterns from successful inspirations."""
    sqrt3 = math.sqrt(3)
    
    # Pattern from inspiration 2 that achieved 0.242855 - this is our best reference
    pattern1 = np.array([
        [0, 0, 0],           # center
        [0, 1.8, 0],         # top
        [sqrt3*0.9, -0.9, 0], # top-right  
        [-sqrt3*0.9, -0.9, 0], # top-left
        [0, -1.8, 0],        # bottom
        [sqrt3*0.9, 0.9, 0], # bottom-right
        [-sqrt3*0.9, 0.9, 0], # bottom-left
        [sqrt3*1.8, 0, 0],   # far right
        [-sqrt3*1.8, 0, 0],  # far left
        [sqrt3*0.9, 2.7, 0], # top-top-right
        [-sqrt3*0.9, 2.7, 0] # top-top-left
    ])
    
    # Pattern from inspiration 3 that achieved 0.237852 - slightly different
    pattern2 = np.array([
        [0, 0, 0],           # center
        [0, 1.85, 0],        # top
        [sqrt3*0.925, -0.925, 0], # top-right  
        [-sqrt3*0.925, -0.925, 0], # top-left
        [0, -1.85, 0],       # bottom
        [sqrt3*0.925, 0.925, 0], # bottom-right
        [-sqrt3*0.925, 0.925, 0], # bottom-left
        [sqrt3*1.85, 0, 0],  # far right
        [-sqrt3*1.85, 0, 0], # far left
        [sqrt3*0.925, 2.775, 0], # top-top-right
        [-sqrt3*0.925, 2.775, 0] # top-top-left
    ])
    
    # Symmetric pattern that works well
    pattern3 = np.array([
        [0, 0, 0],           # center
        [0, 1.9, 0],         # top
        [sqrt3*0.95, -0.95, 0], # top-right  
        [-sqrt3*0.95, -0.95, 0], # top-left
        [0, -1.9, 0],        # bottom
        [sqrt3*0.95, 0.95, 0], # bottom-right
        [-sqrt3*0.95, 0.95, 0], # bottom-left
        [sqrt3*1.9, 0, 0],   # far right
        [-sqrt3*1.9, 0, 0],  # far left
        [sqrt3*0.95, 2.85, 0], # top-top-right
        [-sqrt3*0.95, 2.85, 0] # top-top-left
    ])
    
    # Pattern from inspiration 1 that achieved ~0.2408
    pattern4 = np.array([
        [0, 0, 0],           # center
        [0, 1.82, 0],        # top
        [sqrt3*0.91, -0.91, 0], # top-right  
        [-sqrt3*0.91, -0.91, 0], # top-left
        [0, -1.82, 0],       # bottom
        [sqrt3*0.91, 0.91, 0], # bottom-right
        [-sqrt3*0.91, 0.91, 0], # bottom-left
        [sqrt3*1.82, 0, 0],  # far right
        [-sqrt3*1.82, 0, 0], # far left
        [sqrt3*0.91, 2.73, 0], # top-top-right
        [-sqrt3*0.91, 2.73, 0] # top-top-left
    ])
    
    # Additional refined pattern
    pattern5 = np.array([
        [0, 0, 0],           # center
        [0, 1.87, 0],        # top
        [sqrt3*0.935, -0.935, 0], # top-right  
        [-sqrt3*0.935, -0.935, 0], # top-left
        [0, -1.87, 0],       # bottom
        [sqrt3*0.935, 0.935, 0], # bottom-right
        [-sqrt3*0.935, 0.935, 0], # bottom-left
        [sqrt3*1.87, 0, 0],  # far right
        [-sqrt3*1.87, 0, 0], # far left
        [sqrt3*0.935, 2.805, 0], # top-top-right
        [-sqrt3*0.935, 2.805, 0] # top-top-left
    ])
    
    return [pattern1, pattern2, pattern3, pattern4, pattern5]

def improved_local_search(initial_solution, max_iterations=100):
    """Enhanced local search with adaptive perturbations and simulated annealing."""
    current_solution = initial_solution.copy()
    current_score = -1.0 / calculate_outer_hex_side_length(current_solution)
    
    # Track best solution found
    best_solution = current_solution.copy()
    best_score = current_score
    
    # Use a more sophisticated cooling schedule for simulated annealing-like behavior
    for iteration in range(max_iterations):
        # Adaptive perturbation strategy
        test_solution = current_solution.copy()
        
        # Choose which hexagon to perturb (avoid center hexagon for better stability)
        hex_idx = np.random.randint(1, 11)  # Skip index 0 (center hexagon)
        
        # Adjust perturbation strength based on iteration progress
        progress = iteration / max_iterations
        if progress < 0.3:
            # Early iterations: more aggressive perturbations
            perturbation_scale = 0.2
        elif progress < 0.7:
            # Middle iterations: moderate perturbations
            perturbation_scale = 0.1
        else:
            # Late iterations: fine-tuning
            perturbation_scale = 0.05
        
        test_solution[hex_idx, 0] += np.random.uniform(-perturbation_scale, perturbation_scale)
        test_solution[hex_idx, 1] += np.random.uniform(-perturbation_scale, perturbation_scale)
        test_solution[hex_idx, 2] += np.random.uniform(-10 * perturbation_scale, 10 * perturbation_scale)
        
        # Check if new solution is valid
        outer_radius = calculate_outer_hex_side_length(test_solution)
        if is_valid_configuration(test_solution, (0, 0), outer_radius):
            test_score = -1.0 / outer_radius
            if test_score > current_score:
                current_solution = test_solution
                current_score = test_score
                
                # Update best solution
                if test_score > best_score:
                    best_solution = test_solution.copy()
                    best_score = test_score
            else:
                # Occasionally accept worse solutions to escape local minima
                # Temperature decreases over time
                temperature = max(0.01, 0.1 * (1 - progress))
                if np.random.random() < temperature:  # Accept worse solution with probability based on temperature
                    current_solution = test_solution
                    current_score = test_score
        else:
            # If invalid, revert to previous solution
            pass
    
    return best_solution, best_score

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a sophisticated multi-stage approach with enhanced optimization.
    """
    start_time = time.time()
    
    # Generate multiple high-quality initial patterns
    patterns = generate_best_initial_patterns()
    
    best_result = None
    best_score = float('-inf')
    best_outer_radius = float('inf')
    
    # Try multiple optimization runs with different strategies
    for i, initial_pattern in enumerate(patterns):
        # Early termination if we're running close to time limit
        if time.time() - start_time > 55:  # Leave 5 seconds for final processing
            break
            
        try:
            # Flatten the initial pattern for optimization
            initial_flat = initial_pattern.flatten()
            
            # Define bounds for optimization with more careful ranges
            # For positions: (-10, 10) to balance exploration and feasibility
            # For rotations: (0, 360) degrees - allowing full rotation freedom
            bounds = []
            for j in range(33):  # 11 hexagons * 3 params
                if j % 3 == 2:  # Rotation parameter
                    bounds.append((0, 360))  # Full rotation range
                else:  # Position parameters
                    bounds.append((-10, 10))
            
            # Try differential evolution with reasonable parameters for time constraints
            result_de = differential_evolution(
                lambda x: objective_with_constraints(x, (0, 0)),
                bounds,
                maxiter=100,  # Reduced iterations for faster convergence
                popsize=25,   # Smaller population for efficiency
                seed=42+i,
                disp=False,
                strategy='best1bin',
                atol=1e-5,    # Tighter tolerance for better accuracy
                rtol=1e-5
            )
            
            # Reshape result back
            optimized_pattern = result_de.x.reshape(-1, 3)
            
            # Verify the solution
            outer_radius = calculate_outer_hex_side_length(optimized_pattern)
            if is_valid_configuration(optimized_pattern, (0, 0), outer_radius):
                # Score is negative of 1/outer_radius (since we want to maximize 1/outer_radius)
                score = -1.0 / outer_radius
                if score > best_score:
                    best_score = score
                    best_outer_radius = outer_radius
                    best_result = optimized_pattern.copy()
                    
        except Exception as e:
            continue
    
    # If we didn't find a better solution through DE, try local search on best pattern
    if best_result is None:
        # Fall back to the best of our initial patterns
        best_pattern = generate_best_initial_patterns()[0]  # Use first pattern as fallback
        outer_radius = calculate_outer_hex_side_length(best_pattern)
        best_result = best_pattern.copy()
        best_outer_radius = outer_radius
    
    # Apply improved local search to further refine the solution
    try:
        refined_solution, refined_score = improved_local_search(best_result, max_iterations=50)
        if refined_score > best_score:
            best_result = refined_solution
            best_score = refined_score
            best_outer_radius = 1.0 / (-refined_score)
    except:
        pass
    
    # Final verification and calculation
    outer_hex_data = np.array([0, 0, 0])
    
    return best_result, outer_hex_data, best_outer_radius


# EVOLVE-BLOCK-END
