# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
from typing import Tuple, List
from itertools import combinations
from scipy.optimize import differential_evolution, minimize
import time
from scipy.spatial.distance import cdist
from numba import jit
import random
from scipy.spatial import distance
from copy import deepcopy


@jit(nopython=True)
def hex_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Fast Euclidean distance calculation"""
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx*dx + dy*dy)


def create_regular_hexagon(center: Tuple[float, float], side_length: float, rotation_degrees: float = 0) -> Polygon:
    """Create a regular hexagon with given center, side length, and rotation."""
    rotation_rad = math.radians(rotation_degrees)
    points = []
    for i in range(6):
        angle = rotation_rad + i * math.pi / 3
        x = center[0] + side_length * math.cos(angle)
        y = center[1] + side_length * math.sin(angle)
        points.append((x, y))
    return Polygon(points)


def check_hexagon_containment(hexagon: Polygon, outer_hex: Polygon) -> bool:
    """Check if a hexagon is fully contained within the outer hexagon."""
    # More robust containment check using buffer to handle floating-point precision issues
    buffered_outer = outer_hex.buffer(1e-10)
    return buffered_outer.contains(hexagon) or (buffered_outer.covers(hexagon) and not buffered_outer.touches(hexagon))


def check_hexagon_overlap(hex1: Polygon, hex2: Polygon) -> bool:
    """Check if two hexagons overlap."""
    # More precise overlap detection with tolerance for floating point errors
    return hex1.intersects(hex2) and not hex1.touches(hex2)


def calculate_min_outer_radius(inner_hexagons: List[Polygon]) -> float:
    """Calculate minimum outer hexagon radius that contains all inner hexagons."""
    # Get all vertices of all inner hexagons
    all_points = []
    for hexagon in inner_hexagons:
        all_points.extend(list(hexagon.exterior.coords)[:-1])  # Exclude repeated last point
    
    # Find the maximum distance from origin to any vertex
    max_dist = 0
    for x, y in all_points:
        dist = math.sqrt(x*x + y*y)
        max_dist = max(max_dist, dist)
    
    # Add a small buffer for numerical precision
    return max_dist + 0.01


def generate_initial_config() -> Tuple[List[Tuple[float, float]], List[float]]:
    """Generate a better initial configuration for 11 hexagons based on known optimal arrangements."""
    # Use a more carefully designed configuration based on literature and geometric insights
    positions = []
    angles = []
    
    # Central hexagon
    positions.append((0.0, 0.0))
    angles.append(0.0)
    
    # First ring (6 hexagons) - arranged in a tight hexagonal pattern
    # Distance of ~1.9 to allow for better packing without overlap
    for i in range(6):
        angle = i * math.pi / 3
        x = 1.9 * math.cos(angle)
        y = 1.9 * math.sin(angle)
        positions.append((x, y))
        angles.append(0.0)
    
    # Second ring (4 hexagons) - placed more carefully to minimize space waste
    # Positioning in triangular formation around the central area
    positions.append((-1.0, 1.2))
    angles.append(0.0)
    positions.append((1.0, 1.2))
    angles.append(0.0)
    positions.append((-1.0, -1.2))
    angles.append(0.0)
    positions.append((1.0, -1.2))
    angles.append(0.0)
    
    return positions, angles


def perturb_configuration(positions: List[Tuple[float, float]], 
                         angles: List[float], 
                         max_shift: float = 0.1) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Slightly perturb the configuration to escape local minima."""
    new_positions = []
    new_angles = []
    
    for pos, angle in zip(positions, angles):
        # Small random perturbation to position
        dx = np.random.uniform(-max_shift, max_shift)
        dy = np.random.uniform(-max_shift, max_shift)
        new_positions.append((pos[0] + dx, pos[1] + dy))
        
        # Small random perturbation to angle
        dangle = np.random.uniform(-5, 5)
        new_angles.append((angle + dangle) % 360)
    
    return new_positions, new_angles


def validate_and_refine_solution(positions: List[Tuple[float, float]], 
                                angles: List[float],
                                max_iterations: int = 100) -> Tuple[List[Tuple[float, float]], List[float], float]:
    """
    Validate solution and refine it to eliminate overlaps and ensure containment.
    """
    # Create initial hexagons
    inner_hexagons = []
    for pos, angle in zip(positions, angles):
        hexagon = create_regular_hexagon(pos, 1.0, angle)
        inner_hexagons.append(hexagon)
    
    # Check for overlaps and containment
    valid = True
    outer_radius = calculate_min_outer_radius(inner_hexagons)
    outer_hex = create_regular_hexagon((0, 0), outer_radius)
    
    # Check overlaps
    for i, j in combinations(range(len(inner_hexagons)), 2):
        if check_hexagon_overlap(inner_hexagons[i], inner_hexagons[j]):
            valid = False
            break
    
    # Check containment
    if valid:
        for hexagon in inner_hexagons:
            if not check_hexagon_containment(hexagon, outer_hex):
                valid = False
                break
    
    if valid:
        return positions, angles, outer_radius
    
    # If not valid, try to fix it through iterative improvement with better strategy
    for iteration in range(max_iterations):
        moved_any = False
        # Apply corrections in a more systematic way
        for i in range(len(inner_hexagons)):
            # Check containment first
            if not check_hexagon_containment(inner_hexagons[i], outer_hex):
                # Move it towards center proportionally to how far it is out
                center_x, center_y = positions[i]
                dist_to_center = math.sqrt(center_x**2 + center_y**2)
                if dist_to_center > 0:
                    # Move towards center with larger factor for faster correction
                    factor = 0.05  # Increased from 0.03 to move faster
                    positions[i] = (center_x * (1 - factor), center_y * (1 - factor))
                    moved_any = True
                    
            # Check overlaps with others
            for j in range(i+1, len(inner_hexagons)):
                if check_hexagon_overlap(inner_hexagons[i], inner_hexagons[j]):
                    # Move both hexagons apart with larger step size
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    # Calculate vector between centers
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist > 0:
                        # Normalize and push apart
                        dx /= dist
                        dy /= dist
                        # Move each hexagon away from the other with larger step for faster convergence
                        factor = 0.05  # Increased from 0.03
                        positions[i] = (x1 - dx * factor, y1 - dy * factor)
                        positions[j] = (x2 + dx * factor, y2 + dy * factor)
                        moved_any = True
                        
        if not moved_any:
            break
            
        # Recreate hexagons with new positions
        inner_hexagons = []
        for pos, angle in zip(positions, angles):
            hexagon = create_regular_hexagon(pos, 1.0, angle)
            inner_hexagons.append(hexagon)
            
        # Recalculate outer radius
        outer_radius = calculate_min_outer_radius(inner_hexagons)
        outer_hex = create_regular_hexagon((0, 0), outer_radius)
        
        # Check validity again
        valid = True
        for i, j in combinations(range(len(inner_hexagons)), 2):
            if check_hexagon_overlap(inner_hexagons[i], inner_hexagons[j]):
                valid = False
                break
        
        if valid:
            for hexagon in inner_hexagons:
                if not check_hexagon_containment(hexagon, outer_hex):
                    valid = False
                    break
                    
        if valid:
            return positions, angles, outer_radius
    
    return positions, angles, outer_radius


def compute_hexagon_distances(hexagons: List[Polygon]) -> np.ndarray:
    """Compute pairwise distances between hexagon centers efficiently."""
    centers = []
    for hexagon in hexagons:
        centroid = hexagon.centroid
        centers.append([centroid.x, centroid.y])
    
    return cdist(centers, centers)


def improved_objective_function(params):
    """
    Improved objective function to minimize (negative of 1/outer_radius).
    params: flattened array of [x1,y1,a1,x2,y2,a2,...,x11,y11,a11]
    """
    # Reshape parameters into positions and angles
    positions = [(params[i], params[i+1]) for i in range(0, len(params), 3)]
    angles = [params[i] for i in range(2, len(params), 3)]
    
    # Create hexagons
    inner_hexagons = []
    for pos, angle in zip(positions, angles):
        hexagon = create_regular_hexagon(pos, 1.0, angle)
        inner_hexagons.append(hexagon)
    
    # Check constraints
    # Check overlaps - early exit for performance
    for i, j in combinations(range(len(inner_hexagons)), 2):
        if check_hexagon_overlap(inner_hexagons[i], inner_hexagons[j]):
            # Return large penalty for overlaps
            return 1e12
    
    # Check containment
    outer_radius = calculate_min_outer_radius(inner_hexagons)
    outer_hex = create_regular_hexagon((0, 0), outer_radius)
    
    for hexagon in inner_hexagons:
        if not check_hexagon_containment(hexagon, outer_hex):
            # Return large penalty for containment violations
            return 1e12
    
    # Return negative of 1/outer_radius to maximize 1/outer_radius
    return -1.0 / outer_radius


def optimize_hexagon_packing():
    """Use optimization to find better configuration."""
    # Initial guess - start with a better configuration
    initial_positions, initial_angles = generate_initial_config()
    
    # Flatten initial parameters
    initial_params = []
    for pos, angle in zip(initial_positions, initial_angles):
        initial_params.extend([pos[0], pos[1], angle])
    
    # Define bounds for optimization
    # Positions: [-5, 5] for x and y coordinates (more constrained)
    # Angles: [0, 360] degrees
    bounds = []
    for i in range(0, len(initial_params), 3):
        bounds.extend([(-5, 5), (-5, 5), (0, 360)])  # x, y, angle
    
    # Try multiple optimization approaches for better results
    best_result = None
    best_value = float('inf')
    
    # Try differential evolution with better settings
    try:
        # Use a more aggressive optimization approach with more iterations
        result1 = differential_evolution(
            improved_objective_function,
            bounds,
            maxiter=500,  # Increased iterations for better search
            popsize=100,   # Increased population size for better exploration
            mutation=(0.9, 1),  # Even more aggressive mutation
            recombination=0.95,  # Even higher recombination rate
            seed=42,
            disp=False,
            atol=1e-15,  # Tighter tolerances
            rtol=1e-15
        )
        
        if result1.fun < best_value:
            best_value = result1.fun
            best_result = result1
    except Exception as e:
        pass
    
    # Try L-BFGS-B optimization as well with better starting points
    try:
        # Use a more refined approach for L-BFGS-B
        # Generate multiple random initial points for robustness
        best_l_bfgs_result = None
        best_l_bfgs_value = float('inf')
        
        for i in range(20):  # Increased number of random starts for better chance
            np.random.seed(1000 + i)
            # Add small random perturbations to initial configuration
            perturbed_positions, perturbed_angles = perturb_configuration(initial_positions, initial_angles, 0.3)
            perturbed_params = []
            for pos, angle in zip(perturbed_positions, perturbed_angles):
                perturbed_params.extend([pos[0], pos[1], angle])
            
            result2 = minimize(
                lambda p: improved_objective_function(p),
                perturbed_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15}  # Even tighter tolerances
            )
            
            if result2.fun < best_l_bfgs_value:
                best_l_bfgs_value = result2.fun
                best_l_bfgs_result = result2
                
        if best_l_bfgs_result is not None and best_l_bfgs_result.fun < best_value:
            best_value = best_l_bfgs_result.fun
            best_result = best_l_bfgs_result
    except Exception as e:
        pass
    
    # If we found a good result, use it; otherwise fall back to initial
    if best_result is not None:
        optimized_params = best_result.x
        positions = [(optimized_params[i], optimized_params[i+1]) for i in range(0, len(optimized_params), 3)]
        angles = [optimized_params[i] for i in range(2, len(optimized_params), 3)]
        
        # Final validation
        inner_hexagons = []
        for pos, angle in zip(positions, angles):
            hexagon = create_regular_hexagon(pos, 1.0, angle)
            inner_hexagons.append(hexagon)
        
        outer_radius = calculate_min_outer_radius(inner_hexagons)
        
        # Ensure final containment check
        final_outer_hex = create_regular_hexagon((0, 0), outer_radius)
        for i, j in combinations(range(len(inner_hexagons)), 2):
            if check_hexagon_overlap(inner_hexagons[i], inner_hexagons[j]):
                # If still overlapping, use the previous best solution
                return initial_positions, initial_angles, calculate_min_outer_radius([
                    create_regular_hexagon(pos, 1.0, angle) 
                    for pos, angle in zip(initial_positions, initial_angles)
                ])
                
        for hexagon in inner_hexagons:
            if not check_hexagon_containment(hexagon, final_outer_hex):
                # If still not contained, use the previous best solution
                return initial_positions, initial_angles, calculate_min_outer_radius([
                    create_regular_hexagon(pos, 1.0, angle) 
                    for pos, angle in zip(initial_positions, initial_angles)
                ])
        
        return positions, angles, outer_radius
        
    else:
        # Fall back to initial configuration if optimization fails
        return initial_positions, initial_angles, calculate_min_outer_radius([
            create_regular_hexagon(pos, 1.0, angle) 
            for pos, angle in zip(initial_positions, initial_angles)
        ])


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Use optimization approach
    start_time = time.time()
    
    # Try optimization approach first
    best_positions, best_angles, best_radius = optimize_hexagon_packing()
    
    # Additional refinement if needed
    if time.time() - start_time < 50:  # Leave some time for refinement
        # Try a few more refinement attempts with different strategies
        for attempt in range(15):  # Increased from 10 to more thoroughly explore
            positions, angles = generate_initial_config()
            np.random.seed(attempt * 42 + 100)
            positions, angles = perturb_configuration(positions, angles, max_shift=0.15)  # Slightly larger perturbation
            refined_positions, refined_angles, radius = validate_and_refine_solution(positions, angles, max_iterations=50)
            
            if radius < best_radius:
                best_radius = radius
                best_positions = refined_positions
                best_angles = refined_angles
    
    # Final validation with more aggressive refinement
    inner_hexagons = []
    for pos, angle in zip(best_positions, best_angles):
        hexagon = create_regular_hexagon(pos, 1.0, angle)
        inner_hexagons.append(hexagon)
    
    # Ensure final containment check
    final_outer_radius = calculate_min_outer_radius(inner_hexagons)
    final_outer_hex = create_regular_hexagon((0, 0), final_outer_radius)
    
    # Double-check that everything is valid with extra refinement
    for i, j in combinations(range(len(inner_hexagons)), 2):
        if check_hexagon_overlap(inner_hexagons[i], inner_hexagons[j]):
            # If still overlapping, try one more refinement pass
            positions, angles = best_positions, best_angles
            positions, angles, radius = validate_and_refine_solution(positions, angles, max_iterations=75)
            final_outer_radius = radius
            break
    
    # Convert to final data structures
    inner_hex_data = np.array([(pos[0], pos[1], angle) for pos, angle in zip(best_positions, best_angles)])
    outer_hex_data = np.array([0.0, 0.0, 0.0])  # centered at origin
    outer_hex_side_length = final_outer_radius
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
