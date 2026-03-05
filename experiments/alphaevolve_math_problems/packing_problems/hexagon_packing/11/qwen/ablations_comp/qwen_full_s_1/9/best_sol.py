# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import time
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

def hexagon_vertices(center_x, center_y, angle_deg, side_length=1.0):
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = np.radians(angle_deg)
    # Hexagon vertices in local coordinate system (centered at origin)
    local_vertices = np.array([
        [side_length, 0],
        [side_length/2, side_length * np.sqrt(3)/2],
        [-side_length/2, side_length * np.sqrt(3)/2],
        [-side_length, 0],
        [-side_length/2, -side_length * np.sqrt(3)/2],
        [side_length/2, -side_length * np.sqrt(3)/2]
    ])
    
    # Rotate and translate
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_vertices = local_vertices @ rotation_matrix.T
    global_vertices = rotated_vertices + np.array([center_x, center_y])
    
    return global_vertices


def create_hexagon_polygon(center_x, center_y, angle_deg, side_length=1.0):
    """Create a Shapely Polygon for a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, angle_deg, side_length)
    return Polygon(vertices)


def check_hexagon_containment(hexagon_poly, outer_hex_poly):
    """Check if a hexagon is completely contained within the outer hexagon."""
    return outer_hex_poly.contains(hexagon_poly) or outer_hex_poly.covers(hexagon_poly)


def check_hexagon_intersection(hex1_poly, hex2_poly):
    """Check if two hexagons intersect."""
    return hex1_poly.intersects(hex2_poly)


def calculate_outer_hex_side_length(inner_hex_data, outer_center=(0, 0), outer_angle=0):
    """Calculate the minimum side length of outer hexagon that contains all inner hexagons."""
    # Generate all inner hexagon polygons
    hex_polygons = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle_deg = inner_hex_data[i]
        hex_poly = create_hexagon_polygon(center_x, center_y, angle_deg)
        hex_polygons.append(hex_poly)
    
    # Collect all vertices from all inner hexagons
    all_vertices = []
    for hex_poly in hex_polygons:
        coords = list(hex_poly.exterior.coords)
        all_vertices.extend(coords[:-1])  # Exclude repeated last vertex
    
    if len(all_vertices) == 0:
        return 1.0
    
    all_vertices = np.array(all_vertices)
    
    # Find the center of all vertices (centroid)
    center_x = np.mean(all_vertices[:, 0])
    center_y = np.mean(all_vertices[:, 1])
    
    # Find maximum distance from center to any vertex
    distances = np.sqrt((all_vertices[:, 0] - center_x)**2 + (all_vertices[:, 1] - center_y)**2)
    max_distance = np.max(distances)
    
    # For a regular hexagon with side length s, the distance from center to vertex is s
    # So we need side length >= max_distance
    return max_distance


def evaluate_packing(inner_hex_data, verbose=False):
    """Evaluate a packing configuration and return the inverse of outer hex side length."""
    try:
        # Create all hexagon polygons
        hex_polygons = []
        for i in range(len(inner_hex_data)):
            center_x, center_y, angle_deg = inner_hex_data[i]
            hex_poly = create_hexagon_polygon(center_x, center_y, angle_deg)
            hex_polygons.append(hex_poly)
        
        # Check for overlaps between any pair of hexagons
        num_hexagons = len(hex_polygons)
        for i in range(num_hexagons):
            for j in range(i+1, num_hexagons):
                if check_hexagon_intersection(hex_polygons[i], hex_polygons[j]):
                    if verbose:
                        print(f"Overlap detected between hexagons {i} and {j}")
                    return 0.0  # Invalid configuration due to overlap
        
        # Calculate outer hexagon side length
        outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
        
        # Create outer hexagon polygon for containment check
        outer_hex_poly = create_hexagon_polygon(0, 0, 0, outer_side_length)
        
        # Check that all inner hexagons are contained within outer hexagon
        for i, hex_poly in enumerate(hex_polygons):
            if not check_hexagon_containment(hex_poly, outer_hex_poly):
                if verbose:
                    print(f"Hexagon {i} is not contained in outer hexagon")
                return 0.0  # Invalid configuration due to containment violation
        
        # Return inverse of outer side length (we want to maximize this)
        return 1.0 / outer_side_length if outer_side_length > 0 else 0.0
        
    except Exception as e:
        if verbose:
            print(f"Error in evaluation: {e}")
        return 0.0


def generate_best_known_configuration():
    """Generate the best known configuration that achieves the benchmark."""
    # This configuration from INSPIRATION 1 achieves ~0.2288 fitness
    # Let's use the precise coordinates that work well
    config = np.array([
        [0.0, 0.0, 0.0],           # center
        [0.0, 1.930092, 0.0],      # top
        [0.0, -1.930092, 0.0],     # bottom
        [1.669999, 0.960000, 0.0], # top-right
        [-1.669999, 0.960000, 0.0], # top-left
        [1.669999, -0.960000, 0.0], # bottom-right
        [-1.669999, -0.960000, 0.0], # bottom-left
        [3.339998, 0.0, 0.0],      # far right
        [-3.339998, 0.0, 0.0],     # far left
        [1.669999, 2.880000, 0.0], # top far right
        [-1.669999, 2.880000, 0.0], # top far left
    ])
    
    return config


def optimized_multi_start_approach():
    """Use a focused optimization approach with maximum aggressive parameter tuning."""
    # Start with the best known configuration
    initial_config = generate_best_known_configuration()
    
    # Define bounds for optimization - tighter bounds for better convergence
    bounds = [(-5, 5), (-5, 5), (0, 360)] * 11  # Tighter bounds around the expected region
    
    # Objective function for optimization (minimize negative fitness)
    def objective(params):
        hex_data = params.reshape((11, 3))
        fitness = evaluate_packing(hex_data)
        return -fitness if fitness > 0 else 1000000  # Penalize invalid configs
    
    best_fitness = evaluate_packing(initial_config)
    best_config = initial_config.copy()
    
    # Strategy 1: Most aggressive L-BFGS-B optimization with extreme precision
    try:
        result = minimize(objective, initial_config.flatten(), method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-14})
        
        if -result.fun > best_fitness:
            best_fitness = -result.fun
            best_config = result.x.reshape((11, 3))
            
    except Exception:
        pass
    
    # Strategy 2: Differential evolution with maximum thoroughness and polishing
    try:
        # Very high iterations and population size for thorough global search
        de_result = differential_evolution(objective, bounds, seed=42, maxiter=40, popsize=30, 
                                          disp=False, tol=1e-14, strategy='best1bin')
        
        # Refine with extremely precise local optimization
        refined = minimize(objective, de_result.x, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 70, 'ftol': 1e-14, 'gtol': 1e-14})
        
        if -refined.fun > best_fitness:
            best_fitness = -refined.fun
            best_config = refined.x.reshape((11, 3))
            
    except Exception:
        pass
    
    # Strategy 3: Multiple random restarts with extensive perturbations and multiple methods
    for seed in [123, 456, 789, 999, 111]:
        try:
            # Extensive perturbation approach with more systematic variation
            perturbed = initial_config.copy()
            np.random.seed(seed)
            for i in range(1, len(perturbed)):  # Skip center
                # Even larger perturbations to explore more thoroughly
                perturbed[i][0] += np.random.uniform(-0.5, 0.5)
                perturbed[i][1] += np.random.uniform(-0.5, 0.5)
                # Rotation with wider range
                perturbed[i][2] += np.random.uniform(-20, 20)
                perturbed[i][2] = perturbed[i][2] % 360
            
            # Try multiple optimization methods on this perturbed configuration
            for method in ['L-BFGS-B', 'TNC']:
                try:
                    result = minimize(objective, perturbed.flatten(), method=method, 
                                     bounds=bounds, options={'maxiter': 70, 'ftol': 1e-14})
                    
                    if -result.fun > best_fitness:
                        best_fitness = -result.fun
                        best_config = result.x.reshape((11, 3))
                        break  # Found a good solution, move to next seed
                except Exception:
                    continue
                    
        except Exception:
            continue
    
    # Strategy 4: Try COBYLA optimization method for additional exploration
    try:
        # Try COBYLA with very tight tolerance
        result = minimize(objective, initial_config.flatten(), method='COBYLA', 
                         bounds=bounds, options={'maxiter': 50, 'tol': 1e-14})
        
        if -result.fun > best_fitness:
            best_fitness = -result.fun
            best_config = result.x.reshape((11, 3))
            
    except Exception:
        pass
    
    # Strategy 5: Final ultra-aggressive refinement with multiple passes
    try:
        # Multiple rounds of ultra-precise optimization
        for round_num in range(3):
            if best_fitness > 0.1:
                result = minimize(objective, best_config.flatten(), method='L-BFGS-B', 
                                 bounds=bounds, options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-16})
                
                if -result.fun > best_fitness:
                    best_fitness = -result.fun
                    best_config = result.x.reshape((11, 3))
                    
    except Exception:
        pass
    
    # Strategy 6: Try a more focused optimization on the specific best result
    try:
        # Fine-tune the absolute best result with extremely high precision
        if best_fitness > 0.22:
            result = minimize(objective, best_config.flatten(), method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-16})
            
            if -result.fun > best_fitness:
                best_fitness = -result.fun
                best_config = result.x.reshape((11, 3))
                
    except Exception:
        pass
    
    return best_config


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses a focused optimization approach combining proven configurations with targeted optimization.
    
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) 
                       containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates 
                       and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Start with the best known configuration
    inner_hex_data = generate_best_known_configuration()
    
    # Try optimized multi-start approach to potentially improve upon the known good configuration
    start_time = time.time()
    
    # Use the optimized multi-start approach
    optimized_config = optimized_multi_start_approach()
    
    # Compare the results
    original_fitness = evaluate_packing(inner_hex_data)
    optimized_fitness = evaluate_packing(optimized_config)
    
    # Use the better of the two
    if optimized_fitness > original_fitness:
        inner_hex_data = optimized_config
    
    eval_time = time.time() - start_time
    
    # Calculate final outer hexagon side length
    outer_side_length = calculate_outer_hex_side_length(inner_hex_data)
    
    # Set outer hexagon at center with zero rotation
    outer_hex_data = np.array([0, 0, 0])
    
    # Final verification
    final_fitness = evaluate_packing(inner_hex_data)
    
    # Calculate benchmark ratio
    benchmark_ratio = final_fitness / 0.2544 if final_fitness > 0 else 0.0
    
    return inner_hex_data, outer_hex_data, outer_side_length


# EVOLVE-BLOCK-END
