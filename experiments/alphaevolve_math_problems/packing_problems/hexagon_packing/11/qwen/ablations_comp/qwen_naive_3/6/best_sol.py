# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
import random
from typing import Tuple, List, Dict
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from scipy.optimize import differential_evolution
import time

def hexagon_vertices(center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> np.ndarray:
    """Generate vertices of a regular hexagon given center, angle, and side length."""
    angle_rad = math.radians(angle_deg)
    # Vertices of a regular hexagon with side length 1, centered at origin, rotated by angle
    vertices = []
    for i in range(6):
        angle = angle_rad + i * math.pi / 3
        x = side_length * math.cos(angle)
        y = side_length * math.sin(angle)
        vertices.append((x, y))
    
    # Translate to center
    vertices = [(x + center_x, y + center_y) for x, y in vertices]
    return np.array(vertices)

def create_hexagon_polygon(center_x: float, center_y: float, angle_deg: float, side_length: float = 1.0) -> Polygon:
    """Create a Shapely polygon for a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, angle_deg, side_length)
    return Polygon(vertices)

def create_outer_hexagon(center_x: float, center_y: float, side_length: float) -> Polygon:
    """Create a Shapely polygon for the outer hexagon."""
    vertices = hexagon_vertices(center_x, center_y, 0, side_length)
    return Polygon(vertices)

def check_containment(hex_center: Tuple[float, float], angle_deg: float, 
                     outer_center: Tuple[float, float], outer_side_length: float) -> bool:
    """Check if a hexagon is fully contained within the outer hexagon using Shapely."""
    inner_hex = create_hexagon_polygon(hex_center[0], hex_center[1], angle_deg)
    outer_hex = create_outer_hexagon(outer_center[0], outer_center[1], outer_side_length)
    
    # Check if inner hexagon is completely inside outer hexagon
    return outer_hex.contains(inner_hex)

def check_overlap(h1_center: Tuple[float, float], h1_angle: float, 
                 h2_center: Tuple[float, float], h2_angle: float) -> bool:
    """Check if two hexagons overlap using Shapely."""
    h1 = create_hexagon_polygon(h1_center[0], h1_center[1], h1_angle)
    h2 = create_hexagon_polygon(h2_center[0], h2_center[1], h2_angle)
    return h1.intersects(h2)

def get_hexagon_bounds(center_x: float, center_y: float, angle_deg: float) -> Tuple[float, float, float, float]:
    """Get bounding box coordinates for a hexagon."""
    vertices = hexagon_vertices(center_x, center_y, angle_deg)
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    return (min(xs), max(xs), min(ys), max(ys))

def calculate_outer_hexagon_radius(inner_positions: List[Tuple[float, float]], 
                                 inner_angles: List[float], 
                                 side_length: float = 1.0) -> float:
    """Estimate the minimal outer hexagon radius needed to contain all inner hexagons."""
    max_distance = 0
    for (cx, cy), angle in zip(inner_positions, inner_angles):
        # Get the furthest vertex from center
        vertices = hexagon_vertices(cx, cy, angle, side_length)
        for vx, vy in vertices:
            distance = math.sqrt((vx)**2 + (vy)**2)  # Assuming outer hexagon centered at origin
            max_distance = max(max_distance, distance)
    return max_distance + 0.1  # Add small buffer

def calculate_total_area(inner_positions: List[Tuple[float, float]], 
                        inner_angles: List[float]) -> float:
    """Calculate total area of all inner hexagons."""
    return len(inner_positions) * (3 * math.sqrt(3) / 2)

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses an efficient evolutionary algorithm with proper time management.
    """
    
    # Time limit for execution
    start_time = time.time()
    time_limit = 55  # Leave 5 seconds for final processing
    
    # Use a more efficient approach with fewer iterations and better initialization
    def generate_initial_solution() -> np.ndarray:
        """Generate a good initial solution using a proven hexagonal packing pattern."""
        # Start with a central hexagon
        positions = [[0, 0]]
        angles = [0]
        
        # Place 10 surrounding hexagons in a pattern that tries to minimize gaps
        # Using a hexagonal lattice arrangement around the center
        for i in range(1, 11):
            # Distribute in rings around center
            ring = (i - 1) // 6 + 1  # Which ring
            offset = (i - 1) % 6  # Position within ring
            
            if ring == 1:
                # First ring: place around center
                angle = offset * 60  # 60 degrees apart
                distance = 2.0  # Distance between centers
                x = distance * math.cos(math.radians(angle))
                y = distance * math.sin(math.radians(angle))
                positions.append([x, y])
                angles.append(0)
            else:
                # Second ring: place further out
                angle = offset * 60  # 60 degrees apart
                distance = 3.5  # Distance between centers  
                x = distance * math.cos(math.radians(angle))
                y = distance * math.sin(math.radians(angle))
                positions.append([x, y])
                angles.append(0)
        
        # Initialize with reasonable outer radius estimate
        outer_radius = calculate_outer_hexagon_radius(positions, angles)
        
        # Flatten into parameter array
        params = []
        for pos, angle in zip(positions, angles):
            params.extend(pos + [angle])
        params.append(outer_radius)
        
        return np.array(params)

    def objective_function(params: np.ndarray) -> float:
        """
        Objective function to minimize: negative of 1/outer_hex_radius (maximize 1/outer_hex_radius)
        params: [x1, y1, angle1, x2, y2, angle2, ..., x11, y11, angle11, outer_radius]
        """
        # Extract parameters
        n = 11
        inner_params = params[:3*n].reshape(n, 3)
        outer_radius = params[3*n]
        
        # Extract positions and angles
        positions = inner_params[:, :2]
        angles = inner_params[:, 2]
        
        # Early exit if time limit exceeded
        if time.time() - start_time > time_limit:
            return 1e10
        
        # Check containment constraints
        # Assume outer hexagon is centered at origin
        for i in range(n):
            if not check_containment(positions[i], angles[i], (0, 0), outer_radius):
                return 1e10  # Large penalty for violation
            
        # Check overlap constraints - only check if not already violated
        total_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                # Check if hexagons overlap
                if check_overlap(positions[i], angles[i], positions[j], angles[j]):
                    # Calculate overlap area for penalty
                    h1 = create_hexagon_polygon(positions[i][0], positions[i][1], angles[i])
                    h2 = create_hexagon_polygon(positions[j][0], positions[j][1], angles[j])
                    intersection = h1.intersection(h2)
                    overlap_area = intersection.area
                    # Penalty based on overlap area
                    total_penalty += overlap_area * 10000.0
                    
                    # Early exit if time limit exceeded
                    if time.time() - start_time > time_limit:
                        return 1e10
        
        # Return negative inverse of outer radius plus penalties
        # We want to minimize this, so we maximize 1/outer_radius
        return -1.0 / outer_radius + total_penalty

    def evaluate_and_improve(initial_params: np.ndarray) -> tuple:
        """Run a focused improvement process on the initial solution."""
        best_params = initial_params.copy()
        best_objective = objective_function(best_params)
        
        # Simple local search with controlled iterations
        max_iterations = 5000
        iterations = 0
        
        while iterations < max_iterations and (time.time() - start_time < time_limit):
            # Make small random changes to parameters
            test_params = best_params.copy()
            
            # Randomly select a parameter to perturb
            param_idx = random.randint(0, len(test_params) - 1)
            
            # Perturb selected parameter
            if param_idx < 33:  # Inner hexagon parameters
                if param_idx % 3 == 2:  # Angle parameter
                    test_params[param_idx] += random.uniform(-15, 15)  # Angle change
                    # Keep angle in valid range
                    test_params[param_idx] = test_params[param_idx] % 360
                else:  # Position parameter
                    test_params[param_idx] += random.uniform(-0.3, 0.3)  # Position change
            else:  # Outer radius parameter
                test_params[param_idx] += random.uniform(-0.3, 0.3)  # Radius change
            
            # Ensure bounds
            n = 11
            # Position bounds
            for i in range(n):
                test_params[3*i] = np.clip(test_params[3*i], -10, 10)     # x
                test_params[3*i+1] = np.clip(test_params[3*i+1], -10, 10) # y
                test_params[3*i+2] = np.clip(test_params[3*i+2], -180, 180) # angle
            # Outer radius bounds
            test_params[33] = np.clip(test_params[33], 1.0, 20.0)
            
            # Evaluate objective
            test_obj = objective_function(test_params)
            
            # Accept if better or occasionally accept worse solutions for escape
            if test_obj < best_objective or random.random() < 0.01:
                best_params = test_params.copy()
                best_objective = test_obj
                
            iterations += 1
            
            # Check time limit periodically
            if iterations % 100 == 0 and time.time() - start_time > time_limit:
                break
        
        return best_params, best_objective

    # Generate initial solution
    initial_params = generate_initial_solution()
    
    # Improve the initial solution
    best_params, best_objective = evaluate_and_improve(initial_params)
    
    # Try a few more restarts for better results
    for restart in range(3):
        if time.time() - start_time > time_limit:
            break
            
        # Start with perturbed version of best solution
        current_params = best_params + np.random.normal(0, 0.1, len(best_params))
        
        # Apply bounds
        n = 11
        for i in range(n):
            current_params[3*i] = np.clip(current_params[3*i], -10, 10)     # x
            current_params[3*i+1] = np.clip(current_params[3*i+1], -10, 10) # y
            current_params[3*i+2] = np.clip(current_params[3*i+2], -180, 180) # angle
        current_params[33] = np.clip(current_params[33], 1.0, 20.0)
        
        # Improve this restart
        current_params, current_obj = evaluate_and_improve(current_params)
        
        if current_obj < best_objective:
            best_objective = current_obj
            best_params = current_params.copy()

    # Extract results
    inner_params = best_params[:33].reshape(11, 3)
    outer_radius = best_params[33]
    
    # Convert to proper format
    inner_hex_data = inner_params.copy()
    outer_hex_data = np.array([0, 0, 0])  # Centered at origin
    
    return inner_hex_data, outer_hex_data, outer_radius


# EVOLVE-BLOCK-END
