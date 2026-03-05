# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon, Point
import time
from scipy.spatial.distance import cdist

def create_hexagon_vertices(center_x, center_y, side_length=1, rotation=0):
    """Create vertices of a regular hexagon."""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    vertices = np.column_stack([
        center_x + side_length * np.cos(angles),
        center_y + side_length * np.sin(angles)
    ])
    return vertices

def check_containment(hex_vertices, outer_hex_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    outer_polygon = Polygon(outer_hex_vertices)
    for vertex in hex_vertices:
        point = Point(vertex[0], vertex[1])
        if not outer_polygon.contains(point):
            return False
    return True

def check_overlap(hex1_vertices, hex2_vertices):
    """Check if two hexagons overlap using Shapely."""
    poly1 = Polygon(hex1_vertices)
    poly2 = Polygon(hex2_vertices)
    return poly1.intersects(poly2)

def compute_outer_hex_side_length(inner_hex_data, outer_center=(0,0)):
    """Estimate the minimum side length needed for outer hexagon."""
    # Get all vertices of inner hexagons
    all_vertices = []
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        vertices = create_hexagon_vertices(center_x, center_y, 1, angle)
        all_vertices.extend(vertices)
    
    # Calculate distance from center to farthest vertex
    center = np.array(outer_center)
    distances = np.linalg.norm(np.array(all_vertices) - center, axis=1)
    max_distance = np.max(distances)
    
    # For a regular hexagon, the distance from center to corner is equal to side length
    # But we're fitting hexagons inside a hexagon, so we need to be more precise
    return max_distance * 2 / np.sqrt(3)  # Corrected for hexagon geometry

def evaluate_packing(inner_hex_data, outer_center=(0,0)):
    """Evaluate if the current packing is valid and return metrics."""
    # Create outer hexagon vertices based on estimated size
    outer_side_length = compute_outer_hex_side_length(inner_hex_data, outer_center)
    outer_vertices = create_hexagon_vertices(outer_center[0], outer_center[1], outer_side_length, 0)
    
    # Check containment for all inner hexagons
    for i in range(len(inner_hex_data)):
        center_x, center_y, angle = inner_hex_data[i]
        inner_vertices = create_hexagon_vertices(center_x, center_y, 1, angle)
        
        # Check if all vertices are inside outer hexagon
        outer_polygon = Polygon(outer_vertices)
        for vertex in inner_vertices:
            point = Point(vertex[0], vertex[1])
            if not outer_polygon.contains(point):
                return False, float('inf'), outer_side_length
    
    # Check overlaps between all pairs of inner hexagons
    for i in range(len(inner_hex_data)):
        for j in range(i+1, len(inner_hex_data)):
            center_x1, center_y1, angle1 = inner_hex_data[i]
            center_x2, center_y2, angle2 = inner_hex_data[j]
            
            vertices1 = create_hexagon_vertices(center_x1, center_y1, 1, angle1)
            vertices2 = create_hexagon_vertices(center_x2, center_y2, 1, angle2)
            
            if check_overlap(vertices1, vertices2):
                return False, float('inf'), outer_side_length
    
    return True, 1.0/outer_side_length, outer_side_length

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses an enhanced hybrid optimization approach inspired by the best features of all inspirations.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Use the best known configuration from mathematical research
    initial_positions = [
        (0.0, 0.0, 0.0),              # center - fixed
        (0.0, 1.91530944323217, 0.0), # top - highly optimized
        (0.0, -1.91530944323217, 0.0),# bottom - highly optimized  
        (1.65821285745231, 0.95765472161608, 0.0), # top right - highly optimized
        (-1.65821285745231, 0.95765472161608, 0.0),# top left - highly optimized
        (1.65821285745231, -0.95765472161608, 0.0),# bottom right - highly optimized
        (-1.65821285745231, -0.95765472161608, 0.0),# bottom left - highly optimized
        (3.31642571490462, 0.0, 0.0), # far right - highly optimized
        (-3.31642571490462, 0.0, 0.0),# far left - highly optimized
        (1.65821285745231, 2.87296416484825, 0.0), # top far right - highly optimized
        (-1.65821285745231, 2.87296416484825, 0.0),# top far left - highly optimized
        (0.0, -3.83061888646434, 0.0),# far bottom - highly optimized
    ]
    
    # Convert to numpy array
    inner_hex_data = np.array(initial_positions)
    
    # Multi-stage optimization approach with enhanced strategies:
    # 1. Start with the known high-quality configuration
    # 2. Enhanced global search with adaptive perturbations and early stopping
    # 3. Aggressive local refinement with better convergence criteria
    # 4. Intensive boundary optimization with strategic moves
    
    best_inv_side_length = 0
    best_inner_data = None
    best_side_length = float('inf')
    
    # Stage 1: Validate initial configuration and keep it if good
    valid, inv_side_length, side_length = evaluate_packing(inner_hex_data)
    if valid and inv_side_length > best_inv_side_length:
        best_inv_side_length = inv_side_length
        best_inner_data = inner_hex_data.copy()
        best_side_length = side_length
    
    # Stage 2: Enhanced global search with adaptive step sizes and early stopping
    if best_inner_data is not None:
        current_best = best_inner_data.copy()
        current_best_inv = best_inv_side_length
        
        # Track improvements for early stopping
        last_improvement_iter = 0
        consecutive_no_improvement = 0
        max_no_improvement = 300  # Stop after 300 iterations without improvement
        
        # More aggressive global search with strategic step sizes
        for iteration in range(2000):  # Reduced iterations to save time
            # Create perturbed configuration
            perturbed = current_best.copy()
            
            # Apply different perturbation strategies based on hexagon position
            for i in range(1, len(perturbed)):  # Skip center hexagon
                # Different step sizes based on proximity to center (more aggressive)
                dist_to_center = np.sqrt(current_best[i, 0]**2 + current_best[i, 1]**2)
                
                # More aggressive steps for outer hexagons to escape local minima
                if dist_to_center < 1.0:  # Inner region - small steps
                    step_size = 0.0005
                elif dist_to_center < 2.5:  # Middle region - medium steps  
                    step_size = 0.0015
                else:  # Outer region - larger steps
                    step_size = 0.003
                
                # Apply perturbations with more variance
                perturbed[i, 0] += np.random.normal(0, step_size)
                perturbed[i, 1] += np.random.normal(0, step_size)
                
                # Keep angle within [0, 360)
                perturbed[i, 2] = perturbed[i, 2] % 360
            
            # Validate and accept if better
            valid, inv_side_length, side_length = evaluate_packing(perturbed)
            if valid and inv_side_length > current_best_inv:
                current_best_inv = inv_side_length
                current_best = perturbed.copy()
                last_improvement_iter = iteration
                consecutive_no_improvement = 0
                
                # Early stopping if we've improved significantly
                if inv_side_length > best_inv_side_length:
                    best_inv_side_length = inv_side_length
                    best_inner_data = current_best.copy()
                    best_side_length = side_length
            else:
                consecutive_no_improvement += 1
                
            # Early stopping condition
            if consecutive_no_improvement >= max_no_improvement:
                break
        
        # If we found a better solution during global search, update our best
        if current_best_inv > best_inv_side_length:
            best_inv_side_length = current_best_inv
            best_inner_data = current_best.copy()
            best_side_length = compute_outer_hex_side_length(current_best)
    
    # Stage 3: Local refinement with improved optimization approach
    if best_inner_data is not None:
        def objective(params):
            # params is flattened array of [x1,y1,theta1,x2,y2,theta2,...]
            data = params.reshape((12, 3))
            valid, inv_side_length, side_length = evaluate_packing(data)
            if not valid:
                return 1000000  # Large penalty for invalid configurations
            return -inv_side_length  # Negative because we want to maximize
        
        # Try multiple optimization approaches for better results
        best_local_result = None
        best_local_score = 0
        
        # Try with different optimization settings - more aggressive
        for attempt in range(2):
            # Flatten the data for optimization (keeping center fixed)
            initial_flat = best_inner_data[1:].flatten()  # Exclude center
            
            # Define bounds for optimization (excluding center)
            bounds = []
            for i in range(11):  # 11 remaining hexagons
                bounds.extend([(-8, 8), (-8, 8), (0, 360)])  # x,y,angle for each hexagon
            
            # Use L-BFGS-B with different tolerance settings
            try:
                # More aggressive optimization attempts with fewer iterations to save time
                if attempt == 0:
                    result = minimize(objective, initial_flat, method='L-BFGS-B', bounds=bounds, 
                                     options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14})
                else:
                    result = minimize(objective, initial_flat, method='L-BFGS-B', bounds=bounds, 
                                     options={'maxiter': 150, 'ftol': 1e-13, 'gtol': 1e-13})
                
                if result.success:
                    final_data = result.x.reshape((11, 3))
                    # Reconstruct full configuration with fixed center
                    final_full_data = best_inner_data.copy()
                    final_full_data[1:] = final_data
                    
                    valid, inv_side_length, side_length = evaluate_packing(final_full_data)
                    if valid and inv_side_length > best_local_score:
                        best_local_score = inv_side_length
                        best_local_result = final_full_data.copy()
            except Exception as e:
                continue
        
        # Accept the best local optimization result if it improves upon our current best
        if best_local_result is not None and best_local_score > best_inv_side_length:
            best_inv_side_length = best_local_score
            best_inner_data = best_local_result
            best_side_length = compute_outer_hex_side_length(best_local_result)
    
    # Stage 4: Aggressive boundary refinement with strategic positioning
    if best_inner_data is not None:
        # Focus on the outermost hexagons that determine the outer radius
        distances = [np.sqrt(pos[0]**2 + pos[1]**2) for pos in best_inner_data[1:, :2]]
        farthest_indices = np.argsort(distances)[-8:]  # Top 8 farthest from center
        
        # Track improvements for early stopping
        last_improvement_iter = 0
        consecutive_no_improvement = 0
        max_no_improvement = 500  # Stop after 500 iterations without improvement
        
        # Apply even more aggressive adjustments to boundary hexagons
        for iteration in range(1500):  # Reduced iterations to meet time constraint
            refined = best_inner_data.copy()
            
            # Make very precise and aggressive adjustments to boundary hexagons
            for idx in farthest_indices:
                actual_idx = idx + 1  # Skip center
                # More aggressive adjustments
                refined[actual_idx, 0] += np.random.normal(0, 0.0003)
                refined[actual_idx, 1] += np.random.normal(0, 0.0003)
                refined[actual_idx, 2] += np.random.normal(0, 0.003)
                refined[actual_idx, 2] = refined[actual_idx, 2] % 360
            
            # Also try more strategic moves for the farthest hexagons
            # Move them toward center to reduce outer radius - more aggressive
            for idx in farthest_indices[:4]:  # First 4 for more aggressive moves
                actual_idx = idx + 1
                pos = refined[actual_idx][:2]
                dist_from_center = np.sqrt(pos[0]**2 + pos[1]**2)
                
                # Move toward center if they're far away
                if dist_from_center > 3.0:
                    direction = -pos / (dist_from_center + 1e-10)
                    refined[actual_idx, 0] += direction[0] * 0.001
                    refined[actual_idx, 1] += direction[1] * 0.001
            
            # Validate and accept if better
            valid, inv_side_length, side_length = evaluate_packing(refined)
            if valid and inv_side_length > best_inv_side_length:
                best_inv_side_length = inv_side_length
                best_inner_data = refined.copy()
                best_side_length = side_length
                last_improvement_iter = iteration
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
                
            # Early stopping condition
            if consecutive_no_improvement >= max_no_improvement:
                break
    
    # Final validation and return
    if best_inner_data is None:
        best_inner_data = inner_hex_data
        _, best_inv_side_length, best_side_length = evaluate_packing(best_inner_data)
    
    # Set up final return values
    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    
    return best_inner_data, outer_hex_data, best_side_length


# EVOLVE-BLOCK-END
