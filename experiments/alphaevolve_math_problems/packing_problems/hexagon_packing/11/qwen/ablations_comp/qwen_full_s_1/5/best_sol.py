# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon
import math
import random
import time

def get_hexagon_vertices(center, side_length, rotation=0):
    """Get vertices of a regular hexagon"""
    angles = np.linspace(0, 2*np.pi, 7) + np.radians(rotation)
    return [(center[0] + side_length * np.cos(angle), 
             center[1] + side_length * np.sin(angle)) 
            for angle in angles]

def check_hexagon_containment(hexagon, outer_hexagon):
    """Check if hexagon is fully contained within outer hexagon"""
    try:
        hex_poly = Polygon(hexagon)
        outer_poly = Polygon(outer_hexagon)
        return outer_poly.contains(hex_poly)
    except:
        return False

def check_hexagon_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    try:
        poly1 = Polygon(hex1)
        poly2 = Polygon(hex2)
        return poly1.intersects(poly2)
    except:
        return True

def evaluate_packing(inner_hex_data, outer_hex_side_length):
    """Evaluate if a packing is valid and return penalty if invalid"""
    # Create outer hexagon
    outer_center = (0, 0)
    outer_hex = get_hexagon_vertices(outer_center, outer_hex_side_length, 0)
    
    # Check if all inner hexagons are contained and non-overlapping
    total_penalty = 0
    
    inner_hexagons = []
    for i in range(len(inner_hex_data)):
        center = (inner_hex_data[i][0], inner_hex_data[i][1])
        rotation = inner_hex_data[i][2]
        side_length = 1.0  # unit hexagon
        hexagon = get_hexagon_vertices(center, side_length, rotation)
        inner_hexagons.append(hexagon)
        
        # Check containment
        if not check_hexagon_containment(hexagon, outer_hex):
            total_penalty += 1000000  # Large penalty for containment violation
            
        # Check overlaps with other hexagons
        for j in range(i):
            if check_hexagon_overlap(hexagon, inner_hexagons[j]):
                total_penalty += 1000000  # Large penalty for overlap
    
    return total_penalty

def binary_search_min_outer_size(inner_solution, max_size=15.0):
    """Binary search to find minimum outer hexagon size that contains all hexagons"""
    low = 1.0
    high = max_size
    best_size = max_size
    
    # Binary search with high precision
    for _ in range(200):  # Even more iterations for maximum precision
        mid = (low + high) / 2
        penalty = evaluate_packing(inner_solution, mid)
        if penalty == 0:
            best_size = mid
            high = mid
        else:
            low = mid
            
    return best_size

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Uses systematic configuration testing with binary search optimization - the most reliable approach for this problem.
    """
    start_time = time.time()
    
    # Define the most carefully optimized configurations based on mathematical analysis
    # These configurations are specifically designed to minimize outer hexagon size
    configurations = [
        # Configuration 1: Optimized hexagonal close packing (from inspiration 3)
        [(0, 0), (0, 2.0), (1.732, 1.0), (1.732, -1.0), (0, -2.0), (-1.732, -1.0), (-1.732, 1.0),
         (3.464, 0), (-3.464, 0), (1.732, 3.0), (-1.732, 3.0)],
        
        # Configuration 2: Slightly adjusted for tighter packing (from inspiration 1)
        [(0, 0), (0, 2.05), (1.732, 1.025), (1.732, -1.025), (0, -2.05), (-1.732, -1.025), (-1.732, 1.025),
         (3.464, 0), (-3.464, 0), (1.732, 3.075), (-1.732, 3.075)],
        
        # Configuration 3: More compact arrangement (from inspiration 2)
        [(0, 0), (0, 1.95), (1.69, 0.975), (1.69, -0.975), (0, -1.95), (-1.69, -0.975), (-1.69, 0.975),
         (3.38, 0), (-3.38, 0), (1.69, 2.925), (-1.69, 2.925)],
        
        # Configuration 4: Extremely optimized spacing (from inspiration 1)
        [(0, 0), (0, 1.98), (1.72, 0.99), (1.72, -0.99), (0, -1.98), (-1.72, -0.99), (-1.72, 0.99),
         (3.44, 0), (-3.44, 0), (1.72, 2.97), (-1.72, 2.97)],
        
        # Configuration 5: Alternative symmetric pattern (from inspiration 3)
        [(0, 0), (0, 1.9), (1.66, 0.95), (1.66, -0.95), (0, -1.9), (-1.66, -0.95), (-1.66, 0.95),
         (3.32, 0), (-3.32, 0), (1.66, 2.85), (-1.66, 2.85)],
        
        # Configuration 6: Very tight arrangement (from inspiration 2)
        [(0, 0), (0, 2.17), (1.732, 1.085), (1.732, -1.085), (0, -2.17), (-1.732, -1.085), (-1.732, 1.085),
         (3.464, 0), (-3.464, 0), (1.732, 3.255), (-1.732, 3.255)],
        
        # Configuration 7: Precise mathematical arrangement (from inspiration 3)
        [(0, 0), (0, 1.8), (1.559, 0.9), (1.559, -0.9), (0, -1.8), (-1.559, -0.9), (-1.559, 0.9),
         (3.118, 0), (-3.118, 0), (1.559, 2.7), (-1.559, 2.7)],
        
        # Configuration 8: Highly optimized symmetric pattern (from inspiration 1)
        [(0, 0), (0, 2.02), (1.732, 1.01), (1.732, -1.01), (0, -2.02), (-1.732, -1.01), (-1.732, 1.01),
         (3.464, 0), (-3.464, 0), (1.732, 3.03), (-1.732, 3.03)],
        
        # Configuration 9: Compact with specific spacing ratios (from inspiration 2)
        [(0, 0), (0, 2.1), (1.732, 1.05), (1.732, -1.05), (0, -2.1), (-1.732, -1.05), (-1.732, 1.05),
         (3.464, 0), (-3.464, 0), (1.732, 3.15), (-1.732, 3.15)],
        
        # Configuration 10: Another optimized configuration (from inspiration 3)
        [(0, 0), (0, 1.99), (1.72, 0.995), (1.72, -0.995), (0, -1.99), (-1.72, -0.995), (-1.72, 0.995),
         (3.44, 0), (-3.44, 0), (1.72, 2.985), (-1.72, 2.985)],
    ]
    
    best_solution = None
    best_side_length = float('inf')
    
    # Test each configuration
    for i, positions in enumerate(configurations):
        # Create solution with no rotation initially
        solution = np.zeros((11, 3))
        for j, (x, y) in enumerate(positions):
            solution[j] = [x, y, 0]
        
        # Binary search for minimal outer hexagon size
        min_size = binary_search_min_outer_size(solution, 15.0)
        penalty = evaluate_packing(solution, min_size)
        
        # If valid solution, check if it's better
        if penalty == 0 and min_size < best_side_length:
            best_side_length = min_size
            best_solution = solution.copy()
            
        # Early exit if we have a very good solution
        if time.time() - start_time > 55:
            break
    
    # If we still don't have a solution, use a fallback
    if best_solution is None:
        # Use a reliable configuration
        positions = [
            (0, 0), (0, 2.0), (1.732, 1.0), (1.732, -1.0), (0, -2.0), (-1.732, -1.0), (-1.732, 1.0),
            (3.464, 0), (-3.464, 0), (1.732, 3.0), (-1.732, 3.0)
        ]
        best_solution = np.zeros((11, 3))
        for j, (x, y) in enumerate(positions):
            best_solution[j] = [x, y, 0]
        best_side_length = binary_search_min_outer_size(best_solution, 15.0)
    
    # Apply a final local refinement to squeeze out any remaining improvement
    # Try a few more precise local adjustments
    if best_solution is not None:
        refined_solution = best_solution.copy()
        best_refined_size = best_side_length
        
        # Perform focused local search around the best solution
        for _ in range(50):
            # Make small adjustments to several hexagons
            hex_idx = random.randint(0, 10)
            adjustment_magnitude = 0.005
            
            # Adjust x and y positions slightly
            refined_solution[hex_idx, 0] += random.uniform(-adjustment_magnitude, adjustment_magnitude)
            refined_solution[hex_idx, 1] += random.uniform(-adjustment_magnitude, adjustment_magnitude)
            
            # Check if this improved the packing
            size_check = binary_search_min_outer_size(refined_solution, 15.0)
            penalty_check = evaluate_packing(refined_solution, size_check)
            
            # If this is better, keep the change
            if penalty_check == 0 and size_check < best_refined_size:
                best_refined_size = size_check
                best_solution = refined_solution.copy()
            else:
                # Revert the change
                refined_solution = best_solution.copy()
                
            # Early exit if time is running out
            if time.time() - start_time > 55:
                break
        
        best_side_length = best_refined_size
    
    # Ensure all rotations are within [0, 360)
    best_solution[:, 2] = np.mod(best_solution[:, 2], 360)
    
    # Return the solution and outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # Outer hexagon centered at origin, no rotation
    
    return best_solution, outer_hex_data, best_side_length


# EVOLVE-BLOCK-END
