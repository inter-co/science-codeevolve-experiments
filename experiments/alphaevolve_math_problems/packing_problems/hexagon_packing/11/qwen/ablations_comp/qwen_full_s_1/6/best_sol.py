# EVOLVE-BLOCK-START
import numpy as np
from shapely.geometry import Polygon, Point
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
        # Use covers() for better robustness - more reliable than contains
        return outer_poly.covers(hex_poly)
    except Exception:
        # Fallback to vertex containment check with proper error handling
        try:
            hex_poly = Polygon(hexagon)
            outer_poly = Polygon(outer_hexagon)
            # Check that all vertices of the inner hexagon are inside the outer hexagon
            for point in list(hex_poly.exterior.coords):
                if not outer_poly.contains(Point(point)):
                    return False
            return True
        except Exception:
            return False

def check_hexagon_overlap(hex1, hex2):
    """Check if two hexagons overlap"""
    try:
        poly1 = Polygon(hex1)
        poly2 = Polygon(hex2)
        # Check if they intersect and are not just touching
        return poly1.intersects(poly2) and not poly1.touches(poly2)
    except Exception:
        # Fallback: check if any vertex of one polygon is inside the other
        try:
            poly1 = Polygon(hex1)
            poly2 = Polygon(hex2)
            # Check if any vertex of poly1 is inside poly2
            for point in list(poly1.exterior.coords):
                if poly2.contains(Point(point)):
                    return True
            # Check if any vertex of poly2 is inside poly1
            for point in list(poly2.exterior.coords):
                if poly1.contains(Point(point)):
                    return True
            return False
        except Exception:
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

def binary_search_min_outer_size(inner_solution, max_size=15.0, tolerance=1e-6):
    """Binary search to find minimum outer hexagon size that contains all hexagons"""
    low = 1.0
    high = max_size
    best_size = max_size
    
    # Binary search with reduced iterations for better performance
    # Use fewer iterations since we're looking for a very precise answer
    max_iterations = 60  # Reduced from previous value
    
    for _ in range(max_iterations):
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
    # Using the most promising configurations from the inspiration programs, plus some additional ones
    configurations = [
        # Configuration 1: From inspiration 3 - precise mathematical arrangement (very close to optimal)
        [(0, 0), (0, 1.930092), (1.669900, 0.965046), (1.669900, -0.965046), (0, -1.930092), 
         (-1.669900, -0.965046), (-1.669900, 0.965046), (0, 3.860184), (0, -3.860184), 
         (3.860184, 0), (-3.860184, 0)],
        
        # Configuration 2: From inspiration 1 - optimized spacing
        [(0, 0), (0, 1.98), (1.72, 0.99), (1.72, -0.99), (0, -1.98), (-1.72, -0.99), (-1.72, 0.99),
         (3.44, 0), (-3.44, 0), (1.72, 2.97), (-1.72, 2.97)],
        
        # Configuration 3: From inspiration 2 - tight arrangement
        [(0, 0), (0, 1.95), (1.69, 0.975), (1.69, -0.975), (0, -1.95), (-1.69, -0.975), (-1.69, 0.975),
         (3.38, 0), (-3.38, 0), (1.69, 2.925), (-1.69, 2.925)],
        
        # Configuration 4: From inspiration 3 - compact hexagonal packing
        [(0, 0), (0, 2.0), (1.732, 1.0), (1.732, -1.0), (0, -2.0), (-1.732, -1.0), (-1.732, 1.0),
         (3.464, 0), (-3.464, 0), (1.732, 3.0), (-1.732, 3.0)],
        
        # Configuration 5: From inspiration 2 - highly optimized
        [(0, 0), (0, 2.17), (1.732, 1.085), (1.732, -1.085), (0, -2.17), (-1.732, -1.085), (-1.732, 1.085),
         (3.464, 0), (-3.464, 0), (1.732, 3.255), (-1.732, 3.255)],
        
        # Configuration 6: From inspiration 1 - very tight packing
        [(0, 0), (0, 1.925), (1.667, 0.9625), (1.667, -0.9625), (0, -1.925), 
         (-1.667, -0.9625), (-1.667, 0.9625), (0, 3.85), (0, -3.85), 
         (3.85, 0), (-3.85, 0)],
        
        # Configuration 7: From inspiration 3 - precise mathematical ratios
        [(0, 0), (0, 1.8), (1.559, 0.9), (1.559, -0.9), (0, -1.8), (-1.559, -0.9), (-1.559, 0.9),
         (3.118, 0), (-3.118, 0), (1.559, 2.7), (-1.559, 2.7)],
        
        # Configuration 8: From inspiration 2 - alternative arrangement
        [(0, 0), (0, 2.02), (1.732, 1.01), (1.732, -1.01), (0, -2.02), (-1.732, -1.01), (-1.732, 1.01),
         (3.464, 0), (-3.464, 0), (1.732, 3.03), (-1.732, 3.03)],
        
        # Configuration 9: From inspiration 3 - optimized symmetric pattern
        [(0, 0), (0, 1.9), (1.66, 0.95), (1.66, -0.95), (0, -1.9), (-1.66, -0.95), (-1.66, 0.95),
         (3.32, 0), (-3.32, 0), (1.66, 2.85), (-1.66, 2.85)],
        
        # Configuration 10: From inspiration 2 - compact with specific ratios
        [(0, 0), (0, 2.1), (1.732, 1.05), (1.732, -1.05), (0, -2.1), (-1.732, -1.05), (-1.732, 1.05),
         (3.464, 0), (-3.464, 0), (1.732, 3.15), (-1.732, 3.15)],
        
        # Configuration 11: Additional configuration from research - very tight packing
        [(0, 0), (0, 1.93), (1.669, 0.965), (1.669, -0.965), (0, -1.93), (-1.669, -0.965), (-1.669, 0.965),
         (0, 3.86), (0, -3.86), (3.86, 0), (-3.86, 0)],
        
        # Configuration 12: From inspiration - radial arrangement
        [(0, 0), (0, 2.0), (1.732, 1.0), (1.732, -1.0), (0, -2.0), (-1.732, -1.0), (-1.732, 1.0),
         (3.464, 0), (-3.464, 0), (1.732, 3.0), (-1.732, 3.0)],
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
    # Use a more targeted approach with better convergence properties
    if best_solution is not None:
        refined_solution = best_solution.copy()
        best_refined_size = best_side_length
        
        # Try several rounds of more sophisticated refinement
        for round_num in range(3):
            # For each round, make multiple small adjustments
            for _ in range(30):  # Increase iterations for better refinement
                # Make small adjustments to a few hexagons
                hex_indices = random.sample(range(11), min(4, 11))  # Sample up to 4 hexagons
                
                for hex_idx in hex_indices:
                    # Small adjustments to positions
                    adjustment_magnitude = 0.002 * (3 - round_num)  # Decrease magnitude with rounds
                    
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
            
            if time.time() - start_time > 55:
                break
        
        best_side_length = best_refined_size
    
    # Ensure all rotations are within [0, 360)
    best_solution[:, 2] = np.mod(best_solution[:, 2], 360)
    
    # Return the solution and outer hexagon data
    outer_hex_data = np.array([0, 0, 0])  # Outer hexagon centered at origin, no rotation
    
    return best_solution, outer_hex_data, best_side_length


# EVOLVE-BLOCK-END
