# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import time

# Fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

def initialize_circles_hexagonal(n: int) -> np.ndarray:
    """Initialize circles in a hexagonal pattern to get a good starting configuration."""
    circles = np.zeros((n, 3))
    
    # Create a structured initial arrangement with good density
    rows = 5
    cols = 5
    
    # Calculate spacing to fit in the unit square
    padding = 0.05
    spacing_x = (1 - 2*padding) / (cols - 1) if cols > 1 else 0.5
    spacing_y = (1 - 2*padding) / (rows - 1) if rows > 1 else 0.5
    
    # Maximum radius based on spacing
    max_radius = min(spacing_x, spacing_y) * 0.4
    
    # Place circles in a structured grid pattern
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
                
            # Position with some jitter to avoid perfect symmetry
            x = padding + (j * spacing_x) + np.random.normal(0, 0.003)
            y = padding + (i * spacing_y) + np.random.normal(0, 0.003)
            
            # Clamp to valid range
            x = np.clip(x, padding, 1-padding)
            y = np.clip(y, padding, 1-padding)
            
            circles[idx] = [x, y, max_radius]
            idx += 1
    
    # Fill remaining circles with random valid positions using better overlap checking
    for i in range(idx, n):
        attempts = 0
        while attempts < 1000:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            r = np.random.uniform(0.01, 0.15)
            
            # Check containment
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                attempts += 1
                continue
                
            # Check overlap with existing circles efficiently using KDTree
            overlap = False
            if i > 0:
                existing_positions = circles[:i, :2]
                tree = cKDTree(existing_positions)
                # Find neighbors within 2*(r + max_radius) distance
                neighbors = tree.query_ball_point([x, y], 2*(r + max_radius))
                for j in neighbors:
                    cx, cy, cr = circles[j]
                    dist_sq = (x - cx)**2 + (y - cy)**2
                    if dist_sq < (r + cr)**2:
                        overlap = True
                        break
            
            if not overlap:
                circles[i] = [x, y, r]
                break
            attempts += 1
    
    return circles

def calculate_radii_sum(circles: np.ndarray) -> float:
    """Calculate the sum of all circle radii."""
    return np.sum(circles[:, 2])

def compute_overlaps(circles: np.ndarray, tree: cKDTree = None) -> list:
    """Find overlapping pairs using spatial data structure for efficiency."""
    if tree is None:
        centers = circles[:, :2]
        tree = cKDTree(centers)
    
    # Find all pairs within 2*(max_radius) distance
    pairs = tree.query_pairs(2.0 * np.max(circles[:, 2]), output_type='ndarray')
    
    overlaps = []
    for i, j in pairs:
        if i < j:  # Avoid duplicate pairs
            r1, r2 = circles[i, 2], circles[j, 2]
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            distance = np.sqrt(dx*dx + dy*dy)
            if distance < r1 + r2:
                overlaps.append((i, j))
    
    return overlaps

def project_to_feasible(circles: np.ndarray) -> np.ndarray:
    """Project circles to feasible region using a more robust approach."""
    projected = circles.copy()
    
    # Ensure containment constraints
    for i in range(len(projected)):
        x, y, r = projected[i]
        # Clamp radius to stay within bounds
        r = min(r, x, y, 1-x, 1-y)
        # Ensure minimum radius
        r = max(r, 0.001)
        projected[i] = [x, y, r]
    
    # Resolve overlaps more systematically with fewer iterations
    for _ in range(30):  # Reduced iterations for faster execution
        tree = cKDTree(projected[:, :2])
        overlaps = compute_overlaps(projected, tree)
        
        if not overlaps:
            break
            
        # Resolve overlaps by reducing radii and adjusting positions
        for i, j in overlaps:
            r1, r2 = projected[i, 2], projected[j, 2]
            dx = projected[i, 0] - projected[j, 0]
            dy = projected[i, 1] - projected[j, 1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                overlap = (r1 + r2) - distance
                if overlap > 0:
                    # Reduce both radii proportionally
                    reduction = min(overlap * 0.5, r1 * 0.3, r2 * 0.3)
                    if reduction > 0:
                        r1_new = max(0.001, r1 - reduction)
                        r2_new = max(0.001, r2 - reduction)
                        
                        # Adjust positions to move them apart
                        ratio = (r1_new + r2_new) / (r1 + r2)
                        if ratio < 1:
                            x1_new = projected[i, 0] + dx * (1 - ratio) * 0.5
                            y1_new = projected[i, 1] + dy * (1 - ratio) * 0.5
                            x2_new = projected[j, 0] - dx * (1 - ratio) * 0.5
                            y2_new = projected[j, 1] - dy * (1 - ratio) * 0.5
                            
                            # Ensure positions are within bounds
                            x1_new = np.clip(x1_new, r1_new, 1 - r1_new)
                            y1_new = np.clip(y1_new, r1_new, 1 - r1_new)
                            x2_new = np.clip(x2_new, r2_new, 1 - r2_new)
                            y2_new = np.clip(y2_new, r2_new, 1 - r2_new)
                            
                            projected[i] = [x1_new, y1_new, r1_new]
                            projected[j] = [x2_new, y2_new, r2_new]
    
    return projected

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a more aggressive coordinate descent approach with better optimization strategies.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Start with good initialization
    circles = initialize_circles_hexagonal(26)
    circles = project_to_feasible(circles)
    
    # Store best solution
    best_circles = circles.copy()
    best_sum = calculate_radii_sum(best_circles)
    
    print(f"Initial sum of radii: {best_sum:.6f}")
    
    # More aggressive coordinate descent optimization
    max_iterations = 2000  # More iterations for better optimization
    improved = True
    iteration = 0
    
    # Track how many consecutive iterations without improvement
    no_improvement_count = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try to increase radii of individual circles more aggressively
        for i in range(26):
            x, y, r = circles[i]
            
            # Try to increase radius up to maximum allowed
            max_radius = min(x, 1-x, y, 1-y)
            
            # Try larger steps first, then smaller ones for fine-tuning
            step_sizes = [0.005, 0.002, 0.001, 0.0005]
            for step_size in step_sizes:
                if r + step_size > max_radius:
                    continue
                    
                test_circles = circles.copy()
                test_circles[i, 2] = r + step_size
                
                # Check if this would be valid with more thorough checking
                valid = True
                positions = test_circles[:, :2]
                tree = cKDTree(positions)
                neighbors = tree.query_ball_point([x, y], 2*(test_circles[i, 2] + 0.5))
                for j in neighbors:
                    if i != j:
                        dx = test_circles[i, 0] - test_circles[j, 0]
                        dy = test_circles[i, 1] - test_circles[j, 1]
                        dist_sq = dx*dx + dy*dy
                        min_dist_sq = (test_circles[i, 2] + test_circles[j, 2])**2
                        if dist_sq < min_dist_sq:
                            valid = False
                            break
                
                # Also check containment
                if test_circles[i, 0] - test_circles[i, 2] < 0 or \
                   test_circles[i, 0] + test_circles[i, 2] > 1 or \
                   test_circles[i, 1] - test_circles[i, 2] < 0 or \
                   test_circles[i, 1] + test_circles[i, 2] > 1:
                    valid = False
                
                if valid:
                    r += step_size
                    improved = True
                else:
                    break  # Can't increase further with this step size
            
            circles[i, 2] = r
        
        # Position adjustments with more thorough search
        if iteration % 3 == 0 and improved:
            # Try to improve positions with better neighborhood consideration
            for i in range(26):
                x, y, r = circles[i]
                
                # Try systematic position adjustments
                best_x, best_y = x, y
                best_sum_local = calculate_radii_sum(circles)
                best_improved = False
                
                # Test more extensive moves
                moves = [
                    (-0.005, -0.005), (-0.005, 0), (-0.005, 0.005),
                    (0, -0.005), (0, 0), (0, 0.005),
                    (0.005, -0.005), (0.005, 0), (0.005, 0.005),
                    (-0.002, -0.002), (-0.002, 0), (-0.002, 0.002),
                    (0.002, -0.002), (0.002, 0), (0.002, 0.002),
                    (0, -0.002), (0, 0.002)
                ]
                
                for dx, dy in moves:
                    new_x = x + dx
                    new_y = y + dy
                    
                    # Keep within bounds
                    new_x = np.clip(new_x, r, 1-r)
                    new_y = np.clip(new_y, r, 1-r)
                    
                    # Check if this move improves the configuration
                    temp_circles = circles.copy()
                    temp_circles[i, 0] = new_x
                    temp_circles[i, 1] = new_y
                    
                    # More thorough validity check
                    valid = True
                    positions = temp_circles[:, :2]
                    tree = cKDTree(positions)
                    neighbors = tree.query_ball_point([new_x, new_y], 2*(r + 0.5))
                    for j in neighbors:
                        if i != j:
                            dx_check = new_x - temp_circles[j, 0]
                            dy_check = new_y - temp_circles[j, 1]
                            dist_sq = dx_check*dx_check + dy_check*dy_check
                            min_dist_sq = (r + temp_circles[j, 2])**2
                            if dist_sq < min_dist_sq:
                                valid = False
                                break
                    
                    if valid:
                        temp_sum = calculate_radii_sum(temp_circles)
                        if temp_sum > best_sum_local:
                            best_sum_local = temp_sum
                            best_x, best_y = new_x, new_y
                            best_improved = True
                
                # Apply the best move if it improves the solution
                if best_improved:
                    circles[i, 0] = best_x
                    circles[i, 1] = best_y
                    improved = True
        
        # Update best solution if improved
        if improved:
            current_sum = calculate_radii_sum(circles)
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = circles.copy()
                no_improvement_count = 0  # Reset counter when we find an improvement
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1
        
        # Early stopping if no improvement for many iterations
        if no_improvement_count > 100:
            break
        
        # Print progress
        if iteration % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {iteration}: Sum = {current_sum:.6f}, Best = {best_sum:.6f}, Time = {elapsed:.2f}s")
        
        # Early termination if we beat the benchmark
        if best_sum > 2.6358627564136983:
            print("Beat benchmark! Early termination.")
            break
    
    # Final refinement with more aggressive coordinate descent
    print("Starting final refinement...")
    for _ in range(1000):
        improved = False
        
        # Try to improve each circle individually with more extensive search
        for i in range(len(best_circles)):
            original_sum = calculate_radii_sum(best_circles)
            
            # Try more extensive adjustments to position and radius
            for _ in range(20):  # More tries per circle
                test_circles = best_circles.copy()
                
                # Larger random perturbations for more aggressive search
                delta_x = np.random.normal(0, 0.002)  # Increased from 0.0005
                delta_y = np.random.normal(0, 0.002)
                delta_r = np.random.normal(0, 0.001)  # Increased from 0.0001
                
                test_circles[i, 0] += delta_x
                test_circles[i, 1] += delta_y
                test_circles[i, 2] += delta_r
                
                # Ensure within bounds
                test_circles[i, 0] = np.clip(test_circles[i, 0], 
                                           test_circles[i, 2], 1 - test_circles[i, 2])
                test_circles[i, 1] = np.clip(test_circles[i, 1], 
                                           test_circles[i, 2], 1 - test_circles[i, 2])
                test_circles[i, 2] = np.clip(test_circles[i, 2], 0.001, 0.5)
                
                # Project to feasible region
                test_circles = project_to_feasible(test_circles)
                test_sum = calculate_radii_sum(test_circles)
                
                if test_sum > original_sum:
                    best_circles = test_circles.copy()
                    improved = True
                    break
        
        if not improved:
            break
    
    # Final validation
    final_solution = project_to_feasible(best_circles)
    final_sum = calculate_radii_sum(final_solution)
    
    print(f"Final solution found with sum of radii: {final_sum:.6f}")
    print(f"Best sum achieved: {best_sum:.6f}")
    print(f"Time taken: {time.time() - start_time:.2f}s")
    
    return final_solution


# EVOLVE-BLOCK-END
