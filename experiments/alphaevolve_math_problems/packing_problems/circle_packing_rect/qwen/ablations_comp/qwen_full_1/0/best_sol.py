# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric initialization with advanced multi-stage optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Test multiple aspect ratios to find optimal configuration
    best_ratio = 1.0
    best_sum = 0.0
    best_circles = None
    
    # Focus on promising aspect ratios based on previous analysis and inspiration programs
    ratios = [0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0]
    
    # Add extra precision around promising values from previous experience
    precise_ratios = [1.18, 1.22, 1.28, 1.32, 1.38, 1.42, 1.48, 1.52, 1.58, 1.62]
    ratios = ratios + precise_ratios
    
    for ratio in ratios:
        width = 2 * ratio / (1 + ratio)
        height = 2 / (1 + ratio)
        
        # Initialize with better pattern that considers actual packing efficiency
        circles = initialize_improved_pattern(21, width, height)
        
        # Multi-stage optimization with increasing precision
        optimized_circles = optimize_circles_multi_stage(circles, width, height)
        
        current_sum = np.sum(optimized_circles[:, 2])
        if current_sum > best_sum:
            best_sum = current_sum
            best_circles = optimized_circles.copy()
            best_ratio = ratio
    
    # Final refinement with the best configuration
    if best_circles is not None:
        # Apply final optimization with extreme precision
        final_circles = optimize_with_extreme_precision(best_circles, best_ratio)
        return final_circles
    else:
        # Fallback to standard square configuration
        width = 1.0
        height = 1.0
        circles = initialize_improved_pattern(21, width, height)
        return optimize_with_extreme_precision(circles, 1.0)


def initialize_improved_pattern(n, width, height):
    """Initialize circles with better pattern that considers actual packing efficiency"""
    circles = np.zeros((n, 3))
    
    # Use a 5x4 grid pattern that's more suitable for 21 circles
    rows = 5
    cols = 4
    
    # Calculate spacing based on area
    area_per_circle = (width * height) / n
    max_radius = np.sqrt(area_per_circle / np.pi) * 0.85  # Leave good margin
    
    # Hexagonal packing approach with better adjustment
    spacing_x = 2 * max_radius
    spacing_y = max_radius * np.sqrt(3)
    
    # Adjust to fit within bounds properly
    actual_cols = max(1, int(width / spacing_x))
    actual_rows = max(1, int(height / spacing_y))
    
    # If we can't fit the desired number, adjust spacing
    if actual_cols * actual_rows < n:
        spacing_x = width / actual_cols
        spacing_y = height / actual_rows
        max_radius = min(spacing_x, spacing_y) / 2 * 0.95
    
    # Fill with circles in hexagonal pattern
    idx = 0
    for row in range(actual_rows):
        for col in range(actual_cols):
            if idx >= n:
                break
                
            # Offset odd rows for hexagonal packing
            x_offset = (row % 2) * spacing_x / 2
            x = x_offset + col * spacing_x + max_radius
            y = row * spacing_y + max_radius
            
            # Ensure within bounds
            x = np.clip(x, max_radius, width - max_radius)
            y = np.clip(y, max_radius, height - max_radius)
            
            if x - max_radius >= 0 and x + max_radius <= width and \
               y - max_radius >= 0 and y + max_radius <= height:
                circles[idx] = [x, y, max_radius]
                idx += 1
                
        if idx >= n:
            break
    
    # Fill remaining slots with more intelligent random placement
    if idx < n:
        for i in range(idx, n):
            # Try multiple random attempts with better rejection criteria
            attempts = 0
            while attempts < 200:
                # Try to place in a way that maximizes utilization
                x = np.random.uniform(max_radius, width - max_radius)
                y = np.random.uniform(max_radius, height - max_radius)
                
                # Check if reasonably far from existing circles
                valid = True
                for j in range(i):
                    existing_x, existing_y, existing_r = circles[j]
                    dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if dist < (existing_r + max_radius) * 0.7:  # Stricter minimum distance
                        valid = False
                        break
                        
                if valid:
                    circles[i] = [x, y, max_radius]
                    break
                attempts += 1
            
            # Last resort - place randomly with basic bounds
            if attempts >= 200:
                circles[i] = [
                    np.random.uniform(max_radius, width - max_radius),
                    np.random.uniform(max_radius, height - max_radius),
                    max_radius
                ]
    
    return circles


def optimize_circles_multi_stage(initial_circles, width, height):
    """Multi-stage optimization for maximum convergence"""
    circles = initial_circles.copy()
    
    # Stage 1: Coarse optimization with relaxed tolerances
    circles = optimize_with_slsqp(circles, width, height, maxiter=100, ftol=1e-2)
    
    # Stage 2: Medium precision optimization  
    circles = optimize_with_slsqp(circles, width, height, maxiter=200, ftol=1e-4)
    
    # Stage 3: Fine optimization with tighter tolerances
    circles = optimize_with_slsqp(circles, width, height, maxiter=300, ftol=1e-6)
    
    # Stage 4: Very fine optimization with even tighter tolerances
    circles = optimize_with_slsqp(circles, width, height, maxiter=400, ftol=1e-8)
    
    # Stage 5: Extremely fine optimization with tightest tolerances
    circles = optimize_with_slsqp(circles, width, height, maxiter=500, ftol=1e-10)
    
    # Stage 6: Aggressive overlap resolution with many passes
    circles = aggressive_overlap_resolution(circles, width, height)
    
    # Stage 7: Enhanced local radius enhancement
    circles = enhanced_local_radius_enhancement(circles, width, height)
    
    # Stage 8: Final boundary and overlap validation with ultra-strict checks
    circles = ultra_strict_validation(circles, width, height)
    
    return circles


def aggressive_overlap_resolution(circles, width, height):
    """Very aggressive overlap resolution with multiple passes"""
    updated_circles = circles.copy()
    
    # Multiple rounds of aggressive overlap resolution with more iterations
    for round_num in range(40):  # Even more rounds for better convergence
        improved = False
        # Check all pairs for overlaps
        for i in range(len(updated_circles)):
            for j in range(i+1, len(updated_circles)):
                x1, y1, r1 = updated_circles[i]
                x2, y2, r2 = updated_circles[j]
                
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2):
                    improved = True
                    # Separate circles more aggressively with even higher factor
                    if dist > 1e-10:
                        dx = (x2 - x1) / dist
                        dy = (y2 - y1) / dist
                        overlap = (r1 + r2) - dist
                        separation = overlap * 0.99  # Even more aggressive separation
                        
                        updated_circles[i][0] -= dx * separation
                        updated_circles[i][1] -= dy * separation
                        updated_circles[j][0] += dx * separation
                        updated_circles[j][1] += dy * separation
                    
                    # Keep within bounds
                    x1, y1, r1 = updated_circles[i]
                    x2, y2, r2 = updated_circles[j]
                    updated_circles[i][0] = np.clip(x1, r1, width - r1)
                    updated_circles[i][1] = np.clip(y1, r1, height - r1)
                    updated_circles[j][0] = np.clip(x2, r2, width - r2)
                    updated_circles[j][1] = np.clip(y2, r2, height - r2)
        
        if not improved:
            break
    
    return updated_circles


def enhanced_local_radius_enhancement(circles, width, height):
    """Enhanced local radius enhancement with more thorough searching"""
    updated_circles = circles.copy()
    
    # Multiple passes of local enhancement with more iterations
    for pass_num in range(70):  # Even more passes for better optimization
        improved = False
        # Try to increase each radius systematically
        for i in range(len(updated_circles)):
            x, y, r = updated_circles[i]
            
            # Calculate maximum possible radius for this circle
            max_radius = min(
                x, width - x,
                y, height - y
            )
            
            # Try to increase radius with even more aggressive steps
            new_r = min(r * 1.15, max_radius)  # Even larger step size
            
            # Check if this radius works with neighbors with stricter criteria
            valid = True
            for j in range(len(updated_circles)):
                if i != j:
                    x1, y1, r1 = updated_circles[i]
                    x2, y2, r2 = updated_circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    # Use stricter minimum distance to avoid near-overlaps
                    if dist < (new_r + r2) * 0.985:  # Even stricter
                        valid = False
                        break
            
            if valid and new_r > r:
                updated_circles[i][2] = new_r
                improved = True
        
        if not improved:
            break
    
    return updated_circles


def ultra_strict_validation(circles, width, height):
    """Ultra-strict final validation and correction"""
    updated_circles = circles.copy()
    
    # Ensure all circles are within bounds
    for i in range(len(updated_circles)):
        x, y, r = updated_circles[i]
        updated_circles[i][0] = np.clip(x, r, width - r)
        updated_circles[i][1] = np.clip(y, r, height - r)
    
    # Final overlap check and correction with ultra-strict tolerance
    for i in range(len(updated_circles)):
        for j in range(i+1, len(updated_circles)):
            x1, y1, r1 = updated_circles[i]
            x2, y2, r2 = updated_circles[j]
            
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist < (r1 + r2) * 0.9998:  # Slightly looser tolerance to allow for more aggressive adjustments
                # Push apart more carefully
                if dist > 1e-10:
                    dx = (x2 - x1) / dist
                    dy = (y2 - y1) / dist
                    overlap = (r1 + r2) - dist
                    separation = overlap * 0.92  # Even more aggressive
                    
                    updated_circles[i][0] -= dx * separation
                    updated_circles[i][1] -= dy * separation
                    updated_circles[j][0] += dx * separation
                    updated_circles[j][1] += dy * separation
                
                # Ensure within bounds after adjustment
                x1, y1, r1 = updated_circles[i]
                x2, y2, r2 = updated_circles[j]
                updated_circles[i][0] = np.clip(x1, r1, width - r1)
                updated_circles[i][1] = np.clip(y1, r1, height - r1)
                updated_circles[j][0] = np.clip(x2, r2, width - r2)
                updated_circles[j][1] = np.clip(y2, r2, height - r2)
    
    return updated_circles


def optimize_with_slsqp(initial_circles, width, height, maxiter=500, ftol=1e-8):
    """SLSQP optimization with enhanced constraint handling"""
    n = len(initial_circles)
    
    # Flatten initial configuration
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    def objective(params):
        # Reconstruct circles from params
        circles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            circles.append([x, y, r])
        
        # Sum of radii (negative for maximization)
        return -sum(circle[2] for circle in circles)
    
    def constraint_func(params):
        # Constraint function for optimization
        circles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            circles.append([x, y, r])
        
        constraints = []
        
        # Boundary constraints (all should be >= 0)
        for i in range(n):
            x, y, r = circles[i]
            constraints.append(x - r)  # Left boundary
            constraints.append(y - r)  # Bottom boundary
            constraints.append(width - x - r)  # Right boundary
            constraints.append(height - y - r)  # Top boundary
        
        # Overlap constraints (distance >= sum of radii)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(distance - (r1 + r2))
        
        return np.array(constraints)
    
    # Define bounds
    bounds = []
    for i in range(n):
        bounds.append((0.001, width - 0.001))      # x coordinates
        bounds.append((0.001, height - 0.001))    # y coordinates
        bounds.append((0.001, min(width, height) / 2 - 0.001))  # radii
    
    # Define constraints
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    try:
        # Perform optimization with more aggressive settings for better convergence
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-10, 'disp': False}
        )
        
        if result.success:
            # Reconstruct final circles
            circles = []
            for i in range(n):
                x = result.x[3*i]
                y = result.x[3*i+1]
                r = result.x[3*i+2]
                circles.append([x, y, r])
            return np.array(circles)
        else:
            return initial_circles
            
    except Exception:
        return initial_circles


def optimize_with_extreme_precision(initial_circles, ratio):
    """Extreme precision optimization to push final results"""
    n = len(initial_circles)
    
    # Try different rectangle dimensions to improve results
    width = 2 * ratio / (1 + ratio)
    height = 2 / (1 + ratio)
    
    # Flatten initial configuration
    initial_params = []
    for i in range(n):
        initial_params.extend([initial_circles[i][0], initial_circles[i][1], initial_circles[i][2]])
    
    def objective(params):
        # Reconstruct circles from params
        circles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            circles.append([x, y, r])
        
        # Sum of radii (negative for maximization)
        return -sum(circle[2] for circle in circles)
    
    def constraint_func(params):
        # Constraint function for optimization
        circles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            circles.append([x, y, r])
        
        constraints = []
        
        # Boundary constraints (all should be >= 0)
        for i in range(n):
            x, y, r = circles[i]
            constraints.append(x - r)  # Left boundary
            constraints.append(y - r)  # Bottom boundary
            constraints.append(width - x - r)  # Right boundary
            constraints.append(height - y - r)  # Top boundary
        
        # Overlap constraints (distance >= sum of radii)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                constraints.append(distance - (r1 + r2))
        
        return np.array(constraints)
    
    # Define bounds
    bounds = []
    for i in range(n):
        bounds.append((0.001, width - 0.001))      # x coordinates
        bounds.append((0.001, height - 0.001))    # y coordinates
        bounds.append((0.001, min(width, height) / 2 - 0.001))  # radii
    
    # Define constraints
    cons = {
        'type': 'ineq',
        'fun': constraint_func
    }
    
    try:
        # Perform optimization with extreme precision
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14, 'disp': False}
        )
        
        if result.success:
            # Reconstruct final circles
            circles = []
            for i in range(n):
                x = result.x[3*i]
                y = result.x[3*i+1]
                r = result.x[3*i+2]
                circles.append([x, y, r])
            return np.array(circles)
        else:
            return initial_circles
            
    except Exception:
        return initial_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
