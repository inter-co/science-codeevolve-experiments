# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import math
import random
from typing import Tuple, List
import copy

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining multiple optimization strategies.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    
    # Use multiple restart strategies with different optimization methods
    best_circles = None
    best_sum = 0
    
    # Strategy 1: High-quality initialization + simulated annealing
    for restart in range(5):
        circles = generate_high_quality_initial(n)
        circles = simulated_annealing(circles)
        circles = local_optimization(circles)
        total_radius = np.sum(circles[:, 2])
        if total_radius > best_sum:
            best_sum = total_radius
            best_circles = circles.copy()
    
    # Strategy 2: Mathematical optimization with multiple starting points
    for restart in range(3):
        circles = generate_focused_initial(n)
        circles = optimize_with_multiple_methods(circles)
        total_radius = np.sum(circles[:, 2])
        if total_radius > best_sum:
            best_sum = total_radius
            best_circles = circles.copy()
    
    # Strategy 3: Enhanced local search
    if best_circles is not None:
        circles = enhanced_local_search(best_circles)
        total_radius = np.sum(circles[:, 2])
        if total_radius > best_sum:
            best_circles = circles.copy()
    
    # Strategy 4: Additional aggressive refinement
    if best_circles is not None:
        circles = aggressive_refinement(best_circles)
        total_radius = np.sum(circles[:, 2])
        if total_radius > best_sum:
            best_circles = circles.copy()
    
    # Fallback to a robust initial configuration if nothing worked well
    if best_circles is None:
        best_circles = generate_high_quality_initial(n)
    
    return best_circles

def generate_high_quality_initial(n: int) -> np.ndarray:
    """Generate high-quality initial configuration using geometric insights"""
    # Create a pattern inspired by dense packings
    circles = []
    
    # Layer 1: Central dense cluster
    center_x, center_y = 0.5, 0.5
    radius = 0.15
    num_in_layer = 8
    
    for i in range(num_in_layer):
        angle = 2 * math.pi * i / num_in_layer
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        circles.append([x, y, 0.08])
    
    # Layer 2: Outer ring
    outer_radius = 0.3
    num_in_outer = 12
    
    for i in range(num_in_outer):
        angle = 2 * math.pi * i / num_in_outer
        x = center_x + outer_radius * math.cos(angle)
        y = center_y + outer_radius * math.sin(angle)
        circles.append([x, y, 0.06])
    
    # Fill remaining positions with random valid placements
    while len(circles) < n:
        x = np.random.uniform(0.05, 0.95)
        y = np.random.uniform(0.05, 0.95)
        radius = np.random.uniform(0.03, 0.12)
        
        # Check collision with existing circles
        valid = True
        for circ in circles:
            dx = x - circ[0]
            dy = y - circ[1]
            dist_sq = dx*dx + dy*dy
            min_dist_sq = (radius + circ[2])**2
            
            if dist_sq < min_dist_sq:
                valid = False
                break
        
        if valid:
            circles.append([x, y, radius])
    
    return np.array(circles[:n])

def generate_focused_initial(n: int) -> np.ndarray:
    """Generate focused initial configuration with better density"""
    # Create a more systematic approach
    circles = []
    
    # Start with a hexagonal pattern in the center
    rows = 5
    cols = 5
    spacing_x = 0.8 / cols
    spacing_y = 0.8 / rows
    
    max_radius = min(spacing_x, spacing_y) * 0.35
    
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
                
            offset = (i % 2) * 0.5
            x = (j + offset) * spacing_x + 0.1
            y = i * spacing_y + 0.1
            
            if x + max_radius <= 1 and y + max_radius <= 1 and x - max_radius >= 0 and y - max_radius >= 0:
                circles.append([x, y, max_radius])
    
    # Fill remaining positions
    while len(circles) < n:
        x = np.random.uniform(max_radius, 1 - max_radius)
        y = np.random.uniform(max_radius, 1 - max_radius)
        # Use smaller radii for more flexibility
        radius = np.random.uniform(0.02, 0.1)
        circles.append([x, y, radius])
    
    return np.array(circles[:n])

def simulated_annealing(circles: np.ndarray) -> np.ndarray:
    """Enhanced simulated annealing optimization for circle packing"""
    n = len(circles)
    current_circles = circles.copy()
    current_energy = -calculate_total_radius(current_circles)  # Negative for maximization
    
    # Annealing parameters with better tuning
    temperature = 1.0
    min_temperature = 0.0001
    cooling_rate = 0.9995
    max_iterations = 10000
    
    # Track best solution
    best_circles = current_circles.copy()
    best_energy = current_energy
    
    for iteration in range(max_iterations):
        # Cool down temperature
        if temperature < min_temperature:
            temperature = min_temperature
        
        # Generate neighbor solution with adaptive perturbation
        neighbor_circles = create_adaptive_neighbor(current_circles)
        
        # Calculate energy difference
        neighbor_energy = -calculate_total_radius(neighbor_circles)
        delta_energy = neighbor_energy - current_energy
        
        # Accept or reject based on Metropolis criterion
        if delta_energy > 0 or np.random.random() < np.exp(delta_energy / temperature):
            current_circles = neighbor_circles
            current_energy = neighbor_energy
            
            # Update best solution if improved
            if current_energy < best_energy:
                best_energy = current_energy
                best_circles = current_circles.copy()
        
        # Occasionally do larger jumps
        if iteration % 500 == 0 and iteration > 0:
            temperature = max(temperature * 0.9, min_temperature)
        
        # Cool down
        temperature *= cooling_rate
    
    return best_circles

def calculate_total_radius(circles: np.ndarray) -> float:
    """Calculate total sum of radii"""
    return np.sum(circles[:, 2])

def create_adaptive_neighbor(circles: np.ndarray) -> np.ndarray:
    """Create a neighboring solution with adaptive perturbation sizes"""
    neighbor = circles.copy()
    
    # Select a random circle to modify
    idx = np.random.randint(0, len(neighbor))
    
    # Modify with adaptive probabilities and sizes
    if np.random.random() < 0.6:  # 60% chance to modify position
        # Larger perturbation for exploration
        perturbation_size = 0.02 if np.random.random() < 0.3 else 0.005
        neighbor[idx][0] += np.random.normal(0, perturbation_size)
        neighbor[idx][1] += np.random.normal(0, perturbation_size)
        
        # Keep within bounds
        neighbor[idx][0] = np.clip(neighbor[idx][0], neighbor[idx][2], 1 - neighbor[idx][2])
        neighbor[idx][1] = np.clip(neighbor[idx][1], neighbor[idx][2], 1 - neighbor[idx][2])
    else:  # 40% chance to modify radius
        # Smaller radius perturbation
        neighbor[idx][2] += np.random.normal(0, 0.008)
        neighbor[idx][2] = np.clip(neighbor[idx][2], 0.001, 0.5)
    
    return neighbor

def local_optimization(circles: np.ndarray) -> np.ndarray:
    """Refine solution using multiple optimization techniques"""
    # First, run mathematical optimization on the current solution
    circles = optimize_with_slsqp(circles)
    
    # Then run trust-constr optimization if available
    try:
        circles = optimize_with_trust_constr(circles)
    except:
        pass
    
    # Finally perform local refinement
    circles = local_refinement(circles)
    
    return circles

def optimize_with_multiple_methods(circles: np.ndarray) -> np.ndarray:
    """Try multiple optimization methods to find better solutions"""
    # Try SLSQP first
    result1 = optimize_with_slsqp(circles)
    score1 = calculate_total_radius(result1)
    
    # Try Trust-constr if available
    result2 = circles.copy()
    try:
        result2 = optimize_with_trust_constr(circles)
        score2 = calculate_total_radius(result2)
    except:
        score2 = score1
    
    # Return the better result
    return result1 if score1 >= score2 else result2

def optimize_with_slsqp(circles: np.ndarray) -> np.ndarray:
    """Optimize using SLSQP method with enhanced settings"""
    n = len(circles)
    
    # Flatten the optimization variables: [x1, y1, r1, x2, y2, r2, ...]
    initial_vars = []
    for i in range(n):
        x, y, r = circles[i]
        initial_vars.extend([x, y, r])
    
    # Define constraints more carefully
    def constraint_containment(vars_list):
        constraints = []
        for i in range(n):
            x = vars_list[3*i]
            y = vars_list[3*i + 1]
            r = vars_list[3*i + 2]
            
            # x - r >= 0
            constraints.append(x - r)
            # x + r <= 1
            constraints.append(1 - x - r)
            # y - r >= 0
            constraints.append(y - r)
            # y + r <= 1
            constraints.append(1 - y - r)
        return np.array(constraints)
    
    def constraint_overlap(vars_list):
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1 = vars_list[3*i]
                y1 = vars_list[3*i + 1]
                r1 = vars_list[3*i + 2]
                x2 = vars_list[3*j]
                y2 = vars_list[3*j + 1]
                r2 = vars_list[3*j + 2]
                
                # Distance squared between centers
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                
                # Minimum distance squared for non-overlap
                min_dist_sq = (r1 + r2)**2
                
                # Constraint: dist_sq >= min_dist_sq (for non-overlap)
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)
    
    # Objective function to maximize (negative because scipy minimizes)
    def objective(vars_list):
        return -sum(vars_list[2::3])  # Sum of all radii (negative for maximization)
    
    # Create constraint dictionaries
    constraints = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_overlap(x)}
    ]
    
    # Bounds for variables: [0.001, 0.999] for x,y and [0.001, 0.5] for radii
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])
    
    try:
        # Run optimization with enhanced settings
        result = minimize(objective, initial_vars, method='SLSQP', 
                         bounds=bounds, constraints=constraints, 
                         options={'maxiter': 6000, 'ftol': 1e-12, 'gtol': 1e-12})
        
        if result.success:
            # Extract optimized values
            optimized_vars = result.x
            circles = []
            for i in range(n):
                x = optimized_vars[3*i]
                y = optimized_vars[3*i + 1]
                r = optimized_vars[3*i + 2]
                circles.append([x, y, r])
            return np.array(circles)
    except Exception as e:
        # If optimization fails, return original circles
        pass
    
    return circles

def optimize_with_trust_constr(circles: np.ndarray) -> np.ndarray:
    """Optimize using trust-constr method"""
    n = len(circles)
    
    # Flatten the optimization variables: [x1, y1, r1, x2, y2, r2, ...]
    initial_vars = []
    for i in range(n):
        x, y, r = circles[i]
        initial_vars.extend([x, y, r])
    
    # Define constraints
    def constraint_containment(vars_list):
        constraints = []
        for i in range(n):
            x = vars_list[3*i]
            y = vars_list[3*i + 1]
            r = vars_list[3*i + 2]
            
            # x - r >= 0
            constraints.append(x - r)
            # x + r <= 1
            constraints.append(1 - x - r)
            # y - r >= 0
            constraints.append(y - r)
            # y + r <= 1
            constraints.append(1 - y - r)
        return np.array(constraints)
    
    def constraint_overlap(vars_list):
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                x1 = vars_list[3*i]
                y1 = vars_list[3*i + 1]
                r1 = vars_list[3*i + 2]
                x2 = vars_list[3*j]
                y2 = vars_list[3*j + 1]
                r2 = vars_list[3*j + 2]
                
                # Distance squared between centers
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                
                # Minimum distance squared for non-overlap
                min_dist_sq = (r1 + r2)**2
                
                # Constraint: dist_sq >= min_dist_sq (for non-overlap)
                constraints.append(dist_sq - min_dist_sq)
        return np.array(constraints)
    
    # Objective function to maximize (negative because scipy minimizes)
    def objective(vars_list):
        return -sum(vars_list[2::3])  # Sum of all radii (negative for maximization)
    
    # Create constraint dictionaries
    constraints = [
        {'type': 'ineq', 'fun': lambda x: constraint_containment(x)},
        {'type': 'ineq', 'fun': lambda x: constraint_overlap(x)}
    ]
    
    # Bounds for variables: [0.001, 0.999] for x,y and [0.001, 0.5] for radii
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])
    
    try:
        # Run optimization with trust-constr method
        result = minimize(objective, initial_vars, method='trust-constr', 
                         bounds=bounds, constraints=constraints, 
                         options={'maxiter': 4000, 'ftol': 1e-12})
        
        if result.success:
            # Extract optimized values
            optimized_vars = result.x
            circles = []
            for i in range(n):
                x = optimized_vars[3*i]
                y = optimized_vars[3*i + 1]
                r = optimized_vars[3*i + 2]
                circles.append([x, y, r])
            return np.array(circles)
    except Exception as e:
        # If optimization fails, return original circles
        pass
    
    return circles

def local_refinement(circles: np.ndarray) -> np.ndarray:
    """Enhanced local refinement to improve solution quality"""
    # Try to slightly increase radii where possible without violating constraints
    max_iter = 400
    for iteration in range(max_iter):
        improved = False
        for i in range(len(circles)):
            # Try to slightly increase radius
            old_radius = circles[i][2]
            # Use a more aggressive increase rate
            new_radius = min(old_radius * 1.025, 0.5)  # Even more aggressive increase
            
            # Check if we can actually increase the radius
            valid = True
            for j in range(len(circles)):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (new_radius + circles[j][2])**2
                    
                    if dist_sq < min_dist_sq:
                        valid = False
                        break
            
            # Also check containment
            if (valid and 
                circles[i][0] + new_radius <= 1 and 
                circles[i][1] + new_radius <= 1 and
                circles[i][0] - new_radius >= 0 and
                circles[i][1] - new_radius >= 0):
                circles[i][2] = new_radius
                improved = True
        
        if not improved:
            break
    
    return circles

def enhanced_local_search(circles: np.ndarray) -> np.ndarray:
    """Perform enhanced local search to squeeze out additional improvements"""
    # Try to improve individual circles one by one
    improved = True
    iteration = 0
    max_iterations = 50
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try increasing each circle's radius individually
        for i in range(len(circles)):
            old_radius = circles[i][2]
            new_radius = min(old_radius * 1.015, 0.5)
            
            # Check if we can increase this radius
            valid = True
            for j in range(len(circles)):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (new_radius + circles[j][2])**2
                    
                    if dist_sq < min_dist_sq:
                        valid = False
                        break
            
            # Check containment
            if (valid and 
                circles[i][0] + new_radius <= 1 and 
                circles[i][1] + new_radius <= 1 and
                circles[i][0] - new_radius >= 0 and
                circles[i][1] - new_radius >= 0):
                circles[i][2] = new_radius
                improved = True
    
    return circles

def aggressive_refinement(circles: np.ndarray) -> np.ndarray:
    """Aggressive refinement to maximize any remaining gains"""
    # Perform multiple rounds of very fine-grained optimization
    for round_num in range(3):
        # Try to increase all radii by small amounts
        for i in range(len(circles)):
            old_radius = circles[i][2]
            # Very small increases for maximum precision
            new_radius = min(old_radius * 1.005, 0.5)
            
            # Check if we can actually increase the radius
            valid = True
            for j in range(len(circles)):
                if i != j:
                    dx = circles[i][0] - circles[j][0]
                    dy = circles[i][1] - circles[j][1]
                    dist_sq = dx*dx + dy*dy
                    min_dist_sq = (new_radius + circles[j][2])**2
                    
                    if dist_sq < min_dist_sq:
                        valid = False
                        break
            
            # Also check containment
            if (valid and 
                circles[i][0] + new_radius <= 1 and 
                circles[i][1] + new_radius <= 1 and
                circles[i][0] - new_radius >= 0 and
                circles[i][1] - new_radius >= 0):
                circles[i][2] = new_radius
    
    return circles

def validate_solution(circles: np.ndarray) -> bool:
    """Validate that all constraints are satisfied"""
    n = len(circles)
    
    # Check containment
    for i in range(n):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return False
    
    # Check non-overlap using spatial indexing
    tree = cKDTree(circles[:, :2])
    max_radius = max(circles[:, 2]) if len(circles) > 0 else 0.1
    
    for i in range(n):
        x, y, r = circles[i]
        
        # Find nearby circles
        nearby_indices = tree.query_ball_point([x, y], 2 * max_radius)
        
        # Check actual overlap
        for j in nearby_indices:
            if i != j:
                x2, y2, r2 = circles[j]
                dx = x - x2
                dy = y - y2
                dist_sq = dx*dx + dy*dy
                min_dist_sq = (r + r2)**2
                
                if dist_sq < min_dist_sq:
                    return False
    
    return True


# EVOLVE-BLOCK-END
