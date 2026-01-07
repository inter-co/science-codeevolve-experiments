# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import random
from typing import Tuple, List
import time
import math

# Global constants
N_CIRCLES = 32
TIME_LIMIT = 55.0         # seconds

def initialize_positions() -> np.ndarray:
    """Initialize positions using a better hexagonal grid approach like INSPIRATION 2"""
    # Create a more optimal hexagonal grid pattern
    grid_size = int(math.ceil(math.sqrt(N_CIRCLES)))
    positions = []
    
    # Better hexagonal layout with more uniform spacing - inspired by INSPIRATION 2
    for i in range(grid_size):
        for j in range(grid_size):
            if len(positions) < N_CIRCLES:
                # Hexagonal offset pattern
                offset = 0.5 if j % 2 == 0 else 0.0
                x = (i + offset) / (grid_size - 1) if grid_size > 1 else 0.5
                y = (j + 0.5) / (grid_size - 1) if grid_size > 1 else 0.5
                # Add small random perturbation for better exploration (more aggressive than before)
                x += random.uniform(-0.03, 0.03)
                y += random.uniform(-0.03, 0.03)
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                positions.append([x, y])
    
    # Fill remaining positions with strategic random points
    while len(positions) < N_CIRCLES:
        x = random.uniform(0.05, 0.95)
        y = random.uniform(0.05, 0.95)
        positions.append([x, y])
        
    return np.array(positions[:N_CIRCLES])

def fitness_function(circles: np.ndarray) -> float:
    """Calculate fitness as sum of radii (higher is better)"""
    return np.sum(circles[:, 2])

def create_initial_solution() -> np.ndarray:
    """Create initial solution using better initialization like INSPIRATION 2"""
    # Use better initialization approach
    positions = initialize_positions()
    
    # Use more reasonable initial radii based on spacing - even larger to allow more growth
    initial_radii = [0.12] * N_CIRCLES  # Slightly larger initial radii
    
    # Create initial circles array
    circles = np.zeros((N_CIRCLES, 3))
    for i in range(N_CIRCLES):
        circles[i] = [positions[i][0], positions[i][1], initial_radii[i]]
    
    # Enforce constraints to ensure validity
    circles = enforce_constraints(circles)
    return circles

def enforce_constraints(circles: np.ndarray) -> np.ndarray:
    """Enforce all constraints on circle positions"""
    # Ensure all circles are within bounds
    for i in range(N_CIRCLES):
        x, y, r = circles[i]
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    # Resolve overlaps with iterative adjustment
    for _ in range(100):
        overlaps = find_overlaps(circles)
        if not overlaps:
            break
            
        for i, j in overlaps:
            # Move circles apart along the line connecting centers
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                overlap = (circles[i, 2] + circles[j, 2]) - dist
                if overlap > 0:
                    # Move them apart
                    scale = overlap / (dist * 2.0)
                    circles[i, 0] += dx * scale
                    circles[i, 1] += dy * scale
                    circles[j, 0] -= dx * scale
                    circles[j, 1] -= dy * scale
                    
                    # Adjust radii to prevent future overlaps - more aggressive
                    max_reduction = min(overlap * 0.3, circles[i, 2] * 0.3, circles[j, 2] * 0.3)
                    if max_reduction > 0:
                        circles[i, 2] -= max_reduction * 0.5
                        circles[j, 2] -= max_reduction * 0.5
                        
                    # Keep within bounds
                    circles[i, 0] = max(circles[i, 2], min(1-circles[i, 2], circles[i, 0]))
                    circles[i, 1] = max(circles[i, 2], min(1-circles[i, 2], circles[i, 1]))
                    circles[j, 0] = max(circles[j, 2], min(1-circles[j, 2], circles[j, 0]))
                    circles[j, 1] = max(circles[j, 2], min(1-circles[j, 2], circles[j, 1]))
    
    # Final pass to ensure all constraints
    for i in range(N_CIRCLES):
        x, y, r = circles[i]
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        circles[i] = [x, y, r]
    
    return circles

def find_overlaps(circles: np.ndarray) -> List[Tuple[int, int]]:
    """Find overlapping pairs of circles efficiently"""
    overlaps = []
    
    # Use spatial indexing for better performance
    positions = circles[:, :2]
    distances = cdist(positions, positions)
    
    for i in range(N_CIRCLES):
        for j in range(i+1, N_CIRCLES):
            # Check if circles are overlapping
            if distances[i, j] < (circles[i, 2] + circles[j, 2]):
                overlaps.append((i, j))
    
    return overlaps

def optimize_with_scipy(circles: np.ndarray) -> np.ndarray:
    """Use scipy optimization to refine the solution with improved constraints"""
    try:
        # Flatten the circles array for scipy optimization
        n = N_CIRCLES
        initial_params = []
        for i in range(n):
            initial_params.extend([circles[i, 0], circles[i, 1], circles[i, 2]])
        
        # Define objective function (negative because we want to maximize sum of radii)
        def objective(params):
            total_radius = 0
            for i in range(0, len(params), 3):
                total_radius += params[i+2]  # radius is third element
            return -total_radius  # negative because we minimize
        
        # Constraint functions
        def contain_constraint(params):
            result = []
            for i in range(0, len(params), 3):
                x, y, r = params[i], params[i+1], params[i+2]
                # Each circle must be contained within [0,1]x[0,1]
                result.extend([
                    x - r,      # x - r >= 0
                    y - r,      # y - r >= 0
                    1 - x - r,  # 1 - x - r >= 0
                    1 - y - r   # 1 - y - r >= 0
                ])
            return np.array(result)
        
        def overlap_constraint(params):
            result = []
            # Check all pairs of circles
            for i in range(0, len(params), 3):
                for j in range(i + 3, len(params), 3):
                    x1, y1, r1 = params[i], params[i+1], params[i+2]
                    x2, y2, r2 = params[j], params[j+1], params[j+2]
                    # Distance between centers minus sum of radii must be >= 0
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    result.append(dist - (r1 + r2))
            return np.array(result)
        
        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': contain_constraint},
            {'type': 'ineq', 'fun': overlap_constraint}
        ]
        
        # Optimization with SLSQP - even higher iterations and better tolerances like INSPIRATION 2
        result = minimize(
            objective,
            initial_params,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-8}
        )
        
        if result.success:
            # Convert back to circles array
            optimized_circles = circles.copy()
            for i in range(n):
                optimized_circles[i] = [
                    result.x[i*3],
                    result.x[i*3 + 1],
                    max(0.001, result.x[i*3 + 2])
                ]
            return optimized_circles
    except Exception:
        # If optimization fails, return original circles
        pass
    
    return circles

def local_refinement(circles: np.ndarray) -> np.ndarray:
    """Apply local refinement using physics-inspired approach like INSPIRATION 2"""
    # Use more efficient distance computation and better repulsion model
    max_refinement_iter = 300  # More iterations to allow better convergence
    learning_rate = 0.015      # Slightly higher learning rate for more aggressive movement
    
    for iteration in range(max_refinement_iter):
        # Calculate pairwise distances efficiently using scipy
        positions = circles[:, :2]
        distances = cdist(positions, positions)
        
        # Create force matrix for repulsion
        forces = np.zeros_like(positions)
        
        # Compute repulsion forces between overlapping circles
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                distance = distances[i, j]  # Already computed
                
                # Apply force if circles are overlapping or very close
                if distance < (circles[i, 2] + circles[j, 2]):
                    # Soft repulsion force with better scaling - more aggressive
                    force_magnitude = max(0, (circles[i, 2] + circles[j, 2] - distance) / (distance + 1e-6))
                    force_x = force_magnitude * dx / (distance + 1e-6)
                    force_y = force_magnitude * dy / (distance + 1e-6)
                    
                    forces[i, 0] += force_x
                    forces[i, 1] += force_y
                    forces[j, 0] -= force_x
                    forces[j, 1] -= force_y
        
        # Update positions with forces and learning rate
        for i in range(N_CIRCLES):
            # Apply forces to update positions
            circles[i, 0] += learning_rate * forces[i, 0]
            circles[i, 1] += learning_rate * forces[i, 1]
            
            # Ensure circles stay within bounds
            circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
            circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
        
        # Try to increase radii where possible with more aggressive approach
        for i in range(N_CIRCLES):
            if circles[i, 2] < 0.45:  # Allow even higher radii for better packing
                safe_to_increase = True
                # Check if increasing would violate constraints with more relaxed tolerance
                for j in range(N_CIRCLES):
                    if i != j:
                        dx = circles[i, 0] - circles[j, 0]
                        dy = circles[i, 1] - circles[j, 1]
                        distance = np.sqrt(dx*dx + dy*dy)
                        # Even less strict check to allow more aggressive radius increases
                        if distance < (circles[i, 2] + 0.0001 + circles[j, 2]):
                            safe_to_increase = False
                            break
                
                if safe_to_increase:
                    # Increase radius more aggressively with even smaller increments
                    circles[i, 2] = min(0.45, circles[i, 2] + 0.003)
    
    # Final boundary correction and cleanup
    for i in range(N_CIRCLES):
        circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1 - circles[i, 2])
        circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1 - circles[i, 2])
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses an enhanced hybrid approach combining better initialization, advanced optimization, 
    and post-processing refinement (like INSPIRATION 2).

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    start_time = time.time()
    
    # Create initial solution using better initialization
    circles = create_initial_solution()
    
    # Apply scipy optimization to the initial solution for fine-tuning
    circles = optimize_with_scipy(circles)
    
    # Apply local refinement
    circles = local_refinement(circles)
    
    # Final constraint enforcement
    circles = enforce_constraints(circles)
    
    # Try a few more optimization attempts with random perturbations - more attempts for better results
    best_fitness = fitness_function(circles)
    best_solution = circles.copy()
    
    # Try several local refinements with different strategies
    for attempt in range(6):  # Even more attempts for better exploration
        if time.time() - start_time > TIME_LIMIT:
            break
            
        # Slightly perturb and re-optimize
        perturbed = circles.copy()
        for i in range(N_CIRCLES):
            if random.random() < 0.45:  # Even higher chance of perturbation
                perturbed[i, 0] += random.uniform(-0.035, 0.035)
                perturbed[i, 1] += random.uniform(-0.035, 0.035)
                # Keep within bounds
                perturbed[i, 0] = max(perturbed[i, 2], min(1-perturbed[i, 2], perturbed[i, 0]))
                perturbed[i, 1] = max(perturbed[i, 2], min(1-perturbed[i, 2], perturbed[i, 1]))
        
        # Re-optimize the perturbed solution
        refined = optimize_with_scipy(perturbed)
        refined = local_refinement(refined)
        refined_fitness = fitness_function(refined)
        
        if refined_fitness > best_fitness:
            best_fitness = refined_fitness
            best_solution = refined
    
    # Final constraint enforcement
    best_solution = enforce_constraints(best_solution)
    
    return best_solution


# EVOLVE-BLOCK-END
