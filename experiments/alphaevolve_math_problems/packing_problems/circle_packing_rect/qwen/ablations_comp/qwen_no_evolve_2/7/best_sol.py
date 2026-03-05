# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import distance
from scipy.optimize import minimize
import random
from typing import Tuple
import time
from itertools import combinations
import math
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
from numba import jit, prange
import heapq
from scipy.spatial.distance import cdist

@jit(nopython=True)
def compute_distance_sq(x1, y1, x2, y2):
    """Fast distance squared computation"""
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy

@jit(nopython=True)
def check_overlap_fast(circles, i, j):
    """Fast overlap check"""
    x1, y1, r1 = circles[i]
    x2, y2, r2 = circles[j]
    dist_sq = compute_distance_sq(x1, y1, x2, y2)
    return dist_sq < (r1 + r2) * (r1 + r2)

@jit(nopython=True)
def check_bounds_fast(circles, width, height):
    """Fast boundary check"""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x < r or x > width - r or y < r or y > height - r:
            return False
    return True

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.
    Uses a hybrid approach combining geometric initialization, simulated annealing, and local optimization.
    
    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    # Rectangle dimensions: width + height = 2
    # Try different aspect ratios to find optimal configuration
    best_result = None
    best_sum = 0
    
    # Test several aspect ratios - focus on those that typically work well for circle packing
    # Try more precise ratios around optimal values
    ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5]
    
    for ratio in ratios:
        width = 1.0
        height = 1.0 / ratio if ratio > 1 else ratio
        
        # Multi-scale approach: start with better initialization
        circles = initialize_better(width, height, 21)
        
        # Apply simulated annealing for global optimization
        circles = simulated_annealing_optimization(circles, width, height, iterations=3000)
        
        # Refine with local optimization
        circles = refine_circles(circles, width, height)
        
        # Calculate sum of radii
        total_radius = np.sum(circles[:, 2])
        
        if total_radius > best_sum:
            best_sum = total_radius
            best_result = circles.copy()
    
    return best_result if best_result is not None else generate_default_solution(1.0, 1.0, 21)

def initialize_better(width: float, height: float, n: int) -> np.ndarray:
    """Better initialization using hexagonal packing idea and strategic placement"""
    circles = np.zeros((n, 3))
    
    # For 21 circles, use a more sophisticated approach:
    # 1. Try to place circles in a hexagonal lattice pattern
    # 2. Place additional circles in strategic positions
    
    # Estimate how many rows/columns for hexagonal packing
    rows = int(np.ceil(np.sqrt(n * 1.2)))  # Allow for more efficient packing
    cols = int(np.ceil(n / rows))
    
    # Calculate spacing for hexagonal packing
    # Using triangular lattice for better packing efficiency
    spacing_x = width / (cols + 1) if cols > 0 else width
    spacing_y = height / (rows + 1) if rows > 0 else height
    
    # For hexagonal packing, adjust spacing
    hex_spacing_x = spacing_x * 0.866  # sqrt(3)/2
    hex_spacing_y = spacing_y * 0.75   # 3/4
    
    count = 0
    
    # Fill hexagonal pattern
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            # Offset every other row for hexagonal packing
            offset = (i % 2) * (hex_spacing_x / 2)
            x = spacing_x * (j + 1) + offset
            y = spacing_y * (i + 1)
            
            # Add systematic randomization to avoid perfect patterns
            x += np.random.uniform(-hex_spacing_x/10, hex_spacing_x/10)
            y += np.random.uniform(-hex_spacing_y/10, hex_spacing_y/10)
            
            # Ensure within bounds
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))
            
            # Initial radius - based on available space
            max_radius = min(x, width - x, y, height - y)
            # Start with a higher percentage of max possible radius for better optimization
            radius = max_radius * 0.30  # Slightly higher starting point
            
            circles[count] = [x, y, radius]
            count += 1
            
        if count >= n:
            break
    
    # Add additional circles in corners and edges for better distribution
    # More strategic corner positions
    corner_positions = [
        (0.1 * width, 0.1 * height),
        (0.9 * width, 0.1 * height),
        (0.1 * width, 0.9 * height),
        (0.9 * width, 0.9 * height),
        (0.5 * width, 0.1 * height),
        (0.5 * width, 0.9 * height),
        (0.1 * width, 0.5 * height),
        (0.9 * width, 0.5 * height),
        (0.25 * width, 0.25 * height),
        (0.75 * width, 0.25 * height),
        (0.25 * width, 0.75 * height),
        (0.75 * width, 0.75 * height)
    ]
    
    # Fill remaining positions with corner/edge placements
    remaining = n - count
    for i in range(remaining):
        if count + i >= n:
            break
        # Use corner positions cyclically
        pos_idx = i % len(corner_positions)
        x, y = corner_positions[pos_idx]
        
        # Add small random offset
        x += np.random.uniform(-width * 0.03, width * 0.03)
        y += np.random.uniform(-height * 0.03, height * 0.03)
        
        # Ensure within bounds
        x = max(0.01, min(width - 0.01, x))
        y = max(0.01, min(height - 0.01, y))
        
        # Initial radius based on distance from center and edges
        max_radius = min(x, width - x, y, height - y)
        radius = max_radius * 0.20
        
        circles[count + i] = [x, y, radius]
    
    count += remaining
    
    # Improve by clustering to distribute circles more evenly
    if n > 10:
        points = circles[:, :2]
        try:
            # Use fewer clusters for better distribution
            kmeans = KMeans(n_clusters=min(4, n//5), random_state=42, n_init=5)
            clusters = kmeans.fit_predict(points)
            
            # Adjust radii based on cluster density
            for i in range(len(clusters)):
                cluster_id = clusters[i]
                # Get points in same cluster
                cluster_points = points[clusters == cluster_id]
                if len(cluster_points) > 1:
                    # Compute average distance to other points in cluster
                    distances = [np.linalg.norm(points[i] - p) for p in cluster_points if not np.allclose(points[i], p)]
                    if distances:
                        avg_dist = np.mean(distances)
                        if avg_dist > 0.01:
                            # Reduce radius to allow for better packing
                            circles[i, 2] = min(circles[i, 2], avg_dist * 0.15)
        except:
            pass  # Fallback to basic initialization if clustering fails
    
    return circles

def simulated_annealing_optimization(circles: np.ndarray, width: float, height: float, iterations: int = 3000) -> np.ndarray:
    """Apply simulated annealing to improve circle packing"""
    current_circles = circles.copy()
    current_energy = -compute_total_radius(current_circles)  # Negative because we want to maximize
    
    # Better temperature schedule for more thorough exploration
    temp = 0.2
    cooling_rate = 0.999
    
    # Track best solution found
    best_circles = current_circles.copy()
    best_energy = current_energy
    
    # More diverse perturbation strategies with better weights
    perturbation_types = ['position', 'radius', 'both']
    perturbation_weights = [0.5, 0.3, 0.2]  # Emphasize position changes more
    
    for iteration in range(iterations):
        # Generate neighbor solution by perturbing one circle
        new_circles = current_circles.copy()
        circle_idx = np.random.randint(0, len(new_circles))
        
        # Choose perturbation type with weighted probabilities
        pert_type = np.random.choice(perturbation_types, p=perturbation_weights)
        
        # Position perturbation - adapt to rectangle size
        pos_perturbation = 0.03 * min(width, height)
        
        if pert_type in ['position', 'both']:
            new_circles[circle_idx, 0] += np.random.normal(0, pos_perturbation * 0.7)
            new_circles[circle_idx, 1] += np.random.normal(0, pos_perturbation * 0.7)
        
        # Radius perturbation
        if pert_type in ['radius', 'both']:
            radius_perturbation = 0.02
            new_circles[circle_idx, 2] += np.random.normal(0, radius_perturbation * np.max([new_circles[circle_idx, 2], 0.01]))
        
        # Keep within bounds
        new_circles[circle_idx, 0] = max(new_circles[circle_idx, 2], 
                                       min(width - new_circles[circle_idx, 2], 
                                           new_circles[circle_idx, 0]))
        new_circles[circle_idx, 1] = max(new_circles[circle_idx, 2], 
                                       min(height - new_circles[circle_idx, 2], 
                                           new_circles[circle_idx, 1]))
        new_circles[circle_idx, 2] = max(0.001, new_circles[circle_idx, 2])
        
        # Check constraints and compute energy
        if check_all_constraints_fast(new_circles, width, height):
            new_energy = -compute_total_radius(new_circles)
            
            # Accept or reject based on Metropolis criterion
            delta_energy = new_energy - current_energy
            if delta_energy > 0 or np.random.rand() < np.exp(delta_energy / temp):
                current_circles = new_circles
                current_energy = new_energy
                
                # Update best solution if improved
                if new_energy < best_energy:  # Since we're minimizing negative energy
                    best_energy = new_energy
                    best_circles = new_circles.copy()
        else:
            # If constraints violated, still try to accept with penalty
            # This helps escape local minima
            if np.random.rand() < 0.02:  # Lower acceptance rate for constraint violations
                current_circles = new_circles
                current_energy = -compute_total_radius(new_circles)
        
        # Cool down temperature
        temp *= cooling_rate
        
        # Occasionally reset temperature to escape local minima
        if iteration % 300 == 0 and temp < 0.001:
            temp = 0.02
    
    return best_circles

def compute_total_radius(circles: np.ndarray) -> float:
    """Compute the sum of all radii"""
    return np.sum(circles[:, 2])

@jit(nopython=True)
def check_all_constraints_fast(circles, width, height):
    """Fast constraint checking using Numba"""
    # Check boundary constraints
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x < r or x > width - r or y < r or y > height - r:
            return False
    
    # Check overlap constraints
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            if check_overlap_fast(circles, i, j):
                return False
    
    return True

def check_all_constraints(circles: np.ndarray, width: float, height: float) -> bool:
    """Check if all constraints are satisfied (for compatibility)"""
    # Use fast version for actual runtime
    return check_all_constraints_fast(circles, width, height)

def optimize_circle_positions(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize circle positions using constrained optimization"""
    n = len(circles)
    
    # Flatten parameters: [x0, y0, r0, x1, y1, r1, ...]
    initial_params = circles.flatten()
    
    def objective(params):
        # Reconstruct circles
        reconstructed = params.reshape(-1, 3)
        # Objective: maximize sum of radii (minimize negative sum)
        return -np.sum(reconstructed[:, 2])
    
    def constraint_func(params):
        # Reconstruct circles
        reconstructed = params.reshape(-1, 3)
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            x, y, r = reconstructed[i]
            # Ensure circles don't exceed boundaries
            constraints.extend([
                x - r,  # left boundary
                width - x - r,  # right boundary
                y - r,  # bottom boundary
                height - y - r  # top boundary
            ])
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = reconstructed[i]
                x2, y2, r2 = reconstructed[j]
                dist_sq = compute_distance_sq(x1, y1, x2, y2)
                # Constraint: dist^2 >= (r1 + r2)^2
                constraints.append(dist_sq - (r1 + r2)**2)
        
        return np.array(constraints)
    
    # Create constraints dictionary
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Optimize
    try:
        result = minimize(objective, initial_params, method='SLSQP', constraints=cons, 
                         options={'maxiter': 150, 'ftol': 1e-6})
        if result.success:
            return result.x.reshape(-1, 3)
    except Exception as e:
        # If optimization fails, return original circles
        pass
    
    # Return original if optimization fails
    return circles

def refine_circles(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Refine circle configuration using multi-stage optimization"""
    refined = circles.copy()
    
    # Stage 1: Global optimization using constrained optimization
    refined = optimize_circle_positions(refined, width, height)
    
    # Stage 2: Local refinement with boundary-aware adjustments
    for _ in range(30):  # More refinement passes
        # Adjust positions to avoid overlaps and respect boundaries
        for i in range(len(refined)):
            x, y, r = refined[i]
            
            # Keep within bounds
            x = max(r, min(width - r, x))
            y = max(r, min(height - r, y))
            
            # Adjust radius to maximize it while respecting constraints
            max_radius = min(x, width - x, y, height - y)
            
            # Check overlaps with other circles more efficiently
            new_radius = max_radius
            for j in range(len(refined)):
                if i != j:
                    x2, y2, r2 = refined[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    if dist > 0.001:
                        # Maximum radius without overlapping this circle
                        max_radius_for_this = dist - r2
                        new_radius = min(new_radius, max_radius_for_this)
            
            # Ensure positive radius
            new_radius = max(0.001, new_radius)
            refined[i] = [x, y, new_radius]
    
    # Stage 3: Final validation and adjustment with more aggressive overlap resolution
    refined = validate_and_correct(refined, width, height, max_iterations=50)
    
    return refined

def validate_and_correct(circles: np.ndarray, width: float, height: float, max_iterations: int = 50) -> np.ndarray:
    """Ensure all constraints are satisfied"""
    corrected_circles = circles.copy()
    
    # First, handle boundary violations
    for i in range(len(corrected_circles)):
        x, y, r = corrected_circles[i]
        # Correct positions that violate boundaries
        corrected_circles[i, 0] = max(r, min(width - r, x))
        corrected_circles[i, 1] = max(r, min(height - r, y))
    
    # Then resolve overlaps through iterative correction
    for _ in range(max_iterations):
        overlaps = []
        for i in range(len(corrected_circles)):
            for j in range(i+1, len(corrected_circles)):
                x1, y1, r1 = corrected_circles[i]
                x2, y2, r2 = corrected_circles[j]
                
                dx = x2 - x1
                dy = y2 - y1
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < (r1 + r2):
                    overlaps.append((i, j, distance, r1 + r2))
        
        if not overlaps:
            break
            
        # Resolve the most severe overlap first
        overlaps.sort(key=lambda x: x[3] - x[2])  # Sort by overlap amount
        i, j, dist, sum_radii = overlaps[-1]
        
        # Push circles apart along the line connecting centers
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > 0.001:  # Avoid division by zero
            push_amount = (sum_radii - distance) / 2
            dx_norm = dx / distance
            dy_norm = dy / distance
            
            # Move both circles away from each other with more aggressive pushing
            push_amount *= 1.5  # Even more aggressive
            corrected_circles[i, 0] -= dx_norm * push_amount
            corrected_circles[i, 1] -= dy_norm * push_amount
            corrected_circles[j, 0] += dx_norm * push_amount
            corrected_circles[j, 1] += dy_norm * push_amount
            
            # Keep within bounds
            corrected_circles[i, 0] = max(corrected_circles[i, 2], 
                                        min(width - corrected_circles[i, 2], 
                                            corrected_circles[i, 0]))
            corrected_circles[i, 1] = max(corrected_circles[i, 2], 
                                        min(height - corrected_circles[i, 2], 
                                            corrected_circles[i, 1]))
            corrected_circles[j, 0] = max(corrected_circles[j, 2], 
                                        min(width - corrected_circles[j, 2], 
                                            corrected_circles[j, 0]))
            corrected_circles[j, 1] = max(corrected_circles[j, 2], 
                                        min(height - corrected_circles[j, 2], 
                                            corrected_circles[j, 1]))
    
    return corrected_circles

def generate_default_solution(width: float, height: float, n: int) -> np.ndarray:
    """Fallback solution if optimization fails"""
    circles = np.zeros((n, 3))
    
    # Better grid approach with more careful spacing
    grid_size = int(np.ceil(np.sqrt(n)))
    spacing_x = width / (grid_size + 1)
    spacing_y = height / (grid_size + 1)
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= n:
                break
            x = spacing_x * (i + 1)
            y = spacing_y * (j + 1)
            # Use a slightly larger radius to start with
            radius = min(spacing_x, spacing_y) / 2.5
            
            # Ensure it's within bounds
            x = max(radius, min(width - radius, x))
            y = max(radius, min(height - radius, y))
            
            circles[count] = [x, y, radius]
            count += 1
            
        if count >= n:
            break
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
