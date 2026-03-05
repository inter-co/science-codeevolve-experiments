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
    
    # Test several aspect ratios - focusing on more promising ones
    # Focus on ratios that typically yield good results for circle packing
    ratios = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    
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
    """Better initialization using hexagonal packing and strategic placement"""
    circles = np.zeros((n, 3))
    
    # Use a more sophisticated approach based on circle packing theory
    # For 21 circles, create a pattern that resembles hexagonal close packing
    # but with some randomness to avoid getting stuck in poor local optima
    
    # Determine grid dimensions - use a more balanced approach
    rows = int(np.ceil(np.sqrt(n * 1.1)))  # Slightly more rows for better distribution
    cols = int(np.ceil(n / rows))
    
    # Calculate spacing for hexagonal packing
    # Using a tighter hexagonal packing for better space utilization
    spacing_x = width / (cols + 1) if cols > 0 else width
    spacing_y = height / (rows + 1) if rows > 0 else height
    
    # Hexagonal packing spacing factors
    hex_spacing_x = spacing_x * 0.866  # sqrt(3)/2 for horizontal spacing
    hex_spacing_y = spacing_y * 0.75   # 3/4 for vertical spacing
    
    count = 0
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            # Offset every other row for hexagonal packing
            offset = (i % 2) * (hex_spacing_x / 2)
            x = spacing_x * (j + 1) + offset
            y = spacing_y * (i + 1)
            
            # Add more systematic randomization to escape local minima
            x += np.random.uniform(-hex_spacing_x/10, hex_spacing_x/10)
            y += np.random.uniform(-hex_spacing_y/10, hex_spacing_y/10)
            
            # Ensure within bounds with safety margin
            x = max(0.01, min(width - 0.01, x))
            y = max(0.01, min(height - 0.01, y))
            
            # Initial radius - based on available space but start with a higher value
            max_radius = min(x, width - x, y, height - y)
            # Start with a higher percentage of max possible radius for better optimization
            radius = max_radius * 0.4  # Slightly higher starting point
            
            circles[count] = [x, y, radius]
            count += 1
            
        if count >= n:
            break
    
    # Improve by clustering to distribute circles more evenly
    if n > 10:
        points = circles[:, :2]
        try:
            # Use fewer clusters for better distribution
            n_clusters = min(5, n//4) if n > 15 else min(3, n//3)
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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
                                circles[i, 2] = min(circles[i, 2], avg_dist * 0.25)
        except:
            pass  # Fallback to basic initialization if clustering fails
    
    return circles

def simulated_annealing_optimization(circles: np.ndarray, width: float, height: float, iterations: int = 3000) -> np.ndarray:
    """Apply simulated annealing to improve circle packing"""
    current_circles = circles.copy()
    current_energy = -compute_total_radius(current_circles)  # Negative because we want to maximize
    
    # Improved temperature schedule and parameters
    temp = 0.2
    cooling_rate = 0.996
    
    # Track best solution found
    best_circles = current_circles.copy()
    best_energy = current_energy
    
    # Keep track of recent improvements for adaptive cooling
    recent_improvements = 0
    improvement_threshold = 30
    
    for iteration in range(iterations):
        # Generate neighbor solution by perturbing one circle
        new_circles = current_circles.copy()
        circle_idx = np.random.randint(0, len(new_circles))
        
        # Use different perturbation strategies based on iteration
        if iteration < iterations // 4:
            # Early phase: larger perturbations for global exploration
            pos_perturbation = 0.08 * min(width, height)
            radius_perturbation = 0.05
        elif iteration < iterations // 2:
            # Middle phase: moderate perturbations
            pos_perturbation = 0.03 * min(width, height)
            radius_perturbation = 0.02
        else:
            # Late phase: small perturbations for fine-tuning
            pos_perturbation = 0.015 * min(width, height)
            radius_perturbation = 0.01
        
        # Perturb position and radius differently for better exploration
        new_circles[circle_idx, 0] += np.random.normal(0, pos_perturbation)
        new_circles[circle_idx, 1] += np.random.normal(0, pos_perturbation)
        
        # Radius perturbation with bounded adjustment
        old_radius = new_circles[circle_idx, 2]
        new_circles[circle_idx, 2] += np.random.normal(0, radius_perturbation * max(old_radius, 0.01))
        
        # Keep within bounds
        new_circles[circle_idx, 0] = max(new_circles[circle_idx, 2], 
                                       min(width - new_circles[circle_idx, 2], 
                                           new_circles[circle_idx, 0]))
        new_circles[circle_idx, 1] = max(new_circles[circle_idx, 2], 
                                       min(height - new_circles[circle_idx, 2], 
                                           new_circles[circle_idx, 1]))
        new_circles[circle_idx, 2] = max(0.001, new_circles[circle_idx, 2])
        
        # Check constraints and compute energy
        if check_all_constraints(new_circles, width, height):
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
                    recent_improvements = 0  # Reset counter when we improve
                else:
                    recent_improvements += 1
        else:
            # If constraints violated, still try to accept with penalty
            # This helps escape local minima
            if np.random.rand() < 0.02:  # Lower probability to accept constraint violations
                current_circles = new_circles
                current_energy = -compute_total_radius(new_circles)
        
        # Adaptive cooling - cool faster when we're making progress, slower when stuck
        if recent_improvements > improvement_threshold:
            temp *= cooling_rate * 0.9  # Cool faster when stuck
        else:
            temp *= cooling_rate
            
        # Occasionally reset temperature to escape local minima
        if iteration % 300 == 0 and temp < 0.001:
            temp = 0.02
    
    return best_circles

def compute_total_radius(circles: np.ndarray) -> float:
    """Compute the sum of all radii"""
    return np.sum(circles[:, 2])

def check_all_constraints(circles: np.ndarray, width: float, height: float) -> bool:
    """Check if all constraints are satisfied - optimized version"""
    n = len(circles)
    
    # Check boundary constraints efficiently
    if np.any(circles[:, 0] < circles[:, 2]) or \
       np.any(circles[:, 0] > width - circles[:, 2]) or \
       np.any(circles[:, 1] < circles[:, 2]) or \
       np.any(circles[:, 1] > height - circles[:, 2]):
        return False
    
    # Check overlap constraints more efficiently using vectorized operations
    # This is more efficient than nested loops for large numbers of circles
    if n > 1:
        # Create coordinate arrays for vectorized computation
        x_coords = circles[:, 0]
        y_coords = circles[:, 1]
        r_coords = circles[:, 2]
        
        # Compute pairwise differences
        diff_x = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
        diff_y = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
        sum_radii = r_coords[:, np.newaxis] + r_coords[np.newaxis, :]
        
        # Compute squared distances and compare with squared sum of radii
        dist_sq = diff_x**2 + diff_y**2
        # Create mask for pairs that are not the same circle
        mask = ~np.eye(n, dtype=bool)
        
        # Check if any pair violates the non-overlap constraint
        if np.any(dist_sq[mask] < sum_radii[mask]**2):
            return False
    
    return True

def optimize_circle_positions(circles: np.ndarray, width: float, height: float) -> np.ndarray:
    """Optimize circle positions using constrained optimization with better error handling"""
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
        
        # Non-overlap constraints - more efficient vectorized version
        if n > 1:
            x_coords = reconstructed[:, 0]
            y_coords = reconstructed[:, 1]
            r_coords = reconstructed[:, 2]
            
            # Compute pairwise differences
            diff_x = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
            diff_y = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
            sum_radii = r_coords[:, np.newaxis] + r_coords[np.newaxis, :]
            
            # Compute squared distances and compare with squared sum of radii
            dist_sq = diff_x**2 + diff_y**2
            # Create mask for pairs that are not the same circle
            mask = ~np.eye(n, dtype=bool)
            
            # Add overlap constraints (dist^2 >= (r1 + r2)^2)
            overlap_constraints = dist_sq[mask] - sum_radii[mask]**2
            constraints.extend(overlap_constraints.tolist())
        
        return np.array(constraints)
    
    # Create constraints dictionary
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Optimize with more robust settings
    try:
        result = minimize(objective, initial_params, method='SLSQP', constraints=cons, 
                         options={'maxiter': 1000, 'ftol': 1e-6, 'eps': 1e-6})
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
    for _ in range(30):  # More refinement passes for better results
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
                    if dist > 0:
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
    """Ensure all constraints are satisfied with more aggressive correction"""
    corrected_circles = circles.copy()
    
    # First, handle boundary violations
    for i in range(len(corrected_circles)):
        x, y, r = corrected_circles[i]
        # Correct positions that violate boundaries
        corrected_circles[i, 0] = max(r, min(width - r, x))
        corrected_circles[i, 1] = max(r, min(height - r, y))
    
    # Then resolve overlaps through iterative correction
    for iteration in range(max_iterations):
        # Find all overlaps efficiently
        overlaps = []
        n = len(corrected_circles)
        
        if n > 1:
            # Vectorized overlap detection
            x_coords = corrected_circles[:, 0]
            y_coords = corrected_circles[:, 1]
            r_coords = corrected_circles[:, 2]
            
            # Compute pairwise differences
            diff_x = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
            diff_y = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
            sum_radii = r_coords[:, np.newaxis] + r_coords[np.newaxis, :]
            
            # Compute squared distances and compare with squared sum of radii
            dist_sq = diff_x**2 + diff_y**2
            # Create mask for pairs that are not the same circle
            mask = ~np.eye(n, dtype=bool)
            
            # Find all overlapping pairs
            overlap_mask = dist_sq[mask] < sum_radii[mask]**2
            if np.any(overlap_mask):
                # Extract indices of overlapping pairs
                overlap_indices = np.where(overlap_mask)[0]
                overlap_pairs = []
                for idx in overlap_indices:
                    # Convert flat index back to 2D indices
                    i = idx // (n - 1)
                    j = idx % (n - 1)
                    if j >= i:
                        j += 1
                    overlap_pairs.append((i, j))
                
                # Process overlaps in order of severity
                overlap_pairs_with_severity = []
                for i, j in overlap_pairs:
                    x1, y1, r1 = corrected_circles[i]
                    x2, y2, r2 = corrected_circles[j]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    severity = (r1 + r2) - dist  # How much they overlap
                    overlap_pairs_with_severity.append((i, j, severity))
                
                overlap_pairs_with_severity.sort(key=lambda x: x[2], reverse=True)
                
                # Resolve most severe overlaps first
                for i, j, _ in overlap_pairs_with_severity[:3]:  # Resolve top 3 overlaps per iteration
                    # Push circles apart along the line connecting centers
                    x1, y1, r1 = corrected_circles[i]
                    x2, y2, r2 = corrected_circles[j]
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance > 0.001:  # Avoid division by zero
                        push_amount = (r1 + r2 - distance) / 2
                        
                        # Apply more aggressive pushing for better results
                        push_amount *= 1.8
                        
                        dx_norm = dx / distance
                        dy_norm = dy / distance
                        
                        # Move both circles away from each other
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
    
    # Simple grid approach with better spacing and consideration of rectangle dimensions
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
            # Use a slightly larger radius to get better baseline performance
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
