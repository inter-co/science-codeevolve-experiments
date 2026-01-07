# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import math
import random
from typing import Tuple
from itertools import combinations
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a novel approach based on discrete differential geometry and symmetry exploitation.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    np.random.seed(42)  # For reproducibility
    random.seed(42)
    
    # Core concept: Exploit symmetry and discrete curvature properties
    # We'll construct a configuration that approximates the optimal using:
    # 1. Fundamental domain construction with rotational symmetry
    # 2. Discrete curvature-based circle placement
    # 3. Optimization through constrained geometric flow
    
    def create_symmetric_initial_config():
        """
        Create an initial configuration using symmetry principles.
        This uses a combination of regular hexagonal tiling with symmetry breaking.
        """
        n = 32
        circles = np.zeros((n, 3))
        
        # Group circles into clusters that respect symmetry
        # We'll use 4 quadrants with different symmetries
        quadrant_size = n // 4
        remainder = n % 4
        
        # Define four symmetric regions
        regions = [
            [(0, 0.5), (0.5, 1)],   # Bottom-left
            [(0.5, 1), (0, 0.5)],   # Top-right
            [(0, 0.5), (0, 0.5)],   # Bottom-right
            [(0.5, 1), (0.5, 1)]    # Top-left
        ]
        
        # Generate a pattern that exploits the discrete curvature
        # of the optimal packing using a Voronoi-inspired approach
        
        # Generate points in a structured way to approximate good packing
        points = []
        radii = []
        
        # Start with a central region and expand outward with symmetry
        center_points = [
            (0.25, 0.25), (0.75, 0.25), 
            (0.25, 0.75), (0.75, 0.75),
            (0.5, 0.5)  # Central point
        ]
        
        # Add additional points around each center with careful spacing
        for i, (cx, cy) in enumerate(center_points):
            if i < 4:  # Corner centers
                # Place 2-3 points around each corner
                for j in range(2 + (i % 2)):
                    angle = j * np.pi/3 + i * np.pi/2
                    dist = 0.1 + 0.05 * j
                    px = cx + dist * np.cos(angle)
                    py = cy + dist * np.sin(angle)
                    if 0 <= px <= 1 and 0 <= py <= 1:
                        points.append([px, py])
            elif i == 4:  # Center point
                # Place 10 points around the center in spiral pattern
                for j in range(10):
                    angle = j * 0.5
                    dist = 0.05 + 0.05 * j / 10
                    px = cx + dist * np.cos(angle)
                    py = cy + dist * np.sin(angle)
                    if 0 <= px <= 1 and 0 <= py <= 1:
                        points.append([px, py])
        
        # Fill remaining spots with a systematic approach
        while len(points) < n:
            # Try to place points in a way that maximizes minimum distances
            candidate_x = random.uniform(0.05, 0.95)
            candidate_y = random.uniform(0.05, 0.95)
            
            # Check if this point is far enough from existing points
            valid = True
            for px, py in points:
                dist = math.sqrt((candidate_x - px)**2 + (candidate_y - py)**2)
                if dist < 0.1:  # Minimum distance threshold
                    valid = False
                    break
            
            if valid:
                points.append([candidate_x, candidate_y])
        
        points = points[:n]
        
        # Calculate initial radii based on proximity to neighbors
        points_array = np.array(points)
        distances = cdist(points_array, points_array)
        
        for i in range(n):
            # Find minimum distance to other points (excluding self)
            dists = distances[i]
            dists[i] = np.inf  # Exclude self-distance
            
            # Minimum distance to any other point
            min_dist = np.min(dists)
            
            # Maximum possible radius is half the minimum distance
            max_radius = min_dist / 2.0
            
            # Ensure we don't exceed bounds
            x, y = points[i]
            max_radius = min(max_radius, x, 1-x, y, 1-y)
            
            # Use a reasonable fraction of maximum as initial value
            radii.append(max(0.001, min(0.1, max_radius * 0.8)))
        
        # Create final circles array
        for i in range(n):
            circles[i] = [points[i][0], points[i][1], radii[i]]
            
        return circles
    
    def discrete_curvature_optimization(circles):
        """
        Apply a discrete curvature-based optimization that respects geometric constraints.
        This mimics a geometric flow where each circle adjusts based on local curvature.
        """
        n = len(circles)
        max_iter = 1000
        tolerance = 1e-6
        
        # Convert to arrays for easier manipulation
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()
        
        for iteration in range(max_iter):
            # Calculate forces between circles
            forces = np.zeros_like(positions)
            
            # Compute pairwise interactions
            for i in range(n):
                for j in range(i+1, n):
                    pos_i = positions[i]
                    pos_j = positions[j]
                    r_i = radii[i]
                    r_j = radii[j]
                    
                    # Vector from i to j
                    dx = pos_j[0] - pos_i[0]
                    dy = pos_j[1] - pos_i[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    # If circles overlap, apply repulsive force
                    if dist < (r_i + r_j):
                        # Force magnitude inversely proportional to distance
                        force_magnitude = (r_i + r_j - dist) / (dist + 1e-10)
                        
                        # Normalize direction
                        if dist > 1e-10:
                            dx /= dist
                            dy /= dist
                        
                        # Apply force to both circles (equal and opposite)
                        forces[i, 0] += force_magnitude * dx
                        forces[i, 1] += force_magnitude * dy
                        forces[j, 0] -= force_magnitude * dx
                        forces[j, 1] -= force_magnitude * dy
            
            # Apply boundary constraints (reflective boundaries)
            for i in range(n):
                # Handle boundary collisions
                if positions[i, 0] - radii[i] < 0:
                    forces[i, 0] = abs(forces[i, 0])  # Push away from left edge
                if positions[i, 0] + radii[i] > 1:
                    forces[i, 0] = -abs(forces[i, 0])  # Push away from right edge
                if positions[i, 1] - radii[i] < 0:
                    forces[i, 1] = abs(forces[i, 1])  # Push away from bottom edge
                if positions[i, 1] + radii[i] > 1:
                    forces[i, 1] = -abs(forces[i, 1])  # Push away from top edge
            
            # Update positions
            step_size = 0.01
            positions += step_size * forces
            
            # Ensure circles stay within bounds and maintain valid radii
            for i in range(n):
                # Keep within unit square
                positions[i, 0] = np.clip(positions[i, 0], radii[i], 1 - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], 1 - radii[i])
                
                # Ensure positive radii
                radii[i] = max(radii[i], 0.001)
            
            # Check convergence
            if np.all(np.abs(forces) < tolerance):
                break
        
        # Final refinement using a more precise optimization
        # This is a simplified version of what would be done with more advanced methods
        return np.column_stack([positions, radii])
    
    def optimize_circles(circles):
        """
        Perform a local optimization to improve the configuration.
        """
        n = len(circles)
        positions = circles[:, :2].copy()
        radii = circles[:, 2].copy()
        
        # Simple gradient ascent on sum of radii with constraints
        # We'll use a projected gradient approach
        
        learning_rate = 0.001
        max_iterations = 500
        
        for _ in range(max_iterations):
            # Calculate gradients (simplified approach)
            # In practice, this would involve computing the gradient of the objective
            # with respect to positions and radii, subject to constraints
            
            # For now, let's do a simple heuristic refinement:
            
            # 1. Try to increase radii where possible without violating constraints
            for i in range(n):
                # Compute the minimum distance to any other circle
                min_dist = float('inf')
                for j in range(n):
                    if i != j:
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        min_dist = min(min_dist, dist)
                
                # Maximum radius without overlapping others
                max_radius = min_dist / 2.0
                
                # Maximum radius due to boundaries
                bound_radius = min(
                    positions[i, 0], 1 - positions[i, 0],
                    positions[i, 1], 1 - positions[i, 1]
                )
                
                # New radius is limited by both constraints
                new_radius = min(max_radius, bound_radius)
                
                # Increase radius gradually (avoid large jumps)
                radii[i] = min(radii[i] + 0.001, new_radius)
            
            # 2. Move positions to improve packing
            for i in range(n):
                # Try to move to a better position
                old_pos = positions[i].copy()
                best_pos = old_pos.copy()
                best_radius = radii[i]
                
                # Test small movements in different directions
                for dx in [-0.01, 0, 0.01]:
                    for dy in [-0.01, 0, 0.01]:
                        new_x = old_pos[0] + dx
                        new_y = old_pos[1] + dy
                        
                        # Check if within bounds
                        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                            # Check if this improves the configuration
                            temp_radii = radii.copy()
                            temp_positions = positions.copy()
                            temp_positions[i] = [new_x, new_y]
                            
                            # Recalculate radii for improved packing
                            new_radius = calculate_new_radius(temp_positions, temp_radii, i)
                            if new_radius > radii[i]:
                                best_pos = [new_x, new_y]
                                best_radius = new_radius
                
                positions[i] = best_pos
                radii[i] = best_radius
        
        return np.column_stack([positions, radii])
    
    def calculate_new_radius(positions, radii, index):
        """Calculate the maximum possible radius for a given position."""
        n = len(positions)
        min_dist = float('inf')
        
        for i in range(n):
            if i != index:
                dx = positions[index][0] - positions[i][0]
                dy = positions[index][1] - positions[i][1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = min(min_dist, dist)
        
        # Maximum radius without overlapping others
        max_radius = min_dist / 2.0
        
        # Maximum radius due to boundaries
        bound_radius = min(
            positions[index][0], 1 - positions[index][0],
            positions[index][1], 1 - positions[index][1]
        )
        
        return min(max_radius, bound_radius)
    
    def validate_solution(circles):
        """Validate that solution meets all constraints"""
        n = len(circles)
        
        # Check containment
        for i in range(n):
            x, y, r = circles[i]
            if r > x or r > (1-x) or r > y or r > (1-y):
                return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist < (r1+r2):
                    return False
        return True
    
    # Main algorithm: Create initial configuration and refine it
    start_time = time.time()
    
    # Step 1: Create symmetric initial configuration
    circles = create_symmetric_initial_config()
    
    # Step 2: Apply discrete curvature optimization
    circles = discrete_curvature_optimization(circles)
    
    # Step 3: Local optimization
    circles = optimize_circles(circles)
    
    # Step 4: Final validation and refinement
    if not validate_solution(circles):
        # If validation fails, try a more conservative approach
        circles = create_symmetric_initial_config()
    
    # Final check to ensure all constraints are met
    if not validate_solution(circles):
        # Last resort: use a grid-based approach with careful radius adjustment
        n = 32
        circles = np.zeros((n, 3))
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 1.0 / (grid_size + 1)
        radius = spacing / 2.0
        
        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= n:
                    break
                x = (j + 1) * spacing
                y = (i + 1) * spacing
                circles[idx] = [x, y, radius]
                idx += 1
        
        # Adjust to maximize sum
        for i in range(n):
            circles[i, 2] = min(circles[i, 2], 
                              circles[i, 0], 1-circles[i, 0],
                              circles[i, 1], 1-circles[i, 1])
    
    return circles


# EVOLVE-BLOCK-END
