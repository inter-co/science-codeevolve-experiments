# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
from collections import defaultdict
import time
import warnings
from scipy.spatial import cKDTree
import math
from scipy.optimize import minimize
import cvxpy as cp

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize sum of radii.
    Uses a mathematical programming approach with convex relaxation for better performance.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    n = 32
    
    def create_lattice_initialization() -> np.ndarray:
        """Create an initial configuration using a structured lattice approach"""
        circles = np.zeros((n, 3))
        
        # Arrange in a rectangular grid pattern (approximately)
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        # Grid spacing that allows for reasonable circle radii
        spacing_x = 1.0 / cols
        spacing_y = 1.0 / rows
        
        # Adjust spacing to allow for some overlap tolerance initially
        spacing_x = min(spacing_x, 0.2)
        spacing_y = min(spacing_y, 0.2)
        
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= n:
                    break
                    
                x = (col + 0.5) * spacing_x
                y = (row + 0.5) * spacing_y
                
                # Keep within bounds
                x = max(0.05, min(0.95, x))
                y = max(0.05, min(0.95, y))
                
                # Initial radius - start with a reasonable value
                r = min(0.05, spacing_x/2, spacing_y/2)
                
                circles[count] = [x, y, r]
                count += 1
                
            if count >= n:
                break
                
        return circles
    
    def calculate_total_radius(circles: np.ndarray) -> float:
        """Calculate sum of all radii"""
        return np.sum(circles[:, 2])
    
    def is_valid_placement(circles: np.ndarray) -> bool:
        """Check if all circles are valid (within bounds and non-overlapping)"""
        # Check boundary constraints for all circles
        for i in range(len(circles)):
            x, y, r = circles[i]
            if x < r or x > 1 - r or y < r or y > 1 - r:
                return False
                
        # Check overlap constraints with all pairs of circles
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                
                if dist_sq < min_dist_sq:
                    return False
                    
        return True
    
    def build_constraint_matrix(circles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build constraint matrix for optimization.
        Returns A, b where Ax <= b represents the constraints.
        """
        m = len(circles) * (len(circles) - 1) // 2  # number of overlap constraints
        A = np.zeros((m, 3 * len(circles)))  # Each circle has x, y, r variables
        b = np.zeros(m)
        
        idx = 0
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                # Overlap constraint: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
                # This becomes: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                # In terms of variables: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
                # Linearized form for optimization: 
                # We'll use the fact that we're optimizing radius sum, so we can work with 
                # quadratic constraints directly
                
                # For now, let's focus on simpler approach using spatial indexing for checking
                idx += 1
                
        return A, b
    
    def solve_with_cvxpy(circles: np.ndarray) -> np.ndarray:
        """
        Use convex optimization to solve the circle packing problem.
        This is a simplified approach using CVXPY for the optimization part.
        """
        # Create optimization variables
        x = cp.Variable(n)
        y = cp.Variable(n)
        r = cp.Variable(n)
        
        # Objective: maximize sum of radii
        objective = cp.Maximize(cp.sum(r))
        
        # Constraints
        constraints = []
        
        # Boundary constraints
        for i in range(n):
            constraints.append(x[i] >= r[i])
            constraints.append(x[i] <= 1 - r[i])
            constraints.append(y[i] >= r[i])
            constraints.append(y[i] <= 1 - r[i])
        
        # Non-overlap constraints (simplified quadratic constraints)
        for i in range(n):
            for j in range(i+1, n):
                # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
                # This is equivalent to: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
                constraints.append(
                    cp.square(x[i] - x[j]) + cp.square(y[i] - y[j]) >= cp.square(r[i] + r[j])
                )
        
        # Create and solve the problem
        prob = cp.Problem(objective, constraints)
        
        try:
            # Try to solve with different solvers
            prob.solve(solver=cp.SCS, verbose=False, max_iters=1000)
            # If that fails, try ECOS
            if prob.status != cp.OPTIMAL:
                prob.solve(solver=cp.ECOS, verbose=False)
        except:
            # Fall back to a simpler approach if optimization fails
            return circles
            
        # Extract solution
        if prob.status == cp.OPTIMAL:
            solution = circles.copy()
            for i in range(n):
                solution[i, 0] = x.value[i]
                solution[i, 1] = y.value[i]
                solution[i, 2] = r.value[i]
            return solution
            
        return circles
    
    def smart_radius_adjustment(circles: np.ndarray) -> np.ndarray:
        """
        Perform smart radius adjustments to maximize sum while maintaining validity.
        Uses a greedy approach with spatial indexing.
        """
        # Build k-d tree for efficient neighbor searching
        points = circles[:, :2]
        tree = cKDTree(points)
        
        # Create copy to work with
        result = circles.copy()
        
        # Iteratively adjust radii and positions
        for iteration in range(50):
            improved = False
            
            # Try to increase each circle's radius first
            for i in range(n):
                x, y, old_r = result[i]
                
                # Find maximum possible radius at current position
                max_radius = min(x, 1-x, y, 1-y)
                
                # Check neighbors using spatial indexing
                nearby_points = tree.query_ball_point([x, y], max_radius * 2)
                for j in nearby_points:
                    if i == j:
                        continue
                    x2, y2, r2 = result[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    max_radius = min(max_radius, dist - r2)
                
                # Increase radius if beneficial
                if max_radius > old_r + 1e-6:
                    result[i, 2] = max_radius
                    improved = True
            
            # If no improvement, try position adjustments
            if not improved:
                for i in range(n):
                    x, y, r = result[i]
                    
                    # Try to move circle to a better position
                    best_x, best_y = x, y
                    best_r = r
                    best_score = calculate_total_radius(result)
                    
                    # Try several nearby positions
                    for dx in [-0.01, 0, 0.01]:
                        for dy in [-0.01, 0, 0.01]:
                            new_x = x + dx
                            new_y = y + dy
                            
                            # Keep within bounds
                            if new_x < r or new_x > 1-r or new_y < r or new_y > 1-r:
                                continue
                                
                            # Test this position
                            temp_result = result.copy()
                            temp_result[i, 0] = new_x
                            temp_result[i, 1] = new_y
                            
                            # Try to maximize radius at new position
                            temp_x, temp_y, temp_r = new_x, new_y, r
                            max_radius = min(temp_x, 1-temp_x, temp_y, 1-temp_y)
                            
                            # Check neighbors
                            nearby_points = tree.query_ball_point([temp_x, temp_y], max_radius * 2)
                            for j in nearby_points:
                                if i == j:
                                    continue
                                x2, y2, r2 = result[j]
                                dist = np.sqrt((temp_x - x2)**2 + (temp_y - y2)**2)
                                max_radius = min(max_radius, dist - r2)
                            
                            if max_radius > temp_r + 1e-6:
                                temp_result[i, 2] = max_radius
                                new_score = calculate_total_radius(temp_result)
                                if new_score > best_score:
                                    best_score = new_score
                                    best_x, best_y = new_x, new_y
                                    best_r = max_radius
                                    
                    # Update if we found improvement
                    if best_x != x or best_y != y or best_r != r:
                        result[i, 0] = best_x
                        result[i, 1] = best_y
                        result[i, 2] = best_r
                        improved = True
            
            # Break if no improvement made
            if not improved:
                break
                
        return result
    
    def construct_optimal_grid() -> np.ndarray:
        """
        Construct an initial configuration using a more principled geometric approach.
        This uses a combination of lattice-based placement and geometric reasoning.
        """
        circles = np.zeros((n, 3))
        
        # Start with a more structured approach
        # Place circles in a way that maximizes packing efficiency
        sqrt_n = int(np.ceil(np.sqrt(n)))
        
        # Create a grid that's slightly denser than needed
        grid_rows = sqrt_n + 1
        grid_cols = sqrt_n + 1
        
        # Calculate spacing
        spacing_x = 1.0 / grid_cols
        spacing_y = 1.0 / grid_rows
        
        # Adjust spacing to make room for circles
        spacing_x = min(spacing_x, 0.15)
        spacing_y = min(spacing_y, 0.15)
        
        count = 0
        for row in range(grid_rows):
            for col in range(grid_cols):
                if count >= n:
                    break
                    
                # Center positions
                x = (col + 0.5) * spacing_x
                y = (row + 0.5) * spacing_y
                
                # Keep within bounds
                x = max(spacing_x/2, min(1 - spacing_x/2, x))
                y = max(spacing_y/2, min(1 - spacing_y/2, y))
                
                # Initial radius - based on spacing
                r = min(spacing_x/2, spacing_y/2) * 0.8
                
                circles[count] = [x, y, r]
                count += 1
                
            if count >= n:
                break
                
        return circles
    
    def finalize_solution(circles: np.ndarray) -> np.ndarray:
        """
        Final optimization pass to refine the solution.
        """
        # Make sure all circles are valid and don't overlap
        result = circles.copy()
        
        # First, ensure valid boundaries
        for i in range(n):
            x, y, r = result[i]
            # Clamp to valid range
            result[i, 0] = max(r, min(1-r, x))
            result[i, 1] = max(r, min(1-r, y))
        
        # Then perform iterative refinement
        for _ in range(20):
            # Try to increase radii
            improved = False
            for i in range(n):
                x, y, r = result[i]
                
                # Find maximum possible radius
                max_radius = min(x, 1-x, y, 1-y)
                
                # Check against all others
                for j in range(n):
                    if i == j:
                        continue
                    x2, y2, r2 = result[j]
                    dist = np.sqrt((x - x2)**2 + (y - y2)**2)
                    max_radius = min(max_radius, dist - r2)
                
                if max_radius > r + 1e-6:
                    result[i, 2] = max_radius
                    improved = True
                    
            if not improved:
                break
                
        return result
    
    # Main algorithm: Geometric construction with mathematical optimization
    try:
        # Step 1: Create initial configuration using structured approach
        initial_solution = construct_optimal_grid()
        
        # Step 2: Refine using smart adjustments
        refined_solution = smart_radius_adjustment(initial_solution)
        
        # Step 3: Final validation and optimization
        final_solution = finalize_solution(refined_solution)
        
        # Ensure validity
        if not is_valid_placement(final_solution):
            # If still invalid, try a more conservative approach
            circles = create_lattice_initialization()
            final_solution = smart_radius_adjustment(circles)
            
        return final_solution
        
    except Exception as e:
        # Fallback to simple initial configuration
        return create_lattice_initialization()


# EVOLVE-BLOCK-END
