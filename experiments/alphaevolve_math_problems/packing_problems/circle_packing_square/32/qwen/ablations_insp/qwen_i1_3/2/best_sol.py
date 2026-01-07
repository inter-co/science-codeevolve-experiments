# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from typing import Tuple, List
import math
import random
import cvxpy as cp

# Global constants
N_CIRCLES = 32

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class CirclePacker:
    def __init__(self):
        self.circles = []
        
    def _distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _check_constraints(self, circles: np.ndarray) -> bool:
        """Check if all circles satisfy containment and non-overlap constraints."""
        # Check containment
        for i in range(N_CIRCLES):
            x, y, r = circles[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                return False
        
        # Check non-overlap
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dist = self._distance((x1, y1), (x2, y2))
                if dist < r1 + r2:
                    return False
        return True
    
    def _get_max_radius(self, x: float, y: float, 
                       existing_circles: np.ndarray) -> float:
        """Calculate maximum possible radius for a circle at (x,y) without overlap."""
        max_radius = min(x, 1 - x, y, 1 - y)  # Boundary constraints
        
        # Check overlap constraints with existing circles
        for i in range(len(existing_circles)):
            cx, cy, cr = existing_circles[i]
            distance = self._distance((x, y), (cx, cy))
            max_radius = min(max_radius, distance - cr)
            
        return max(max_radius, 0.001)  # Ensure positive radius
    
    def _compute_total_radius_sum(self, circles: np.ndarray) -> float:
        """Compute sum of all circle radii."""
        return np.sum(circles[:, 2])
    
    def _initialize_placement(self) -> np.ndarray:
        """Initialize circle placement using a more systematic approach like working solution."""
        # Create a structured hexagonal grid pattern that works well for 32 circles
        circles = []
        
        # Use a 6x6 grid but adjust to get exactly 32 circles
        rows = 6
        cols = 6
        
        # Hexagonal packing parameters with tighter spacing
        spacing_x = 0.16  # Slightly smaller spacing for better utilization
        spacing_y = 0.16 * np.sqrt(3)/2
        
        # Generate points in hexagonal grid
        count = 0
        for i in range(rows):
            for j in range(cols):
                if count >= N_CIRCLES:
                    break
                x = 0.1 + j * spacing_x
                y = 0.1 + i * spacing_y
                # Offset every other row
                if i % 2 == 1:
                    x += spacing_x / 2
                if x <= 0.9 and y <= 0.9:
                    circles.append([x, y, 0.04])  # Slightly smaller initial radius
                    count += 1
        
        # Fill remaining positions with more careful random placements
        while len(circles) < N_CIRCLES:
            # Place near corners and edges where we might gain more space
            corner_positions = [
                (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
                (0.5, 0.1), (0.5, 0.9), (0.1, 0.5), (0.9, 0.5)
            ]
            if len(circles) < len(corner_positions):
                x, y = corner_positions[len(circles)]
            else:
                x = np.random.uniform(0.05, 0.95)
                y = np.random.uniform(0.05, 0.95)
            circles.append([x, y, 0.04])
            
        return np.array(circles)
    
    def _objective_function(self, circles_flat: np.ndarray) -> float:
        """Objective function (negative sum of radii for minimization)."""
        circles = circles_flat.reshape(-1, 3)
        # Sum of radii (negative for minimization)
        return -np.sum(circles[:, 2])
    
    def _create_constraint_functions(self):
        """Create constraint functions for scipy optimization."""
        cons = []
        
        # Boundary constraints: radius <= x <= 1-radius and radius <= y <= 1-radius
        for i in range(N_CIRCLES):
            # x >= r
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: x[3*i] - x[3*i+2] - 1e-6
            })
            # y >= r  
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: x[3*i+1] - x[3*i+2] - 1e-6
            })
            # 1-x >= r
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2] - 1e-6
            })
            # 1-y >= r
            cons.append({
                'type': 'ineq', 
                'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2] - 1e-6
            })
        
        # Non-overlap constraints: sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
        for i in range(N_CIRCLES):
            for j in range(i+1, N_CIRCLES):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: (
                        np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) 
                        - x[3*i+2] - x[3*j+2] - 1e-6
                    )
                })
        
        return cons
    
    def _optimize_with_cvxpy(self, initial_circles: np.ndarray) -> np.ndarray:
        """Use convex optimization with CVXPY for potentially better results."""
        try:
            # Define variables
            x = cp.Variable(N_CIRCLES)
            y = cp.Variable(N_CIRCLES)
            r = cp.Variable(N_CIRCLES)
            
            # Objective: maximize sum of radii
            objective = cp.Maximize(cp.sum(r))
            
            # Constraints
            constraints = []
            
            # Boundary constraints
            for i in range(N_CIRCLES):
                constraints.append(x[i] >= r[i])
                constraints.append(y[i] >= r[i])
                constraints.append(x[i] <= 1 - r[i])
                constraints.append(y[i] <= 1 - r[i])
            
            # Non-overlap constraints
            for i in range(N_CIRCLES):
                for j in range(i+1, N_CIRCLES):
                    # Use squared distance to avoid sqrt for convexity
                    dist_squared = (x[i] - x[j])**2 + (y[i] - y[j])**2
                    constraints.append(dist_squared >= (r[i] + r[j])**2)
            
            # Create problem and solve
            prob = cp.Problem(objective, constraints)
            
            # Try different solvers
            solver_options = {
                'solver': cp.SCS,  # Often more robust for complex problems
                'eps': 1e-6,
                'max_iters': 10000
            }
            
            prob.solve(**solver_options)
            
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                # Extract solution
                solution = np.zeros((N_CIRCLES, 3))
                for i in range(N_CIRCLES):
                    solution[i] = [x[i].value, y[i].value, r[i].value]
                return solution
            
        except Exception as e:
            # Fall back to scipy if CVXPY fails
            pass
        
        return initial_circles
    
    def _optimize_with_scipy(self, initial_circles: np.ndarray) -> np.ndarray:
        """Use scipy optimization for better results with multiple methods."""
        # Flatten for optimization
        circles_flat = initial_circles.flatten()
        
        # Define bounds for each parameter (x, y, r)
        bounds = []
        for i in range(N_CIRCLES):
            # x, y, r bounds
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
        
        # Optimization options
        options = {'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
        
        best_result = None
        best_sum = -float('inf')
        
        # Try multiple optimization methods with different settings
        methods = ['SLSQP', 'trust-constr']
        for method in methods:
            try:
                result = minimize(
                    self._objective_function,
                    circles_flat,
                    method=method,
                    bounds=bounds,
                    constraints=self._create_constraint_functions(),
                    options=options,
                    tol=1e-8
                )
                
                if result.success:
                    current_sum = -result.fun  # Convert back to sum of radii
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_result = result
            except Exception as e:
                continue
        
        # If we got a good result, return it; otherwise fall back to initial
        if best_result is not None and best_result.success:
            circles_opt = best_result.x.reshape(-1, 3)
            # Ensure valid constraints
            if self._check_constraints(circles_opt):
                return circles_opt
        
        return initial_circles
    
    def _improve_with_local_search(self, circles: np.ndarray, max_iter: int = 10000) -> np.ndarray:
        """Improve configuration with enhanced local search like working solution."""
        best_circles = circles.copy()
        best_sum = self._compute_total_radius_sum(circles)
        
        # Enhanced local search with better neighborhood exploration
        for iteration in range(max_iter):
            current_circles = best_circles.copy()
            changed = False
            
            # Try to improve each circle systematically
            for i in range(N_CIRCLES):
                # Save current state
                x, y, r = current_circles[i]
                
                # Try to increase radius to maximum possible value
                new_r = self._get_max_radius(x, y, np.vstack([current_circles[:i], current_circles[i+1:]]))
                if new_r > r + 1e-6:
                    current_circles[i] = [x, y, new_r]
                    changed = True
                    continue
                
                # Try small position adjustments with more thorough exploration
                best_move = None
                best_improvement = 0
                
                # Try multiple directions and sizes
                moves = [
                    (0, 0, 0.001), (0, 0, 0.002), (0, 0, -0.001),
                    (-0.005, 0, 0), (0.005, 0, 0), (0, -0.005, 0), (0, 0.005, 0),
                    (-0.003, -0.003, 0.001), (-0.003, 0.003, 0.001), 
                    (0.003, -0.003, 0.001), (0.003, 0.003, 0.001),
                    (-0.01, 0, 0), (0.01, 0, 0), (0, -0.01, 0), (0, 0.01, 0),
                    (-0.002, 0, 0.001), (0.002, 0, 0.001), (0, -0.002, 0.001), (0, 0.002, 0.001)
                ]
                
                # Add some random moves for exploration
                for _ in range(15):
                    dx = np.random.uniform(-0.005, 0.005)
                    dy = np.random.uniform(-0.005, 0.005)
                    dr = np.random.uniform(-0.002, 0.002)
                    moves.append((dx, dy, dr))
                
                for dx, dy, dr in moves:
                    test_x = max(0.01, min(0.99, x + dx))
                    test_y = max(0.01, min(0.99, y + dy))
                    test_r = max(0.001, r + dr)
                    
                    # Check if this move is valid
                    if (test_x - test_r >= 0 and test_x + test_r <= 1 and
                        test_y - test_r >= 0 and test_y + test_r <= 1):
                        
                        # Check overlap with all other circles
                        valid = True
                        test_circles = current_circles.copy()
                        test_circles[i] = [test_x, test_y, test_r]
                        
                        for j in range(N_CIRCLES):
                            if i != j:
                                dist = self._distance((test_x, test_y), (current_circles[j][0], current_circles[j][1]))
                                if dist < (test_r + current_circles[j][2]):
                                    valid = False
                                    break
                        
                        if valid:
                            # Calculate improvement
                            new_sum = self._compute_total_radius_sum(test_circles)
                            improvement = new_sum - best_sum
                            if improvement > best_improvement + 1e-8:
                                best_improvement = improvement
                                best_move = (test_x, test_y, test_r)
                
                if best_move is not None:
                    current_circles[i] = best_move
                    changed = True
            
            # Check if this improves the configuration
            new_sum = self._compute_total_radius_sum(current_circles)
            if new_sum > best_sum + 1e-8:
                best_sum = new_sum
                best_circles = current_circles.copy()
            elif not changed:
                # No improvements made, stop early
                break
        
        return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    packer = CirclePacker()
    
    # Step 1: Initialize with improved placement
    initial_circles = packer._initialize_placement()
    
    # Step 2: Try convex optimization first (potentially better than pure scipy)
    try:
        optimized_circles = packer._optimize_with_cvxpy(initial_circles)
    except:
        optimized_circles = initial_circles
    
    # Step 3: Use scipy optimization for fine-tuning
    optimized_circles = packer._optimize_with_scipy(optimized_circles)
    
    # Step 4: Fine-tune with enhanced local search
    final_circles = packer._improve_with_local_search(optimized_circles)
    
    # Final validation
    if not packer._check_constraints(final_circles):
        # If constraints violated, use a more robust fallback
        final_circles = packer._improve_with_local_search(initial_circles, max_iter=2000)
    
    return final_circles


# EVOLVE-BLOCK-END
