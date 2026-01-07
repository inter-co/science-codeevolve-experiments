# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import random
import time

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal initialization with aggressive optimization.
    
    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    n = 32
    start_time = time.time()
    
    # Hexagonal grid initialization
    def initialize_hexagonal(n: int) -> np.ndarray:
        """Initialize circles using a hexagonal grid pattern."""
        # Create hexagonal grid points
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))
        
        points = []
        for i in range(rows):
            for j in range(cols):
                if len(points) >= n:
                    break
                # Hexagonal offset for alternating rows
                x = (j + 0.5 + (i % 2) * 0.5) / cols
                y = (i + 0.5) / rows
                # Add slight randomness to break symmetry
                x += random.uniform(-0.005, 0.005)
                y += random.uniform(-0.005, 0.005)
                points.append([x, y])
        
        points = points[:n]
        
        # Initialize with appropriate radii based on proximity to boundaries
        radii = []
        for i, (x, y) in enumerate(points):
            # Base radius based on distance to boundaries
            edge_dist = min(x, 1-x, y, 1-y)
            base_radius = min(0.05, edge_dist * 0.8)
            # Add some randomness to prevent perfect symmetry
            radius = max(0.01, base_radius * (0.8 + 0.4 * random.random()))
            radii.append(radius)
        
        circles = np.column_stack([points, radii])
        
        # Apply repulsion to improve initial distribution
        for _ in range(200):
            for i in range(n):
                for j in range(i+1, n):
                    x1, y1, r1 = circles[i]
                    x2, y2, r2 = circles[j]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = max(np.sqrt(dx*dx + dy*dy), 1e-8)
                    
                    if dist < 0.15:
                        # Repulsion force
                        force = 0.0015 * (0.15 - dist) / (dist + 1e-8)
                        circles[i, 0] += force * dx
                        circles[i, 1] += force * dy
                        circles[j, 0] -= force * dx
                        circles[j, 1] -= force * dy
                        
                        # Keep within bounds
                        circles[i, 0] = np.clip(circles[i, 0], r1, 1-r1)
                        circles[i, 1] = np.clip(circles[i, 1], r1, 1-r1)
                        circles[j, 0] = np.clip(circles[j, 0], r2, 1-r2)
                        circles[j, 1] = np.clip(circles[j, 1], r2, 1-r2)
        
        # Final boundary correction
        for i in range(n):
            x, y, r = circles[i]
            circles[i] = [np.clip(x, r, 1-r), np.clip(y, r, 1-r), r]
        
        return circles
    
    # Fast constraint checking with early termination
    def fast_constraint_check(circles: np.ndarray) -> tuple[bool, float]:
        """Fast constraint validation."""
        # Check containment
        positions = circles[:, :2]
        radii = circles[:, 2]
        
        # Vectorized containment check
        containment_ok = (
            (positions[:, 0] - radii >= 0) &
            (positions[:, 0] + radii <= 1) &
            (positions[:, 1] - radii >= 0) &
            (positions[:, 1] + radii <= 1)
        )
        
        if not np.all(containment_ok):
            return False, 0.0
        
        # Check overlaps using spatial indexing
        tree = cKDTree(positions)
        pairs = tree.query_pairs(2 * np.max(radii), p=2)
        
        for i, j in pairs:
            if i < j:
                dist = np.linalg.norm(positions[i] - positions[j])
                r_i, r_j = radii[i], radii[j]
                if dist < r_i + r_j:
                    return False, np.sum(radii)
        
        return True, np.sum(radii)
    
    # Aggressive optimization with multiple strategies
    def aggressive_optimization(initial_circles: np.ndarray) -> np.ndarray:
        """Aggressive optimization with multiple refinement strategies."""
        current = initial_circles.copy()
        best_circles = current.copy()
        best_sum = np.sum(current[:, 2])
        
        # Track improvement for early stopping
        last_improvement = 0
        patience = 50
        
        # Main optimization loop
        for iteration in range(500):  # More iterations for better convergence
            if time.time() - start_time > 55:  # Time limit
                break
                
            improved = False
            
            # Strategy 1: Try to expand radii aggressively
            for i in range(n):
                x, y, r = current[i]
                max_radius = min(x, 1-x, y, 1-y)
                
                # Try larger steps first for faster progress
                steps = [0.02, 0.01, 0.005, 0.002, 0.001]
                for step in steps:
                    if step > max_radius - r:
                        continue
                    new_r = min(r + step, max_radius)
                    if new_r > r + 1e-6:
                        test_circles = current.copy()
                        test_circles[i, 2] = new_r
                        
                        valid, _ = fast_constraint_check(test_circles)
                        if valid:
                            current = test_circles
                            improved = True
                            break
            
            # Strategy 2: Position optimization with diverse moves
            for i in range(n):
                x, y, r = current[i]
                old_pos = [x, y]
                
                # Try many different moves to find improvements
                moves = [
                    # Small moves
                    (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01),
                    # Diagonal moves
                    (0.005, 0.005), (-0.005, -0.005), (0.005, -0.005), (-0.005, 0.005),
                    # Medium moves
                    (0.02, 0), (0, 0.02), (-0.02, 0), (0, -0.02),
                    # Larger moves for escape
                    (0.05, 0), (0, 0.05), (-0.05, 0), (0, -0.05),
                    # Very small moves for fine-tuning
                    (0.001, 0.001), (-0.001, -0.001)
                ]
                
                for dx, dy in moves:
                    new_x = x + dx
                    new_y = y + dy
                    
                    # Keep within bounds
                    new_x = np.clip(new_x, r, 1-r)
                    new_y = np.clip(new_y, r, 1-r)
                    
                    # Test move
                    test_circles = current.copy()
                    test_circles[i, 0] = new_x
                    test_circles[i, 1] = new_y
                    
                    valid, _ = fast_constraint_check(test_circles)
                    if valid:
                        current = test_circles
                        improved = True
                        break
            
            # Update best solution
            current_sum = np.sum(current[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = current.copy()
                last_improvement = iteration
            elif iteration - last_improvement > patience:
                # Early stopping if no improvement
                break
        
        return best_circles
    
    # Multi-start approach with different initialization strategies
    best_solution = None
    best_sum = 0
    
    # Strategy 1: Hexagonal initialization with aggressive optimization
    try:
        initial = initialize_hexagonal(n)
        optimized = aggressive_optimization(initial)
        valid, sum_radii = fast_constraint_check(optimized)
        
        if valid and sum_radii > best_sum:
            best_sum = sum_radii
            best_solution = optimized.copy()
    except Exception as e:
        pass
    
    # Strategy 2: Multiple restarts with different approaches
    for restart in range(15):
        try:
            # Random initialization with better constraints
            circles = np.zeros((n, 3))
            for i in range(n):
                x = random.uniform(0.05, 0.95)
                y = random.uniform(0.05, 0.95)
                r = random.uniform(0.02, 0.08)
                circles[i] = [x, y, r]
            
            # Apply aggressive optimization
            refined = aggressive_optimization(circles)
            valid, sum_radii = fast_constraint_check(refined)
            
            if valid and sum_radii > best_sum:
                best_sum = sum_radii
                best_solution = refined.copy()
        except Exception as e:
            continue
    
    # Fallback to a good known configuration if needed
    if best_solution is None:
        # Create a structured pattern that typically works well
        best_solution = np.zeros((n, 3))
        idx = 0
        for i in range(6):
            for j in range(6):
                if idx >= n:
                    break
                x = 0.1 + j * 0.15
                y = 0.1 + i * 0.15
                r = 0.04
                best_solution[idx] = [x, y, r]
                idx += 1
        # Fill remaining slots with smaller circles
        for i in range(idx, n):
            best_solution[i] = [0.5, 0.5, 0.02]
    
    return best_solution


# EVOLVE-BLOCK-END
