# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple
import random
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining geometric insights and efficient optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Use a better initial layout based on hexagonal packing principles
    circles = _hexagonal_initial_layout(n)
    
    # Strategy 1: Multi-start optimization with better constraints
    best_sum = -float('inf')
    best_circles = circles.copy()
    
    # Run with multiple random restarts to avoid local minima
    start_time = time.time()
    for attempt in range(15):  # More attempts for better chance of finding good solution
        if time.time() - start_time > 55:  # Leave some time for final processing
            break
            
        try:
            # Slightly perturb the initial solution for diversity
            perturbed_circles = circles.copy()
            for i in range(n):
                # Add small random perturbation to x, y, r
                perturbed_circles[i, 0] += np.random.uniform(-0.015, 0.015)
                perturbed_circles[i, 1] += np.random.uniform(-0.015, 0.015)
                perturbed_circles[i, 2] += np.random.uniform(-0.008, 0.008)
                
                # Ensure bounds
                perturbed_circles[i, 0] = np.clip(perturbed_circles[i, 0], 0.001, 0.999)
                perturbed_circles[i, 1] = np.clip(perturbed_circles[i, 1], 0.001, 0.999)
                perturbed_circles[i, 2] = np.clip(perturbed_circles[i, 2], 0.001, 0.5)
            
            optimized = _optimize_with_constraints(perturbed_circles)
            
            current_sum = np.sum(optimized[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_circles = optimized.copy()
                
        except Exception as e:
            continue
    
    # Strategy 2: Improved evolutionary approach with better selection and more sophisticated operators
    if time.time() - start_time < 50:
        try:
            evol_result = _enhanced_evolutionary_approach(n)
            evol_sum = np.sum(evol_result[:, 2])
            if evol_sum > best_sum:
                best_sum = evol_sum
                best_circles = evol_result
        except Exception:
            pass
    
    # Strategy 3: Local search refinement with better neighborhood exploration
    if time.time() - start_time < 58:
        try:
            refined = _improved_local_search(best_circles)
        except Exception:
            refined = best_circles
    
    return refined if 'refined' in locals() else best_circles

def _hexagonal_initial_layout(n: int) -> np.ndarray:
    """Create a better initial layout inspired by hexagonal packing with systematic approach"""
    circles = np.zeros((n, 3))
    
    # Create a more structured hexagonal pattern
    # For 26 circles, we can arrange them in roughly 5 rows with alternating offset
    rows = 5
    cols = 5
    
    # Calculate spacing based on circle count
    spacing_x = 0.85 / (cols - 1) if cols > 1 else 0.5
    spacing_y = 0.85 / (rows - 1) if rows > 1 else 0.5
    
    # Create hexagonal pattern with offset rows
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
                
            # Offset every other row for better packing
            x_offset = spacing_x * 0.5 if i % 2 == 1 else 0.0
            x = 0.075 + j * spacing_x + x_offset
            y = 0.075 + i * spacing_y
            
            # Keep within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            # Initial radius based on distance to boundaries and position
            min_edge_dist = min(x, 1-x, y, 1-y)
            base_radius = min_edge_dist * 0.3
            
            # Adjust based on position - center circles get larger radii
            center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            if center_dist < 0.25:
                radius = min(base_radius, 0.18)
            elif center_dist < 0.5:
                radius = min(base_radius, 0.15)
            else:
                radius = min(base_radius, 0.12)
            
            # Ensure reasonable minimum
            radius = max(radius, 0.015)
            
            circles[idx] = [x, y, radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining positions strategically
    for i in range(idx, n):
        # Place in corners and edge regions with strategic positioning
        if i % 7 == 0:
            # Bottom-left corner
            x = np.random.uniform(0.05, 0.12)
            y = np.random.uniform(0.05, 0.12)
        elif i % 7 == 1:
            # Top-right corner
            x = np.random.uniform(0.88, 0.95)
            y = np.random.uniform(0.88, 0.95)
        elif i % 7 == 2:
            # Top-left corner
            x = np.random.uniform(0.05, 0.12)
            y = np.random.uniform(0.88, 0.95)
        elif i % 7 == 3:
            # Bottom-right corner
            x = np.random.uniform(0.88, 0.95)
            y = np.random.uniform(0.05, 0.12)
        elif i % 7 == 4:
            # Center-top
            x = np.random.uniform(0.4, 0.6)
            y = np.random.uniform(0.85, 0.92)
        elif i % 7 == 5:
            # Center-bottom
            x = np.random.uniform(0.4, 0.6)
            y = np.random.uniform(0.05, 0.12)
        else:
            # Edge-center
            if random.random() < 0.5:
                x = np.random.uniform(0.05, 0.12)
                y = np.random.uniform(0.3, 0.7)
            else:
                x = np.random.uniform(0.88, 0.95)
                y = np.random.uniform(0.3, 0.7)
            
        # Radius based on proximity to edges and center
        min_edge_dist = min(x, 1-x, y, 1-y)
        radius = min(min_edge_dist * 0.4, 0.15)
        radius = max(radius, 0.02)
        
        circles[i] = [x, y, radius]
    
    return circles

def _optimize_with_constraints(initial_circles: np.ndarray) -> np.ndarray:
    """Use constrained optimization with better handling of constraints and smarter approach"""
    n = len(initial_circles)
    
    # Flatten initial solution
    initial_vars = []
    for i in range(n):
        initial_vars.extend([initial_circles[i, 0], initial_circles[i, 1], initial_circles[i, 2]])
    
    # Define bounds for each variable (x, y, r)
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x coordinate
        bounds.append((0.001, 0.999))  # y coordinate
        bounds.append((0.001, 0.5))    # radius
    
    # Define objective function (negative because we want to maximize)
    def objective(vars_array):
        circles_array = vars_array.reshape(-1, 3)
        return -np.sum(circles_array[:, 2])
    
    # More efficient constraint functions
    def containment_constraint(vars_array):
        circles_array = vars_array.reshape(-1, 3)
        result = []
        for i in range(n):
            x, y, r = circles_array[i]
            # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
            result.extend([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.array(result)
    
    def overlap_constraint(vars_array):
        circles_array = vars_array.reshape(-1, 3)
        result = []
        # Vectorized computation for better performance - compute only unique pairs
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                # Distance squared minus (r1 + r2)^2 should be >= 0
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                result.append(dist_sq - (r1 + r2)**2)
        return np.array(result)
    
    # Create constraint dictionaries for scipy
    cons = [
        {'type': 'ineq', 'fun': lambda x: containment_constraint(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraint(x)}
    ]
    
    # Run optimization with better settings
    try:
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 500, 'ftol': 1e-7, 'eps': 1e-7},
            tol=1e-7
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Clip to ensure validity
            for i in range(n):
                optimized_circles[i, 0] = np.clip(optimized_circles[i, 0], 0.001, 0.999)
                optimized_circles[i, 1] = np.clip(optimized_circles[i, 1], 0.001, 0.999)
                optimized_circles[i, 2] = np.clip(optimized_circles[i, 2], 0.001, 0.5)
            return optimized_circles
    except Exception:
        pass
    
    return initial_circles

def _enhanced_evolutionary_approach(n: int) -> np.ndarray:
    """Enhanced evolutionary algorithm with better selection pressure and operators"""
    # Population size
    pop_size = 50  # Larger population for better exploration
    generations = 25  # Fewer generations but better selection
    
    # Initialize population with better starting points
    population = []
    for _ in range(pop_size):
        individual = _hexagonal_initial_layout(n)
        population.append(individual)
    
    # Evaluate fitness with more sophisticated penalty system
    def evaluate_fitness(individual):
        # Calculate sum of radii
        total_radius = np.sum(individual[:, 2])
        
        # More sophisticated penalty system
        penalty = 0
        
        # Check containment with proper bounds checking
        for i in range(n):
            x, y, r = individual[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 100000  # Much higher penalty for boundary violations
        
        # Check overlaps with more careful computation
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = individual[i]
                x2, y2, r2 = individual[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                if dist_sq < min_dist_sq:
                    # Penalty proportional to violation amount
                    violation = min_dist_sq - dist_sq
                    penalty += violation * 5000
        
        return total_radius - penalty
    
    # Evolution loop with better selection
    for gen in range(generations):
        # Evaluate all individuals
        fitness_scores = [evaluate_fitness(ind) for ind in population]
        
        # Tournament selection with better parameters
        sorted_indices = np.argsort(fitness_scores)[::-1]
        # Keep top 60% instead of 50%
        selected = [population[i] for i in sorted_indices[:int(pop_size * 0.6)]]
        
        # Create new population through crossover and mutation
        new_population = selected.copy()
        
        # Elitism - keep the best individual
        best_individual = selected[0]
        new_population.append(best_individual)
        
        # Generate offspring through better crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection with size 3
            parent1 = selected[random.randint(0, len(selected)-1)]
            parent2 = selected[random.randint(0, len(selected)-1)]
            
            # Blend crossover (more sophisticated than uniform)
            child = parent1.copy()
            alpha = random.random() * 0.5 + 0.25  # Blend factor between 0.25 and 0.75
            
            for i in range(n):
                # Blend positions and radii
                child[i, 0] = alpha * parent1[i, 0] + (1 - alpha) * parent2[i, 0]
                child[i, 1] = alpha * parent1[i, 1] + (1 - alpha) * parent2[i, 1]
                child[i, 2] = alpha * parent1[i, 2] + (1 - alpha) * parent2[i, 2]
            
            # Adaptive mutation with different rates for different components
            for i in range(n):
                if random.random() < 0.1:  # Mutation rate
                    # Different step sizes for different components
                    child[i, 0] += np.random.normal(0, 0.01)  # Position step
                    child[i, 1] += np.random.normal(0, 0.01)
                    child[i, 2] += np.random.normal(0, 0.005)  # Radius step
                    
                    # Keep within bounds
                    child[i, 0] = np.clip(child[i, 0], 0.001, 0.999)
                    child[i, 1] = np.clip(child[i, 1], 0.001, 0.999)
                    child[i, 2] = np.clip(child[i, 2], 0.001, 0.5)
            
            new_population.append(child)
        
        population = new_population
    
    # Return best individual
    fitness_scores = [evaluate_fitness(ind) for ind in population]
    best_idx = np.argmax(fitness_scores)
    return population[best_idx]

def _improved_local_search(circles: np.ndarray) -> np.ndarray:
    """More sophisticated local search with better neighborhood exploration"""
    n = len(circles)
    
    # Use a more thorough local search approach
    improved = True
    iterations = 0
    max_iterations = 100  # More iterations for better search
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try improving each circle systematically
        for i in range(n):
            best_x, best_y, best_r = circles[i]
            best_sum = np.sum(circles[:, 2])
            
            # Try more strategic adjustments with adaptive steps
            adjustments = [
                (0.002, 0.002, 0.002),   # Small positive
                (-0.002, -0.002, -0.002), # Small negative
                (0, 0, 0.003),           # Increase radius only
                (0, 0, -0.003),          # Decrease radius only
                (0.003, 0, 0),           # Move right
                (-0.003, 0, 0),          # Move left
                (0, 0.003, 0),           # Move up
                (0, -0.003, 0),          # Move down
                (0.001, 0.001, 0),       # Small move diagonally
                (-0.001, -0.001, 0),     # Small move diagonally opposite
            ]
            
            # Also try some larger moves occasionally
            if random.random() < 0.1:
                adjustments.extend([
                    (0.01, 0.01, 0.005),
                    (-0.01, -0.01, -0.005),
                ])
            
            for dx, dy, dr in adjustments:
                new_x = best_x + dx
                new_y = best_y + dy
                new_r = best_r + dr
                
                # Check bounds
                if (0.001 <= new_x - new_r and new_x + new_r <= 0.999 and
                    0.001 <= new_y - new_r and new_y + new_r <= 0.999 and
                    0.001 <= new_r <= 0.5):
                    
                    # Test new configuration
                    test_circles = circles.copy()
                    test_circles[i] = [new_x, new_y, new_r]
                    
                    # Check overlap constraints with all others
                    overlap_ok = True
                    for j in range(n):
                        if i != j:
                            dist_sq = (test_circles[i, 0] - test_circles[j, 0])**2 + \
                                      (test_circles[i, 1] - test_circles[j, 1])**2
                            min_dist_sq = (test_circles[i, 2] + test_circles[j, 2])**2
                            if dist_sq < min_dist_sq:
                                overlap_ok = False
                                break
                    
                    if overlap_ok:
                        new_sum = np.sum(test_circles[:, 2])
                        if new_sum > best_sum:
                            best_x, best_y, best_r = new_x, new_y, new_r
                            best_sum = new_sum
                            improved = True
            
            circles[i] = [best_x, best_y, best_r]
    
    # Final cleanup to ensure all constraints are met
    for i in range(n):
        x, y, r = circles[i]
        # Ensure containment
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
