# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import math
from typing import Tuple
import random
from collections import defaultdict
from itertools import combinations
import time

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square to maximize the sum of radii.
    Uses a hybrid approach combining evolutionary algorithm, geometric insights, 
    and gradient-based optimization.
    
    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # Use a more sophisticated initial layout based on known good configurations
    circles = _better_initial_layout(n)
    
    # Try multiple optimization strategies
    best_sum = -float('inf')
    best_circles = circles.copy()
    
    # Strategy 1: Multiple restarts with different initializations
    for attempt in range(8):
        try:
            # Create diverse initial solutions
            if attempt < 4:
                # Use better initial layout with some randomness
                perturbed_circles = _better_initial_layout(n)
                # Add small random perturbation
                for i in range(n):
                    perturbed_circles[i, 0] += np.random.uniform(-0.03, 0.03)
                    perturbed_circles[i, 1] += np.random.uniform(-0.03, 0.03)
                    perturbed_circles[i, 2] += np.random.uniform(-0.015, 0.015)
            else:
                # Use evolutionary approach to generate initial solution
                perturbed_circles = _evolutionary_approach(n)
            
            # Ensure bounds are respected
            for i in range(n):
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
    
    # Strategy 2: Enhanced evolutionary approach with better operators
    evol_result = _enhanced_evolutionary_approach(n)
    evol_sum = np.sum(evol_result[:, 2])
    if evol_sum > best_sum:
        best_sum = evol_sum
        best_circles = evol_result
    
    # Strategy 3: Global optimization with better constraints handling
    global_result = _global_optimization_approach(n)
    global_sum = np.sum(global_result[:, 2])
    if global_sum > best_sum:
        best_sum = global_sum
        best_circles = global_result
    
    # Final refinement with more aggressive local search
    refined = _improved_local_optimization(best_circles)
    
    return refined

def _better_initial_layout(n: int) -> np.ndarray:
    """Create a better initial heuristic layout inspired by hexagonal packing"""
    circles = np.zeros((n, 3))
    
    # For 26 circles, we'll use a more sophisticated approach
    # Create a grid-like pattern with hexagonal packing considerations
    
    # First, try a 5x5 grid pattern with adjustments
    rows = 5
    cols = 5
    
    # Calculate spacing with better consideration of boundary constraints
    spacing_x = 0.85 / (cols - 1) if cols > 1 else 0.5
    spacing_y = 0.85 / (rows - 1) if rows > 1 else 0.5
    
    # Create hexagonal pattern with proper spacing
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
                
            # Offset every other row for hexagonal packing
            x_offset = spacing_x * 0.5 if i % 2 == 1 else 0.0
            x = 0.075 + j * spacing_x + x_offset
            y = 0.075 + i * spacing_y
            
            # Keep within bounds
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            # Initial radius based on distance to boundaries and position
            min_edge_dist = min(x, 1-x, y, 1-y)
            # Base radius with adjustment for center vs edge positioning
            base_radius = min_edge_dist * 0.35
            
            # Circles closer to center get larger radii, but with better distribution
            center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            if center_dist < 0.25:
                radius = min(base_radius, 0.15)
            elif center_dist < 0.5:
                radius = min(base_radius, 0.12)
            else:
                radius = min(base_radius, 0.09)
            
            # Ensure reasonable minimum
            radius = max(radius, 0.01)
            
            circles[idx] = [x, y, radius]
            idx += 1
            
        if idx >= n:
            break
    
    # Fill remaining positions strategically
    remaining_positions = n - idx
    for i in range(remaining_positions):
        # Place in corners or along edges with careful spacing
        corner_positions = [
            (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9),
            (0.05, 0.5), (0.5, 0.05), (0.95, 0.5), (0.5, 0.95)
        ]
        
        if i < len(corner_positions):
            x, y = corner_positions[i]
        else:
            # Random placement near edges but not too close to corners
            side = i % 4
            if side == 0:  # Bottom edge
                x = np.random.uniform(0.1, 0.9)
                y = 0.05
            elif side == 1:  # Top edge
                x = np.random.uniform(0.1, 0.9)
                y = 0.95
            elif side == 2:  # Left edge
                x = 0.05
                y = np.random.uniform(0.1, 0.9)
            else:  # Right edge
                x = 0.95
                y = np.random.uniform(0.1, 0.9)
        
        # Radius based on proximity to edges and center
        min_edge_dist = min(x, 1-x, y, 1-y)
        radius = min(min_edge_dist * 0.35, 0.12)
        
        # Adjust for center distance
        center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
        if center_dist < 0.3:
            radius = min(radius, 0.12)
        elif center_dist < 0.6:
            radius = min(radius, 0.1)
        
        radius = max(radius, 0.01)
        
        circles[idx + i] = [x, y, radius]
    
    return circles

def _optimize_with_constraints(initial_circles: np.ndarray) -> np.ndarray:
    """Use constrained optimization with better handling of constraints"""
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
    
    # More efficient constraint definitions
    def containment_constraints(vars_array):
        circles_array = vars_array.reshape(-1, 3)
        result = []
        for i in range(n):
            x, y, r = circles_array[i]
            # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
            result.extend([x - r, 1 - x - r, y - r, 1 - y - r])
        return np.array(result)
    
    # Overlap constraints using vectorized operations for efficiency
    def overlap_constraints(vars_array):
        circles_array = vars_array.reshape(-1, 3)
        result = []
        # Generate all pairs once
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
        {'type': 'ineq', 'fun': lambda x: containment_constraints(x)},
        {'type': 'ineq', 'fun': lambda x: overlap_constraints(x)}
    ]
    
    # Run optimization with better parameters
    try:
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-8, 'eps': 1e-6},
            callback=lambda x: None  # No callback for performance
        )
        
        if result.success:
            optimized_circles = result.x.reshape(-1, 3)
            # Clip to ensure validity
            for i in range(n):
                optimized_circles[i, 0] = np.clip(optimized_circles[i, 0], 0.001, 0.999)
                optimized_circles[i, 1] = np.clip(optimized_circles[i, 1], 0.001, 0.999)
                optimized_circles[i, 2] = np.clip(optimized_circles[i, 2], 0.001, 0.5)
            return optimized_circles
    except Exception as e:
        # If optimization fails, return original
        pass
    
    return initial_circles

def _enhanced_evolutionary_approach(n: int) -> np.ndarray:
    """Use a more sophisticated evolutionary algorithm approach"""
    # Population size
    pop_size = 30
    generations = 70
    
    # Initialize population with better diversity
    population = []
    for _ in range(pop_size):
        individual = _better_initial_layout(n)
        # Add some random variation to increase diversity
        for i in range(n):
            individual[i, 0] += np.random.uniform(-0.02, 0.02)
            individual[i, 1] += np.random.uniform(-0.02, 0.02)
            individual[i, 2] += np.random.uniform(-0.01, 0.01)
        population.append(individual)
    
    # Evaluate fitness
    def evaluate_fitness(individual):
        # Calculate sum of radii
        total_radius = np.sum(individual[:, 2])
        
        # Penalize constraint violations more heavily
        penalty = 0
        
        # Check containment
        for i in range(n):
            x, y, r = individual[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                penalty += 5000
        
        # Check overlaps with early termination
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = individual[i]
                x2, y2, r2 = individual[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                min_dist_sq = (r1 + r2)**2
                if dist_sq < min_dist_sq:
                    # Penalty proportional to violation amount
                    penalty += (min_dist_sq - dist_sq) * 200
        
        return total_radius - penalty
    
    # Evolution loop
    for gen in range(generations):
        # Evaluate all individuals
        fitness_scores = [evaluate_fitness(ind) for ind in population]
        
        # Select best individuals using tournament selection
        sorted_indices = np.argsort(fitness_scores)[::-1]
        selected = [population[i] for i in sorted_indices[:pop_size//2]]
        
        # Create new population through crossover and mutation
        new_population = selected.copy()
        
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = selected[random.randint(0, len(selected)-1)]
            parent2 = selected[random.randint(0, len(selected)-1)]
            
            # Crossover with uniform and blend crossover
            child = parent1.copy()
            for i in range(n):
                if random.random() < 0.5:
                    child[i] = parent2[i]
                # Blend crossover for continuous variables
                elif random.random() < 0.2:
                    alpha = random.random()
                    child[i, 0] = alpha * parent1[i, 0] + (1 - alpha) * parent2[i, 0]
                    child[i, 1] = alpha * parent1[i, 1] + (1 - alpha) * parent2[i, 1]
                    child[i, 2] = alpha * parent1[i, 2] + (1 - alpha) * parent2[i, 2]
            
            # Mutation with adaptive rates
            for i in range(n):
                if random.random() < 0.15:  # Higher mutation rate
                    # Mutate position and/or radius
                    child[i, 0] += np.random.normal(0, 0.015)
                    child[i, 1] += np.random.normal(0, 0.015)
                    child[i, 2] += np.random.normal(0, 0.008)
                    
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

def _global_optimization_approach(n: int) -> np.ndarray:
    """Try a different optimization approach with better initialization"""
    # Start with a good layout
    circles = _better_initial_layout(n)
    
    # Try a few different optimization approaches
    best_result = circles.copy()
    best_sum = np.sum(circles[:, 2])
    
    # Try with different solvers
    from scipy.optimize import differential_evolution
    
    # Flatten for differential evolution
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.5)])
    
    def objective(vars_array):
        circles_array = vars_array.reshape(-1, 3)
        return -np.sum(circles_array[:, 2])
    
    def constraint_function(vars_array):
        circles_array = vars_array.reshape(-1, 3)
        result = []
        
        # Containment constraints
        for i in range(n):
            x, y, r = circles_array[i]
            result.extend([x - r, 1 - x - r, y - r, 1 - y - r])
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                x1, y1, r1 = circles_array[i]
                x2, y2, r2 = circles_array[j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                result.append(dist_sq - (r1 + r2)**2)
        
        return np.array(result)
    
    # Try differential evolution for global search
    try:
        de_result = differential_evolution(
            objective,
            bounds,
            constraints=[{'type': 'ineq', 'fun': constraint_function}],
            maxiter=500,
            popsize=15,
            seed=42
        )
        
        if de_result.success:
            optimized_circles = de_result.x.reshape(-1, 3)
            # Validate and clip
            for i in range(n):
                optimized_circles[i, 0] = np.clip(optimized_circles[i, 0], 0.001, 0.999)
                optimized_circles[i, 1] = np.clip(optimized_circles[i, 1], 0.001, 0.999)
                optimized_circles[i, 2] = np.clip(optimized_circles[i, 2], 0.001, 0.5)
            
            current_sum = np.sum(optimized_circles[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = optimized_circles
    except:
        pass
    
    return best_result

def _improved_local_optimization(circles: np.ndarray) -> np.ndarray:
    """Apply more sophisticated local optimization to improve the solution"""
    n = len(circles)
    
    # More aggressive local search with better neighborhood exploration
    improved = True
    iterations = 0
    max_iterations = 150
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try improving each circle systematically
        for i in range(n):
            best_x, best_y, best_r = circles[i]
            best_sum = np.sum(circles[:, 2])
            
            # Better neighborhood search with adaptive step sizes
            step_sizes = [0.002, 0.005, 0.01, 0.02]
            
            for step in step_sizes:
                # Grid search around current position
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        for dr in [-step*0.3, 0, step*0.3]:
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
                                
                                # Efficient overlap checking - only check with nearby circles
                                overlap_ok = True
                                # Find nearby circles to check against
                                nearby_indices = []
                                for j in range(n):
                                    if i != j:
                                        dist_sq = (test_circles[i, 0] - test_circles[j, 0])**2 + \
                                                  (test_circles[i, 1] - test_circles[j, 1])**2
                                        # Only check if potentially overlapping
                                        if dist_sq < 4*(test_circles[i, 2] + test_circles[j, 2])**2:
                                            nearby_indices.append(j)
                                
                                for j in nearby_indices:
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
    
    # Final cleanup and validation
    for i in range(n):
        x, y, r = circles[i]
        # Ensure containment constraints
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        circles[i] = [x, y, r]
    
    return circles


# EVOLVE-BLOCK-END
