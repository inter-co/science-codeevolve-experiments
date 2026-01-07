# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, distance
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List
import time
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from functools import partial

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def is_valid_placement(circles: np.ndarray, new_circle: Tuple[float, float, float]) -> bool:
    """Check if a new circle placement is valid (no overlaps, fully contained)."""
    x, y, r = new_circle
    
    # Check containment
    if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
        return False
    
    # Check overlap with existing circles
    for i in range(len(circles)):
        if i >= len(circles):
            break
        cx, cy, cr = circles[i]
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        if distance < r + cr:
            return False
    
    return True

def compute_total_radius(circles: np.ndarray) -> float:
    """Compute the sum of all circle radii."""
    return np.sum(circles[:, 2])

def evaluate_circles(individual: list) -> tuple:
    """Evaluate a circle configuration and return fitness."""
    # Convert individual to circles array
    circles = np.array(individual).reshape(-1, 3)
    
    # Check if all circles are valid
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
            return (0,)  # Invalid placement
        
        for j in range(i):
            cx, cy, cr = circles[j]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < r + cr:
                return (0,)  # Overlapping circles
    
    # Return negative sum of radii (since we want to maximize)
    return (-compute_total_radius(circles),)

def initialize_grid(n: int) -> np.ndarray:
    """Initialize circles using a grid-based approach for better distribution."""
    # Create a grid pattern
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Adjust to ensure we have enough positions
    while rows * cols < n:
        cols += 1
    
    # Create circle positions
    circles = np.zeros((n, 3))
    
    # Grid spacing
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    # Initialize with small radii
    for i in range(n):
        row = i // cols
        col = i % cols
        
        x = (col + 0.5) * spacing_x
        y = (row + 0.5) * spacing_y
        
        # Initial radius - based on grid spacing and some randomness
        max_radius = min(spacing_x, spacing_y) * 0.4
        radius = np.random.uniform(0.1 * max_radius, 0.8 * max_radius)
        
        circles[i] = (x, y, radius)
    
    return circles

def initialize_random(n: int) -> np.ndarray:
    """Initialize circles randomly with some validation."""
    circles = np.zeros((n, 3))
    
    for i in range(n):
        attempts = 0
        while attempts < 1000:
            # Random position
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            
            # Random radius (smaller to avoid immediate overlaps)
            max_radius = min(x, 1-x, y, 1-y) * 0.4
            if max_radius <= 0:
                attempts += 1
                continue
                
            radius = np.random.uniform(0.01, max_radius)
            
            # Check if this circle is valid with current configuration
            new_circle = (x, y, radius)
            valid = True
            
            # Check against existing circles
            for j in range(i):
                cx, cy, cr = circles[j]
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < radius + cr:
                    valid = False
                    break
            
            if valid:
                circles[i] = new_circle
                break
            attempts += 1
        
        # If we couldn't place a valid circle, use fallback
        if attempts >= 1000:
            x = np.random.uniform(0.05, 0.95)
            y = np.random.uniform(0.05, 0.95)
            max_radius = min(x, 1-x, y, 1-y) * 0.3
            circles[i] = (x, y, max_radius)
    
    return circles

def evolve_circles(circles: np.ndarray, max_generations: int = 100) -> np.ndarray:
    """Use evolutionary algorithm to optimize circle placement."""
    n = len(circles)
    
    # Define the optimization problem
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Define gene representation: [x, y, r] for each circle
    def create_individual():
        # Start with current circles
        individual = []
        for i in range(n):
            x, y, r = circles[i]
            # Add small random variation to initial positions
            x += np.random.normal(0, 0.02)
            y += np.random.normal(0, 0.02)
            r += np.random.normal(0, 0.01)
            
            # Keep within bounds
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            r = np.clip(r, 0.001, 0.4)
            
            individual.extend([x, y, r])
        return creator.Individual(individual)
    
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation function
    toolbox.register("evaluate", evaluate_circles)
    
    # Register genetic operators
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.02, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Create population
    pop = toolbox.population(n=50)
    
    # Run evolution
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    try:
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, 
                                          ngen=max_generations, stats=stats, 
                                          halloffame=hof, verbose=False)
        best_individual = hof[0]
        return np.array(best_individual).reshape(-1, 3)
    except Exception as e:
        # If evolution fails, return the original circles
        return circles

def optimize_circles_local(circles: np.ndarray, max_iter: int = 2000) -> np.ndarray:
    """Local optimization using a more sophisticated approach."""
    n = len(circles)
    best_circles = circles.copy()
    best_sum = compute_total_radius(best_circles)
    
    # Try to improve each circle individually
    for iteration in range(max_iter):
        improved = False
        
        # Try to improve each circle
        for i in range(n):
            old_x, old_y, old_r = best_circles[i]
            
            # Try various perturbations
            best_candidate = (old_x, old_y, old_r)
            best_candidate_sum = best_sum
            
            # Try small improvements
            for _ in range(50):
                # Small random perturbations
                dx = np.random.normal(0, 0.005)
                dy = np.random.normal(0, 0.005)
                dr = np.random.normal(0, 0.002)
                
                new_x = max(0.01, min(0.99, old_x + dx))
                new_y = max(0.01, min(0.99, old_y + dy))
                new_r = max(0.001, min(0.4, old_r + dr))
                
                # Check if this change is valid
                test_circles = best_circles.copy()
                test_circles[i] = (new_x, new_y, new_r)
                
                # Validate against all other circles
                valid = True
                for j in range(n):
                    if i != j:
                        x1, y1, r1 = test_circles[i]
                        x2, y2, r2 = test_circles[j]
                        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                        if dist < r1 + r2:
                            valid = False
                            break
                
                if valid:
                    # Check containment
                    if new_x - new_r >= 0 and new_x + new_r <= 1 and \
                       new_y - new_r >= 0 and new_y + new_r <= 1:
                        
                        # Calculate new sum
                        new_sum = compute_total_radius(test_circles)
                        
                        if new_sum > best_candidate_sum:
                            best_candidate = (new_x, new_y, new_r)
                            best_candidate_sum = new_sum
                            improved = True
            
            # Apply the best improvement found
            if improved:
                best_circles[i] = best_candidate
                best_sum = best_candidate_sum
        
        # Early stopping if no improvement
        if not improved:
            break
    
    return best_circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    circles = np.zeros((n, 3))
    
    # Start with a good initial configuration using grid-based approach
    circles = initialize_grid(n)
    
    # Run evolutionary optimization
    circles = evolve_circles(circles, max_generations=50)
    
    # Local optimization refinement
    circles = optimize_circles_local(circles, max_iter=1000)
    
    # Final local search with more intensive optimization
    circles = optimize_circles_local(circles, max_iter=2000)
    
    return circles


# EVOLVE-BLOCK-END
