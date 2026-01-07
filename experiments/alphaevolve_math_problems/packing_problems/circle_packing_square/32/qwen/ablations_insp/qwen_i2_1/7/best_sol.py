# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import math

# Global constants for the algorithm
MAX_ITERATIONS = 15000
TEMP_START = 1.0
TEMP_DECAY = 0.9995
MIN_TEMP = 1e-6

def initialize_hexagonal_grid(n):
    """Initialize circle positions using a hexagonal grid pattern"""
    # Calculate grid parameters
    rows = int(math.sqrt(n))
    cols = int(math.ceil(n / rows))
    
    # Create hexagonal grid points
    spacing_x = 0.9 / (cols + 1)
    spacing_y = 0.9 / (rows + 1)
    
    positions = []
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = 0.05 + (j + 1) * spacing_x
            y = 0.05 + (i + 1) * spacing_y
            # Offset every other row for hexagonal packing
            if i % 2 == 1:
                x += spacing_x / 2
            positions.append([x, y])
    
    # Ensure we have exactly n positions
    while len(positions) < n:
        positions.append([random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)])
    
    return np.array(positions[:n])

def calculate_max_radius(pos, circles, idx):
    """Calculate maximum possible radius for circle at position pos without overlapping others"""
    max_r = min(pos[0], 1-pos[0], pos[1], 1-pos[1])  # Boundary constraints
    
    # Check overlap with existing circles
    for i, (cx, cy, r) in enumerate(circles):
        if i != idx:
            dist = math.sqrt((pos[0] - cx)**2 + (pos[1] - cy)**2)
            max_r = min(max_r, dist - r)
    
    return max(max_r, 0.001)  # Ensure minimum radius

def evaluate_fitness(circles):
    """Evaluate fitness as sum of all radii"""
    return sum(circle[2] for circle in circles)

def move_circle(circles, idx, step_size=0.01):
    """Move a single circle slightly in a random direction"""
    new_circles = [circle[:] for circle in circles]  # Deep copy
    
    dx = random.uniform(-step_size, step_size)
    dy = random.uniform(-step_size, step_size)
    
    new_x = max(0.001, min(0.999, circles[idx][0] + dx))
    new_y = max(0.001, min(0.999, circles[idx][1] + dy))
    
    # Recalculate maximum possible radius
    max_r = calculate_max_radius([new_x, new_y], new_circles, idx)
    new_circles[idx] = [new_x, new_y, max_r]
    
    return new_circles

def simulate_annealing():
    """Main simulated annealing implementation with better cooling and acceptance"""
    n = 32
    # Initialize with hexagonal grid
    positions = initialize_hexagonal_grid(n)
    
    # Initialize circles with small radii
    circles = [[pos[0], pos[1], 0.01] for pos in positions]
    
    # Set up temperature schedule
    temp = TEMP_START
    best_circles = [circle[:] for circle in circles]
    best_fitness = evaluate_fitness(best_circles)
    
    # Main optimization loop
    for iteration in range(MAX_ITERATIONS):
        # Randomly select a circle to perturb
        idx = random.randint(0, n-1)
        
        # Generate neighbor solution
        new_circles = move_circle(circles, idx)
        
        # Evaluate new solution
        new_fitness = evaluate_fitness(new_circles)
        
        # Accept or reject based on Metropolis criterion
        delta = new_fitness - evaluate_fitness(circles)
        if delta > 0 or random.random() < math.exp(delta / temp):
            circles = new_circles
            
            # Update best solution
            if new_fitness > best_fitness:
                best_circles = [circle[:] for circle in new_circles]
                best_fitness = new_fitness
                
        # Cool down temperature
        temp = max(MIN_TEMP, temp * TEMP_DECAY)
        
        # Early stopping condition
        if temp < MIN_TEMP:
            break
    
    return best_circles

def refine_solution(circles):
    """Apply local optimization to refine the solution"""
    # Use a more aggressive refinement approach
    improved = True
    max_iterations = 1000
    
    for iteration in range(max_iterations):
        if not improved:
            break
        improved = False
        
        # Try to improve each circle individually with better search
        for i in range(len(circles)):
            # Store current state
            original_pos = circles[i][:2]
            original_radius = circles[i][2]
            
            # Try several moves in different directions
            best_move = None
            best_fitness = evaluate_fitness(circles)
            
            # Try larger steps for faster exploration, but also smaller ones for fine-tuning
            step_sizes = [0.05, 0.02, 0.01, 0.005]
            moves_per_step = 30  # More moves for better search
            
            for step_size in step_sizes:
                # Try moves per step size for better search
                for _ in range(moves_per_step):
                    dx = random.uniform(-step_size, step_size)
                    dy = random.uniform(-step_size, step_size)
                    
                    new_x = max(0.001, min(0.999, circles[i][0] + dx))
                    new_y = max(0.001, min(0.999, circles[i][1] + dy))
                    
                    # Calculate new radius
                    max_r = calculate_max_radius([new_x, new_y], circles, i)
                    
                    if max_r > original_radius:
                        test_circles = [c[:] for c in circles]
                        test_circles[i] = [new_x, new_y, max_r]
                        new_fitness = evaluate_fitness(test_circles)
                        
                        if new_fitness > best_fitness:
                            best_fitness = new_fitness
                            best_move = (new_x, new_y, max_r)
            
            # Apply best move if found
            if best_move:
                circles[i] = list(best_move)
                improved = True
    
    return circles

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    Uses a hybrid approach combining hexagonal initialization, simulated annealing, and local refinement.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates 
                 of the i-th circle of radius r.
    """
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run simulated annealing to get good initial solution
    circles = simulate_annealing()
    
    # Refine the solution with local optimization
    circles = refine_solution(circles)
    
    # Convert to numpy array
    result = np.array(circles)
    
    return result


# EVOLVE-BLOCK-END
