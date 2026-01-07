# EVOLVE-BLOCK-START
import numpy as np
import pymunk
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import time

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing26() -> np.ndarray:
    """
    Places 26 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (26,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 26
    
    # --- Phase 1: Physics-based initial placement using Pymunk ---
    # Goal: Get a non-overlapping, contained arrangement for small, fixed radii.
    
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = 0.9 # Apply damping to reduce energy and settle faster

    # Add static boundaries for the unit square
    static_body = space.static_body
    thickness = 0.001 # Small thickness for boundaries to ensure collision
    
    # Left, Bottom, Right, Top walls
    walls = [
        pymunk.Segment(static_body, (0, 0), (0, 1), thickness),
        pymunk.Segment(static_body, (0, 0), (1, 0), thickness),
        pymunk.Segment(static_body, (1, 0), (1, 1), thickness),
        pymunk.Segment(static_body, (0, 1), (1, 1), thickness)
    ]
    for wall in walls:
        wall.friction = 1.0
        wall.elasticity = 0.1 # Make collisions slightly bouncy to help separate
        space.add(wall)

    # Create circles
    initial_r_pymunk = 0.03 # Small initial radius for pymunk simulation
    circles_pymunk_bodies = [] # Store bodies to retrieve positions
    np.random.seed(42) # For reproducibility of initial positions

    for i in range(n):
        # Initial random positions, slightly offset from boundaries to avoid immediate heavy collision
        x = np.random.uniform(initial_r_pymunk * 2, 1 - initial_r_pymunk * 2)
        y = np.random.uniform(initial_r_pymunk * 2, 1 - initial_r_pymunk * 2)
        
        # Give some initial velocity to help "shake" them into place
        vx = np.random.uniform(-0.05, 0.05) # Reduced initial velocity
        vy = np.random.uniform(-0.05, 0.05)

        mass = 1
        moment = pymunk.moment_for_circle(mass, 0, initial_r_pymunk)
        body = pymunk.Body(mass, moment)
        body.position = x, y
        body.velocity = vx, vy
        shape = pymunk.Circle(body, initial_r_pymunk)
        shape.friction = 0.5
        shape.elasticity = 0.1 # Slightly less bouncy
        space.add(body, shape)
        circles_pymunk_bodies.append(body) # Only need the body to get position later

    # Simulate to settle circles
    simulation_steps = 3000 # Increased number of steps for better settling
    dt = 1.0 / 60.0 # Time step (e.g., 60 FPS)

    for _ in range(simulation_steps):
        space.step(dt)
    
    # Extract positions and prepare initial guess for Scipy
    initial_guess_scipy = np.empty(n * 3)
    initial_r_scipy_start = 0.05 # Initial radius for Scipy optimization, slightly larger

    for i, body in enumerate(circles_pymunk_bodies):
        initial_guess_scipy[i*3] = body.position.x
        initial_guess_scipy[i*3+1] = body.position.y
        initial_guess_scipy[i*3+2] = initial_r_scipy_start # Use the larger radius for Scipy start

    # --- Phase 2: Optimization with Scipy ---
    # Objective function: minimize negative sum of radii
    def objective(params):
        current_circles = params.reshape((n, 3))
        # Ensure radii are positive for calculations
        current_circles[:, 2] = np.maximum(current_circles[:, 2], 1e-9) 
        return -np.sum(current_circles[:, 2])

    # Bounds for (x, y, r) for each circle
    lb = np.array([0.0, 0.0, 1e-9] * n) # r_i must be > 0
    ub = np.array([1.0, 1.0, 0.5] * n) # r_i must be <= 0.5
    bounds = Bounds(lb, ub)

    # Nonlinear constraints for containment and non-overlap
    def constraints_func(params):
        current_circles = params.reshape((n, 3))
        
        # Ensure radii are positive for calculations, although bounds should handle this
        r = np.maximum(current_circles[:, 2], 1e-9)
        x = current_circles[:, 0]
        y = current_circles[:, 1]

        constraints = []

        # Containment constraints (4*N constraints)
        for i in range(n):
            constraints.append(x[i] - r[i])       # x - r >= 0
            constraints.append(1 - x[i] - r[i])   # 1 - x - r >= 0
            constraints.append(y[i] - r[i])       # y - r >= 0
            constraints.append(1 - y[i] - r[i])   # 1 - y - r >= 0
        
        # Non-overlap constraints (N*(N-1)/2 constraints)
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                min_dist_sq = (r[i] + r[j])**2
                constraints.append(dist_sq - min_dist_sq) 

        return np.array(constraints)

    num_total_constraints = n * 4 + n * (n - 1) // 2
    nonlinear_constraint = NonlinearConstraint(
        constraints_func,
        lb=np.zeros(num_total_constraints),
        ub=np.full(num_total_constraints, np.inf),
        jac='2-point' # Use numerical approximation for Jacobian
    )

    # Clamp initial guess from pymunk to ensure it respects bounds for scipy.optimize
    for i in range(n):
        r_val = initial_guess_scipy[i*3+2] # Current initial radius for scipy
        initial_guess_scipy[i*3] = np.clip(initial_guess_scipy[i*3], r_val, 1 - r_val) # x
        initial_guess_scipy[i*3+1] = np.clip(initial_guess_scipy[i*3+1], r_val, 1 - r_val) # y
        initial_guess_scipy[i*3+2] = np.clip(initial_guess_scipy[i*3+2], 1e-9, 0.5) # r

    # Run optimization
    res = minimize(
        objective,
        initial_guess_scipy,
        method='SLSQP', # Good for this type of problem
        bounds=bounds,
        constraints=[nonlinear_constraint],
        options={'maxiter': 3000, 'ftol': 1e-8, 'disp': False} # Increased maxiter and ftol for precision
    )

    if not res.success:
        print(f"Optimization warning: {res.message}. Final sum_radii might be suboptimal.")
        # If optimization fails, still return the best found solution.

    optimized_circles = res.x.reshape((n, 3))
    
    # Final clamping to ensure strict adherence to constraints, might lose a tiny bit of sum_radii
    # This step is crucial because optimization might slightly violate constraints at convergence.
    for i in range(n):
        r_val = optimized_circles[i, 2]
        r_val = np.maximum(r_val, 1e-9) # Ensure radius is positive
        optimized_circles[i, 2] = r_val
        
        # Ensure positions are strictly within r and 1-r
        optimized_circles[i, 0] = np.clip(optimized_circles[i, 0], r_val, 1 - r_val)
        optimized_circles[i, 1] = np.clip(optimized_circles[i, 1], r_val, 1 - r_val)

    return optimized_circles


# EVOLVE-BLOCK-END
