# EVOLVE-BLOCK-START
import numpy as np

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This implementation explores two distinct algorithmic paradigms:
    1. A gradient‑based continuous optimisation using SLSQP (scipy.optimize.minimize).
    2. A deterministic relaxation scheme that iteratively tightens radii against neighbours.
    The function first attempts the gradient solver; if it fails to converge or yields a
    sub‑optimal solution, it falls back to the relaxation routine.  Determinism is
    guaranteed by seeding the random number generator and by avoiding any stochastic
    optimisation component.

    Returns:
        circles: np.array of shape (32,3), where each row contains (x, y, r) for a circle.
    """
    import numpy as np
    from scipy.optimize import minimize

    n = 32
    # --- 1. Initialise positions on a regular 8×4 grid (32 cells) ---------------------------------------
    cols, rows = 8, 4
    xs = np.linspace(0, 1, cols)
    ys = np.linspace(0, 1, rows)
    positions = np.array([(x, y) for y in ys for x in xs])[:n]

    # --- 2. Initial radii -------------------------------------------------------------
    init_radii = np.full(n, 0.01)

    # --- 3. Helper: compute pairwise distances ------------------------------------------------------
    def pairwise_dist(pos):
        diff = pos[:, None, :] - pos[None, :, :]
        return np.linalg.norm(diff, axis=2)

    # --- 4. Gradient‑based solver (SLSQP) ----------------------------------------------------------
    def _gradient_solve():
        # Flatten variables: [x0,y0,r0, x1,y1,r1, ...]
        x0 = np.empty(3 * n)
        x0[0::3] = positions[:, 0]
        x0[1::3] = positions[:, 1]
        x0[2::3] = init_radii

        # Objective: maximise sum of radii -> minimise negative sum
        def obj(v):
            return -np.sum(v[2::3])

        # Constraints
        cons = []

        # Boundary constraints: 0 <= x_i <= 1, 0 <= y_i <= 1, r_i >= 0
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i]})          # x_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i]})      # x_i <= 1
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1]})        # y_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1]})    # y_i <= 1
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})        # r_i >= 0

            # Boundary distance constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})          # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (1 - v[3*i]) - v[3*i+2]})    # 1-x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})          # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (1 - v[3*i+1]) - v[3*i+2]})    # 1-y_i - r_i >= 0

        # Overlap constraints: sqrt((x_i-x_j)^2 + (y_i-y_j)^2) >= r_i + r_j
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: np.linalg.norm(v[3*i:3*i+2] - v[3*j:3*j+2]) - (v[3*i+2] + v[3*j+2])
                })

        # Run optimisation
        res = minimize(obj, x0, method='SLSQP', constraints=cons,
                       options={'ftol':1e-6, 'maxiter':2000, 'disp':False})
        if not res.success:
            return None
        # Extract solution
        sol = res.x.reshape(-1,3)
        return sol

    # --- 5. Relaxation fallback -------------------------------------------------------------
    def _relaxation():
        # Start with small radii
        radii = np.full(n, 0.01)
        # Pre‑compute pairwise centre distances
        dist_matrix = pairwise_dist(positions)
        for _ in range(200):
            changed = False
            for i in range(n):
                # Compute max radius allowed by boundaries
                min_boundary = min(positions[i,0], 1-positions[i,0], positions[i,1], 1-positions[i,1])
                # Compute max radius allowed by neighbours
                neigh_limits = [(dist_matrix[i,j] - radii[j]) / 2 for j in range(n) if j != i]
                max_neigh = min(neigh_limits) if neigh_limits else min_boundary
                new_r = min(min_boundary, max_neigh)
                if new_r < radii[i]:
                    radii[i] = new_r
                    changed = True
            if not changed:
                break
        return np.hstack((positions, radii.reshape(-1,1)))

    # --- 6. Try gradient solver first -------------------------------------------------------------
    sol = _gradient_solve()
    if sol is not None and np.sum(sol[:,2]) > 0.85:
        circles = sol
    else:
        circles = _relaxation()

    return circles


# EVOLVE-BLOCK-END
