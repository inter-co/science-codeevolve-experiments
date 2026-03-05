# EVOLVE-BLOCK-START
import numpy as np
import math
import random
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    base = None
    creator = None
    tools = None
    algorithms = None
try:
    from scipy.spatial import KDTree
except ImportError:
    KDTree = None
from scipy.optimize import minimize

def circle_packing32(strategy: str = 'hex', seed: int = 42) -> np.ndarray:
    """
    Generate 32 non-overlapping circles inside the unit square maximizing the sum of radii.
    Two strategies are available:
        - 'hex' : deterministic hexagonal lattice packing.
        - 'evo' : simple evolutionary algorithm using DEAP.
    Parameters
    ----------
    strategy : str
        Choice of strategy.
    seed : int
        Random seed for reproducibility.
    Returns
    -------
    np.ndarray
        Array of shape (32,3) with (x,y,r) for each circle.
    """
    np.random.seed(seed)
    random.seed(seed)
    n = 32

    if strategy == 'hex':
        rng = np.random.default_rng(seed)
        # Determine optimal rows and columns for hexagonal packing
        best_s = 0.0
        best_rc = (1, n)
        for r in range(1, n + 1):
            c = -(-n // r)  # ceil division
            s1 = 1.0 / c
            s2 = 1.0 / ((r - 1) * math.sqrt(3) / 2 + 1)
            s = min(s1, s2)
            if s > best_s:
                best_s = s
                best_rc = (r, c)
        r_rows, c_cols = best_rc
        s = best_s
        R = s / 2.0
        vertical = math.sqrt(3) / 2 * s

        # Generate initial positions
        positions = []
        for i in range(r_rows):
            for j in range(c_cols):
                if len(positions) >= n:
                    break
                x = j * s + R
                y = i * vertical + R
                if i % 2 == 1:
                    x += s / 2
                # Ensure within bounds
                if x - R < 0 or x + R > 1 or y - R < 0 or y + R > 1:
                    continue
                positions.append([x, y])
            if len(positions) >= n:
                break
        # Pad if necessary
        while len(positions) < n:
            positions.append([np.random.uniform(0.01, 0.99), np.random.uniform(0.01, 0.99)])
            # start with tiny radius; will be inflated later
        positions = np.array(positions, dtype=float)

        # Helper: recompute radii for all circles given positions
        def compute_radii(pos: np.ndarray) -> np.ndarray:
            d_boundary = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]),
                                    np.minimum(pos[:, 1], 1 - pos[:, 1]))
            tree = KDTree(pos)
            d_neighbor, _ = tree.query(pos, k=2)
            rad = np.minimum(d_boundary, d_neighbor[:, 1] / 2.0)
            return rad

        # --- Inflation helper: iteratively enlarge radii while respecting constraints ----
        def _inflate(positions: np.ndarray, radii: np.ndarray,
                     max_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
            for _ in range(max_iter):
                changed = False
                for i in range(n):
                    xi, yi = positions[i]
                    max_r = min(xi, 1.0 - xi, yi, 1.0 - yi)
                    for j in range(n):
                        if j == i:
                            continue
                        xj, yj = positions[j]
                        d = np.hypot(xi - xj, yi - yj) - radii[j]
                        if d < max_r:
                            max_r = d
                    new_r = min(max_r, radii[i] + 0.01)
                    if new_r > radii[i] + tol:
                        radii[i] = new_r
                        changed = True
                if not changed:
                    break
            return np.clip(radii, 0.01, 0.5)

        # Start with zero radii and inflate
        init_radii = np.zeros(n)
        init_radii = _inflate(positions, init_radii)
        best_pos = positions.copy()
        best_rad = init_radii.copy()
        best_sum = best_rad.sum()

        # --- Constraint‑based SLSQP optimizer (Inspired by Insp 2) ---
        # Build bounds: x,y in [0,1], r in [0,0.5]
        bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

        # Boundary constraints
        def _boundary_fun(i, dim, sign):
            def fun(v):
                coord = v[i] if dim == 0 else v[n + i]
                rad   = v[2 * n + i]
                return coord - rad if sign == 1 else 1.0 - coord - rad
            return fun

        # Overlap constraints
        def _overlap_fun(i, j):
            def fun(v):
                xi, yi = v[i], v[n + i]
                xj, yj = v[j], v[n + j]
                ri, rj = v[2 * n + i], v[2 * n + j]
                dist2 = (xi - xj) ** 2 + (yi - yj) ** 2
                return dist2 - (ri + rj) ** 2
            return fun

        cons = []
        for idx in range(n):
            cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 0,  1)})
            cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 0, -1)})
            cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 1,  1)})
            cons.append({'type': 'ineq', 'fun': _boundary_fun(idx, 1, -1)})

        for i in range(n):
            for j in range(i + 1, n):
                cons.append({'type': 'ineq', 'fun': _overlap_fun(i, j)})

        def objective(v):
            return -np.sum(v[2 * n:])

        best_solution = None
        best_sum = -np.inf

        # Multiple randomized restarts
        for restart in range(5):
            # Perturb initial positions slightly
            pos_pert = best_pos + rng.normal(scale=0.02, size=best_pos.shape)
            pos_pert = np.clip(pos_pert, 0, 1)
            # Perturb radii slightly
            rad_pert = best_rad + rng.normal(scale=0.01, size=best_rad.shape)
            rad_pert = np.clip(rad_pert, 0.01, 0.5)
            init_vars = np.concatenate([pos_pert[:,0], pos_pert[:,1], rad_pert])

            res = minimize(
                objective,
                init_vars,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'ftol': 1e-9, 'maxiter': 4000, 'disp': False}
            )

            if res.success:
                xs = res.x[:n]
                ys = res.x[n:2 * n]
                rs = res.x[2 * n:3 * n]
                sum_r = np.sum(rs)
                if sum_r > best_sum:
                    best_sum = sum_r
                    best_solution = np.column_stack([xs, ys, rs])

        # Fallback to current best if SLSQP fails
        if best_solution is None:
            best_solution = np.column_stack([best_pos, best_rad])

        circles = best_solution
        return circles

    elif strategy == 'evo':
        if base is None:
            raise ImportError("DEAP is required for evolutionary strategy but is not installed.")
        # DEAP setup
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        # Individual: 32 circles each with (x,y,r)
        # bounds: x,y in [0,1], r in [0,0.5]
        def random_circle():
            return [np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,0.5)]
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         random_circle, n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Fitness evaluation
        def eval_individual(individual):
            circles = np.array(individual).reshape((n,3))
            total_r = np.sum(circles[:,2])
            penalty = 0.0
            # Boundary penalty
            for x,y,r in circles:
                if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                    penalty += 1000 * (r - min(x, 1-x, y, 1-y))
            # Overlap penalty
            for i in range(n):
                xi, yi, ri = circles[i]
                for j in range(i+1, n):
                    xj, yj, rj = circles[j]
                    d = math.hypot(xi-xj, yi-yj)
                    if d < ri + rj:
                        penalty += 1000 * (ri + rj - d)
            return (total_r - penalty,)

        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutUniformFloat, low=[0,0,0], up=[1,1,0.5], indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=50)
        hof = tools.HallOfFame(1)

        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50,
                            halloffame=hof, verbose=False)

        best = hof[0]
        circles = np.array(best).reshape((n,3))
        return circles


# EVOLVE-BLOCK-END
