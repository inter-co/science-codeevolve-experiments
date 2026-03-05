# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt

def _tighten_radii(pos: np.ndarray, rad: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Deterministically tighten each radius to the maximum allowed by the current positions
    and other circles.  This is a local refinement step that often improves the total sum.
    """
    n = pos.shape[0]
    for _ in range(iterations):
        for i in range(n):
            max_r = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            d = np.hypot(pos[i, 0] - pos[:, 0], pos[i, 1] - pos[:, 1]) - rad
            max_r = min(max_r, np.min(d[d > 0]))
            rad[i] = max(0.0, max_r)
    return rad

# You can define functions outside the main function below.
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.
    This implementation follows a three‑stage pipeline:
    1.  A fine 8×4 grid initialization with radii set to 90 % of the cell radius.
    2.  A simulated‑annealing search that perturbs both positions and radii.
    3.  A deterministic radius‑refinement loop that tightens each circle to the largest
        feasible size given the final positions.
    The result is then refined once more with SLSQP and a final radius tightening.
    """
    import numpy as np
    import scipy.optimize as opt
    import math

    n = 32
    best_overall = None
    best_sum_overall = -np.inf

    for restart in range(3):
        rng = np.random.default_rng(42 + restart)

        # Stage 1 – Grid initialization
        cols, rows = 8, 4
        dx, dy = 1.0 / cols, 1.0 / rows
        xs = np.linspace(dx / 2, 1 - dx / 2, cols)
        ys = np.linspace(dy / 2, 1 - dy / 2, rows)
        grid_positions = np.array([(x, y) for y in ys for x in xs])[:n]
        init_radius = min(dx, dy) / 2 * 0.9
        radii = np.full(n, init_radius)
        circles = np.column_stack((grid_positions, radii))

        # Stage 2 – Simulated annealing
        T = 0.05
        Tmin = 1e-4
        alpha = 0.995
        steps = 80000

        best = circles.copy()
        best_sum = best[:, 2].sum()

        for step in range(steps):
            i = rng.integers(n)

            if rng.random() < 0.5:
                # Perturb radius
                new_radii = best[:, 2].copy()
                delta = rng.uniform(-0.005, 0.005)
                new_radii[i] = max(0.0, new_radii[i] + delta)
                new_positions = best[:, :2]
            else:
                # Perturb position
                new_positions = best[:, :2].copy()
                move = rng.uniform(-0.04, 0.04, size=2)
                new_positions[i] += move
                new_positions[i] = np.clip(new_positions[i], 0, 1)
                new_radii = best[:, 2]

            # Containment check
            if not np.all((new_radii <= new_positions[:, 0]) & (new_positions[:, 0] <= 1 - new_radii) &
                          (new_radii <= new_positions[:, 1]) & (new_positions[:, 1] <= 1 - new_radii)):
                continue

            # Overlap check (squared distances)
            dxs = new_positions[:, 0] - new_positions[i, 0]
            dys = new_positions[:, 1] - new_positions[i, 1]
            dists2 = dxs ** 2 + dys ** 2
            dists2[i] = np.inf
            if np.any(dists2 < (new_radii[i] + new_radii) ** 2):
                continue

            new_sum = new_radii.sum()
            delta_obj = new_sum - best_sum

            if delta_obj > 0 or rng.random() < math.exp(delta_obj / T):
                best[:, :2] = new_positions
                best[:, 2] = new_radii
                if new_sum > best_sum:
                    best_sum = new_sum

            T = max(Tmin, T * alpha)

        # Deterministic radius tightening after SA
        best[:, 2] = _tighten_radii(best[:, :2], best[:, 2], iterations=20)

        # Stage 3 – SLSQP refinement
        x0 = np.empty(3 * n)
        x0[:n] = best[:, 0]
        x0[n:2 * n] = best[:, 1]
        x0[2 * n:] = best[:, 2]

        cons = []
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2 * n + i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2 * n + i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n + i] - v[2 * n + i]})
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n + i] - v[2 * n + i]})
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({'type': 'ineq',
                             'fun': lambda v, i=i, j=j: (v[i] - v[j]) ** 2 + (v[n + i] - v[n + j]) ** 2
                             - (v[2 * n + i] + v[2 * n + j]) ** 2})

        def obj(v):
            return -np.sum(v[2 * n:])

        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

        res = opt.minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'ftol': 1e-6, 'maxiter': 12000, 'disp': False})

        circles = np.zeros((n, 3))
        if res.success:
            circles[:, 0] = res.x[:n]
            circles[:, 1] = res.x[n:2 * n]
            circles[:, 2] = res.x[2 * n:]

        # Final deterministic tightening
        circles[:, 2] = _tighten_radii(circles[:, :2], circles[:, 2], iterations=12)

        # Small radius‑only SA for fine‑tuning
        for _ in range(5000):
            i = rng.integers(n)
            new_radii = circles[:, 2].copy()
            delta = rng.uniform(-0.001, 0.001)
            new_radii[i] = max(0.0, new_radii[i] + delta)

            # Check containment
            if not np.all((new_radii <= circles[:, 0]) & (circles[:, 0] <= 1 - new_radii) &
                          (new_radii <= circles[:, 1]) & (circles[:, 1] <= 1 - new_radii)):
                continue

            # Overlap check
            dxs = circles[:, 0] - circles[i, 0]
            dys = circles[:, 1] - circles[i, 1]
            dists2 = dxs ** 2 + dys ** 2
            dists2[i] = np.inf
            if np.any(dists2 < (new_radii[i] + new_radii) ** 2):
                continue

            new_sum = new_radii.sum()
            if new_sum > circles[:, 2].sum():
                circles[:, 2] = new_radii

        sum_r = circles[:, 2].sum()
        if sum_r > best_sum_overall:
            best_sum_overall = sum_r
            best_overall = circles.copy()

    return best_overall


# EVOLVE-BLOCK-END
