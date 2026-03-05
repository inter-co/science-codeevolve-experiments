# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _tighten_radii(pos: np.ndarray, rad: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Deterministically tighten each radius to the maximum allowed by the current positions
    and other circles.  This is a local refinement step that often improves the total sum.
    """
    n = pos.shape[0]
    for _ in range(iterations):
        # Boundary limits for each circle
        max_r = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]),
                           np.minimum(pos[:, 1], 1 - pos[:, 1]))
        # Pairwise distances minus current radii
        diff = pos[:, None, :] - pos[None, :, :]          # shape (n, n, 2)
        dists = np.sqrt(np.sum(diff ** 2, axis=2)) - rad[None, :]
        np.fill_diagonal(dists, np.inf)                    # ignore self
        min_dists = np.min(dists, axis=1)                  # nearest neighbour clearance
        max_r = np.minimum(max_r, min_dists)               # tightest possible radius
        rad = np.maximum(0.0, max_r)                       # enforce non‑negative
    return rad

# You can define functions outside the main function below.
def _hex_initialization(n: int) -> np.ndarray:
    """
    Generate an initial set of n circles using a hexagonal lattice.
    The function searches over lattice spacing to find a configuration
    that yields a large sum of radii.  The returned array has shape (n,3)
    with columns (x, y, r).
    """
    best_sum = 0.0
    best_circles = None
    for s in np.linspace(0.05, 0.5, 46):
        vert_spacing = s * np.sqrt(3) / 2
        xs = np.arange(0, 1 + 1e-9, s)
        ys = np.arange(0, 1 + 1e-9, vert_spacing)
        points = []
        for i, y in enumerate(ys):
            offset = (s / 2) if (i % 2) else 0.0
            for x in xs:
                px = x + offset
                if px <= 1:
                    points.append((px, y))
        points = np.array(points)
        inside = (points[:, 0] >= 0) & (points[:, 0] <= 1) & (points[:, 1] >= 0) & (points[:, 1] <= 1)
        points = points[inside]
        if len(points) < n:
            continue
        radii = np.full(len(points), s / 2)
        boundary = np.minimum(
            np.minimum(points[:, 0], 1 - points[:, 0]),
            np.minimum(points[:, 1], 1 - points[:, 1])
        )
        radii = np.minimum(radii, boundary)
        idx = np.argsort(-radii)
        selected = idx[:n]
        sum_r = radii[selected].sum()
        if sum_r > best_sum:
            best_sum = sum_r
            best_circles = np.column_stack((points[selected], radii[selected]))
    if best_circles is None:
        best_circles = np.empty((0, 3))
    return best_circles
# Remember that any function used in parallel computation must be defined globally and not locally.

def circle_packing32() -> np.ndarray:
    start_time = time.time()
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
    rng = np.random.default_rng(12345)

    # Stage 1 – Hexagonal lattice initialization
    init = _hex_initialization(32)
    if init.shape[0] < 32:
        # Fallback to grid if hex fails
        cols, rows = 8, 4
        dx, dy = 1.0 / cols, 1.0 / rows
        xs = np.linspace(dx / 2, 1 - dx / 2, cols)
        ys = np.linspace(dy / 2, 1 - dy / 2, rows)
        grid_positions = np.array([(x, y) for y in ys for x in xs])[:32]
        init_radius = min(dx, dy) / 2 * 0.95
        radii = np.full(32, init_radius)
        init = np.column_stack((grid_positions, radii))
    circles = init.copy()

    # Stage 2 – Simulated annealing
    T = 0.02
    Tmin = 1e-4
    alpha = 0.995
    steps = 800000

    best = circles.copy()
    best_sum = best[:, 2].sum()

    for step in range(steps):
        i = rng.integers(n)

        if rng.random() < 0.5:
            # Perturb radius
            new_radii = best[:, 2].copy()
            delta = rng.uniform(-0.015, 0.015)
            new_radii[i] = max(0.0, new_radii[i] + delta)
            new_positions = best[:, :2]
        else:
            # Perturb position
            new_positions = best[:, :2].copy()
            move = rng.uniform(-0.07, 0.07, size=2)
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
    best[:, 2] = _tighten_radii(best[:, :2], best[:, 2], iterations=80)

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
                       options={'ftol': 1e-12, 'maxiter': 60000, 'disp': False})

    circles = np.zeros((n, 3))
    if res.success:
        circles[:, 0] = res.x[:n]
        circles[:, 1] = res.x[n:2 * n]
        circles[:, 2] = res.x[2 * n:]

    # Final deterministic tightening
    circles[:, 2] = _tighten_radii(circles[:, :2], circles[:, 2], iterations=80)

    # Local radius refinement after SLSQP
    rng = np.random.default_rng(12345)
    current_sum = np.sum(circles[:, 2])
    for _ in range(40000):
        i = rng.integers(n)
        new_r = circles[i, 2] + rng.uniform(-0.005, 0.005)
        new_r = np.clip(new_r, 0.0, 0.5)
        # Containment
        if not (new_r <= circles[i, 0] <= 1 - new_r and
                new_r <= circles[i, 1] <= 1 - new_r):
            continue
        # Overlap
        dx = circles[:, 0] - circles[i, 0]
        dy = circles[:, 1] - circles[i, 1]
        dists2 = dx**2 + dy**2
        dists2[i] = np.inf
        if np.any(dists2 < (new_r + circles[:, 2])**2):
            continue
        new_sum = current_sum - circles[i, 2] + new_r
        if new_sum > current_sum:
            circles[i, 2] = new_r
            current_sum = new_sum

    sum_radii = np.sum(circles[:, 2])
    elapsed = time.time() - start_time
    print(f"sum_radii = {sum_radii:.6f}, time = {elapsed:.3f}s")
    return circles


# EVOLVE-BLOCK-END
