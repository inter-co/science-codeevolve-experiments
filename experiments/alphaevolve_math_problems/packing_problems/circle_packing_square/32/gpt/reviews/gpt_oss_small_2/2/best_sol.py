# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

# deterministic seed for reproducibility
_RNG_SEED = 12345
rng = np.random.default_rng(_RNG_SEED)

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

def _grid_initialization(n: int) -> np.ndarray:
    """
    Fallback grid initialization used if the hex lattice fails to produce enough circles.
    """
    cols, rows = 8, 4
    dx, dy = 1.0 / cols, 1.0 / rows
    xs = np.linspace(dx / 2, 1 - dx / 2, cols)
    ys = np.linspace(dy / 2, 1 - dy / 2, rows)
    grid_positions = np.array([(x, y) for y in ys for x in xs])[:n]
    init_radius = min(dx, dy) / 2 * 0.95
    radii = np.full(n, init_radius)
    return np.column_stack((grid_positions, radii))

def _simulated_annealing(circles: np.ndarray, steps: int = 400000,
                         T_init: float = 0.02, alpha: float = 0.995) -> np.ndarray:
    """
    Simulated‑annealing search that perturbs both positions and radii.
    """
    best = circles.copy()
    best_sum = best[:, 2].sum()
    T = T_init
    for step in range(steps):
        i = rng.integers(circles.shape[0])
        new = best.copy()
        if rng.random() < 0.5:
            # Perturb radius
            delta = rng.uniform(-0.015, 0.015)
            new[i, 2] = max(0.0, new[i, 2] + delta)
        else:
            # Perturb position
            move = rng.uniform(-0.07, 0.07, size=2)
            new[i, :2] += move
            new[i, :2] = np.clip(new[i, :2], new[i, 2], 1 - new[i, 2])
        # Containment check
        if not np.all((new[:, 2] <= new[:, 0]) & (new[:, 0] <= 1 - new[:, 2]) &
                      (new[:, 2] <= new[:, 1]) & (new[:, 1] <= 1 - new[:, 2])):
            continue
        # Overlap check
        dxs = new[:, 0] - new[i, 0]
        dys = new[:, 1] - new[i, 1]
        dists2 = dxs ** 2 + dys ** 2
        dists2[i] = np.inf
        if np.any(dists2 < (new[i, 2] + new[:, 2]) ** 2):
            continue
        new_sum = new[:, 2].sum()
        delta_obj = new_sum - best_sum
        if delta_obj > 0 or rng.random() < np.exp(delta_obj / T):
            best = new
            best_sum = new_sum
        T = max(T * alpha, 1e-4)
    return best

def _slsqp_refinement(circles: np.ndarray) -> np.ndarray:
    """
    Final SLSQP refinement enforcing all constraints exactly.
    """
    n = circles.shape[0]
    x0 = np.empty(3 * n)
    x0[:n] = circles[:, 0]
    x0[n:2 * n] = circles[:, 1]
    x0[2 * n:] = circles[:, 2]

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
    if res.success:
        circles[:, 0] = res.x[:n]
        circles[:, 1] = res.x[n:2 * n]
        circles[:, 2] = res.x[2 * n:]
    return circles

def circle_packing32() -> np.ndarray:
    """
    Public API: returns an array of shape (32,3) containing the optimal
    circle positions and radii found by the deterministic simulated annealing
    routine, followed by deterministic tightening and SLSQP refinement.
    """
    start_time = time.time()
    # Stage 1 – Hexagonal lattice initialization
    init = _hex_initialization(32)
    if init.shape[0] < 32:
        init = _grid_initialization(32)
    # Stage 2 – Simulated annealing
    sa = _simulated_annealing(init)
    sa[:, 2] = _tighten_radii(sa[:, :2], sa[:, 2], iterations=80)
    # Stage 3 – SLSQP refinement
    refined = _slsqp_refinement(sa)
    refined[:, 2] = _tighten_radii(refined[:, :2], refined[:, 2], iterations=100)
    # Final local simulated‑annealing to escape small local optima
    T = 0.005
    alpha = 0.99
    steps = 2000
    for _ in range(steps):
        i = rng.integers(32)
        move = rng.uniform(-0.01, 0.01, size=2)
        new_pos = refined[:, :2].copy()
        new_pos[i] += move
        new_pos[i] = np.clip(new_pos[i], 0, 1)
        # Containment check
        if not np.all((refined[:, 2] <= new_pos[:, 0]) & (new_pos[:, 0] <= 1 - refined[:, 2]) &
                      (refined[:, 2] <= new_pos[:, 1]) & (new_pos[:, 1] <= 1 - refined[:, 2])):
            continue
        # Overlap check
        dxs = new_pos[:, 0] - new_pos[i, 0]
        dys = new_pos[:, 1] - new_pos[i, 1]
        dists2 = dxs ** 2 + dys ** 2
        dists2[i] = np.inf
        if np.any(dists2 < (refined[:, 2] + refined[:, 2]) ** 2):
            continue
        new_rads = _tighten_radii(new_pos, refined[:, 2].copy(), iterations=5)
        if new_rads.sum() > refined[:, 2].sum():
            refined[:, :2] = new_pos
            refined[:, 2] = new_rads
    # Final tightening after the local SA
    refined[:, 2] = _tighten_radii(refined[:, :2], refined[:, 2], iterations=20)
    elapsed = time.time() - start_time
    sum_radii = np.sum(refined[:, 2])
    print(f"sum_radii = {sum_radii:.6f}, time = {elapsed:.3f}s")
    return refined


# EVOLVE-BLOCK-END
