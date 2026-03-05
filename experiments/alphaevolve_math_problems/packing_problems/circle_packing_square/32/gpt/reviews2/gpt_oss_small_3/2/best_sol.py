# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def local_refine(circles: np.ndarray,
                 steps: int = 20000,
                 radius_step: float = 0.005,
                 position_step: float = 0.01) -> np.ndarray:
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    r = circles[:, 2].copy()

    np.random.seed(42)

    best_pos = pos.copy()
    best_r = r.copy()
    best_sum = r.sum()

    for _ in range(steps):
        i = np.random.randint(n)

        if np.random.rand() < 0.5:
            delta = (np.random.rand() * 2.0 - 1.0) * radius_step
            new_r = r[i] + delta
            new_r = max(new_r, 0.0)

            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if j == i:
                    continue
                d = np.linalg.norm(pos[i] - pos[j]) - r[j]
                if d < max_r:
                    max_r = d
            max_r = max(max_r, 0.0)
            new_r = min(new_r, max_r)

            if new_r > r[i]:
                r[i] = new_r
                cur_sum = r.sum()
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_pos = pos.copy()
                    best_r = r.copy()
        else:
            delta = (np.random.rand(2) * 2.0 - 1.0) * position_step
            new_pos = pos[i] + delta
            new_pos[0] = np.clip(new_pos[0], r[i], 1.0 - r[i])
            new_pos[1] = np.clip(new_pos[1], r[i], 1.0 - r[i])

            ok = True
            for j in range(n):
                if j == i:
                    continue
                if np.linalg.norm(new_pos - pos[j]) < r[i] + r[j]:
                    ok = False
                    break

            if ok:
                pos[i] = new_pos
                cur_sum = r.sum()
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_pos = pos.copy()
                    best_r = r.copy()

    return np.column_stack((best_pos, best_r))

def evolutionary_refine(circles: np.ndarray,
                         n_mutations: int = 10,
                         rng: np.random.Generator = None) -> np.ndarray:
    """
    Lightweight evolutionary mutation strategy to escape local optima.
    Each mutation selects a random circle, perturbs its radius and position,
    and accepts the change only if it yields a higher total radius sum
    while preserving all constraints.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    best = circles.copy()
    best_sum = best[:, 2].sum()
    n = best.shape[0]

    for _ in range(n_mutations):
        i = rng.integers(n)

        # Propose new radius
        new_r = best[i, 2] + rng.normal(scale=0.01)
        new_r = max(0.0, min(new_r, 0.5))

        # Propose new position
        new_pos = best[i, :2] + rng.normal(scale=0.02, size=2)
        new_pos[0] = np.clip(new_pos[0], new_r, 1.0 - new_r)
        new_pos[1] = np.clip(new_pos[1], new_r, 1.0 - new_r)

        # Check overlap constraints
        ok = True
        for j in range(n):
            if j == i:
                continue
            if np.linalg.norm(new_pos - best[j, :2]) < new_r + best[j, 2]:
                ok = False
                break

        if ok:
            new_circles = best.copy()
            new_circles[i, :2] = new_pos
            new_circles[i, 2] = new_r
            new_sum = new_circles[:, 2].sum()
            if new_sum > best_sum:
                best = new_circles
                best_sum = new_sum

    return best

def circle_packing32() -> np.ndarray:
    n = 32
    rng = np.random.default_rng(42)

    # Structured initial guess: hexagonal lattice inside the unit square
    s = 0.12
    points = []
    for i in range(8):
        for j in range(9):
            x = (j + 0.5 * (i % 2)) * s
            y = i * s * np.sqrt(3) / 2
            if x <= 1 and y <= 1:
                points.append((x, y))
    pts = np.array(points)

    # Pick the n points with largest minimal distance to the boundary
    d_boundary = np.minimum(np.minimum(pts[:, 0], 1 - pts[:, 0]),
                            np.minimum(pts[:, 1], 1 - pts[:, 1]))
    idx = np.argsort(-d_boundary)[:n]
    x0 = np.empty(3 * n)
    x0[:n] = pts[idx, 0]
    x0[n:2 * n] = pts[idx, 1]

    # Initial radii: min(boundary distance, half nearest‑neighbor distance)
    d_boundary = np.minimum(np.minimum(pts[idx, 0], 1 - pts[idx, 0]),
                            np.minimum(pts[idx, 1], 1 - pts[idx, 1]))
    diff = pts[idx][:, None, :] - pts[idx][None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    d_nearest = np.min(dist, axis=1)
    init_radii = np.minimum(d_boundary, d_nearest / 2.0)
    x0[2 * n:] = init_radii

    def obj(v):
        return -np.sum(v[2 * n :])

    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    cons = []

    # Containment constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n + i] - v[2 * n + i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n + i] - v[2 * n + i]})

    # Non‑overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[i] - v[j]) ** 2 + (v[n + i] - v[n + j]) ** 2
                - (v[2 * n + i] + v[2 * n + j]) ** 2
            })

    # First SLSQP run
    res = minimize(
        obj,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 15000, 'ftol': 1e-9, 'disp': False}
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    sol = res.x
    circles = np.column_stack((sol[:n], sol[n:2 * n], sol[2 * n:]))

    # Deterministic local refinement
    circles = local_refine(circles, steps=20000, radius_step=0.005, position_step=0.01)

    # Final polishing SLSQP
    x0_refined = circles.ravel()
    res2 = minimize(
        obj,
        x0_refined,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False}
    )
    if res2.success:
        sol = res2.x
        circles = np.column_stack((sol[:n], sol[n:2 * n], sol[2 * n:]))
    return circles


# EVOLVE-BLOCK-END
