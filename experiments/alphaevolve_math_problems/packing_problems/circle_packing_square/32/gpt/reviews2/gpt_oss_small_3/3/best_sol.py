# EVOLVE-BLOCK-START
import numpy as np
import math
import logging

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Helper: generate a hexagonal initial packing guess
# ------------------------------------------------------------------
def _hexagonal_initial(n: int, r0: float = 0.05) -> np.ndarray:
    """
    Create a deterministic hexagonal lattice of `n` circles with radius `r0`.
    """
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    dx = 2 * r0
    dy = r0 * np.sqrt(3)
    circles = np.zeros((n, 3))
    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= n:
                break
            x = r0 + col * dx
            y = r0 + row * dy
            circles[idx] = [x, y, r0]
            idx += 1
    return circles

# ------------------------------------------------------------------
# Constraint checker (used by physics scaling and local search)
# ------------------------------------------------------------------
def _check_constraints(positions: np.ndarray, radii: np.ndarray) -> bool:
    """
    Verify that all circles are inside the unit square and non-overlapping.
    """
    # Boundary constraints
    if np.any(radii > positions[:, 0]) or np.any(radii > 1 - positions[:, 0]):
        return False
    if np.any(radii > positions[:, 1]) or np.any(radii > 1 - positions[:, 1]):
        return False

    # Pairwise overlap constraints
    n = positions.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < radii[i] + radii[j] - 1e-12:  # small tolerance
                return False
    return True

# ------------------------------------------------------------------
# Physics‑based scaling: uniformly enlarge radii until a violation occurs
# ------------------------------------------------------------------
def _physics_scale(positions: np.ndarray, radii: np.ndarray, max_iter: int = 2000) -> np.ndarray:
    """
    Starting from the given radii, uniformly scale all radii by a factor > 1
    until constraints are violated, then backtrack slightly.  Returns the
    best radii found.
    """
    best_radii = radii.copy()
    best_sum = radii.sum()
    scale = 1.05  # initial scaling factor > 1

    for _ in range(max_iter):
        new_radii = best_radii * scale
        if _check_constraints(positions, new_radii):
            best_radii = new_radii
            best_sum = best_radii.sum()
        else:
            scale *= 0.99  # reduce scaling factor gradually
            if scale < 1.001:
                break  # cannot scale further

    return best_radii

# ------------------------------------------------------------------
# Main constructor
# ------------------------------------------------------------------
def local_refine(circles: np.ndarray,
                 steps: int = 6000,
                 radius_step: float = 0.01,
                 position_step: float = 0.02) -> np.ndarray:
    """
    Perform a deterministic local search to improve a circle packing.
    """
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
            # Radius update
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
            # Position update
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

    return np.column_stack((best_pos, best_r.reshape(-1, 1)))

def circle_packing32() -> np.ndarray:
    """
    Construct a packing of 32 non‑overlapping circles inside the unit square
    that maximizes the sum of radii.  The function uses a deterministic
    hexagonal initialization, SLSQP optimization, and a deterministic
    local search refinement.
    """
    import numpy as np
    from scipy.optimize import minimize

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
    d_boundary = np.minimum(np.minimum(pts[:, 0], 1 - pts[:, 0]),
                            np.minimum(pts[:, 1], 1 - pts[:, 1]))
    idx = np.argsort(-d_boundary)[:n]
    x0 = np.empty(3 * n)
    x0[:n] = pts[idx, 0]
    x0[n:2 * n] = pts[idx, 1]
    # Compute initial radii from the lattice: min(boundary, half nearest‑neighbor distance)
    d_boundary = np.minimum(np.minimum(pts[idx, 0], 1 - pts[idx, 0]),
                            np.minimum(pts[idx, 1], 1 - pts[idx, 1]))
    diff = pts[idx][:, None, :] - pts[idx][None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    d_nearest = np.min(dist, axis=1)
    init_radii = np.minimum(d_boundary, d_nearest / 2.0)
    x0[2 * n:] = init_radii

    # Objective: negative sum of radii
    def obj(v):
        return -np.sum(v[2 * n :])

    # Bounds
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    # Constraints
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

    # Run optimizer with more iterations
    res = minimize(
        obj,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 12000, 'ftol': 1e-9, 'disp': False}
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    sol = res.x
    circles = np.column_stack((sol[:n], sol[n:2 * n], sol[2 * n:]))

    # Deterministic local refinement with a larger search budget
    circles = local_refine(circles, steps=40000)

    # Final polishing with SLSQP
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
    # Re‑apply deterministic local refinement to capture any remaining improvements
    circles = local_refine(circles, steps=20000)
    return circles


# EVOLVE-BLOCK-END
