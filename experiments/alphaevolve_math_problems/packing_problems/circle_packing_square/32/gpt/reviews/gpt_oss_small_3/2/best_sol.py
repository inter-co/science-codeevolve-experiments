# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------------
# Penalty‑based objective for simulated annealing
# ------------------------------------------------------------------
def _compute_objective(circles: np.ndarray, penalty_factor: float = 1e4) -> float:
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    sum_r = rs.sum()

    # Containment penalties
    cont = np.minimum(np.minimum(xs, 1 - xs), np.minimum(ys, 1 - ys))
    cont_pen = np.sum(np.maximum(0, rs - cont) ** 2)

    # Pairwise overlap penalties
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    d = np.sqrt(dx * dx + dy * dy)
    rsum = rs[:, None] + rs[None, :]
    mask = np.triu(np.ones_like(d), k=1).astype(bool)
    overlap = d[mask] - rsum[mask]
    overlap_pen = np.sum(np.maximum(0, -overlap) ** 2)

    return sum_r - penalty_factor * (cont_pen + overlap_pen)

# ------------------------------------------------------------------
# Enforce all constraints by clipping radii
# ------------------------------------------------------------------
def _enforce_constraints(circles: np.ndarray, iterations: int = 20) -> np.ndarray:
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    for _ in range(iterations):
        # Wall constraints
        rs = np.minimum(rs, xs)
        rs = np.minimum(rs, 1 - xs)
        rs = np.minimum(rs, ys)
        rs = np.minimum(rs, 1 - ys)

        # Pairwise constraints
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        allowed = d - rs[None, :]
        np.fill_diagonal(allowed, np.inf)
        min_allowed = np.min(allowed, axis=1)
        rs = np.minimum(rs, min_allowed)
        rs = np.maximum(rs, 0.0)

    circles[:, 2] = rs
    return circles

# ------------------------------------------------------------------
# Deterministic simulated annealing
# ------------------------------------------------------------------
def _simulated_annealing(
    n: int,
    max_iter: int = 20000,
    seed: int = 0,
    T0: float = 0.1,
    cooling: float = 0.9995,
    pos_step: float = 0.05,
    rad_step: float = 0.02,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Initialize with very small radii
    circles = rng.uniform(0.01, 0.99, size=(n, 3))
    circles[:, 2] = 0.01
    circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], 1 - circles[:, 2])
    circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], 1 - circles[:, 2])

    best = circles.copy()
    best_obj = _compute_objective(best)
    T = T0

    for _ in range(max_iter):
        new = best.copy()
        idx = rng.integers(n)

        if rng.random() < 0.5:
            # Move the centre
            delta = (rng.random(2) - 0.5) * pos_step
            new[idx, 0:2] += delta
            new[idx, 0] = np.clip(new[idx, 0], new[idx, 2], 1 - new[idx, 2])
            new[idx, 1] = np.clip(new[idx, 1], new[idx, 2], 1 - new[idx, 2])
        else:
            # Change the radius
            delta_r = (rng.random() - 0.5) * rad_step
            new_r = new[idx, 2] + delta_r
            new_r = max(0.001, min(new_r, 0.5))
            new[idx, 2] = new_r
            new[idx, 0] = np.clip(new[idx, 0], new_r, 1 - new_r)
            new[idx, 1] = np.clip(new[idx, 1], new_r, 1 - new_r)

        new_obj = _compute_objective(new)
        if new_obj > best_obj or rng.random() < np.exp((new_obj - best_obj) / T):
            best = new
            best_obj = new_obj

        T *= cooling

    return best

# ------------------------------------------------------------------
# Local refinement with SLSQP
# ------------------------------------------------------------------
def _local_refine_slsqp(circles: np.ndarray, maxiter: int = 5000) -> np.ndarray:
    n = circles.shape[0]
    x0 = circles.flatten()
    bounds = [(0, None)] * 3 * n
    cons = []

    # Containment constraints
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda x, i=i: x[3 * i] - x[3 * i + 2]})
        cons.append({"type": "ineq", "fun": lambda x, i=i: x[3 * i + 1] - x[3 * i + 2]})
        cons.append({"type": "ineq", "fun": lambda x, i=i: 1 - x[3 * i] - x[3 * i + 2]})
        cons.append({"type": "ineq", "fun": lambda x, i=i: 1 - x[3 * i + 1] - x[3 * i + 2]})

    # Non‑overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda x, i=i, j=j: (
                        x[3 * i] - x[3 * j]
                    ) ** 2
                    + (x[3 * i + 1] - x[3 * j + 1]) ** 2
                    - (x[3 * i + 2] + x[3 * j + 2]) ** 2,
                }
            )

    def obj(x):
        return -np.sum(x[2::3])

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-9, "maxiter": maxiter, "disp": False},
    )

    if not res.success:
        return circles

    sol = res.x
    refined = np.column_stack((sol[0::3], sol[1::3], sol[2::3]))
    return refined

# ------------------------------------------------------------------
# Physics‑based radius expansion
# ------------------------------------------------------------------
def circle_packing32_physics(circles: np.ndarray, delta: float = 1e-4, max_iter: int = 2000) -> np.ndarray:
    """
    Refines a given circle configuration by incrementally increasing radii while preserving constraints.
    """
    n = circles.shape[0]
    grid = circles[:, :2]
    radii = circles[:, 2].copy()

    for _ in range(max_iter):
        improved = False
        for i in range(n):
            xi, yi = grid[i]
            new_r = radii[i] + delta
            if new_r > min(xi, yi, 1 - xi, 1 - yi):
                continue
            overlap = False
            for j in range(n):
                if i == j:
                    continue
                xj, yj = grid[j]
                rj = radii[j]
                if (xi - xj) ** 2 + (yi - yj) ** 2 < (new_r + rj) ** 2 - 1e-12:
                    overlap = True
                    break
            if not overlap:
                radii[i] = new_r
                improved = True
        if not improved:
            break

    return np.column_stack([grid, radii])

# ------------------------------------------------------------------
# Main entry point: deterministic SA + physics + SLSQP
# ------------------------------------------------------------------
def circle_packing32(seed: int = 0, max_iter: int = 2000, tol: float = 1e-6) -> np.ndarray:
    """
    Returns a 32×3 array of (x, y, r) for a near‑optimal circle packing.
    The routine runs a deterministic simulated‑annealing search, enforces all constraints,
    refines the result with a physics‑based radius expansion, and finally applies SLSQP.
    """
    n = 32

    # Step 1: simulated annealing
    sa_circles = _simulated_annealing(n, max_iter=20000, seed=seed)

    # Step 2: enforce constraints (robustness)
    sa_circles = _enforce_constraints(sa_circles, iterations=20)

    # Step 3: physics‑based refinement
    phys_circles = circle_packing32_physics(sa_circles, delta=1e-4, max_iter=2000)

    # Step 4: local SLSQP refinement
    refined = _local_refine_slsqp(phys_circles, maxiter=5000)

    # Fallback if all fails
    if refined is None or not np.isfinite(refined).all():
        # Simple equal‑radius grid fallback
        xs = np.linspace(0.5 / 8, 1 - 0.5 / 8, 8)
        ys = np.linspace(0.5 / 4, 1 - 0.5 / 4, 4)
        grid = np.array([(x, y) for x in xs for y in ys])[:n]
        return np.column_stack([grid, np.full(n, 0.0625)])

    return refined


# EVOLVE-BLOCK-END
