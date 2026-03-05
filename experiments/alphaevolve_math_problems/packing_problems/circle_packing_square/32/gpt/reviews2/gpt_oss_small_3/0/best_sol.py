# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
import sys
import math

# Fixed random seed for reproducibility
_rng = np.random.default_rng(42)

# ------------------------------------------------------------------
# Helper functions for radius maximisation (inspired by INSPIRATION 2)
# ------------------------------------------------------------------
def _compute_radii(pos: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """
    Compute maximal radii for given positions respecting containment and pairwise
    non‑overlap constraints.  Radii are recomputed iteratively until convergence.
    """
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]          # (n, n, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))          # (n, n)
    boundary = np.minimum(
        np.minimum(pos[:, 0], 1 - pos[:, 0]),
        np.minimum(pos[:, 1], 1 - pos[:, 1])
    )  # (n,)

    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    inter = dist - radii[None, :]                     # (n, n)
    inter[~mask] = np.inf
    min_inter = np.min(inter, axis=1)                 # (n,)

    new_radii = np.minimum(boundary, min_inter / 2.0)
    new_radii = np.maximum(new_radii, 0.0)
    return new_radii

def _maximise_radii(pos: np.ndarray, max_iter: int = 50, eps: float = 1e-6) -> np.ndarray:
    """
    Iteratively update radii until convergence for a fixed set of positions.
    """
    n = pos.shape[0]
    radii = np.full(n, 0.01)  # start with a small radius
    for _ in range(max_iter):
        new_radii = _compute_radii(pos, radii)
        if np.max(np.abs(new_radii - radii)) < eps:
            break
        radii = new_radii
    return radii

# ------------------------------------------------------------------
# Local refinement routine (simulated‑annealing style)
# ------------------------------------------------------------------
def _local_refine(circles: np.ndarray, max_iter: int = 2000, step: float = 0.01, seed: int = 42) -> np.ndarray:
    """
    Perform a lightweight local search that perturbs circle positions
    and recomputes maximal radii.  Only moves that increase the total
    radius sum are accepted.
    """
    rng = np.random.default_rng(seed)
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    radii = circles[:, 2].copy()
    best_pos, best_radii = pos.copy(), radii.copy()
    best_sum = radii.sum()

    for _ in range(max_iter):
        i = rng.integers(n)
        new_pos = pos.copy()
        new_pos[i] += rng.normal(size=2) * step
        new_pos[i] = np.clip(new_pos[i], 0, 1)

        new_radii = _maximise_radii(new_pos)
        new_sum = new_radii.sum()

        if new_sum > best_sum:
            pos, radii = new_pos, new_radii
            best_pos, best_radii = pos.copy(), radii.copy()
            best_sum = new_sum

    return np.hstack((best_pos, best_radii[:, None]))

# ------------------------------------------------------------------
# LP radius optimiser for fixed positions
# ------------------------------------------------------------------
def _optimize_radii_lp(pos: np.ndarray) -> np.ndarray:
    """
    Given fixed circle centres, solve a linear program to maximise the sum of radii.
    This is a fast, deterministic step that tightens radii after any position
    optimisation.  The formulation follows the one used in INSPIRATION 2.
    """
    n = pos.shape[0]
    c = -np.ones(n)

    A = []
    b = []

    # Containment constraints
    for i in range(n):
        coeff = np.zeros(n)
        coeff[i] = 1
        A.append(coeff); b.append(pos[i, 0])          # x_i
        A.append(coeff); b.append(1 - pos[i, 0])      # 1 - x_i
        A.append(coeff); b.append(pos[i, 1])          # y_i
        A.append(coeff); b.append(1 - pos[i, 1])      # 1 - y_i

    # Non‑overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) + 1e-9
            coeff = np.zeros(n)
            coeff[i] = 1
            coeff[j] = 1
            A.append(coeff); b.append(dist)

    A = np.array(A)
    b = np.array(b)
    bounds = [(0, None)] * n

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    if res.success:
        return res.x
    else:
        return np.full(n, 0.01)

# ------------------------------------------------------------------
# Initialisation strategies
# ------------------------------------------------------------------
def _hex_initialization(n: int):
    """Generate positions and a common radius using a hexagonal lattice."""
    lo, hi = 0.0, 0.5
    for _ in range(50):
        mid = (lo + hi) / 2
        rows = int(math.floor((1 - 2 * mid) / (math.sqrt(3) * mid))) + 1
        cols = int(math.floor((1 - 2 * mid) / (2 * mid))) + 1
        if rows * cols >= n:
            lo = mid
        else:
            hi = mid
    r = lo
    circles = np.zeros((n, 3))
    count = 0
    y = r
    row = 0
    while count < n and y <= 1 - r + 1e-12:
        offset = r if row % 2 == 0 else 2 * r
        x = offset
        while count < n and x <= 1 - r + 1e-12:
            circles[count] = [x, y, r]
            count += 1
            x += 2 * r
        y += math.sqrt(3) * r
        row += 1
    while count < n:
        circles[count] = [r, r, r]
        count += 1
    return circles[:, :2], circles[:, 2]

def _grid_initialization(n: int):
    """6×6 grid truncated to n points with a small random perturbation."""
    grid_cols, grid_rows = 6, 6
    xs = np.linspace(0.5 / grid_cols, 1 - 0.5 / grid_cols, grid_cols)
    ys = np.linspace(0.5 / grid_rows, 1 - 0.5 / grid_rows, grid_rows)
    grid_positions = [(x, y) for y in ys for x in xs]
    init_pos = np.array(grid_positions[:n])
    init_pos += _rng.uniform(-0.01, 0.01, size=init_pos.shape)
    init_pos = np.clip(init_pos, 0.0, 1.0)
    init_r = np.full(n, 0.05)
    return init_pos, init_r

# ------------------------------------------------------------------
# Main public API
# ------------------------------------------------------------------
def circle_packing32() -> np.ndarray:
    """
    Places 32 non‑overlapping circles in the unit square to maximize the sum of radii.
    The algorithm explores several deterministic initializations,
    refines each with SLSQP, and finally applies a lightweight local refinement
    to squeeze extra radius.  After the final refinement, a fast LP step tightens
    the radii for the obtained positions.
    """
    n = 32
    init_methods = [_hex_initialization, _grid_initialization]
    best_sum = -np.inf
    best_circles = None

    # Helper to extract indices
    def idx_x(i): return i
    def idx_y(i): return n + i
    def idx_r(i): return 2 * n + i

    # Objective: minimize negative sum of radii
    def objective(v):
        r = v[2 * n:3 * n]
        return -np.sum(r)

    for init_fn in init_methods:
        init_pos, init_r = init_fn(n)
        x0 = np.concatenate([init_pos[:, 0], init_pos[:, 1], init_r])

        bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

        cons = []

        # Vectorized pairwise non‑overlap constraints
        pair_indices = np.triu_indices(n, k=1)
        i_pairs, j_pairs = pair_indices

        def pair_constraints(v):
            xi = v[idx_x(i_pairs)]
            yi = v[idx_y(i_pairs)]
            ri = v[idx_r(i_pairs)]
            xj = v[idx_x(j_pairs)]
            yj = v[idx_y(j_pairs)]
            rj = v[idx_r(j_pairs)]
            dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
            return dist_sq - (ri + rj) ** 2

        cons.append({'type': 'ineq', 'fun': pair_constraints})

        # Boundary constraints (vectorized)
        def boundary_constraints(v):
            x = v[0:n]
            y = v[n:2*n]
            r = v[2*n:3*n]
            return np.concatenate([
                x - r,                    # x >= r
                1.0 - x - r,              # x <= 1 - r
                y - r,                    # y >= r
                1.0 - y - r               # y <= 1 - r
            ])

        cons.append({'type': 'ineq', 'fun': boundary_constraints})

        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9, 'maxiter': 8000, 'disp': False}
        )

        if not res.success:
            continue

        sol = res.x
        circles = np.empty((n, 3))
        for i in range(n):
            circles[i, 0] = sol[idx_x(i)]
            circles[i, 1] = sol[idx_y(i)]
            circles[i, 2] = sol[idx_r(i)]

        radius_sum = np.sum(circles[:, 2])
        if radius_sum > best_sum:
            best_sum = radius_sum
            best_circles = circles

    # Fallback: if all runs failed, use hexagonal lattice
    if best_circles is None:
        pos, r = _hex_initialization(n)
        best_circles = np.column_stack((pos, r))

    # Final local refinement to squeeze extra radius
    best_circles = _local_refine(best_circles, max_iter=2000, step=0.01, seed=42)

    # Tighten radii for the obtained positions using a fast LP
    best_circles[:, 2] = _optimize_radii_lp(best_circles[:, :2])

    return best_circles


# EVOLVE-BLOCK-END
