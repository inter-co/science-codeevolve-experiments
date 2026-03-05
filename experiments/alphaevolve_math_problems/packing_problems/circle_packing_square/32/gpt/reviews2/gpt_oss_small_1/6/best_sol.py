# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def circle_packing32() -> np.ndarray:
    """
    Deterministic, high‑performance circle packing of 32 circles in a unit square.
    Pipeline (inspired by the best‑scoring reference solution):
    1. 6×6 hexagonal‑offset grid (first 32 cells) → good starting layout.
    2. Linear program to compute the maximal radii for those fixed positions.
    3. One high‑accuracy SLSQP run (with several perturbed starts) to jointly optimise
       positions and radii while respecting all constraints.
    4. Greedy radius‑squeeze step that iteratively enlarges each circle to its
       maximum allowed radius given the current positions.
    5. Optional tiny radius‑increment loop to exploit any remaining slack.
    Determinism is guaranteed by a fixed random seed and the absence of stochastic
    optimisation components.
    """
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # 1. Hexagonal‑grid initialization (6×6, first 32 cells)
    # ------------------------------------------------------------------
    rows, cols = 6, 6
    spacing_x = 1.0 / (cols + 1)   # 0.125
    spacing_y = 1.0 / (rows + 1)   # ≈0.1667
    init_x, init_y = [], []
    for j in range(rows):
        for i in range(cols):
            if len(init_x) >= 32:
                break
            offset = 0.5 * spacing_x if j % 2 == 1 else 0.0
            init_x.append((i + 1) * spacing_x + offset)
            init_y.append((j + 1) * spacing_y)
        if len(init_x) >= 32:
            break
    init_x = np.array(init_x, dtype=np.float64)
    init_y = np.array(init_y, dtype=np.float64)

    # ------------------------------------------------------------------
    # 2. LP for maximal radii at fixed positions
    # ------------------------------------------------------------------
    def _lp_max_radii(pos: np.ndarray) -> np.ndarray:
        n = pos.shape[0]
        c = -np.ones(n)  # maximise radii
        bounds = []
        for i in range(n):
            x, y = pos[i]
            max_r = min(x, 1 - x, y, 1 - y)
            bounds.append((0.0, max_r))
        A_ub = []
        b_ub = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.hypot(dx, dy)
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        res = minimize(
            lambda r: -np.sum(r),
            np.full(n, 0.01),
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "ineq", "fun": lambda r: b_ub - A_ub @ r},
            options={"ftol": 1e-9, "maxiter": 5000, "disp": False},
        )
        if res.success:
            return res.x
        # Fallback to trivial radii if LP fails
        return np.zeros(n)

    init_radii = _lp_max_radii(np.column_stack([init_x, init_y]))

    # ------------------------------------------------------------------
    # 3. SLSQP optimisation (multi‑start with perturbations)
    # ------------------------------------------------------------------
    n = 32
    v0 = np.concatenate([init_x, init_y, init_radii])
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    def objective(v: np.ndarray) -> float:
        return -np.sum(v[64:])  # radii start at index 64

    def constraints(v: np.ndarray) -> np.ndarray:
        x = v[:32]
        y = v[32:64]
        r = v[64:]
        cons = []

        # Containment
        cons.extend(x - r)
        cons.extend(y - r)
        cons.extend(1.0 - x - r)
        cons.extend(1.0 - y - r)

        # Non‑overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist2 = dx * dx + dy * dy
        rsum = r[:, None] + r[None, :]
        mask = np.triu_indices(32, k=1)
        cons.extend(dist2[mask] - rsum[mask] ** 2)

        return np.array(cons)

    cons_dict = {"type": "ineq", "fun": constraints}

    best_res = None
    best_sum = -np.inf

    for _ in range(12):
        # Perturb positions slightly
        pert = rng.uniform(-0.02, 0.02, size=(32, 2))
        x_start = np.clip(init_x + pert[:, 0], 0.01, 0.99)
        y_start = np.clip(init_y + pert[:, 1], 0.01, 0.99)
        v_start = np.concatenate([x_start, y_start, init_radii])

        res = minimize(
            objective,
            v_start,
            method="SLSQP",
            bounds=bounds,
            constraints=cons_dict,
            options={"maxiter": 20000, "ftol": 1e-9, "disp": False},
        )

        if res.success:
            sum_r = -objective(res.x)
            if sum_r > best_sum:
                best_sum = sum_r
                best_res = res

    # Final refinement from best solution
    if best_res is not None:
        v_start = best_res.x
        res_refine = minimize(
            objective,
            v_start,
            method="SLSQP",
            bounds=bounds,
            constraints=cons_dict,
            options={"maxiter": 30000, "ftol": 1e-9, "disp": False},
        )
        if res_refine.success:
            best_res = res_refine

    # ------------------------------------------------------------------
    # 4. Greedy radius‑squeeze after positions are fixed
    # ------------------------------------------------------------------
    if best_res is not None and best_res.success:
        x_opt = best_res.x[:32]
        y_opt = best_res.x[32:64]
        r_opt = best_res.x[64:]
    else:
        x_opt, y_opt, r_opt = init_x, init_y, init_radii

    # Iteratively enlarge each circle to the maximum allowed radius
    for _ in range(20):
        changed = False
        centers = np.column_stack([x_opt, y_opt])
        dists = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        boundary_limits = np.minimum(
            np.minimum(x_opt, 1 - x_opt),
            np.minimum(y_opt, 1 - y_opt),
        )
        neigh_limits = dists - r_opt[:, None]
        np.fill_diagonal(neigh_limits, np.inf)
        max_r = np.minimum(boundary_limits, neigh_limits.min(axis=1))
        new_r = np.maximum(0.0, max_r)
        if np.any(np.abs(new_r - r_opt) > 1e-9):
            r_opt = new_r
            changed = True
        if not changed:
            break

    # Optional tiny radius‑increment loop to exploit remaining slack
    for _ in range(2000):
        for i in range(32):
            new_r = r_opt[i] + 1e-4
            if new_r > 0.5:
                continue
            # Check boundary
            if x_opt[i] - new_r < 0 or x_opt[i] + new_r > 1:
                continue
            if y_opt[i] - new_r < 0 or y_opt[i] + new_r > 1:
                continue
            # Check overlap with others
            dx = x_opt[i] - x_opt
            dy = y_opt[i] - y_opt
            dist_sq = dx * dx + dy * dy
            dist_sq[i] = np.inf
            required_sq = (new_r + r_opt) ** 2
            if np.any(dist_sq < required_sq):
                continue
            r_opt[i] = new_r

    circles = np.column_stack([x_opt, y_opt, r_opt])
    return circles


# EVOLVE-BLOCK-END
