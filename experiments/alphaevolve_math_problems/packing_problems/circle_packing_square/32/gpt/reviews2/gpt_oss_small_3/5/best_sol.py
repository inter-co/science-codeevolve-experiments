# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------------
# Deterministic local refinement after SLSQP
# ------------------------------------------------------------------
def local_refine(circles: np.ndarray,
                 steps: int = 60000,
                 radius_step: float = 0.02,
                 position_step: float = 0.04) -> np.ndarray:
    """
    Perform a deterministic local search to improve a circle packing.

    Parameters
    ----------
    circles : np.ndarray
        Array of shape (n,3) with (x,y,r) for each circle.
    steps : int, default 60000
        Number of random modification steps.
    radius_step : float, default 0.02
        Maximum change in radius per step.
    position_step : float, default 0.04
        Maximum change in position per step.

    Returns
    -------
    np.ndarray
        Improved circle packing.
    """
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    r = circles[:, 2].copy()

    # Use a dedicated RNG for reproducibility and speed
    rng = np.random.default_rng(42)

    best_pos = pos.copy()
    best_r = r.copy()
    best_sum = r.sum()

    # Early‑exit counter: stop if no improvement in last 2000 steps
    no_improve = 0
    for _ in range(steps):
        i = rng.integers(n)

        if rng.random() < 0.5:
            # Radius update
            delta = (rng.random() * 2.0 - 1.0) * radius_step
            new_r = r[i] + delta
            new_r = max(new_r, 0.0)

            # Compute maximum allowed radius given current layout
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            # Vectorized distance to other circles
            d = np.linalg.norm(pos[i] - pos, axis=1) - r
            d[i] = np.inf  # ignore self
            max_r = min(max_r, d.min())
            max_r = max(max_r, 0.0)
            new_r = min(new_r, max_r)

            if new_r > r[i]:
                r[i] = new_r
                cur_sum = r.sum()
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_pos = pos.copy()
                    best_r = r.copy()
                    no_improve = 0
                else:
                    no_improve += 1
        else:
            # Position update
            delta = (rng.random(2) * 2.0 - 1.0) * position_step
            new_pos = pos[i] + delta
            new_pos[0] = np.clip(new_pos[0], r[i], 1.0 - r[i])
            new_pos[1] = np.clip(new_pos[1], r[i], 1.0 - r[i])

            # Check overlaps with vectorized distances
            d = np.linalg.norm(new_pos - pos, axis=1)
            d[i] = np.inf  # ignore self
            if (d >= r[i] + r).all():
                pos[i] = new_pos
                cur_sum = r.sum()
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_pos = pos.copy()
                    best_r = r.copy()
                    no_improve = 0
                else:
                    no_improve += 1

        if no_improve >= 2000:
            break

    return np.column_stack((best_pos, best_r))

# ------------------------------------------------------------------
# A deterministic optimization-based approach using SLSQP
# ------------------------------------------------------------------
# The strategy is to formulate the circle packing as a constrained
# nonlinear optimization problem.  The decision vector consists of
# 3*n variables: x_i, y_i and r_i for each circle.  The objective is
# to maximize the sum of radii, which we implement as minimizing the
# negative sum.  All geometric constraints (containment and
# non-overlap) are expressed as inequality constraints suitable for
# SLSQP.  A fixed random seed guarantees reproducibility.
# ------------------------------------------------------------------

def circle_packing32() -> np.ndarray:
    """
    Places 32 non-overlapping circles in the unit square in order to maximize the sum of radii.

    Returns:
        circles: np.array of shape (32,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 32
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Structured initial guess: hexagonal lattice inside the unit square
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
    # Compute initial radii from the lattice: min(boundary, half nearest‑neighbor distance)
    # ------------------------------------------------------------------
    # Boundary distances
    d_boundary = np.minimum(np.minimum(pts[idx, 0], 1 - pts[idx, 0]),
                            np.minimum(pts[idx, 1], 1 - pts[idx, 1]))
    # Pairwise distances
    diff = pts[idx][:, None, :] - pts[idx][None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    d_nearest = np.min(dist, axis=1)
    # Initial radii
    init_radii = np.minimum(d_boundary, d_nearest / 2.0)
    x0[2 * n:] = init_radii

    # Objective: negative sum of radii (SLSQP minimizes)
    def obj(v):
        return -np.sum(v[2 * n :])

    # Bounds: x, y in [0,1]; radii in [0,0.5] (0.5 is a safe upper bound)
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n

    # Constraints
    cons = []

    # Containment constraints: each circle must stay inside the unit square
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2 * n + i]})          # x_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2 * n + i]})    # 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n + i] - v[2 * n + i]})      # y_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n + i] - v[2 * n + i]})# 1 - y_i - r_i >= 0

    # Non-overlap constraints: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[i] - v[j]) ** 2 + (v[n + i] - v[n + j]) ** 2
                - (v[2 * n + i] + v[2 * n + j]) ** 2
            })

    # Run the optimizer with a larger iteration budget
    # Multiple deterministic restarts to escape local optima
    best_circles = None
    best_sum = -np.inf
    rng = np.random.default_rng(42)
    for restart in range(3):
        # Perturb initial guess slightly after the first run
        if restart == 0:
            x0_guess = x0.copy()
        else:
            noise = rng.normal(scale=0.01, size=x0.shape)
            x0_guess = x0 + noise
            # Clip to bounds
            x0_guess[:n] = np.clip(x0_guess[:n], 0.0, 1.0)
            x0_guess[n:2*n] = np.clip(x0_guess[n:2*n], 0.0, 1.0)
            x0_guess[2*n:] = np.clip(x0_guess[2*n:], 0.0, 0.5)

        res = minimize(
            obj,
            x0_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 12000, 'ftol': 1e-9, 'disp': False}
        )

        if not res.success:
            continue

        sol = res.x
        circles = np.column_stack((sol[:n], sol[n:2 * n], sol[2 * n:]))
        # Perform deterministic local refinement to improve packing
        circles = local_refine(circles, steps=40000)

        cur_sum = np.sum(circles[:, 2])
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_circles = circles

    if best_circles is None:
        # Fallback to original solution if all restarts fail
        best_circles = circles

    # Final polishing with SLSQP to capture any remaining improvements
    x0_refined = best_circles.ravel()
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
        best_circles = np.column_stack((sol[:n], sol[n:2 * n], sol[2 * n:]))
    
    # ---------- LP to tighten radii given positions ----------
    # Solve a linear program: maximize sum(r) subject to containment and non‑overlap
    # with fixed positions from best_circles.
    xs = best_circles[:, 0]
    ys = best_circles[:, 1]
    c = -np.ones(n)  # maximize sum(r) => minimize -sum(r)
    A_ub = []
    b_ub = []

    # Containment constraints
    for i in range(n):
        A_ub.append([0]*n); A_ub[-1][i] = 1.0
        b_ub.append(xs[i])          # r_i <= x_i
        A_ub.append([0]*n); A_ub[-1][i] = 1.0
        b_ub.append(1.0 - xs[i])    # r_i <= 1 - x_i
        A_ub.append([0]*n); A_ub[-1][i] = 1.0
        b_ub.append(ys[i])          # r_i <= y_i
        A_ub.append([0]*n); A_ub[-1][i] = 1.0
        b_ub.append(1.0 - ys[i])    # r_i <= 1 - y_i

    # Non‑overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            dist = np.hypot(xs[i]-xs[j], ys[i]-ys[j])
            row = [0]*n
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    from scipy.optimize import linprog
    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(0, None)] * n, method='highs')
    if res_lp.success:
        best_circles[:, 2] = res_lp.x
    else:
        # If LP fails, keep the SLSQP radii
        pass

    return best_circles

    sol = res.x
    circles = np.column_stack((sol[:n], sol[n:2 * n], sol[2 * n:]))
    # Perform deterministic local refinement to improve packing
    # Increase local refinement to explore a larger search space
    circles = local_refine(circles, steps=20000)
    
    # Final polishing with SLSQP to capture any remaining improvements
    x0_refined = circles.ravel()
    # Final polishing with a moderate number of iterations
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
