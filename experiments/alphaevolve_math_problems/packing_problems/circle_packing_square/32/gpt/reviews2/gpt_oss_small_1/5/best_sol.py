# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize

# Helper functions for a deterministic hexagonal packing heuristic.

def generate_hex_positions(n: int) -> np.ndarray:
    """
    Generate up to n points in a hexagonal lattice inside the unit square.
    The lattice has 6 rows with alternating 6 and 5 points per row.
    """
    rows = []
    num_rows = 6
    for r in range(num_rows):
        if r % 2 == 0:
            count = 6
            xs = np.linspace(0, 1, count)
        else:
            count = 5
            spacing = 1 / (count - 1)
            xs = np.linspace(spacing / 2, 1 - spacing / 2, count)
        y = (r + 0.5) / num_rows
        for x in xs:
            rows.append([x, y])
    pos = np.array(rows, dtype=float)
    return pos[:n]

# Linear‑programming helper: compute maximal radii for fixed positions
def _compute_max_radii(positions: np.ndarray) -> np.ndarray:
    """
    Solve linear program to maximize sum of radii given fixed positions.
    positions: shape (n,2)
    Returns radii array of shape (n,)
    """
    n = positions.shape[0]
    # Objective: maximize sum r_i -> minimize -sum r_i
    c = -np.ones(n)

    # Bounds for radii: 0 <= r_i <= min(x,1-x,y,1-y)
    bounds = []
    for i in range(n):
        x, y = positions[i]
        max_r = min(x, 1 - x, y, 1 - y)
        bounds.append((0.0, max_r))

    # Inequality constraints: r_i + r_j <= d_ij
    A_ub = []
    b_ub = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            d = np.hypot(dx, dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x
    else:
        # Fallback: return zeros
        return np.zeros(n)

def compute_radii(pos: np.ndarray) -> np.ndarray:
    """
    Compute the maximal radius for each circle at given positions
    using linear programming to ensure optimality.
    """
    return _compute_max_radii(pos)

def gradient_refinement(pos: np.ndarray, radii: np.ndarray,
                        iterations: int = 10, eps: float = 1e-4) -> (np.ndarray, np.ndarray):
    """
    Perform a finite‑difference gradient refinement of the positions
    to locally increase the sum of radii. This follows the scheme
    used in the first inspiration program.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = best_radii.sum()

    for _ in range(iterations):
        grad = np.zeros_like(best_pos)
        for i in range(best_pos.shape[0]):
            for d in range(2):
                pert = np.zeros_like(best_pos)
                pert[i, d] = eps
                r_plus = compute_radii(best_pos + pert)
                r_minus = compute_radii(best_pos - pert)
                grad[i, d] = (r_plus.sum() - r_minus.sum()) / (2 * eps)
        norm = np.linalg.norm(grad)
        if norm == 0:
            break
        # More aggressive step for finer search
        step = 0.002 * grad / norm
        best_pos += step
        best_pos = np.clip(best_pos, 0.0, 1.0)
        best_radii = compute_radii(best_pos)
        best_sum = best_radii.sum()

    return best_pos, best_radii

# Alias for backward compatibility
local_search = gradient_refinement

def _compute_radii_simple(positions: np.ndarray) -> np.ndarray:
    """
    Fast heuristic: r_i = min(boundary distance, half of the nearest neighbor distance).
    """
    # Boundary distances
    bd = np.minimum(np.minimum(positions[:, 0], 1.0 - positions[:, 0]),
                    np.minimum(positions[:, 1], 1.0 - positions[:, 1]))
    # Pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    half_dist = dist / 2.0
    radii = np.minimum(bd, np.min(half_dist, axis=1))
    return radii

def hill_climb_lp(pos: np.ndarray, radii: np.ndarray,
                  rng: np.random.Generator, iters: int = 1000, step: float = 0.01) -> (np.ndarray, np.ndarray):
    """
    Small hill‑climb phase that evaluates each move with the exact LP solver.
    Increased iterations for finer polishing.
    """
    best_pos = pos.copy()
    best_radii = radii.copy()
    best_sum = best_radii.sum()

    for _ in range(iters):
        idx = rng.integers(best_pos.shape[0])
        new_pos = best_pos.copy()
        new_pos[idx] += rng.uniform(-step, step, size=2)
        new_pos[idx] = np.clip(new_pos[idx], 0.0, 1.0)

        new_radii = compute_radii(new_pos)
        new_sum = new_radii.sum()

        if new_sum > best_sum:
            best_pos = new_pos
            best_radii = new_radii
            best_sum = new_sum

    return best_pos, best_radii


def _physics_relaxation(pos: np.ndarray, radii: np.ndarray,
                        steps: int = 5000, dt: float = 0.1) -> (np.ndarray, np.ndarray):
    """
    Vectorized repulsive force simulation that slightly separates overlapping circles.
    Uses the current radii as constraints and keeps circles inside the unit square.
    """
    n = pos.shape[0]
    for _ in range(steps):
        dx = pos[:, None, 0] - pos[None, :, 0]
        dy = pos[:, None, 1] - pos[None, :, 1]
        dist = np.hypot(dx, dy)
        dist_safe = np.where(dist == 0, 1e-12, dist)

        # Overlap detection
        overlap = radii[:, None] + radii[None, :] - dist
        mask = (overlap > 0) & (dist > 1e-12)

        fx = np.where(mask, (dx / dist_safe) * overlap, 0.0)
        fy = np.where(mask, (dy / dist_safe) * overlap, 0.0)

        forces = np.zeros((n, 2), dtype=np.float64)
        forces[:, 0] = fx.sum(axis=1)
        forces[:, 1] = fy.sum(axis=1)

        # Boundary forces
        left_mask = pos[:, 0] - radii < 0
        forces[left_mask, 0] += (radii[left_mask] - pos[left_mask, 0])
        right_mask = pos[:, 0] + radii > 1
        forces[right_mask, 0] -= (pos[right_mask, 0] + radii[right_mask] - 1)
        bottom_mask = pos[:, 1] - radii < 0
        forces[bottom_mask, 1] += (radii[bottom_mask] - pos[bottom_mask, 1])
        top_mask = pos[:, 1] + radii > 1
        forces[top_mask, 1] -= (pos[top_mask, 1] + radii[top_mask] - 1)

        pos += dt * forces
        pos[:, 0] = np.clip(pos[:, 0], radii, 1 - radii)
        pos[:, 1] = np.clip(pos[:, 1], radii, 1 - radii)

    return pos, radii


def _final_slsqp_refine(pos: np.ndarray, radii: np.ndarray,
                        rng: np.random.Generator, maxiter: int = 2000) -> (np.ndarray, np.ndarray):
    """
    Final SLSQP refinement that jointly optimizes positions and radii.
    Starts from the best configuration after physics relaxation.
    """
    n = pos.shape[0]
    x0 = np.empty(3 * n)
    x0[0::3] = pos[:, 0]
    x0[1::3] = pos[:, 1]
    x0[2::3] = radii

    def obj(v: np.ndarray) -> float:
        return -np.sum(v[2::3])  # maximize sum of radii

    cons = []

    # Boundary constraints
    for i in range(n):
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i]})           # x >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1 - v[3 * i]})       # x <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i + 1]})       # y >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: 1 - v[3 * i + 1]})   # y <= 1
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i + 2]})       # r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i] - v[3 * i + 2]})   # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1 - v[3 * i]) - v[3 * i + 2]}) # 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: v[3 * i + 1] - v[3 * i + 2]})   # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda v, i=i: (1 - v[3 * i + 1]) - v[3 * i + 2]}) # 1 - y - r >= 0

    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append(
                {"type": "ineq",
                 "fun": lambda v, i=i, j=j: np.linalg.norm(v[3 * i:3 * i + 2] - v[3 * j:3 * j + 2])
                 - (v[3 * i + 2] + v[3 * j + 2])}
            )

    res = minimize(obj, x0, method="SLSQP", constraints=cons,
                   options={"ftol": 1e-6, "maxiter": maxiter, "disp": False})

    if res.success:
        sol = res.x.reshape(-1, 3)
        return sol[:, 0:2], sol[:, 2]
    else:
        # Fallback: return original
        return pos, radii


def circle_packing32() -> np.ndarray:
    """
    Hybrid deterministic hexagonal initialization + large‑scale random search
    using a fast heuristic for quick evaluation, followed by LP optimization,
    physics relaxation, and extensive polishing.
    """
    rng = np.random.default_rng(42)

    n = 32
    # Three starting configurations
    starts = []

    # 1️⃣ Hexagonal
    hex_pos = generate_hex_positions(n)
    starts.append(hex_pos)

    # 2️⃣ Random uniform
    rand_pos = rng.uniform(0.0, 1.0, size=(n, 2))
    starts.append(rand_pos)

    # 3️⃣ Perturbed hexagonal
    pert_pos = hex_pos.copy()
    pert_pos += rng.uniform(-0.05, 0.05, size=(n, 2))
    pert_pos = np.clip(pert_pos, 0.0, 1.0)
    starts.append(pert_pos)

    best_overall_pos = None
    best_overall_radii = None
    best_overall_sum = -np.inf

    # Random search parameters
    n_iter = 40000
    step_size = 0.15

    for init_pos in starts:
        pos = init_pos.copy()
        radii = _compute_radii_simple(pos)
        sum_r = radii.sum()

        for _ in range(n_iter):
            idx = rng.integers(n)
            new_pos = pos.copy()
            new_pos[idx] += rng.uniform(-step_size, step_size, size=2)
            new_pos[idx] = np.clip(new_pos[idx], 0.0, 1.0)

            new_radii = _compute_radii_simple(new_pos)
            new_sum = new_radii.sum()

            if new_sum > sum_r:
                pos, sum_r = new_pos, new_sum

        # Evaluate with LP
        opt_radii = compute_radii(pos)
        opt_sum = opt_radii.sum()

        if opt_sum > best_overall_sum:
            best_overall_sum = opt_sum
            best_overall_pos = pos.copy()
            best_overall_radii = opt_radii.copy()

    # Gradient refinement on the best positions
    best_overall_pos, best_overall_radii = gradient_refinement(
        best_overall_pos, best_overall_radii)

    # LP‑based hill‑climb polish
    best_overall_pos, best_overall_radii = hill_climb_lp(
        best_overall_pos, best_overall_radii, rng, iters=1500, step=0.01)

    # Second fine‑grained random search around the best solution
    for _ in range(20000):
        idx = rng.integers(n)
        new_pos = best_overall_pos.copy()
        new_pos[idx] += rng.uniform(-0.05, 0.05, size=2)
        new_pos[idx] = np.clip(new_pos[idx], 0.0, 1.0)

        new_radii = _compute_radii_simple(new_pos)
        new_sum = new_radii.sum()

        if new_sum > best_overall_radii.sum():
            best_overall_pos = new_pos
            best_overall_radii = compute_radii(best_overall_pos)

    # Physics relaxation to slightly separate circles
    best_overall_pos, best_overall_radii = _physics_relaxation(
        best_overall_pos, best_overall_radii, steps=5000, dt=0.1)

    # Final LP after physics
    best_overall_radii = compute_radii(best_overall_pos)

    # Final SLSQP refinement for maximal sum
    best_overall_pos, best_overall_radii = _final_slsqp_refine(
        best_overall_pos, best_overall_radii, rng, maxiter=2000)

    circles = np.column_stack((best_overall_pos, best_overall_radii))
    return circles


# EVOLVE-BLOCK-END
