# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

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
                  rng: np.random.Generator, iters: int = 500, step: float = 0.01) -> (np.ndarray, np.ndarray):
    """
    Small hill‑climb phase that evaluates each move with the exact LP solver.
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


def circle_packing32() -> np.ndarray:
    """
    Hybrid deterministic hexagonal initialization + large‑scale random search
    using a fast heuristic for quick evaluation, followed by LP optimization
    and local polishing.
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
        best_overall_pos, best_overall_radii, rng, iters=500, step=0.01)

    circles = np.column_stack((best_overall_pos, best_overall_radii))
    return circles


# EVOLVE-BLOCK-END
