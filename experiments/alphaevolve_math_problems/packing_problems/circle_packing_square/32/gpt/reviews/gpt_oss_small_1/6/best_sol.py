# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import math

# ------------------------------------------------------------------
# Improved physics relaxation using vectorized forces
# ------------------------------------------------------------------
def _physics_relaxation(circles: np.ndarray, steps: int = 40000, dt: float = 0.2) -> np.ndarray:
    """
    Repulsive force simulation to separate overlapping circles and push them towards the square centre.
    Vectorized implementation for speed.
    """
    n = len(circles)
    for _ in range(steps):
        # pairwise distances
        dx = circles[:, 0][:, None] - circles[:, 0][None, :]
        dy = circles[:, 1][:, None] - circles[:, 1][None, :]
        dist = np.hypot(dx, dy)
        # avoid division by zero
        dist_safe = np.where(dist == 0, 1e-12, dist)
        min_dist = circles[:, 2][:, None] + circles[:, 2][None, :]
        mask = (dist < min_dist) & (dist > 1e-12)
        overlap = np.where(mask, min_dist - dist, 0.0)
        fx = np.where(mask, (dx / dist_safe) * overlap, 0.0)
        fy = np.where(mask, (dy / dist_safe) * overlap, 0.0)
        forces = np.zeros((n, 2), dtype=np.float64)
        forces[:, 0] = fx.sum(axis=1)
        forces[:, 1] = fy.sum(axis=1)
        # boundary forces
        r = circles[:, 2]
        left_mask = circles[:, 0] - r < 0
        forces[left_mask, 0] += (r[left_mask] - circles[left_mask, 0])
        right_mask = circles[:, 0] + r > 1
        forces[right_mask, 0] -= (circles[right_mask, 0] + r[right_mask] - 1)
        bottom_mask = circles[:, 1] - r < 0
        forces[bottom_mask, 1] += (r[bottom_mask] - circles[bottom_mask, 1])
        top_mask = circles[:, 1] + r > 1
        forces[top_mask, 1] -= (circles[top_mask, 1] + r[top_mask] - 1)
        # update positions
        circles[:, 0] += dt * forces[:, 0]
        circles[:, 1] += dt * forces[:, 1]
        # clip to stay inside
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], 1 - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], 1 - circles[:, 2])
    return circles

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

# ------------------------------------------------------------------
# Greedy max‑radius routine that iteratively sets each circle’s radius
# to the largest feasible value given the current configuration.
# ------------------------------------------------------------------
def _max_radius_iterative(circles: np.ndarray, tol: float = 1e-5, max_iter: int = 200) -> np.ndarray:
    """
    Greedy max‑radius routine that iteratively sets each circle’s radius to the
    largest feasible value given the current configuration.
    """
    n = len(circles)
    for _ in range(max_iter):
        changed = False
        centers = circles[:, :2]
        dists = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        # boundary limits
        max_r = np.minimum(np.minimum(circles[:, 0], 1 - circles[:, 0]),
                           np.minimum(circles[:, 1], 1 - circles[:, 1]))
        # neighbor limits
        neigh = dists - circles[:, 2][:, None]
        np.fill_diagonal(neigh, np.inf)
        max_r = np.minimum(max_r, neigh.min(axis=1))
        new_r = np.maximum(0.0, max_r)
        if np.any(np.abs(new_r - circles[:, 2]) > tol):
            circles[:, 2] = new_r
            changed = True
        if not changed:
            break
    return circles

# ------------------------------------------------------------------
# Fine‑tune: attempt tiny radius increases on all circles
# ------------------------------------------------------------------
def _fine_tune(circles: np.ndarray, rng: np.random.Generator, attempts: int = 5000, step: float = 0.0005) -> np.ndarray:
    """
    Try to increase each circle’s radius by a tiny step if it remains feasible.
    """
    n = len(circles)
    for _ in range(attempts):
        for i in range(n):
            new_r = circles[i, 2] + step
            if circles[i, 0] - new_r < 0 or circles[i, 0] + new_r > 1 or \
               circles[i, 1] - new_r < 0 or circles[i, 1] + new_r > 1:
                continue
            dx = circles[i, 0] - circles[:, 0]
            dy = circles[i, 1] - circles[:, 1]
            dist_sq = dx * dx + dy * dy
            dist_sq[i] = np.inf
            required_sq = (new_r + circles[:, 2]) ** 2
            if np.any(dist_sq < required_sq):
                continue
            circles[i, 2] = new_r
    return circles

# ------------------------------------------------------------------
# Simulated annealing tweak
# ------------------------------------------------------------------
def _annealing_tweak(circles: np.ndarray, rng: np.random.Generator, n_iter: int = 400000) -> np.ndarray:
    """
    Stochastic perturbation of positions and radii to escape local optima.
    """
    n = len(circles)
    T0, Tmin, alpha = 3.0, 1e-6, 0.995
    T = T0
    for _ in range(n_iter):
        idx = rng.integers(0, n)
        delta_r = rng.uniform(-0.001, 0.001)
        new_r = circles[idx, 2] + delta_r
        delta_x = rng.uniform(-0.01, 0.01)
        delta_y = rng.uniform(-0.01, 0.01)
        new_x = circles[idx, 0] + delta_x
        new_y = circles[idx, 1] + delta_y
        if new_r < 0 or new_x - new_r < 0 or new_x + new_r > 1 or new_y - new_r < 0 or new_y + new_r > 1:
            continue
        dx = new_x - circles[:, 0]
        dy = new_y - circles[:, 1]
        dist_sq = dx * dx + dy * dy
        dist_sq[idx] = np.inf
        required_sq = (new_r + circles[:, 2]) ** 2
        if np.any(dist_sq < required_sq):
            continue
        current_sum = circles[:, 2].sum()
        new_sum = current_sum - circles[idx, 2] + new_r
        if new_sum > current_sum or rng.random() < math.exp((new_sum - current_sum) / T):
            circles[idx, 0] = new_x
            circles[idx, 1] = new_y
            circles[idx, 2] = new_r
        T = max(T * alpha, Tmin)
    return circles

# ------------------------------------------------------------------
# Global radius scaling (push all radii up until first overlap)
# ------------------------------------------------------------------
def _global_scale(circles: np.ndarray) -> np.ndarray:
    """
    Scale all radii by the largest factor s ∈ (0,1] that keeps the configuration feasible.
    Binary search over s for 20 iterations.
    """
    n = len(circles)
    low, high = 0.0, 1.0
    for _ in range(20):
        mid = (low + high) / 2
        scaled = circles[:, 2] * mid
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.hypot(circles[i, 0] - circles[j, 0], circles[i, 1] - circles[j, 1])
                if dist < scaled[i] + scaled[j]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            low = mid
        else:
            high = mid
    scale = low
    circles[:, 2] *= scale
    return circles

# ------------------------------------------------------------------
# Final feasibility correction
# ------------------------------------------------------------------
def _final_feasibility(circles: np.ndarray) -> np.ndarray:
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(circles[i, 0] - circles[j, 0], circles[i, 1] - circles[j, 1])
            min_dist = circles[i, 2] + circles[j, 2]
            if dist < min_dist:
                shrink = (min_dist - dist) / 2 + 1e-6
                circles[i, 2] = max(0, circles[i, 2] - shrink)
                circles[j, 2] = max(0, circles[j, 2] - shrink)
    return circles

def gradient_refinement(pos: np.ndarray, radii: np.ndarray,
                        iterations: int = 20, eps: float = 1e-4) -> (np.ndarray, np.ndarray):
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
        step = 0.001 * grad / norm
        best_pos += step
        best_pos = np.clip(best_pos, 0.0, 1.0)
        best_radii = compute_radii(best_pos)
        best_sum = best_radii.sum()

    return best_pos, best_radii

# Alias for backward compatibility
local_search = gradient_refinement

def circle_packing32() -> np.ndarray:
    """
    Random initialization followed by LP radius optimisation, physics relaxation,
    global scaling, extensive annealing tweak, final max‑radius pass, fine‑tune,
    and feasibility correction.
    """
    rng = np.random.default_rng(42)

    n = 32
    # Random initial positions with a moderate starting radius
    init_r = 0.1
    circles = np.zeros((n, 3))
    circles[:, 0] = rng.uniform(init_r, 1 - init_r, size=n)
    circles[:, 1] = rng.uniform(init_r, 1 - init_r, size=n)
    circles[:, 2] = init_r

    # Physics relaxation to separate overlapping circles
    circles = _physics_relaxation(circles, steps=30000, dt=0.15)

    # Recompute radii with LP after relaxation
    new_radii = compute_radii(circles[:, :2])
    circles[:, 2] = new_radii

    # Max radius iterations to adjust radii given current positions
    circles = _max_radius_iterative(circles)

    # Global radius scaling to squeeze in more radius
    circles = _global_scale(circles)

    # Annealing tweak to escape local optima
    circles = _annealing_tweak(circles, rng, n_iter=400000)

    # Final max‑radius pass after annealing
    circles = _max_radius_iterative(circles)

    # Gradient refinement of positions using exact LP evaluation
    circles[:, :2], circles[:, 2] = gradient_refinement(circles[:, :2], circles[:, 2])

    # Second fine‑tune to push radii as far as possible
    circles = _fine_tune(circles, rng, attempts=7000, step=0.0005)

    # Final feasibility correction
    circles = _final_feasibility(circles)

    return circles


# EVOLVE-BLOCK-END
