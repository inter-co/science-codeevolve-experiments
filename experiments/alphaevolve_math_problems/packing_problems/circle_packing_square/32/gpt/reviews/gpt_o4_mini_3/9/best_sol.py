# EVOLVE-BLOCK-START
import numpy as np
import random
import math
import time
from scipy.optimize import linprog, minimize

# Modular circle packing strategies for methodological diversity.

def greedy_random_packing(n=32, samples=5000, seed=42):
    """
    Greedy random sequential adsorption.
    Places circles one by one by sampling candidate positions and selecting the maximal feasible radius.
    """
    random.seed(seed)
    np.random.seed(seed)
    circles = []
    for i in range(n):
        best_r = 0.0
        best_pos = (0.0, 0.0)
        for _ in range(samples):
            x, y = random.random(), random.random()
            # Max radius limited by boundary
            r = min(x, 1.0 - x, y, 1.0 - y)
            # Shrink r to avoid overlaps with existing circles
            for (cx, cy, cr) in circles:
                dist = math.hypot(x - cx, y - cy)
                r = min(r, dist - cr)
            if r > best_r:
                best_r = r
                best_pos = (x, y)
        if best_r > 1e-6:
            circles.append((best_pos[0], best_pos[1], best_r))
        else:
            # Fallback to a tiny circle if no feasible region found
            circles.append((random.random(), random.random(), 1e-3))
    return np.array(circles)

def lattice_packing(n=32, rows=6, cols=6):
    """
    Simple lattice-based equal-radius packing as an analytic baseline.
    """
    positions = []
    r = 1.0 / (2 * max(rows, cols))
    for i in range(rows):
        for j in range(cols):
            if len(positions) >= n:
                break
            x = (j + 0.5) * 2 * r
            y = (i + 0.5) * 2 * r
            positions.append((x, y))
        if len(positions) >= n:
            break
    radii = [r] * len(positions)
    return np.column_stack((np.array(positions), np.array(radii)))

# LP-based radius optimization for fixed positions
def optimize_radii(xs, ys):
    """
    Optimize circle radii for given positions using linear programming.
    xs, ys: arrays of shape (n,)
    Returns array of shape (n,3): x, y, optimized r
    """
    n = len(xs)
    # Objective: maximize sum of r_i => minimize -sum r_i
    c = [-1.0] * n
    A = []
    b = []
    # Boundary constraints: r_i <= x_i, 1-x_i, y_i, 1-y_i
    for i in range(n):
        ai = [0.0]*n
        ai[i] = 1.0
        A.append(ai.copy()); b.append(xs[i])
        A.append(ai.copy()); b.append(1.0 - xs[i])
        A.append(ai.copy()); b.append(ys[i])
        A.append(ai.copy()); b.append(1.0 - ys[i])
    # Pairwise non-overlap: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i+1, n):
            dij = math.hypot(xs[i]-xs[j], ys[i]-ys[j])
            aij = [0.0]*n
            aij[i] = 1.0
            aij[j] = 1.0
            A.append(aij)
            b.append(dij)
    # Solve LP
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')
    if res.success:
        rs = res.x
    else:
        # Fallback to minimal boundary radii
        rs = np.minimum(xs, np.minimum(1-xs, np.minimum(ys, 1-ys)))
    return np.column_stack((xs, ys, rs))

# Local stochastic search: jitter circle positions and re-optimize radii (inspired by Inspiration Program 2)
def local_search(xs, ys, rs, iters=500, noise_scale=0.02, seed=42):
    """
    Perform stochastic local refinement by randomly perturbing one circle at a time
    and re-optimizing radii. Retains improvements in total radius sum.
    """
    random.seed(seed)
    best_xs = xs.copy()
    best_ys = ys.copy()
    best_pack = np.column_stack((best_xs, best_ys, rs.copy()))
    best_sum = np.sum(rs)
    for _ in range(iters):
        i = random.randrange(len(xs))
        # Perturb the i-th circle's position
        cand_xs = best_xs.copy()
        cand_ys = best_ys.copy()
        cand_xs[i] = min(max(cand_xs[i] + random.uniform(-noise_scale, noise_scale), 0.0), 1.0)
        cand_ys[i] = min(max(cand_ys[i] + random.uniform(-noise_scale, noise_scale), 0.0), 1.0)
        # Re-optimize radii for new positions
        cand_pack = optimize_radii(cand_xs, cand_ys)
        cand_sum = np.sum(cand_pack[:, 2])
        if cand_sum > best_sum:
            best_sum = cand_sum
            best_xs, best_ys = cand_pack[:, 0].copy(), cand_pack[:, 1].copy()
            best_pack = cand_pack.copy()
    return best_pack

# Simulated Annealing–based circle packing (inspired by Inspiration Program 1)
def simulated_annealing_packing(n=32, time_limit=10, seed=42):
    """
    Simulated Annealing for diverse global exploration.
    """
    random.seed(seed)
    np.random.seed(seed)
    start = time.time()

    # Initialize on a coarse grid with small radii
    grid = int(np.ceil(np.sqrt(n)))
    coords = [((i + 0.5)/grid, (j + 0.5)/grid)
              for i in range(grid) for j in range(grid)]
    coords = coords[:n]
    circles = np.array([[x, y, 1.0/(2*grid)] for x, y in coords])

    def is_feasible(sol):
        xs, ys, rs = sol[:,0], sol[:,1], sol[:,2]
        # Boundary constraints
        if np.any(xs - rs < 0) or np.any(xs + rs > 1) \
           or np.any(ys - rs < 0) or np.any(ys + rs > 1):
            return False
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                if (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 < (rs[i]+rs[j])**2:
                    return False
        return True

    T, T_end, alpha = 0.1, 1e-4, 0.995
    best = circles.copy()
    current = circles.copy()
    best_val = best[:,2].sum()

    while time.time() - start < time_limit:
        idx = random.randrange(n)
        cand = current.copy()
        if random.random() < 0.5:
            # jitter position
            cand[idx,0] = np.clip(cand[idx,0] + np.random.normal(scale=0.01),
                                  cand[idx,2], 1-cand[idx,2])
            cand[idx,1] = np.clip(cand[idx,1] + np.random.normal(scale=0.01),
                                  cand[idx,2], 1-cand[idx,2])
        else:
            # tweak radius
            cand[idx,2] = abs(cand[idx,2] + np.random.normal(scale=0.001))
        if not is_feasible(cand):
            continue
        val = cand[:,2].sum()
        delta = val - current[:,2].sum()
        if delta > 0 or random.random() < math.exp(delta/T):
            current = cand
            if val > best_val:
                best = cand.copy()
                best_val = val
        T *= alpha
        if T < T_end:
            break
    return best

# SLSQP-based continuous refinement using SciPy minimize (inspired by Inspiration Program 1)
def slsqp_refine(circles, maxiter=200):
    """
    Continuous refinement: maximize sum of radii with SLSQP.
    circles: array of shape (n,3) [x, y, r]
    """
    n = len(circles)
    x0 = np.hstack([circles[:,0], circles[:,1], circles[:,2]])
    def obj(x):
        return -np.sum(x[2*n:])
    cons = []
    # Boundary constraints and non-overlap
    for i in range(n):
        # r_i ≤ x_i ≤ 1−r_i
        cons.append({'type':'ineq','fun': lambda x,i=i:  x[i] - x[2*n+i]})
        cons.append({'type':'ineq','fun': lambda x,i=i:  1 - x[i] - x[2*n+i]})
        # r_i ≤ y_i ≤ 1−r_i
        cons.append({'type':'ineq','fun': lambda x,i=i:  x[n+i] - x[2*n+i]})
        cons.append({'type':'ineq','fun': lambda x,i=i:  1 - x[n+i] - x[2*n+i]})
    # Pairwise non-overlap: dist^2 ≥ (r_i+r_j)^2
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type':'ineq',
                'fun': lambda x,i=i,j=j:
                    (x[i]-x[j])**2 + (x[n+i]-x[n+j])**2
                    - (x[2*n+i] + x[2*n+j])**2
            })
    res = minimize(obj, x0, method='SLSQP', constraints=cons,
                   options={'ftol':1e-6, 'maxiter':maxiter, 'disp':False})
    if res.success:
        sol = res.x
        return np.column_stack((sol[:n], sol[n:2*n], sol[2*n:]))
    return circles

def circle_packing32() -> np.ndarray:
    """
    Aggregates three strategies—greedy, lattice, simulated annealing—
    refines each via LP, local search, and final SLSQP, then returns the best.
    """
    # 1) Greedy + LP + local + SLSQP
    pack1 = greedy_random_packing()
    pack1 = optimize_radii(pack1[:,0], pack1[:,1])
    pack1 = local_search(pack1[:,0], pack1[:,1], pack1[:,2])
    pack1 = slsqp_refine(pack1)

    # 2) Lattice + LP + local + SLSQP
    pack2 = lattice_packing()
    pack2 = optimize_radii(pack2[:,0], pack2[:,1])
    pack2 = local_search(pack2[:,0], pack2[:,1], pack2[:,2])
    pack2 = slsqp_refine(pack2)

    # 3) Simulated Annealing + LP + local + SLSQP
    pack3 = simulated_annealing_packing()
    pack3 = optimize_radii(pack3[:,0], pack3[:,1])
    pack3 = local_search(pack3[:,0], pack3[:,1], pack3[:,2])
    pack3 = slsqp_refine(pack3)

    # Compare total sums and then one final SLSQP refine on best
    sums = [np.sum(p[:,2]) for p in (pack1, pack2, pack3)]
    best = [pack1, pack2, pack3][int(np.argmax(sums))]
    return slsqp_refine(best)


# EVOLVE-BLOCK-END
