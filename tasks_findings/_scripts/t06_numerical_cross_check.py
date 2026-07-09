"""Numerical cross-check for Task T06.

Loads the cached utility grid from `game_experiments/_cache_v2/C1.pkl` (the
symmetric two-player configuration w1=w2=1, |G_1|=|G_2|=5000) and evaluates
the finite-difference mixed partial

    D u_k (i, j)  :=  ( U[i+1, j+1, k] - U[i+1, j-1, k]
                       - U[i-1, j+1, k] + U[i-1, j-1, k] )
                     / ( (eps[i+1] - eps[i-1]) * (eps[j+1] - eps[j-1]) )

for each player k in {0, 1}. The Monderer-Shapley symmetry condition demands
    eta_2 * D u_1 (i, j)  =  eta_1 * D u_2 (i, j)
for a fixed weight ratio eta_2/eta_1 across the grid.

We show:
  1. On the diagonal (eps_1 = eps_2), the ratio D u_1 / D u_2 == 1 (up to
     floating-point noise), consistent with mechanism-exchange symmetry.
  2. Off the diagonal, the ratio varies over ~two orders of magnitude and
     changes sign, so no positive constant weight ratio can satisfy MS-sym.
  3. In fact, at pairs (a, b) and (b, a) the sign of the ratio flips, which
     directly falsifies MS-sym under any *positive* weight vector.

The script writes a compact table with 8 off-diagonal points and 4 diagonal
reference points and prints a one-line MS-consistency verdict.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[2] / "game_experiments/_cache_v2/C1.pkl"


def cross_partial(U: np.ndarray, eps: np.ndarray, i: int, j: int, k: int) -> float:
    """Central second-difference mixed partial d^2 U[:,:,k] / (deps_i deps_j)."""
    num = (
        U[i + 1, j + 1, k]
        - U[i + 1, j - 1, k]
        - U[i - 1, j + 1, k]
        + U[i - 1, j - 1, k]
    )
    den = (eps[i + 1] - eps[i - 1]) * (eps[j + 1] - eps[j - 1])
    return num / den


def main() -> None:
    with open(CACHE, "rb") as fh:
        data = pickle.load(fh)
    sw = data["sweep"]
    U = np.array(sw["U"])           # shape (K, K, 2)
    feas = np.array(sw["feasible"]) # shape (K, K)
    eps = np.array(data["actions"]) # shape (K,)
    K = U.shape[0]

    # Collect (i, j, D1, D2, ratio) for all interior grid cells with feasible
    # 3x3 neighbourhood.
    rows = []
    for i in range(1, K - 1):
        for j in range(1, K - 1):
            if not feas[i - 1: i + 2, j - 1: j + 2].all():
                continue
            d1 = cross_partial(U, eps, i, j, 0)
            d2 = cross_partial(U, eps, i, j, 1)
            if abs(d2) < 1e-15:
                continue
            rows.append((i, j, eps[i], eps[j], d1, d2, d1 / d2))

    if not rows:
        raise RuntimeError("no feasible interior cells with 3x3 neighbourhood")

    ratios = np.array([r[6] for r in rows])
    on_diag = np.array([r[0] == r[1] for r in rows])
    off_diag_ratios = ratios[~on_diag]

    print("=" * 70)
    print("C1 cache: w1=w2=1, |G_1|=|G_2|=5000, K=22 log-spaced budgets")
    print("=" * 70)
    print(f"Interior feasible cells:            {len(rows)}")
    print(f"Diagonal cells (eps_1 = eps_2):     {int(on_diag.sum())}")
    print(f"Off-diagonal cells:                 {int((~on_diag).sum())}")
    print()

    # Verdict.
    n_pos = int((off_diag_ratios > 0).sum())
    n_neg = int((off_diag_ratios < 0).sum())
    print(f"Off-diagonal ratio spread:")
    print(f"  min           = {off_diag_ratios.min():+.3e}")
    print(f"  max           = {off_diag_ratios.max():+.3e}")
    print(f"  # sign>0      = {n_pos}")
    print(f"  # sign<0      = {n_neg}")
    print()
    if n_pos > 0 and n_neg > 0:
        print("VERDICT: MS-sym infeasible. Off-diagonal ratios span both signs,")
        print("so no *positive* weight pair (eta_1, eta_2) can satisfy")
        print("eta_2 D u_1 = eta_1 D u_2 across the grid.")
    else:
        print("VERDICT: off-diagonal ratios have a fixed sign; check spread.")
    print()

    # Diagonal reference table.
    print("Diagonal cells (mechanism exchange-symmetry check; ratio ~= 1):")
    print(f"  {'eps':>8} {'D^2 u_1':>13} {'D^2 u_2':>13} {'ratio':>10}")
    for row in rows:
        i, j, ei, ej, d1, d2, r = row
        if i != j:
            continue
        if i not in (2, 6, 10, 14, 18):
            continue
        print(f"  {ei:>8.4f} {d1:>+13.3e} {d2:>+13.3e} {r:>+10.4f}")
    print()

    # Top-8 off-diagonal magnitudes.
    print("Off-diagonal cells with largest |ratio|:")
    print(f"  {'(eps_1, eps_2)':>18} {'D^2 u_1':>13} {'D^2 u_2':>13} {'ratio':>10}")
    top = sorted([r for r in rows if r[0] != r[1]], key=lambda r: -abs(r[6]))[:8]
    for i, j, ei, ej, d1, d2, r in top:
        print(f"  ({ei:>7.3f}, {ej:>7.3f}) {d1:>+13.3e} {d2:>+13.3e} {r:>+10.4f}")
    print()

    # Direct sign-flip pair check.
    ratio_map = {(i, j): r for (i, j, ei, ej, d1, d2, r) in rows}
    flips = []
    for (i, j), r in ratio_map.items():
        if i >= j:
            continue
        if (j, i) in ratio_map and r * ratio_map[(j, i)] < 0:
            flips.append((i, j, r, ratio_map[(j, i)]))
    print(f"(i, j) vs (j, i) sign-flip pairs: {len(flips)}")
    if flips:
        i, j, ra, rb = flips[len(flips) // 2]
        print(f"  Example: (eps_1, eps_2) = ({eps[i]:.3f}, {eps[j]:.3f})  ratio = {ra:+.3f}")
        print(f"           swapped         = ({eps[j]:.3f}, {eps[i]:.3f})  ratio = {rb:+.3f}")


if __name__ == "__main__":
    main()
