"""Numerical corroboration of the 2-player self-sorting proposition (Task 07).

Reads the C1 mechanism cache (`game_experiments/_cache_v2/C1.pkl`), which
holds a K=22 log-spaced sweep at symmetric groups |G_1|=|G_2|=5000 and
symmetric weights w_1=w_2=1. From it we extract the *weight-independent*
mechanism state (sigma[i,j], A[i,j,:], feasible[i,j]) and re-derive the
utility grid at asymmetric weights via `sweep_from_grid`, then locate
pure NEs on that fine grid.

Reports:

  1. Mechanism symmetry (Assumption (A1) of the proposition):
     sigma(x,y) = sigma(y,x) and A_own(x,y) = A_other(y,x).

  2. First-order signs used by the proof (from lem:externality):
     dA_own / deps_own > 0, dA_own / deps_other <= 0, dsigma / deps_i < 0.

  3. Structural submodularity (Assumption (A3) of the proposition):
     f(x,y) := A_own(x,y) is submodular in (x, y), i.e., the discrete
     cross-difference [f(x_hi,y_lo) - f(x_lo,y_lo)] - [f(x_hi,y_hi) - f(x_lo,y_hi)]
     is >= 0 on every feasible 2x2 cell.

  4. **Self-sorting** (the proposition): a table of (w_1/w_2 ratio,
     eps_ordering) pairs on a sweep w_1 in {1/8, 1/4, 1/2, 1, 2, 4, 8},
     w_2 = 1. For every ratio > 1 (resp. < 1), every pure NE must satisfy
     eps_1 < eps_2 (resp. eps_1 > eps_2). At ratio = 1, the set of NEs is
     closed under (i,j) -> (j,i).

Also runs a rotated sweep w_1 = 1, w_2 in {1/8, ..., 8} as a redundancy
check that the ordering is symmetric across relabelling.

Prints all results to stdout; no plots. Reproduce with

    python tasks_findings/_scripts/07_check.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.continuous import sweep_from_grid, find_pure_NE_2player

CACHE = ROOT / "game_experiments" / "_cache_v2" / "C1.pkl"


def load_mech_grid() -> dict:
    with open(CACHE, "rb") as fh:
        d = pickle.load(fh)
    sw = d["sweep"]
    return dict(
        actions=sw["actions"],
        sigma=sw["sigma"],
        A=sw["A"],
        feasible=sw["feasible"],
    )


def check_symmetry(mech: dict) -> None:
    sigma = mech["sigma"]
    A = mech["A"]
    feas = mech["feasible"]
    K = len(mech["actions"])
    # sigma(x,y) = sigma(y,x)
    both = feas & feas.T
    sig_err = float(np.nanmax(np.abs(sigma - sigma.T)[both])) if both.any() else 0.0
    # A_own(x,y) = A_other(y,x)  i.e.  A[i,j,0] == A[j,i,1]
    swap_err = 0.0
    for i in range(K):
        for j in range(K):
            if feas[i, j] and feas[j, i]:
                swap_err = max(
                    swap_err,
                    abs(float(A[i, j, 0] - A[j, i, 1])),
                    abs(float(A[i, j, 1] - A[j, i, 0])),
                )
    print("[A1] mechanism swap symmetry (|G_1|=|G_2|):")
    print(f"     max |sigma(x,y) - sigma(y,x)| = {sig_err:.2e}")
    print(f"     max |A(x,y,0) - A(y,x,1)|     = {swap_err:.2e}")


def check_first_order_signs(mech: dict) -> None:
    """dA/deps_own > 0, dA/deps_other <= 0, dsigma/deps_i < 0."""
    A = mech["A"]
    sig = mech["sigma"]
    feas = mech["feasible"]
    K = len(mech["actions"])

    def _monotone_frac(kind: str) -> tuple[int, int]:
        ok = tot = 0
        if kind == "A_own_up":
            # A[i,j,0] is increasing in i at fixed j
            for j in range(K):
                prev = None
                for i in range(K):
                    if not feas[i, j]:
                        prev = None
                        continue
                    cur = float(A[i, j, 0])
                    if prev is not None:
                        tot += 1
                        ok += cur > prev
                    prev = cur
        elif kind == "A_other_down":
            # A[i,j,0] non-increasing in j at fixed i
            for i in range(K):
                prev = None
                for j in range(K):
                    if not feas[i, j]:
                        prev = None
                        continue
                    cur = float(A[i, j, 0])
                    if prev is not None:
                        tot += 1
                        ok += cur <= prev + 1e-12
                    prev = cur
        elif kind == "sigma_down":
            # sigma[i,j] strictly decreasing in i at fixed j
            for j in range(K):
                prev = None
                for i in range(K):
                    if not feas[i, j]:
                        prev = None
                        continue
                    cur = float(sig[i, j])
                    if prev is not None:
                        tot += 1
                        ok += cur < prev
                    prev = cur
        return ok, tot

    for kind, desc in [
        ("A_own_up",     "dA_own / deps_own > 0        "),
        ("A_other_down", "dA_own / deps_other <= 0     "),
        ("sigma_down",   "dsigma / deps_i < 0          "),
    ]:
        ok, tot = _monotone_frac(kind)
        print(f"     {desc}: {ok}/{tot} = {ok/tot:.3f} monotone")


def check_submodularity(mech: dict) -> None:
    """f(x,y) = A_own(x,y) submodular in (x, y):
    D2 f := [f(x_hi, y_lo) - f(x_lo, y_lo)] - [f(x_hi, y_hi) - f(x_lo, y_hi)] >= 0.
    """
    A = mech["A"][:, :, 0]  # shape (K, K)
    feas = mech["feasible"]
    K = A.shape[0]
    ok = tot = 0
    worst = 0.0
    for i_lo in range(K - 1):
        for i_hi in range(i_lo + 1, K):
            for j_lo in range(K - 1):
                for j_hi in range(j_lo + 1, K):
                    if not (feas[i_hi, j_lo] and feas[i_lo, j_lo]
                            and feas[i_hi, j_hi] and feas[i_lo, j_hi]):
                        continue
                    d = (A[i_hi, j_lo] - A[i_lo, j_lo]) \
                        - (A[i_hi, j_hi] - A[i_lo, j_hi])
                    tot += 1
                    if d >= -1e-12:
                        ok += 1
                    else:
                        worst = min(worst, float(d))
    print(f"[A3] f(x,y)=A_own(x,y) submodular: {ok}/{tot} = "
          f"{ok/tot:.3f} of feasible 2x2 blocks (worst neg dip: {worst:.2e})")


def sort_table(mech: dict, w1s, w2s) -> list[dict]:
    """Sweep (w_1, w_2), locate all pure NEs, classify eps ordering."""
    actions = mech["actions"]
    rows = []
    for w1 in w1s:
        for w2 in w2s:
            sw = sweep_from_grid(mech, w1=w1, w2=w2)
            nes = find_pure_NE_2player(sw)
            n_below = n_above = n_equal = 0
            for (i, j) in nes:
                if i < j:  n_above += 1   # eps_1 < eps_2
                elif i > j: n_below += 1
                else:       n_equal += 1
            rows.append(dict(w1=w1, w2=w2, ratio=w1/w2, n_ne=len(nes),
                             n_eps1_lt=n_above, n_eps1_gt=n_below,
                             n_eps1_eq=n_equal,
                             nes=[(float(actions[i]), float(actions[j])) for (i, j) in nes]))
    return rows


def print_sort_table(rows: list[dict]) -> None:
    print(f"{'w_1':>7} {'w_2':>6} {'w1/w2':>7} {'|NE|':>4} "
          f"{'eps1<eps2':>10} {'eps1=eps2':>10} {'eps1>eps2':>10}  verdict")
    for r in rows:
        if r["ratio"] > 1:
            expected = "eps1 < eps2"
            ok = r["n_eps1_gt"] == 0
        elif r["ratio"] < 1:
            expected = "eps1 > eps2"
            ok = r["n_eps1_lt"] == 0
        else:
            expected = "symmetric"
            ok = r["n_eps1_lt"] == r["n_eps1_gt"]
        verdict = ("OK " if ok else "FAIL") + f"  ({expected})"
        print(f"{r['w1']:>7.3g} {r['w2']:>6.3g} {r['ratio']:>7.3g} "
              f"{r['n_ne']:>4d} {r['n_eps1_lt']:>10d} {r['n_eps1_eq']:>10d} "
              f"{r['n_eps1_gt']:>10d}  {verdict}")


def main() -> None:
    mech = load_mech_grid()
    K = len(mech["actions"])
    print(f"Loaded C1 mechanism grid, K={K}, "
          f"eps in [{mech['actions'][0]:.3g}, {mech['actions'][-1]:.3g}].\n")

    check_symmetry(mech)
    print()
    print("[FOC signs] on the K x K mechanism grid (lem:externality):")
    check_first_order_signs(mech)
    print()
    check_submodularity(mech)
    print()

    # ------- Main table: w_1 sweep, w_2 = 1 -------
    ws = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    print("[Sort table] Sweep w_1 with w_2 = 1  (Proposition prediction: "
          "w_1 > w_2  ==>  eps_1 <= eps_2).")
    rows = sort_table(mech, w1s=ws, w2s=[1.0])
    print_sort_table(rows)
    print()

    # ------- Rotated table: w_2 sweep, w_1 = 1 -------
    print("[Rotated table] Sweep w_2 with w_1 = 1  (redundancy check: "
          "same proposition under relabelling).")
    rows = sort_table(mech, w1s=[1.0], w2s=ws)
    print_sort_table(rows)
    print()

    # ------- Summary count -------
    all_rows = sort_table(mech, w1s=ws, w2s=ws)
    n_correct = n_total = 0
    for r in all_rows:
        if r["n_ne"] == 0:
            continue
        n_total += 1
        if r["ratio"] > 1 and r["n_eps1_gt"] == 0:
            n_correct += 1
        elif r["ratio"] < 1 and r["n_eps1_lt"] == 0:
            n_correct += 1
        elif r["ratio"] == 1 and r["n_eps1_lt"] == r["n_eps1_gt"]:
            n_correct += 1
    print(f"[Summary] {n_correct}/{n_total} (w_1, w_2) cells satisfy the "
          f"self-sorting prediction; sweep {len(ws)} x {len(ws)} = "
          f"{len(ws)**2} cells.")


if __name__ == "__main__":
    main()
