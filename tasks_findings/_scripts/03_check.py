"""Numerical sanity check for Proposition (No symmetric pure NE).

For each available equilibrium cache, enumerate every diagonal profile
(eps_i, eps_i) on the action grid and check whether it survives as a pure
NE. Report the maximum-gap unilateral deviation at every diagonal profile
so the (negative) result is quantitative rather than binary.

We interpret a profile (i, j) as a pure NE if
    U[i, j, 0] >= max_k U[k, j, 0]  (player 1 cannot deviate)
    U[i, j, 1] >= max_k U[i, k, 1]  (player 2 cannot deviate)
where the max is over feasible actions only.

The proposition claims that no diagonal profile (i, i) is a pure NE in the
symmetric 2-player PEG. We verify this on:
    - C1 (K=22, w=1, |G_1|=|G_2|=5000) -- fully symmetric baseline
    - C2 rows -- swept group-size ratio; for the r=1 row we also expect no
      diagonal NE; for r != 1 the game is not symmetric so the proposition
      does not apply, but the enumeration is still informative
    - C5 rows if present
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
CACHE = ROOT / "game_experiments" / "_cache_v2"

TOL = 1e-10


def diagonal_ne_scan(
    U: np.ndarray, feasible: np.ndarray, actions: np.ndarray
) -> list[dict[str, Any]]:
    """Enumerate all diagonal profiles and report NE status and worst-case gap."""
    K = len(actions)
    rows = []
    for i in range(K):
        if not feasible[i, i]:
            rows.append(
                dict(idx=i, eps=float(actions[i]), feasible=False, is_NE=False,
                     gap1=float("nan"), gap2=float("nan"))
            )
            continue
        u1_here = U[i, i, 0]
        u2_here = U[i, i, 1]
        # best unilateral deviation for each player among feasible actions
        col_feas = feasible[:, i]
        row_feas = feasible[i, :]
        u1_col = np.where(col_feas, U[:, i, 0], -np.inf)
        u2_row = np.where(row_feas, U[i, :, 1], -np.inf)
        best1 = float(u1_col.max())
        best2 = float(u2_row.max())
        gap1 = best1 - u1_here
        gap2 = best2 - u2_here
        is_NE = (gap1 <= TOL) and (gap2 <= TOL)
        rows.append(
            dict(idx=i, eps=float(actions[i]), feasible=True, is_NE=bool(is_NE),
                 gap1=float(gap1), gap2=float(gap2))
        )
    return rows


def print_table(name: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n=== Diagonal-profile scan: {name} ===")
    print(f"{'idx':>4} {'eps':>10} {'feasible':>9} {'gap1':>12} {'gap2':>12} {'NE?':>5}")
    for r in rows:
        eps = f"{r['eps']:.4g}"
        g1 = f"{r['gap1']:.3e}" if r["feasible"] else "  --  "
        g2 = f"{r['gap2']:.3e}" if r["feasible"] else "  --  "
        ne = "YES" if r["is_NE"] else "no"
        print(f"{r['idx']:>4} {eps:>10} {str(r['feasible']):>9} {g1:>12} {g2:>12} {ne:>5}")
    n_diag_ne = sum(1 for r in rows if r["is_NE"])
    print(f"  --> diagonal NEs found: {n_diag_ne}/{len(rows)}")


def scan_c1() -> tuple[int, list[dict[str, Any]]]:
    with (CACHE / "C1.pkl").open("rb") as fh:
        d = pickle.load(fh)
    sw = d["sweep"]
    rows = diagonal_ne_scan(sw["U"], sw["feasible"], sw["actions"])
    print_table("C1 (K=22, w=1, |G_1|=|G_2|=5000, fully symmetric)", rows)
    # cross-check: none of the reported off-diagonal NEs sit on the diagonal
    nes = d["nes"]
    diag_in_nes = [(i, j) for (i, j) in nes if i == j]
    print(f"  cache-reported pure NEs: {nes}")
    print(f"  of which on the diagonal: {diag_in_nes}")
    return sum(1 for r in rows if r["is_NE"]), rows


def scan_c2() -> int:
    with (CACHE / "C2.pkl").open("rb") as fh:
        d = pickle.load(fh)
    tot = 0
    for row in d["rows"]:
        rat = row["ratio"]
        rows = diagonal_ne_scan(row["U"], row["feasible"], row["actions"])
        label = f"C2 ratio={rat:g}, |G_1|={row['g1']}, |G_2|={row['g2']}"
        applies = "APPLIES" if row["g1"] == row["g2"] else "not symmetric"
        print_table(f"{label} [{applies}]", rows)
        nes = row["nes"]
        diag_nes = [(i, j) for (i, j) in nes if i == j]
        print(f"  cache-reported pure NEs: {nes}")
        print(f"  of which on the diagonal: {diag_nes}")
        if row["g1"] == row["g2"]:
            tot += sum(1 for r in rows if r["is_NE"])
    return tot


def scan_c5() -> int:
    p = CACHE / "C5.pkl"
    if not p.exists():
        return 0
    with p.open("rb") as fh:
        d = pickle.load(fh)
    # C5 stores rows with an actions/U/feasible triple keyed by the swept
    # parameter (V family etc.). Only diag-NE-relevant if fully symmetric.
    tot = 0
    for row in d.get("rows", []):
        if "U" not in row or "feasible" not in row or "actions" not in row:
            continue
        rows = diagonal_ne_scan(row["U"], row["feasible"], row["actions"])
        label = "C5 row=" + str({k: v for k, v in row.items() if not hasattr(v, "shape")})[:80]
        print_table(label, rows)
        tot += sum(1 for r in rows if r["is_NE"])
    return tot


if __name__ == "__main__":
    n_c1, _ = scan_c1()
    n_c2 = scan_c2()
    n_c5 = scan_c5()
    total = n_c1 + n_c2 + n_c5
    print("\n=== SUMMARY ===")
    print(f"Total diagonal pure NEs found across symmetric-hypothesis instances: {total}")
    print("Prediction of Proposition (No symmetric pure NE): 0.")
    if total == 0:
        print("Sanity check PASSES.")
    else:
        print("Sanity check FAILS -- investigate parameter regime.")
