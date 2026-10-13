"""T25 -- Export the C6 best-response geometry as .dat tables for the
native pgfplots figure ``paper/tex/fig_C6_tikz.tex``.

The best-response correspondence of the symmetric 2-player PEG is
two-branched: the own-payoff column u_1(., eps_2) has a tight and a
loose local maximum on either side of the diagonal trough
(prop:no-symmetric-ne), and the argmax flips between the branches as
the opponent's budget crosses a threshold.  A single connected argmax
line is therefore misleading (it draws branch flips as jumps); the
figure instead shows the eps-best-response bands -- every action whose
payoff is within TOL of the column maximum -- as point bands, plus the
raw argmax points and the 48 raw pure NEs (which lie exactly on both
raw argmax loci).

Reads the K=200 cache ``game_experiments/_cache_v2/C6_K200.pkl``
(cache-hit only, seconds on the login node).  Writes:

  paper/figures/data/c6_band_p1.dat   (eps1 eps2) near-BR band, player 1
  paper/figures/data/c6_band_p2.dat   (eps1 eps2) near-BR band, player 2
  paper/figures/data/c6_argmax_p1.dat (eps1 eps2) raw argmax, player 1
  paper/figures/data/c6_argmax_p2.dat (eps1 eps2) raw argmax, player 2
  paper/figures/data/c6_nes.dat       (eps1 eps2) the 48 pure NEs
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
CACHE = ROOT / "game_experiments" / "_cache_v2" / "C6_K200.pkl"
OUT = ROOT / "paper" / "figures" / "data"

TOL = 0.02  # utility distance to the column max that counts as near-BR


def near_br_band(U: np.ndarray, feas: np.ndarray, player: int):
    """(own_eps_idx, opp_eps_idx) pairs within TOL of the column max."""
    K = U.shape[0]
    pairs = []
    for j in range(K):
        if player == 0:
            col = U[:, j, 0].copy()
            col[~feas[:, j]] = -np.inf
        else:
            col = U[j, :, 1].copy()
            col[~feas[j, :]] = -np.inf
        if np.all(np.isneginf(col)):
            continue
        top = col.max()
        for k in np.where(col >= top - TOL)[0]:
            pairs.append((k, j) if player == 0 else (j, k))
    return pairs


def analytic_frontier(actions: np.ndarray) -> list:
    """The participation frontier in closed form.

    On the frontier player 2's rate sits at the participation floor
    PBAR, so the batch constraint pins player 1's rate at
    B/|G| - PBAR; sigma then solves player 1's budget alone, and
    eps2_min is one accountant evaluation at (sigma, PBAR).  Matches
    the grid frontier within half a grid cell; the grid's ragged
    cells are bisection tolerance scattered around this curve.
    """
    import math
    from game.mechanism import epsilon_rdp
    STEPS, DELTA, PBAR = 390, 1e-12, 1e-5
    P1 = 128.0 / 5000.0 - PBAR
    pts = []
    for e1 in np.geomspace(actions[0], actions[-1], 300):
        lo, hi = 0.3, 200.0
        if epsilon_rdp(hi, P1, STEPS, DELTA) > e1: continue
        if epsilon_rdp(lo, P1, STEPS, DELTA) < e1: continue
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            if epsilon_rdp(mid, P1, STEPS, DELTA) > e1:
                lo = mid
            else:
                hi = mid
        e2 = epsilon_rdp(math.sqrt(lo * hi), PBAR, STEPS, DELTA)
        if actions[0] <= e2 <= actions[-1]:
            pts.append((e1, e2))
    return pts


def main() -> None:
    data = pickle.load(open(CACHE, "rb"))
    U = data["sweep"]["U"]
    feas = data["sweep"]["feasible"]
    actions = data["actions"]
    br1, br2, nes = data["br1"], data["br2"], data["nes"]

    OUT.mkdir(parents=True, exist_ok=True)

    def write(name, pairs):
        path = OUT / name
        with open(path, "w") as f:
            f.write("eps1 eps2\n")
            for (i, j) in pairs:
                f.write(f"{actions[i]:.6g} {actions[j]:.6g}\n")
        print(f"wrote {path} ({len(pairs)} points)")

    write("c6_band_p1.dat", near_br_band(U, feas, player=0))
    write("c6_band_p2.dat", near_br_band(U, feas, player=1))
    write("c6_argmax_p1.dat",
          [(br1[j], j) for j in range(len(actions)) if br1[j] >= 0])
    write("c6_argmax_p2.dat",
          [(i, br2[i]) for i in range(len(actions)) if br2[i] >= 0])
    write("c6_nes.dat", list(nes))

    import sys
    sys.path.insert(0, str(ROOT))
    frontier = analytic_frontier(actions)
    with open(OUT / "c6_frontier.dat", "w") as f:
        f.write("eps1 eps2\n")
        for (x, y) in frontier:
            f.write(f"{x:.6g} {y:.6g}\n")
    print(f"wrote {OUT/'c6_frontier.dat'} ({len(frontier)} points)")

    on = sum((i == br1[j]) and (j == br2[i]) for (i, j) in nes)
    assert on == len(nes), "NE off the raw argmax loci"
    print(f"all {len(nes)} NEs lie on both raw argmax loci")


if __name__ == "__main__":
    main()
