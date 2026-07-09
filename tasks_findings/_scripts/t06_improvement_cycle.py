"""Improvement cycle on the C1 cache: rules out an ordinal potential.

Monderer & Shapley (1996), Lemma 2.5: a finite game is an ordinal potential
game iff it has no *improvement cycle* -- a closed sequence of unilateral
deviations, each strictly improving the deviating player's utility, that
returns to the starting profile.

The Monderer-Shapley cross-partial test (T06 symbolic script) rules out
weighted potentials, but not ordinal potentials. This script exhibits an
improvement cycle on the C1 discretisation, closing that gap.

For a 2-player game with utility grid U[i, j, k] (player k, action i for
player 1, action j for player 2), an improvement cycle of length 4 is a
sequence
    (i0, j0) -> (i1, j0)  strict improvement for player 1
              -> (i1, j1)  strict improvement for player 2
              -> (i0, j1)  strict improvement for player 1
              -> (i0, j0)  strict improvement for player 2.
We search the C1 grid for the first such cycle.

Since the C1 configuration is fully symmetric (w1 = w2, |G_1| = |G_2|) and
its cross-partial vanishes on the diagonal, any obstruction must live in the
finite-cycle geometry rather than the derivative signature -- which is
exactly what an improvement cycle witnesses.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[2] / "game_experiments/_cache_v2/C1.pkl"


def find_improvement_cycle_4(U: np.ndarray, feas: np.ndarray):
    """Return the first improvement cycle of length 4 found on the grid.

    Iterates lexicographically over (i0, j0, i1, j1) with i0 != i1 and
    j0 != j1, and returns the tuple as soon as all four utility inequalities
    hold *strictly*.
    """
    K = U.shape[0]
    U0 = U[..., 0]  # player 1 utility
    U1 = U[..., 1]  # player 2 utility

    for i0 in range(K):
        for j0 in range(K):
            if not feas[i0, j0]:
                continue
            for i1 in range(K):
                if i1 == i0 or not feas[i1, j0]:
                    continue
                # step 1: player 1 improves from i0 -> i1 at j0
                if not (U0[i1, j0] > U0[i0, j0]):
                    continue
                for j1 in range(K):
                    if j1 == j0 or not feas[i1, j1] or not feas[i0, j1]:
                        continue
                    # step 2: player 2 improves from j0 -> j1 at i1
                    if not (U1[i1, j1] > U1[i1, j0]):
                        continue
                    # step 3: player 1 improves from i1 -> i0 at j1
                    if not (U0[i0, j1] > U0[i1, j1]):
                        continue
                    # step 4: player 2 improves from j1 -> j0 at i0
                    if not (U1[i0, j0] > U1[i0, j1]):
                        continue
                    return (i0, j0, i1, j1)
    return None


def main() -> None:
    with open(CACHE, "rb") as fh:
        data = pickle.load(fh)
    sw = data["sweep"]
    U = np.array(sw["U"])
    feas = np.array(sw["feasible"])
    eps = np.array(data["actions"])

    print("=" * 70)
    print("Improvement-cycle search on the C1 grid")
    print("=" * 70)
    print(f"Grid size:              K = {U.shape[0]}")
    print(f"Feasible cells:         {int(feas.sum())} / {feas.size}")
    print()

    cycle = find_improvement_cycle_4(U, feas)
    if cycle is None:
        print("No length-4 improvement cycle found.")
        return
    i0, j0, i1, j1 = cycle
    print(f"Improvement cycle found: (i0, j0, i1, j1) = {cycle}")
    print(
        f"Action budgets:          eps_1 in ({eps[i0]:.3f}, {eps[i1]:.3f}), "
        f"eps_2 in ({eps[j0]:.3f}, {eps[j1]:.3f})"
    )
    print()
    print("Sequence of unilateral improvements (strict):")
    step = 0
    profile = [i0, j0]

    def show_step(who: str, before: tuple, after: tuple, u_before: float, u_after: float):
        nonlocal step
        step += 1
        print(
            f"  step {step} ({who}): eps = ({eps[before[0]]:.3f},"
            f" {eps[before[1]]:.3f}) -> ({eps[after[0]]:.3f}, {eps[after[1]]:.3f})"
            f"    u_{who} : {u_before:+.5e} -> {u_after:+.5e}    delta = {u_after - u_before:+.3e}"
        )

    # (i0, j0) -> (i1, j0), player 1
    show_step("1", (i0, j0), (i1, j0), U[i0, j0, 0], U[i1, j0, 0])
    # (i1, j0) -> (i1, j1), player 2
    show_step("2", (i1, j0), (i1, j1), U[i1, j0, 1], U[i1, j1, 1])
    # (i1, j1) -> (i0, j1), player 1
    show_step("1", (i1, j1), (i0, j1), U[i1, j1, 0], U[i0, j1, 0])
    # (i0, j1) -> (i0, j0), player 2
    show_step("2", (i0, j1), (i0, j0), U[i0, j1, 1], U[i0, j0, 1])
    print()
    print("Conclusion: the four strict inequalities close a cycle around")
    print(f"({eps[i0]:.3f}, {eps[j0]:.3f}) with all deviations improving for")
    print("the deviating player. By Monderer-Shapley (1996, Lemma 2.5) the")
    print("PEG is not an ordinal potential game.")


if __name__ == "__main__":
    main()
