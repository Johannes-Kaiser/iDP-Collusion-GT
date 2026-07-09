"""
Phase D sanity checks for the FL-population experiments.

D1 (oracle self-consistency): for one population, brute-force the best response
of one player against a fixed profile by *direct* PLD evaluation (uncached
mechanism solve, no significant-digit rounding on sigma/p) and confirm it
matches the cached/rounded oracle's argmax. Guards against cache-rounding bugs.

D3 (feasibility audit): count how many best-response candidate profiles are
rejected as infeasible for a sample population, and inspect a few rejects to
confirm the participation floor -- not a solver failure -- is doing the
exclusion.

Run on CPU (Slurm). Small: a handful of solves per check.
"""
from __future__ import annotations

import argparse

import numpy as np

from game.mechanism import solve_mechanism
from game.advantage import mia_advantage_subsampled_gaussian
from game_experiments.flpop_common import (
    sample_population, Oracle, value_fn, EXPECTED_BATCH, DELTA, ACTION_GRID,
)
from game_experiments.flpop_a4_scan import CANDIDATE_GRIDS


def direct_utility(pop, profile, i, disc):
    """Player i's utility on `profile` via a *direct*, uncached, unrounded solve."""
    mech = solve_mechanism(
        eps_target=[float(e) for e in profile],
        group_sizes=[int(g) for g in pop.group_sizes],
        expected_batch_size=EXPECTED_BATCH, steps=pop.steps, delta=DELTA,
    )
    if not mech.feasible:
        return None
    A_i = mia_advantage_subsampled_gaussian(
        sigma=float(mech.sigma), sample_rate=float(mech.sample_rates[i]),
        steps=pop.steps, discretization=disc)
    return float(value_fn(mech.sigma) - pop.weights[i] * A_i)


def check_d1(pid: int, grid_name: str, disc: float, n_probe: int = 3) -> bool:
    actions = CANDIDATE_GRIDS[grid_name]
    pop = sample_population(pid)
    orc = Oracle(pop, actions=actions, persist=False, discretization=disc)
    rng = np.random.default_rng(pid)
    ok = True
    for probe in range(n_probe):
        base = rng.choice(actions, size=pop.n)
        i = int(rng.integers(pop.n))
        # oracle argmax (cached / sig6-rounded path)
        orc_u = {}
        for a in actions:
            t = list(base); t[i] = float(a)
            res = orc.evaluate(t)
            orc_u[a] = float(res.utilities[i]) if res.feasible else None
        orc_best = max((a for a in actions if orc_u[a] is not None),
                       key=lambda a: orc_u[a])
        # direct argmax (uncached / unrounded)
        dir_u = {}
        for a in actions:
            t = list(base); t[i] = float(a)
            dir_u[a] = direct_utility(pop, t, i, disc)
        dir_best = max((a for a in actions if dir_u[a] is not None),
                       key=lambda a: dir_u[a])
        max_abs = max(
            abs(orc_u[a] - dir_u[a])
            for a in actions if orc_u[a] is not None and dir_u[a] is not None)
        agree = (orc_best == dir_best)
        ok = ok and agree
        print(f"[D1 pop{pid} probe{probe}] player{i}: oracle_argmax={orc_best} "
              f"direct_argmax={dir_best} agree={agree} max|du|={max_abs:.2e}")
    return ok


def check_d3(pid: int, grid_name: str, disc: float, n_random: int = 60) -> None:
    actions = CANDIDATE_GRIDS[grid_name]
    pop = sample_population(pid)
    orc = Oracle(pop, actions=actions, persist=False, discretization=disc)
    rng = np.random.default_rng(500 + pid)
    n_infeasible = 0
    rejects = []
    for _ in range(n_random):
        prof = rng.choice(actions, size=pop.n)
        res = orc.evaluate(prof)
        if not res.feasible:
            n_infeasible += 1
            if len(rejects) < 3:
                min_p = float(res.sample_rates.min())
                argmin = int(np.argmin(res.sample_rates))
                rejects.append((float(prof[argmin]), int(pop.group_sizes[argmin]),
                                min_p, float(res.sigma)))
    print(f"[D3 pop{pid}] {n_infeasible}/{n_random} random profiles infeasible "
          f"({100*n_infeasible/n_random:.0f}%)")
    for eps, g, p, s in rejects:
        print(f"   reject: min sample-rate p={p:.2e} at (eps={eps}, |G|={g}), "
              f"sigma={s:.3f}  -> participation floor 1e-5 {'binds' if p < 1e-5 else 'NOT binding (inspect!)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="d1", choices=["d1", "d3", "all"])
    ap.add_argument("--pops", default="0,1,2")
    ap.add_argument("--grid", default="default", choices=list(CANDIDATE_GRIDS))
    ap.add_argument("--disc", type=float, default=1e-2)
    args = ap.parse_args()
    pops = [int(x) for x in args.pops.split(",")]

    if args.check in ("d1", "all"):
        all_ok = all(check_d1(p, args.grid, args.disc) for p in pops)
        print(f"[D1] overall {'PASS' if all_ok else 'FAIL'}")
    if args.check in ("d3", "all"):
        for p in pops:
            check_d3(p, args.grid, args.disc)


if __name__ == "__main__":
    main()
