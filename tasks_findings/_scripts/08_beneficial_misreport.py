"""
T08 -- Numerical witness that the sampling-based iDP mechanism, viewed as a
direct-revelation mechanism on the budget-declaration channel, is *not*
dominant-strategy incentive compatible.

Idea. The identity mechanism (Boenisch et al.) uses the declared budget
epsilon_i directly as the input to solve_mechanism. Suppose for contradiction
that this is DSIC: then there is a truth-telling map f: w_i -> epsilon-hat
such that reporting f(w_i) is a *dominant strategy* for player i, i.e.
optimal against every opponent profile. We falsify DSIC by exhibiting two
opponent profiles eps_2^A, eps_2^B at which the best-response of player 1
lands at strictly different budgets. No single f(w_1) can be optimal against
both, so DSIC fails.

We also print one explicit "beneficial misreport" at which the epsilon-value
that maximises player 1's utility in scenario A is strictly worse than a
different epsilon-value in scenario B (with a numerical utility gap).

Parameters mirror the C5 experiment cache: two silos, |G|=25000, E_b=128,
I = 5*50000/128 = 1953 steps, delta=1e-12. Player weights w_i = 1.0.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from game.mechanism import solve_mechanism_cached
from game.advantage import mia_advantage_subsampled_gaussian
from game.utility import value_fn


# ---------- fixed setting -----------------------------------------------------
GROUP_SIZES = (25_000, 25_000)
E_B = 128
STEPS = int(5 * 50_000 / 128)  # 1953, matching the C5 cache
DELTA = 1e-12
W1 = 1.0  # privacy weight of player 1 (the "true type")

# Log-spaced budget grid matching the K=22 grid used in the paper (a coarser
# subset is enough to exhibit the DSIC failure).
EPS_GRID = np.geomspace(0.5, 64.0, 12)


def evaluate(eps1: float, eps2: float) -> dict:
    res = solve_mechanism_cached(
        eps_target=[float(eps1), float(eps2)],
        group_sizes=list(GROUP_SIZES),
        expected_batch_size=E_B,
        steps=STEPS,
        delta=DELTA,
    )
    p1 = float(res.sample_rates[0])
    p2 = float(res.sample_rates[1])
    sigma = float(res.sigma)
    A1 = mia_advantage_subsampled_gaussian(sigma=sigma, sample_rate=p1, steps=STEPS)
    A2 = mia_advantage_subsampled_gaussian(sigma=sigma, sample_rate=p2, steps=STEPS)
    u1 = value_fn(sigma) - W1 * A1
    return {
        "eps1": float(eps1),
        "eps2": float(eps2),
        "sigma": sigma,
        "p1": p1,
        "p2": p2,
        "A1": A1,
        "A2": A2,
        "u1": u1,
        "feasible": bool(res.feasible),
    }


def best_response(eps2: float) -> dict:
    """Argmax over EPS_GRID of player-1 utility at fixed eps2, restricted to
    feasible profiles."""
    rows = [evaluate(e1, eps2) for e1 in EPS_GRID]
    feas = [r for r in rows if r["feasible"]]
    if not feas:
        raise RuntimeError(f"No feasible profile at eps2={eps2}")
    best = max(feas, key=lambda r: r["u1"])
    return {"rows": rows, "best": best}


def main() -> None:
    print("=" * 72)
    print("T08 -- DSIC failure of budget-declaration mechanism (numerical)")
    print("=" * 72)
    print(
        f"|G|={GROUP_SIZES}  E_b={E_B}  I={STEPS}  delta={DELTA}  w_1={W1}"
    )
    print()

    # Two opponent scenarios: high budget vs low budget.
    scenarios = [("A: eps_2 = 4.0", 4.0), ("B: eps_2 = 0.5", 0.5)]

    br = {}
    for label, eps2 in scenarios:
        print(f"--- Scenario {label}: player-1 utility over epsilon grid ---")
        info = best_response(eps2)
        for r in info["rows"]:
            marker = "  <-- argmax" if r is info["best"] else ""
            feas = "F" if r["feasible"] else "-"
            print(
                f"  [{feas}] eps_1={r['eps1']:7.3f}  sigma={r['sigma']:7.3f}  "
                f"p_1={r['p1']:.4f}  A_1={r['A1']:.4f}  u_1={r['u1']:+.4f}"
                f"{marker}"
            )
        br[label] = info["best"]
        print()

    labelA, labelB = scenarios[0][0], scenarios[1][0]
    bA = br[labelA]
    bB = br[labelB]

    print("=" * 72)
    print("DSIC test: is there a single dominant epsilon-report for player 1?")
    print("=" * 72)
    print(
        f"  argmax in {labelA}: eps_1 = {bA['eps1']:7.3f}   u_1 = {bA['u1']:+.4f}"
    )
    print(
        f"  argmax in {labelB}: eps_1 = {bB['eps1']:7.3f}   u_1 = {bB['u1']:+.4f}"
    )
    if abs(bA["eps1"] - bB["eps1"]) > 1e-9:
        print()
        print("  --> best-response of player 1 differs across opponent scenarios.")
        print("  --> no dominant strategy: the identity mechanism is NOT DSIC.")
    else:
        print("  --> best-response coincides; failure not witnessed on this grid.")

    # A single, explicit beneficial-misreport: take the argmax in scenario A as
    # the "candidate truthful report" and evaluate it in scenario B. Compare to
    # the argmax in scenario B. The gap is the strict utility gain from
    # misreporting.
    print()
    print("=" * 72)
    print("Explicit beneficial misreport (scenario B):")
    print("=" * 72)
    cand = evaluate(bA["eps1"], scenarios[1][1])
    print(
        f"  Reporting the {labelA} argmax eps_1={bA['eps1']:7.3f} in {labelB}: "
        f"u_1 = {cand['u1']:+.4f}"
    )
    print(
        f"  Reporting the {labelB} argmax eps_1={bB['eps1']:7.3f} in {labelB}: "
        f"u_1 = {bB['u1']:+.4f}"
    )
    gain = bB["u1"] - cand["u1"]
    print(f"  Strict utility gain from misreport: Delta u_1 = {gain:+.4f}")

    # Full best-response schedule across many eps_2 to show the failure is not
    # a corner artefact.
    print()
    print("=" * 72)
    print("Best-response schedule eps_1^*(eps_2) at w_1=1.0:")
    print("=" * 72)
    br_schedule = []
    for eps2 in EPS_GRID:
        try:
            b = best_response(float(eps2))["best"]
            br_schedule.append((eps2, b["eps1"], b["u1"]))
            print(
                f"  eps_2 = {eps2:7.3f}   -->   eps_1^* = {b['eps1']:7.3f}   "
                f"u_1^* = {b['u1']:+.4f}"
            )
        except RuntimeError as exc:
            print(f"  eps_2 = {eps2:7.3f}   -->   {exc}")

    distinct = sorted({round(x[1], 6) for x in br_schedule})
    print()
    print(
        f"  #distinct best-response epsilons across opponent grid: {len(distinct)}"
    )
    print(f"  distinct values: {distinct}")
    print(
        "  ==> the identity mechanism has no dominant-strategy report for w_1=1.0."
    )


if __name__ == "__main__":
    main()
