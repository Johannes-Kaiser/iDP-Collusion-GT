"""T04 numerical corroboration.

Cache-hit-only evaluation of the slack decomposition
    slack_i = c_bar(eps_i) - A_i(eps)
across the C1, C2, C5 caches in game_experiments/_cache_v2/. Produces the
demonstration table and the monotonicity summary that back
prop:slack-monotonicity in the findings.

Run:
    python -m tasks_findings._scripts.04_slack_numerics
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

CACHE = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory/game_experiments/_cache_v2")
DELTA = 1e-12


def cbar(e: np.ndarray | float) -> np.ndarray | float:
    """Post-processing-tight DP MIA advantage bound of Hayes--Balle 2024
    / Kaissis--Ziller--Cohen 2024:  (e^eps - 1 + 2 delta) / (e^eps + 1).
    """
    return (np.exp(e) - 1.0 + 2.0 * DELTA) / (np.exp(e) + 1.0)


def _monotone_stats(slack, feas):
    """Count consecutive (j -> j+1) pairs that are weakly non-decreasing
    in the j direction (i fixed) over the feasibility mask."""
    K = slack.shape[0]
    n_pairs = 0
    n_dec = 0
    worst = 0.0
    for i in range(K):
        js = np.where(feas[i])[0]
        for a, b in zip(js[:-1], js[1:]):
            n_pairs += 1
            if slack[i, b] < slack[i, a] - 1e-12:
                n_dec += 1
                worst = max(worst, slack[i, a] - slack[i, b])
    return n_pairs, n_dec, worst


def _idx(actions: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(actions - target)))


def analyse_C1():
    with open(CACHE / "C1.pkl", "rb") as f:
        d = pickle.load(f)
    sweep = d["sweep"]
    actions = np.array(sweep["actions"])
    A = np.array(sweep["A"])
    feas = np.array(sweep["feasible"])
    K = len(actions)

    slack1 = cbar(actions[:, None]) - A[..., 0]
    slack1[~feas] = np.nan

    print("=== C1 (2-player symmetric, K=22) ===")
    print(f"  grid: {K} log-spaced points in [{actions[0]:.3f}, {actions[-1]:.3f}]")
    print(f"  feasible cells: {int(feas.sum())}/{K*K}")
    print(f"  slack in feasible region: min={np.nanmin(slack1):.4f}, "
          f"med={np.nanmedian(slack1):.4f}, max={np.nanmax(slack1):.4f}")
    print(f"  slack negativity violations: {int((slack1 < 0).sum())}")
    print(f"  slack > cbar violations: {int(((slack1 - cbar(actions[:, None])) > 0).sum())}")

    n, n_dec, worst = _monotone_stats(slack1, feas)
    print(f"  monotonicity in eps_j: {n - n_dec}/{n} nondecreasing "
          f"({100*(n-n_dec)/n:.1f}%); worst decrease {worst:.2e}")

    print("\n  Anchor table (eps_i x eps_j):")
    print(f"  {'eps_i':>7} {'eps_j':>7} {'cbar(eps_i)':>12} {'A_1':>8} {'slack_1':>10}")
    for ei in [1.0, 5.04, 32.0]:
        i = _idx(actions, ei)
        cb = cbar(actions[i])
        for ej in [1.0, 5.04, 32.0]:
            j = _idx(actions, ej)
            if feas[i, j]:
                print(f"  {actions[i]:7.3f} {actions[j]:7.3f} {cb:12.4f} "
                      f"{A[i,j,0]:8.4f} {cb - A[i,j,0]:10.4f}")


def analyse_C2():
    with open(CACHE / "C2.pkl", "rb") as f:
        d = pickle.load(f)
    n_ratios = len(d["rows"])
    K_c2 = len(d["rows"][0]["actions"])
    print(f"\n=== C2 (2-player, {n_ratios} group-ratio slices, K={K_c2}) ===")
    tot_pairs = 0
    tot_dec = 0
    worst = 0.0
    for r in d["rows"]:
        actions = np.array(r["actions"])
        A = np.array(r["A"])
        feas = np.array(r["feasible"])
        slack1 = cbar(actions[:, None]) - A[..., 0]
        slack1[~feas] = np.nan
        n, n_dec, w = _monotone_stats(slack1, feas)
        tot_pairs += n
        tot_dec += n_dec
        worst = max(worst, w)
        print(f"  ratio={r['ratio']:5.2f} (|G_1|={r['g1']:5d}, |G_2|={r['g2']:5d}): "
              f"feas={int(feas.sum()):3d}, {n - n_dec}/{n} monotone, "
              f"slack in [{np.nanmin(slack1):.3f},{np.nanmax(slack1):.3f}]")
    print(f"  pooled: {tot_pairs - tot_dec}/{tot_pairs} monotone "
          f"({100*(tot_pairs-tot_dec)/tot_pairs:.1f}%); worst dec {worst:.2e}")


def analyse_C5():
    with open(CACHE / "C5.pkl", "rb") as f:
        d = pickle.load(f)
    rows = d["rows"]
    teps = sorted({r["target_eps"] for r in rows})
    props = sorted({r["proportion"] for r in rows})
    print("\n=== C5 (5-player, target vs. others) ===")

    tot_pairs = 0
    tot_dec = 0
    for te in teps:
        for pr in props:
            seq = sorted(
                [r for r in rows if r["target_eps"] == te and r["proportion"] == pr],
                key=lambda r: r["other_eps"],
            )
            if not seq:
                continue
            slacks = [cbar(te) - r["A_target"] for r in seq]
            strict_dec = sum(1 for a, b in zip(slacks[:-1], slacks[1:])
                             if b < a - 1e-12)
            tot_pairs += len(seq) - 1
            tot_dec += strict_dec
            print(f"  target_eps={te:5.1f} pi={pr:.2f}: n={len(seq):2d}, "
                  f"slack range [{min(slacks):.4f},{max(slacks):.4f}], "
                  f"strict_dec={strict_dec}")
    print(f"  pooled: {tot_pairs - tot_dec}/{tot_pairs} monotone")

    print("\n  Anchor sequence at target_eps=4, pi=0.6:")
    seq = sorted(
        [r for r in rows if r["target_eps"] == 4.0 and r["proportion"] == 0.6],
        key=lambda r: r["other_eps"],
    )
    print(f"  {'eps_other':>10} {'A_target':>10} {'slack_target':>14} {'sigma':>8}")
    for r in seq:
        s = cbar(r["target_eps"]) - r["A_target"]
        eo = r["other_eps"]
        at = r["A_target"]
        sg = r["sigma"]
        print(f"  {eo:10.3f} {at:10.4f} {s:14.4f} {sg:8.4f}")


def demo_delta_neutralisation():
    """Illustrate prop:delta-neutralisation by taking every pair of feasible
    profiles in C1 that share eps_1, computing |A_1(eps) - A_1(eps')|, and
    reporting the maximum swing per anchor eps_1. If a Delta-cap were
    enforced, the swing would be bounded by Delta by construction; without
    a cap the swing is dictated by the range of the slack."""
    with open(CACHE / "C1.pkl", "rb") as f:
        d = pickle.load(f)
    sweep = d["sweep"]
    actions = np.array(sweep["actions"])
    A = np.array(sweep["A"])
    feas = np.array(sweep["feasible"])
    slack1 = cbar(actions[:, None]) - A[..., 0]
    slack1[~feas] = np.nan
    print("\n=== Delta-neutralisation illustration (C1) ===")
    print(f"  {'eps_1':>7} {'cbar':>8} {'A_1 min':>10} {'A_1 max':>10} "
          f"{'swing = slack range':>22}")
    for i in range(len(actions)):
        row_A = A[i, :, 0][feas[i]]
        row_s = slack1[i][feas[i]]
        if row_A.size < 2:
            continue
        print(f"  {actions[i]:7.3f} {cbar(actions[i]):8.4f} "
              f"{row_A.min():10.4f} {row_A.max():10.4f} "
              f"{row_s.max() - row_s.min():22.4f}")


if __name__ == "__main__":
    analyse_C1()
    analyse_C2()
    analyse_C5()
    demo_delta_neutralisation()
