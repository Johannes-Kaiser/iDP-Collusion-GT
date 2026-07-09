"""
A4 pre-flight feasibility / spread scan for the FL-population experiments.

For each candidate action grid and each of the first ``--pops`` populations,
evaluate the level-2 oracle on the six uniform profiles and ``--n-random``
random profiles, and record:
  * feasibility rate (random profiles),
  * sigma range,
  * per-uniform-value population-median A_i (the spread the grid must show),
  * A_i range, wall time per solve.

Acceptance (plan A4): >=90% of random profiles feasible; sigma in [0.4, 6];
across the six uniform profiles the population-median A_i spans ~[0.05, 0.5];
median solve time <= 2 s (at DISC_FAST).

Run under Slurm (CPU). Parallelises over (grid, pop) tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from game_experiments.flpop_common import (
    CACHE_DIR, DISC_FAST, sample_population, Oracle,
)

# Candidate grids to compare (all log-spaced on their endpoints).
CANDIDATE_GRIDS = {
    "default": (10.0, 14.0, 19.0, 26.0, 37.0, 50.0),   # geomspace(10,50,6)
    "low":     (5.0, 7.0, 10.0, 15.0, 21.0, 30.0),     # geomspace(5,30,6)
    "mid":     (6.0, 9.0, 13.0, 19.0, 28.0, 40.0),     # geomspace(6,40,6)
}


def scan_one(args):
    grid_name, grid, pid, n_random, disc = args
    pop = sample_population(pid)
    orc = Oracle(pop, actions=grid, persist=False, discretization=disc)
    rng = np.random.default_rng(1000 + pid)
    # uniform profiles
    uni_medA = []
    sigmas = []
    times = []
    for e in grid:
        t0 = time.time()
        res = orc.evaluate([e] * pop.n)
        times.append(time.time() - t0)
        uni_medA.append(float(np.median(res.advantages)))
        sigmas.append(res.sigma)
    # random profiles
    feas = 0
    A_lo, A_hi = np.inf, -np.inf
    for _ in range(n_random):
        prof = rng.choice(grid, size=pop.n)
        t0 = time.time()
        res = orc.evaluate(prof)
        times.append(time.time() - t0)
        if res.feasible:
            feas += 1
            A_lo = min(A_lo, float(res.advantages.min()))
            A_hi = max(A_hi, float(res.advantages.max()))
            sigmas.append(res.sigma)
    return dict(
        grid=grid_name, pid=pid,
        uni_medA=uni_medA,
        sigma_min=float(np.min(sigmas)), sigma_max=float(np.max(sigmas)),
        feas_rate=feas / max(n_random, 1),
        A_lo=(None if not np.isfinite(A_lo) else A_lo),
        A_hi=(None if not np.isfinite(A_hi) else A_hi),
        med_time=float(np.median(times)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--grids", default="default,low,mid")
    ap.add_argument("--disc", type=float, default=DISC_FAST)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("PEG_N_WORKERS", 4)))
    args = ap.parse_args()

    grids = [g.strip() for g in args.grids.split(",")]
    tasks = [
        (gn, CANDIDATE_GRIDS[gn], pid, args.n_random, args.disc)
        for gn in grids for pid in range(args.pops)
    ]
    print(f"[A4] {len(tasks)} tasks, {args.workers} workers, disc={args.disc}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(scan_one, tasks):
            results.append(r)
            print(f"  [{r['grid']:8s} pop{r['pid']:02d}] feas={r['feas_rate']:.2f} "
                  f"sigma[{r['sigma_min']:.3f},{r['sigma_max']:.3f}] "
                  f"uni_medA={[round(a,3) for a in r['uni_medA']]} "
                  f"t~{r['med_time']:.2f}s", flush=True)

    out = CACHE_DIR / "a4_scan.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Per-grid aggregate summary.
    print("\n[A4] === SUMMARY (aggregated across populations) ===")
    for gn in grids:
        rs = [r for r in results if r["grid"] == gn]
        med_uni = np.median(np.array([r["uni_medA"] for r in rs]), axis=0)
        feas = np.mean([r["feas_rate"] for r in rs])
        smin = np.min([r["sigma_min"] for r in rs])
        smax = np.max([r["sigma_max"] for r in rs])
        mt = np.median([r["med_time"] for r in rs])
        print(f"  {gn:8s}: feas={feas:.3f}  sigma[{smin:.3f},{smax:.3f}]  "
              f"median-uni-A(low->high)={[round(a,3) for a in med_uni]}  "
              f"med_solve={mt:.2f}s")
    print(f"\n[A4] wrote {out}")


if __name__ == "__main__":
    main()
