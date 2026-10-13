"""Compare the H1 measured instance's stored mechanism solutions against
the fixed solve_mechanism (post bisection-bug fix, 2026-07-13).

For each of the 25 profiles on the 5x5 grid eps in {2,4,8,16,32} (Adult,
STEPS=4680), re-solve with the fixed solver and report the relative sigma
and rate deviations from the values stored in
game_experiments/_cache_H1/n2_adult/<idx>/profile_meta.txt (which
parameterised the DP-SGD shadow-model training). Decides whether the GPU
retraining of H1 is required.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.mechanism import solve_mechanism

EPS_GRID = [2.0, 4.0, 8.0, 16.0, 32.0]
STEPS = 4680
DELTA = 1e-12
CACHE = ROOT / "game_experiments" / "_cache_H1" / "n2_adult"


def main() -> None:
    worst = 0.0
    print("idx  (e1,e2)      sigma_old  sigma_new  d_sigma   d_p1     d_p2")
    for idx in range(25):
        i, j = divmod(idx, 5)
        e1, e2 = EPS_GRID[i], EPS_GRID[j]
        meta = (CACHE / str(idx) / "profile_meta.txt").read_text()
        s_old = float(re.search(r"sigma=([0-9.e+-]+)", meta).group(1))
        p_old = [float(x) for x in
                 re.findall(r"np\.float64\(([0-9.e+-]+)\)", meta)]
        res = solve_mechanism([e1, e2], [5000, 5000], 128, STEPS, DELTA)
        ds = abs(res.sigma - s_old) / s_old
        dp = [abs(res.sample_rates[k] - p_old[k]) / max(p_old[k], 1e-12)
              for k in range(2)]
        worst = max(worst, ds, *dp)
        print(f"{idx:3d}  ({e1:4.0f},{e2:4.0f})  {s_old:9.4f}  "
              f"{res.sigma:9.4f}  {ds*100:6.2f}%  {dp[0]*100:6.2f}%  "
              f"{dp[1]*100:6.2f}%")
    print(f"worst relative deviation across all profiles: {worst*100:.2f}%")


if __name__ == "__main__":
    main()
