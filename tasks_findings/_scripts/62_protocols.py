"""Protocol robustness of the dynamics claims (register row A6).

Reruns the attractor census on the K=200 cache (T=390) under four
update styles and compares against the alternating-BR results of
Secs. 11/21:
  alt     alternating best response (baseline: P2 replies, then P1)
  simu    simultaneous best response (both jump at once)
  damp    damped: move only half the distance (in log eps) toward the
          best response each round (both players, simultaneous)
  rand    random order each round (seeded), census over 25 restarts

Questions: does the lambda=2 cycle survive gentler play? does the
lambda=1 attractor census (cycle share, wall share, symmetric share)
depend on the protocol? does first-mover determinism disappear when
nobody strictly moves first?
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.continuous import best_response_curve, sweep_from_grid

mech = pickle.load(open(
    ROOT / "game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl",
    "rb"))
acts = np.asarray(mech["actions"])
K = len(acts)
LOG = np.log(acts)


def nearest(val):
    return int(np.argmin(np.abs(LOG - np.log(val))))


def run_census(lam, proto, n_starts=400, seed=7):
    sweep = sweep_from_grid(mech, lam, lam)
    feas = sweep["feasible"]
    BR1 = best_response_curve(sweep, 0)
    BR2 = best_response_curve(sweep, 1)
    rng = np.random.default_rng(seed)
    starts = [(int(i), int(j)) for i, j in
              zip(rng.integers(0, K, n_starts), rng.integers(0, K, n_starts))
              if feas[i, j] and BR1[j] >= 0 and BR2[i] >= 0]

    def step(i, j, t):
        if proto == "alt":
            j2 = int(BR2[i]); return int(BR1[j2]), j2
        if proto == "simu":
            return int(BR1[j]), int(BR2[i])
        if proto == "damp":
            i2, j2 = int(BR1[j]), int(BR2[i])
            return (nearest(np.exp(0.5 * (LOG[i] + LOG[i2]))),
                    nearest(np.exp(0.5 * (LOG[j] + LOG[j2]))))
        if proto == "rand":
            if rng.random() < 0.5:
                j2 = int(BR2[i]); return int(BR1[j2]), j2
            i2 = int(BR1[j]); return i2, int(BR2[i2])
        raise ValueError(proto)

    census = {}
    for (i0, j0) in starts:
        seen, i, j = {}, i0, j0
        for t in range(400):
            if (i, j) in seen:
                cyc, (a, b) = [], (i, j)
                while True:
                    cyc.append((a, b))
                    a, b = step(a, b, t)
                    if (a, b) == (i, j) or len(cyc) > 60:
                        break
                key = ("fp", cyc[0]) if len(cyc) == 1 else ("cyc", len(cyc))
                census[key] = census.get(key, 0) + 1
                break
            seen[(i, j)] = t
            i, j = step(i, j, t)
        else:
            census[("nonconv", 0)] = census.get(("nonconv", 0), 0) + 1
    return census, len(starts)


for lam in (1.0, 2.0):
    print(f"===== lambda = {lam} =====")
    for proto in ("alt", "simu", "damp", "rand"):
        census, n = run_census(lam, proto)
        tot_cyc = sum(v for (k, x), v in census.items() if k == "cyc")
        tot_fp = sum(v for (k, x), v in census.items() if k == "fp")
        nonc = census.get(("nonconv", 0), 0)
        top = sorted(census.items(), key=lambda kv: -kv[1])[:4]
        desc = ", ".join(
            (f"fp{tuple(round(float(acts[z]), 2) for z in x)}" if k == "fp"
             else (f"{x}-cycle" if k == "cyc" else "nonconv")) + f":{v}"
            for (k, x), v in top)
        print(f"  {proto:5s}: {n} starts -> cycles {100*tot_cyc/n:.0f}%, "
              f"fixed pts {100*tot_fp/n:.0f}%, nonconv {100*nonc/n:.0f}% | top: {desc}")
