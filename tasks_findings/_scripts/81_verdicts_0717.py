"""Pre-registered analyses for the 2026-07-17 landings (verdicts in
nummeric_eval.md Secs. 20.1 and 20.3).

(1) 52b/52c (jobs 180626/180627): exact-ladder deviation sweeps at the
    symmetric candidates {2.30, 2.43, 2.548, 2.60}. u2 = V(sigma) -
    lam*A2 along the 21-point p2 grid; gap = max u2 - u2(diagonal).
    Result: diagonal is the argmax at ALL candidates; at 2.548 the
    margin over the nearest sampled deviations is 1.2e-3 / 4.2e-3
    (vs 4.3e-4 certified bracket width). Sec. 20.1 verdict resolved.

(2) 55 vs 56 (job 180594): identical 220-point window [0.9, 1.2],
    lambda = 2, 89 vs 761 Renyi orders. Payoffs shift <= 7.9e-4;
    teeth shrink 3x and multiply 52 -> 119; the two 2-cycles collapse
    onto 4 pure NE at eps in [1.025, 1.029] (two symmetric). Sec.
    20.3 decision rule outcome (b).
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "_scripts"


def gap_analysis(lam=1.0):
    for name in ("52c_t390_saddle", "52b_t390_fixedpoint"):
        d = pickle.load(open(ROOT / f"{name}.pkl", "rb"))
        for entry in d["columns"][390]:
            rows = [r for r in entry["rows"] if r["ok"]]
            sig = np.array([r["sigma"] for r in rows])
            A2 = np.array([r["A2"] for r in rows])
            diag = np.array([r.get("diag", False) for r in rows])
            u2 = 1.0 / (1.0 + sig**2) - lam * A2
            k = int(np.where(diag)[0][0])
            gap = u2.max() - u2[k]
            order = np.argsort(u2)[::-1]
            margin = u2[k] - u2[order[1]] if order[0] == k else -gap
            print(f"e1 = {entry['e1']}: gap = {gap:+.6f}, "
                  f"diag-vs-runner-up margin = {margin:+.6f}")


def zoom_compare(lam=2.0):
    old = pickle.load(open(ROOT / "55_cycle_zoom.pkl", "rb"))
    new = pickle.load(open(ROOT / "56_cycle_zoom_dense.pkl", "rb"))
    acts = old["actions"]
    out = {}
    for tag, d in (("89", old), ("761", new)):
        V = 1.0 / (1.0 + d["sigma"]**2)
        U1 = np.where(d["feasible"], V - lam * d["A"][:, :, 0], -np.inf)
        U2 = np.where(d["feasible"], V - lam * d["A"][:, :, 1], -np.inf)
        B1, B2 = np.argmax(U1, axis=0), np.argmax(U2, axis=1)
        j = np.abs(np.diff(acts[B2]))
        step = np.median(np.diff(acts))
        nes = [(i, k) for i in range(len(acts)) for k in range(len(acts))
               if U1[i, k] >= U1[:, k].max() - 1e-14
               and U2[i, k] >= U2[i, :].max() - 1e-14]
        print(f"{tag} orders: teeth {np.sum(j > 3*step)}, max tooth "
              f"{j.max():.4f}, pure NE {len(nes)}: "
              f"{[(round(acts[a],4), round(acts[b],4)) for a, b in nes]}")
        out[tag] = (U2, B1, B2)
    dU = np.abs(out['761'][0] - out['89'][0])
    dU = dU[np.isfinite(dU)]
    print(f"payoff shift 89->761: median {np.median(dU):.2e}, "
          f"max {dU.max():.2e}")


if __name__ == "__main__":
    print("== (1) exact-ladder symmetric-NE verification, lambda = 1")
    gap_analysis()
    print("\n== (2) dense-order zoom comparison, lambda = 2")
    zoom_compare()
