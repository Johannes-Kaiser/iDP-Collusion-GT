"""Alternative plays on the PEG, cache-only / closed-form. Reproduces
every number quoted in nummeric_eval.md Sec. 22:

(a) Stackelberg commitment on the K=200 cache (lambda in {1, 2}):
    leader maximizes u1(x, BR2(x)).
(b) Collaborative (team) game, symmetric: welfare argmax vs diagonal
    (fair-share) optimum vs best pure NE.
(c) Heterogeneous collaborative game: lambda ratios (K=200 cache,
    weights applied post hoc) and group-size ratios (C2 cache).
(d) High-T behavior in the GDP limit (gdp_limit_game): Stackelberg,
    fairness premium, and the lambda-rescaling probe at T = 25000.

Run: ./.venv/bin/python tasks_findings/_scripts/59_alt_plays.py
     > tasks_findings/_scripts/59_alt_plays_summary.txt
"""
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.continuous import (best_response_curve, find_pure_NE_2player,
                             sweep_from_grid)

mech = pickle.load(open(
    ROOT / "game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl",
    "rb"))
acts = np.asarray(mech["actions"])

print("== (a) Stackelberg (K=200 cache) ==")
for lam in (1.0, 2.0):
    sweep = sweep_from_grid(mech, lam, lam)
    U, feas = sweep["U"], sweep["feasible"]
    BR2 = best_response_curve(sweep, 1)
    vals = np.array([U[i, BR2[i], 0]
                     if BR2[i] >= 0 and feas[i, BR2[i]] else -np.inf
                     for i in range(len(acts))])
    k = int(np.argmax(vals))
    print(f"  lam={lam}: commit {acts[k]:.4f} -> reply {acts[BR2[k]]:.4f}; "
          f"leader {vals[k]:.5f}, follower {U[k, BR2[k], 1]:.5f}")

print("== (b) collaborative game, symmetric (K=200 cache) ==")
for lam in (1.0, 2.0):
    sweep = sweep_from_grid(mech, lam, lam)
    U, feas = sweep["U"], sweep["feasible"]
    W = np.where(feas, U[:, :, 0] + U[:, :, 1], -np.inf)
    i, j = np.unravel_index(np.argmax(W), W.shape)
    diag = max(W[k, k] for k in range(len(acts)))
    nes = find_pure_NE_2player(sweep)
    Wne = max((W[a, b] for a, b in nes), default=float("nan"))
    print(f"  lam={lam}: welfare opt ({acts[i]:.3f}, {acts[j]:.3f}) "
          f"W={W[i, j]:.4f}; fair diag {diag:.4f}; best NE {Wne:.4f}")

print("== (c) heterogeneity ==")
for r in (1.0, 1.5, 2.0, 4.0, 8.0):
    sweep = sweep_from_grid(mech, 1.0, r)
    U, feas = sweep["U"], sweep["feasible"]
    W = np.where(feas, U[:, :, 0] + U[:, :, 1], -np.inf)
    i, j = np.unravel_index(np.argmax(W), W.shape)
    diag = max(W[k, k] for k in range(len(acts)))
    nes = find_pure_NE_2player(sweep)
    Wne = max((W[a, b] for a, b in nes), default=float("nan"))
    print(f"  lam2/lam1={r}: opt ({acts[i]:.2f}, {acts[j]:.2f}) W={W[i, j]:.4f} "
          f"| equal-budget {diag:.4f} | best NE {Wne:.4f}")
d = pickle.load(open(ROOT / "game_experiments/_cache_v2/C2.pkl", "rb"))
for row in d["rows"]:
    a = np.asarray(row["actions"])
    W = np.where(row["feasible"], row["U"][:, :, 0] + row["U"][:, :, 1],
                 -np.inf)
    i, j = np.unravel_index(np.argmax(W), W.shape)
    print(f"  G ratio {row['ratio']}: G=({row['g1']},{row['g2']}), "
          f"opt ({a[i]:.2f}, {a[j]:.2f}) W={W[i, j]:.4f}")

print("== (d) GDP limit, high T ==")
import game_experiments.gdp_limit_game as g


def metrics(T, lam, n1=30, n2=37):
    e1s = np.geomspace(0.5, 64.0, n1)
    best_st, W_best, W_diag = None, -9e9, -9e9
    kinds = {"hi": 0, "int": 0, "lo": 0}
    for e1 in e1s:
        rows = g.column(T, float(e1), n_p=n2)
        cand = []
        for row in rows:
            if not (0.5 - 1e-9 <= row["e2"] <= 64.0 + 1e-9):
                continue
            p1 = g.PSUM - row["p2"]
            u1 = g.value(row["sigma"]) - lam * g.advantage(row["sigma"], p1, T)
            u2 = g.value(row["sigma"]) - lam * row["A2"]
            cand.append((u1, u2, row))
            W_best = max(W_best, u1 + u2)
            if row["diag"]:
                W_diag = max(W_diag, 2 * u2)
        if not cand:
            continue
        k = int(np.argmax([c[1] for c in cand]))
        kinds["hi" if k == len(cand) - 1 else ("lo" if k == 0 else "int")] += 1
        if best_st is None or cand[k][0] > best_st[0]:
            best_st = (cand[k][0], cand[k][1], float(e1), cand[k][2]["e2"])
    return best_st, W_best, W_diag, kinds


for (T, lam) in [(390, 1.0), (1560, 1.0), (6240, 1.0), (25000, 1.0),
                 (25000, 0.5), (25000, 0.35), (25000, 0.25), (25000, 0.125)]:
    st, Wb, Wd, kinds = metrics(float(T), lam)
    prem = (Wb - Wd) / Wb * 100 if Wb > 0 else float("nan")
    print(f"  T={T:6d} lam={lam:5.3f}: kinds {kinds} | Stackelberg "
          f"{st[2]:6.2f}->{st[3]:6.2f} ({st[0]:.4f}/{st[1]:.4f}) | "
          f"W {Wb:.4f} fair {Wd:.4f} premium {prem:5.1f}%")
