"""Audit of the paper's quantitative claims against the post-solver-fix caches.

Cache-only (no mechanism solves). Every block prints the paper's claim and
the value the current cache gives. Output: 82_paper_audit.txt.

    .venv/bin/python tasks_findings/_scripts/82_paper_audit.py > tasks_findings/_scripts/82_paper_audit.txt
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "game_experiments" / "_cache_v2"
mech = pickle.load(open(C / "mech_grid_K200_g5000x5000_b128_e5.pkl", "rb"))
acts, SIG, A, F = mech["actions"], mech["sigma"], mech["A"], mech["feasible"]
K = len(acts)


def utilities(V, w1=1.0, w2=1.0, sig=SIG, AA=A, FF=F):
    v = V(sig)
    U = np.stack([v - w1 * AA[:, :, 0], v - w2 * AA[:, :, 1]], axis=-1)
    U[~FF] = -np.inf
    return U


def pure_ne(U, FF=F):
    br1 = np.argmax(U[:, :, 0], axis=0)          # best row for each column
    br2 = np.argmax(U[:, :, 1], axis=1)          # best column for each row
    return [(i, j) for i in range(U.shape[0]) for j in range(U.shape[1])
            if FF[i, j] and br1[j] == i and br2[i] == j]


def idx(e):
    return int(np.argmin(np.abs(np.log(acts) - np.log(e))))


V1 = lambda s: 1 / (1 + s ** 2)
print("# Paper audit against post-fix caches (K=200 canonical grid)\n")

# ---------------------------------------------------------------- 1 NE set
U = utilities(V1)
nes = pure_ne(U)
diag = [acts[i] for i, j in nes if i == j]
print(f"[1] pure NE at lambda=1: {len(nes)}, diagonal: {np.round(diag, 4)}")
off = [(i, j) for i, j in nes if i < j]
ratios = [acts[j] / acts[i] for i, j in off]
print(f"    ratio band {min(ratios):.2f}-{max(ratios):.2f}, median {np.median(ratios):.2f}")

# ---------------------------------------------------------------- 2 V-shape ablation
shapes = {
    "V1 1/(1+s^2)": V1,
    "V2 1/(1+s)": lambda s: 1 / (1 + s),
    "V3 exp(-s^2/2)": lambda s: np.exp(-s ** 2 / 2),
    "V4 exp(-s)": lambda s: np.exp(-s),
    "V5 -log(1+s)": lambda s: -np.log1p(s),
    "V6 -s": lambda s: -s,
}
print("\n[2] V-shape ablation (paper Fig. 14 / tab:H5: 48/60/48/44/48/54 NE, 100% off-diagonal)")
base = np.array([(np.log10(acts[i]), np.log10(acts[j])) for i, j in nes])
for name, V in shapes.items():
    n_ = pure_ne(utilities(V))
    d_ = [round(acts[i], 3) for i, j in n_ if i == j]
    pts = np.array([(np.log10(acts[i]), np.log10(acts[j])) for i, j in n_])
    dist = np.mean([np.min(np.linalg.norm(pts - b, axis=1)) for b in base]) if len(pts) else np.nan
    print(f"    {name:16s} #NE={len(n_):3d}  diagonal={d_}  mean V1->V dist={dist:.3f} dec")

# ---------------------------------------------------------------- 3 common lambda sweep + PoEV
iu = idx(5.73)
sum_uni = A[iu, iu, 0] + A[iu, iu, 1]
print(f"\n[3] common-lambda sweep; uniform baseline eps={acts[iu]:.3f}, sum A={sum_uni:.3f}")
for lam in (0.125, 0.25, 0.5, 1, 2, 4):
    n_ = pure_ne(utilities(V1, lam, lam))
    if not n_:
        print(f"    lambda={lam}: 0 NE")
        continue
    po = np.array([(A[i, j, 0] + A[i, j, 1] - sum_uni) / sum_uni for i, j in n_])
    dg = [round(acts[i], 3) for i, j in n_ if i == j]
    print(f"    lambda={lam}: {len(n_)} NE, diag {dg}, PoEV [{po.min():+.2f}, {po.max():+.2f}] "
          f"median {np.median(po):+.2f}, inflationary {np.mean(po > 0):.0%}")
sumA = {(i, j): A[i, j, 0] + A[i, j, 1] for i, j in nes}
best = min(sumA, key=sumA.get)
print(f"    lambda=1 least-aggregate-advantage NE: ({acts[best[0]]:.3f}, {acts[best[1]]:.3f}) sumA={sumA[best]:.3f}")
W = {(i, j): U[i, j, 0] + U[i, j, 1] for i, j in nes}
wb = max(W, key=W.get)
print(f"    lambda=1 welfare-best NE: ({acts[wb[0]]:.3f}, {acts[wb[1]]:.3f}) W={W[wb]:.4f}")

# ---------------------------------------------------------------- 4 diagonal trough stats
gains = []
for k in range(K):
    if not F[k, k]:
        continue
    col = U[:, k, 0]
    gains.append(col.max() - U[k, k, 0])
gains = np.array(gains)
print(f"\n[4] diagonal: {len(gains)} feasible cells, median best-deviation gain {np.median(gains):.3f}, "
      f"cells with zero gain {np.sum(gains <= 1e-12)}  (paper: 200 cells, median 0.122)")

# ---------------------------------------------------------------- 5 Hawk-Dove restriction
t, l = idx(1.12), idx(7.13)
uT_L, uL_T = U[t, l, 0], U[l, t, 0]
uT_T, uL_L = U[t, t, 0], U[l, l, 0]
q = (uL_L - uT_L) / ((uT_T - uL_T) - (uT_L - uL_L))
val = q * uT_T + (1 - q) * uT_L
print(f"\n[5] Hawk-Dove at ({acts[t]:.3f},{acts[l]:.3f}): tight vs loose {uT_L:.3f}/{U[t, l, 1]:.3f}; "
      f"(T,T)={uT_T:.3f} (L,L)={uL_L:.3f}; mixed P(tight)={q:.3f}, value={val:.3f}  "
      f"(paper: 0.505/0.253, 0.445, 0.349)")

# ---------------------------------------------------------------- 6 non-DSIC witness
for e2 in (4.0, 0.5):
    j = idx(e2)
    print(f"[6] best report of P1 vs eps2={acts[j]:.3f}: {acts[np.argmax(U[:, j, 0])]:.3f}  "
          f"(paper: 2.92 vs 4.0, 1.21 vs 0.5)")

# ---------------------------------------------------------------- 7 group size (C2, 9-point grid)
c2 = pickle.load(open(C / "C2.pkl", "rb"))
print("\n[7] group-size sweep (paper: minority eps1* in [1.5,3] vs ~10; A1<0.01 vs A2 in [0.24,0.32])")
for r in c2["rows"]:
    a = r["actions"]
    print(f"    |G1|/|G2|={r['ratio']}: NE " +
          ", ".join(f"({a[i]:.2f},{a[j]:.2f}|A={r['A'][i, j, 0]:.3f},{r['A'][i, j, 1]:.3f})"
                    for i, j in r["nes"]))

# ---------------------------------------------------------------- 8 sensitivity-based control
sc = pickle.load(open(ROOT / "game_experiments/_cache_scale/C7_K200.pkl", "rb"))
print("\n[8] sensitivity-based, common lambda (paper: single diagonal NE at 0.25/0.5/1)")
for r in sc["rows"]:
    a = r["actions"]
    print(f"    lambda={r.get('w', r.get('lam'))}: " + ", ".join(f"({a[i]:.2f},{a[j]:.2f})" for i, j in r["nes"]))

# ---------------------------------------------------------------- 9 K=22 overview (Fig. 3)
c1 = pickle.load(open(C / "C1.pkl", "rb"))
a = c1["actions"]
print(f"\n[9] K=22 pure NE ({len(c1['nes'])}): " + ", ".join(f"({a[i]:.2f},{a[j]:.2f})" for i, j in c1["nes"])
      + "  (Fig. 3 caption: every pure NE off the diagonal; app:sensitivity: 8 NE at K=22)")

# ---------------------------------------------------------------- 10 externality certificate
for name, ax in (("cross dA1/deps2", 1), ("own dA1/deps1", 0)):
    d = np.diff(A[:, :, 0], axis=ax)
    f = (F[1:, :] & F[:-1, :]) if ax == 0 else (F[:, 1:] & F[:, :-1])
    d = d[f]
    print(f"[10] {name}: N={d.size} neg={np.sum(d < 0)} pos={np.sum(d > 0)} tie={np.sum(d == 0)}")

# ---------------------------------------------------------------- 11 n-player Kendall tau (fixed analysis)
from scipy.stats import kendalltau
h3 = pickle.load(open(C / "H3_nplayer.pkl", "rb"))
print("\n[11] n-player heterogeneous sort compliance, Kendall tau(w, eps) < 0 "
      "(raw FP histories, last 50 rounds; paper: 100% n<=8, 83% n=10)")
for e in h3:
    if e["w_regime"] != "heterogeneous":
        continue
    n = e["n"]
    w = np.geomspace(0.25, 4, n)
    if e["equilibria"] is not None:
        profs = [tuple(p) for p in e["equilibria"]]
    else:
        profs = [tuple(float(x) for x in p) for h in e["histories"] for p in h[-50:]]
    taus = [kendalltau(w, p)[0] for p in profs]
    taus = np.array([t for t in taus if not np.isnan(t)])
    print(f"    n={n:2d}: {len(profs)} profiles, tau<0 in {np.mean(taus < 0):.0%}, "
          f"tau=nan (all-equal) {len(profs) - len(taus)}")
for e in h3:
    if e["w_regime"] == "symmetric":
        print(f"    symmetric n={e['n']}: modal profile {e['summary']['modal_profile']}")
