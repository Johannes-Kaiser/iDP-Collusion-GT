"""Holistic robustness sweep over the assumption register
(nummeric_eval.md Secs. 23-24).

Design: (i) one-factor-at-a-time around the baseline (main effects per
assumption axis), (ii) a jointly randomized sample over ALL axes at
once (interactions), (iii) a T=2 replication subset (does composition
change any verdict). Every configuration gets the same value-level
battery; results post-process into an invariance matrix
(finding x axis -> holds / flips / scoped).

Battery per configuration (2-player; n=3 gets a reduced diagonal-only
battery):
  b1 trough: fraction of sampled columns whose utility slice dips
     below both ends (the "some data at some risk is worst" claim)
  b2 corner dominance: BR kind census over columns (lo/int/hi)
  b3 equilibrium taxonomy: floor pair / ceiling / kind-switch (wall
     segment) flags from corner ties
  b4 symmetric stability: diagonal argmax location + deviation gap
     (best column reply minus staying) at the diagonal optimum
  b5 welfare: specialization premium = (grid welfare max - diagonal
     welfare max) / |grid max|
  b6 Stackelberg: leader value and role (commit tight vs loose)
     vs the diagonal-optimum payoff

Env: PEG_N_WORKERS, PEG_OUT (60_robustness.pkl), PEG_SEED,
PEG_N_RANDOM (default 160), PEG_T2 (1 = include T=2 subset).
"""
from __future__ import annotations

import itertools
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game_experiments.robust_game import Game  # noqa: E402

N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
N_RANDOM = int(os.environ.get("PEG_N_RANDOM", "160"))
SEED = int(os.environ.get("PEG_SEED", "20260715"))
WITH_T2 = os.environ.get("PEG_T2", "1") == "1"
OUT = ROOT / "tasks_findings" / "_scripts" / os.environ.get(
    "PEG_OUT", "60_robustness.pkl")

BASE = dict(T=1, G=(5000, 5000), pbar=1e-5, delta=1e-12,
            box=(0.5, 64.0), V_form="inv_sq", cost_form="linear",
            lam=0.7)

AXES = dict(
    V_form=["inv_sq", "inv", "exp"],
    cost_form=["linear", "quad"],
    lam=[0.05, 0.7, 2.0, 28.0],
    lam_quad=[200.0, 1000.0, 3000.0],     # weights for the quad shape
    G=[(500, 500), (5000, 5000), (50000, 50000), (2000, 8000)],
    pbar=[1e-4, 1e-5, 1e-6],
    delta=[1e-10, 1e-12, 1e-14],
    box=[(0.5, 64.0), (0.25, 128.0), (1.0, 16.0)],
    het=[1.0, 4.0],                        # lam2/lam1
)


def battery(cfg: dict) -> dict:
    g = Game(**{k: v for k, v in cfg.items() if k != "tag"})
    lo, hi = g.box
    cols = np.geomspace(lo, hi, 7)
    n_samp = 48
    out = dict(cfg=cfg, ok=True)
    kinds, troughs = [], 0
    W_best, W_diag = -np.inf, -np.inf
    stack = (-np.inf, None)
    diag_u, diag_es = [], np.geomspace(lo, hi, 26)

    # diagonal scan (n-player symmetric)
    n = len(g.G)
    for e in diag_es:
        r = g.utilities([float(e)] * n)
        diag_u.append(r[0][0] if r else np.nan)
        if r and n == 2:
            W_diag = max(W_diag, r[0][0] + r[0][1])
    diag_u = np.array(diag_u)
    fin = np.isfinite(diag_u)
    out["diag_argmax"] = float(diag_es[fin][np.argmax(diag_u[fin])]) if fin.any() else np.nan
    out["diag_interior"] = bool(fin.any() and
                                0 < int(np.argmax(np.where(fin, diag_u, -np.inf))) < len(diag_es) - 1)

    if n > 2:                       # reduced battery for n >= 3
        out["kinds"] = None
        return out

    for e1 in cols:
        try:
            f_lo, f_hi = g.feasible_e2(float(e1))
        except Exception:
            continue
        a, b = max(lo, f_lo * (1 + 1e-9)), min(hi, f_hi * (1 - 1e-9))
        if not (a < b):
            continue
        xs = np.geomspace(a, b, n_samp)
        us1, us2 = np.full(n_samp, np.nan), np.full(n_samp, np.nan)
        for k, e2 in enumerate(xs):
            r = g.utilities([float(e1), float(e2)])
            if r:
                us1[k], us2[k] = r[0][0], r[0][1]
        finc = np.isfinite(us2)
        if finc.sum() < 5:
            continue
        u2 = us2[finc]; u1 = us1[finc]; xv = xs[finc]
        k = int(np.argmax(u2))
        kinds.append("lo" if k == 0 else ("hi" if k == len(u2) - 1 else "int"))
        if u2.min() < min(u2[0], u2[-1]) - 1e-12:
            troughs += 1
        Wc = np.nanmax(u1 + u2)
        W_best = max(W_best, float(Wc))
        if u1[k] > stack[0]:
            stack = (float(u1[k]), float(e1), float(xv[k]), float(u2[k]))
        # deviation gap at the diagonal point of this column (if inside)
        if xv[0] <= e1 <= xv[-1]:
            ud = float(np.interp(np.log(e1), np.log(xv), u2))
            gap = float(u2.max() - ud)
            out.setdefault("gaps", []).append((float(e1), gap))

    out["kinds"] = {kk: kinds.count(kk) for kk in ("lo", "int", "hi")}
    out["trough_frac"] = troughs / max(len(kinds), 1)
    out["W_best"] = float(W_best)
    out["W_diag"] = float(W_diag)
    out["premium"] = (float((W_best - W_diag) / abs(W_best))
                      if np.isfinite(W_best) and np.isfinite(W_diag)
                      and W_best != 0 else np.nan)
    out["stackelberg"] = stack
    return out


def configs():
    cfgs = [dict(BASE, tag="baseline")]
    # one-factor-at-a-time
    for ax, levels in AXES.items():
        for v in levels:
            c = dict(BASE)
            if ax == "lam_quad":
                c["cost_form"], c["lam"] = "quad", v
            elif ax == "het":
                c["lam"] = (0.7, 0.7 * v)
            elif ax == "lam":
                c["lam"] = v
            else:
                c[ax] = v
            if c != BASE:
                cfgs.append(dict(c, tag=f"ofat:{ax}={v}"))
    # jointly randomized
    rng = np.random.default_rng(SEED)
    for k in range(N_RANDOM):
        cost = rng.choice(AXES["cost_form"])
        lam = (float(np.exp(rng.uniform(np.log(0.02), np.log(50)))) if cost == "linear"
               else float(np.exp(rng.uniform(np.log(50), np.log(5000)))))
        if rng.random() < 0.25:
            lam = (lam, lam * float(np.exp(rng.uniform(0, np.log(6)))))
        c = dict(T=1,
                 G=AXES["G"][rng.integers(len(AXES["G"]))],
                 pbar=float(rng.choice(AXES["pbar"])),
                 delta=float(rng.choice(AXES["delta"])),
                 box=AXES["box"][rng.integers(len(AXES["box"]))],
                 V_form=str(rng.choice(AXES["V_form"])),
                 cost_form=str(cost), lam=lam, tag=f"rand:{k}")
        cfgs.append(c)
    # 3-player spot checks
    for Gs in [(3333, 3333, 3334), (1000, 4000, 5000)]:
        cfgs.append(dict(BASE, G=Gs, tag=f"n3:{Gs}"))
    # T=2 replication subset
    if WITH_T2:
        picks = [dict(BASE), dict(BASE, lam=2.0), dict(BASE, lam=28.0),
                 dict(BASE, V_form="inv"), dict(BASE, V_form="exp"),
                 dict(BASE, cost_form="quad", lam=1000.0),
                 dict(BASE, G=(500, 500)), dict(BASE, pbar=1e-6),
                 dict(BASE, box=(0.25, 128.0)), dict(BASE, lam=(0.7, 2.8))]
        for k, c in enumerate(picks):
            cfgs.append(dict(c, T=2, tag=f"t2:{k}"))
    return cfgs


def main() -> None:
    cfgs = configs()
    print(f"{len(cfgs)} configurations", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(battery, c): c for c in cfgs}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as ex:
                r = dict(cfg=futs[f], ok=False, err=str(ex))
            results.append(r)
            if len(results) % 20 == 0:
                print(f"{len(results)}/{len(cfgs)}", flush=True)
                pickle.dump(results, open(OUT, "wb"))
    pickle.dump(results, open(OUT, "wb"))
    print(f"wrote {OUT} ({len(results)} results)")


if __name__ == "__main__":
    main()
