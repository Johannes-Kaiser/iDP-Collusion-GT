"""Grid-free equilibrium points of the symmetric 2-player PEG.

Computes, by continuous root/fixed-point search on the exact mechanism
(Newton-tight solves, batch_rtol=1e-6, PLD advantage):

  (3a) the symmetric equilibrium eps* : the fixed point of the local
       own-argmax map phi(eps) = argmax_e u1(e, eps) near the diagonal,
       found by bisection on phi(eps) - eps with golden-section for the
       inner argmax; plus a full-range global-argmax check at eps*.

  (3b) the frontier equilibrium pair: the loose player's unconstrained
       own-FOC zero G(el) = du_l/d el at (e_t = f(el), el), with
       f the analytic participation frontier; plus the tight player's
       corner condition and both players' global-argmax checks; plus
       the flatness profile |G(el)| and u_l variation across the
       observed grid string el in [4, 12].

Writes tasks_findings/_scripts/29_exact_equilibria.pkl and prints a
summary. Run via slurm_scripts/run_exact_eq.sh.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from game.advantage import mia_advantage_subsampled_gaussian as adv
from game.mechanism import epsilon_rdp, solve_mechanism
from game.utility import value_fn

STEPS, DELTA = 390, 1e-12
PBAR = 1e-5
P_LOOSE = 128.0 / 5000.0 - PBAR
N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
N_BISECT = int(os.environ.get("EQ_N_BISECT", "25"))
GOLD_TOL = float(os.environ.get("EQ_GOLD_TOL", "5e-4"))
BATCH_RTOL = float(os.environ.get("EQ_BATCH_RTOL", "1e-6"))

import logging
logging.getLogger("absl").setLevel(logging.ERROR)
OUT = ROOT / "tasks_findings" / "_scripts" / "29_exact_equilibria.pkl"

_cache: dict = {}


def u1(e1: float, e2: float) -> float:
    key = (round(e1, 10), round(e2, 10))
    if key in _cache:
        return _cache[key]
    r = solve_mechanism([e1, e2], [5000, 5000], 128, STEPS, DELTA,
                        batch_rtol=BATCH_RTOL)
    a1 = adv(r.sigma, float(r.sample_rates[0]), STEPS)
    val = value_fn(r.sigma) - a1
    if not r.feasible:
        val = -2.0
    _cache[key] = val
    return val


def golden_argmax(e2, lo, hi, tol=None):
    tol = GOLD_TOL if tol is None else tol
    """argmax_e u1(e, e2) over [lo, hi] in log-eps, golden section."""
    gr = (math.sqrt(5) - 1) / 2
    a, b = math.log(lo), math.log(hi)
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = u1(math.exp(c), e2), u1(math.exp(d), e2)
    while (b - a) > tol:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = u1(math.exp(c), e2)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = u1(math.exp(d), e2)
    e = math.exp((a + b) / 2)
    return e, u1(e, e2)


def frontier(eps_loose: float) -> float:
    lo, hi = 1e-3, 1e5
    if epsilon_rdp(hi, P_LOOSE, STEPS, DELTA) > eps_loose:
        return 0.0
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if epsilon_rdp(mid, P_LOOSE, STEPS, DELTA) > eps_loose:
            lo = mid
        else:
            hi = mid
    return epsilon_rdp(math.sqrt(lo * hi), PBAR, STEPS, DELTA)


def G_loose(el: float, h_rel: float = 2e-3) -> float:
    """Unconstrained own-FOC derivative of the loose player at the
    frontier profile (e_t = f(el), el): d u_loose / d e_loose."""
    et = frontier(el)
    h = el * h_rel
    return (u1(el + h, et) - u1(el - h, et)) / (2 * h)


def flat_point(el: float):
    et = frontier(el)
    return dict(el=el, et=et, G=G_loose(el), u=u1(el, et))


def main() -> None:
    res = {}

    # ---- (3a) symmetric equilibrium --------------------------------
    def psi(e):
        # local argmax within +-15% of the diagonal
        phi, _ = golden_argmax(e, e * 0.85, e * 1.15)
        return math.log(phi) - math.log(e)

    lo, hi = 2.5, 3.5
    plo, phi_ = psi(lo), psi(hi)
    print(f"[3a] psi(2.5)={plo:+.5f} psi(3.5)={phi_:+.5f}")
    if plo * phi_ < 0:
        for _ in range(N_BISECT):
            mid = math.sqrt(lo * hi)
            pm = psi(mid)
            print(f'[3a] bisect: eps={mid:.4f} psi={pm:+.5f}', flush=True)
            if pm * plo > 0:
                lo, plo = mid, pm
            else:
                hi, phi_ = mid, pm
        e_star = math.sqrt(lo * hi)
    else:
        e_star = float("nan")
    # global check across the whole action range
    g_arg, g_val = golden_argmax(e_star, 0.5, 64.0, tol=2e-4)
    res["symmetric"] = dict(e_star=e_star, u_at=u1(e_star, e_star),
                            global_argmax=g_arg, global_val=g_val)
    print(f"[3a] symmetric fixed point eps* = {e_star:.4f}; "
          f"global argmax vs eps* : {g_arg:.4f} "
          f"(u {g_val:.6f} vs {u1(e_star, e_star):.6f})")

    # ---- (3b) frontier equilibrium ----------------------------------
    els = np.geomspace(4.0, 12.0, 17)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        prof = list(pool.map(flat_point, [float(e) for e in els]))
    res["flatness"] = prof
    print("[3b] flatness profile along the frontier "
          "(el, et=f(el), G=du_l/del, u_l):")
    for r in prof:
        print(f"    el={r['el']:7.3f} et={r['et']:.4f} "
              f"G={r['G']:+.6f} u={r['u']:.6f}")
    # root of G by bisection on the profile's sign changes
    roots = []
    for k in range(len(prof) - 1):
        if prof[k]["G"] * prof[k + 1]["G"] < 0:
            a, b = prof[k]["el"], prof[k + 1]["el"]
            ga = prof[k]["G"]
            for _ in range(20):
                m = math.sqrt(a * b)
                gm = G_loose(m)
                if gm * ga > 0:
                    a, ga = m, gm
                else:
                    b = m
            el_star = math.sqrt(a * b)
            et_star = frontier(el_star)
            # corner condition for the tight player: deviation up loses
            hu = et_star * 5e-3
            corner = u1(et_star + hu, el_star) - u1(et_star, el_star)
            # global checks
            gl_arg, gl_val = golden_argmax(et_star, 0.5, 64.0, tol=2e-4)
            gt_arg, gt_val = golden_argmax(el_star, 0.5, 64.0, tol=2e-4)
            roots.append(dict(el=el_star, et=et_star,
                              corner_dev_gain=corner,
                              loose_global=(gl_arg, gl_val,
                                            u1(el_star, et_star)),
                              tight_global=(gt_arg, gt_val,
                                            u1(et_star, el_star))))
            print(f"[3b] FOC root: (e_t, e_l) = ({et_star:.4f}, "
                  f"{el_star:.4f}); tight corner dev gain {corner:+.2e}")
            print(f"      loose global argmax {gl_arg:.4f} "
                  f"(u {gl_val:.6f} vs {u1(el_star, et_star):.6f}); "
                  f"tight global argmax {gt_arg:.4f} "
                  f"(u {gt_val:.6f} vs {u1(et_star, el_star):.6f})")
    res["frontier_roots"] = roots
    us = [r["u"] for r in prof]
    print(f"[3b] u_l variation across the string el in [4,12]: "
          f"max-min = {max(us)-min(us):.2e}")

    pickle.dump(res, open(OUT, "wb"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
