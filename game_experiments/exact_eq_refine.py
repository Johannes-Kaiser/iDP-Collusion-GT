"""Refinement of the exact-equilibrium search (follow-up to run 179691).

(3a') Dense psi scan on eps in [2.80, 3.32]: psi(eps) = log phi(eps) -
      log eps with phi the local (+-15% window) own-argmax. The fast run
      showed psi dips to +0.0036 near 2.96 at a 1.5e-3 noise floor and
      only crosses zero at the branch-jump point 3.2555 (rejected by the
      global check). This scan resolves whether the dip actually
      crosses (exact symmetric fixed point) or stays positive (flat
      near-equilibrium at grid resolution). Any sign change is bisected;
      any root gets a full-range global-argmax check.

(3b') Frontier flatness, evaluated a hair INSIDE the feasible region:
      e_t = f(e_l) * (1 + ET_OFFSET) so finite differences never cross
      the feasibility cliff (run 179691 evaluated on the knife edge and
      got the -2 infeasibility sentinel at 12/17 points).

Writes tasks_findings/_scripts/30_exact_eq_refine.pkl.
Run via slurm_scripts/run_exact_eq_refine.sh.
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
sys.path.insert(0, str(ROOT / "game_experiments"))

import exact_equilibria as eq  # noqa: E402  (reads EQ_* env at import)

N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))
ET_OFFSET = float(os.environ.get("EQ_ET_OFFSET", "2e-3"))
OUT = ROOT / "tasks_findings" / "_scripts" / "30_exact_eq_refine.pkl"


def psi_point(e: float) -> dict:
    phi, u_at = eq.golden_argmax(e, e * 0.85, e * 1.15)
    return dict(eps=e, phi=phi, psi=math.log(phi) - math.log(e), u=u_at)


def flat_point_inside(el: float) -> dict:
    et = eq.frontier(el) * (1.0 + ET_OFFSET)
    h = el * 2e-3
    up, um, u0 = eq.u1(el + h, et), eq.u1(el - h, et), eq.u1(el, et)
    ok = all(v > -1.5 for v in (up, um, u0))
    G = (up - um) / (2 * h) if ok else float("nan")
    return dict(el=el, et=et, G=G, u=u0, feasible=ok)


def main() -> None:
    res = {}

    # ---- (3a') dense psi scan ---------------------------------------
    grid = np.geomspace(2.80, 3.32, 27)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        scan = list(pool.map(psi_point, [float(e) for e in grid]))
    res["psi_scan"] = scan
    print("[3a'] psi scan:")
    for r in scan:
        print(f"    eps={r['eps']:.4f} phi={r['phi']:.4f} "
              f"psi={r['psi']:+.5f} u={r['u']:.6f}", flush=True)

    roots = []
    for k in range(len(scan) - 1):
        if scan[k]["psi"] * scan[k + 1]["psi"] < 0:
            a, b = scan[k]["eps"], scan[k + 1]["eps"]
            pa = scan[k]["psi"]
            for _ in range(12):
                m = math.sqrt(a * b)
                pm = psi_point(m)["psi"]
                print(f"[3a'] bisect [{a:.4f},{b:.4f}]: "
                      f"eps={m:.4f} psi={pm:+.5f}", flush=True)
                if pm * pa > 0:
                    a, pa = m, pm
                else:
                    b = m
            e_root = math.sqrt(a * b)
            g_arg, g_val = eq.golden_argmax(e_root, 0.5, 64.0, tol=2e-4)
            u_at = eq.u1(e_root, e_root)
            roots.append(dict(e_root=e_root, u_at=u_at,
                              global_argmax=g_arg, global_val=g_val))
            print(f"[3a'] root eps={e_root:.4f}; global argmax "
                  f"{g_arg:.4f} (u {g_val:.6f} vs {u_at:.6f})")
    res["psi_roots"] = roots
    if not roots:
        print("[3a'] no sign change: psi stays one-signed on the scan; "
              "the dip is a flat near-fixed-point, not an exact root")

    # ---- (3b') frontier flatness, inside the cliff -------------------
    els = np.geomspace(4.0, 12.0, 17)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        prof = list(pool.map(flat_point_inside, [float(e) for e in els]))
    res["flatness"] = prof
    print(f"[3b'] frontier profile at et = f(el)*(1+{ET_OFFSET:g}):")
    for r in prof:
        print(f"    el={r['el']:7.3f} et={r['et']:.4f} "
              f"G={r['G']:+.6e} u={r['u']:.6f} feas={r['feasible']}")
    us = [r["u"] for r in prof if r["feasible"]]
    if us:
        print(f"[3b'] u_l variation across feasible points: "
              f"max-min = {max(us) - min(us):.2e}")

    g_roots = []
    for k in range(len(prof) - 1):
        ga, gb = prof[k]["G"], prof[k + 1]["G"]
        if not (math.isnan(ga) or math.isnan(gb)) and ga * gb < 0:
            a, b = prof[k]["el"], prof[k + 1]["el"]
            for _ in range(20):
                m = math.sqrt(a * b)
                gm = flat_point_inside(m)["G"]
                if gm * ga > 0:
                    a, ga = m, gm
                else:
                    b = m
            el_star = math.sqrt(a * b)
            et_star = eq.frontier(el_star) * (1.0 + ET_OFFSET)
            hu = et_star * 5e-3
            corner = eq.u1(et_star + hu, el_star) - eq.u1(et_star, el_star)
            gl_arg, gl_val = eq.golden_argmax(et_star, 0.5, 64.0, tol=2e-4)
            gt_arg, gt_val = eq.golden_argmax(el_star, 0.5, 64.0, tol=2e-4)
            g_roots.append(dict(el=el_star, et=et_star,
                                corner_dev_gain=corner,
                                loose_global=(gl_arg, gl_val,
                                              eq.u1(el_star, et_star)),
                                tight_global=(gt_arg, gt_val,
                                              eq.u1(et_star, el_star))))
            print(f"[3b'] FOC root: (e_t, e_l) = ({et_star:.4f}, "
                  f"{el_star:.4f}); tight corner dev gain {corner:+.2e}")
            print(f"      loose global argmax {gl_arg:.4f} "
                  f"(u {gl_val:.6f} vs {eq.u1(el_star, et_star):.6f}); "
                  f"tight global argmax {gt_arg:.4f} "
                  f"(u {gt_val:.6f} vs {eq.u1(et_star, el_star):.6f})")
    res["frontier_roots"] = g_roots

    pickle.dump(res, open(OUT, "wb"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
