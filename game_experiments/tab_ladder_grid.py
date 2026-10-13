"""Exact-ladder mechanism grid on the K x K action box via (sigma, p)
tabulation, run twice on offset node meshes to measure interpolation
error empirically.

Everything the game needs factors through two smooth, monotone scalar
fields of the exact composition: the budget E_T(sigma, p) and the
advantage Adv_T(sigma, p) (multi_step_game.Ladder — exact PLD ladder,
no accountant). Instead of one ladder solve per grid cell (~40 s x
40,000), we evaluate the ladder on a modest rectangular node mesh in
(log sigma, log p), interpolate with a bicubic spline, and run the
mechanism inversion (batch root-find in sigma with per-player rate
inversions) on the interpolants. The whole K x K grid then costs
milliseconds per cell.

Error control, two independent instruments:
  * Offset meshes. Mesh A is n_s x n_p log-spaced; mesh B places its
    interior nodes at the geometric midpoints of mesh A's cells (with
    shared endpoints, so coverage is identical). Both meshes produce a
    full mechanism grid; their per-cell disagreement in sigma, A, and
    u (after the inversion, i.e. in game coordinates) is an empirical
    bound on the interpolation error where it matters.
  * Anchors. tasks_findings/_scripts/52_t390_ladder.pkl stores per-row
    (sigma, p2, e2, A2) from the expensive per-cell ladder solves of
    job 180196 — direct ground truth for both interpolants, off-node
    by construction.

The output grid mirrors precompute_mechanism_grid's schema exactly
(actions / sigma / A / feasible / frontier_eps_min) so that
sweep_from_grid, find_pure_NE_2player, and the regret tooling work
unchanged.

Phases (CLI):
  nodes   --mesh A|B      evaluate the ladder on the node mesh
  grid    --mesh A|B      build interpolants, solve the K x K grid
  compare                 offset-mesh disagreement + anchor errors
Run each phase under Slurm; `nodes` is the only expensive one.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq

from game_experiments.multi_step_game import Ladder, PSUM, PBAR

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "_cache_v2"
RDP_GRID = CACHE / "mech_grid_K200_g5000x5000_b128_e5.pkl"
ANCHORS = ROOT.parent / "tasks_findings" / "_scripts" / "52_t390_ladder.pkl"

G = int(os.environ.get("PEG_G", "5000"))
B = 128
T_DEFAULT = 390
K_DEFAULT = 200

# Node-mesh extent. sigma must cover every sigma* the exact game can
# select on the box [0.5, 64]^2 (the RDP grid's feasible range is
# [0.368, 3.64]; exact accounting needs slightly less noise at the top
# and can go slightly below at the bottom — margins on both ends).
SIGMA_RANGE = (0.26, 4.8)
P_RANGE = (PBAR, PSUM - PBAR)
N_SIGMA = 44
N_P = 40

EPS_MIN, EPS_MAX = 0.5, 64.0
BATCH_RTOL = 1e-6          # spline evals are cheap; converge the batch hard
FLOOR_TOL = 1e-12          # p_i >= PBAR * (1 - FLOOR_TOL) counts as feasible


def node_mesh(mesh: str):
    """Mesh A: log-spaced. Mesh B: geometric midpoints of A's cells with
    shared endpoints (same coverage, interior fully staggered)."""
    sa = np.geomspace(*SIGMA_RANGE, N_SIGMA)
    pa = np.geomspace(*P_RANGE, N_P)
    if mesh == "A":
        return sa, pa
    sb = np.concatenate([[sa[0]], np.sqrt(sa[:-1] * sa[1:]), [sa[-1]]])
    pb = np.concatenate([[pa[0]], np.sqrt(pa[:-1] * pa[1:]), [pa[-1]]])
    return sb, pb


def nodes_path(mesh: str, T: int) -> Path:
    return CACHE / f"tab_nodes_{mesh}_T{T}_g{G}.pkl"


def grid_path(mesh: str, T: int, K: int) -> Path:
    return CACHE / f"exact_grid_K{K}_T{T}_g{G}_mesh{mesh}.pkl"


def _eval_node(args):
    s, p, T = args
    lad = Ladder(s, p, T)
    return lad.eps_budget(), lad.adv()


def run_nodes(mesh: str, T: int, n_workers: int) -> None:
    sn, pn = node_mesh(mesh)
    path = nodes_path(mesh, T)
    E = np.full((len(sn), len(pn)), np.nan)
    A = np.full((len(sn), len(pn)), np.nan)
    if path.exists():
        with open(path, "rb") as f:
            prev = pickle.load(f)
        if prev["sigma_nodes"].shape == sn.shape and np.allclose(prev["sigma_nodes"], sn):
            E, A = prev["E"], prev["Adv"]
    todo = [(i, j) for i in range(len(sn)) for j in range(len(pn))
            if not np.isfinite(E[i, j])]
    print(f"mesh {mesh}: {len(sn)}x{len(pn)} nodes, {len(todo)} to do", flush=True)
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_eval_node, (float(sn[i]), float(pn[j]), T)): (i, j)
                for (i, j) in todo}
        done = 0
        for fut in as_completed(futs):
            i, j = futs[fut]
            E[i, j], A[i, j] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(todo):
                with open(path, "wb") as f:
                    pickle.dump(dict(sigma_nodes=sn, p_nodes=pn, E=E, Adv=A,
                                     T=T, G=G, B=B, mesh=mesh), f)
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(todo)} ({rate:.2f}/s, "
                      f"eta {(len(todo)-done)/max(rate,1e-9)/60:.0f} min)",
                      flush=True)
    with open(path, "wb") as f:
        pickle.dump(dict(sigma_nodes=sn, p_nodes=pn, E=E, Adv=A,
                         T=T, G=G, B=B, mesh=mesh), f)
    print(f"nodes done -> {path}", flush=True)


class TabModel:
    """Bicubic interpolants of log E and log Adv over (log sigma, log p),
    plus the mechanism inversion run on them."""

    def __init__(self, mesh: str, T: int):
        with open(nodes_path(mesh, T), "rb") as f:
            d = pickle.load(f)
        if not np.all(np.isfinite(d["E"])) or not np.all(np.isfinite(d["Adv"])):
            raise RuntimeError("node table incomplete")
        self.mesh, self.T = mesh, T
        self.ls = np.log(d["sigma_nodes"])
        self.lp = np.log(d["p_nodes"])
        self._logE = RectBivariateSpline(self.ls, self.lp, np.log(d["E"]),
                                         kx=3, ky=3)
        self._logA = RectBivariateSpline(self.ls, self.lp, np.log(d["Adv"]),
                                         kx=3, ky=3)

    def eps(self, sigma: float, p: float) -> float:
        return math.exp(float(self._logE(math.log(sigma), math.log(p))))

    def adv(self, sigma: float, p: float) -> float:
        return min(math.exp(float(self._logA(math.log(sigma), math.log(p)))), 1.0)

    def rate_for_eps(self, sigma: float, eps: float):
        """phi(sigma, eps): p with E(sigma, p) = eps. Returns (p, clamp)
        with clamp in {None, "low", "high"}: "low" = the contract binds
        only below the participation floor (required p < PBAR), "high" =
        it cannot be exhausted within the batch (required p > PSUM-PBAR).
        A clamped rate is NOT a solution of E = eps — the caller must
        treat any clamped coordinate as infeasible; the clamped value
        only keeps the batch bisection bracketed and monotone."""
        ls, le = math.log(sigma), math.log(eps)
        f = lambda lp: float(self._logE(ls, lp)) - le
        if f(self.lp[0]) >= 0.0:
            return 0.0, "low"                # even p = PBAR exceeds eps
        if f(self.lp[-1]) <= 0.0:
            return math.exp(self.lp[-1]), "high"
        return math.exp(brentq(f, self.lp[0], self.lp[-1],
                               xtol=1e-13, rtol=8.9e-16)), None

    def solve_cell(self, e1: float, e2: float):
        """Mirror of solve_mechanism on the interpolants: geometric
        bisection of sigma on the batch equation, feasibility = batch
        converged and both rates >= PBAR."""
        target = float(B)

        def batch(ls: float):
            s = math.exp(ls)
            r1, c1 = self.rate_for_eps(s, e1)
            r2, c2 = self.rate_for_eps(s, e2)
            return G * (r1 + r2) - target, r1, r2, c1, c2

        lo, hi = self.ls[0], self.ls[-1]
        if batch(hi)[0] < 0.0 or batch(lo)[0] > 0.0:
            return None                      # batch unreachable in range
        f = lambda ls: batch(ls)[0]
        ls_star = brentq(f, lo, hi, xtol=1e-12, rtol=8.9e-16)
        bb, r1, r2, c1, c2 = batch(ls_star)
        s_star = math.exp(ls_star)
        # Feasible = the true system is solved: batch met AND both
        # contracts actually binding (no clamped coordinate). A "low"
        # clamp is the participation wall (required p < PBAR); a "high"
        # clamp means the loose contract cannot be exhausted within the
        # batch — both are infeasible in the deployed convention.
        feasible = (abs(bb) < BATCH_RTOL * target * 10
                    and c1 is None and c2 is None
                    and r1 >= PBAR * (1 - 1e-9) and r2 >= PBAR * (1 - 1e-9))
        return dict(sigma=s_star, p1=r1, p2=r2, feasible=feasible,
                    A1=self.adv(s_star, r1) if r1 > 0 else np.nan,
                    A2=self.adv(s_star, r2) if r2 > 0 else np.nan)

    def frontier_eps_min(self, e_opp: float) -> float:
        """Budget the floor player is pinned to when the opponent holds
        e_opp: opponent takes the whole batch minus the floor."""
        p_opp = PSUM - PBAR
        f = lambda ls: float(self._logE(ls, math.log(p_opp))) - math.log(e_opp)
        if f(self.ls[0]) * f(self.ls[-1]) > 0:
            return float("nan")          # sigma-bar outside the table range
        ls_bar = brentq(f, self.ls[0], self.ls[-1], xtol=1e-13)
        return self.eps(math.exp(ls_bar), PBAR)


def run_grid(mesh: str, T: int, K: int) -> None:
    model = TabModel(mesh, T)
    if RDP_GRID.exists() and K == K_DEFAULT:
        with open(RDP_GRID, "rb") as f:
            actions = pickle.load(f)["actions"]
    else:
        actions = np.geomspace(EPS_MIN, EPS_MAX, K)
    S = np.full((K, K), np.nan)
    A = np.full((K, K, 2), np.nan)
    feas = np.zeros((K, K), dtype=bool)
    t0 = time.time()
    for i in range(K):
        e1 = float(actions[i])
        for j in range(i, K):
            sol = model.solve_cell(e1, float(actions[j]))
            if sol is None or not sol["feasible"]:
                continue
            S[i, j] = S[j, i] = sol["sigma"]
            A[i, j, 0] = A[j, i, 1] = sol["A1"]
            A[i, j, 1] = A[j, i, 0] = sol["A2"]
            feas[i, j] = feas[j, i] = True
        if (i + 1) % 20 == 0:
            print(f"  row {i+1}/{K} ({time.time()-t0:.0f}s)", flush=True)
    fem = np.array([model.frontier_eps_min(float(e)) for e in actions])
    out = dict(actions=actions, sigma=S, A=A, feasible=feas,
               frontier_eps_min=fem,
               meta=dict(accounting="exact-ladder-tabulated", mesh=mesh,
                         T=T, G=G, B=B, delta=1e-12,
                         sigma_range=SIGMA_RANGE, n_nodes=(N_SIGMA, N_P)))
    with open(grid_path(mesh, T, K), "wb") as f:
        pickle.dump(out, f)
    print(f"grid done: feasible {feas.mean():.4f} -> {grid_path(mesh, T, K)}",
          flush=True)


def run_compare(T: int, K: int) -> None:
    lines = [f"# Offset-mesh comparison, exact-ladder tabulated grid, "
             f"T={T}, K={K}, G={G}", ""]

    def say(s=""):
        lines.append(s)
        print(s, flush=True)

    # --- anchor errors: interpolant vs the per-cell ladder solves of 180196
    models = {m: TabModel(m, T) for m in ("A", "B")}
    if ANCHORS.exists():
        with open(ANCHORS, "rb") as f:
            anch = pickle.load(f)
        rows = [r for col in anch["columns"].get(T, [])
                for r in col["rows"] if r.get("ok")]
        say(f"## Anchor check vs 52_t390_ladder.pkl ({len(rows)} rows)")
        for m, mod in (models.items() if rows else ()):
            ee = np.array([abs(mod.eps(r["sigma"], r["p2"]) - r["e2"])
                           / max(r["e2"], 1e-12) for r in rows])
            ea = np.array([abs(mod.adv(r["sigma"], r["p2"]) - r["A2"])
                           / max(r["A2"], 1e-12) for r in rows])
            say(f"mesh {m}: eps rel err median {np.median(ee):.2e} "
                f"p99 {np.quantile(ee, 0.99):.2e} max {ee.max():.2e}; "
                f"adv rel err median {np.median(ea):.2e} "
                f"p99 {np.quantile(ea, 0.99):.2e} max {ea.max():.2e}")
        say()

    # --- grid-level disagreement (in game coordinates, after inversion)
    ga, gb = (pickle.load(open(grid_path(m, T, K), "rb")) for m in ("A", "B"))
    fa, fb = ga["feasible"], gb["feasible"]
    both = fa & fb
    dis = fa ^ fb
    say("## Offset-mesh grid disagreement")
    say(f"feasible: mesh A {fa.mean():.4f}, mesh B {fb.mean():.4f}, "
        f"classification disagreement {dis.sum()} cells "
        f"({dis.mean():.2e} of grid)")
    ds = np.abs(ga["sigma"][both] - gb["sigma"][both]) / ga["sigma"][both]
    dA = np.abs(ga["A"][both] - gb["A"][both]).max(axis=-1)
    say(f"sigma rel diff: median {np.median(ds):.2e} "
        f"p99 {np.quantile(ds, 0.99):.2e} max {ds.max():.2e}")
    say(f"advantage abs diff: median {np.median(dA):.2e} "
        f"p99 {np.quantile(dA, 0.99):.2e} max {dA.max():.2e}")
    from game.continuous import sweep_from_grid, find_pure_NE_2player
    sa = sweep_from_grid(ga, 1.0, 1.0)
    sb = sweep_from_grid(gb, 1.0, 1.0)
    dU = np.abs(sa["U"] - sb["U"]).max(axis=-1)[both]
    say(f"utility (lam=1) abs diff: median {np.median(dU):.2e} "
        f"p99 {np.quantile(dU, 0.99):.2e} max {dU.max():.2e}")
    nea, neb = find_pure_NE_2player(sa), find_pure_NE_2player(sb)
    say(f"pure NE at lam=1: mesh A {len(nea)}, mesh B {len(neb)}")
    say()
    say("Verdict rule: interpolation is grid-proof where the offset-mesh "
        "utility disagreement sits well below the payoff scales it is "
        "asked to resolve (sawtooth ~1e-4, gap ~1e-3); cells failing "
        "that, and all classification-disagreement cells, need per-cell "
        "ladder solves before any equilibrium claim.")
    out = CACHE / f"tab_compare_T{T}_K{K}.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["nodes", "grid", "compare"])
    ap.add_argument("--mesh", choices=["A", "B"])
    ap.add_argument("--T", type=int, default=T_DEFAULT)
    ap.add_argument("--K", type=int, default=K_DEFAULT)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("PEG_N_WORKERS", "8")))
    a = ap.parse_args()
    if a.phase == "nodes":
        run_nodes(a.mesh, a.T, a.workers)
    elif a.phase == "grid":
        run_grid(a.mesh, a.T, a.K)
    else:
        run_compare(a.T, a.K)


if __name__ == "__main__":
    main()
