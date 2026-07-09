"""
Experiment E1: best-response dynamics on realistic FL-consortium populations.

One array task = one population. For a population we run:
  * sequential best-response dynamics from two initialisations (grid midpoint
    and grid top), with cycle detection (Phase B1);
  * the uniform-best baseline and an approximate planner optimum (Phase B2);
  * per-participant and global level-2 measurements (Phase B3),

and write a single JSON result to ``_cache_flpop/E1_pop{pid}.json`` (per-task
file, never a shared writable file). Aggregation across populations (Phase B4)
and the boundary/feasibility audits (Phase D2/D3) run in ``--aggregate`` mode.

All quantities are level-2 (PLD-exact analytic advantage). Runs on CPU under
Slurm; the 6 best-response / planner candidate solves per step are evaluated in
parallel across ``PEG_N_WORKERS`` workers.

Usage:
    python -m game_experiments.run_flpop_e1 --pop 0 [--grid default] [--disc 0.01]
    python -m game_experiments.run_flpop_e1 --aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from game_experiments.flpop_common import (
    CACHE_DIR, ROOT, DISC_FAST, DISC_EXACT, ACTION_GRID, N_PLAYERS,
    sample_population, Population, gini, value_fn, _round_eps, _worker_solve,
)
from game_experiments.flpop_a4_scan import CANDIDATE_GRIDS

RESULT_DIR = CACHE_DIR


def variant_suffix(w_mu: float) -> str:
    """File-name suffix distinguishing weight-scale variants (empty for the
    plan default w_mu = 0)."""
    return "" if abs(w_mu) < 1e-12 else f"_wmu{w_mu:+.2f}"


# --------------------------------------------------------------------------
# Parallel, multiset-keyed profile cache.
# --------------------------------------------------------------------------
class ParCache:
    """Caches level-2 solves keyed by the multiset of (round(eps), |G_i|) pairs.
    ``fill`` computes all uncached profiles in a batch, in parallel."""

    def __init__(self, pop: Population, disc: float, workers: int,
                 path: Path | None, persist: bool = True):
        self.pop = pop
        self.gs = [int(g) for g in pop.group_sizes]
        self.steps = pop.steps
        self.disc = float(disc)
        self.workers = max(1, int(workers))
        self.path = path
        self.persist = persist and path is not None
        self.store: Dict[Tuple, Dict] = {}
        self.n_solves = 0
        if self.persist and path.exists():
            try:
                with open(path, "rb") as f:
                    self.store = pickle.load(f)
            except (EOFError, pickle.UnpicklingError):
                self.store = {}
        self._pool = None

    def _pool_get(self):
        if self._pool is None and self.workers > 1:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        self.save()

    def save(self):
        if not self.persist:
            return
        tmp = self.path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.store, f)
        tmp.replace(self.path)

    def _key(self, eps_vec: Sequence[float]) -> Tuple:
        return tuple(sorted(
            (_round_eps(e), g) for e, g in zip(eps_vec, self.gs)
        ))

    def _record(self, eps_vec, res) -> None:
        pa: Dict = {}
        for i in range(self.pop.n):
            pa[(_round_eps(eps_vec[i]), self.gs[i])] = (
                float(res["rates"][i]), float(res["advs"][i]))
        self.store[self._key(eps_vec)] = dict(
            sigma=res["sigma"], feasible=res["feasible"], pa=pa)

    def fill(self, profiles: List[Sequence[float]]) -> None:
        # Deduplicate uncached multisets, keep one representative eps_vec each.
        need: Dict[Tuple, Sequence[float]] = {}
        for prof in profiles:
            k = self._key(prof)
            if k not in self.store and k not in need:
                need[k] = list(prof)
        if not need:
            return
        tasks = [(tuple(float(e) for e in prof), tuple(self.gs),
                  self.steps, self.disc, self.pop.e_b) for prof in need.values()]
        self.n_solves += len(tasks)
        pool = self._pool_get()
        if pool is not None and len(tasks) > 1:
            for eps_t, res in pool.map(_worker_solve, tasks):
                self._record(eps_t, res)
        else:
            for eps_t, res in map(_worker_solve, tasks):
                self._record(eps_t, res)

    def get(self, eps_vec: Sequence[float]):
        """Return (sigma, p_vec, A_vec, feasible) for one profile."""
        k = self._key(eps_vec)
        if k not in self.store:
            self.fill([eps_vec])
        rec = self.store[k]
        n = self.pop.n
        p = np.empty(n); A = np.empty(n)
        for i in range(n):
            p[i], A[i] = rec["pa"][(_round_eps(eps_vec[i]), self.gs[i])]
        return rec["sigma"], p, A, rec["feasible"]

    def utilities(self, eps_vec: Sequence[float]) -> Tuple[np.ndarray, float, np.ndarray, bool]:
        sigma, p, A, feas = self.get(eps_vec)
        u = value_fn(sigma) - self.pop.weights * A
        return u, sigma, A, feas


# --------------------------------------------------------------------------
# Best-response dynamics (B1).
# --------------------------------------------------------------------------
def best_response_dynamics(cache: ParCache, actions: Sequence[float],
                           start: Sequence[float], max_rounds: int = 200) -> Dict:
    n = cache.pop.n
    w = cache.pop.weights
    current = [float(x) for x in start]
    history = [tuple(current)]
    seen = {tuple(current): 0}
    for r in range(max_rounds):
        changed = False
        for i in range(n):
            cands = []
            for a in actions:
                t = list(current); t[i] = float(a); cands.append(t)
            cache.fill(cands)
            cur_val = current[i]
            best_a, best_u = cur_val, -np.inf
            for a in actions:
                t = list(current); t[i] = float(a)
                sigma, p, A, feas = cache.get(t)
                if not feas:
                    continue
                ui = value_fn(sigma) - w[i] * A[i]
                if ui > best_u + 1e-12 or (abs(ui - best_u) <= 1e-12 and a == cur_val):
                    best_u, best_a = ui, float(a)
            if best_a != current[i]:
                current[i] = best_a; changed = True
        key = tuple(current)
        history.append(key)
        if not changed:
            return dict(profile=current, rounds=r + 1, converged=True,
                        cycling=False, cycle=None)
        if key in seen:
            cycle = history[seen[key]:len(history) - 1]  # distinct cycle profiles
            return dict(profile=current, rounds=r + 1, converged=False,
                        cycling=True, cycle=[list(c) for c in cycle])
        seen[key] = len(history) - 1
    return dict(profile=current, rounds=max_rounds, converged=False,
                cycling=False, cycle=None)


# --------------------------------------------------------------------------
# Baselines (B2).
# --------------------------------------------------------------------------
def uniform_best(cache: ParCache, actions: Sequence[float]) -> Tuple[List[float], float]:
    n = cache.pop.n
    profiles = [[float(a)] * n for a in actions]
    cache.fill(profiles)
    best_prof, best_sum = None, -np.inf
    for a in actions:
        prof = [float(a)] * n
        u, sigma, A, feas = cache.utilities(prof)
        if not feas:
            continue
        s = float(u.sum())
        if s > best_sum:
            best_sum, best_prof = s, prof
    return best_prof, best_sum


def planner_optimum(cache: ParCache, actions: Sequence[float],
                    starts: List[List[float]], max_sweeps: int = 40) -> List[float]:
    n = cache.pop.n
    w = cache.pop.weights
    best_prof, best_sum = None, -np.inf
    for start in starts:
        current = [float(x) for x in start]
        for _ in range(max_sweeps):
            changed = False
            for i in range(n):
                cands = []
                for a in actions:
                    t = list(current); t[i] = float(a); cands.append(t)
                cache.fill(cands)
                best_a, best_val = current[i], -np.inf
                for a in actions:
                    t = list(current); t[i] = float(a)
                    sigma, p, A, feas = cache.get(t)
                    if not feas:
                        continue
                    su = n * value_fn(sigma) - float(np.dot(w, A))
                    if su > best_val + 1e-12:
                        best_val, best_a = su, float(a)
                if best_a != current[i]:
                    current[i] = best_a; changed = True
            if not changed:
                break
        u, sigma, A, feas = cache.utilities(current)
        if feas and float(u.sum()) > best_sum:
            best_sum, best_prof = float(u.sum()), current
    return best_prof


# --------------------------------------------------------------------------
# Measurements (B3).
# --------------------------------------------------------------------------
def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    if len(set(np.round(y, 9))) < 2:
        return 0.0
    r = spearmanr(x, y).correlation
    return float(r) if np.isfinite(r) else 0.0


def kendall(x, y) -> float:
    from scipy.stats import kendalltau
    r = kendalltau(x, y).correlation
    return float(r) if np.isfinite(r) else 0.0


def _profile_measure(cache: ParCache, profile: List[float]) -> Dict:
    pop = cache.pop
    u, sigma, A, feas = cache.utilities(profile)
    _, p, _, _ = cache.get(profile)
    return dict(
        eps=[float(e) for e in profile], sigma=float(sigma), V=float(value_fn(sigma)),
        A=[float(a) for a in A], u=[float(x) for x in u], p=[float(x) for x in p],
        gini_A=gini(A), feasible=bool(feas),
    )


def measure_population(pop: Population, grid_name: str, actions: Sequence[float],
                       disc: float, workers: int, max_rounds: int,
                       w_mu: float = 0.0) -> Dict:
    # NOTE: the parcache stores only (sigma, p, A) per multiset -- weight-free
    # quantities -- so it is shared across weight-scale (w_mu) variants.
    # Non-default mechanism scales (E_b, epochs) get their own namespace.
    from game_experiments.flpop_common import EXPECTED_BATCH, EPOCHS
    mech = ("" if (pop.e_b == EXPECTED_BATCH and pop.epochs == EPOCHS)
            else f"_{pop.mech_tag}")
    path = RESULT_DIR / f"parcache_pop{pop.pid:03d}_{grid_name}{mech}.pkl"
    cache = ParCache(pop, disc=disc, workers=workers, path=path)
    n = pop.n
    a = list(actions)
    mid = float(a[len(a) // 2 - 1])   # third of six
    top = float(a[-1])

    # Best-response dynamics from two inits.
    br_mid = best_response_dynamics(cache, actions, [mid] * n, max_rounds)
    br_top = best_response_dynamics(cache, actions, [top] * n, max_rounds)

    # Baselines.
    uni_prof, _ = uniform_best(cache, actions)
    rng = np.random.default_rng(9000 + pop.pid)
    planner_starts = [[float(v)] * n for v in a[1:]] + [list(rng.choice(a, size=n))]
    plan_prof = planner_optimum(cache, actions, planner_starts)

    def eq_summary(br: Dict) -> Dict:
        if br["cycling"] and br["cycle"]:
            # cycle-average each per-player measurement over the cycle profiles
            profs = br["cycle"]
            eps_arr = np.array(profs, dtype=float)              # (T, n)
            A_stack, u_stack, p_stack, sig_list, gini_list = [], [], [], [], []
            for pr in profs:
                m = _profile_measure(cache, pr)
                A_stack.append(m["A"]); u_stack.append(m["u"])
                p_stack.append(m["p"]); sig_list.append(m["sigma"])
                gini_list.append(m["gini_A"])
            eps_star = eps_arr.mean(axis=0)
            A_mean = np.array(A_stack).mean(axis=0)
            u_mean = np.array(u_stack).mean(axis=0)
            p_mean = np.array(p_stack).mean(axis=0)
            return dict(
                eps=[float(x) for x in eps_star], A=[float(x) for x in A_mean],
                u=[float(x) for x in u_mean], p=[float(x) for x in p_mean],
                sigma=float(np.mean(sig_list)), V=float(np.mean([value_fn(s) for s in sig_list])),
                gini_A=float(np.mean(gini_list)),
            )
        else:
            m = _profile_measure(cache, br["profile"])
            return dict(eps=m["eps"], A=m["A"], u=m["u"], p=m["p"],
                        sigma=m["sigma"], V=m["V"], gini_A=m["gini_A"])

    eq_mid = eq_summary(br_mid)
    eq_top = eq_summary(br_top)
    uni_m = _profile_measure(cache, uni_prof)
    plan_m = _profile_measure(cache, plan_prof)

    w = pop.weights.tolist()
    gs = pop.group_sizes.tolist()
    M = pop.M

    def sorting(eq) -> Dict:
        return dict(
            rho_w_eps=spearman(w, eq["eps"]),
            tau_w_eps=kendall(w, eq["eps"]),
            rho_eps_A=spearman(eq["eps"], eq["A"]),
        )

    cache.close()
    n_boundary_lo = int(sum(1 for e in eq_mid["eps"] if e <= a[0] + 1e-9))
    n_boundary_hi = int(sum(1 for e in eq_mid["eps"] if e >= a[-1] - 1e-9))

    return dict(
        pid=pop.pid, n=n, grid=grid_name, actions=a, disc=disc, M=M,
        w_mu=float(w_mu),
        w=w, group_sizes=gs, contrib=[g / M for g in gs],
        # primary (midpoint-init) equilibrium
        eq=eq_mid, eq_top=eq_top, uniform=uni_m, planner=plan_m,
        uniform_profile=uni_prof, planner_profile=plan_prof,
        sorting_mid=sorting(eq_mid), sorting_top=sorting(eq_top),
        gini_eq=eq_mid["gini_A"], gini_uniform=uni_m["gini_A"], gini_planner=plan_m["gini_A"],
        br_mid=dict(rounds=br_mid["rounds"], converged=br_mid["converged"],
                    cycling=br_mid["cycling"],
                    cycle_len=(len(br_mid["cycle"]) if br_mid["cycle"] else 0)),
        br_top=dict(rounds=br_top["rounds"], converged=br_top["converged"],
                    cycling=br_top["cycling"],
                    cycle_len=(len(br_top["cycle"]) if br_top["cycle"] else 0)),
        n_boundary_lo=n_boundary_lo, n_boundary_hi=n_boundary_hi,
        n_solves=cache.n_solves,
    )


# --------------------------------------------------------------------------
# Aggregation (B4, D2, D3).
# --------------------------------------------------------------------------
def _load_rows(grid_name: str, w_mu: float) -> List[Dict]:
    rows = []
    for f in sorted(RESULT_DIR.glob("E1_pop*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("grid") == grid_name and abs(r.get("w_mu", 0.0) - w_mu) < 1e-9:
            rows.append(r)
    return rows


def aggregate(grid_name: str, w_mu: float = 0.0) -> None:
    from scipy.stats import wilcoxon
    rows = _load_rows(grid_name, w_mu)
    if not rows:
        print(f"[aggregate] no results for grid={grid_name} w_mu={w_mu}")
        return
    print(f"[aggregate] {len(rows)} populations, grid={grid_name} w_mu={w_mu}")

    rho_w = np.array([r["sorting_mid"]["rho_w_eps"] for r in rows])
    tau_w = np.array([r["sorting_mid"]["tau_w_eps"] for r in rows])
    rho_A = np.array([r["sorting_mid"]["rho_eps_A"] for r in rows])
    g_eq = np.array([r["gini_eq"] for r in rows])
    g_uni = np.array([r["gini_uniform"] for r in rows])
    g_plan = np.array([r["gini_planner"] for r in rows])

    conv = sum(r["br_mid"]["converged"] for r in rows)
    cyc = sum(r["br_mid"]["cycling"] for r in rows)

    # Boundary audit (D2)
    n_players_total = sum(r["n"] for r in rows)
    n_lo = sum(r["n_boundary_lo"] for r in rows)
    n_hi = sum(r["n_boundary_hi"] for r in rows)

    diff = g_eq - g_uni
    frac_higher = float(np.mean(diff > 0))
    try:
        wstat = wilcoxon(g_eq, g_uni)
        wp = float(wstat.pvalue)
    except ValueError:
        wp = float("nan")

    def q(x):
        return (float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75)))

    summary = dict(
        grid=grid_name, n_populations=len(rows),
        converged=conv, cycling=cyc,
        rho_w_eps=dict(median=q(rho_w)[0], iqr=[q(rho_w)[1], q(rho_w)[2]]),
        tau_w_eps=dict(median=q(tau_w)[0], iqr=[q(tau_w)[1], q(tau_w)[2]]),
        rho_eps_A=dict(median=q(rho_A)[0], iqr=[q(rho_A)[1], q(rho_A)[2]]),
        gini_eq=dict(median=q(g_eq)[0], iqr=[q(g_eq)[1], q(g_eq)[2]]),
        gini_uniform=dict(median=q(g_uni)[0], iqr=[q(g_uni)[1], q(g_uni)[2]]),
        gini_planner=dict(median=q(g_plan)[0], iqr=[q(g_plan)[1], q(g_plan)[2]]),
        gini_diff_median=float(np.median(diff)),
        gini_eq_higher_frac=frac_higher, wilcoxon_p=wp,
        boundary_frac=float((n_lo + n_hi) / n_players_total),
        boundary_lo_frac=float(n_lo / n_players_total),
        boundary_hi_frac=float(n_hi / n_players_total),
        median_solves=float(np.median([r["n_solves"] for r in rows])),
    )
    summary["w_mu"] = w_mu
    out = RESULT_DIR / f"E1_summary_{grid_name}{variant_suffix(w_mu)}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[aggregate] wrote {out}")


# --------------------------------------------------------------------------
# Main figure (B6): three level-2 panels.
# --------------------------------------------------------------------------
def make_figure(grid_name: str, w_mu: float = 0.0) -> None:
    import matplotlib.pyplot as plt
    from game.style import (apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE,
                            ACCENT_GREY, two_column_size)
    from game_experiments.flpop_common import lorenz_curve
    apply_paper_style()
    FIG_DIR = ROOT / "paper" / "figures"

    rows = _load_rows(grid_name, w_mu)
    if not rows:
        print(f"[figure] no results for grid={grid_name} w_mu={w_mu}")
        return

    w_all = np.concatenate([np.array(r["w"]) for r in rows])
    eps_all = np.concatenate([np.array(r["eq"]["eps"]) for r in rows])
    A_all = np.concatenate([np.array(r["eq"]["A"]) for r in rows])
    rho_w = np.median([r["sorting_mid"]["rho_w_eps"] for r in rows])
    rho_A = np.median([r["sorting_mid"]["rho_eps_A"] for r in rows])
    g_eq = np.array([r["gini_eq"] for r in rows])
    g_uni = np.array([r["gini_uniform"] for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=two_column_size(rows=1, aspect=0.34))

    # Panel A: w vs eps* (log-log), per-decile median overlay.
    ax = axes[0]
    ax.scatter(w_all, eps_all, s=6, alpha=0.15, color=PRIMARY_BLUE,
               edgecolors="none")
    order = np.argsort(w_all)
    ws, es = w_all[order], eps_all[order]
    deciles = np.array_split(np.arange(len(ws)), 10)
    dm_w = [np.median(ws[d]) for d in deciles]
    dm_e = [np.median(es[d]) for d in deciles]
    ax.plot(dm_w, dm_e, "-o", color=PRIMARY_ORANGE, markersize=3, lw=1.4,
            label="per-decile median")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"privacy weight $w_i$")
    ax.set_ylabel(r"equilibrium budget $\varepsilon_i^\ast$")
    ax.set_title("(a) Preferences sort budgets")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.04, 0.06, rf"median $\rho(w,\varepsilon^\ast)={rho_w:.2f}$",
            transform=ax.transAxes, fontsize=7)

    # Panel B: eps* vs realized A (log-log).
    ax = axes[1]
    ax.scatter(eps_all, A_all, s=6, alpha=0.15, color=PRIMARY_BLUE,
               edgecolors="none")
    order = np.argsort(eps_all)
    es2, As2 = eps_all[order], A_all[order]
    dec2 = np.array_split(np.arange(len(es2)), 10)
    ax.plot([np.median(es2[d]) for d in dec2], [np.median(As2[d]) for d in dec2],
            "-o", color=PRIMARY_ORANGE, markersize=3, lw=1.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"equilibrium budget $\varepsilon_i^\ast$")
    ax.set_ylabel(r"realized advantage $A_i$")
    ax.set_title("(b) Budgets set realized risk")
    ax.text(0.04, 0.9, rf"median $\rho(\varepsilon^\ast,A)={rho_A:.2f}$",
            transform=ax.transAxes, fontsize=7)

    # Panel C: Lorenz curves (averaged) + Gini inset.
    ax = axes[2]
    m = 200
    grid_f = np.linspace(0, 1, m + 1)
    eq_curves, uni_curves = [], []
    for r in rows:
        fe, ce = lorenz_curve(r["eq"]["A"])
        fu, cu = lorenz_curve(r["uniform"]["A"])
        eq_curves.append(np.interp(grid_f, fe, ce))
        uni_curves.append(np.interp(grid_f, fu, cu))
    eq_mean = np.mean(eq_curves, axis=0)
    uni_mean = np.mean(uni_curves, axis=0)
    ax.plot([0, 1], [0, 1], ":", color=ACCENT_GREY, lw=0.8)
    ax.plot(grid_f, uni_mean, "--", color=PRIMARY_BLUE, lw=1.4,
            label=f"uniform-best (Gini {np.median(g_uni):.2f})")
    ax.plot(grid_f, eq_mean, "-", color=PRIMARY_ORANGE, lw=1.6,
            label=f"equilibrium (Gini {np.median(g_eq):.2f})")
    ax.set_xlabel("population fraction (sorted by $A_i$)")
    ax.set_ylabel("cumulative share of realized advantage")
    ax.set_title("(c) Equilibrium concentrates risk")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    # Gini inset (paired box).
    axin = ax.inset_axes([0.62, 0.12, 0.34, 0.34])
    axin.boxplot([g_uni, g_eq], widths=0.5, showfliers=False,
                 medianprops=dict(color=ACCENT_GREY))
    axin.set_xticks([1, 2]); axin.set_xticklabels(["unif", "eq"], fontsize=6)
    axin.set_ylabel("Gini", fontsize=6)
    axin.tick_params(labelsize=6)

    fig.tight_layout()
    stem = FIG_DIR / "fig_flpop_equilibrium"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)
    print(f"[figure] wrote {stem.with_suffix('.pdf')} from {len(rows)} populations")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=None, help="single population id")
    ap.add_argument("--grid", default="default", choices=list(CANDIDATE_GRIDS))
    ap.add_argument("--disc", type=float, default=DISC_FAST)
    ap.add_argument("--max-rounds", type=int, default=200)
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("PEG_N_WORKERS", 6)))
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--w-mu", type=float, default=0.0,
                    help="location of log-weight distribution "
                         "(median weight = exp(w_mu); plan default 0)")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.grid, args.w_mu)
        return
    if args.plot:
        make_figure(args.grid, args.w_mu)
        return

    if args.pop is None:
        raise SystemExit("give --pop N or --aggregate")

    actions = CANDIDATE_GRIDS[args.grid]
    pop = sample_population(args.pop, w_mu=args.w_mu)
    import time
    t0 = time.time()
    res = measure_population(pop, args.grid, actions, args.disc,
                             args.workers, args.max_rounds, w_mu=args.w_mu)
    res["wall_seconds"] = time.time() - t0
    out = RESULT_DIR / f"E1_pop{args.pop:03d}{variant_suffix(args.w_mu)}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[E1 pop{args.pop}] grid={args.grid} solves={res['n_solves']} "
          f"BRmid rounds={res['br_mid']['rounds']} conv={res['br_mid']['converged']} "
          f"cyc={res['br_mid']['cycling']} "
          f"rho(w,eps)={res['sorting_mid']['rho_w_eps']:.3f} "
          f"rho(eps,A)={res['sorting_mid']['rho_eps_A']:.3f} "
          f"gini_eq={res['gini_eq']:.3f} gini_uni={res['gini_uniform']:.3f} "
          f"({res['wall_seconds']:.0f}s) -> {out.name}")


if __name__ == "__main__":
    main()
