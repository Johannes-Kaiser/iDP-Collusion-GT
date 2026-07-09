"""
Experiment E2 -- adaptive agents discover the equilibrium.

Myopic agents that know only their own utility (no equilibrium solver) are run
on one fixed representative FL-consortium population, and we ask whether their
long-run play reaches the same outcomes (equilibrium, inequality) that the PEG
predicts. All dynamics run on the *level-2* (PLD-exact, mechanism-specific)
advantage via the shared oracle.

Modes
-----
  references : uniform sweep, PEG equilibrium (sequential best response from the
               grid midpoint), planner-approximate (greedy coordinate ascent on
               sum-utility, 5 starts). Constant baseline lines for the figures.
  learn      : the adaptive dynamics. --rule in {myopic, noisy, exp3}.
                 myopic (primary): activate one uniform-random agent/iteration,
                   she tries one adjacent grid value (reflect at ends), moves iff
                   own utility strictly improves by > MOVE_TOL.
                 noisy (appendix): same, with additive N(0, NOISE_SD) on each
                   utility observation (accept on the noisy comparison).
                 exp3 (appendix): per-agent EXP3 over the 6 arms, reward = own
                   utility rescaled to [0,1]; activated agent plays one arm-draw.
  figures    : build fig_flpop_learning and fig_flpop_trajectory.

Physics and constants are frozen in ``_flpop_e2_oracle`` (delta=1e-12, E_b=256,
epochs=10, grid eps in {10,14,19,26,37,50}, n=20). Dynamics use PLD
discretisation 1e-2 (|dA|~1e-2 vs 1e-3, plan D/failure-modes); final equilibria
are re-verified at 1e-3.

Parallelism: PEG_N_WORKERS worker processes evaluate unique profiles; the
learning loop runs the 50 seeds in lock-step so each iteration's candidate
profiles form one parallel batch over a shared cache.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The private oracle pulls its physics constants (grid, E_b rule, epochs,
# delta, sampler) from game_experiments.flpop_common when that module exists,
# so E1 and E2 share one config source (see _flpop_e2_oracle.CONFIG_SOURCE).
from game_experiments._flpop_e2_oracle import (
    make_oracle, sample_population, config_tag, ACTIONS, CONFIG_SOURCE,
)
_ORACLE_SRC = f"_flpop_e2_oracle(config={CONFIG_SOURCE})"

CACHE_DIR = ROOT / "game_experiments" / "_cache_flpop_e2"
FIG_DIR = ROOT / "paper" / "figures"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))

# Dynamics hyper-parameters (frozen per plan C).
POP_ID = int(os.environ.get("FLPOP_E2_POP", "0"))
POP_N = os.environ.get("FLPOP_E2_N")          # None -> sampler default n=20
N_SEEDS = 50
T_ITERS = 500
LOG_EVERY = 5
REGRET_EVERY = 25            # regret needs 20*6 oracle calls / seed / point
MOVE_TOL = 1e-6
NOISE_SD = 0.05
DISC_DYN = 1e-2
DISC_VERIFY = 1e-3


# ==========================================================================
# helpers
# ==========================================================================

def gini(x: Sequence[float]) -> float:
    a = np.sort(np.asarray(x, dtype=float))
    n = len(a)
    if n == 0 or a.sum() <= 0:
        return 0.0
    return float((2.0 * np.sum(np.arange(1, n + 1) * a) / (n * a.sum())) - (n + 1) / n)


def _idx_to_eps(idx: np.ndarray) -> List[float]:
    return [float(ACTIONS[int(i)]) for i in idx]


# ==========================================================================
# reference profiles
# ==========================================================================

def compute_references(orc, pool) -> Dict:
    n = orc.pop.n
    K = len(ACTIONS)

    # uniform sweep -> utilitarian-best uniform
    uni_profiles = [[a] * n for a in ACTIONS]
    uni_evs = orc.evaluate_many(uni_profiles, pool=pool)
    uni_rows = []
    for a, ev in zip(ACTIONS, uni_evs):
        su = float(ev["u"].sum()) if ev["feasible"] else float("-inf")
        uni_rows.append(dict(eps=float(a), sum_u=su, V=ev["V"], sigma=ev["sigma"],
                             meanA=float(ev["A"].mean()), gini=gini(ev["A"]),
                             feasible=ev["feasible"]))
    best_uni = max(uni_rows, key=lambda r: r["sum_u"])
    bu_ev = uni_evs[[r["eps"] for r in uni_rows].index(best_uni["eps"])]

    # PEG equilibrium: sequential best response
    def seq_br(init_idx: int, label: str):
        idx = np.full(n, init_idx, dtype=int)
        for rnd in range(200):
            changed = False
            for i in range(n):
                # evaluate all K own-actions for player i vs current others
                cand_profiles, cand_idx = [], []
                for a_idx in range(K):
                    t = idx.copy(); t[i] = a_idx
                    cand_profiles.append(_idx_to_eps(t)); cand_idx.append(a_idx)
                evs = orc.evaluate_many(cand_profiles, pool=pool)
                best_a = int(idx[i])
                best_u = (evs[best_a]["u"][i] if evs[best_a]["feasible"]
                          else float("-inf"))
                for a_idx, ev in zip(cand_idx, evs):
                    if not ev["feasible"]:
                        continue
                    if float(ev["u"][i]) > best_u + 1e-9:
                        best_u = float(ev["u"][i]); best_a = a_idx
                if best_a != int(idx[i]):
                    idx[i] = best_a; changed = True
            print(f"[refs] BR({label}) round {rnd+1} changed={changed} "
                  f"profile={_idx_to_eps(idx)}", flush=True)
            if not changed:
                return idx, True
        return idx, False

    mid_idx = K // 2 - 1 if K == 6 else K // 2   # third-of-six: eps=19
    eq_idx, converged = seq_br(mid_idx, "mid")
    eq_ev = orc.evaluate_many([_idx_to_eps(eq_idx)], pool=pool)[0]
    eq_top_idx, converged_top = seq_br(K - 1, "top")
    eq_top_ev = orc.evaluate_many([_idx_to_eps(eq_top_idx)], pool=pool)[0]

    # planner-approximate: greedy coordinate ascent on sum-u, 5 starts
    rng = np.random.default_rng(0)
    starts = [np.full(n, k, dtype=int) for k in range(1, 5)]  # uniform idx 1..4
    starts.append(rng.integers(0, K, size=n))                  # one random
    best_planner_idx, best_planner_val = None, float("-inf")
    for s0 in starts:
        pidx = s0.copy()
        for _ in range(200):
            changed = False
            for i in rng.permutation(n):
                cand_profiles = []
                for a_idx in range(K):
                    t = pidx.copy(); t[i] = a_idx
                    cand_profiles.append(_idx_to_eps(t))
                evs = orc.evaluate_many(cand_profiles, pool=pool)
                sums = [float(ev["u"].sum()) if ev["feasible"] else float("-inf")
                        for ev in evs]
                bi = int(np.argmax(sums))
                if bi != int(pidx[i]) and sums[bi] > sums[int(pidx[i])] + 1e-9:
                    pidx[i] = bi; changed = True
            if not changed:
                break
        val = float(orc.evaluate_many([_idx_to_eps(pidx)], pool=pool)[0]["u"].sum())
        print(f"[refs] planner start done: sum_u={val:.4f} "
              f"profile={_idx_to_eps(pidx)}", flush=True)
        if val > best_planner_val:
            best_planner_val = val; best_planner_idx = pidx.copy()
    pl_ev = orc.evaluate_many([_idx_to_eps(best_planner_idx)], pool=pool)[0]

    # exhaustive pure-NE enumeration on the K^n grid (n <= 5 only)
    pure_nes = None
    if K ** n <= 10_000:
        import itertools
        all_idx = list(itertools.product(range(K), repeat=n))
        all_evs = orc.evaluate_many(
            [_idx_to_eps(np.array(t)) for t in all_idx], pool=pool)
        umap = {t: ev for t, ev in zip(all_idx, all_evs)}
        pure_nes = []
        for t, ev in umap.items():
            if not ev["feasible"]:
                continue
            ok = True
            for i in range(n):
                ui = float(ev["u"][i])
                for a_idx in range(K):
                    if a_idx == t[i]:
                        continue
                    alt = t[:i] + (a_idx,) + t[i + 1:]
                    ev2 = umap[alt]
                    if ev2["feasible"] and float(ev2["u"][i]) > ui + 1e-9:
                        ok = False; break
                if not ok:
                    break
            if ok:
                pure_nes.append(dict(
                    eps=_idx_to_eps(np.array(t)), sigma=ev["sigma"], V=ev["V"],
                    sum_u=float(ev["u"].sum()), gini=gini(ev["A"]),
                    avg_eps=float(np.mean(_idx_to_eps(np.array(t)))),
                    meanA=float(ev["A"].mean())))
        print(f"[refs] brute force: {len(pure_nes)} pure NE(s) on the "
              f"{K}^{n} grid", flush=True)

    # critical-weight measurement at the all-bottom corner: for each player,
    # w*_i = dV / dA_i for the unilateral 10 -> 14 deviation. She profits from
    # deviating upward iff w_i < w*_i; corner-pooling at w ~ 1 means w* << 1.
    corner = np.zeros(n, dtype=int)
    corner_ev = orc.evaluate_many([_idx_to_eps(corner)], pool=pool)[0]
    dev_profiles = []
    for i in range(n):
        t = corner.copy(); t[i] = 1
        dev_profiles.append(_idx_to_eps(t))
    dev_evs = orc.evaluate_many(dev_profiles, pool=pool)
    w_crit = []
    for i, ev in enumerate(dev_evs):
        dV = float(ev["V"] - corner_ev["V"])
        dA = float(ev["A"][i] - corner_ev["A"][i])
        w_crit.append(dV / dA if dA > 0 else float("inf"))
    print(f"[refs] corner critical weights w* (dev 10->14): "
          f"median={np.median(w_crit):.4f} range=[{min(w_crit):.4f},"
          f"{max(w_crit):.4f}]", flush=True)

    def pack(idx_vec, ev, label):
        eps = _idx_to_eps(idx_vec)
        return dict(label=label, idx=idx_vec.tolist(), eps=eps,
                    sigma=ev["sigma"], V=ev["V"], avg_eps=float(np.mean(eps)),
                    var_eps=float(np.var(eps)), meanA=float(ev["A"].mean()),
                    mean_u=float(ev["u"].mean()), sum_u=float(ev["u"].sum()),
                    gini=gini(ev["A"]), feasible=ev["feasible"],
                    A=ev["A"].tolist(), u=ev["u"].tolist(),
                    p=ev["p"].tolist())

    refs = dict(
        pop_id=orc.pop.pop_id, n=n, M=orc.pop.M, steps=orc.pop.steps,
        group_sizes=orc.pop.group_sizes.tolist(),
        weights=orc.pop.weights.tolist(),
        actions=list(ACTIONS),
        uniform_sweep=uni_rows,
        uniform_best=pack(np.full(n, [r["eps"] for r in uni_rows].index(best_uni["eps"])), bu_ev, "uniform-best"),
        equilibrium=pack(eq_idx, eq_ev, "PEG-equilibrium"),
        equilibrium_converged=converged,
        equilibrium_top=pack(eq_top_idx, eq_top_ev, "PEG-equilibrium-topinit"),
        equilibrium_top_converged=converged_top,
        planner=pack(best_planner_idx, pl_ev, "planner-approx"),
        pure_nes=pure_nes,          # None unless K^n small enough to enumerate
        w_crit_corner=list(map(float, w_crit)),
    )
    return refs


# ==========================================================================
# learning dynamics (lock-step over seeds)
# ==========================================================================

def _reflect(idx: int, direction: int, K: int) -> int:
    j = idx + direction
    if j < 0 or j >= K:
        j = idx - direction
    return int(np.clip(j, 0, K - 1))


def run_learning(orc, pool, rule: str, n_seeds: int, T: int,
                 u_lo: float = None, u_hi: float = None,
                 exp3_gamma: float = 0.1) -> Dict:
    n = orc.pop.n
    K = len(ACTIONS)
    mid = K // 2 - 1 if K == 6 else K // 2   # third-of-six: eps=19
    rngs = [np.random.default_rng(1000 + s) for s in range(n_seeds)]

    # initial state: all agents at grid midpoint
    state = np.full((n_seeds, n), mid, dtype=int)

    # EXP3 per-agent arm weights
    if rule == "exp3":
        w = np.ones((n_seeds, n, K))

    log_iters, regret_iters = [], []
    # metric buffers: dict metric -> list over log points of arrays (n_seeds,)
    M = {k: [] for k in ["avg_eps", "var_eps", "V", "avg_u", "avg_A", "gini_A"]}
    regret_log = []   # list over regret points of arrays (n_seeds,)
    state_hist = []   # list over log points of state.copy() -> (n_seeds, n) idx

    def log_metrics(it):
        profs = [_idx_to_eps(state[s]) for s in range(n_seeds)]
        evs = orc.evaluate_many(profs, pool=pool)
        log_iters.append(it)
        M["avg_eps"].append(np.array([np.mean(_idx_to_eps(state[s])) for s in range(n_seeds)]))
        M["var_eps"].append(np.array([np.var(_idx_to_eps(state[s])) for s in range(n_seeds)]))
        M["V"].append(np.array([evs[s]["V"] for s in range(n_seeds)]))
        M["avg_u"].append(np.array([float(evs[s]["u"].mean()) for s in range(n_seeds)]))
        M["avg_A"].append(np.array([float(evs[s]["A"].mean()) for s in range(n_seeds)]))
        M["gini_A"].append(np.array([gini(evs[s]["A"]) for s in range(n_seeds)]))
        state_hist.append(state.copy())

    def log_regret(it):
        # for every seed, every agent, evaluate all 6 own-actions vs current others
        profs, meta = [], []
        for s in range(n_seeds):
            for i in range(n):
                for a_idx in range(K):
                    t = state[s].copy(); t[i] = a_idx
                    profs.append(_idx_to_eps(t)); meta.append((s, i, a_idx))
        evs = orc.evaluate_many(profs, pool=pool)
        # per (s,i): best own utility - current own utility
        by = {}
        for (s, i, a_idx), ev in zip(meta, evs):
            ui = float(ev["u"][i]) if ev["feasible"] else float("-inf")
            by.setdefault((s, i), {})[a_idx] = ui
        reg = np.zeros(n_seeds)
        for s in range(n_seeds):
            per_agent = []
            for i in range(n):
                d = by[(s, i)]
                cur = d[int(state[s, i])]
                per_agent.append(max(d.values()) - cur)
            reg[s] = float(np.mean(per_agent))
        regret_iters.append(it)
        regret_log.append(reg)

    log_metrics(0)
    log_regret(0)

    for it in range(1, T + 1):
        if rule in ("myopic", "noisy"):
            # each seed activates one agent, tries one adjacent value
            cand_state = state.copy()
            act_i = np.empty(n_seeds, dtype=int)
            cand_idx = np.empty(n_seeds, dtype=int)
            profs_cur, profs_cand = [], []
            for s in range(n_seeds):
                i = int(rngs[s].integers(n))
                direction = 1 if rngs[s].random() < 0.5 else -1
                j = _reflect(int(state[s, i]), direction, K)
                act_i[s] = i; cand_idx[s] = j
                profs_cur.append(_idx_to_eps(state[s]))
                t = state[s].copy(); t[i] = j
                profs_cand.append(_idx_to_eps(t))
            ev_cur = orc.evaluate_many(profs_cur, pool=pool)
            ev_cand = orc.evaluate_many(profs_cand, pool=pool)
            for s in range(n_seeds):
                i = int(act_i[s]); j = int(cand_idx[s])
                if j == int(state[s, i]):
                    continue
                u_cur = float(ev_cur[s]["u"][i]) if ev_cur[s]["feasible"] else float("-inf")
                u_cand = float(ev_cand[s]["u"][i]) if ev_cand[s]["feasible"] else float("-inf")
                if rule == "noisy":
                    u_cur = u_cur + rngs[s].normal(0, NOISE_SD)
                    u_cand = u_cand + rngs[s].normal(0, NOISE_SD)
                if u_cand > u_cur + MOVE_TOL:
                    state[s, i] = j
        elif rule == "exp3":
            # each seed activates one agent; she samples an arm from EXP3, plays
            profs, meta = [], []
            for s in range(n_seeds):
                i = int(rngs[s].integers(n))
                wi = w[s, i]
                probs = (1 - exp3_gamma) * wi / wi.sum() + exp3_gamma / K
                arm = int(rngs[s].choice(K, p=probs))
                t = state[s].copy(); t[i] = arm
                profs.append(_idx_to_eps(t)); meta.append((s, i, arm, probs))
            evs = orc.evaluate_many(profs, pool=pool)
            for (s, i, arm, probs), ev in zip(meta, evs):
                state[s, i] = arm
                u = float(ev["u"][i]) if ev["feasible"] else u_lo
                reward = np.clip((u - u_lo) / (u_hi - u_lo + 1e-12), 0.0, 1.0)
                est = reward / probs[arm]
                w[s, i, arm] *= np.exp(exp3_gamma * est / K)
                w[s, i] /= w[s, i].max()   # renormalise for numerical stability
        else:
            raise ValueError(rule)

        if it % LOG_EVERY == 0:
            log_metrics(it)
        if it % REGRET_EVERY == 0:
            log_regret(it)
        if it % 50 == 0:
            print(f"[learn:{rule}] iter {it}/{T} "
                  f"avg_u={M['avg_u'][-1].mean():.4f} "
                  f"gini_A={M['gini_A'][-1].mean():.4f} "
                  f"cache: eval={orc.n_eval} hit={orc.n_hit}", flush=True)
            orc.save()

    # stack metrics
    for k in M:
        M[k] = np.array(M[k])          # (n_logpoints, n_seeds)
    regret_arr = np.array(regret_log)  # (n_regret_points, n_seeds)

    return dict(
        rule=rule, n_seeds=n_seeds, T=T,
        log_iters=np.array(log_iters),
        regret_iters=np.array(regret_iters),
        metrics={k: v for k, v in M.items()},
        regret=regret_arr,
        final_state=state.copy(),
        state_hist=np.array(state_hist),     # (n_logpoints, n_seeds, n) idx
    )


# ==========================================================================
# end-state NE analysis
# ==========================================================================

def analyse_end_states(orc, pool, final_state, eq_gini: float) -> Dict:
    n_seeds, n = final_state.shape
    K = len(ACTIONS)
    # best response index for every (seed, agent) at the final profile
    profs, meta = [], []
    for s in range(n_seeds):
        for i in range(n):
            for a_idx in range(K):
                t = final_state[s].copy(); t[i] = a_idx
                profs.append(_idx_to_eps(t)); meta.append((s, i, a_idx))
    evs = orc.evaluate_many(profs, pool=pool)
    util = {}
    for (s, i, a_idx), ev in zip(meta, evs):
        ui = float(ev["u"][i]) if ev["feasible"] else float("-inf")
        util.setdefault((s, i), {})[a_idx] = ui

    is_ne = np.zeros(n_seeds, dtype=bool)
    is_approx_ne = np.zeros(n_seeds, dtype=bool)   # every player within 1 grid step of her BR
    frac_at_br = np.zeros(n_seeds)
    final_gini = np.zeros(n_seeds)
    for s in range(n_seeds):
        ok_ne = True; ok_approx = True; at_br = 0
        for i in range(n):
            d = util[(s, i)]
            cur = int(final_state[s, i])
            br = max(d, key=lambda a: d[a])
            if d[br] > d[cur] + MOVE_TOL:
                ok_ne = False
            if abs(br - cur) > 1:
                ok_approx = False
            if br == cur:
                at_br += 1
        is_ne[s] = ok_ne
        is_approx_ne[s] = ok_approx
        frac_at_br[s] = at_br / n
        fev = orc.evaluate_many([_idx_to_eps(final_state[s])], pool=pool)[0]
        final_gini[s] = gini(fev["A"])

    return dict(
        frac_pure_ne=float(is_ne.mean()),
        frac_approx_ne=float(is_approx_ne.mean()),
        mean_frac_at_br=float(frac_at_br.mean()),
        final_gini_mean=float(final_gini.mean()),
        final_gini_sd=float(final_gini.std()),
        eq_gini=float(eq_gini),
        gini_gap_to_eq=float(final_gini.mean() - eq_gini),
        final_gini=final_gini.tolist(),
        is_ne=is_ne.tolist(),
    )


# ==========================================================================
# entry point
# ==========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["references", "learn", "figures", "all"])
    ap.add_argument("--rule", default="myopic",
                    choices=["myopic", "noisy", "exp3"])
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--T", type=int, default=T_ITERS)
    ap.add_argument("--disc", type=float, default=DISC_DYN)
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--n", type=int, default=(int(POP_N) if POP_N else None),
                    help="population size (default: sampler default, 20)")
    ap.add_argument("--w-mu", type=float, default=0.0,
                    help="log-mean of the weight distribution "
                         "(track A critical-scale: -3.51)")
    args = ap.parse_args()

    pop = sample_population(POP_ID, n=args.n, w_mu=args.w_mu)
    # run tag: identifies (population id, n, weight regime, physics config).
    tag = f"pop{POP_ID}_n{pop.n}_wmu{args.w_mu:g}_{config_tag(pop.M)}"
    print(f"[E2] oracle source: {_ORACLE_SRC}  tag={tag} "
          f"(M={pop.M} E_b={pop.expected_batch} I={pop.steps} "
          f"w_median={float(np.median(pop.weights)):.4f}) "
          f"workers={args.workers} disc={args.disc}", flush=True)

    # oracle cache is weight-free: shared across w_mu (no wmu in the name).
    cache_path = (CACHE_DIR /
                  f"oracle_pop{POP_ID}_n{pop.n}_{config_tag(pop.M)}_disc{args.disc:g}.pkl")
    orc = make_oracle(POP_ID, cache_path=cache_path,
                      discretization=args.disc, n=args.n, w_mu=args.w_mu)
    pool = orc.make_pool(args.workers) if args.workers > 1 else None
    t0 = time.time()

    try:
        if args.mode in ("references", "all"):
            print("[E2] computing reference profiles ...", flush=True)
            refs = compute_references(orc, pool)
            with open(CACHE_DIR / f"references_{tag}.pkl", "wb") as f:
                pickle.dump(refs, f)
            orc.save()
            print(f"[E2] references done in {time.time()-t0:.0f}s "
                  f"(new evals={orc.n_eval})", flush=True)
            print(f"   equilibrium eps={refs['equilibrium']['eps']}")
            print(f"   equilibrium gini={refs['equilibrium']['gini']:.4f} "
                  f"V={refs['equilibrium']['V']:.4f} avg_eps={refs['equilibrium']['avg_eps']:.2f} "
                  f"converged={refs['equilibrium_converged']}")
            print(f"   uniform-best eps={refs['uniform_best']['eps'][0]} "
                  f"gini={refs['uniform_best']['gini']:.4f} V={refs['uniform_best']['V']:.4f}")
            print(f"   planner eps={refs['planner']['eps']}")
            print(f"   planner gini={refs['planner']['gini']:.4f} "
                  f"V={refs['planner']['V']:.4f} avg_eps={refs['planner']['avg_eps']:.2f}")

        if args.mode in ("learn", "all"):
            # need u_lo/u_hi for exp3 reward scaling from references
            refs_path = CACHE_DIR / f"references_{tag}.pkl"
            u_lo = u_hi = None
            if refs_path.exists():
                with open(refs_path, "rb") as f:
                    refs = pickle.load(f)
                all_u = refs["equilibrium"]["u"] + refs["planner"]["u"] + refs["uniform_best"]["u"]
                u_lo, u_hi = float(min(all_u)), float(max(all_u))
            rules = [args.rule]
            for rule in rules:
                print(f"[E2] learning rule={rule} seeds={args.n_seeds} T={args.T}",
                      flush=True)
                res = run_learning(orc, pool, rule, args.n_seeds, args.T,
                                   u_lo=u_lo, u_hi=u_hi)
                eqg = refs["equilibrium"]["gini"] if refs_path.exists() else float("nan")
                res["end_state"] = analyse_end_states(orc, pool, res["final_state"], eqg)
                with open(CACHE_DIR / f"learn_{rule}_{tag}.pkl", "wb") as f:
                    pickle.dump(res, f)
                orc.save()
                es = res["end_state"]
                print(f"[E2] rule={rule} done in {time.time()-t0:.0f}s "
                      f"(new evals total={orc.n_eval})", flush=True)
                print(f"   frac pure NE={es['frac_pure_ne']:.2f} "
                      f"frac approx-NE(<=1 step)={es['frac_approx_ne']:.2f} "
                      f"mean frac-at-BR={es['mean_frac_at_br']:.2f}")
                print(f"   final gini={es['final_gini_mean']:.4f}+/-{es['final_gini_sd']:.4f} "
                      f"eq gini={es['eq_gini']:.4f} gap={es['gini_gap_to_eq']:+.4f}")
                print(f"   final mean regret={res['regret'][-1].mean():.5f} "
                      f"init mean regret={res['regret'][0].mean():.5f}")

        if args.mode == "figures":
            from game_experiments import _flpop_e2_figs  # noqa
            _flpop_e2_figs.build_all(tag)
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
