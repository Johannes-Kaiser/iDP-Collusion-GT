"""T26 -- anatomy of symmetry breaking in the symmetric 2-player PEG.

Verifies, on the cached K=200 mechanism grid (and the K=22 C1 cache for the
mixed-equilibrium computation), the game-theoretic mechanism behind the
no-symmetric-pure-NE result:

  (1) diagonal trough: at the diagonal first-order candidate, u_1(., eps*)
      has a strict local MINIMUM in the own action (deviation gains in both
      directions);
  (2) strategic substitutes: the discrete cross-difference of u_1 in
      (eps_1, eps_2) is negative near the diagonal, so best responses move
      AWAY from a deviating opponent (no recovery);
  (3) best-response dynamics started at every feasible diagonal profile
      leave the diagonal and terminate at off-diagonal pure NEs;
  (4) the 2x2 restriction of the game to the two budgets of a mirror NE
      pair is a Hawk-Dove (anti-coordination) game;
  (5) the symmetric game has a symmetric MIXED equilibrium (Nash 1951);
      we approximate it by one-population replicator dynamics on the K=22
      payoff matrix and report the lottery-over-roles it induces.

Outputs:
  tasks_findings/_scripts/26_symmetry_summary.txt
  paper/figures/fig_symmetry_breaking.{pdf,svg}
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game.style import (
    apply_paper_style, PRIMARY_BLUE, PRIMARY_ORANGE, ACCENT_GREY,
    column_size,
)

CACHE200 = ROOT / "game_experiments" / "_cache_v2" / "mech_grid_K200_g5000x5000_b128_e5.pkl"
CACHE22 = ROOT / "game_experiments" / "_cache_v2" / "C1.pkl"
FIG_PDF = ROOT / "paper" / "figures" / "fig_symmetry_breaking.pdf"
FIG_SVG = ROOT / "paper" / "figures" / "fig_symmetry_breaking.svg"
SUMMARY = ROOT / "tasks_findings" / "_scripts" / "26_symmetry_summary.txt"

W = 1.0  # canonical privacy weight


def V(sigma):
    return 1.0 / (1.0 + sigma ** 2)


def load_k200():
    d = pickle.load(open(CACHE200, "rb"))
    a = np.asarray(d["actions"])
    sigma = np.asarray(d["sigma"])
    A = np.asarray(d["A"])  # (K, K, 2); axis 0 = player 1's action
    feas = np.asarray(d["feasible"], dtype=bool)
    U1 = np.where(feas, V(sigma) - W * A[:, :, 0], -np.inf)
    U2 = np.where(feas, V(sigma) - W * A[:, :, 1], -np.inf)
    return a, U1, U2, A, feas


def pure_nes(U1, U2):
    K = U1.shape[0]
    br1 = np.argmax(U1, axis=0)          # br1[j]: best i against eps_2 = a[j]
    br2 = np.argmax(U2, axis=1)          # br2[i]: best j against eps_1 = a[i]
    nes = [(i, j) for j in range(K) for i in [br1[j]] if br2[i] == j
           and np.isfinite(U1[i, j])]
    return br1, br2, sorted(set(nes))


def main():
    out = []
    say = lambda s: (print(s), out.append(s))

    a, U1, U2, A, feas = load_k200()
    K = len(a)
    br1, br2, nes = pure_nes(U1, U2)
    n_diag = sum(1 for i, j in nes if i == j)
    say(f"# T26 symmetry-breaking anatomy (K={K}, w={W})")
    say(f"pure NEs: {len(nes)}, on diagonal: {n_diag}")

    # ---- (1) diagonal trough -------------------------------------------
    # Find diagonal cells where the discrete own-action derivative changes
    # sign (first-order candidates), then test for local min.
    diag_ok = np.array([feas[k, k] for k in range(K)])
    cand = []
    for k in range(1, K - 1):
        if not (diag_ok[k] and feas[k - 1, k] and feas[k + 1, k]):
            continue
        d_lo = U1[k, k] - U1[k - 1, k]
        d_hi = U1[k + 1, k] - U1[k, k]
        if d_lo <= 0 <= d_hi or (d_lo >= 0 >= d_hi):
            gain_up = np.nanmax(U1[k + 1:, k]) - U1[k, k]
            gain_dn = np.nanmax(U1[:k, k]) - U1[k, k]
            local_min = (d_lo < 0) and (d_hi > 0)
            cand.append((k, local_min, gain_up, gain_dn))
    say("\n# (1) diagonal first-order candidates (own-action sign change):")
    for k, lm, gu, gd in cand:
        say(f"  eps*={a[k]:7.3f}  local_min={lm}   "
            f"best deviation gain up={gu:+.4f}  down={gd:+.4f}")
    # global check: no diagonal cell is a NE; report min deviation gain
    gains = []
    for k in range(K):
        if diag_ok[k] and np.isfinite(U1[:, k]).any():
            gains.append(np.nanmax(U1[:, k]) - U1[k, k])
    gains = np.array(gains)
    say(f"  all feasible diagonal cells: min unilateral deviation gain "
        f"= {gains.min():.2e} (>0 everywhere: {bool((gains > 0).all())})")

    # ---- (2) strategic substitutes near the diagonal -------------------
    say("\n# (2) cross-difference sign near the diagonal "
        "(substitutes <=> negative):")
    for h in (4, 8, 16):
        signs = []
        for k in range(h, K - h):
            cells = [(k + h, k + h), (k + h, k - h), (k - h, k + h),
                     (k - h, k - h)]
            if not all(feas[i, j] for i, j in cells):
                continue
            dd = (U1[k + h, k + h] - U1[k + h, k - h]
                  - U1[k - h, k + h] + U1[k - h, k - h])
            signs.append(dd < 0)
        if signs:
            say(f"  stride h={h:2d}: {100 * np.mean(signs):5.1f}% negative "
                f"({len(signs)} diagonal cells)")

    # BR slope against opponent action (median discrete slope, feasible part)
    j_ok = np.array([np.isfinite(U1[:, j]).any() for j in range(K)])
    jj = np.where(j_ok)[0]
    slopes = np.diff(np.log(a[br1[jj]])) / np.diff(np.log(a[jj]))
    say(f"  BR_1 vs eps_2: median local slope = {np.median(slopes):+.2f} "
        f"({100 * np.mean(slopes < 0):.0f}% of steps negative)")

    # ---- (3) best-response dynamics from diagonal starts ---------------
    say("\n# (3) alternating best-response dynamics from diagonal starts:")
    end_off, end_diag, steps_all, ended_ne = 0, 0, [], 0
    sample_traj = None
    for k in range(K):
        if not diag_ok[k]:
            continue
        i, j = k, k
        traj = [(i, j)]
        for t in range(200):
            i2 = br1[j]
            j2 = br2[i2]
            if (i2, j2) == (i, j):
                break
            i, j = i2, j2
            traj.append((i, j))
        steps_all.append(len(traj) - 1)
        if i != j:
            end_off += 1
        else:
            end_diag += 1
        if (i, j) in nes:
            ended_ne += 1
        if sample_traj is None and k == K // 2:
            sample_traj = traj
    say(f"  starts: {end_off + end_diag}; end off-diagonal: {end_off}; "
        f"end on diagonal: {end_diag}")
    say(f"  ended at a pure NE: {ended_ne}; median steps: "
        f"{np.median(steps_all):.0f}")

    # ---- (4) 2x2 restriction is Hawk-Dove ------------------------------
    # take the mirror pair (1.91, 3.35) <-> indices via nearest actions
    lo = int(np.argmin(np.abs(a - 1.91)))
    hi = int(np.argmin(np.abs(a - 3.35)))
    say("\n# (4) 2x2 restriction to budgets "
        f"(T={a[lo]:.2f}, L={a[hi]:.2f})  [T=tight, L=loose]:")
    for (i, j), tag in [((lo, lo), "(T,T)"), ((lo, hi), "(T,L)"),
                        ((hi, lo), "(L,T)"), ((hi, hi), "(L,L)")]:
        say(f"  {tag}: u1={U1[i, j]:+.4f}  u2={U2[i, j]:+.4f}")
    tt_dev = U1[hi, lo] > U1[lo, lo]   # deviate T->L when opp plays T
    ll_dev = U1[lo, hi] > U1[hi, hi]   # deviate L->T when opp plays L
    tl_ne = (U1[lo, hi] >= U1[hi, hi]) and (U2[lo, hi] >= U2[lo, lo])
    lt_ne = (U1[hi, lo] >= U1[lo, lo]) and (U2[hi, lo] >= U2[hi, hi])
    say(f"  anti-coordination: (T,T) unstable={bool(tt_dev)}, "
        f"(L,L) unstable={bool(ll_dev)}, "
        f"(T,L) NE={bool(tl_ne)}, (L,T) NE={bool(lt_ne)}")

    # ---- (5) symmetric mixed equilibrium (K=22 game) --------------------
    c1 = pickle.load(open(CACHE22, "rb"))
    a22 = np.asarray(c1["actions"])
    U22 = np.asarray(c1["sweep"]["U"])
    A22 = np.asarray(c1["sweep"]["A"])
    f22 = np.asarray(c1["sweep"]["feasible"], dtype=bool)
    # Infeasible profiles (participation violated) are excluded from the
    # game; for the mixed extension we give them a strictly dominated
    # penalty payoff so no equilibrium puts weight on them.
    PEN = -2.0
    M = np.where(f22, U22[:, :, 0], PEN)
    K22 = len(a22)
    say("\n# (5) symmetric mixed equilibria, exact support enumeration "
        f"(K=22 full game, infeasible profiles = penalty {PEN}):")

    from itertools import combinations

    def symmetric_mixed_nes(M, max_support=5, tol=1e-9):
        found = []
        n_act = M.shape[0]
        for size in range(2, max_support + 1):
            for S in combinations(range(n_act), size):
                S = list(S)
                # solve (M x)_i = v on S, sum x = 1, x >= 0 on S
                A_ = np.zeros((size + 1, size + 1))
                b_ = np.zeros(size + 1)
                for r, i in enumerate(S):
                    A_[r, :size] = M[i, S]
                    A_[r, size] = -1.0
                A_[size, :size] = 1.0
                b_[size] = 1.0
                try:
                    sol = np.linalg.solve(A_, b_)
                except np.linalg.LinAlgError:
                    continue
                x_, v = sol[:size], sol[size]
                if (x_ < -tol).any():
                    continue
                x_ = np.clip(x_, 0, None)
                x_ /= x_.sum()
                pay_full = M[:, S] @ x_
                if pay_full.max() > v + 1e-7:
                    continue
                xfull = np.zeros(n_act)
                xfull[S] = x_
                found.append((S, xfull, v))
            if found:
                break  # report the smallest-support equilibria
        return found

    nes_mixed = symmetric_mixed_nes(M)
    if not nes_mixed:
        say("  none found up to support size 5")
    for S, x, v in nes_mixed:
        supp_str = ", ".join(f"{a22[i]:.2f}:{x[i]:.3f}" for i in S)
        p_same = float((x ** 2).sum())
        gap = 0.0
        ea = 0.0
        A1f = A22[:, :, 0]
        for ii in S:
            for jj in S:
                w_ = x[ii] * x[jj]
                gap += w_ * abs(A1f[ii, jj] - A1f[jj, ii])
                ea += w_ * A1f[ii, jj]
        say(f"  support {{{supp_str}}}  value={v:+.4f}")
        say(f"    P[same draw]={p_same:.3f} -> asymmetric realisation "
            f"w.p. {1 - p_same:.3f};  E|A_1-A_2|={gap:.3f};  E[A_i]={ea:.3f}")

    SUMMARY.write_text("\n".join(out) + "\n")
    print(f"\nwrote {SUMMARY}")

    # ---- figure ---------------------------------------------------------
    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=column_size(rows=1, aspect=0.42),
                             gridspec_kw=dict(wspace=0.42))

    # panel (a): own-utility slice at the diagonal candidate
    lm_cand = [(k, min(gu, gd)) for k, lm, gu, gd in cand if lm]
    k_star = max(lm_cand, key=lambda t: t[1])[0] if lm_cand else cand[0][0]
    ax = axes[0]
    mask = np.isfinite(U1[:, k_star])
    ax.plot(a[mask], U1[mask, k_star], "-", color=PRIMARY_BLUE, lw=1.2)
    ax.plot(a[k_star], U1[k_star, k_star], "o", color=PRIMARY_ORANGE,
            markersize=5, zorder=3)
    ax.axvline(a[k_star], color=ACCENT_GREY, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"own budget $\varepsilon_1$")
    ax.set_ylabel(r"$u_1$ against $\varepsilon_2 = \varepsilon^\ast$")
    ax.set_title("(a) diagonal candidate is a trough", fontsize=8)

    # panel (b): BR-dynamics trajectories from diagonal starts
    ax = axes[1]
    ax.plot([a[0], a[-1]], [a[0], a[-1]], "--", color=ACCENT_GREY, lw=0.6)
    starts = [k for k in range(0, K, 25) if diag_ok[k]]
    for s_idx, k in enumerate(starts):
        i, j = k, k
        traj = [(i, j)]
        for t in range(200):
            i2 = br1[j]; j2 = br2[i2]
            if (i2, j2) == (i, j):
                break
            i, j = i2, j2
            traj.append((i, j))
        xs = [a[i] for i, _ in traj]
        ys = [a[j] for _, j in traj]
        col = plt.cm.viridis(s_idx / max(1, len(starts) - 1))
        ax.plot(xs, ys, "-", color=col, lw=0.8, alpha=0.8)
        ax.plot(xs[0], ys[0], "o", color=col, markersize=3)
        ax.plot(xs[-1], ys[-1], "*", color=col, markersize=8,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title("(b) BR dynamics leave the diagonal", fontsize=8)

    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    print(f"wrote {FIG_PDF}")


if __name__ == "__main__":
    main()
