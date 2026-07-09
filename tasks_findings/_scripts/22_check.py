"""
Task T22 -- Roughgarden (lambda, mu)-smoothness scan for the PEG.

Runs a symbolic + numerical analysis of Roughgarden's welfare-smoothness
inequality
    sum_i u_i(eps_i^opt, eps_{-i})  >=  lambda * SW(eps^opt)  -  mu * SW(eps),
specialised to the 2-player PEG at the K=200 mechanism grid.

Behaviour:

    1) For the C1 setting (symmetric, w1=w2=1) and every C7 privacy weight
       w in {0.125, 0.25, 0.5, 1, 2, 4}, load the K=200 mechanism grid,
       derive the K x K utility surface via game.continuous.sweep_from_grid,
       find the socially-optimal profile (SW-argmax on the feasible cells),
       and for every feasible cell compute the unilateral-deviation sum
       U1[i_opt, j] + U2[i, j_opt].

    2) Scan the (lambda, mu) rectangle on a 41x41 grid,
       lambda in [0.05, 2.00], mu in [0.00, 0.99], report the deficit
       surface deficit(lambda, mu) = max_s (lambda*SW_opt - mu*SW[s] - dev_sum[s])
       (deficit <= 0  =>  smoothness inequality holds everywhere).

    3) For each admissible (lambda, mu), evaluate the pure-NE PoA bound
       lambda / (1 - mu) and report the argmin over admissible pairs;
       verify against the T10 anchors:
           worst-NE / social-opt in raw social welfare
           worst-NE / social-opt in aggregate A (PoEV+1)

    4) Sympy check of the Roughgarden smoothness reduction for the PEG
       utility u_i = V(sigma) - w_i A_i with V(sigma)=1/(1+sigma^2):
       symbolically expand the smoothness LHS-RHS gap and record the
       V-side and A-side contributions, so the constants (lambda*, mu*)
       have an analytic interpretation.

Outputs (under tasks_findings/_scripts/_out/22/):

    * summary.json         -- (lambda*, mu*, PoA_bound) per w and C1 anchors
    * deficit_C1_w1.npz    -- raw deficit surface for C1 (w=1)
    * deficit_C7_w*.npz    -- raw deficit surface for each C7 w
    * verbatim_stdout.txt  -- captured stdout, sourced verbatim by the
                              findings note.

No `paper/` edits, no Slurm.  Runtime ~5 s login-node CPU.
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.continuous import sweep_from_grid, find_pure_NE_2player  # noqa: E402

CACHE = ROOT / "game_experiments" / "_cache_v2"
OUT = Path(__file__).resolve().parent / "_out" / "22"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Symbolic reduction (sympy)
# ---------------------------------------------------------------------------
def symbolic_reduction() -> str:
    """Symbolic expansion of the smoothness gap for u_i = V(sigma) - w_i A_i.

    Report the algebraic structure of
        Delta(s, s^opt; lambda, mu)
        := lambda * SW(s^opt) - mu * SW(s)
           - sum_i u_i(s^opt_i, s_{-i}).

    Under the shared-V structure, the term decomposes into a V-only part and
    an A-only part; we print both and identify which cross-partial signs
    (T20 / T23) each part inherits.
    """
    import sympy as sp

    out = io.StringIO()
    print("=" * 70, file=out)
    print("(0) Symbolic Roughgarden smoothness gap for the PEG", file=out)
    print("=" * 70, file=out)

    # Two players; the general n-player expansion is analogous.
    sigma_s, sigma_opt = sp.symbols("sigma_s sigma_opt", positive=True)
    sigma_d1, sigma_d2 = sp.symbols("sigma^{1d} sigma^{2d}", positive=True)
    A1_s, A2_s, A1_opt, A2_opt = sp.symbols("A^{s}_1 A^{s}_2 A^{opt}_1 A^{opt}_2",
                                            nonnegative=True)
    A1_d, A2_d = sp.symbols("A^{1d}_1 A^{2d}_2", nonnegative=True)
    w1, w2, lam, mu = sp.symbols("w_1 w_2 lambda mu", positive=True)

    V = sp.Function("V")

    SW_s = V(sigma_s) - w1 * A1_s + V(sigma_s) - w2 * A2_s
    SW_opt = V(sigma_opt) - w1 * A1_opt + V(sigma_opt) - w2 * A2_opt
    # Unilateral deviation of player 1 to opt_1 (profile: opt_1, s_2):
    #   sigma_{1d} := sigma(opt_1, s_2),  A_1 there is A_1^{1d}.
    #   (player 2's action is s_2, its advantage is A_2^{1d,unused}.)
    #   The smoothness LHS is u_1 at (opt_1, s_2) + u_2 at (s_1, opt_2).
    u1_dev = V(sigma_d1) - w1 * A1_d
    u2_dev = V(sigma_d2) - w2 * A2_d

    dev_sum = u1_dev + u2_dev
    gap = lam * SW_opt - mu * SW_s - dev_sum
    gap_expanded = sp.expand(gap)

    print("SW(s^opt)         =", SW_opt, file=out)
    print("SW(s)             =", SW_s, file=out)
    print("dev_sum LHS       =", dev_sum, file=out)
    print("gap (RHS - LHS)   =", gap_expanded, file=out)
    print(file=out)

    # V-only contribution:
    V_part = (lam * (V(sigma_opt) + V(sigma_opt))
              - mu * (V(sigma_s) + V(sigma_s))
              - V(sigma_d1) - V(sigma_d2))
    A_part = (- lam * (w1 * A1_opt + w2 * A2_opt)
              + mu * (w1 * A1_s + w2 * A2_s)
              + w1 * A1_d + w2 * A2_d)
    print("V-part (V-side of the gap):", file=out)
    print("  ", sp.expand(V_part), file=out)
    print("A-part (A-side of the gap):", file=out)
    print("  ", sp.expand(A_part), file=out)
    print("(V-part + A-part - gap) =", sp.simplify(V_part + A_part - gap_expanded),
          "  (must be 0)", file=out)
    print(file=out)
    print("Interpretation.", file=out)
    print(
        "  The V-part is negative iff\n"
        "     V(sigma_d1) + V(sigma_d2)\n"
        "       >= 2*lambda*V(sigma_opt) - 2*mu*V(sigma_s).\n"
        "  Because V is monotone-decreasing and sigma_d_k <= sigma_s at every\n"
        "  unilateral deviation to opt (increasing one player's eps drops sigma\n"
        "  by lem:externality), V(sigma_d_k) >= V(sigma_s), so the V-part\n"
        "  admits the (crude) sufficient condition\n"
        "     2*V(sigma_s) >= 2*lambda*V(sigma_opt) - 2*mu*V(sigma_s)\n"
        "     <=> V(sigma_s) * (1 + mu) >= lambda*V(sigma_opt).\n"
        "  This is trivially satisfied when V(s) = V(opt) and mu >= lambda - 1.\n",
        file=out,
    )
    print(
        "  The A-part is negative iff\n"
        "     w_1 * A^{1d}_1 + w_2 * A^{2d}_2\n"
        "       <= lambda*(w_1 A^{opt}_1 + w_2 A^{opt}_2)\n"
        "          - mu*(w_1 A^{s}_1 + w_2 A^{s}_2).\n"
        "  This is the hard direction: A^{1d}_1 = A_1(opt_1, s_2), and\n"
        "  when s_2 is small (adversarial coalition), the externality on\n"
        "  player 1 makes A^{1d}_1 LARGE even though player 1 is playing opt_1.\n"
        "  See Section 3 of the findings note for the analytic bound.\n",
        file=out,
    )

    return out.getvalue()


# ---------------------------------------------------------------------------
# Numerical smoothness scan
# ---------------------------------------------------------------------------
def V_of_sigma(sigma_arr):
    return 1.0 / (1.0 + sigma_arr * sigma_arr)


def smoothness_scan(mech_grid: dict, w1: float, w2: float,
                    lam_grid: np.ndarray, mu_grid: np.ndarray) -> dict:
    """Return the deficit surface, admissible frontier and best PoA bound
    for the given (w1, w2) K x K sweep.

    The scan uses s^* = SW-argmax (the pure-NE PoA bound only needs the
    inequality at s^* = OPT; extending to CCE would require the universal
    quantifier, which we discuss below).
    """
    sweep = sweep_from_grid(mech_grid, w1=w1, w2=w2)
    A = sweep["A"]; feas = sweep["feasible"]; sigma = sweep["sigma"]
    K = A.shape[0]
    V = V_of_sigma(sigma)

    U1 = V - w1 * A[:, :, 0]
    U2 = V - w2 * A[:, :, 1]
    SW = np.where(feas, U1 + U2, np.nan)

    # Find OPT = argmax over feasible cells.
    sw_flat = np.where(feas, U1 + U2, -np.inf).ravel()
    opt_flat = int(np.argmax(sw_flat))
    i_opt, j_opt = opt_flat // K, opt_flat % K
    SW_opt = float(SW[i_opt, j_opt])

    # Unilateral deviation sum: player 1 to i_opt, player 2 to j_opt.
    # At profile (eps_i, eps_j), the deviation-sum LHS of smoothness is
    #     u_1(eps^opt_1, eps_j) + u_2(eps_i, eps^opt_2)
    #   = U1[i_opt, j] + U2[i, j_opt]
    dev = np.full((K, K), -np.inf)
    for i in range(K):
        for j in range(K):
            if not feas[i, j]:
                continue
            # Both unilateral-deviation profiles must be feasible; else the
            # LHS is -infty (smoothness fails there trivially, but only if
            # we insist on feasibility being preserved. We treat infeasible
            # deviations as "player refuses to deviate" and set the term
            # to the equilibrium utility instead -- this is the LEAST
            # generous reading and can only tighten the deficit).
            if feas[i_opt, j]:
                u1_dev = float(U1[i_opt, j])
            else:
                u1_dev = float(U1[i, j])
            if feas[i, j_opt]:
                u2_dev = float(U2[i, j_opt])
            else:
                u2_dev = float(U2[i, j])
            dev[i, j] = u1_dev + u2_dev

    # Deficit(lambda, mu) = max_s (lambda*SW_opt - mu*SW[s] - dev_sum[s])
    deficit = np.full((len(lam_grid), len(mu_grid)), np.nan)
    for a, lam in enumerate(lam_grid):
        for b, mu in enumerate(mu_grid):
            rhs = lam * SW_opt - mu * SW - dev
            # ignore infeasible cells (NaN in SW)
            defmax = float(np.nanmax(rhs))
            deficit[a, b] = defmax

    # Admissible pairs (deficit <= tol).
    #
    # Correct PoA bound for a *welfare-maximisation* game with smoothness
    #     sum_i u_i(s^*_i, s_{-i}) >= lambda * SW(s^*) - mu * SW(s)
    # is
    #     PoA(pure NE) = SW(OPT) / SW(NE)  <=  (1 + mu) / lambda,
    # not lambda/(1-mu) (which is the cost-minimisation convention). We
    # search for the pair minimising (1 + mu) / lambda among admissible
    # pairs; the necessary constraint at s = OPT is lambda <= 1 + mu
    # (else deficit at OPT itself is positive), which is baked into the
    # numerical admissibility test.
    tol = 1e-9
    adm = deficit <= tol
    if adm.any():
        ii, jj = np.where(adm)
        bounds = (1.0 + mu_grid[jj]) / lam_grid[ii]
        k_best = int(np.argmin(bounds))
        best = dict(
            lam=float(lam_grid[ii[k_best]]),
            mu=float(mu_grid[jj[k_best]]),
            PoA_bound=float(bounds[k_best]),
        )
    else:
        best = dict(lam=None, mu=None, PoA_bound=None,
                    min_deficit=float(np.nanmin(deficit)))

    # Analytic best-lambda frontier: for each mu, lambda_max(mu) is the
    # largest lambda satisfying the smoothness inequality; explicitly,
    #     lambda_max(mu) = min_s (dev_sum[s] + mu * SW[s]) / SW(opt)
    # over the feasible cells. The pure-NE PoA bound is then
    #     PoA_bound(mu) = (1 + mu) / lambda_max(mu).
    # We report the mu that minimises this.
    mask = np.isfinite(dev) & np.isfinite(SW)
    dev_flat = dev[mask]
    sw_flat_all = SW[mask]
    mu_fine = np.linspace(0.0, 5.0, 501)
    frontier = np.empty_like(mu_fine)
    for k, mu in enumerate(mu_fine):
        lam_max = float(np.min((dev_flat + mu * sw_flat_all) / SW_opt))
        frontier[k] = (1.0 + mu) / lam_max if lam_max > 0 else float("inf")
    k_best_analytic = int(np.argmin(frontier))
    mu_best = float(mu_fine[k_best_analytic])
    lam_best = float(np.min((dev_flat + mu_best * sw_flat_all) / SW_opt))
    PoA_best = float(frontier[k_best_analytic])
    analytic = dict(lam=lam_best, mu=mu_best, PoA_bound=PoA_best)

    # Also: SW-based PoA anchor across pure NEs.
    nes = find_pure_NE_2player(sweep)
    if nes:
        sw_at_nes = np.array([float(SW[i, j]) for (i, j) in nes])
        worst_ne = float(np.min(sw_at_nes))
        best_ne = float(np.max(sw_at_nes))
    else:
        worst_ne = best_ne = float("nan")

    return dict(
        opt=(int(i_opt), int(j_opt)),
        opt_eps=(float(sweep["actions"][i_opt]),
                 float(sweep["actions"][j_opt])),
        SW_opt=SW_opt,
        n_ne=len(nes),
        worst_NE_SW=worst_ne,
        best_NE_SW=best_ne,
        SW_ratio_worst=(SW_opt / worst_ne) if worst_ne > 0 else float("nan"),
        deficit_surface=deficit,
        lam_grid=lam_grid,
        mu_grid=mu_grid,
        best=best,
        analytic=analytic,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        # (0) symbolic reduction
        print(symbolic_reduction())

        # (1) load K=200 mechanism grid
        grid = pickle.load(open(CACHE / "mech_grid_K200_g5000x5000_b128_e5.pkl",
                                 "rb"))
        K = len(grid["actions"])
        print("=" * 70)
        print(f"(1) K=200 mechanism grid loaded: {K}x{K} cells, "
              f"{int(grid['feasible'].sum())} feasible")

        lam_grid = np.linspace(0.05, 2.00, 41)
        mu_grid = np.linspace(0.00, 0.99, 41)

        summary = dict(K=K, C1=None, C7={})

        # (2) C1: symmetric, w1=w2=1
        print()
        print("=" * 70)
        print("(2) C1 smoothness scan (w1 = w2 = 1)")
        print("=" * 70)
        r = smoothness_scan(grid, 1.0, 1.0, lam_grid, mu_grid)
        print(f"  OPT idx    = {r['opt']}, eps={r['opt_eps']}")
        print(f"  SW(OPT)    = {r['SW_opt']:.4f}")
        print(f"  pure NEs   = {r['n_ne']}")
        print(f"  worst NE SW= {r['worst_NE_SW']:.4f}")
        print(f"  best NE SW = {r['best_NE_SW']:.4f}")
        print(f"  SW(OPT) / worst-NE-SW = {r['SW_ratio_worst']:.3f}")
        print(f"  min deficit over lambda,mu grid = "
              f"{float(np.nanmin(r['deficit_surface'])):.4g}")
        if r["best"]["PoA_bound"] is not None:
            print(f"  best grid (lambda, mu, PoA_bound=(1+mu)/lambda) = "
                  f"({r['best']['lam']:.3f}, {r['best']['mu']:.3f}, "
                  f"{r['best']['PoA_bound']:.3f})")
        else:
            print(f"  NO admissible (lambda, mu). Min-deficit = "
                  f"{r['best']['min_deficit']:.4g}")
        print(f"  ANALYTIC best over the frontier lambda_max(mu):")
        print(f"    lambda={r['analytic']['lam']:.4f}  mu={r['analytic']['mu']:.4f}  "
              f"PoA_bound=(1+mu)/lambda={r['analytic']['PoA_bound']:.3f}")
        np.savez(OUT / "deficit_C1_w1.npz",
                 deficit=r["deficit_surface"],
                 lam=lam_grid, mu=mu_grid)
        summary["C1"] = dict(
            opt_eps=r["opt_eps"], SW_opt=r["SW_opt"], n_ne=r["n_ne"],
            worst_NE_SW=r["worst_NE_SW"], best_NE_SW=r["best_NE_SW"],
            SW_ratio_worst=r["SW_ratio_worst"],
            best_lam_grid=r["best"]["lam"], best_mu_grid=r["best"]["mu"],
            PoA_bound_grid=r["best"]["PoA_bound"],
            lam_analytic=r["analytic"]["lam"],
            mu_analytic=r["analytic"]["mu"],
            PoA_bound_analytic=r["analytic"]["PoA_bound"],
            min_deficit=float(np.nanmin(r["deficit_surface"])),
        )

        # (3) C7 sweep -- symmetric w in {0.125, 0.25, 0.5, 1, 2, 4}
        print()
        print("=" * 70)
        print("(3) C7 symmetric-w sweep")
        print("=" * 70)
        for w in [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]:
            r = smoothness_scan(grid, w, w, lam_grid, mu_grid)
            print(f"  w={w:5.3f}  n_NE={r['n_ne']:2d}  "
                  f"SW(OPT)={r['SW_opt']:6.3f}  "
                  f"worst_NE_SW={r['worst_NE_SW']:6.3f}  "
                  f"SW-ratio(OPT/worst-NE)={r['SW_ratio_worst']:6.2f}  "
                  f"analytic lambda={r['analytic']['lam']:.3f} "
                  f"mu={r['analytic']['mu']:.3f} "
                  f"PoA<={r['analytic']['PoA_bound']:.3f}")
            np.savez(OUT / f"deficit_C7_w{w}.npz",
                     deficit=r["deficit_surface"],
                     lam=lam_grid, mu=mu_grid)
            summary["C7"][str(w)] = dict(
                opt_eps=r["opt_eps"], SW_opt=r["SW_opt"], n_ne=r["n_ne"],
                worst_NE_SW=r["worst_NE_SW"], best_NE_SW=r["best_NE_SW"],
                SW_ratio_worst=r["SW_ratio_worst"],
                best_lam_grid=r["best"]["lam"], best_mu_grid=r["best"]["mu"],
                PoA_bound_grid=r["best"]["PoA_bound"],
                lam_analytic=r["analytic"]["lam"],
                mu_analytic=r["analytic"]["mu"],
                PoA_bound_analytic=r["analytic"]["PoA_bound"],
                min_deficit=float(np.nanmin(r["deficit_surface"])),
            )

    (OUT / "verbatim_stdout.txt").write_text(buf.getvalue())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2,
                                                  default=float))
    print(buf.getvalue())


if __name__ == "__main__":
    main()
