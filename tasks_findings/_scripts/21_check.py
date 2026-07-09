"""
T21 numerical certificate for supermodular NE existence on the PEG.

For the supermodular route we need the sign of d^2 u_i / (d eps_i d eps_j)
on the feasible region. Under the reparametrisation a_i := -eps_i, the
game is supermodular iff d^2 u_i / (d eps_i d eps_j) <= 0 on the operating
sublattice.

We centre-difference the K x K x 2 utility surface stored in
`game_experiments/_cache_v2/C1.pkl` (fully symmetric 2-player scenario,
K=22 log-grid on [0.5, 64]) and in `_cache_v2/C2.pkl` (7 group-size
ratios, K=9 log-grid) at every interior feasible cell and report

  * fraction of cells with d^2 u_i / (d eps_i d eps_j) <= 0 (i.e.
    utility submodular in eps, hence supermodular in a := -eps),
  * fraction on the "small-p face" identified via A_i < 0.05
    (proxy for p_i in the small-p regime of T20's corollary),
  * MCS spot check: how the extremal off-diagonal NEs of C2 shift as the
    group-size ratio |G_1|/|G_2| varies.

Sympy also verifies the leading-order small-p cross-partial identity
predicted by T20 and used as the analytic input in T21's Lemma.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import sympy as sp

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "game_experiments" / "_cache_v2"


# ---------------------------------------------------------------------------
# (0) Sympy: leading-order cross-partial of u_i in the small-p regime.
# u_i = V(sigma) - w_i * A_i,  A_i = p_i * AdvG(sigma) + O(p^2).
# ---------------------------------------------------------------------------
def sympy_check():
    print("=" * 72)
    print("[0] Sympy: cross-partial of u_i in eps at leading order in p (T20)")
    print("=" * 72)
    eps_i, eps_j = sp.symbols("eps_i eps_j", real=True)
    sigma = sp.Function("sigma")(eps_i, eps_j)
    p = sp.Function("p_i")(eps_i, eps_j)
    AdvG = sp.Function("A")(sigma)  # unsubsampled MIA advantage of Gaussian
    V = sp.Function("V")(sigma)
    w = sp.symbols("w", positive=True)

    Ai_lead = p * AdvG
    u = V - w * Ai_lead

    cp = sp.diff(u, eps_i, eps_j).expand()
    print("d^2 u_i / (d eps_i d eps_j) =")
    print(sp.pretty(cp))

    # Substitute symbolic signs used in the T21 lemma.
    # We can't evaluate signs symbolically without more structure, but we
    # can factor and note the dominant term.
    dp_i = sp.diff(p, eps_i)
    dp_j = sp.diff(p, eps_j)
    ds_i = sp.diff(sigma, eps_i)
    ds_j = sp.diff(sigma, eps_j)
    ds_ij = sp.diff(sigma, eps_i, eps_j)
    dp_ij = sp.diff(p, eps_i, eps_j)
    dV = sp.diff(V, sigma)
    dVV = sp.diff(V, sigma, 2)
    dA = sp.diff(AdvG, sigma)
    dAA = sp.diff(AdvG, sigma, 2)
    # Symbolically rewrite the cross partial via chain rule
    cp_manual = (
        dVV * ds_i * ds_j + dV * ds_ij
        - w * (
            dp_ij * AdvG
            + dp_i * dA * ds_j
            + dp_j * dA * ds_i
            + p * (dAA * ds_i * ds_j + dA * ds_ij)
        )
    )
    diff = sp.simplify(cp - cp_manual)
    print(f"\n(cross-partial expansion residual = {diff})")
    assert diff == 0, "expansion should match"
    print("Chain-rule expansion matches.")

    # T20 signs on the small-p face:
    #   dV/dsigma = V' < 0,  d2V/dsigma2 = V'' -- sign depends on V family;
    #   dA/dsigma = AdvG' < 0 (T20 sec 4.2),
    #   dsigma/deps_k < 0 for all k (lem:externality),
    #   dp_i/deps_j < 0 for j != i, dp_i/deps_i > 0.
    #   At symmetric groups d2 p_i / (deps_i deps_j) = 0 (T06 lem cross-part-A).
    #
    # At leading order in p (T20), the dominant surviving term is
    #   -w * dp_i * dA * ds_j - w * dp_j * dA * ds_i
    #     = -w * AdvG' * (dp_i * ds_j + dp_j * ds_i).
    # With (dp_i>0, ds_j<0) and (dp_j<0, ds_i<0), the bracket is
    #   (+)(-) + (-)(-) = -|.| + |.|,  ambiguous individually,
    # but T20's dominance calc pins the sign of Adv_G(s)/s + Adv_G'(s) > 0
    # which forces the aggregate to be signed. The certificate below
    # verifies the aggregate sign directly on C1 and C2.
    print("\nSee C1/C2 checks below for the aggregate sign of d^2 u_i on the")
    print("feasible region and the small-p face.")


# ---------------------------------------------------------------------------
# Finite-difference cross partial on a K x K x 2 utility grid.
# ---------------------------------------------------------------------------
def cross_partial_grid(U_pl: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Return the centred-difference d^2 U / (d eps_1 d eps_2) at interior
    cells; NaN elsewhere."""
    K = U_pl.shape[0]
    out = np.full((K, K), np.nan)
    for i in range(1, K - 1):
        for j in range(1, K - 1):
            num = (
                U_pl[i + 1, j + 1]
                - U_pl[i + 1, j - 1]
                - U_pl[i - 1, j + 1]
                + U_pl[i - 1, j - 1]
            )
            den = (actions[i + 1] - actions[i - 1]) * (actions[j + 1] - actions[j - 1])
            out[i, j] = num / den
    return out


def _summary(cp: np.ndarray, feas: np.ndarray, label: str, small_p_mask=None):
    interior = np.zeros_like(feas, dtype=bool)
    interior[1:-1, 1:-1] = True
    mask = feas & interior & ~np.isnan(cp)
    if not mask.any():
        print(f"  [{label}] no interior feasible cells")
        return
    vals = cp[mask]
    n = vals.size
    n_neg = int((vals <= 0).sum())
    n_pos = int((vals > 0).sum())
    print(
        f"  [{label}] n={n}  d^2u<=0: {n_neg} ({100*n_neg/n:5.1f}%)"
        f"  d^2u>0: {n_pos} ({100*n_pos/n:5.1f}%)"
        f"  range=[{vals.min():+.3e}, {vals.max():+.3e}]"
    )
    if small_p_mask is not None:
        smask = mask & small_p_mask
        if smask.any():
            svals = cp[smask]
            sn = svals.size
            sn_neg = int((svals <= 0).sum())
            print(
                f"    small-p (A_i<0.05): n={sn}  d^2u<=0: {sn_neg} "
                f"({100*sn_neg/sn:5.1f}%)  worst positive: "
                f"{svals.max():+.3e}"
            )


def check_C1():
    print("\n" + "=" * 72)
    print("[1] C1: 2-player symmetric, |G_1|=|G_2|=5000, K=22")
    print("=" * 72)
    d = pickle.load((CACHE / "C1.pkl").open("rb"))
    sw = d["sweep"]
    actions = sw["actions"]
    U = sw["U"]        # (K, K, 2)
    A = sw["A"]        # (K, K, 2)
    feas = sw["feasible"]

    for pl in (0, 1):
        cp = cross_partial_grid(U[:, :, pl], actions)
        small_p_mask = A[:, :, pl] < 0.05
        _summary(cp, feas, f"player {pl+1}", small_p_mask=small_p_mask)


def check_C2():
    print("\n" + "=" * 72)
    print("[2] C2: 7 group-size ratios, K=9")
    print("=" * 72)
    d = pickle.load((CACHE / "C2.pkl").open("rb"))

    for row in d["rows"]:
        actions = row["actions"]
        U = row["U"]
        A = row["A"]
        feas = row["feasible"]
        print(f"\n  ratio |G_1|/|G_2| = {row['ratio']:.3g}  "
              f"(|G_1|={row['g1']}, |G_2|={row['g2']})")
        for pl in (0, 1):
            cp = cross_partial_grid(U[:, :, pl], actions)
            small_p_mask = A[:, :, pl] < 0.05
            _summary(cp, feas, f"    player {pl+1}", small_p_mask=small_p_mask)


def mcs_spot_check_C2():
    """Monotone comparative statics spot check.

    For asymmetric groups the T21 Corollary predicts that the extremal NE
    eps_i (extreme in each player's coordinate) shifts monotonically as
    |G_i| grows: a larger group carries a *smaller* eps at the extremal
    NEs (higher effective privacy sensitivity per capita). We report the
    min/max of eps_1 and eps_2 across the enumerated NEs in each C2 row
    and check monotonicity in the group-size ratio.
    """
    print("\n" + "=" * 72)
    print("[3] MCS spot check on C2: extremal NE shift vs group-size ratio")
    print("=" * 72)
    d = pickle.load((CACHE / "C2.pkl").open("rb"))
    rows = []
    for row in d["rows"]:
        actions = row["actions"]
        nes = row["nes"]
        if not nes:
            rows.append((row["ratio"], row["g1"], row["g2"], None, None, None, None))
            continue
        e1 = [actions[i] for (i, _) in nes]
        e2 = [actions[j] for (_, j) in nes]
        rows.append((row["ratio"], row["g1"], row["g2"],
                     min(e1), max(e1), min(e2), max(e2)))
    print(f"  {'ratio':>8s} {'|G1|':>6s} {'|G2|':>6s}  "
          f"{'e1_min':>7s} {'e1_max':>7s} {'e2_min':>7s} {'e2_max':>7s}")
    for (r, g1, g2, a, b, c, e) in rows:
        if a is None:
            print(f"  {r:>8.3g} {g1:>6d} {g2:>6d}  (no NE on grid)")
        else:
            print(f"  {r:>8.3g} {g1:>6d} {g2:>6d}  "
                  f"{a:>7.3f} {b:>7.3f} {c:>7.3f} {e:>7.3f}")
    # Report Kendall tau between ratio and e1_min / e2_min (should be
    # negative-of-negative i.e. monotone in the direction Corollary predicts).
    good = [row for row in rows if row[3] is not None]
    if len(good) >= 3:
        ratios = np.array([row[0] for row in good])
        e1_mins = np.array([row[3] for row in good])
        e2_mins = np.array([row[5] for row in good])
        # simple sign-agreement
        def agree(x, y):
            n = 0; a = 0
            for i in range(len(x)):
                for j in range(i+1, len(x)):
                    dx = np.sign(x[j]-x[i])
                    dy = np.sign(y[j]-y[i])
                    if dx == 0 or dy == 0:
                        continue
                    n += 1
                    if dx == dy:
                        a += 1
            return a, n
        a1, n1 = agree(ratios, e1_mins)
        a2, n2 = agree(ratios, e2_mins)
        print(f"\n  concordant pairs (ratio, e1_min): {a1}/{n1}")
        print(f"  concordant pairs (ratio, e2_min): {a2}/{n2}")


if __name__ == "__main__":
    sympy_check()
    check_C1()
    check_C2()
    mcs_spot_check_C2()
