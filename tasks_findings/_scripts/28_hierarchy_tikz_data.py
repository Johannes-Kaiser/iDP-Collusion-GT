"""T28 -- data for the TikZ advantage-hierarchy figure (Sec. II-C of main2).

Trade-off curves are computed EXACTLY from the composed privacy-loss
distribution (dp_accounting) as the optimal Neyman-Pearson ROC, so they are
smooth. Constructions and cross-checks follow Dong, Roth, Su, "Gaussian
Differential Privacy" (arXiv:1905.02383).

The privacy loss distribution stores, for the "upper" hypothesis P (add),
probabilities q_i at loss values l_i = (lower_loss + i) * disc, plus an
infinity mass at l = +inf; the "lower" hypothesis Q (remove) has
Q(l_i) = e^{-l_i} q_i and no infinity mass. The optimal test that rejects Q
when the loss exceeds a threshold t traces the trade-off curve
  alpha(t) = Q(loss > t) = sum_{l_i > t} e^{-l_i} q_i          (type-I error)
  T(t)     = P(loss <= t) = 1 - [sum_{l_i > t} q_i + inf_mass] (type-II error)
and the advantage (total variation, = level-2 A_PLD) is
  Adv = max_t [P(loss>t) - Q(loss>t)]
      = max_t [sum_{l_i>t} q_i (1 - e^{-l_i}) + inf_mass].
For the Gaussian mechanism this equals the closed form 2*Phi(mu/2)-1
(Dong-Roth-Su), which we assert as a correctness check.

Five Poisson-subsampled Gaussian mechanisms (shades of blue), single
composition (I=1) so the accountant is exact, calibrated to pass through
the shared contract point (eps0, delta0) = (10, 1e-12) by bisecting the
noise multiplier sigma while sweeping the subsampling rate p over a wide
range; each is verified to hit delta0 within 1%. The (eps,delta) envelope
is plotted only in the trade-off and advantage panels (level-1 worst case).

Outputs: paper/figures/data/hier_*.dat and 28_hierarchy_summary.txt.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dp_accounting import pld as dp_pld

# composition count, prefix, and contract point are argv-overridable so the
# same generator produces the I=1 paper figure and the calibration-point
# comparison rows.  Usage: 28_hierarchy_tikz_data.py [STEPS] [PREFIX] [EPS0] [DELTA0]
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "hier"
EPS0 = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
DELTA0 = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-12
P_VALUES = [0.001, 0.01, 0.1, 0.3, 0.6, 1.0]  # p=1 is the plain Gaussian
MECHS = [(p, STEPS) for p in P_VALUES]
DISC = 1e-4
DATA = ROOT / "paper" / "figures" / "data"
SUMMARY = (ROOT / "tasks_findings" / "_scripts"
           / f"28_hierarchy_summary_{PREFIX}.txt")

# dense near eps=0 so the profile's saturation at delta(0)=TV renders smoothly
EPS_PLOT = np.unique(np.concatenate([
    np.linspace(0.0, 1.2 * EPS0, 240),
    np.geomspace(1e-4, 1.0, 60)]))
ALPHA_ENV = np.unique(np.concatenate([
    np.geomspace(1e-8, 1e-2, 120), np.linspace(1e-2, 1 - 1e-5, 300),
    1 - np.geomspace(1e-8, 1e-2, 120)]))
FLOOR = 1e-14
N_CURVE = 400  # points sampled along each ROC for plotting


def make_pld(sigma, p, steps):
    return dp_pld.privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=sigma, sampling_prob=p,
        value_discretization_interval=DISC,
        use_connect_dots=True).self_compose(steps)


def calibrate_sigma(p, steps):
    lo, hi = 0.05, 120.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if make_pld(mid, p, steps).get_delta_for_epsilon(EPS0) > DELTA0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-7 * hi:
            break
    return 0.5 * (lo + hi)


def roc_tradeoff(obj):
    """Exact optimal trade-off curve (alpha, T) and advantage from the PLD."""
    pmf = obj._pmf_add if not obj._symmetric else obj._pmf_add
    disc = pmf._discretization
    lower = pmf._lower_loss
    q = np.asarray(pmf._probs, dtype=float)       # P(loss = l_i), upper hyp.
    inf_mass = float(pmf._infinity_mass)
    n = len(q)
    losses = (lower + np.arange(n)) * disc
    # sweep threshold from high loss to low: include mass with l_i > t.
    # process in descending loss order (arrays are ascending in i).
    p_desc = q[::-1]                               # P mass, high loss first
    l_desc = losses[::-1]
    qmass_desc = np.exp(-l_desc) * p_desc          # Q mass at same losses
    P_gt = np.cumsum(p_desc) + inf_mass            # P(loss > t) incl inf mass
    Q_gt = np.cumsum(qmass_desc)                   # Q(loss > t)
    # prepend the empty-rejection point (t = +inf): reject nothing
    P_gt = np.concatenate([[inf_mass], P_gt])
    Q_gt = np.concatenate([[0.0], Q_gt])
    alpha = Q_gt                                   # type-I error
    T = 1.0 - P_gt                                 # type-II error
    adv = float(np.max(P_gt - Q_gt))               # total variation
    # monotone, dedup, subsample for plotting
    order = np.argsort(alpha)
    alpha, T = alpha[order], T[order]
    keep = np.concatenate([[True], np.diff(alpha) > 1e-9])
    alpha, T = alpha[keep], T[keep]
    idx = np.unique(np.linspace(0, len(alpha) - 1, N_CURVE).astype(int))
    return alpha[idx], np.clip(T[idx], 0, 1), adv


def subsampled_gaussian_tradeoff(sigma, p):
    """Exact, stable trade-off of a SINGLE-STEP (p, sigma)-subsampled Gaussian.
    Hypotheses on the per-example statistic (sensitivity 1): out ~ N(0,sigma^2)
    vs in ~ (1-p) N(0,sigma^2) + p N(1,sigma^2). The likelihood ratio is
    monotone in x, so the optimal test thresholds x; with alpha = 1 - Phi(t/sigma)
    (i.e. t = sigma * Phi^{-1}(1-alpha)),
        power(t) = (1-p)(1-Phi(t/sigma)) + p(1-Phi((t-1)/sigma)),
        T(alpha) = 1 - power.
    This reproduces G_mu exactly at p=1 (mu = 1/sigma) and is free of the
    PLD-reconstruction instability. (Single composition only.)"""
    alpha = np.unique(np.clip(np.concatenate([
        np.geomspace(1e-7, 1e-2, 250), np.linspace(1e-2, 1 - 1e-2, 700),
        1 - np.geomspace(1e-7, 1e-2, 250)]), 0.0, 1.0))
    t = sigma * norm.ppf(np.clip(1 - alpha, 1e-15, 1 - 1e-15))
    power = (1 - p) * (1 - norm.cdf(t / sigma)) + p * (1 - norm.cdf((t - 1) / sigma))
    T = np.clip(1 - power, 0.0, 1.0)
    adv = float(np.max(power - alpha))
    return alpha, T, adv


def f_eps_delta(eps, delta, alpha):
    return np.maximum.reduce([
        np.zeros_like(alpha),
        1 - delta - math.exp(eps) * alpha,
        math.exp(-eps) * (1 - delta - alpha)])


def dual_tradeoff(obj):
    """Stable trade-off T(alpha) = sup_eps f_{eps,delta(eps)}(alpha) from the
    profile, with a dense signed eps grid.  Numerically robust at heavy
    composition where the direct ROC reconstruction is unstable."""
    alpha = np.unique(np.concatenate([
        np.geomspace(1e-8, 1e-2, 200), np.linspace(1e-2, 1 - 1e-5, 500),
        1 - np.geomspace(1e-8, 1e-2, 200)]))
    eps_grid = np.concatenate([-np.geomspace(1e-3, 25.0, 400)[::-1],
                               [0.0], np.geomspace(1e-3, 25.0, 400)])
    deltas = np.array([obj.get_delta_for_epsilon(float(e)) for e in eps_grid])
    T = np.zeros_like(alpha)
    for e, d in zip(eps_grid, deltas):
        T = np.maximum(T, f_eps_delta(e, d, alpha))
    idx = np.unique(np.linspace(0, len(alpha) - 1, N_CURVE).astype(int))
    return alpha[idx], np.clip(T[idx], 0, 1)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    out = []
    say = lambda s: (print(s, flush=True), out.append(s))
    say(f"# T28 hierarchy TikZ data (exact ROC trade-off): "
        f"contract eps0={EPS0:g}, delta0={DELTA0:g}, PLD disc={DISC}")

    # --- correctness check: pure Gaussian ROC advantage vs closed form ---
    from scipy.optimize import brentq
    def gdp_delta(eps, mu):
        return norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2)
    mu = brentq(lambda m: gdp_delta(EPS0, m) - DELTA0, 0.05, 20.0, xtol=1e-12)
    g = make_pld(mu, 1.0, 1)  # sampling_prob=1 -> plain Gaussian, sigma=mu^-1? check
    # For p=1 the "subsampled" Gaussian is the plain Gaussian N(0,sigma^2) vs
    # N(1,sigma^2) with mu_GDP = 1/sigma. Calibrate sigma so 1/sigma = mu:
    g = make_pld(1.0 / mu, 1.0, 1)
    _, _, adv_g = roc_tradeoff(g)
    adv_closed = 2 * norm.cdf(mu / 2) - 1
    say(f"check: Gaussian mu={mu:.4f} ROC adv={adv_g:.4f} vs closed "
        f"{adv_closed:.4f} (|diff|={abs(adv_g-adv_closed):.1e})")
    assert abs(adv_g - adv_closed) < 3e-3, "ROC advantage disagrees with GDP closed form"

    advs = []
    for k, (p, steps) in enumerate(MECHS, start=1):
        sigma = calibrate_sigma(p, steps)
        obj = make_pld(sigma, p, steps)
        d_at = obj.get_delta_for_epsilon(EPS0)
        rel = abs(d_at - DELTA0) / DELTA0
        assert rel < 0.01, f"mech {k} off budget: delta={d_at:.3e} rel {rel:.3f}"
        prof = np.array([obj.get_delta_for_epsilon(float(e)) for e in EPS_PLOT])
        # advantage = total variation = delta(0): exact and numerically stable.
        adv = obj.get_delta_for_epsilon(0.0)
        advs.append(adv)
        # trade-off curve: computed EXACTLY from the two 1-D hypothesis
        # distributions of the single-step subsampled Gaussian -- smooth and
        # free of the PLD-reconstruction instability (I=1 only).
        assert steps == 1, "direct trade-off is single-composition only"
        alpha, T, adv_direct = subsampled_gaussian_tradeoff(sigma, p)
        assert abs(adv_direct - adv) < 3e-3, (
            f"mech {k}: direct trade-off adv {adv_direct:.4f} vs "
            f"delta(0) {adv:.4f}")
        say(f"mech {k}: p={p:g} I={steps} -> sigma={sigma:.5f}, "
            f"delta(eps0)={d_at:.4e} (rel {rel:.4f}), Adv=delta(0)={adv:.4f}")
        np.savetxt(DATA / f"{PREFIX}_profile_{k}.dat",
                   np.column_stack([EPS_PLOT, np.maximum(prof, FLOOR)]),
                   header="eps delta", comments="")
        np.savetxt(DATA / f"{PREFIX}_tradeoff_{k}.dat",
                   np.column_stack([alpha, T]), header="alpha beta", comments="")

    # --- envelope: level-1 worst case, trade-off + advantage only ---
    toff_env = f_eps_delta(EPS0, DELTA0, ALPHA_ENV)
    adv_env = (math.exp(EPS0) - 1 + 2 * DELTA0) / (math.exp(EPS0) + 1)
    say(f"envelope: A_(eps,delta) = {adv_env:.6f}")
    np.savetxt(DATA / f"{PREFIX}_tradeoff_env.dat",
               np.column_stack([ALPHA_ENV, np.clip(toff_env, 0, 1)]),
               header="alpha beta", comments="")

    with open(DATA / f"{PREFIX}_adv.dat", "w") as f:
        f.write("index adv\n")
        for k, adv in enumerate(advs, start=1):
            f.write(f"{k} {adv:.6f}\n")
        f.write(f"{len(advs)+1} {adv_env:.6f}\n")

    say(f"advantage spread: [{min(advs):.3f}, {max(advs):.3f}]")
    SUMMARY.write_text("\n".join(out) + "\n")
    say(f"wrote {DATA}/hier_*.dat and {SUMMARY}")


if __name__ == "__main__":
    main()
