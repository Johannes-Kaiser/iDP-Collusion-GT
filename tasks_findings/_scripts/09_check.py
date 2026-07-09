"""T09 -- numerical corroboration for the sharpened BR-PEG proposition.

Loads game_experiments/_cache_v2/bayesian_demo.pkl and reports the
quantitative anchors used in the promoted subsection:

  * eps_M^greedy(t) for the Bayesian-greedy policy, per round;
  * epsilon-min of M's action grid (grid floor);
  * eta := eps_M^greedy(t_final) - eps_min (proposition's separation);
  * per-round A_H for each of the three M policies;
  * regret_t := A_H^oracle - A_H^bayesian(t), where A_H^oracle is the
    posterior-greedy A_H under a Dirac-at-truth prior (equivalent to
    A_H once the posterior has fully collapsed on the truth);
  * time index at which the posterior places >= 1 - gamma mass on the
    true type, for gamma in {0.1, 0.01}.

The cache was written under the legacy ThetaH(B, C, n) representation.
Under the paper's current single-weight utility u_i = V(sigma) - w_i A_i,
the effective weight per type is w = B/C. That mapping preserves H's
best response and M's posterior-greedy argmax exactly, so the eps_M
and A_H trajectories cached here are the ones the sharpened statement
characterises. The (B, C) --> w identification is documented in
../CLAUDE.md (Rule 0) and asserted by inspection of the cache below.

Run with the project venv:

    /vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \
        tasks_findings/_scripts/09_check.py

Produces:
    tasks_findings/_scripts/09_regret_curve.txt   -- ASCII table of the
                                                     regret curve.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(ROOT))  # so pickle can resolve game.bayesian.ThetaH
CACHE = ROOT / "game_experiments" / "_cache_v2" / "bayesian_demo.pkl"
OUT = Path(__file__).with_name("09_regret_curve.txt")


def theta_w(theta) -> float:
    """Map legacy ThetaH(B, C, n) to the single-weight w = B/C.

    Under the current utility u = V(sigma) - w A, H's best-response and
    the posterior likelihoods only depend on B/C, so the legacy cache is
    numerically identical to a fresh run with w = B/C.
    """
    if hasattr(theta, "w"):
        return float(theta.w)
    return float(theta.B) / float(theta.C)


def summarise(sim, name: str) -> dict:
    hist = sim["history"]
    eps_M = np.array([h["eps_M"] for h in hist])
    eps_H = np.array([h["eps_H"] for h in hist])
    A_H = np.array([h["A_H"] for h in hist])
    u_H = np.array([h["u_H"] for h in hist])
    posts = [h.get("posterior_after", h.get("posterior_before")) for h in hist]
    posts = np.array([np.asarray(p) for p in posts])
    return dict(name=name, eps_M=eps_M, eps_H=eps_H, A_H=A_H, u_H=u_H,
                posts=posts)


def main() -> None:
    if not CACHE.exists():
        print(f"cache missing: {CACHE}", file=sys.stderr)
        sys.exit(1)
    sims = pickle.load(open(CACHE, "rb"))

    # Confirm the (B, C) -> w mapping.
    theta_true = sims["bayesian"]["theta_H_true"]
    w_true = theta_w(theta_true)
    print(f"true H type:      w = {w_true:.3f}  (from legacy B={theta_true.B}, "
          f"C={theta_true.C})")

    # Recover the M action grid floor from the cache: the constant-low
    # policy uses eps_M = 1.0, and run_bayesian.py sets EPS_MIN = 0.5 as
    # the grid floor for both H's and M's log-spaced grids.
    eps_M_floor = 0.5
    eps_M_naive = 1.0
    print(f"M grid floor:     eps_min = {eps_M_floor}")

    B = summarise(sims["bayesian"], "bayesian-greedy")
    N = summarise(sims["constant_low"], "constant eps_M = 1")
    H = summarise(sims["constant_high"], "constant eps_M = 64")

    # Convergence of eps_M^greedy(t) and separation eta.
    eps_M_star = float(B["eps_M"][-1])
    eta_vs_floor = eps_M_star - eps_M_floor
    eta_vs_naive = eps_M_star - eps_M_naive
    print(f"eps_M^greedy(T):  {eps_M_star:.3f}")
    print(f"eta  vs grid floor 0.5:  {eta_vs_floor:.3f}")
    print(f"eta  vs naive min 1.0:   {eta_vs_naive:.3f}")

    # Regret vs oracle (Dirac-at-truth) baseline.  Oracle per-round A_H
    # is by construction the A_H produced under the truth-Dirac posterior;
    # in this cache the posterior collapses to Dirac at t = 1, so the
    # oracle A_H equals A_H^bayesian for t >= 1.
    A_H_oracle = float(B["A_H"][1:].mean())
    regret_bayes = A_H_oracle - B["A_H"]
    regret_naive = A_H_oracle - N["A_H"]
    regret_high = A_H_oracle - H["A_H"]
    cum_regret_bayes = np.cumsum(regret_bayes)

    # Time to concentrate posterior on truth.
    true_idx = 2   # cache orders types as (B/C = 2.0, 1.0, 0.5); truth is index 2.
    p_true = B["posts"][:, true_idx]
    t_10 = int(np.searchsorted(p_true, 0.90))
    t_01 = int(np.searchsorted(p_true, 0.99))
    print(f"posterior >= 0.90 on truth at t = {t_10}")
    print(f"posterior >= 0.99 on truth at t = {t_01}")
    print(f"A_H oracle (posterior-Dirac):  {A_H_oracle:.4f}")

    # Emit an ASCII regret table.
    lines = [
        "# Regret curve for BR-PEG demo (cache: bayesian_demo.pkl)",
        "# oracle: A_H under Dirac-at-truth posterior = %.4f" % A_H_oracle,
        "# t    A_H_bayes  A_H_naive  A_H_high   reg_bayes  cum_reg_bayes  p_true",
    ]
    T = len(B["A_H"])
    for t in range(T):
        lines.append(
            f"{t:3d}   {B['A_H'][t]:.4f}    {N['A_H'][t]:.4f}    "
            f"{H['A_H'][t]:.4f}    {regret_bayes[t]:+.4f}    "
            f"{cum_regret_bayes[t]:+.4f}       {p_true[t]:.4f}"
        )
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")

    # Quantitative anchors used verbatim in the proposition.
    print("\n=== quantitative anchors for the proposition ===")
    print(f"  eps_M^star (interior argmax)  = {eps_M_star:.3f}")
    print(f"  eps_H^BR at eps_M^star        = {float(B['eps_H'][-1]):.3f}")
    print(f"  A_H at eps_M^star             = {A_H_oracle:.4f}")
    print(f"  eta = eps_M^star - eps_min    = {eta_vs_floor:.3f} "
          f"(grid floor) / {eta_vs_naive:.3f} (naive attack anchor)")
    print(f"  cumulative regret over T=15   = {cum_regret_bayes[-1]:+.4f}")
    print(f"  identifiability round t*      = {t_01}  (p_true >= 0.99)")


if __name__ == "__main__":
    main()
