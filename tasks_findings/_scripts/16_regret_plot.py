"""T16 -- H8 BR-PEG regret curve (redesigned).

Rewrite driven by the finding that the original 3-type demo
(Theta = {w=0.5, 1.0, 2.0}) has such a type-separating log-Gaussian
likelihood that M's posterior collapses in one round, making the "O(1)
regret" claim visually trivial: BR-PEG jumps to the full-information
optimum at t=1 and stays there forever. The reader sees a step function
and cannot judge how BR-PEG's Bayesian design *earns* its regret --
identification is essentially free because the setup is too clean.

This rewrite introduces two sources of ambiguity that make the learning
curve real:

  1.  Overlapping type space: Theta = {w in {0.6, 0.8, 1.0, 1.4, 2.0}},
      the two middle types differ only by 20% in privacy weight and
      induce nearly identical (eps_H, u_H) observations.

  2.  Stochastic observation noise: M observes (eps_H, u_H) corrupted
      by log-Gaussian noise of width sigma_obs. This is the honest
      model of what M actually sees on any real deployment (H's action
      and utility are estimated from finite proxies), and it makes the
      likelihood posterior a proper Bayesian update rather than a
      delta-function classifier.

Under this setup, BR-PEG's cumulative regret against the full-information
Stackelberg oracle grows sub-linearly (concave, plateaus once the
posterior has sharpened enough to consistently pick the oracle arm),
which is the ordinary regime the Stackelberg-learning literature covers.

We compare four policies against a single Stackelberg oracle
(A_H_oracle = A_H under the constant eps_M that maximises E_{true type}
A_H, evaluated at the true type):

  * BR-PEG greedy with a correctly-specified prior on Theta:
    sub-linear regret that plateaus once the posterior identifies the
    truth; the "well-tuned Bayesian" story.
  * BR-PEG greedy with a mis-specified prior (support excludes the
    truth): the posterior converges to the nearest wrong type and the
    policy locks onto that type's greedy arm, paying linear regret at
    (asymptotically) the uniform-belief slope; quantifies the *cost of
    prior mis-specification* -- learning buys nothing once the truth
    is outside the support.
  * EXP3 (Auer et al. 2002), a black-box adversarial-bandit no-regret
    algorithm over the same eps_M arm set with no prior at all:
    sub-linear asymptotically, but over practical horizons (T <= 60)
    its forced exploration makes it the *worst* of the four; the
    "atheoretical baseline that a reviewer would compare against".
  * Uniform-belief constant M (single-round argmax under U(Theta)):
    exactly linear regret; the naive stationary policy.

The figure is a single panel (no title, self-contained, style-matched
to fig_h5_v_shape). Mean cumulative regret over N_SEEDS seeds with a
+/- 1 SE band.

Reproduce (about 2-3 minutes on the login node; the mechanism lookup is
cached to `_cache_v2/H8_regret_lookup.pkl` after the first run and the
downstream simulation is pure numpy):

    /vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \\
        tasks_findings/_scripts/16_regret_plot.py

Outputs:
  paper/figures/fig_h8_brpeg_regret.pdf
  paper/figures/fig_h8_brpeg_regret.svg
  tasks_findings/_scripts/16_regret_summary.txt
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(ROOT))

from game.bayesian import H_best_response
from game.utility import value_fn
from game.style import (
    apply_paper_style, PRIMARY_BLUE, PRIMARY_ORANGE, ACCENT_GREY,
    column_size,
)

LOOKUP_CACHE = ROOT / "game_experiments" / "_cache_v2" / "H8_regret_lookup.pkl"
FIG_PDF = ROOT / "paper" / "figures" / "fig_h8_brpeg_regret.pdf"
FIG_SVG = ROOT / "paper" / "figures" / "fig_h8_brpeg_regret.svg"
SUMMARY = Path(__file__).with_name("16_regret_summary.txt")

# ---------------------------------------------------------------------------
# Instance definition
# ---------------------------------------------------------------------------

# Overlapping type space. The two middle types (0.8 and 1.0) induce nearly
# identical (eps_H, u_H) observations, so distinguishing them requires
# accumulating evidence over multiple rounds.
TYPE_WS = [0.6, 0.8, 1.0, 1.4, 2.0]
N_H = 5000
N_M = 5000

# eps grids
EPS_H_MIN, EPS_H_MAX = 0.5, 64.0
K_H = 9
EPS_H_GRID = np.logspace(np.log10(EPS_H_MIN), np.log10(EPS_H_MAX), K_H)

# eps_M arm set (kept small so EXP3 has a fair chance; 7 arms in log-space)
EPS_M_MIN, EPS_M_MAX = 1.0, 32.0
K_M = 7
EPS_M_GRID = np.logspace(np.log10(EPS_M_MIN), np.log10(EPS_M_MAX), K_M)

EXPECTED_BATCH = 128
STEPS = int(5 * (N_H + N_M) / EXPECTED_BATCH)
DELTA = 1e-12

# True type index. w = 0.6 is chosen because (i) its Stackelberg-oracle
# arm (eps_M = 3.17, A_H* = 0.454) differs from the uniform-prior argmax
# arm (eps_M = 1.78), so a policy that never learns pays linear regret;
# and (ii) at the uniform-prior arm the three low-w types {0.6, 0.8, 1.0}
# induce the *same* eps_H best response and differ only by ~0.06-0.13 in
# u_H -- below the observation noise -- so identifying the truth takes
# multiple rounds of evidence and the learning curve is real.
TRUE_TYPE_IDX = 0  # w = 0.6

# Observation noise (log-Gaussian on both eps_H and u_H).
# Chosen so that a single round provides evidence but not certainty;
# calibrated to give a multi-round posterior evolution on this instance.
SIGMA_OBS = 0.15

# Simulation parameters
T_MAX = 60
N_SEEDS = 40

# Prior variants for BR-PEG.
# Correctly specified: uniform prior on full Theta (includes truth).
PRIOR_CORRECT = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) / 5.0
# Mis-specified: prior support excludes the truth.
_mis = np.ones(len(TYPE_WS))
_mis[TRUE_TYPE_IDX] = 0.0
PRIOR_MISSPEC = _mis / _mis.sum()

# Horizon markers reported in the summary table
T_MARKERS = [10, 20, 40, T_MAX]


# ---------------------------------------------------------------------------
# Precomputed lookup table
# ---------------------------------------------------------------------------

def build_lookup() -> dict:
    """For each (type_idx, eps_M_idx), compute H's best-response and the
    resulting (eps_H, u_H, A_H, sigma) tuple. All types share n_H=N_H, so
    the mechanism solves are shared across types via the in-memory lru
    cache in `game.mechanism`."""
    n_types = len(TYPE_WS)
    n_eps_M = len(EPS_M_GRID)
    eps_H = np.zeros((n_types, n_eps_M))
    u_H = np.zeros((n_types, n_eps_M))
    A_H = np.zeros((n_types, n_eps_M))
    sigma = np.zeros((n_types, n_eps_M))
    feas = np.zeros((n_types, n_eps_M), dtype=bool)

    for k, w in enumerate(TYPE_WS):
        for j, eps_M in enumerate(EPS_M_GRID):
            best_eps, res = H_best_response(
                w_H=float(w), eps_M=float(eps_M),
                n_H=N_H, n_M=N_M,
                eps_grid=EPS_H_GRID,
                expected_batch=EXPECTED_BATCH, steps=STEPS, delta=DELTA,
            )
            if not res.get("feasible", False):
                continue
            eps_H[k, j] = best_eps
            A_H[k, j] = res["A_H"]
            sigma[k, j] = res["sigma"]
            u_H[k, j] = value_fn(res["sigma"]) - w * res["A_H"]
            feas[k, j] = True
        print(f"[lookup] type w={w:.2f} done "
              f"(eps_M range feasible: {feas[k].sum()}/{n_eps_M})")
    return dict(eps_H=eps_H, u_H=u_H, A_H=A_H, sigma=sigma, feas=feas,
                type_ws=np.array(TYPE_WS), eps_M_grid=EPS_M_GRID.copy(),
                eps_H_grid=EPS_H_GRID.copy())


def get_lookup() -> dict:
    if LOOKUP_CACHE.exists():
        with LOOKUP_CACHE.open("rb") as f:
            lu = pickle.load(f)
        # Sanity check the cached lookup matches the current instance.
        if (list(lu["type_ws"]) == TYPE_WS
                and len(lu["eps_M_grid"]) == K_M
                and abs(lu["eps_M_grid"][0] - EPS_M_GRID[0]) < 1e-9
                and abs(lu["eps_M_grid"][-1] - EPS_M_GRID[-1]) < 1e-9):
            print(f"[lookup] loaded cache: {LOOKUP_CACHE}")
            return lu
        print(f"[lookup] cache stale; rebuilding")
    t0 = time.time()
    print(f"[lookup] building fresh...")
    lu = build_lookup()
    LOOKUP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with LOOKUP_CACHE.open("wb") as f:
        pickle.dump(lu, f)
    print(f"[lookup] built in {time.time()-t0:.1f}s -> {LOOKUP_CACHE}")
    return lu


# ---------------------------------------------------------------------------
# Policy implementations (all cheap; no mechanism solves)
# ---------------------------------------------------------------------------

def posterior_greedy_arm(posterior: np.ndarray, A_H_pred: np.ndarray,
                         feas: np.ndarray) -> int:
    """Argmax over eps_M arms of E_{theta ~ posterior} [A_H | theta].
    Types with infeasible arms are excluded from that arm's average
    (rescaled by the mass of types with a feasible arm)."""
    K_M_ = A_H_pred.shape[1]
    scores = np.full(K_M_, -np.inf)
    for j in range(K_M_):
        w_j = posterior * feas[:, j]
        Z = w_j.sum()
        if Z <= 0.0:
            continue
        scores[j] = (w_j * A_H_pred[:, j]).sum() / Z
    return int(np.argmax(scores))


def posterior_update(posterior: np.ndarray, arm: int,
                     eps_H_obs: float, u_H_obs: float,
                     eps_H_pred: np.ndarray, u_H_pred: np.ndarray,
                     feas: np.ndarray, sigma_obs: float) -> np.ndarray:
    """Bayesian update of the type posterior. Likelihood is Gaussian on
    log eps_H and linear u_H with width sigma_obs, matching the
    (correctly-specified) observation noise process."""
    new_post = np.zeros_like(posterior)
    for k in range(len(posterior)):
        if posterior[k] < 1e-12 or not feas[k, arm]:
            continue
        d_eps = np.log(max(eps_H_pred[k, arm], 1e-6)) - np.log(max(eps_H_obs, 1e-6))
        d_u = u_H_pred[k, arm] - u_H_obs
        log_lik = -0.5 * ((d_eps / sigma_obs) ** 2 + (d_u / sigma_obs) ** 2)
        new_post[k] = posterior[k] * np.exp(log_lik)
    Z = new_post.sum()
    if Z > 0.0:
        return new_post / Z
    return posterior.copy()


def simulate_brpeg(prior: np.ndarray, lu: dict, rng: np.random.Generator,
                   T: int, arm_oracle: int, A_H_oracle: float,
                   sigma_obs: float) -> np.ndarray:
    """One trajectory of BR-PEG. Returns per-round regret r_t."""
    posterior = prior.copy()
    A_H_pred = lu["A_H"]; eps_H_pred = lu["eps_H"]
    u_H_pred = lu["u_H"]; feas = lu["feas"]
    true_A_H = A_H_pred[TRUE_TYPE_IDX]
    true_eps_H = eps_H_pred[TRUE_TYPE_IDX]
    true_u_H = u_H_pred[TRUE_TYPE_IDX]
    r = np.zeros(T)
    for t in range(T):
        arm = posterior_greedy_arm(posterior, A_H_pred, feas)
        # Observed reward = A_H at true type for this arm
        A_H_t = float(true_A_H[arm])
        r[t] = A_H_oracle - A_H_t
        # Observation model: log-Gaussian noise on eps_H, additive Gaussian on u_H
        eps_H_obs = float(true_eps_H[arm]) * float(np.exp(sigma_obs * rng.standard_normal()))
        u_H_obs = float(true_u_H[arm]) + float(sigma_obs * rng.standard_normal())
        posterior = posterior_update(
            posterior, arm, eps_H_obs, u_H_obs,
            eps_H_pred, u_H_pred, feas, sigma_obs,
        )
    return r


def simulate_exp3(lu: dict, rng: np.random.Generator, T: int,
                  arm_oracle: int, A_H_oracle: float) -> np.ndarray:
    """EXP3 (Auer et al. 2002) over the eps_M arms, using A_H as reward
    (bounded above by A_H_ub = 0.5, then normalised to [0,1]).
    Anytime version with per-round eta_t = sqrt(log K / (t*K)) to avoid
    a horizon dependency."""
    K = len(EPS_M_GRID)
    true_A_H = lu["A_H"][TRUE_TYPE_IDX]
    true_feas = lu["feas"][TRUE_TYPE_IDX]
    R = np.zeros(K)  # cumulative importance-weighted rewards
    A_H_ub = 0.5     # empirical upper bound on the reward scale
    r = np.zeros(T)
    for t in range(1, T + 1):
        eta = np.sqrt(np.log(K) / (t * K))
        w = np.exp(eta * (R - R.max()))
        # Mask out perpetually infeasible arms so EXP3 does not put mass
        # on them.
        w = w * true_feas
        p = w / w.sum()
        # Mixing with uniform to keep exploration alive (gamma=1/sqrt(t))
        gamma = min(1.0, np.sqrt(K / t))
        p = (1 - gamma) * p + gamma * (true_feas / true_feas.sum())
        arm = int(rng.choice(K, p=p))
        reward = float(true_A_H[arm]) / A_H_ub
        r[t - 1] = A_H_oracle - float(true_A_H[arm])
        # Importance-weighted update
        R[arm] += reward / p[arm]
    return r


def simulate_uniform_belief(lu: dict, T: int, arm_uniform: int,
                            A_H_oracle: float) -> np.ndarray:
    A_H_t = float(lu["A_H"][TRUE_TYPE_IDX, arm_uniform])
    return np.full(T, A_H_oracle - A_H_t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    lu = get_lookup()
    A_H = lu["A_H"]; feas = lu["feas"]

    # Oracle: arm that maximises A_H under the *true* type
    scores_true = np.where(feas[TRUE_TYPE_IDX], A_H[TRUE_TYPE_IDX], -np.inf)
    arm_oracle = int(np.argmax(scores_true))
    A_H_oracle = float(scores_true[arm_oracle])
    eps_M_oracle = float(EPS_M_GRID[arm_oracle])

    # Uniform-belief-constant M: argmax over eps_M of E_{theta ~ U(Theta)} A_H
    # (single-round expected argmax under the uniform prior; a *stationary*
    # policy that never updates).
    uni_scores = np.full(len(EPS_M_GRID), -np.inf)
    for j in range(len(EPS_M_GRID)):
        m = feas[:, j]
        if m.any():
            uni_scores[j] = float(A_H[m, j].mean())
    arm_uniform = int(np.argmax(uni_scores))
    A_H_uniform_per_round = float(A_H[TRUE_TYPE_IDX, arm_uniform])

    print(f"[oracle]  arm={arm_oracle}  eps_M={eps_M_oracle:.3f}  "
          f"A_H*={A_H_oracle:.4f}")
    print(f"[uniform] arm={arm_uniform}  eps_M={EPS_M_GRID[arm_uniform]:.3f}  "
          f"A_H={A_H_uniform_per_round:.4f}")
    print(f"[types]   feasible-arm counts: "
          f"{[int(feas[k].sum()) for k in range(len(TYPE_WS))]}")

    # Trace lookup values around the truth to confirm ambiguity.
    print(f"\n[ambiguity check] A_H(theta, oracle_arm={arm_oracle}) "
          f"across types:")
    for k, w in enumerate(TYPE_WS):
        eh = lu["eps_H"][k, arm_oracle]
        uh = lu["u_H"][k, arm_oracle]
        ah = lu["A_H"][k, arm_oracle]
        star = "  <- truth" if k == TRUE_TYPE_IDX else ""
        print(f"  w={w:.2f}  eps_H={eh:6.3f}  u_H={uh:+.4f}  A_H={ah:.4f}{star}")

    # -- Multi-seed simulation --
    T = T_MAX
    R_brpeg_correct = np.zeros((N_SEEDS, T))
    R_brpeg_misspec = np.zeros((N_SEEDS, T))
    R_exp3 = np.zeros((N_SEEDS, T))
    R_uniform = np.zeros((N_SEEDS, T))

    for s in range(N_SEEDS):
        # Independent rng per seed and per policy: paired noise is not
        # required and independent rngs keep the estimators unbiased.
        rng_correct = np.random.default_rng(1000 + s)
        rng_misspec = np.random.default_rng(2000 + s)
        rng_exp3 = np.random.default_rng(3000 + s)
        r_c = simulate_brpeg(PRIOR_CORRECT, lu, rng_correct,
                             T, arm_oracle, A_H_oracle, SIGMA_OBS)
        r_m = simulate_brpeg(PRIOR_MISSPEC, lu, rng_misspec,
                             T, arm_oracle, A_H_oracle, SIGMA_OBS)
        r_e = simulate_exp3(lu, rng_exp3, T, arm_oracle, A_H_oracle)
        r_u = simulate_uniform_belief(lu, T, arm_uniform, A_H_oracle)
        R_brpeg_correct[s] = np.cumsum(r_c)
        R_brpeg_misspec[s] = np.cumsum(r_m)
        R_exp3[s] = np.cumsum(r_e)
        R_uniform[s] = np.cumsum(r_u)

    def mean_se(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = R.mean(axis=0)
        se = R.std(axis=0, ddof=1) / np.sqrt(R.shape[0])
        return mu, se

    mu_c, se_c = mean_se(R_brpeg_correct)
    mu_m, se_m = mean_se(R_brpeg_misspec)
    mu_e, se_e = mean_se(R_exp3)
    mu_u, se_u = mean_se(R_uniform)

    # -- Summary text --
    lines = [
        "# T16 / H8 BR-PEG regret summary (redesigned)",
        f"# instance: 5 overlapping types w in {TYPE_WS}",
        f"#           n_H=n_M={N_H}, sigma_obs={SIGMA_OBS}, T={T}, "
        f"N_SEEDS={N_SEEDS}",
        f"# eps_M grid ({K_M} arms, log-spaced [{EPS_M_MIN},{EPS_M_MAX}]): "
        + " ".join(f"{e:.2f}" for e in EPS_M_GRID),
        f"# oracle arm={arm_oracle}, eps_M*={eps_M_oracle:.3f}, "
        f"A_H*={A_H_oracle:.4f}",
        f"# uniform-belief arm={arm_uniform}, "
        f"eps_M_uni={EPS_M_GRID[arm_uniform]:.3f}, "
        f"A_H_uni={A_H_uniform_per_round:.4f}",
        f"# prior (correct)     = {PRIOR_CORRECT.round(3).tolist()}",
        f"# prior (misspecified)= {PRIOR_MISSPEC.round(3).tolist()}",
        "",
        "# Mean cumulative regret R(T) +/- 1 SE over seeds",
        f"# {'T':>4}  {'BR-PEG(correct)':>22} {'BR-PEG(mis)':>18} "
        f"{'EXP3':>18} {'uniform':>18}",
    ]
    for Tm in T_MARKERS:
        idx = Tm - 1
        lines.append(
            f"  {Tm:>4}  {mu_c[idx]:>10.3f} +/- {se_c[idx]:>5.3f}    "
            f"{mu_m[idx]:>10.3f} +/- {se_m[idx]:>5.3f}    "
            f"{mu_e[idx]:>10.3f} +/- {se_e[idx]:>5.3f}    "
            f"{mu_u[idx]:>10.3f} +/- {se_u[idx]:>5.3f}"
        )
    SUMMARY.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {SUMMARY}")

    # -----------------------------------------------------------------
    # Figure. Single panel, no title, style-matched to fig_h5_v_shape.
    # -----------------------------------------------------------------
    apply_paper_style()
    fig, ax = plt.subplots(1, 1, figsize=column_size(rows=1, aspect=0.75))

    ts = np.arange(1, T + 1)

    def band(mu, se, color, ls, lw, label, zorder):
        ax.fill_between(ts, mu - se, mu + se,
                        color=color, alpha=0.15, linewidth=0, zorder=zorder)
        ax.plot(ts, mu, ls, color=color, lw=lw, label=label, zorder=zorder + 1)

    # Uniform-belief: exactly linear -- plot first behind others
    band(mu_u, se_u, ACCENT_GREY, "-", 1.2,
         "uniform-belief constant  ($O(T)$)", zorder=1)
    # Misspecified BR-PEG: linear, locks onto the wrong type's arm
    band(mu_m, se_m, PRIMARY_BLUE, "--", 1.2,
         "BR-PEG (mis-specified prior)", zorder=3)
    # EXP3: sub-linear asymptotically, worst over this horizon
    band(mu_e, se_e, PRIMARY_ORANGE, "-", 1.4,
         "EXP3 bandit  (no prior)", zorder=5)
    # Correctly-specified BR-PEG: sub-linear, plateaus
    band(mu_c, se_c, PRIMARY_BLUE, "-", 1.6,
         "BR-PEG (correct prior)", zorder=7)

    ax.set_xlim(1, T)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"horizon $T$")
    ax.set_ylabel(r"cumulative regret $R(T)$")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left",
              handletextpad=0.5, labelspacing=0.3)

    plt.tight_layout()
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF)
    fig.savefig(FIG_SVG)
    plt.close(fig)
    print(f"wrote {FIG_PDF}")
    print(f"wrote {FIG_SVG}")


if __name__ == "__main__":
    main()
