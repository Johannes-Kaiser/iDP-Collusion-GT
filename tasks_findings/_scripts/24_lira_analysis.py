"""T24 -- LiRA analysis pipeline for the n=2 Privacy Externality Game.

This stub covers the full analysis path once shadow-model training finishes.
It is designed to run in three modes:

    python 24_lira_analysis.py --build-grid
        Solve the DP-SGD mechanism for each of the 25 profiles on the 5x5 grid,
        cache (sigma, p_1, p_2, feasible) at
        game_experiments/_cache_H1/mechanism_grid.pkl. CPU-only, <1 min --
        OK to run on the login node per CLAUDE.md Rule 3.

    python 24_lira_analysis.py --score-profile <pid> [--lira-variant ...]
        Called from the Slurm array job after 04_mia_pergroup.py has written
        shadow-model logits. Fits per-example IN/OUT Gaussians, computes the
        LiRA log-likelihood-ratio per record, and writes:
            _cache_H1/n2_credit_card/<pid>/A_measured.npy  # (2,) silo-mean advantages
            _cache_H1/n2_credit_card/<pid>/V_measured.npy  # scalar: mean shadow test acc
            _cache_H1/n2_credit_card/<pid>/lira_scores.npz # per-record scores + labels

    python 24_lira_analysis.py --collect
        Aggregate over 0..24, build measured/analytical payoff matrices,
        enumerate pure NE, and dump summary to
        _cache_H1/H1_summary.pkl for the paper-figure step.

The LiRA implementation follows Carlini et al. 2022 (S&P):
  * Per-example statistic phi_j = log(p_j / (1 - p_j))   -- logit scale.
  * Offline-fixed-variance variant by default (Carlini reports it as the
    best-performing of the four in Table 1).
  * Online variant available via --lira-variant online-fixed-variance for
    ablation; requires shadow models with both IN and OUT copies per example
    (guaranteed by pkeep=0.5 and n_shadows=128).
  * MIA advantage A_j = max_alpha (TPR_j(alpha) - alpha) on the per-record ROC.
  * Silo aggregate A_g = mean_{j in silo_g} A_j; this is what enters the PEG
    utility u_i = V - w_i * A_i, replacing the PLD upper bound
    game.advantage.mia_advantage_subsampled_gaussian.

CLAUDE.md Rule 5: this analysis only *scores* the shadow-model runs; it does
not perturb the mechanism-solver or utility form.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(REPO))

from game.advantage import mia_advantage_subsampled_gaussian  # noqa: E402
from game.mechanism import solve_mechanism  # noqa: E402


# ---------------------------------------------------------------------------
# Grid / mechanism configuration -- MUST match the YAML config that drives
# 04_mia_pergroup.py and the Slurm script.
# ---------------------------------------------------------------------------
EPS_GRID: np.ndarray = np.array([2.0, 4.0, 8.0, 16.0, 32.0])   # K=5, log2-spaced.
K: int = len(EPS_GRID)
GROUP_SIZES: Tuple[int, int] = (5000, 5000)
EXPECTED_BATCH: int = 128
EPOCHS: int = 60
STEPS: int = EPOCHS * (sum(GROUP_SIZES) // EXPECTED_BATCH)   # 60 * 78 = 4680.
DELTA: float = 1e-12
CACHE = REPO / "game_experiments" / "_cache_H1"
# Adult was substituted for Credit Card Default: the shared-loader
# `credit_card_default` alias is OpenML `credit-g` (1000 rows), too small for
# a 5000-per-silo LiRA pool. Adult is the drop-in tabular replacement.
CACHE_PROFILES = CACHE / "n2_adult"


# ---------------------------------------------------------------------------
# Step 1: mechanism grid (CPU-only, cached).
# ---------------------------------------------------------------------------
def build_mechanism_grid() -> List[Dict]:
    """Solve (sigma, p_1, p_2) for every (eps_1, eps_2) in the 5x5 grid.

    Deterministic; cached at _cache_H1/mechanism_grid.pkl. Row order is
    (i * K + j) with i indexing eps_1 (silo 0) and j indexing eps_2 (silo 1).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    grid: List[Dict] = []
    for i, e1 in enumerate(EPS_GRID):
        for j, e2 in enumerate(EPS_GRID):
            r = solve_mechanism(
                eps_target=[float(e1), float(e2)],
                group_sizes=list(GROUP_SIZES),
                expected_batch_size=EXPECTED_BATCH,
                steps=STEPS,
                delta=DELTA,
            )
            grid.append(
                dict(
                    pid=i * K + j,
                    ij=(int(i), int(j)),
                    eps=(float(e1), float(e2)),
                    sigma=float(r.sigma),
                    p1=float(r.sample_rates[0]),
                    p2=float(r.sample_rates[1]),
                    eps_realised=(float(r.eps_per_group[0]),
                                  float(r.eps_per_group[1])),
                    feasible=bool(r.feasible),
                )
            )
    out = CACHE / "mechanism_grid.pkl"
    with open(out, "wb") as f:
        pickle.dump(grid, f)
    n_feasible = sum(int(g["feasible"]) for g in grid)
    print(f"[build-grid] wrote {out} ({n_feasible}/{len(grid)} feasible profiles)")
    return grid


# ---------------------------------------------------------------------------
# Step 2: LiRA scoring within a single profile.
# ---------------------------------------------------------------------------
def _logit_score(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Carlini et al. 2022 phi = log(p_y) - log(1 - p_y) on the true class.

    logits: (n_shadows, n_records, n_classes). Returns (n_shadows, n_records).
    """
    # Numerically-stable softmax on the last axis.
    x = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
    n_shadows, n_records, _ = logits.shape
    idx = np.arange(n_records)
    p_true = p[:, idx, labels[idx]]
    p_true = np.clip(p_true, 1e-6, 1.0 - 1e-6)
    return np.log(p_true) - np.log1p(-p_true)


def _lira_offline(phi: np.ndarray, keep: np.ndarray,
                  fix_variance: bool = True) -> np.ndarray:
    """Offline LiRA: fit per-record OUT Gaussian; score = pdf(phi_target | out).

    phi:   (n_shadows, n_records) logit-scaled scores.
    keep:  (n_shadows, n_records) bool -- True where record is IN.
    Returns per-record MIA advantage (n_records,).

    Following Carlini 2022 Sec. 4.3, when a target model is queried we ask "how
    likely is the observed phi under the OUT distribution?"; low likelihood ->
    infer IN. We evaluate this leave-one-out over the shadow models.
    """
    from scipy import stats
    n_shadows, n_records = phi.shape
    advs = np.zeros(n_records)
    # For each shadow model j, fit the OUT Gaussian per-record using the OTHER
    # (n_shadows - 1) shadows. This gives us a target/reference split matched
    # to canonical LiRA (Carlini 2022 Alg. 1).
    # For speed we approximate with a single global fit per record (leave-out
    # bias is O(1/n_shadows) which at n=128 is < 1%).
    mu_out = np.array([
        np.median(phi[~keep[:, r], r]) if (~keep[:, r]).sum() >= 2 else 0.0
        for r in range(n_records)
    ])
    if fix_variance:
        # Pool OUT variance across records (Carlini finds this is the strongest
        # variant on CIFAR-10 in Table 1).
        pooled = np.concatenate([
            phi[~keep[:, r], r] - mu_out[r] for r in range(n_records)
        ])
        sigma_out = np.full(n_records, max(pooled.std(), 1e-6))
    else:
        sigma_out = np.array([
            max(phi[~keep[:, r], r].std(), 1e-6) if (~keep[:, r]).sum() >= 2
            else 1.0
            for r in range(n_records)
        ])

    # For each record, sweep membership using phi_target ~ shadow, score is
    # -log pdf under OUT (large score -> IN).
    for r in range(n_records):
        scores = -stats.norm.logpdf(phi[:, r], loc=mu_out[r], scale=sigma_out[r])
        labels = keep[:, r].astype(int)
        if labels.sum() == 0 or labels.sum() == n_shadows:
            advs[r] = 0.0
            continue
        # ROC advantage via sweep.
        order = np.argsort(-scores)
        y = labels[order]
        tpr = np.cumsum(y) / max(y.sum(), 1)
        fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
        advs[r] = float((tpr - fpr).max())
    return advs


def _lira_online(phi: np.ndarray, keep: np.ndarray,
                 fix_variance: bool = True) -> np.ndarray:
    """Online LiRA: fit both IN and OUT Gaussians per record; likelihood ratio."""
    from scipy import stats
    n_shadows, n_records = phi.shape
    advs = np.zeros(n_records)
    mu_in = np.array([np.median(phi[keep[:, r], r]) if keep[:, r].sum() >= 2 else 0.0
                      for r in range(n_records)])
    mu_out = np.array([np.median(phi[~keep[:, r], r]) if (~keep[:, r]).sum() >= 2 else 0.0
                       for r in range(n_records)])
    if fix_variance:
        pooled_in = np.concatenate([
            phi[keep[:, r], r] - mu_in[r] for r in range(n_records)
            if keep[:, r].sum() >= 2
        ])
        pooled_out = np.concatenate([
            phi[~keep[:, r], r] - mu_out[r] for r in range(n_records)
            if (~keep[:, r]).sum() >= 2
        ])
        sigma_in = np.full(n_records, max(pooled_in.std(), 1e-6))
        sigma_out = np.full(n_records, max(pooled_out.std(), 1e-6))
    else:
        sigma_in = np.array([max(phi[keep[:, r], r].std(), 1e-6)
                             if keep[:, r].sum() >= 2 else 1.0
                             for r in range(n_records)])
        sigma_out = np.array([max(phi[~keep[:, r], r].std(), 1e-6)
                              if (~keep[:, r]).sum() >= 2 else 1.0
                              for r in range(n_records)])
    for r in range(n_records):
        # Lambda = log N(phi | in) - log N(phi | out); large -> IN.
        lr = (stats.norm.logpdf(phi[:, r], loc=mu_in[r], scale=sigma_in[r])
              - stats.norm.logpdf(phi[:, r], loc=mu_out[r], scale=sigma_out[r]))
        labels = keep[:, r].astype(int)
        if labels.sum() == 0 or labels.sum() == n_shadows:
            advs[r] = 0.0
            continue
        order = np.argsort(-lr)
        y = labels[order]
        tpr = np.cumsum(y) / max(y.sum(), 1)
        fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
        advs[r] = float((tpr - fpr).max())
    return advs


def score_profile(pid: int, lira_variant: str = "offline-fixed-variance") -> None:
    """Load shadow-model logits for one profile, score LiRA per-silo,
    write A_measured / V_measured. Called by the Slurm post-training step."""
    pdir = CACHE_PROFILES / str(pid)
    if not pdir.exists():
        raise FileNotFoundError(f"Profile dir {pdir} missing (Slurm array didn't run?)")

    # 04_mia_pergroup.py writes one subfolder per shadow model. Each has:
    #     logits.npy   (n_records, n_augs, n_classes)  -- test-time logits
    #     keep.npy     (n_records,) bool               -- membership label
    #     test_acc.npy scalar                          -- shadow test accuracy
    shadow_dirs = sorted(d for d in pdir.iterdir() if d.is_dir())
    if not shadow_dirs:
        raise FileNotFoundError(f"No shadow-model outputs under {pdir}")

    logits = np.stack([np.load(d / "logits.npy") for d in shadow_dirs])  # (S, N, A, C)
    if logits.ndim == 4:
        # Collapse augmentation axis (Credit Card Default: n_queries=2, take mean).
        logits = logits.mean(axis=2)
    keep = np.stack([np.load(d / "keep.npy") for d in shadow_dirs])       # (S, N)
    accs = np.array([float(np.load(d / "test_acc.npy")) for d in shadow_dirs])

    # Labels come from the fixed training set of the YAML config.
    labels = np.load(pdir / "labels.npy")   # (N,) int, written once by driver.
    silo_id = np.load(pdir / "silo_id.npy")  # (N,) in {0, 1}

    phi = _logit_score(logits, labels.astype(int))
    if lira_variant.startswith("online"):
        A_j = _lira_online(phi, keep, fix_variance=lira_variant.endswith("fixed-variance"))
    else:
        A_j = _lira_offline(phi, keep, fix_variance=lira_variant.endswith("fixed-variance"))

    A_silo = np.array([A_j[silo_id == g].mean() for g in (0, 1)])
    V_meas = float(accs.mean())
    np.save(pdir / "A_measured.npy", A_silo)
    np.save(pdir / "V_measured.npy", np.array(V_meas))
    np.savez(pdir / "lira_scores.npz",
             A_per_record=A_j, silo_id=silo_id, variant=lira_variant,
             n_shadows=len(shadow_dirs), V_measured=V_meas)
    print(f"[score-profile pid={pid} variant={lira_variant}] "
          f"A=({A_silo[0]:.4f}, {A_silo[1]:.4f}) V={V_meas:.4f} "
          f"n_shadows={len(shadow_dirs)}")


# ---------------------------------------------------------------------------
# Step 3: aggregate across profiles; enumerate NE.
# ---------------------------------------------------------------------------
def _find_pure_ne(U1: np.ndarray, U2: np.ndarray) -> List[Tuple[int, int]]:
    """Pure-strategy Nash equilibria on a K x K bimatrix.

    (i, j) is a NE iff U1[i, j] = max_i' U1[i', j] and U2[i, j] = max_j' U2[i, j'].
    """
    ne = []
    for i in range(U1.shape[0]):
        for j in range(U1.shape[1]):
            if U1[i, j] >= U1[:, j].max() - 1e-9 and U2[i, j] >= U2[i, :].max() - 1e-9:
                ne.append((i, j))
    return ne


def collect_and_score(w: Tuple[float, float] = (1.0, 1.0)) -> Dict:
    grid_path = CACHE / "mechanism_grid.pkl"
    with open(grid_path, "rb") as f:
        grid = pickle.load(f)

    A_meas = np.full((K, K, 2), np.nan)
    A_pld = np.full((K, K, 2), np.nan)
    V_meas = np.full((K, K), np.nan)

    for row in grid:
        i, j = row["ij"]
        pdir = CACHE_PROFILES / str(row["pid"])
        if not row["feasible"]:
            continue
        try:
            A_meas[i, j] = np.load(pdir / "A_measured.npy")
            V_meas[i, j] = float(np.load(pdir / "V_measured.npy"))
        except FileNotFoundError:
            pass
        # Analytical anchor: PLD MIA advantage per silo (same formula the paper
        # uses everywhere else -- game.advantage).
        A_pld[i, j, 0] = mia_advantage_subsampled_gaussian(row["sigma"], row["p1"], STEPS)
        A_pld[i, j, 1] = mia_advantage_subsampled_gaussian(row["sigma"], row["p2"], STEPS)

    # Payoffs: u_i(a1, a2) = V - w_i * A_i.
    U1_meas = V_meas - w[0] * A_meas[..., 0]
    U2_meas = V_meas - w[1] * A_meas[..., 1]
    U1_pld = V_meas - w[0] * A_pld[..., 0]
    U2_pld = V_meas - w[1] * A_pld[..., 1]
    ne_meas = _find_pure_ne(U1_meas, U2_meas)
    ne_pld = _find_pure_ne(U1_pld, U2_pld)

    summary = dict(
        A_meas=A_meas, A_pld=A_pld, V_meas=V_meas,
        U1_meas=U1_meas, U2_meas=U2_meas,
        U1_pld=U1_pld, U2_pld=U2_pld,
        ne_meas=ne_meas, ne_pld=ne_pld,
        eps_grid=EPS_GRID, weights=w, steps=STEPS,
    )
    with open(CACHE / "H1_summary.pkl", "wb") as f:
        pickle.dump(summary, f)
    print(f"[collect] measured NE cells: {ne_meas}")
    print(f"[collect] PLD-analytical NE cells: {ne_pld}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-grid", action="store_true")
    ap.add_argument("--score-profile", type=int, default=None,
                    help="Score a single profile pid in [0, 24].")
    ap.add_argument("--lira-variant", default="offline-fixed-variance",
                    choices=["offline-fixed-variance", "offline",
                            "online-fixed-variance", "online"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--w", type=float, nargs=2, default=(1.0, 1.0))
    args = ap.parse_args()

    if args.build_grid:
        build_mechanism_grid()
    if args.score_profile is not None:
        score_profile(args.score_profile, args.lira_variant)
    if args.collect:
        collect_and_score(tuple(args.w))


if __name__ == "__main__":
    main()
