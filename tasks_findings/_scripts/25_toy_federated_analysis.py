"""T25 -- LiRA analysis for the toy federated n=3 Privacy Externality Game.

Companion of tasks_findings/25_toy_federated_n3_shadow_model.md.

Design principle: the LiRA scoring path is n-agnostic. We reuse T24's
implementation verbatim (`tasks_findings/_scripts/24_lira_analysis.py`) for
`_logit_score`, `_lira_offline`, `_lira_online`, and the pure-NE search
`_find_pure_ne`; only the mechanism grid, the per-silo aggregation, and the
NE enumeration on a K x K x K cube differ.

Three modes, mirroring T24:

    python 25_toy_federated_analysis.py --build-grid
        Solve the DP-SGD mechanism for each of the 27 profiles on the
        3x3x3 grid, cache (sigma, p_1, p_2, p_3, feasible) at
        game_experiments/_cache_T25/mechanism_grid.pkl. CPU-only, <2 min,
        safe on the login node.

    python 25_toy_federated_analysis.py --score-profile <pid> [--lira-variant ...]
        Called from the Slurm array job after 04_mia_pergroup.py has
        finished writing federated shadow-model logits. Writes:
            _cache_T25/n3_credit_card/<pid>/A_measured.npy   # (3,)
            _cache_T25/n3_credit_card/<pid>/V_measured.npy   # scalar
            _cache_T25/n3_credit_card/<pid>/lira_scores.npz  # per-record

    python 25_toy_federated_analysis.py --collect
        Aggregate over 0..26, build measured and PLD-analytical payoff
        cubes, enumerate pure NE on the 3x3x3 tensor, and dump summary to
        _cache_T25/T25_summary.pkl for the paper-figure step.

CLAUDE.md Rule 5: does NOT modify the mechanism-solver or utility form.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tasks_findings" / "_scripts"))

# Reuse T24 verbatim -- the scoring math is silo-count-agnostic.
import importlib.util as _iu
_T24_PATH = REPO / "tasks_findings" / "_scripts" / "24_lira_analysis.py"
_spec = _iu.spec_from_file_location("t24_lira", _T24_PATH)
_t24 = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_t24)  # type: ignore[union-attr]

_logit_score = _t24._logit_score
_lira_offline = _t24._lira_offline
_lira_online = _t24._lira_online
_find_pure_ne_bimatrix = _t24._find_pure_ne  # renamed to keep bimatrix scope clear.

from game.advantage import mia_advantage_subsampled_gaussian  # noqa: E402
from game.mechanism import solve_mechanism  # noqa: E402


# ---------------------------------------------------------------------------
# Grid / mechanism configuration -- MUST match the YAML config and the
# Slurm script.
# ---------------------------------------------------------------------------
EPS_GRID: np.ndarray = np.array([2.0, 8.0, 32.0])         # coarse toy grid, K=3.
K: int = len(EPS_GRID)
N_PLAYERS: int = 3
GROUP_SIZES: Tuple[int, int, int] = (10000, 10000, 10000)
EXPECTED_BATCH: int = 128
EPOCHS: int = 60
STEPS: int = EPOCHS * (sum(GROUP_SIZES) // EXPECTED_BATCH)   # 60 * 234 = 14040.
DELTA: float = 1e-12
CACHE = REPO / "game_experiments" / "_cache_T25"
CACHE_PROFILES = CACHE / "n3_credit_card"


# ---------------------------------------------------------------------------
# Step 1: mechanism grid.
# ---------------------------------------------------------------------------
def build_mechanism_grid() -> List[Dict]:
    """Solve (sigma, p_1, p_2, p_3) for every profile on the 3x3x3 grid.

    Row order is pid = i * K^2 + j * K + k with (i, j, k) indexing
    (eps_1, eps_2, eps_3). Infeasible profiles (any p_g below the
    solver's participation floor) are flagged and reported.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    grid: List[Dict] = []
    for pid, (i, j, k) in enumerate(product(range(K), repeat=N_PLAYERS)):
        e1, e2, e3 = EPS_GRID[i], EPS_GRID[j], EPS_GRID[k]
        r = solve_mechanism(
            eps_target=[float(e1), float(e2), float(e3)],
            group_sizes=list(GROUP_SIZES),
            expected_batch_size=EXPECTED_BATCH,
            steps=STEPS,
            delta=DELTA,
        )
        grid.append(
            dict(
                pid=pid,
                ijk=(int(i), int(j), int(k)),
                eps=(float(e1), float(e2), float(e3)),
                sigma=float(r.sigma),
                p1=float(r.sample_rates[0]),
                p2=float(r.sample_rates[1]),
                p3=float(r.sample_rates[2]),
                eps_realised=tuple(float(x) for x in r.eps_per_group),
                feasible=bool(r.feasible),
            )
        )
    out = CACHE / "mechanism_grid.pkl"
    with open(out, "wb") as f:
        pickle.dump(grid, f)
    n_feasible = sum(int(g["feasible"]) for g in grid)
    print(f"[build-grid] wrote {out} ({n_feasible}/{len(grid)} feasible profiles)")
    infeasible = [g["eps"] for g in grid if not g["feasible"]]
    if infeasible:
        print(f"[build-grid] infeasible profiles: {infeasible}")
    return grid


# ---------------------------------------------------------------------------
# Step 2: LiRA scoring for one profile.
# ---------------------------------------------------------------------------
def score_profile(pid: int, lira_variant: str = "offline-fixed-variance") -> None:
    """Load federated shadow-model logits, score LiRA per silo (3 silos).

    The federated pipeline writes one subfolder per shadow federation
    (each subfolder is a full FedSGD run with per-silo (sigma, p_g)
    noise). Each subfolder contains the same schema as T24:
        logits.npy   (n_records, n_augs, n_classes)
        keep.npy     (n_records,) bool
        test_acc.npy scalar
    plus the driver writes labels.npy and silo_id.npy (in {0, 1, 2})
    once at the profile-directory root.
    """
    pdir = CACHE_PROFILES / str(pid)
    if not pdir.exists():
        raise FileNotFoundError(f"Profile dir {pdir} missing (Slurm array didn't run?)")

    shadow_dirs = sorted(d for d in pdir.iterdir() if d.is_dir())
    if not shadow_dirs:
        raise FileNotFoundError(f"No shadow-model outputs under {pdir}")

    logits = np.stack([np.load(d / "logits.npy") for d in shadow_dirs])  # (S, N, A, C)
    if logits.ndim == 4:
        logits = logits.mean(axis=2)
    keep = np.stack([np.load(d / "keep.npy") for d in shadow_dirs])       # (S, N)
    accs = np.array([float(np.load(d / "test_acc.npy")) for d in shadow_dirs])

    labels = np.load(pdir / "labels.npy")     # (N,) int
    silo_id = np.load(pdir / "silo_id.npy")   # (N,) in {0, 1, 2}

    phi = _logit_score(logits, labels.astype(int))
    if lira_variant.startswith("online"):
        A_j = _lira_online(phi, keep, fix_variance=lira_variant.endswith("fixed-variance"))
    else:
        A_j = _lira_offline(phi, keep, fix_variance=lira_variant.endswith("fixed-variance"))

    A_silo = np.array([A_j[silo_id == g].mean() for g in range(N_PLAYERS)])
    V_meas = float(accs.mean())
    np.save(pdir / "A_measured.npy", A_silo)
    np.save(pdir / "V_measured.npy", np.array(V_meas))
    np.savez(pdir / "lira_scores.npz",
             A_per_record=A_j, silo_id=silo_id, variant=lira_variant,
             n_shadows=len(shadow_dirs), V_measured=V_meas)
    print(f"[score-profile pid={pid} variant={lira_variant}] "
          f"A=({A_silo[0]:.4f}, {A_silo[1]:.4f}, {A_silo[2]:.4f}) V={V_meas:.4f} "
          f"n_shadows={len(shadow_dirs)}")


# ---------------------------------------------------------------------------
# Step 3: pure-NE search on the K x K x K payoff cube.
# ---------------------------------------------------------------------------
def _find_pure_ne_cube(U: np.ndarray) -> List[Tuple[int, int, int]]:
    """Enumerate pure Nash equilibria on a K x K x K x N_PLAYERS payoff cube.

    U[i, j, k, g] is player g's payoff under joint action (i, j, k).
    (i, j, k) is a NE iff no player g gains by unilaterally deviating.
    """
    assert U.shape == (K, K, K, N_PLAYERS), f"unexpected U shape {U.shape}"
    ne: List[Tuple[int, int, int]] = []
    for i, j, k in product(range(K), repeat=N_PLAYERS):
        if (U[i, j, k, 0] >= U[:, j, k, 0].max() - 1e-9
                and U[i, j, k, 1] >= U[i, :, k, 1].max() - 1e-9
                and U[i, j, k, 2] >= U[i, j, :, 2].max() - 1e-9):
            ne.append((i, j, k))
    return ne


def collect_and_score(w: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict:
    grid_path = CACHE / "mechanism_grid.pkl"
    with open(grid_path, "rb") as f:
        grid = pickle.load(f)

    A_meas = np.full((K, K, K, N_PLAYERS), np.nan)
    A_pld = np.full((K, K, K, N_PLAYERS), np.nan)
    V_meas = np.full((K, K, K), np.nan)

    for row in grid:
        i, j, k = row["ijk"]
        pdir = CACHE_PROFILES / str(row["pid"])
        if not row["feasible"]:
            continue
        try:
            A_meas[i, j, k] = np.load(pdir / "A_measured.npy")
            V_meas[i, j, k] = float(np.load(pdir / "V_measured.npy"))
        except FileNotFoundError:
            pass
        A_pld[i, j, k, 0] = mia_advantage_subsampled_gaussian(row["sigma"], row["p1"], STEPS)
        A_pld[i, j, k, 1] = mia_advantage_subsampled_gaussian(row["sigma"], row["p2"], STEPS)
        A_pld[i, j, k, 2] = mia_advantage_subsampled_gaussian(row["sigma"], row["p3"], STEPS)

    # Payoff cubes: u_g = V - w_g * A_g.
    U_meas = np.stack([V_meas - w[g] * A_meas[..., g] for g in range(N_PLAYERS)], axis=-1)
    U_pld = np.stack([V_meas - w[g] * A_pld[..., g] for g in range(N_PLAYERS)], axis=-1)
    ne_meas = _find_pure_ne_cube(U_meas)
    ne_pld = _find_pure_ne_cube(U_pld)

    summary = dict(
        A_meas=A_meas, A_pld=A_pld, V_meas=V_meas,
        U_meas=U_meas, U_pld=U_pld,
        ne_meas=ne_meas, ne_pld=ne_pld,
        eps_grid=EPS_GRID, weights=w, steps=STEPS, n_players=N_PLAYERS,
    )
    with open(CACHE / "T25_summary.pkl", "wb") as f:
        pickle.dump(summary, f)
    print(f"[collect] measured NE cells (i,j,k): {ne_meas}")
    print(f"[collect] PLD-analytical NE cells:  {ne_pld}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-grid", action="store_true")
    ap.add_argument("--score-profile", type=int, default=None,
                    help="Score a single profile pid in [0, 26].")
    ap.add_argument("--lira-variant", default="offline-fixed-variance",
                    choices=["offline-fixed-variance", "offline",
                             "online-fixed-variance", "online"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--w", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    args = ap.parse_args()

    if args.build_grid:
        build_mechanism_grid()
    if args.score_profile is not None:
        score_profile(args.score_profile, args.lira_variant)
    if args.collect:
        collect_and_score(tuple(args.w))


if __name__ == "__main__":
    main()
