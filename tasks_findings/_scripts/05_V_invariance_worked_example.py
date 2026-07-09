"""
Worked numerical example for T05 -- Qualitative invariance of the
model-utility function V.

Reads the cached (A, sigma, feasible) tensor from
`game_experiments/_cache_v2/C1.pkl` -- the canonical 2-player, K=22 grid,
|G_1|=|G_2|=5000, w_i=1 -- and re-composes player utilities under a
family of admissible V. Since A_i and sigma are V-free by construction
(they are functionals of the mechanism's (C1)+(C2) solution alone), no
new mechanism-solver call is required.

We report, per V, the number of pure Nash equilibria on the grid, how
many sit off-diagonal, and the induced BR indices along the diagonal.
The point of the script is to show that the *qualitative* NE structure
(existence, asymmetric location) is preserved across an admissible
class -- V(sigma) = 1/(1+sigma^2) is a representative, not a load-
bearing choice.

Run:
    source .venv/bin/activate
    python3 tasks_findings/_scripts/05_V_invariance_worked_example.py

No Slurm, no cache write: this is a read-only recomposition.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "game_experiments" / "_cache_v2" / "C1.pkl"


def pure_ne(U: np.ndarray, feasible: np.ndarray) -> list[tuple[int, int]]:
    """Enumerate pure NE of a 2-player game with utility tensor U[i,j,p]."""
    K = U.shape[0]
    U0 = np.where(feasible, U[:, :, 0], -np.inf)
    U1 = np.where(feasible, U[:, :, 1], -np.inf)
    col_max = U0.max(axis=0)  # best response of player 0 for each j
    row_max = U1.max(axis=1)  # best response of player 1 for each i
    ne = []
    tol = 1e-12
    for i in range(K):
        for j in range(K):
            if not feasible[i, j]:
                continue
            if U0[i, j] >= col_max[j] - tol and U1[i, j] >= row_max[i] - tol:
                ne.append((i, j))
    return ne


def main() -> None:
    with CACHE.open("rb") as f:
        d = pickle.load(f)
    sweep = d["sweep"]
    A = sweep["A"]          # (K, K, 2), player-realised MIA advantage
    sigma = sweep["sigma"]  # (K, K), mechanism noise multiplier
    feas = sweep["feasible"]
    actions = sweep["actions"]
    K = len(actions)

    print(f"Grid: K={K}, eps in [{actions[0]:.3f}, {actions[-1]:.3f}]")
    print(f"Feasible cells: {int(feas.sum())}/{feas.size}")
    print(f"sigma range on feasible cells: "
          f"[{sigma[feas].min():.3f}, {sigma[feas].max():.3f}]")

    # Admissible V candidates: continuous, strictly decreasing, bounded on
    # the feasible sigma range. We include:
    #   * the canonical decaying-rational choice used in the paper,
    #   * an exponential (strictly convex),
    #   * a strictly concave choice, and
    #   * an affine (linear) choice.
    V_family: list[tuple[str, callable]] = [
        ("1/(1+s^2)  (canonical)", lambda s: 1.0 / (1.0 + s * s)),
        ("exp(-s)    (convex)  ", lambda s: np.exp(-s)),
        ("-log(1+s)  (concave) ", lambda s: -np.log1p(s)),
        ("-s         (linear)  ", lambda s: -s),
    ]

    # Note on affine transforms. A positive-affine transform of the
    # *whole* utility u_i -> a_i u_i + b_i (a_i>0) is well known to
    # preserve best-response sets and hence NE. A positive-affine
    # transform of V alone is NOT such a transform: it rescales the
    # trade-off between the shared V and the private A_i term and is
    # equivalent to a per-player rescaling of w_i. We therefore do not
    # bundle a V-affine test with the family above -- the proposition's
    # invariance class is the class of admissible V under a *fixed*
    # weight vector, not the joint class of (V, w) with weights adapted.

    header = f"{'V(sigma)':<30}  {'#NE':>4}  {'asym':>4}  {'sym':>4}"
    print()
    print(header)
    print("-" * len(header))

    canonical_ne: list[tuple[int, int]] | None = None
    for label, V in V_family:
        Vs = V(sigma)
        U = np.stack([Vs - A[:, :, 0], Vs - A[:, :, 1]], axis=-1)
        nes = pure_ne(U, feas)
        n_asym = sum(1 for (i, j) in nes if i != j)
        n_sym = sum(1 for (i, j) in nes if i == j)
        print(f"{label:<30}  {len(nes):>4}  {n_asym:>4}  {n_sym:>4}")
        if canonical_ne is None:
            canonical_ne = nes
    print()

    # Sanity check: positive-affine transform of the *whole* utility
    # (u_i -> a_i u_i + b_i, a_i>0) preserves best responses and hence
    # NE. This is a classical fact and orthogonal to the paper's V-
    # invariance claim; we verify it here to make explicit which
    # invariance class is at stake.
    Vs = V_family[0][1](sigma)
    U0 = np.stack([Vs - A[:, :, 0], Vs - A[:, :, 1]], axis=-1)
    a = np.array([2.5, 3.7])
    b = np.array([-1.1, 0.4])
    U_affine = a[None, None, :] * U0 + b[None, None, :]
    nes_affine = pure_ne(U_affine, feas)
    assert set(nes_affine) == set(canonical_ne), (
        "Positive-affine per-player transform of u_i moved the NE set: "
        f"expected {canonical_ne}, got {nes_affine}"
    )
    print("Positive-affine per-player transform of u_i: NE set unchanged. OK")

    # Externality-sign check: the signs of dsigma/deps_j and dA_i/deps_j
    # are V-free -- they are properties of the (A, sigma) arrays. We
    # verify them directly on the cache.
    ds_di = np.diff(sigma, axis=0)  # over i-index (player 0's eps)
    ds_dj = np.diff(sigma, axis=1)  # over j-index (player 1's eps)
    feas_i = feas[:-1, :] & feas[1:, :]
    feas_j = feas[:, :-1] & feas[:, 1:]
    print(f"sign(dsigma/deps_i)<0 on {(ds_di[feas_i] < 0).mean() * 100:.1f}% "
          f"of interior feasible edges (expected 100.0%)")
    print(f"sign(dsigma/deps_j)<0 on {(ds_dj[feas_j] < 0).mean() * 100:.1f}% "
          f"of interior feasible edges (expected 100.0%)")

    # A_i sign check: A_i should be non-increasing in eps_j (player j's
    # budget loosening reduces player i's realised MIA advantage on the
    # feasible region -- see Lemma "externality").
    dA0_dj = np.diff(A[:, :, 0], axis=1)
    dA1_di = np.diff(A[:, :, 1], axis=0)
    print(f"sign(dA_0/deps_1)<=0 on "
          f"{(dA0_dj[feas_j] <= 0).mean() * 100:.1f}% "
          f"of interior feasible edges")
    print(f"sign(dA_1/deps_0)<=0 on "
          f"{(dA1_di[feas_i] <= 0).mean() * 100:.1f}% "
          f"of interior feasible edges")


if __name__ == "__main__":
    main()
