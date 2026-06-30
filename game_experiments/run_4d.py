"""
C9 - 4D sensitivity of the NE locus.

We sweep both player-asymmetry ratios at once:
    rB = B_1/B_2  ∈  {0.5, 1, 2}
    rC = C_1/C_2  ∈  {0.5, 1, 2}
with B_2 = C_2 = 1 fixed.

For each (rB, rC) we run the full 2-player sweep on a moderate grid K=12
and find the pure NEs. We then represent the four-dimensional dependence
(rB, rC) -> {(eps_1, eps_2)} two ways:

    (i)  a 3x3 grid of small-multiples scatter plots, one per (rB, rC),
         showing the NE positions in (eps_1, eps_2).
    (ii) a single 2-panel summary with two scalars per (rB, rC):
         - the asymmetry index  S  =  mean log(eps_top/eps_bot)
           (averaged over the NEs of that (rB, rC))
         - the realised-vulnerability gap  Delta_A  =  mean |A_1 - A_2|
         each on a heatmap over (rB, rC).
"""
from __future__ import annotations

import os, pickle, sys, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game.continuous import (
    ContinuousGame, sweep_2player_grid, find_pure_NE_2player,
)
from game.utility import PlayerProfile
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
    column_size, two_column_size,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / "game_experiments" / "_cache_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

apply_paper_style()

DELTA = 1e-12
EPS_MIN, EPS_MAX = 0.5, 64.0
KGRID = 12   # for tractability across the 3x3 sweep


def make_game(B=(1.0, 1.0), C=(1.0, 1.0), K=KGRID) -> ContinuousGame:
    group_sizes = (5000, 5000)
    expected_batch = 128
    steps = int(5 * sum(group_sizes) / expected_batch)
    players = (PlayerProfile("P1", B=B[0], C=C[0]),
               PlayerProfile("P2", B=B[1], C=C[1]))
    return ContinuousGame(
        players=players, group_sizes=group_sizes,
        expected_batch_size=expected_batch, steps=steps, delta=DELTA,
        eps_min=EPS_MIN, eps_max=EPS_MAX, K=K,
    )


def experiment_C9():
    print(f"[C9] 4D NE sensitivity (B/B and C/C), K={KGRID}")
    cache = CACHE_DIR / f"C9_K{KGRID}.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        rBs = [0.5, 1.0, 2.0]
        rCs = [0.5, 1.0, 2.0]
        rows = []
        for rB in rBs:
            for rC in rCs:
                t0 = time.time()
                game = make_game(B=(rB, 1.0), C=(rC, 1.0), K=KGRID)
                sweep = sweep_2player_grid(game)
                nes = find_pure_NE_2player(sweep)
                rows.append({
                    "rB": rB, "rC": rC, "nes": nes,
                    "actions": sweep["actions"],
                    "U": sweep["U"], "A": sweep["A"],
                    "S": sweep["sigma"], "feasible": sweep["feasible"],
                })
                print(f"  rB={rB}, rC={rC}: |NE|={len(nes)} ({time.time()-t0:.0f}s)")
        data = dict(rows=rows, rBs=rBs, rCs=rCs)
        pickle.dump(data, open(cache, "wb"))

    # ---------------------------------------------------------
    # Plot (i): 3x3 small-multiples, one panel per (rB, rC).
    # ---------------------------------------------------------
    rBs = data["rBs"]; rCs = data["rCs"]
    fig, axes = plt.subplots(len(rBs), len(rCs),
                              figsize=two_column_size(rows=1.5, aspect=0.4),
                              sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    rows = data["rows"]
    # index rows by (rB, rC)
    rows_grid = {(r["rB"], r["rC"]): r for r in rows}
    for i, rB in enumerate(rBs):
        for j, rC in enumerate(rCs):
            ax = axes[i, j]
            r = rows_grid[(rB, rC)]
            actions = r["actions"]
            for (ni, nj) in r["nes"]:
                ax.loglog(actions[ni], actions[nj], "o",
                           color=PRIMARY_ORANGE, markersize=4,
                           markeredgecolor="white", markeredgewidth=0.3)
            ax.plot([EPS_MIN, EPS_MAX], [EPS_MIN, EPS_MAX],
                    "--", color=ACCENT_GREY, lw=0.4, alpha=0.5)
            ax.set_xlim(EPS_MIN, EPS_MAX)
            ax.set_ylim(EPS_MIN, EPS_MAX)
            if i == 0:
                ax.set_title(f"$C_1/C_2 = {rC:g}$", fontsize=8)
            if j == 0:
                ax.set_ylabel(rf"$B_1/B_2={rB:g}$" + "\n" + r"$\varepsilon_2$",
                               fontsize=7)
            if i == len(rBs) - 1:
                ax.set_xlabel(r"$\varepsilon_1$")
    plt.tight_layout()
    out1 = FIG_DIR / "fig_C9_smallmultiples.pdf"
    fig.savefig(out1)
    fig.savefig(out1.with_suffix(".svg"))
    plt.close()
    print(f"  -> {out1.name}")

    # ---------------------------------------------------------
    # Plot (ii): heatmap of (asymmetry, A-gap) over (rB, rC).
    # ---------------------------------------------------------
    nb, nc = len(rBs), len(rCs)
    S = np.full((nb, nc), np.nan)
    DA = np.full((nb, nc), np.nan)
    NEcount = np.zeros((nb, nc), dtype=int)
    for i, rB in enumerate(rBs):
        for j, rC in enumerate(rCs):
            r = rows_grid[(rB, rC)]
            nes = r["nes"]
            NEcount[i, j] = len(nes)
            if not nes:
                continue
            actions = r["actions"]
            asym = []
            da = []
            for (ni, nj) in nes:
                a1, a2 = actions[ni], actions[nj]
                A1 = r["A"][ni, nj, 0]; A2 = r["A"][ni, nj, 1]
                asym.append(abs(np.log(a1) - np.log(a2)))
                da.append(abs(A1 - A2))
            S[i, j] = float(np.mean(asym))
            DA[i, j] = float(np.mean(da))

    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.42))
    for ax, M, title, cmap in [
        (axes[0], S, r"asymmetry $\langle|\log(\varepsilon_1/\varepsilon_2)|\rangle$", "Blues"),
        (axes[1], DA, r"realised-$A$ gap $\langle|A_1-A_2|\rangle$", "Oranges"),
    ]:
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(nc))
        ax.set_xticklabels([f"{rC:g}" for rC in rCs])
        ax.set_yticks(range(nb))
        ax.set_yticklabels([f"{rB:g}" for rB in rBs])
        ax.set_xlabel(r"$C_1/C_2$")
        ax.set_ylabel(r"$B_1/B_2$")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Annotate NE count in each cell
        for i in range(nb):
            for j in range(nc):
                ax.text(j, i, f"{NEcount[i,j]}", ha="center", va="center",
                        color="black", fontsize=6)
    plt.tight_layout()
    out2 = FIG_DIR / "fig_C9_heatmap.pdf"
    fig.savefig(out2)
    fig.savefig(out2.with_suffix(".svg"))
    plt.close()
    print(f"  -> {out2.name}")
    return data


if __name__ == "__main__":
    experiment_C9()
