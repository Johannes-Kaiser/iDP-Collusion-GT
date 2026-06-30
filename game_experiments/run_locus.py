"""
C6: Fine-grid (K=44) sweep of the symmetric 2-player game to map the NE locus.
C7: (B/C)-ratio sensitivity at K=22 (both players' weights vary together).
C8: Asymmetric (B_1, B_2) or (C_1, C_2) at K=22.

Each writes its own cache + figure. C6 is heavy (~40 min); C7/C8 are
moderate.
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
    best_response_curve,
)
from game.utility import PlayerProfile
from game.style import (
    apply_paper_style, PRIMARY_ORANGE, PRIMARY_BLUE, ACCENT_GREY,
    column_size, two_column_size, orange_palette, blue_palette,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / "game_experiments" / "_cache_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

apply_paper_style()

DELTA = 1e-12
EPS_MIN, EPS_MAX = 0.5, 64.0


def make_2player(B=(1.0, 1.0), C=(1.0, 1.0), K=22,
                  group_sizes=(5000, 5000),
                  eps_min=EPS_MIN, eps_max=EPS_MAX) -> ContinuousGame:
    M = sum(group_sizes)
    expected_batch = 128
    steps = int(5 * M / expected_batch)
    players = (PlayerProfile("P1", B=B[0], C=C[0]),
               PlayerProfile("P2", B=B[1], C=C[1]))
    return ContinuousGame(
        players=players, group_sizes=group_sizes,
        expected_batch_size=expected_batch, steps=steps, delta=DELTA,
        eps_min=eps_min, eps_max=eps_max, K=K,
    )


# --------------------------------------------------------------------------
# C6: Fine-grid NE locus
# --------------------------------------------------------------------------
def experiment_C6(K: int = 44):
    print(f"[C6] Fine-grid NE locus, K={K}")
    cache = CACHE_DIR / f"C6_K{K}.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        game = make_2player(K=K)
        t0 = time.time()
        sweep = sweep_2player_grid(game)
        nes = find_pure_NE_2player(sweep)
        br1 = best_response_curve(sweep, player=0)
        br2 = best_response_curve(sweep, player=1)
        print(f"  computed in {time.time()-t0:.1f}s, |NE|={len(nes)}")
        data = dict(sweep=sweep, nes=nes, br1=br1, br2=br2,
                    actions=sweep["actions"], K=K)
        pickle.dump(data, open(cache, "wb"))

    actions = data["actions"]
    sweep = data["sweep"]
    K = data["K"]

    # Plot the BR loci + NEs alone, plus an inset of u_1 with BR overlay.
    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.42),
                              gridspec_kw=dict(width_ratios=[1, 1]))

    # Panel (a): BR loci + NEs
    ax = axes[0]
    br1 = data["br1"]; br2 = data["br2"]
    m1 = br1 >= 0
    m2 = br2 >= 0
    ax.loglog(actions[br1[m1]], actions[m1],
               "o-", color=PRIMARY_BLUE, markersize=2.5, lw=0.8,
               label=r"$\varepsilon_1 = \mathrm{BR}_1(\varepsilon_2)$")
    ax.loglog(actions[m2], actions[br2[m2]],
               "s-", color=PRIMARY_ORANGE, markersize=2.5, lw=0.8,
               label=r"$\varepsilon_2 = \mathrm{BR}_2(\varepsilon_1)$")
    for (ni, nj) in data["nes"]:
        ax.scatter([actions[ni]], [actions[nj]], marker="*",
                    s=80, c="black", edgecolors="white", linewidth=0.5,
                    zorder=10)
    # Add diagonal as a reference
    ax.plot([actions[0], actions[-1]], [actions[0], actions[-1]],
            "--", color=ACCENT_GREY, lw=0.5, alpha=0.6, label="diagonal")
    ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title(f"BR loci and pure NEs (K={K})")
    ax.legend(frameon=False, fontsize=6, loc="lower right")

    # Panel (b): u_1 heatmap with NE markers
    ax = axes[1]
    U = sweep["U"]; feas = sweep["feasible"]
    U_plot = np.where(feas, U[:, :, 0], np.nan)
    im = ax.imshow(U_plot, origin="lower", aspect="auto", cmap="Blues",
                    extent=[0, len(actions)-1, 0, len(actions)-1])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$u_1$")
    # NE markers in index space
    for (ni, nj) in data["nes"]:
        ax.scatter([nj], [ni], marker="*", s=70, c="black",
                    edgecolors="white", linewidth=0.5, zorder=10)
    tick_pos = np.linspace(0, len(actions)-1, 6, dtype=int)
    tick_lab = [f"{actions[i]:.1f}" for i in tick_pos]
    ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab)
    ax.set_yticks(tick_pos); ax.set_yticklabels(tick_lab)
    ax.set_xlabel(r"$\varepsilon_2$"); ax.set_ylabel(r"$\varepsilon_1$")
    ax.set_title(f"$u_1$ with NEs marked")

    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig_C6_locus_K{K}.pdf")
    fig.savefig(FIG_DIR / f"fig_C6_locus_K{K}.svg")
    plt.close()
    print(f"  -> fig_C6_locus_K{K}.pdf  (|NE|={len(data['nes'])})")
    return data


# --------------------------------------------------------------------------
# C7: (B/C)-ratio symmetric sweep
# --------------------------------------------------------------------------
def experiment_C7(K: int = 22):
    print(f"[C7] (B/C)-ratio symmetric sensitivity, K={K}")
    cache = CACHE_DIR / f"C7_K{K}.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        ratios = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # B/C
        rows = []
        for r in ratios:
            # Keep C = 1, scale B
            game = make_2player(B=(r, r), C=(1.0, 1.0), K=K)
            sweep = sweep_2player_grid(game)
            nes = find_pure_NE_2player(sweep)
            rows.append({"ratio": r, "nes": nes,
                         "actions": sweep["actions"],
                         "U": sweep["U"], "A": sweep["A"],
                         "S": sweep["sigma"],
                         "feasible": sweep["feasible"]})
            print(f"  B/C={r}: |NE|={len(nes)}")
        data = dict(rows=rows, K=K)
        pickle.dump(data, open(cache, "wb"))

    rows = data["rows"]
    K = data["K"]
    fig, ax = plt.subplots(figsize=column_size(aspect=0.72))
    cmap = blue_palette(len(rows))
    for ip, r_data in enumerate(rows):
        actions = r_data["actions"]
        nes = r_data["nes"]
        for (ni, nj) in nes:
            ax.loglog(actions[ni], actions[nj], "o",
                       color=cmap[ip], markersize=5,
                       markeredgecolor="white", markeredgewidth=0.4)
    # Legend entries
    handles = []
    for ip, r_data in enumerate(rows):
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color=cmap[ip], marker="o",
                                lw=0, markersize=5,
                                label=f"B/C={r_data['ratio']:g}"))
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="best")
    ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title("NE locations across symmetric $B/C$ ratios")
    # Diagonal reference
    ax.plot([0.5, 64], [0.5, 64], "--", color=ACCENT_GREY, lw=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig_C7_BC_ratio.pdf")
    fig.savefig(FIG_DIR / f"fig_C7_BC_ratio.svg")
    plt.close()
    print(f"  -> fig_C7_BC_ratio.pdf")
    return data


# --------------------------------------------------------------------------
# C8: Asymmetric (B_1, B_2) or (C_1, C_2)
# --------------------------------------------------------------------------
def experiment_C8(K: int = 22):
    print(f"[C8] Asymmetric (C_1, C_2) sensitivity, K={K}")
    cache = CACHE_DIR / f"C8_K{K}.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        # Vary C_1/C_2 ratio: player 1 more or less privacy-sensitive than player 2
        ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
        rows = []
        for r in ratios:
            # C_1 = r, C_2 = 1
            game = make_2player(B=(1.0, 1.0), C=(r, 1.0), K=K)
            sweep = sweep_2player_grid(game)
            nes = find_pure_NE_2player(sweep)
            rows.append({"ratio": r, "nes": nes,
                         "actions": sweep["actions"],
                         "U": sweep["U"], "A": sweep["A"],
                         "S": sweep["sigma"],
                         "feasible": sweep["feasible"]})
            print(f"  C_1/C_2={r}: |NE|={len(nes)}")
        data = dict(rows=rows, K=K)
        pickle.dump(data, open(cache, "wb"))

    rows = data["rows"]
    K = data["K"]
    fig, ax = plt.subplots(figsize=column_size(aspect=0.72))
    cmap = orange_palette(len(rows))
    for ip, r_data in enumerate(rows):
        actions = r_data["actions"]
        nes = r_data["nes"]
        for (ni, nj) in nes:
            ax.loglog(actions[ni], actions[nj], "s",
                       color=cmap[ip], markersize=5,
                       markeredgecolor="white", markeredgewidth=0.4)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=cmap[ip], marker="s",
                       lw=0, markersize=5,
                       label=f"$C_1/C_2 = {r_data['ratio']:g}$")
               for ip, r_data in enumerate(rows)]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="best")
    ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title("NE shift under asymmetric privacy weights $C_1 \\neq C_2$")
    ax.plot([0.5, 64], [0.5, 64], "--", color=ACCENT_GREY, lw=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig_C8_asymmetric_C.pdf")
    fig.savefig(FIG_DIR / f"fig_C8_asymmetric_C.svg")
    plt.close()
    print(f"  -> fig_C8_asymmetric_C.pdf")
    return data


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of C6,C7,C8.")
    parser.add_argument("--K6", type=int, default=44,
                        help="Grid size for C6.")
    parser.add_argument("--K7", type=int, default=22,
                        help="Grid size for C7.")
    parser.add_argument("--K8", type=int, default=22,
                        help="Grid size for C8.")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else None

    EXPERIMENTS = [
        ("C6", lambda: experiment_C6(args.K6)),
        ("C7", lambda: experiment_C7(args.K7)),
        ("C8", lambda: experiment_C8(args.K8)),
    ]
    for name, fn in EXPERIMENTS:
        if only is None or name in only:
            fn()
