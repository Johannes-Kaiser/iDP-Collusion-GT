"""
C6: Fine-grid (K=200) sweep of the symmetric 2-player game to map the NE locus.
C7: Symmetric w sweep at K=200 (both players share the same privacy weight).
C8: Asymmetric w_1 / w_2 at K=200.

Utility: u_i(eps) = V(sigma(eps)) - w_i A_i(eps).

Parallelism: the expensive step is a single mechanism-grid precomputation
(shared across all weight configurations at the same (K, group sizes,
batch, steps, delta)). It is farmed out to a process pool of size
PEG_N_WORKERS (default 1). Weight-dependent utility/NE evaluations are
then cheap O(K^2) per weight setting.
"""
from __future__ import annotations

import os, pickle, sys, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_N_WORKERS = int(os.environ.get("PEG_N_WORKERS", "1"))

from game.continuous import (
    precompute_mechanism_grid, sweep_from_grid,
    find_pure_NE_2player, best_response_curve,
)
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
GROUP_SIZES = (5000, 5000)
EXPECTED_BATCH = 128
EPOCHS = 5
STEPS = int(EPOCHS * sum(GROUP_SIZES) / EXPECTED_BATCH)


# --------------------------------------------------------------------------
# Shared mechanism grid: computed once, reused across every weight setting.
# --------------------------------------------------------------------------

def load_or_compute_mech_grid(K: int) -> dict:
    """Load the mechanism grid from disk if cached; otherwise compute and save.

    The cache key encodes (K, group sizes, batch, steps, delta). Different
    experiments at the same base params (which is all of C6/C7/C8) share it.
    """
    tag = f"mech_grid_K{K}_g{GROUP_SIZES[0]}x{GROUP_SIZES[1]}_b{EXPECTED_BATCH}_e{EPOCHS}"
    path = CACHE_DIR / f"{tag}.pkl"
    if path.exists():
        print(f"[mech-grid] loading cached grid: {path.name}")
        return pickle.load(open(path, "rb"))
    print(f"[mech-grid] computing K={K} with {_N_WORKERS} workers...")
    t0 = time.time()
    grid = precompute_mechanism_grid(
        K=K, group_sizes=GROUP_SIZES,
        expected_batch_size=EXPECTED_BATCH,
        steps=STEPS, delta=DELTA,
        eps_min=EPS_MIN, eps_max=EPS_MAX,
        n_workers=_N_WORKERS,
    )
    dt = time.time() - t0
    n_feasible = int(grid["feasible"].sum())
    print(f"[mech-grid] done in {dt:.1f}s  feasible cells: {n_feasible}/{K*K}")
    pickle.dump(grid, open(path, "wb"))
    return grid


# --------------------------------------------------------------------------
# C6: Fine-grid NE locus (symmetric w=1)
# --------------------------------------------------------------------------
def experiment_C6(K: int = 200):
    print(f"[C6] Fine-grid NE locus, K={K}")
    cache = CACHE_DIR / f"C6_K{K}.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        grid = load_or_compute_mech_grid(K)
        sweep = sweep_from_grid(grid, w1=1.0, w2=1.0)
        nes = find_pure_NE_2player(sweep)
        br1 = best_response_curve(sweep, player=0)
        br2 = best_response_curve(sweep, player=1)
        data = dict(sweep=sweep, nes=nes, br1=br1, br2=br2,
                    actions=sweep["actions"], K=K)
        pickle.dump(data, open(cache, "wb"))
        print(f"  |NE|={len(nes)}")

    actions = data["actions"]
    sweep = data["sweep"]
    K = data["K"]

    fig, axes = plt.subplots(1, 2, figsize=two_column_size(rows=1, aspect=0.42),
                              gridspec_kw=dict(width_ratios=[1, 1]))
    ax = axes[0]
    br1 = data["br1"]; br2 = data["br2"]
    m1 = br1 >= 0
    m2 = br2 >= 0
    ax.loglog(actions[br1[m1]], actions[m1],
               "o-", color=PRIMARY_BLUE, markersize=1.8, lw=0.5,
               label=r"$\varepsilon_1 = \mathrm{BR}_1(\varepsilon_2)$")
    ax.loglog(actions[m2], actions[br2[m2]],
               "s-", color=PRIMARY_ORANGE, markersize=1.8, lw=0.5,
               label=r"$\varepsilon_2 = \mathrm{BR}_2(\varepsilon_1)$")
    for (ni, nj) in data["nes"]:
        ax.scatter([actions[ni]], [actions[nj]], marker="*",
                    s=60, c="black", edgecolors="white", linewidth=0.4,
                    zorder=10)
    ax.plot([actions[0], actions[-1]], [actions[0], actions[-1]],
            "--", color=ACCENT_GREY, lw=0.5, alpha=0.6, label="diagonal")
    ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title(f"BR loci and pure NEs (K={K})")
    ax.legend(frameon=False, fontsize=6, loc="lower right")

    ax = axes[1]
    U = sweep["U"]; feas = sweep["feasible"]
    U_plot = np.where(feas, U[:, :, 0], np.nan)
    im = ax.imshow(U_plot, origin="lower", aspect="auto", cmap="Blues",
                    extent=[0, len(actions)-1, 0, len(actions)-1])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$u_1$")
    for (ni, nj) in data["nes"]:
        ax.scatter([nj], [ni], marker="*", s=50, c="black",
                    edgecolors="white", linewidth=0.4, zorder=10)
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
# C7: Symmetric w sweep
# --------------------------------------------------------------------------
def experiment_C7(K: int = 200):
    print(f"[C7] symmetric w sensitivity, K={K}")
    cache = CACHE_DIR / f"C7_K{K}.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        grid = load_or_compute_mech_grid(K)
        ws = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
        rows = []
        for w in ws:
            sweep = sweep_from_grid(grid, w1=w, w2=w)
            nes = find_pure_NE_2player(sweep)
            rows.append({"w": w, "nes": nes,
                         "actions": sweep["actions"],
                         "U": sweep["U"], "A": sweep["A"],
                         "S": sweep["sigma"],
                         "feasible": sweep["feasible"]})
            print(f"  w={w}: |NE|={len(nes)}")
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
                       color=cmap[ip], markersize=3.5,
                       markeredgecolor="white", markeredgewidth=0.3,
                       alpha=0.8)
    handles = []
    for ip, r_data in enumerate(rows):
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color=cmap[ip], marker="o",
                                lw=0, markersize=5,
                                label=f"$w={r_data['w']:g}$"))
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="best")
    ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title("NE locations across symmetric privacy weights $w$")
    ax.plot([0.5, 64], [0.5, 64], "--", color=ACCENT_GREY, lw=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig_C7_w_ratio.pdf")
    fig.savefig(FIG_DIR / f"fig_C7_w_ratio.svg")
    plt.close()
    print(f"  -> fig_C7_w_ratio.pdf")
    return data


# --------------------------------------------------------------------------
# C8: Asymmetric w_1 / w_2 with w_2 = 1
# --------------------------------------------------------------------------
def experiment_C8(K: int = 200):
    print(f"[C8] Asymmetric w_1 / w_2 sensitivity, K={K}")
    cache = CACHE_DIR / f"C8_K{K}.pkl"
    if cache.exists():
        data = pickle.load(open(cache, "rb"))
    else:
        grid = load_or_compute_mech_grid(K)
        ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
        rows = []
        for r in ratios:
            sweep = sweep_from_grid(grid, w1=r, w2=1.0)
            nes = find_pure_NE_2player(sweep)
            rows.append({"ratio": r, "nes": nes,
                         "actions": sweep["actions"],
                         "U": sweep["U"], "A": sweep["A"],
                         "S": sweep["sigma"],
                         "feasible": sweep["feasible"]})
            print(f"  w_1/w_2={r}: |NE|={len(nes)}")
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
                       color=cmap[ip], markersize=3.5,
                       markeredgecolor="white", markeredgewidth=0.3,
                       alpha=0.8)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=cmap[ip], marker="s",
                       lw=0, markersize=5,
                       label=f"$w_1/w_2 = {r_data['ratio']:g}$")
               for ip, r_data in enumerate(rows)]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="best")
    ax.set_xlabel(r"$\varepsilon_1$"); ax.set_ylabel(r"$\varepsilon_2$")
    ax.set_title("NE shift under asymmetric privacy weights $w_1 \\neq w_2$")
    ax.plot([0.5, 64], [0.5, 64], "--", color=ACCENT_GREY, lw=0.5, alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"fig_C8_asymmetric_w.pdf")
    fig.savefig(FIG_DIR / f"fig_C8_asymmetric_w.svg")
    plt.close()
    print(f"  -> fig_C8_asymmetric_w.pdf")
    return data


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of C6,C7,C8.")
    parser.add_argument("--K6", type=int, default=200)
    parser.add_argument("--K7", type=int, default=200)
    parser.add_argument("--K8", type=int, default=200)
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
