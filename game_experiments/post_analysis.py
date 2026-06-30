"""Cross-cache post-analysis:
 * Finds pure NE in the 2-player aware and naive games (from E1 cache).
 * Computes the Price of Excess Vulnerability at each pure NE.
 * Compares to a uniform-budget baseline.

Writes a LaTeX-friendly table to paper/tex/table_NE.tex.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "game_experiments" / "_cache"
TEX_DIR = ROOT / "paper" / "tex"


def find_pure_nes_2p(U, actions):
    K = len(actions)
    nes = []
    for i in range(K):
        for j in range(K):
            i_br = int(np.nanargmax(U[:, j, 0]))
            j_br = int(np.nanargmax(U[i, :, 1]))
            if i_br == i and j_br == j:
                nes.append((i, j))
    return nes


def main():
    with open(CACHE_DIR / "E1.pkl", "rb") as f:
        d = pickle.load(f)
    actions = d["actions"]
    K = len(actions)

    # Uniform baseline: every player at the middle of the grid
    mid = K // 2
    base_idx = (mid, mid)

    rows = []
    for reg, U, A in [("Aware", d["U_aw"], d["A_aw"]),
                       ("Naive", d["U_nv"], d["A_aw"])]:  # A is the same physically
        nes = find_pure_nes_2p(U, actions)
        A_base = float(np.nanmean(A[base_idx[0], base_idx[1], :]))
        if not nes:
            rows.append(dict(regime=reg, ne_text="(no pure NE)",
                             A1=None, A2=None, PoEV=None, sigma=None))
        for (i, j) in nes:
            A_avg = float(np.nanmean(A[i, j, :]))
            poev = (A_avg - A_base) / max(A_base, 1e-9)
            rows.append(dict(
                regime=reg, ne_text=f"({actions[i]:g}, {actions[j]:g})",
                A1=float(A[i, j, 0]), A2=float(A[i, j, 1]),
                PoEV=poev, sigma=float(d["S_aw"][i, j])
                                  if reg == "Aware" else None,
            ))
    # Write LaTeX table
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pure Nash equilibria of the symmetric 2-player PEG (action grid "
        r"$\mathcal{A} = \{1, 2, 4, 8, 16, 32, 64\}$, $|G_1|=|G_2|=5000$, $E_b=128$, "
        r"$5$ epochs, $\delta=10^{-12}$). $\PoEV$ is computed against a uniform "
        r"baseline at $\eps_1 = \eps_2 = 8$.}",
        r"\label{tab:NE}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Regime & $(\eps_1^*, \eps_2^*)$ & $A_1$ & $A_2$ & $\PoEV$ \\",
        r"\midrule",
    ]
    for r in rows:
        if r["A1"] is None:
            lines.append(rf"{r['regime']} & {r['ne_text']} & --- & --- & --- \\")
        else:
            lines.append(
                rf"{r['regime']} & {r['ne_text']} & {r['A1']:.3f} & "
                rf"{r['A2']:.3f} & {r['PoEV']:+.2f} \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path = TEX_DIR / "table_NE.tex"
    out_path.write_text("\n".join(lines))
    print("Wrote", out_path)
    print()
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
