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

    # Find the social optimum (maximises total utility) for each regime
    rows = []
    for reg, U, A in [("Aware", d["U_aw"], d["A_aw"]),
                       ("Naive", d["U_nv"], d["A_aw"])]:  # A is physically the same
        nes = find_pure_nes_2p(U, actions)
        total_u = U[:, :, 0] + U[:, :, 1]
        # Mask infeasible
        total_u = np.where(np.isnan(total_u), -np.inf, total_u)
        i_opt, j_opt = np.unravel_index(np.argmax(total_u), total_u.shape)
        A_opt = float(np.nanmean(A[i_opt, j_opt, :]))
        # If A_opt = 0 (degenerate), fall back to a large value
        A_opt_safe = max(A_opt, 1e-3)
        if not nes:
            rows.append(dict(
                regime=reg, ne_text="(no pure NE)",
                A1=None, A2=None, PoEV=None, sigma=None,
                A_opt=A_opt, opt=(float(actions[i_opt]), float(actions[j_opt])),
            ))
        for (i, j) in nes:
            A_avg = float(np.nanmean(A[i, j, :]))
            poev = (A_avg - A_opt) / A_opt_safe
            rows.append(dict(
                regime=reg, ne_text=f"({actions[i]:g}, {actions[j]:g})",
                A1=float(A[i, j, 0]), A2=float(A[i, j, 1]),
                PoEV=poev, sigma=float(d["S_aw"][i, j])
                                  if reg == "Aware" else None,
                A_opt=A_opt, opt=(float(actions[i_opt]), float(actions[j_opt])),
            ))
    # Write LaTeX table
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pure Nash equilibria of the symmetric 2-player PEG (action grid "
        r"$\mathcal{A} = \{1, 2, 4, 8, 16, 32, 64\}$, $|G_1|=|G_2|=5000$, $E_b=128$, "
        r"$5$ epochs, $\delta=10^{-12}$, $B_i = C_i = 1$, $V(\sigma) = 1/(1+\sigma)$). "
        r"$A_i$ is the realised MIA advantage of player $i$ at the equilibrium and "
        r"$A^{\textrm{opt}}$ is the realised MIA advantage at the social-optimum "
        r"profile $\boldsymbol{\eps}^{\textrm{opt}}$.}",
        r"\label{tab:NE}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Regime & $(\eps_1^*, \eps_2^*)$ & $A_1$ & $A_2$ & "
        r"$\boldsymbol{\eps}^{\textrm{opt}}$ & $A^{\textrm{opt}}$ \\",
        r"\midrule",
    ]
    for r in rows:
        if r["A1"] is None:
            lines.append(rf"{r['regime']} & {r['ne_text']} & --- & --- & --- & --- \\")
        else:
            opt_eps = r["opt"]
            lines.append(
                rf"{r['regime']} & {r['ne_text']} & {r['A1']:.3f} & "
                rf"{r['A2']:.3f} & "
                rf"$({opt_eps[0]:g}, {opt_eps[1]:g})$ & {r['A_opt']:.3f} \\"
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
