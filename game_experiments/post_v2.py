"""Post-analysis for continuous-action C1 sweep.

Reads C1.pkl (the 2-player payoff sweep) and writes paper/tex/table_NE_v2.tex
with a clean enumeration of pure NEs.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "game_experiments" / "_cache_v2" / "C1.pkl"
TEX = ROOT / "paper" / "tex" / "table_NE_v2.tex"


def main():
    if not CACHE.exists():
        print(f"missing cache {CACHE}, skipping")
        TEX.write_text("% Cache not yet generated.\n")
        return
    d = pickle.load(open(CACHE, "rb"))
    actions = d["actions"]
    sweep = d["sweep"]
    nes = d["nes"]
    A = sweep["A"]
    U = sweep["U"]
    S = sweep["sigma"]
    feas = sweep["feasible"]

    # Social optimum (over feasible grid)
    total_u = np.where(feas, U[:, :, 0] + U[:, :, 1], -np.inf)
    iopt, jopt = np.unravel_index(np.argmax(total_u), total_u.shape)
    A_opt = float((A[iopt, jopt, 0] + A[iopt, jopt, 1]) / 2.0)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pure Nash equilibria of the symmetric 2-player PEG on the "
        r"log-spaced action grid $\mathcal{A}_K = $ 22 points in $[0.5, 64]$, "
        r"$|G_1|=|G_2|=5000$, $E_b=128$, $5$ epochs, $\delta=10^{-12}$, "
        r"$B_i = C_i = 1$, $V(\sigma) = 1/(1+\sigma^2)$. Social optimum is "
        r"the feasible action profile maximising $u_1+u_2$.}",
        r"\label{tab:NE}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & $(\varepsilon_1^*, \varepsilon_2^*)$ & $A_1$ & $A_2$ & $\sigma$ \\",
        r"\midrule",
    ]
    if not nes:
        lines.append(r"NE & (none) & --- & --- & --- \\")
    for k, (i, j) in enumerate(nes):
        lines.append(
            rf"NE$_{{{k+1}}}$ & $({actions[i]:.2f}, {actions[j]:.2f})$ & "
            rf"{A[i, j, 0]:.3f} & {A[i, j, 1]:.3f} & {S[i, j]:.3f} \\"
        )
    lines.append(
        rf"social opt & $({actions[iopt]:.2f}, {actions[jopt]:.2f})$ & "
        rf"{A[iopt, jopt, 0]:.3f} & {A[iopt, jopt, 1]:.3f} & "
        rf"{S[iopt, jopt]:.3f} \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TEX.write_text("\n".join(lines))
    print("Wrote", TEX)
    print(f"NEs: {[(actions[i],actions[j]) for (i,j) in nes]}")
    print(f"social opt: ({actions[iopt]:.2f}, {actions[jopt]:.2f}), A_avg={A_opt:.3f}")


if __name__ == "__main__":
    main()
