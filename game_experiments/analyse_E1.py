"""Find pure Nash equilibria from the cached E1 payoff matrix."""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "game_experiments" / "_cache" / "E1.pkl", "rb") as f:
    data = pickle.load(f)

for label, U in [("Aware", data["U_aw"]), ("Naive", data["U_nv"])]:
    actions = data["actions"]
    K = len(actions)
    A_arr = data["A_aw"] if label == "Aware" else data["A_aw"]  # advantages are same
    print(f"== {label} regime ==")
    print("Best response of P1 (argmax over rows for each column):")
    for j, e2 in enumerate(actions):
        i_star = int(np.nanargmax(U[:, j, 0]))
        print(f"  eps2={e2:g}: BR_1 = eps1={actions[i_star]:g}  (u_1={U[i_star, j, 0]:.3f})")
    print("Best response of P2 (argmax over cols for each row):")
    for i, e1 in enumerate(actions):
        j_star = int(np.nanargmax(U[i, :, 1]))
        print(f"  eps1={e1:g}: BR_2 = eps2={actions[j_star]:g}  (u_2={U[i, j_star, 1]:.3f})")
    print("Pure NE (where each is best-responding):")
    nes = []
    for i, e1 in enumerate(actions):
        for j, e2 in enumerate(actions):
            i_br = int(np.nanargmax(U[:, j, 0]))
            j_br = int(np.nanargmax(U[i, :, 1]))
            if i_br == i and j_br == j:
                A1 = data["A_aw"][i, j, 0]
                A2 = data["A_aw"][i, j, 1]
                print(f"  NE: ({e1:g}, {e2:g})  u=({U[i,j,0]:.3f}, {U[i,j,1]:.3f})  A=({A1:.3f}, {A2:.3f})")
                nes.append((e1, e2))
    if not nes:
        print("  none.")
    print()
