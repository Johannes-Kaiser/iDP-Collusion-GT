import sys, numpy as np
sys.path.insert(0, "/vol/miltank/users/kaiserj/Collusion_Gametheory")
from game.mechanism import solve_mechanism
from game.advantage import mia_advantage_subsampled_gaussian
from game.utility import value_fn
eps2 = 2.4985
grid = np.logspace(np.log10(0.5), np.log10(64), 200)
i0 = int(np.argmin(np.abs(grid-2.4985)))
print("own-utility column around the diagonal cell, tight tolerances:")
for k in range(i0-3, i0+4):
    e1 = float(grid[k])
    r = solve_mechanism([e1, eps2], [5000,5000], 128, 390, 1e-12)
    B = 5000*(r.sample_rates[0]+r.sample_rates[1])
    A1 = mia_advantage_subsampled_gaussian(r.sigma, float(r.sample_rates[0]), 390)
    u1 = value_fn(r.sigma) - A1
    print(f"  eps1={e1:.4f}  sigma={r.sigma:.6f}  B={B:.4f}  u1={u1:.7f}")
