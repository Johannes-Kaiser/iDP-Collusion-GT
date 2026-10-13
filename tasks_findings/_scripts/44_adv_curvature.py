"""Curvature of the composed advantage Adv_T(p) = delta_T(0) in p, at
fixed sigma, across composition counts T. T=1: exactly linear (p * TV of
the unsampled pair). T=2: exact quadrature. T>2: dp_accounting
connect-the-dots (pessimistic; discretization 1e-5 -> curvature signal
must exceed the bracket width to count)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import numpy as np
from dp_accounting.pld import privacy_loss_distribution as pld_lib

import game_experiments.two_step_game as g2

def adv_pld(sigma, p, T, disc=1e-5):
    pld = pld_lib.from_gaussian_mechanism(
        standard_deviation=sigma, sampling_prob=p,
        value_discretization_interval=disc,
        pessimistic_estimate=True, use_connect_dots=True)
    return pld.self_compose(T).get_delta_for_epsilon(0.0)

for sigma in (0.9, 3.0):
    print(f"== sigma={sigma} ==")
    ps = np.linspace(0.002, 0.0256, 9)
    dp2 = ps[1] - ps[0]
    for T in (1, 2, 5, 20, 100, 390):
        if T == 2:
            av = np.array([g2.advantage(sigma, float(p)) for p in ps])
            tag = "exact"
        else:
            av = np.array([adv_pld(sigma, float(p), T) for p in ps])
            tag = "pld"
        curv = np.diff(av, 2) / dp2**2
        ratio = av / ps
        print(f"  T={T:4d} ({tag:5s}): Adv range [{av[0]:.4e},{av[-1]:.4e}] "
              f"d2/dp2 in [{curv.min():+.3e},{curv.max():+.3e}] "
              f"Adv/p {ratio[0]:.4f}->{ratio[-1]:.4f}")
