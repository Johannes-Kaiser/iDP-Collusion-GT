# Measured PEG instance with above-floor attack power

**Priority:** H  **Effort:** H (GPU, Slurm)  **Category:** experiment

## Why
The one measured instance in the paper (main2, Sec. "Does the game survive a
measured adversary?", `fig_H1_n2_measured_NE`) sits at the LiRA measurement
floor on Adult MLP (A^LiRA in [0.107, 0.115], AUC <= 0.506, while the PLD
bound spans [0.004, 0.870]). The measured equilibria therefore collapse to
the high-eps corner and the strategic asymmetry is invisible. The paper
frames this honestly as "the game's asymmetric structure is carried by the
attacker's ability to realise the externality" — but the claim that the
asymmetric NE appears when attack power is above the floor is currently
*untested*. This is the single most valuable missing experiment.

## What
Re-run the H1 pipeline (`experiment_scripts/mia/04_mia_pergroup.py`,
`slurm_scripts/run_lira_n2_shadow.sh`, `tasks_findings/_scripts/24_*`) on a
configuration where LiRA resolves the leakage:
- inject canary-style outliers (mislabeled or duplicated records) per silo
  and score LiRA on the canaries, and/or
- use a more memorization-prone dataset/architecture (e.g. CIFAR-10 subset,
  small CNN, more epochs), and/or
- extend the eps grid downward so the low-eps cells have measurable
  analytical advantage on the trained model.
Success: measured payoff matrix whose NE set is off-diagonal (or an honest
negative result at higher attack power). Update Sec. V-H of main2 and the
figure; the current text already anticipates this experiment
("canary-style outliers" sentence).

## Budget
~25 profiles x 128 shadow models; GPU per Rule 3 (`--gres=gpu:1`), reuse the
array-job template. Estimate 50-100 GPU-h.
