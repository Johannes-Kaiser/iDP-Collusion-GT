# The Privacy Externality Game

Code, experiments, and paper draft for a game-theoretic analysis of
**sampling-based individual differential privacy (iDP)**. In sampling-based
iDP, every record's sampling rate is tied to everyone else's budget through a
shared noise multiplier and a fixed expected batch size. A data contributor's
choice of `(eps, delta)` budget therefore changes the membership-inference
risk of every other contributor. We model this coupling as a game, the
**Privacy Externality Game (PEG)**, and study its equilibria, attacks, and
mitigations.

The companion paper is Kaiser et al., *Your Privacy Depends on Others:
Collusion Vulnerabilities in Individual Differential Privacy* (SaTML 2026,
arXiv 2601.12922). Its code (budget-manipulation attack, LiRA experiments)
still lives in this repository; see [Legacy code](#legacy-code-companion-paper).

> **For LLM agents:** read this file, then `CLAUDE.md` (mandatory working
> rules), then `FINDINGS_MAP.md` (what is known and how much to trust it).
> The rest of this README tells you where everything is.

---

## 1. The game

### Why there is a game at all

Individual DP lets each data contributor sign their own `(eps_i, delta)`
contract. The sampling-based realisation meets all contracts at once by
giving every group its own sampling rate `p_i` while all groups share one
noise multiplier `sigma` and one expected batch size `B`. Because `B` is
fixed, one contributor's budget choice changes the solved `sigma` and the
other groups' sampling rates. As a result, a record's realised
membership-inference risk depends on its own budget *and* on everyone else's.
The nominal `(eps_i, delta)` contract still holds; the advantage an attacker
actually achieves does not stay fixed.

Contributors who understand this can act strategically. A budget is
then a move in a game, not a private preference. The PEG asks what happens when
every contributor picks the budget that is best for them, given what the others
pick.

### Players, actions, payoffs

- **Players:** `n` data owners (2 in most experiments). Player `i` owns a
  group of records `G_i`.
- **Action:** a privacy budget `eps_i` (`delta` is common and fixed).
- **Mechanism:** given all budgets, `game/mechanism.py` solves for one shared
  `sigma` and per-group sampling rates `p_i` so that each group meets its RDP
  budget over `T` DP-SGD steps and the expected batch size equals `B`.
- **Trade-off:** a looser budget lowers `sigma` and raises model quality for
  everyone. It also raises the player's own sampling rate and attack
  advantage, and it changes the other players' advantage too.

Each player's utility is

```
u_i(eps) = V(sigma(eps)) - lambda_i * A_PLD,i(eps)
```

- `V(sigma) = 1 / (1 + sigma^2)`: shared model-quality proxy (`game/utility.py`).
- `A_PLD,i`: player `i`'s membership-inference advantage, computed from the
  privacy loss distribution of the subsampled Gaussian (`game/advantage.py`).
- `lambda_i`: per-player privacy weight. A coalition's malice weight is `kappa`.

**Notation map.** The paper writes `lambda_i`, `A_PLD,i` (macro `\Apld{i}`),
`kappa`. The code writes `w_i`, `A_i`. Do not rename code symbols without
regenerating every figure that prints them. `B_i`, `C_i`, or "benefit-to-cost"
anywhere is stale and should be removed.

### Experimental setup

Baseline instance (used by most of the campaign):

| parameter | value |
|---|---|
| players | 2 groups, `|G1| = |G2| = 5000` records |
| expected batch size | `B = 128` |
| DP-SGD steps | `T = 390` |
| `delta` | `1e-12` |
| privacy weight | `lambda = 1` (also swept, and heterogeneous ratios `lambda_2/lambda_1` up to 8) |
| action grid | `eps in [0.5, 64]`, K = 200 log-spaced points |
| accounting | deployed RDP accountant (89 orders) for the grid; exact PLD / certified composition ladder for the "exact" results |

Variants: `T in {1, 2}` closed-form models, `n = 3–20` players, FL-consortium
populations, a sensitivity-based ("Scale") mechanism, coalitions with malice
weight `kappa`, and a LiRA-measured adversary.

---

## 2. Findings

> **Disclaimer.** These findings are **small-scale and preliminary**. Most
> rest on a single two-player instance (above), a finite action grid, and a
> proxy model-quality function `V`. Several were revised after the
> 2026-07-13 solver fix, and the campaign is ongoing, so **any item here may
> already be outdated**. `FINDINGS_MAP.md` is the current record; trust
> levels there: **A** proved or exact, **B** solid numerics, **C**
> provisional, **D** superseded. Read each finding as evidence for the
> stated setup, not as a general law.

1. **The externality is real (A/B).** When one player tightens their budget,
   `sigma` rises and the other player's sampling rate and attack advantage
   change. On the post-fix K = 200 grid the sign of the effect holds at every
   cell. Two records with the same `(eps, delta)` contract can face
   different realised advantage.
2. **Equilibria are asymmetric corner equilibria (B, exact FOC: A).** The
   K = 200 grid has 43 pure Nash equilibria: 42 mirror pairs plus one on the
   diagonal. The off-diagonal ones lie on the participation frontier: one
   player plays tight, the other loose and carries the batch. Restricted to
   a mirror pair, the game behaves like Hawk–Dove.
3. **Why corners (A).** Under exact accounting, a looser budget first buys
   participation (sampling rate) and only later model quality. This drives
   play to corners. A symmetric equilibrium appears only when a constraint
   pins both players.
4. **Symmetric moderation exists only at composition scale (A).** At
   `T = 390`, `lambda = 1` there is exactly one symmetric equilibrium,
   `eps* ≈ 2.59` under exact accounting (≈ 2.96 on the deployed grid). It is
   locally stable. For `T <= 2` symmetric points exist but are not equilibria.
5. **Some grid equilibria are accountant artefacts (B).** The 89-order RDP
   accountant adds a ~1e-4 payoff sawtooth. It creates and destroys near-tie
   equilibria; the "no pure NE / cycling" result at `lambda = 2` depends on it.
6. **A common privacy weight moves equilibria but restores no symmetry (B).**
   Larger `lambda` shifts the equilibrium cloud to tighter budgets; asymmetric
   play remains.
7. **Heterogeneity sorts roles (A under a condition; B numerically).** The
   less privacy-sensitive player carries the batch (170/174 heterogeneous
   equilibria). Group size does *not* sort roles after the solver fix.
8. **The first mover selects the outcome (B).** Under alternating best
   responses the basin depends only on the first mover's budget. A
   Stackelberg leader commits tight.
9. **Selfish play loses welfare (B).** The planner's optimum is itself
   asymmetric; selfish play lands about 9% below it and never compensates
   the player who carries the batch.
10. **More players (C).** Role sorting survives for `n = 3–10`; at `n = 20`
    every player free-rides at the budget floor.
11. **The effect is mechanism-specific (B).** The sensitivity-based ("Scale")
    iDP mechanism has a single symmetric equilibrium.
12. **Coalitions (B).** A coalition attack is coalition play with
    `kappa -> infinity`. Against a passive target (n = 5), four colluders
    raise its advantage by 64%. A target that best-responds ends up with
    18–51× lower advantage than a passive one.
13. **Measured adversary (C).** With LiRA at the floor, the externality is
    invisible and the equilibrium sits at the accuracy corner.
14. **Mitigation (A/C).** A Δ-iDP cap bounds how much a player's advantage
    can be raised by others. Budget reports are not truthful in dominant
    strategies. Linear pricing cannot stabilise moderate symmetric budgets;
    convex pricing can.
15. **Robustness (B).** Across six choices of `V` the frontier
    equilibria coincide while the diagonal point moves. Varying the
    participation floor keeps all equilibria off the diagonal; PLD
    discretisation changes advantage by at most 4.5e-3. Corner play holds
    in 107/110 linear-cost model variants: the linear cost term produces it.
16. **Structure (A, scoped).** A pure equilibrium exists on the small-`p`
    region; the PEG is not a potential game.

The Gaussian-DP large-`T` limit model (`gdp_limit_game.py`) is a
potential game with a saddle, but it only describes the high-`T` regime and
is not used for claims about the deployed game.

---

## 3. Sources of truth (read before trusting anything)

| rank | source | role |
|---|---|---|
| 1 | `game/`, `game_experiments/` | the code defines the model |
| 2 | `game_experiments/_cache_v2/` (and other `_cache_*`) | regenerated experiment results (pickles, git-ignored) |
| 3 | `nummeric_eval.md` | campaign log for the exact-equilibrium analysis |
| 4 | `FINDINGS_MAP.md` | topic index (T0–T19), trust levels A–D, contradictions register |
| 5 | `PAPER_AUDIT.md` | claim-by-claim audit of the paper against code and data |
| — | `paper/` | **provisional**; never evidence for a claim (CLAUDE.md Rule -1) |

**The 2026-07-13 solver fix is a hard boundary.** An early break in the sigma
bisection in `game/mechanism.py::solve_mechanism` was fixed on that date.
Caches built before it can have sigma off by up to ~10%. Treat any result
built from a pre-fix cache as provisional.

---

## 4. Repository map

### Core library: `game/`

| file | contents |
|---|---|
| `mechanism.py` | sampling-based iDP solver: shared `sigma`, per-group `p_g`, fixed expected batch |
| `mechanism_scale.py` | sensitivity-based ("Scale") iDP solver, for the cross-mechanism comparison |
| `advantage.py` | MIA advantage from the PLD / trade-off function of the subsampled Gaussian |
| `utility.py` | `V(sigma)` and the player utility |
| `game.py` | `Game` evaluator: profiles, payoff grids, best responses |
| `continuous.py` | fine log-grid sweeps, cached mechanism grids |
| `solvers.py` | fictitious play, Pareto fronts, coalition (Stackelberg) sweeps |
| `bayesian.py` | Bayesian repeated PEG (honest vs. malicious player) |
| `style.py` | shared matplotlib style, matched to the companion paper |
| `_smoke.py` | smoke test: 2-player NE via best-response dynamics |

Changes to `utility.py`, `mechanism.py`, `continuous.py`, or experiment grid
defaults change equilibria. Re-run the affected experiments and update the
paper (CLAUDE.md Rule 5).

### Experiments: `game_experiments/`

Each script has a one-line module docstring stating its purpose. Groups:

| group | scripts |
|---|---|
| deployed-grid game (K=200) | `run_continuous.py`, `run_locus.py`, `run_br_trace.py`, `run_mech_dense.py`, `run_cycle_zoom.py`, `patch_grid_analytic_feas.py` |
| exact accounting | `exact_equilibria.py`, `exact_eq_refine.py`, `exact_foc.py`, `exact_foc_grid.py`, `multi_step_game.py`, `double_ladder.py`, `tab_ladder_grid.py`, `t390_local_slice.py`, `run_t_grid.py` |
| closed-form small-T models | `single_step_*.py` (T=1), `two_step_*.py` (T=2) |
| attacks & coalitions | `run_h4.py` (aware target), `run_lambda_pareto.py` |
| n players / FL populations | `run_nplayer.py`, `run_flpop_e1.py`, `run_flpop_e2.py`, `flpop_*.py` |
| mechanism comparison | `run_scale.py` |
| robustness | `run_robustness.py`, `robust_game.py`, `diag_curvature_scan.py` |
| Gaussian-DP limit (demoted) | `gdp_limit_game.py`: high-T regime only, not in the paper's main line |
| verification one-offs | `_verify_*.py`, `audit_h1_sigma.py` |
| legacy drivers | `run_all.py`, `post_analysis.py`, `post_v2.py`, `run_bayesian.py` |

Run scripts as modules from the repo root, e.g.
`python -m game_experiments.run_br_trace --K 22`. Caches go to
`game_experiments/_cache_*/` (git-ignored). Delete a figure's cache before
regenerating that figure.

### Findings and analysis: `tasks_findings/`

- `NN_<topic>.md`: one write-up per research question (propositions,
  robustness sweeps, attacks, n-player results). Numbers match the topic IDs in
  `FINDINGS_MAP.md`.
- `_scripts/`: the analysis and figure scripts behind those write-ups, plus
  their `*_summary.txt` / `*.txt` outputs. `NN_` prefixes match the write-ups.

### Paper: `paper/`

- `main.tex` + `tex/01_intro.tex` … `tex/09_appendix.tex`: the paper, under
  active revision. Build: `cd paper && pdflatex -interaction=nonstopmode main.tex`.
- `refs.bib`: bibliography.
- `figures/`: figure files (`.pdf` included by LaTeX, `.svg` alongside).
  `figures/data/*.dat` are pgfplots tables (git-ignored; regenerate with the
  `tasks_findings/_scripts/` producers).
- `history/`: superseded drafts kept for reference only. Never cite or copy.

Writing rules for the paper (final-artefact voice, no history, terminology,
anonymity) are in `CLAUDE.md`.

### Compute: `slurm_scripts/` (git-ignored)

All non-trivial compute goes through Slurm; CPU unless a job trains a network.
Job scripts (`*.sh`) are kept locally and not versioned. Logs go to `out/`.

### Other top-level files

| path | contents |
|---|---|
| `CLAUDE.md` | rules for working in this repo (paper, figures, Slurm, notation) |
| `FINDINGS_MAP.md` | the findings index; start here for any research question |
| `nummeric_eval.md` | detailed campaign log (setup, runs, data paths, conclusions) |
| `PAPER_AUDIT.md` | which paper claims hold, are stale, or are falsified |
| `COLLUSION_GAME_PLAN.md`, `claude_nodes.md` | early planning notes; historical |
| `.tasks2/` | open task tickets (`NN_<impact>_<effort>_<slug>.md`) |
| `slides/` | findings deck (`build_deck.py`, `make_figs.py`, `peg_findings.pptx`) |
| `requirements.txt` | Python dependencies (use the repo's `.venv`) |

### Legacy code (companion paper)

| path | contents |
|---|---|
| `experiment_scripts/mia/` | budget-manipulation attack and LiRA membership inference |
| `notebook_scripts/` | notebooks for the companion paper's figures |
| `opacus_new/` | modified Opacus with per-sample iDP budgets |
| `differential-privacy/` | external DP accounting library (gitlink) |
| `figures/` | companion-paper figures |

---

## 5. Not in git

Ignored on purpose (see `.gitignore`):

- experiment caches (`game_experiments/_cache_*/`), pickles, `.npy`, `.json`, `.csv`, `.png`;
- Slurm scripts (`*.sh`) and logs (`*.out`, `*.err`);
- pgfplots data tables (`*.dat`);
- source material: reference PDFs at the repo root and `sources/`
  (the writing-voice corpus `sources/voice/voice.md`, its `MANIFEST.md`, and the papers it was distilled from);
- LaTeX build intermediates, `.venv/`, `data/`.

A fresh clone therefore has neither caches nor `.dat` tables. Regenerate them
from `game_experiments/` and `tasks_findings/_scripts/` before rebuilding the
paper's pgfplots figures.

---

## 6. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m game._smoke          # quick sanity check of the solver stack
```
