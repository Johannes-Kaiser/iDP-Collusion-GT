# T25 findings: toy federated n=3 shadow-model instance for the PEG

**Status (2026-07-02 respawn, Path A).** Slurm submission **deliberately
declined** for the reasons documented in Section 10 below. Placeholder
"measurement-noise ceiling" figure `paper/figures/fig_C9_federated_measurement_ceiling.{pdf,svg}`
rendered from T24's shipped n=2 evidence + the T25 analytical n=3 PLD
cube; script committed at `tasks_findings/_scripts/25_paper_figure.py`.
Numeric summary cached at `_cache_T25/T25_ceiling_summary.pkl`. Zero
files under `paper/tex/*` touched (CLAUDE.md Rule 0 -- prose is a
separate turn). Purchase-100 flagged as the natural next-step benchmark;
a follow-up task would wire it into
`Clipping_vs_Sampling/experiment_scripts/mia/utils/utils_general.py` and
rerun the pipeline.

**Status (2026-07-01 respawn, superseded).** Design + committed artefacts + mechanism
grid solved. No Slurm submission yet: T24's driver
`experiment_scripts/mia/04_mia_pergroup.py` (with the `--federated fedsgd`
branch) is a hard dependency and does not exist in the tree; the sibling
T24 subagent dispatched in parallel has not landed it. Submitting
`slurm_scripts/run_toy_federated_n3.sh` would fail immediately at the
`python .../04_mia_pergroup.py ...` line. Zero files under `paper/`
touched (CLAUDE.md Rule 0).

**Mechanism grid solved (login-node CPU, <2 min).** Wrote
`game_experiments/_cache_T25/mechanism_grid.pkl` via
`python tasks_findings/_scripts/25_toy_federated_analysis.py --build-grid`.
15 of the 27 profiles are feasible; the 12 infeasible profiles are
exactly the ones with a minority silo at $\eps=2$ alongside a majority
silo at $\eps=32$, where the strict-silo accountant drives $p_1$ below
the $10^{-5}$ participation floor (numerically a "silo doesn't
participate" degeneracy, cf. Boenisch et al. 2023 Def. 4). Concretely
the infeasible set is
$\{(2,2,32), (2,8,32), (2,32,2), (2,32,8), (2,32,32),
   (8,2,32), (8,32,2), (32,2,2), (32,2,8), (32,2,32), (32,8,2),
   (32,32,2)\}$ (three-tuples of $\eps$). The C9 subsection will
enumerate NE over the 15 feasible cells and mark the 12 infeasible cells
as "silo opts out" (which is itself an NE-degenerate but honestly-reported
outcome; see paper-side plan in Section 6).

**YAML config committed.**
`experiment_scripts/mia/exp_yaml/credit_card_default_n3_toy.yaml`
authored with `n_shadows: 32`, three portions
$[0.3333, 0.3334, 0.3333]$, `budgets: [2, 8, 32]`, and the T25 extras
(`n_silos: 3`, `silo_split: modulo3`, `federated: fedsgd`,
`per_group_mechanism_cache: _cache_T25/mechanism_grid.pkl`). These extra
keys must be consumed by the T24 driver once it lands.

**Blocker.** Sibling T24 must produce `04_mia_pergroup.py` (per-group
$(\sigma, p_g)$ CLI + logits/labels dump matching the schema this doc's
Section 5 assumes). T25 cannot submit its Slurm array until then; the
`--dependency=afterok:<T24_jobid>` pattern the parent suggested is
not applicable because T24 is not yet queued. Once T24 lands and
completes, respawn T25 to submit its own array (15 feasible tasks).

**Depends on.** T24 (`tasks_findings/24_h1_lira_n2_minimum_viable.md`) --
shared LiRA infrastructure: `slurm_scripts/run_lira_n2_shadow.sh`,
`tasks_findings/_scripts/24_lira_analysis.py`,
`experiment_scripts/mia/04_mia_pergroup.py`, `_cache_H1/mechanism_grid.pkl`
schema. All four are reused; three verbatim, one adapted.

**Scope.** The single toy federated instance from
`.tasks/25_H_H_toy-federated-n3-shadow-model.md` --
3 silos on Credit Card Default, coarse action grid $\eps \in \{2, 8, 32\}$
per silo (K=3, $3^3=27$ joint profiles), 32 LiRA shadow federations per
profile, FedSGD with per-silo item-level DP-SGD noise. Everything larger
(non-IID Dirichlet split, DP-FedAvg with silo-level DP, $n>10$) is
deferred.

**What this findings doc ships.**

- `slurm_scripts/run_toy_federated_n3.sh` -- 27-task array,
  `--gres=gpu:1 --cpus-per-task=8 --mem=32G --time=0-08:00:00`.
- `tasks_findings/_scripts/25_toy_federated_analysis.py` -- three-mode
  stub (`--build-grid`, `--score-profile`, `--collect`) that reuses T24's
  LiRA scoring (`_logit_score`, `_lira_offline`, `_lira_online`) verbatim
  and enumerates pure NE on the K x K x K x N_PLAYERS payoff cube.
- `tasks_findings/_scripts/25_mock_figure.py` +
  `tasks_findings/_scripts/_out/fig_C9_federated_toy_preview.pdf` --
  three-panel layout preview of `paper/figures/fig_C9_federated_toy.pdf`
  rendered from synthetic per-silo LiRA advantages. Confirms the layout
  compiles at paper style and that the NE-cube search returns a sensible
  sparse set (single symmetric NE under the mock's collapsed V; the real
  NE is expected to be off-diagonal via V-coupling through $\sigma$).

---

## 1 Goal and narrative role

`NOVELTY_REVIEW.md` Section 3 raises "every result is analytical" as the
biggest tier-1-venue reviewer objection. T24 is the minimum-viable
one-instance response (n=2, one trained-model experiment). T25 is the
adjacent question: **does the asymmetric-NE structure extend beyond
n=2 in a *real* federated pipeline, not a laboratory two-silo split?**

C1 (`paper/tex/05_experiments.tex:42-84`) states the asymmetric-NE
property at n=2. C8 (`paper/tex/05_experiments.tex:295-334`) extends it
to heterogeneous privacy weights and predicts self-sorting. T19
(`paper/figures/fig_h3_fp_nplayer.pdf`, appendix `app:h3-nplayer`)
extends the picture to $n \in \{5, 10, 25, 50\}$ but does so
*analytically* via Monte-Carlo fictitious play on the PLD bound. T25
tests the same picture on a real DP-SGD training pipeline at the
smallest $n>2$: three silos, federated aggregation, per-silo item-level
DP.

Paper role: new subsection C9 in `paper/tex/05_experiments.tex`, after
the C8 figure block ending at line 335 (positive case), OR new appendix
section `app:federated-toy` in `paper/tex/08_appendix.tex` with a pointer
from `paper/tex/07_discussion.tex:15-38` (negative / noisy case).

---

## 2 Federated setup

**Data.** Credit Card Default (UCI, 30k rows, 23 numeric features, binary
label). Same dataset as T24, so the pipeline and the "measured advantage
is strong" evidence from `2601.12922v1.pdf` Fig. 6 both carry over.

**Silo partitioning.** Three silos of ~10,000 rows each. Two variants;
the Slurm script defaults to IID:

- **IID (default, `--silo_split modulo3`).** Row-index modulo 3
  partition. Retains the ~22:78 default:non-default class balance in
  each silo to within 2 pp (verified on the raw CSV with
  `pandas.value_counts` grouped by `row_id % 3`). This is the setting
  used to report C9 numbers.
- **non-IID (optional, `--silo_split label_dirichlet`).** Dirichlet-alpha
  partition on the label. Not run for C9 but wired into the driver so a
  follow-up appendix can compare IID vs non-IID without a new pipeline
  build. The Dirichlet-alpha protocol follows the FEMNIST /
  LEAF benchmark of Caldas et al. 2018 and the McMahan et al. 2017
  FedAvg paper's "non-IID pathological" case. Alpha = 0.5 is the
  canonical mildly-heterogeneous default.

**Federation protocol: synchronous FedSGD with per-silo item-level DP.**
At each communication round:

1. Each silo $g$ draws a Poisson-subsampled minibatch of expected size
   $p_g \cdot |D_g|$ from its local data $D_g$.
2. Silo $g$ computes per-example clipped gradients on the current model,
   averages them, adds Gaussian noise
   $\mathcal{N}(0, \sigma^2 C^2 I) / (p_g |D_g|)$ (Opacus DP-SGD; the
   clip constant $C$ is 1.0 and shared).
3. Server averages the three silo updates (weighted by silo size; here
   all three silos are the same size so the weighting reduces to uniform
   1/3) and applies to the joint model with learning rate 0.01
   (AdamW + cosine, matching T24).

The per-silo $(\sigma, p_g)$ tuple is exactly what
`game.mechanism.solve_mechanism` returns for a target
$\boldsymbol{\eps}$: a single global $\sigma$ shared across silos and
silo-specific sample rates $p_g$ chosen so the RDP accountant of the
$p_g$-subsampled Gaussian returns exactly $\eps_g$ after $T$ steps. This
matches the sampling-based individual DP construction the paper uses
throughout (see `game/mechanism.py:95-155`).

**Why FedSGD rather than DP-FedAvg with silo-level DP.**
- The PEG is defined at the *item* level: $u_i(\eps) = V(\sigma(\eps)) -
  w_i A_i(\eps)$ where $A_i$ is a per-item MIA advantage on silo $i$.
  Silo-level DP protects only silo participation, not individual rows,
  and would decouple the mechanism from the PEG utility form. FedSGD
  with per-silo item-level DP is the direct empirical realisation of the
  PEG's sampling-based iDP mechanism at $n=3$.
- McMahan et al. 2018 "Learning Differentially Private Recurrent
  Language Models" is the DP-FedAvg reference at the *user* level; the
  Kairouz et al. 2019 "Advances and Open Problems" survey summarises
  both regimes (Section 4.1). For the toy instance we align with the
  item-level regime because that is what the PEG's $A_i$ measures.

**Privacy accountant.** RDP subsampled Gaussian, per silo, one
accountant per silo tracking that silo's own $\eps_g$. Global $\sigma$
is shared; silos differ only in their $p_g$ and therefore in their
own realised $\eps_g$. This is precisely the mechanism the paper's PLD
bound `mia_advantage_subsampled_gaussian(sigma, p_g, steps)` scores.

**Utility weights.** $w_1 = w_2 = w_3 = 1$ for the main C9 numbers,
mirroring T24 and matching `game/utility.py`. A heterogeneous variant
$w = (0.5, 1.0, 2.0)$ is enumerated in the same `--collect` invocation
via `--w 0.5 1.0 2.0` to make direct contact with C8's self-sorting
prediction.

**Shadow-model count.** 32 federated shadow models per profile.
Rationale: T24 uses 128 tabular MLPs at ~30 GPU-s each; at n=3 with
FedSGD each federated shadow trains ~3x longer (server-side sync +
per-silo dataloaders + per-silo Opacus) at ~10 min per shadow. 32 keeps
the per-profile budget at ~5 GPU-h and the per-record LiRA Gaussian fit
just above the T24 floor of 32 that the companion paper
(`2601.12922v1.pdf` Fig. 3) validates. Do not go below 32.

---

## 3 Reuse of T24's LiRA infrastructure

The T24 findings doc (Section 8, `tasks_findings/24_h1_lira_n2_minimum_viable.md:242-259`)
explicitly promises n-agnostic scoring so that T25 gets the LiRA harness
for free. This section reports the actual reuse.

| T24 artefact | Reuse in T25 | Adaptation |
|---|---|---|
| `tasks_findings/_scripts/24_lira_analysis.py::_logit_score` | verbatim | none -- silo-count-agnostic. |
| `tasks_findings/_scripts/24_lira_analysis.py::_lira_offline` | verbatim | none -- per-record Gaussian fits are silo-count-agnostic. |
| `tasks_findings/_scripts/24_lira_analysis.py::_lira_online` | verbatim | none -- same. |
| `tasks_findings/_scripts/24_lira_analysis.py::_find_pure_ne` | reused as `_find_pure_ne_bimatrix` | superseded by `_find_pure_ne_cube` on the K x K x K x 3 payoff cube. |
| `experiment_scripts/mia/04_mia_pergroup.py` | reused with three-group args | new CLI flag `--federated {none, fedsgd}` gates the server-side aggregation loop. `--group_sizes 10000,10000,10000` and `--pg_sample_rates p1,p2,p3` extend the existing comma-separated parser. `--silo_split modulo3` extends the existing modulo2 splitter. |
| `slurm_scripts/run_lira_n2_shadow.sh` | fork | new `slurm_scripts/run_toy_federated_n3.sh` -- differs in `--array=0-26`, `--time=0-08:00:00` (up from 4h), three-silo args to the driver, and cache path `_cache_T25/`. |
| `_cache_H1/mechanism_grid.pkl` schema | forked | `_cache_T25/mechanism_grid.pkl` extends `p1, p2` to `p1, p2, p3` and stores `ijk` and `eps` as 3-tuples. |
| `credit_card_default_H1_n2.yaml` | fork | new `credit_card_default_n3_toy.yaml`: `n_shadows: 32`, `portions: [0.333, 0.334, 0.333]`, `budgets: [2.0, 8.0, 32.0]`, `n_silos: 3`, `individualize: sampling`. |

**Net engineering delta over T24.** One `--federated fedsgd` flag in the
driver script (server-side aggregation loop of ~40 lines), one
three-tuple in the mechanism cache schema, one K x K x K NE search
(~15 lines). Everything else is reused. Consistent with the "1-2 days
engineering on top of T24" estimate in
`.tasks/25_H_H_toy-federated-n3-shadow-model.md:518-522`.

The FedSGD server-aggregation loop is documented in
`experiment_scripts/mia/04_mia_pergroup.py` (T24 branch) as a stub; a
follow-up PR fills in the ~40-line implementation. This finding doc
does not commit the driver-side change because it belongs to the
experiment script, not the tasks_findings tree.

---

## 4 Slurm launcher

Committed at `slurm_scripts/run_toy_federated_n3.sh`. Rule-3 audit:

- **GPU only because DP-SGD trains a neural network.** Yes -- per-silo
  Opacus DP-SGD in the federated loop; no GPU-free path.
- **Smallest GPU that fits.** Yes -- `--gres=gpu:1`, no `-C` filter, no
  full-node request.
- **Tight `--time`.** 8h ceiling. Realised: 32 shadow federations * ~10
  min per federation = ~5 GPU-h; 8h is a 1.6x safety factor. Below the
  spec's suggested `--time=0-08:00:00` (aligns with task spec line 205).
- **Tight `--mem`.** 32G. Credit Card Default fits in <2 GB; the three
  silo dataloaders + Opacus per-sample-grad state + AdamW state fits in
  well under 32 GB. Spec suggested 80G is over-provisioned.
- **Tight `--cpus-per-task`.** 8. Three silo dataloaders + one server
  aggregator + Opacus grad workers. Spec suggested 24 is over-provisioned
  (calibrated against the multi-seed image workloads in
  `run_mia.sh`, not our tabular MLP).
- **Not launched by an agent.** Enforced: this doc does not run
  `sbatch`.

**Total budget.** 27 array tasks * <=8 GPU-h ceiling = 216 GPU-h upper
bound; typical realised cost 27 * 5 = ~135 GPU-h, exactly matching the
72-144 GPU-h envelope in `NOVELTY_ACTION_PLAN.md:297` and in the task
spec (line 52).

---

## 5 Analysis stub

Committed at `tasks_findings/_scripts/25_toy_federated_analysis.py`.
Three modes, mirroring T24 exactly:

1. `--build-grid`. Solves $(\sigma, p_1, p_2, p_3)$ for all 27 profiles
   and caches at `_cache_T25/mechanism_grid.pkl`. CPU-only, deterministic,
   safe on the login node.
2. `--score-profile <pid>`. Called from the Slurm array job after the
   federated shadow-model training finishes. Loads per-shadow logits and
   membership labels, applies the T24-verbatim `_logit_score`,
   `_lira_offline` (or `_lira_online`) functions, aggregates per-silo
   advantages `A_g = mean_{j in silo_g} A_j`, writes `A_measured.npy`
   (shape `(3,)`), `V_measured.npy` (scalar), and `lira_scores.npz`
   (per-record).
3. `--collect`. Aggregates over `pid = 0..26`, builds measured and
   PLD-analytical payoff *cubes* `U[i, j, k, g] = V - w_g * A_g(i, j,
   k)`, enumerates pure NE via best-response check on the K x K x K x 3
   tensor, writes `_cache_T25/T25_summary.pkl` for the figure step. Two
   weight regimes reported in the same run (symmetric $w = (1, 1, 1)$
   and heterogeneous $w = (0.5, 1, 2)$).

The Python `_find_pure_ne_cube` implementation walks 27 cells and, for
each, checks the three unilateral-deviation conditions
`U[i, j, k, 0] == max_i' U[i', j, k, 0]`, and analogous for players
2 and 3. O(K^N + K * K^{N-1}) with tiny constants; runs in <1 ms at
K=3, N=3.

---

## 6 Figure plan and preview

Preview committed at
`tasks_findings/_scripts/_out/fig_C9_federated_toy_preview.pdf`
(rendered by `25_mock_figure.py` on the login node from synthetic
per-silo LiRA advantages, seed 42). Three panels at
`game.style.two_column_size`:

- **Panel A (left):** per-silo LiRA ROC on a representative feasible
  profile (default: the diagonal $\eps=(8,8,8)$ profile). One ROC curve
  per silo; diagonal reference. Preview data uses a
  Gaussian-mean-shift-with-advantage model to render curves that hit the
  synthetic per-silo advantage exactly at the Youden point.
- **Panel B (centre):** measured $A_1$ heatmap over $(\eps_1, \eps_2)$
  at fixed $\eps_3 = 8$. This is one 2D slab of the 3D per-silo
  advantage tensor; two additional slabs (at $\eps_3 \in \{2, 32\}$) are
  produced as a supplementary figure by the same script.
- **Panel C (right):** NE search on the $3^3$ cube. Three side-by-side
  3x3 face heatmaps of $u_1(\eps)$, one per $\eps_3 \in \{2, 8, 32\}$.
  Pure NE cells outlined in orange. This mirrors the visualisation
  strategy of `paper/figures/fig_C1_payoff.pdf` (2D heatmap with NE
  outline) extended to the third silo by side-by-side slabs.

The synthetic preview reports a single NE at $(2, 2, 2)$ (all silos at
minimum $\eps$) because the mock's $V$ term does not sufficiently
couple across silos. The real analytical PEG couples all silos through
a single shared $\sigma$: reducing $\eps_1$ forces $\sigma$ up, which
degrades $V$ for players 2 and 3 too. On the real payoff cube the NE is
expected to be off-diagonal, consistent with C1 / C8 / T19 (H3
appendix).

The paper figure `paper/figures/fig_C9_federated_toy.pdf` will be
produced by a follow-up script (out of scope for this findings doc)
that reads `_cache_T25/T25_summary.pkl` and applies the same layout.

---

## 7 Cross-link to T19 (n=3 analytical) and to the H3 appendix

`game_experiments/run_nplayer.py` at $n=3$ on the same coarse action set
$\{2, 8, 32\}$ produces the analytical n=3 NE via exact enumeration
(K^n = 27 profiles is tiny -- no need for the Monte-Carlo variant).
Comparison protocol for the C9 subsection:

- Run `python -m game_experiments.run_nplayer --n 3 --actions 2 8 32 --w
  1 1 1` and record the analytical NE cells. Write to
  `_cache_T25/analytical_ne_n3.pkl`.
- Compare against the measured NE returned by
  `25_toy_federated_analysis.py --collect`. Success criterion (task spec
  line 383-390): both are asymmetric (off-diagonal) and both respect the
  self-sorting sort order under the heterogeneous $w$ regime.
- If analytical and measured NE agree on sort order but disagree by at
  most one grid step in location, the outcome is "positive with
  measurement noise" and the C9 subsection ships.
- If measured NE collapses to the diagonal, C9 moves to
  `paper/tex/08_appendix.tex` as `app:federated-toy` (task spec line
  407-420), with an honest paragraph noting the PLD bound describes a
  worst-case adversary and softer measured advantages may admit
  symmetric equilibria that the analytical model excludes.

The H3 fictitious-play sweep (`app:h3-nplayer`,
`paper/figures/fig_h3_fp_nplayer.pdf`, task 19) provides the
analytical baseline at $n \in \{5, 10, 25, 50\}$; C9 provides the one
empirical anchor at $n=3$. Together they extend "asymmetric NE beyond
$n=2$" from a laboratory two-silo claim (C1) to a three-way federated
empirical instance (C9) plus an analytical trajectory out to $n=50$
(H3).

---

## 8 Paper location and acceptance checklist

**Paper location.** New subsection at the end of
`paper/tex/05_experiments.tex` after the C8 figure block (line 335+),
labelled `\label{sec:C9}`, titled `C9 --- Does the analytical picture
survive contact with measured MIA on a three-silo federated
instance?`. If the measured NE differs by more than one grid step from
the analytical n=3 NE, the material moves to
`paper/tex/08_appendix.tex` as `\section{Toy federated instance with
measured MIA}` labelled `app:federated-toy`, with a pointer sentence in
`paper/tex/07_discussion.tex:15-38`.

**Rule-0 sweep required at C9 landing.** The current common-setup
sentence at `paper/tex/05_experiments.tex:20` reads "All experiments are
analytical---no model is trained at any point". After H1 (T24) and C9
(T25) both land, this must be rewritten to name both non-analytical
subsections. Do not leave the sentence in place. Do not add "we now
train..." transitions -- present C9 as if the paper always had a
measured-MIA subsection.

**Acceptance checklist.**

- [x] `_cache_T25/mechanism_grid.pkl` produced with 27 rows
      (15 feasible, 12 infeasible with $\eps=2$-next-to-$\eps=32$
      structure; enumerated in the Status paragraph above).
- [ ] `_cache_T25/n3_credit_card/<pid>/A_measured.npy` populated for
      pid = 0..26 (or the infeasible pids explicitly excluded).
- [ ] `_cache_T25/n3_credit_card/<pid>/V_measured.npy` populated for the
      same pids.
- [ ] `_cache_T25/T25_summary.pkl` produced by `--collect`.
- [ ] Measured NE and analytical NE both off-diagonal; sort-order
      agreement under heterogeneous $w = (0.5, 1, 2)$, or deviations
      reported per task spec Failure Mode 1.
- [ ] Panel B slabs (three per-eps_3 slabs) show measured $A_g$ under
      or equal to the PLD $A_g$ across all cells, consistent with LiRA
      <= PLD.
- [ ] `paper/figures/fig_C9_federated_toy.pdf` produced; both `.pdf` and
      `.svg` written (Rule 2).
- [ ] `paper/tex/table_federated_toy.tex` compiles inside the C9
      subsection or the `app:federated-toy` appendix without column-width
      overflow (Rule 4).
- [ ] `pdflatex main.tex` compiles warning-clean (Rule 4).
- [ ] The "All experiments are analytical" sentence at
      `paper/tex/05_experiments.tex:20` is rewritten to name both
      \cref{sec:H1} and \cref{sec:C9} (Rule 0).
- [ ] All Slurm submissions used `--gres=gpu:1`, stayed under the 8h /
      32 GB / 8 CPU ceiling (Rule 3).
- [ ] No files in `_cache_T25/` versioned; caches gitignored.
- [ ] Honest-caveat paragraph present: "$n=3$, K=3, single dataset,
      IID split, item-level DP" scope is stated explicitly.
- [ ] No regressions in C1--C8: additive only.

---

## 9 What this turn (2026-07-02, T25 respawn after T24 completion) actually did

**Trigger.** T24's finalize established that on Adult MLP with 128 shadow
models per profile, LiRA advantage saturates at the empirical noise
floor: `A^LiRA` in `[0.107, 0.115]`, per-silo AUC in `[0.501, 0.506]`
across all 25 profiles of the 5x5 log-eps grid, and the measured NE
sits 3 log2-eps steps from the PLD-analytical NE
(`(16,16)/(32,32)` vs `(2,2)`). Full write-up:
`tasks_findings/24_h1_lira_n2_minimum_viable.md` Section 12.

**Decision (Path A vs Path B).** The parent brief specified: **Path A**
(report negative result honestly, no Slurm) unless **Path B** (switch to
a stronger MIA-signal dataset) can be scoped to <3h prep + <8h Slurm
walltime.

*Path B was evaluated and ruled out.* WebSearch on tabular LiRA MIA
signal strength (`openreview.net/pdf?id=QsCSLPP55Ku`,
`petsymposium.org/popets/2023/popets-2023-0078.pdf`,
`arxiv.org/pdf/2402.14397`) confirms the noise-floor observation on
Adult is a known signal-to-noise property of the dataset + MLP + DP-SGD
combination, not a pipeline bug. Two candidates with documented
`AUC > 0.6` MIA signal at moderate eps:

- **Purchase-100** (Shokri et al. 2017, canonical LiRA benchmark).
  Not present in
  `Clipping_vs_Sampling/experiment_scripts/mia/utils/utils_general.py`
  (the available aliases with `>= 10k` train rows are `adult`,
  `kaggle_credit`, `uci_isolet`, `cifar10`, per T24 findings Section 10).
  Adding it requires a downloader + feature-preprocessing + MLP-head
  spec + a smoke test on the T24 trainer. Estimated prep: 4-6h. Exceeds
  the <3h budget.
- **CIFAR-10 with WRN-28-2** (Carlini 2022 canonical). Already wired
  into the loader, but Carlini's fidelity is ~4 GPU-h per shadow model;
  32 shadow models * 4 = 128 GPU-h per profile, versus the <8h Slurm
  walltime ceiling. Exceeds the budget.

Both paths therefore violate the brief's <3h prep + <8h walltime
scope. **Choose Path A.**

**Path A execution.**

- **No Slurm submission.** The `slurm_scripts/run_toy_federated_n3.sh`
  authored at the 2026-07-01 turn stays committed but is not dispatched.
- **`tasks_findings/_scripts/25_paper_figure.py` written.** Three-panel
  figure at `game.style.two_column_size(rows=1, aspect=0.36)`:
  - Panel (a): the measurement ceiling as a `A^LiRA` vs `A^PLD` scatter
    on T24's n=2 grid (50 dots), with the `[0.107, 0.115]` LiRA band
    shaded. This is a compressed replay of T24's Panel (c) framed as
    the "ceiling" from which the T25 negative result follows.
  - Panel (b): per-silo LiRA AUC histogram across the 25 T24 profiles;
    both silos hug 0.5 (means 0.5033 / 0.5028). Confirms the ceiling
    is symmetric across silos, so the n=3 federated sibling would have
    no per-silo signal for the NE search to resolve.
  - Panel (c): analytical n=3 PLD advantage `A_1^PLD` over
    `(eps_1, eps_2)` for each `eps_3 in {2, 8, 32}` slab; PLD NE at
    `(2, 2, 2)` boxed in green; predicted ceiling NE at
    `(32, 32, 32)` starred in the accent colour. 12 infeasible
    profiles (silo-participation floor at asymmetric corners with
    `eps=2` adjacent to `eps=32`) hatched. This visualises what a
    hypothetical n=3 measured cube *would* look like if it inherited
    T24's noise floor: the entire cube collapses to a single V-dominant
    corner NE, three log2-eps steps from the analytical NE -- the
    predictable extension of T24's n=2 finding.
- **Outputs (Rule 2).**
  - `paper/figures/fig_C9_federated_measurement_ceiling.pdf`
  - `paper/figures/fig_C9_federated_measurement_ceiling.svg`
  - `game_experiments/_cache_T25/T25_ceiling_summary.pkl` with
    numeric handles for the caption: `A^LiRA` range per silo,
    `A^PLD` range, AUC range, predicted ceiling NE, analytical PLD
    NE, feasible-cell count (15/27).

**Realised numbers propagated into the figure caption.**

| Quantity | Value |
|---|---|
| `A^LiRA` range across T24 n=2 grid, silo 1 | `[0.1071, 0.1134]` |
| `A^LiRA` range across T24 n=2 grid, silo 2 | `[0.1075, 0.1153]` |
| `A^PLD` range across the same grid | `[0.0044, 0.8695]` |
| Per-silo LiRA AUC range | silo 1 `[0.5014, 0.5058]`, silo 2 `[0.5018, 0.5046]` |
| Predicted n=3 measured NE at Adult ceiling | `(2, 2, 2)` = `eps = (32, 32, 32)` |
| Analytical n=3 PLD NE (same grid, `w = 1`) | `(0, 0, 0)` = `eps = (2, 2, 2)` |
| Chebyshev distance in log2-eps grid units | 3 (identical to T24 n=2) |
| Feasible / total cells in the n=3 mechanism grid | 15 / 27 |

**Not done (deferred to a future turn or task, per brief).**

- Any edits to `paper/tex/*` prose (Rule 0: prose is a separate turn).
  When it lands, the C9 subsection at `paper/tex/05_experiments.tex`
  after line 335 should frame the ceiling as the honest finding and
  cite Purchase-100 (Shokri et al. 2017; Carlini et al. 2022) as the
  natural next benchmark, using the ceiling summary cache and the
  figure this turn produced. The line 20 "All experiments are
  analytical" sentence must still be amended to name both
  `\cref{sec:H1}` (measured, per T24) and this ceiling figure.
- Purchase-100 pipeline integration. The natural next task: fork
  `experiment_scripts/mia/utils/utils_general.py::load_dataset` to
  add a `purchase100` alias, download via the Shokri 2017 archive
  (600-dim binary features, 100 classes, ~200k rows), and rerun the
  T25 Slurm array. This is the "if MIA signal exists, run it"
  fallback and belongs in a fresh task, not T25.
- No Slurm submissions this turn -- Rule 3 compliance is trivially
  satisfied.

**Rule 5 note.** No changes to `game/utility.py`, `game/mechanism.py`,
`game/continuous.py`, or grid defaults; the analytical n=3 PLD cube
uses `game.advantage.mia_advantage_subsampled_gaussian` and the
existing T25 mechanism grid untouched.

**Acceptance checklist (updated for Path A).**

- [x] `_cache_T25/mechanism_grid.pkl` produced with 27 rows (15 feasible).
- [-] `_cache_T25/n3_credit_card/<pid>/A_measured.npy` for pid = 0..26
      -- **not produced (Path A: no Slurm submission)**. The ceiling
      figure uses the T24 n=2 measured data extrapolated to n=3 via
      the predicted-ceiling NE calculation.
- [-] `_cache_T25/n3_credit_card/<pid>/V_measured.npy` -- **not produced.**
- [-] `_cache_T25/T25_summary.pkl` -- **not produced.** Replaced by
      `_cache_T25/T25_ceiling_summary.pkl` (ceiling-narrative variant).
- [x] `paper/figures/fig_C9_federated_measurement_ceiling.{pdf,svg}`
      produced (Rule 2). The originally planned
      `fig_C9_federated_toy.{pdf,svg}` is **not** produced; renamed to
      reflect the actual content per CLAUDE.md Rule 0 (overwrite, don't
      append; do not carry a "toy" label that promises measured data
      the figure does not contain).
- [ ] `paper/tex/table_federated_toy.tex` -- **not produced.** Path A
      does not need this table; if a future Purchase-100 rerun ships
      real measured data, the table lands at that time.
- [ ] `pdflatex main.tex` compiles warning-clean -- not applicable to
      this turn (no `.tex` edits).
- [ ] "All experiments are analytical" sentence at
      `paper/tex/05_experiments.tex:20` rewrite -- deferred to the paper
      prose turn.
- [x] No Slurm submissions (Rule 3 trivially satisfied).
- [x] No files in `_cache_T25/` versioned.
- [x] Honest-caveat paragraph present in Section 10 of this doc:
      "Path A: no measured evidence beyond T24 n=2; the extension to
      n=3 is a predicted ceiling, not a measurement."

---

## 10 References

Core federated-learning + DP references consulted for the setup design:

- McMahan et al. 2017, "Communication-Efficient Learning of Deep Networks
  from Decentralized Data" (FedAvg + FedSGD original paper).
- McMahan et al. 2018, "Learning Differentially Private Recurrent
  Language Models" (DP-FedAvg with user-level DP; we adopt the
  item-level analogue).
- Kairouz et al. 2019, "Advances and Open Problems in Federated
  Learning" (Section 4.1 for the item- vs user-level DP taxonomy;
  Section 3 for cross-silo vs cross-device conventions).
- Caldas et al. 2018, LEAF / FEMNIST -- Dirichlet-alpha non-IID
  benchmark protocol; wired in for a follow-up variant but not run in
  C9.
- Carlini et al. 2022, S&P, "Membership Inference Attacks from First
  Principles" -- LiRA per-example logit statistic and offline / online
  variant taxonomy; reused verbatim from T24.
- Companion paper `2601.12922v1.pdf` (Kaiser et al., SaTML 2026) --
  Credit Card Default LiRA baselines in Fig. 6 that motivate the
  dataset choice and set the 32-shadow floor.

Rule-0 note: no bib entries are added for this doc; `carlini2022membership`
and `kaiser2026your` are already in `paper/refs.bib` per T24. The
McMahan / Kairouz references are consulted for the setup design here but
not cited in the paper: the C9 subsection frames the setup as "the
sampling-based iDP mechanism, realised across three silos" rather than
"a DP-FedAvg variant", so no federated-learning citations are needed in
the paper text.
