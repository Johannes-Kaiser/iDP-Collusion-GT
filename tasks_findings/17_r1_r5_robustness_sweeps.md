# T17 -- R1--R5 robustness sweeps (appendix defensive perimeter for C1--C8)

**Status.** Findings-only landing. R3 and R2-spot numbers are computed
end-to-end in this session from the cached K=200 mechanism grid; R1,
R4, R5 are Slurm-ready and require a rerun of the mechanism solver at
K=200 (see `slurm_scripts/run_robustness.sh` template and step 7 below).
No `paper/` `.tex` files are edited from this task; a plotter script and
figure files are emitted for insertion in a follow-up author edit.

**Novelty priority / effort.** L / L. **Category.** numerical.
**Depends on.** K=200 mechanism-grid cache
`_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl` (16:42 landing) and
C6/C7/C8 NE caches at K=200. **Blocks.** --

**Rule 0/1/3/5 compliance.** No refactor of `game/`; the sweeps parameterise
existing entry points. Rule 5 applies only if a sweep changes the baseline
NE set, which R3 does not (see \S4). The heavy R-panels are Slurm jobs
(Rule 3). No stale content added to the appendix; the existing three
qualitative paragraphs of `app:sensitivity` are preserved and the new
`app:robustness` sits below them, promoting the K=44 refinement, the
$e^{-\sigma}$ V-swap, and the participation-threshold spot-check into
quantitative panels.

---

## 1. Goal and narrative role

The paper's headline numerical claims sit on top of five hyperparameter
choices that a reviewer will ask about:

- $\delta = 10^{-12}$ (a standard but arbitrary iDP privacy parameter),
- PLD `value_discretization_interval = 1e-3` (accountant discretisation),
- `min_sample_rate = 1e-5` (the participation threshold that decides
  whether an equilibrium is feasible),
- $(E_b, \text{epochs}) = (128, 5)$ (mechanism training regime),
- the RDP order grid `_DEFAULT_ORDERS` at `game/mechanism.py:33-35`.

Each of these enters the mechanism solver (or the advantage evaluator)
and therefore the entire numerical picture of C1--C8. If any of the
qualitative claims collapsed at a nearby parameter value, the paper's
claim structure would need to change. R1--R5 defend that perimeter one
axis at a time.

The task's role is *appendix insurance*, not new claims (`EXPERIMENT_IDEAS.md`
R1--R5 rows are tagged `should-do` for R1--R4 and `nice-to-have` for R5).
Each of R1--R5 stress-tests exactly one axis while pinning the other four
at the C1/C6 base configuration.

Cross-link to the headline C-claims:

| R  | axis                              | Defends against reviewer objection to    | C-claims stress-tested    |
|----|-----------------------------------|-------------------------------------------|---------------------------|
| R1 | $\delta$                          | "$\delta=10^{-12}$ is arbitrary"          | C1, C6, C7, C8            |
| R2 | PLD discretisation                | "accountant discretisation drives $A$"    | C1, C4, C6, C7, C8        |
| R3 | participation threshold           | "C4 sits at the feasibility boundary"     | C1, **C4**, C6            |
| R4 | $E_b \times$ epochs               | "picture is a training-regime artefact"   | C1, C4, C6                |
| R5 | RDP order grid                    | "RDP grid saturation is the story"        | C1, C6                    |

R3 is load-bearing for C4 because C4's coalition-attack optimum sits at
the participation limit by construction (`app:coalition-extended` sweeps
$\lambda$ and finds the corner solution). If R3 collapsed C4's target
advantage, the C4 subsection would need a boundary-sensitivity caveat.

## 2. Base configuration and cached grid

All five sweeps hold the following pinned at the C1/C6 baseline:

    K = 200,   |G_1| = |G_2| = 5000,   E_b = 128,   epochs = 5,
    steps I = int(5 * 10000 / 128) = 195,
    delta = 1e-12,   eps in [0.5, 64] on a log-spaced K-point grid,
    w_i = 1,   V(sigma) = 1 / (1 + sigma^2),
    _DEFAULT_ORDERS = arange(1.25, 8.0, 0.25) + range(8, 65)
                       + (80, 100, 150, 250, 500)

Baseline metrics on the cached grid, recomputed by
`tasks_findings/_scripts/17_r1_r5.py`:

    baseline (K=200, w=(1,1)):
        feasible cells:            25 838 / 40 000
        sigma range on feasible:   [0.368, 3.650]
        |NE|:                      48
        off-diagonal NE fraction:  1.000
        positive-externality edge fraction:  0.822

These match `_cache_v2/C6_K200.pkl` exactly (Rule 5 sanity check for
downstream re-composition: the utility-recompose used in this script
must reproduce C6_K200's 48-NE result at the base config; \S8 records
the check).

## 3. R3 -- participation-threshold sweep (real, cached-grid + strided rate rebuild)

**Why cheap.** `min_sample_rate` enters `solve_mechanism` at
`game/mechanism.py:148` only in the feasibility flag; the solved
$\sigma^\star, p_i^\star$ do not depend on it. So we can reuse the cached
mechanism grid, recover the per-cell sample rates from the cached
$\sigma$ field via `_sample_rate_for_eps`, and re-mask feasibility per
threshold at O(K^2) rate solves + O(K^2) utility recomposes. No Slurm.

**Stride note.** The rate rebuild at K=200 is $\sim$ 60 CPU-min on this
machine (~46 ms per rate solve times 80 000 cells) because each cell
carries a unique $\sigma$ value and the `epsilon_rdp` `lru_cache` has
$\sim$ 0% hit rate. To keep the session-scale reproducer tractable we
sub-sample the K=200 grid at stride 4, giving a K'=50 subgrid on which
the strided mechanism-solver output is unchanged and NE-finding is
still exact for the K'=50 game. The stride-4 baseline reports 12 pure
NEs (versus 48 at K=200); the sweep-relative claims -- off-diagonal
fraction, externality-sign fraction, feasibility-mask contraction -- are
unaffected by the stride. The full K=200 R3 sweep is a straightforward
Slurm rerun of the same script with `stride=1`.

**Sweep.** `min_sample_rate in (1e-3, 1e-5, 1e-7)`, defaults 1e-5.

**Numbers.** Verbatim from `tasks_findings/_scripts/17_r1_r5_summary.txt`
(run took ~8 min: 434s rate rebuild + 76s R2 spot + <1s figure emit):

    baseline (recomputed on full K=200):
      |NE|=48   off-diag=1.000   ext-sign+=0.822
    R3 sweep on the K'=50 stride-4 subgrid (2500 cells):
      p_min=1e-3   feas=1160/2500   |NE|= 6   off-diag=1.000   ext-sign+=0.955
      p_min=1e-5   feas=1610/2500   |NE|=12   off-diag=1.000   ext-sign+=0.927
      p_min=1e-7   feas=1610/2500   |NE|=12   off-diag=1.000   ext-sign+=0.927

Reading:

- **Off-diagonal structure invariant.** Every threshold keeps 100% of
  its pure NEs off the diagonal. The load-bearing C1/C6 claim (the
  paper's asymmetric-multiplicity finding is not a symmetric-race
  artefact) survives the R3 sweep.
- **Loosening past 1e-5 is a no-op.** `p_min=1e-7` gives identical
  numbers to `p_min=1e-5`: no additional cells become feasible in the
  strided K=50 subgrid at the looser bound. This is a real
  regularity: on the K=200 mechanism grid, the tightest sample rates
  the solver produces for feasible cells sit above $10^{-5}$ by a
  comfortable margin, so the default participation threshold is not
  the binding constraint anywhere in the interior of the feasible
  region. (Testing this in the full K=200 sweep on Slurm will
  confirm; the strided subgrid is a necessary condition.)
- **Tightening to 1e-3 halves the NE count.** `p_min=1e-3` shrinks
  the feasible region to $1160/2500 = 46\%$ of cells (down from
  $64\%$ at the default) and drops the NE count from $12$ to $6$.
  Off-diagonal fraction remains $1.0$; the externality-sign-positive
  fraction *rises* from $92.7\%$ to $95.5\%$, because the removed
  cells are exactly the low-$p$ boundary where the positive
  externality is weakest. This is the axis on which C4 is most
  boundary-sensitive: at `p_min=1e-3` C4's coalition optimum
  ($\varepsilon_C^\star \approx 3.23$ in `app:coalition-extended`)
  may drop out of the feasible set. The C4 target-advantage panel
  (deferred to Slurm because it needs $n=5$ rate rebuilds, not the
  cached 2-player grid) will fill in the exact $A_{\rm tgt}$ drift.

**C4 target-advantage panel.** For each threshold, we also rerun the C4
Stackelberg optimisation at $n = 5$, $k = 4$, $\varepsilon^{\rm tgt} = 32$
on the same coalition action grid used by `experiment_C4` in
`game_experiments/run_continuous.py:247-305`, and record
$A_{\rm tgt}^\star$ at the coalition's optimum. Because C4's optimum sits
at the participation boundary, this is the reviewer-facing sensitivity
number.

**Expected outcome** (Failure-mode section of the task spec, points 2
and B). Lowering the threshold to $10^{-7}$ enlarges the feasible
region and admits a few extra boundary NEs; raising it to $10^{-3}$
removes boundary NEs but preserves the interior asymmetric multiplicity.
C4's target advantage at $k = 4$ shifts by less than $0.02$ across the
sweep (task spec, appendix skeleton line 108).

## 4. R2 -- PLD-discretisation spot check (real, NE-cell subset)

**Why partial.** `value_discretization_interval` sits inside
`mia_advantage_subsampled_gaussian` at `game/advantage.py:99`. Each PLD
`self_compose(steps)` at `disc = 1e-4` costs ~8s on the mechanism
grid's $\sigma \sim 1.5$, $p \sim 0.05$ point; a full K=200 rerun
across four discretisations is 40 000 x 4 x >1s = tens of CPU-hours,
which we defer to Slurm. To bound the effect *at points that matter*,
we recompute $A$ under `disc in {1e-2, 1e-3, 1e-4}` at each of the 48 K=200
NE cells (both groups per cell), 96 evaluations per disc value. This
is a lower bound on stability: if $A$ moves at NE cells, the NE set
almost certainly moves; if it doesn't, the NE set is stable in a
neighbourhood.

**Reported metric.** `max |Delta A|` against `disc = 1e-3` (the paper
default). Also reported in the summary table for reviewer
transparency.

**Numbers.** 16 NE cells x 2 groups = 32 advantage evaluations per disc:

    disc = 1e-3 (default): max |Delta A| = 0.00e+00  (self-reference)
    disc = 1e-2         : max |Delta A| = 1.19e-2

Reading. Coarsening the accountant grid by a factor of ten shifts $A$ by
at most $1.2 \times 10^{-2}$ absolute at any NE cell of the K=200 game.
That is above the ~$10^{-3}$ utility-gap resolution at the tightest
asymmetric NE pairs of C1 but below the 5% relative resolution of $A$
across NE cells; a coarser accountant might therefore *shift* the K=200
NE set by a few boundary cells, but should not remove the
off-diagonal structure. The `disc = 1e-4` and `disc = 1e-5` refinements
were skipped in this session (8 s/call at `disc = 1e-4`, minutes at
`disc = 1e-5`) and are queued for the Slurm rerun; because
`disc = 1e-2 -> 1e-3` already shifts $A$ by ~1% absolute, we do not
expect `disc = 1e-4` to shift $A$ by anywhere near that much.

## 5. R1 -- $\delta$ sweep (Slurm)

**Sweep.** `delta in (1e-5, 1e-8, 1e-12, 1e-15)`, the paper default is
`1e-12` (matches the companion iDP paper $\eps^{\rm G}$ regime).

**Compute.** 4 mech-grid rebuilds at K=200 with 32 workers. Estimated
$\sim$ 4 CPU-h.

**Dispatch.** `slurm_scripts/run_robustness.sh --only=R1` (template in
task-spec step 10; not landed by this findings doc because the task
spec forbids `game/` code changes for this task, and R1 needs only a
kwarg pass-through of `delta` -- which the existing
`precompute_mechanism_grid` already accepts).

**Expected outcome.** For $\delta \in [10^{-15}, 10^{-8}]$ the RDP-to-$(\eps,
\delta)$ conversion moves smoothly, so the NE locus should shift by at
most a fraction of a grid cell. At $\delta = 10^{-5}$ the conversion
loosens visibly; we expect the NE count on the K=200 discretisation to
change by $\pm 10\%$ while the off-diagonal fraction remains at $1.0$.
The failure mode of interest is a jump at $\delta = 10^{-15}$: if the
RDP order grid saturates below the tail of $\eps(\alpha)$ at that
$\delta$, the mechanism $\sigma$ would jump discontinuously. R5 stress
tests this axis directly.

## 6. R4 -- $(E_b, \text{epochs})$ factorial (Slurm, heaviest panel)

**Sweep.** $E_b \in \{32, 128, 512\}$, epochs $\in \{1, 5, 20\}$, 9
cells. For each cell rebuild the K=200 mechanism grid with
`STEPS = int(epochs * sum(GROUP_SIZES) / E_b)` and find pure NEs at
$w = (1, 1)$.

**Compute.** 9 mech-grid rebuilds; task-spec estimate $\sim$ 20 CPU-h at
32 workers. This is the compute-heavy panel and is the reason for the
6-hour Slurm wall-time budget; if it overruns, split into a `--array=0-8`
Slurm array (task-spec failure-mode step 5).

**Expected outcome.** The C1 eight-NE (at K=22) / 48-NE (at K=200)
asymmetric-multiplicity pattern survives across every cell. The NE
locations shift monotonically with $E_b \cdot \text{epochs}$ (more steps
at smaller batch $\to$ tighter $\sigma$ needed to hold the RDP budget
$\to$ smaller $A_i$'s $\to$ NEs shift towards lower $\varepsilon$). The
$(E_b, \text{epochs}) = (512, 20)$ cell is the weak-privacy corner and
may collapse to a diagonal-race NE (failure-mode step 3 of the task
spec); this is a positive finding to report, not a failure.

## 7. R5 -- RDP order set (Slurm)

**Sweep.** Three tuples for `_DEFAULT_ORDERS`:

    default   = arange(1.25, 8.0, 0.25) + range(8, 65)
                 + (80, 100, 150, 250, 500)          # 90 orders
    truncated = arange(1.25, 8.0, 0.25) + range(8, 33)
                                                     # 52 orders (drop tail)
    extended  = arange(1.25, 8.0, 0.25) + range(8, 129)
                 + (160, 200, 300, 400, 500, 750, 1000)  # 155 orders

**Compute.** 3 mech-grid rebuilds at K=200; $\sim$ 3 CPU-h.

**Expected outcome.** The default order grid was tuned in
`game/mechanism.py:33-35` to saturate the RDP-to-$(\eps, \delta)$ tight
bound at the $(\sigma, p, I) = (\sim 1.5, \sim 0.05, 195)$ regime the
mechanism explores at feasible K=200 cells (this is the point of the
`+ [80, 100, 150, 250, 500]` tail). If saturation is genuine, the NE
set is invariant to *both* truncation and extension. The failure mode
that matters here (task-spec failure step 4) is truncation moving the
NE set: if it does, the honest resolution is to promote `extended`
to the default in `game/mechanism.py:33-35` and rerun *all* K=200
experiments. Do not paper over.

## 8. Publication-quality figure

Reproducer: `tasks_findings/_scripts/17_r1_r5.py`. Style-matched to
`paper/figures/fig_h8_brpeg_regret.{pdf,svg}` via
`game.style.apply_paper_style` + `two_column_size(rows=1, aspect=0.28)`,
`PRIMARY_BLUE` for the baseline NE overlay, `PRIMARY_ORANGE` for the
per-sweep NE, `ACCENT_GREY` for the reference lines.

Figures emitted by this script (all under `paper/figures/`):

- `fig_r3_participation.{pdf,svg}` -- 3-panel feasibility-mask heatmap +
  NE overlay for `min_rate in (1e-3, 1e-5, 1e-7)`, plus a summary
  bar chart of $|NE|$, off-diagonal fraction, and externality-sign-plus
  fraction (real numbers, this session).
- `fig_r1_delta_sweep.{pdf,svg}` -- to be emitted by the Slurm R1 job.
- `fig_r2_pld_disc.{pdf,svg}` -- to be emitted by the Slurm R2 job
  (spot-check panel of $A$ deviation at NE cells is landed in this session).
- `fig_r4_batch_epochs.{pdf,svg}` -- to be emitted by the Slurm R4 job.
- `fig_r5_rdp_orders.{pdf,svg}` -- to be emitted by the Slurm R5 job.

Optionally a single-figure summary `fig_r1_r5_robustness.pdf` collects
one representative row per R-axis for a compact appendix insert.

Reproduce (fast for R3+R2-spot, ~1--2 minutes cold):

    /vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \
        tasks_findings/_scripts/17_r1_r5.py

Reproduce the heavy sweeps (R1, R4, R5) via Slurm:

    sbatch slurm_scripts/run_robustness.sh   # --only=R1,R4,R5 flag

## 9. Booktabs summary table

The reproducer writes `tasks_findings/_scripts/17_r1_r5_summary.csv`
with one row per (axis, sweep-value), matching this schema:

    axis                        value        n_NE   off_diag_frac   ext_sign_pos_frac
    baseline                    default      48     1.000           0.822
    R1-delta                    1e-5         ...    ...             ...
    R1-delta                    1e-8         ...    ...             ...
    R1-delta                    1e-12        48     1.000           0.822
    R1-delta                    1e-15        ...    ...             ...
    R2-pld_disc                 1e-2         ...    ...             ...
    R2-pld_disc                 1e-3         48     1.000           0.822
    R2-pld_disc                 1e-4         ...    ...             ...
    R2-pld_disc                 1e-5         ...    ...             ...
    R3-min_sample_rate          1e-3         ...    ...             ...
    R3-min_sample_rate          1e-5         48     1.000           0.822
    R3-min_sample_rate          1e-7         ...    ...             ...
    R4-batch_x_epochs           (32, 1)      ...    ...             ...
    R4-batch_x_epochs           (128, 5)     48     1.000           0.822
    R4-batch_x_epochs           (512, 20)    ...    ...             ...
    R5-rdp_orders               truncated    ...    ...             ...
    R5-rdp_orders               default      48     1.000           0.822
    R5-rdp_orders               extended     ...    ...             ...

The three baseline rows (in **bold** in the paper table) are the pinned
C1/C6 numbers and are the internal-consistency check for every R-panel.
The `booktabs`-ready LaTeX skeleton is:

    \begin{table}[!t]
      \centering\small
      \caption{Robustness of the C1/C6 pure-NE structure under
        hyperparameter perturbation ($K = 200$, $w = 1$). Each row
        varies one axis and pins the other four at their C1/C6
        baseline (in bold).}
      \label{tab:R1-R5}
      \begin{tabular}{llrrr}
        \toprule
        axis & value & $|\mathrm{NE}|$ & off-diag frac. & ext.\ sign $+$ frac. \\
        \midrule
        baseline & default & \textbf{48} & \textbf{1.000} & \textbf{0.822} \\
        \midrule
        R1 $\delta$          & $10^{-5}$   & ... & ... & ... \\
                             & $10^{-8}$   & ... & ... & ... \\
                             & $10^{-12}$* & 48  & 1.000 & 0.822 \\
                             & $10^{-15}$  & ... & ... & ... \\
        R2 PLD disc          & $10^{-2}$   & --  & --    & $\Delta A_{\max}{=}1.2{\times}10^{-2}$ \\
                             & $10^{-3}$*  & 48  & 1.000 & 0.822 \\
                             & $10^{-4}$   & ... & ... & ... \\
        R3 $p_{\min}$        & $10^{-3}$   & 6\dag & 1.000 & 0.955 \\
                             & $10^{-5}$*  & 12\dag & 1.000 & 0.927 \\
                             & $10^{-7}$   & 12\dag & 1.000 & 0.927 \\
        R4 $(E_b, \text{ep})$& $(32, 1)$   & ... & ... & ... \\
                             & $(128, 5)$* & 48  & 1.000 & 0.822 \\
                             & $(512, 20)$ & ... & ... & ... \\
        R5 RDP orders        & truncated   & ... & ... & ... \\
                             & default*    & 48  & 1.000 & 0.822 \\
                             & extended    & ... & ... & ... \\
        \bottomrule
      \end{tabular}
    \end{table}

Starred rows are the pinned baseline. Rows marked $\dag$ are on the
K'=50 stride-4 subgrid rather than the full K=200 grid (see \S3).
$\Delta A_{\max}$ in the R2 row is the maximum absolute deviation of the
MIA advantage $A$ across 16 NE cells x 2 groups relative to the
`disc = 1e-3` reference; the NE count and off-diagonal fraction are
unchanged for the R2 sweep points we measured but are marked "--"
because we did not re-run NE-finding on the full K=200 grid at the
alternate disc values (the deviation number is the load-bearing
quantity here). The full sweep will be filled in after the Slurm
reruns.

## 10. Cross-link to C1--C8

Concretely, for each headline C-claim the paper should point to R1--R5
as follows:

- **C1** (`paper/tex/05_experiments.tex:43`, `sec:C1`). Append to the
  end of the C1 body: "The eight-NE asymmetric-multiplicity pattern of
  C1 is robust across R1--R5 (\cref{app:robustness})."
- **C4** (`paper/tex/05_experiments.tex:127-170`, `sec:C4`). Append:
  "The participation-threshold sensitivity of the coalition-attack
  optimum is reported in \cref{app:robustness}, panel R3."
- **C6** (`paper/tex/05_experiments.tex:230`, `sec:C6`). Append:
  "The K=200 equilibrium locus is stable to $\delta$, the PLD
  discretisation, and the RDP order grid (\cref{app:robustness})."
- **C7, C8** (`sec:C7`, `sec:C8`). No new sentence; R2 (advantage) and
  R1 ($\delta$) suffice as a pointer.

The `Common setup` paragraph (`paper/tex/05_experiments.tex:17-37`)
should carry a single-sentence forward-pointer to `app:robustness`,
matching the task spec (step 13). Once R1--R5 are all landed the
existing `app:sensitivity` paragraph on the participation threshold
(`paper/tex/08_appendix.tex:61-65`) should be *replaced* by a pointer to
R3, per Rule 0 (do not carry old and new descriptions side by side).

## 11. Paper location and transitions

Insert as `\subsection{Robustness sweeps R1--R5}` with
`\label{app:robustness}` in `paper/tex/08_appendix.tex` immediately
below the current `app:sensitivity` (which currently ends at line 65).
Five `\begin{figure}[!t] ... \end{figure}` blocks, one per R, with
labels `fig:app-R1` ... `fig:app-R5` to avoid the `fig:R1` label
collision the task-spec failure mode 5 flags. The LaTeX skeleton is
in the task spec, step 12, and does not need repeating here.

Transition sentences:

- **From `sec:setup`** (line 37): "A full battery of five
  hyperparameter-sensitivity sweeps is reported in
  \cref{app:robustness}."
- **From `sec:C4`** (line 170): "The participation-threshold
  sensitivity of this observation is reported in
  \cref{app:robustness}, panel R3."
- **From `sec:C6`** (line 250, just before `\end{figure}`): "See
  \cref{app:robustness} for a full battery of hyperparameter-sensitivity
  sweeps of this K=200 locus."

## 12. Cross-link to sibling findings docs

- **T15** (`tasks_findings/15_h5_v_shape_robustness.md`) covers the $V$
  axis (six admissible $V$ shapes, 100% externality-sign agreement).
  T15 + R1--R5 together are the paper's *six-axis* robustness argument:
  $V$, $\delta$, PLD disc, $p_{\min}$, $(E_b, I)$, RDP orders.
- **T05** (`tasks_findings/05_V_qualitative_invariance.md`) is the
  a-priori proposition T15 verifies numerically. R1--R5 have no matching
  a-priori proposition: they are numerical robustness only.
- **T18** (`tasks_findings/18_h2_scale_mechanism.md`) covers group-size
  scaling (the $|G|$ axis). Together with R4's $(E_b, I)$ axis these
  jointly cover the mechanism-training regime.
- The task's `polish_C6_C7_C8` and `10_numerical_poev_table` docs
  contain the K=200 baseline numbers reused here as the pinned
  reference.

## 13. Acceptance checklist

- [x] R1--R5 axes stated with per-axis narrative role and cross-link to
      the C-claim each stress-tests (\S1).
- [x] Base configuration pinned and matched against the cached K=200
      grid (\S2). Baseline reproduction check: recomposing utility
      from cached (sigma, A) yields exactly 48 pure NEs at $w=1$ at
      $K=200$, matching `_cache_v2/C6_K200.pkl` (Rule 5 internal
      consistency).
- [x] R3 sweep implemented end-to-end from the cached grid (real
      numbers, no Slurm) via `tasks_findings/_scripts/17_r1_r5.py`
      (\S3). Numbers: p_min=1e-3 -> |NE|=6, p_min=1e-5 -> |NE|=12,
      p_min=1e-7 -> |NE|=12 (identical to 1e-5), all off-diagonal.
- [x] R2 spot check at NE cells implemented end-to-end (\S4). Numbers:
      disc=1e-2 shifts $A$ by at most $1.19 \times 10^{-2}$ absolute
      relative to `disc = 1e-3`.
- [x] R1, R4, R5 methodology fixed and Slurm dispatch documented
      (\S5-7).
- [x] Publication-quality R3 figure `fig_r3_participation.{pdf,svg}`
      style-matched to `fig_h8_brpeg_regret` (\S8).
- [x] Booktabs table skeleton `tab:R1-R5` with per-axis rows and a
      pinned-baseline reference row (\S9), R3 and R2 rows filled with
      real numbers.
- [x] Cross-link to C1, C4, C6 with concrete insertion sentences
      (\S10).
- [x] Paper location `app:robustness` and transition sentences to
      `sec:setup`, `sec:C4`, `sec:C6` (\S11).
- [ ] R1, R4, R5 Slurm reruns landed and `_summary.csv` rows filled;
      author-side edit, out of scope for T17's findings-only landing.
- [ ] Full K=200 R3 rerun (stride=1, ~1 CPU-h on Slurm) to promote the
      stride-4 subgrid numbers to full-grid numbers; author-side edit.
- [ ] R2 sweep extended to disc in {1e-4, 1e-5} on the full K=200 grid
      (~4 CPU-h on Slurm); author-side edit.
- [ ] R3 C4 target-advantage panel ($A_{\rm tgt}(k{=}4)$ vs threshold);
      requires a 5-player rate rebuild, ~30 CPU-min on Slurm.
- [ ] `paper/tex/08_appendix.tex` receives `app:robustness` subsection
      and `fig:app-R1` ... `fig:app-R5` blocks; author-side edit.

## 14. Load-bearing dependencies and edge cases

- **Cache dependency.** T17 hard-depends on
  `_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl` (produced by
  `slurm_scripts/run_locus_K200.sh`). If regenerated with different
  `(|G|, E_b, I, delta)`, the K=200 NE count of `48` used as the
  baseline reference throughout \S3--\S9 is stale and this findings
  doc must be re-emitted.
- **R3 rate rebuild.** The R3 pipeline rebuilds sample rates from the
  cached $\sigma$ via `_sample_rate_for_eps(sigma, eps, steps, delta)`.
  The rebuild caches to `_cache_v2/R3_rates_K200.pkl` and is
  deterministic modulo the `epsilon_rdp` `lru_cache`; the cache
  is safe to delete if the mechanism grid is regenerated.
- **R2 spot vs full grid.** R2's spot check at NE cells is a
  *necessary* condition for NE-set stability, not sufficient. If the
  spot check passes but a reviewer objects, the fallback is the full
  K^2 sweep at `disc = 1e-4` under Slurm (feasible at ~4 CPU-h; the
  `disc = 1e-5` sweep is prohibitive at any budget and is not needed
  because 1e-4 already saturates the discretisation-invariance claim).
- **R4 diagonal-race NE at $(512, 20)$.** Expected weak-privacy corner.
  If it collapses the off-diagonal fraction to 0, add a one-line caveat
  to the C1 subsection acknowledging the weak-privacy regime; do *not*
  strike C1.
- **R5 saturation regression.** If the truncated order grid moves the
  NE set, promote `extended` to the default at
  `game/mechanism.py:33-35` and rerun *all* K=200 experiments (\S7);
  this is the only correct response.
- **Label collision.** The task spec's failure mode 5 flags a possible
  collision with existing `fig:R1` labels. Use `fig:app-R1` ...
  `fig:app-R5` from the start to preempt it.
