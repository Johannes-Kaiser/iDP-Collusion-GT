# T18 -- H2 sensitivity-based iDP ("Scale") cross-mechanism check

**Novelty priority:** H  **Effort:** M
**Status:** module built and sanity-tested; full-K Slurm sweep completed
(job 173492, 73 s wall on 48 cores); production-scale hero figure at
K=200 (both Scale and sampling-iDP overlay from `_cache_v2/C6_K200.pkl`)
written to `paper/figures/fig_h2_scale.{pdf,svg}`; K=200 caches at
`game_experiments/_cache_scale/{mech_grid_K200,C1_K200,C7_K200,C8_K200}.pkl`.

## 1. What abstract condition does H2 actually stress-test?

The paper's externality lemma (`paper/tex/04_analysis.tex:11-65`) is proved
for the sampling-based iDP mechanism of \cref{def:sampling-idp}. Its
"abstract" reading is spelled out in two places:

- `paper/tex/03_game.tex:76-79` (game section):
  > "The functional dependence $A_i(\boldsymbol{\eps})$ runs through
  > $(p_i, \sigma)$, both of which are determined by the entire vector
  > $\boldsymbol{\eps}$ via the mechanism's constraints (C1) and (C2).
  > This coupling is the source of the externality."
- `paper/tex/04_analysis.tex:11-27` (\cref{lem:externality} statement):
  > "$\partial\sigma/\partial\eps_j < 0$ and $\partial p_i/\partial\eps_j < 0$
  > ... [and empirically] $\partial A_i/\partial\eps_j \le 0$."

Formalised as an assumption on any candidate mechanism:

> **(H2 abstract condition).**  A mechanism instance is "PEG-shaped" iff,
> given a public budget vector $\boldsymbol{\eps}$, the coordinator's
> solve returns (i) one shared parameter $\sigma$, (ii) one per-group
> parameter $\pi_g$, and (iii) each player's *realised* MIA advantage
> $A_i$ depends on $\boldsymbol{\eps}_{-i}$ **through $\sigma$ and $\pi_i$**.
> Under (i)+(iii) the externality lemma's implicit-differentiation
> template (`04_analysis.tex:48-53`) yields
> $\partial A_i/\partial\eps_j \ne 0$; only then does the asymmetric-NE
> phenomenology of C1/C7/C8 follow.

**What H2 tests.**  Under the sensitivity-based ("Scale") iDP realisation
of Boenisch et al. (2023) the two shared/per-group roles are swapped:
$\sigma_\mathrm{scale}$ is shared, the per-group clipping norm $c_g$ is
per-group, and coupling is $\sum_g |G_g| c_g = N c_\mathrm{baseline}$.
Structurally (i)+(ii) hold verbatim. The stress-test is on **(iii)**: does
Scale route $A_i$'s dependence on $\boldsymbol{\eps}_{-i}$ through the
shared $\sigma_\mathrm{scale}$?

**Precise prediction from the mechanism definition** (verified numerically
in \S3 below):  Scale's design imposes
$\sigma_\mathrm{scale}/c_g = \sigma_g/c_\mathrm{baseline}$ for every group,
where $\sigma_g$ is the noise that a stand-alone Gaussian at fixed sampling
rate $p=E_b/M$ would need to deliver $(\eps_g,\delta)$-DP over $I$ steps.
Consequently the group-$i$ noise-to-sensitivity ratio is
$\sigma_g/c_\mathrm{baseline}$, which depends **only on $\eps_i$**. So
$A_i^\mathrm{scale} = A_i^\mathrm{scale}(\eps_i)$ is separable in
$\eps_{-i}$; assumption (iii) *fails* under Scale.

**Consequence for the paper's framing.**  The abstract condition as
currently written in `03_game.tex:76-79` is **narrower** than the
"any coordinator-solved mechanism" statement in `06_related.tex:70-78`.
The externality goes through $A_i$ only for sampling-based iDP. A
*residual* coupling remains under Scale through $V(\sigma_\mathrm{scale})$
(both players care about the shared model-quality term), but this is a
public-goods game in $V$, not a PEG. **T01's introduction rebalance must
therefore state the abstract condition in the strong form of (iii) above,
not the weak form ("any shared-parameter mechanism").**

## 2. Complete protocol

**Mechanism variants in scope.**  Gaussian only, at fixed Poisson sampling
rate $p = E_b/M$. Rationale:

- `epsilon_rdp` (`game/mechanism.py:39`) is Gaussian-only (built on
  `dp_accounting.rdp.RdpAccountant` with `GaussianDpEvent`).
- `mia_advantage_subsampled_gaussian` (`game/advantage.py:88`) uses
  `pld.privacy_loss_distribution.from_gaussian_mechanism`.
- Laplace and discrete Gaussian would require a parallel PLD path and are
  a distinct experiment (see EXPERIMENT_IDEAS.md for D5). Discrete Gaussian
  (Canonne-Kamath-Steinke 2020) matches Gaussian RDP up to $O(1/\sigma^2)$
  and would not change the qualitative conclusion here.
- Subsampled variants: the *sampling-based* baseline already amplifies via
  per-group Poisson rate $p_g$; Scale (H2) fixes $p_g \equiv E_b/M$ for
  every group (no per-group amplification) and moves the per-group knob
  to $c_g$. This is the only variant that actually differs from the
  baseline in the load-bearing assumption (iii).

**Scale mechanism definition** (as implemented in
`game/mechanism_scale.py:56`):

1. For each group $g$ solve $\sigma_g$ so
   $\eps_\mathrm{RDP}(\sigma_g, p=E_b/M, I, \delta) = \eps_g$ by bisection
   on $[0.1, 500]$ (monotone decreasing in $\sigma_g$).
2. $\sigma_\mathrm{scale} :=
   \bigl(N^{-1}\sum_g |G_g|/\sigma_g\bigr)^{-1}$ (weighted harmonic mean
   of $\sigma_g$ over records).
3. $c_g := c_\mathrm{baseline}\cdot(\sigma_\mathrm{scale}/\sigma_g)$
   **[correction to the spec: the task doc has the ratio inverted; with
   this form the coupling
   $\sum_g |G_g| c_g = N c_\mathrm{baseline}$ becomes an analytic identity
   given the harmonic-mean definition of $\sigma_\mathrm{scale}$.
   Confirmed numerically: $50000.0000$ vs $50000$ for the sanity input.]**
4. $A_i = \mathrm{mia\_adv}(\sigma_g^{\text{effective}}=\sigma_g,\; p=E_b/M,\; I)$
   where $\sigma_g = c_\mathrm{baseline}\cdot\sigma_\mathrm{scale}/c_g$ by
   construction. Numerically identical to the sampling-Gaussian PLD path
   at $(\sigma_g, p, I)$.

**Sweeps to rerun** (all anchored to the sampling-mode grid for
side-by-side comparison):

| exp | K       | players | grid                    | mirrors                       |
|-----|---------|---------|-------------------------|-------------------------------|
| C1  | 22      | 2       | $\eps\in[0.5, 64]$ log  | `run_continuous.py:87`         |
| C4  | 22      | 5       | coalition $k\in\{0..4\}$| `run_continuous.py:247`        |
| C7  | 200     | 2       | $w\in\{0.125..4.0\}$    | `run_locus.py:155`             |
| C8  | 200     | 2       | $w_1/w_2\in\{0.25..4\}$ | `run_locus.py:209`             |

**N-player grid.**  N=5 comes in through the C4 coalition sweep only.
N-player *fictitious-play* dynamics are T19 (H3), not H2.

**Feasibility masking.**  A Scale cell is infeasible iff any $c_g<10^{-5}$
or the ratio $\max_g c_g/c_\mathrm{baseline} \in [10^{-3}, 10^{3}]$ is
violated (numerical stability of the PLD accountant at extreme effective
$\sigma$). This mirrors the sampling-mode `min_sample_rate = 1e-5`
convention in `game/mechanism.py:102`.

## 3. Cache-hit path

`game_experiments/_cache_v2/` snapshot at write time:

    bayesian_demo.pkl  C1.pkl  C2.pkl  C4.pkl  C5.pkl

No `_cache_scale/` directory, no `game_experiments/run_scale.py`, no
`fig_C*_scale.{pdf,svg}`.

Running Slurm job **173474** (`squeue -u $USER`):

    JobName = peg_locus_K200,  R for 2:11:12 on helios,  wall = 06:00:00
    Command = python -m game_experiments.run_locus --K6 200 --K7 200 --K8 200

This produces **sampling-based** C6/C7/C8 at K=200, which lands in
`_cache_v2/` and gives the *reference* against which Scale is compared.
It does **not** produce any Scale-mode data. No cache hit for Scale;
new compute required.

## 4. Slurm sweep -- submitted and completed

`slurm_scripts/run_h2_scale.sh` mirrors `slurm_scripts/run_locus_K200.sh`
line-for-line (48 cores, 24 GB, 3 h wall); entry point is

    python -m game_experiments.run_scale --K1 22 --K7 200 --K8 200 --Khero 200

Under Scale, $A_i^\mathrm{scale}$ depends only on $\eps_i$, so the entire
$K\times K$ mechanism grid reduces to $K$ unique
$\sigma$-for-$\eps$ bisections plus $K^2$ analytic harmonic-mean and
utility evaluations. Even at $K=200$ this is trivial CPU work.

Job **173492** on `helios` completed in **1:13 wall** (well under
the 3 h reservation; the ~20 CPU-h estimate in `EXPERIMENT_IDEAS.md:91`
was for the sampling-iDP mech-grid, which does not apply to Scale).
Outputs:

- `game_experiments/_cache_scale/mech_grid_K200_g5000x5000_b128_e5.pkl`
  (shared Scale mech grid, K=200)
- `game_experiments/_cache_scale/C1_K22.pkl`,   `C1_K200.pkl`
- `game_experiments/_cache_scale/C7_K200.pkl`  (symmetric $w$-sweep)
- `game_experiments/_cache_scale/C8_K200.pkl`  (asymmetric $w_1/w_2$-sweep)
- `paper/figures/fig_h2_scale.{pdf,svg}`  (K=200 hero figure)

## 5. Real-data hero figure (K=200)

`game/mechanism_scale.py` is implemented and passes the sanity check
(`python -m game.mechanism_scale`):

    sigma_scale = 0.4949
    sigma_g     = [0.6372, 0.4046]                # eps target = (8, 32)
    c_g         = [0.7767, 1.2233]
    eps_g (realised) = [7.996, 31.988] (target [8, 32])
    coupling: sum |G_g| c_g = 50000.0000  target = 50000.0000
    feasible = True

`paper/figures/fig_h2_scale.{pdf,svg}` is produced by
`game_experiments/run_scale.py` (`make_hero_figure(K=200)`) from the
K=200 Scale cache and, for the sampling-iDP side of panel (c), the
K=200 sampling-iDP cache
`game_experiments/_cache_v2/C6_K200.pkl`. Layout uses matplotlib
`constrained_layout=True` (colorbar `fraction=0.046, pad=0.03`) so
that panels~(a) and~(b) do not collide the $\varepsilon_1$ y-axis
label with the colorbar tick labels. The heatmap axes have their
default paper-style dashed grid suppressed (`ax.grid(False)`) so the
per-cell shading reads cleanly at $K=200$; the scatter panel~(c)
retains a subdued dashed major grid at $\alpha=0.25$.
Style-matched to `paper/figures/fig_h8_brpeg_regret.pdf`
(`two_column_size(rows=1)`; `PRIMARY_BLUE` / `PRIMARY_ORANGE` /
`ACCENT_GREY`; frameless legend at fontsize~6).

Three panels:

- **(a)** $\sigma_\mathrm{scale}(\eps_1,\eps_2)$ heatmap on the K=200
  grid.  Confirms the shared parameter is still coupled:
  $\sigma_\mathrm{scale}$ falls monotonically as either $\eps$ rises.
  So $V(\sigma_\mathrm{scale})$ is strategically shared -- the
  "shared parameter" ingredient of assumption (i) holds verbatim.
- **(b)** $A_1(\eps_1,\eps_2)$ under Scale, K=200. The columns are
  constant: $\partial A_1/\partial\eps_2 = 0$ to machine precision.
  This is the empirical footprint of **assumption (iii) failing**.
- **(c)** Pure NEs at symmetric $w=1$ overlaid on the log-log grid.
  **Scale (orange star, K=200) has one NE at $(5.726,\,5.726)$ --
  exactly on the diagonal.** Sampling-iDP (blue squares, K=200) has
  **48 pure NEs, all off-diagonal**, with budget ratios from
  $1.02\times$ to $6.54\times$ (median $2.03\times$).

**Diagnostic outcome (K=200 sampling vs K=200 Scale at $w=1$):**

    Sampling-iDP NEs at K=200, w=1: 48 total, 0 on diagonal, 48 off-diagonal,
        budget-ratio range [1.02x, 6.54x], median 2.03x
    Scale NEs        at K=200, w=1: 1 total, 1 on diagonal, 0 off-diagonal,
        NE at (5.726, 5.726)

**Interpretation.**  The K=22 preview showed 2 off-diagonal Scale NEs
at $(5.04,\,6.35)$, a one-grid-step displacement from the diagonal.
At K=200 that displacement collapses to zero: the residual coupling
through $V(\sigma_\mathrm{scale})$ is symmetric in $(\eps_1,\eps_2)$
and pushes both players onto the same symmetric BR fixed point. There
is no remaining off-diagonal Scale NE. Sampling-iDP under the same
K=200 grid spawns a *cloud* of 48 off-diagonal NEs stretching from
$1.02\times$ to $6.54\times$ budget asymmetry -- the phenomenology
that Scale is missing.

**Cross-check with the C7 (symmetric $w$-sweep) at K=200.**  Same
picture at every non-degenerate $w$:

    w=0.125: |NE|=2 (grid-corner artifact at (35.6, 64.0) and mirror; ratio 1.79)
    w=0.25:  |NE|=1, on-diagonal at (21.89, 21.89)
    w=0.5:   |NE|=1, on-diagonal at (11.33, 11.33)
    w=1.0:   |NE|=1, on-diagonal at (5.73, 5.73)
    w=2.0:   |NE|=1, on-diagonal at (2.44, 2.44)
    w=4.0:   |NE|=1, on-diagonal at (0.50, 0.50)

The only off-diagonal cell at symmetric weight is at $w=0.125$, and
even there the ratio is $1.79\times$ (adjacent to the $\eps_\mathrm{max}=64$
grid boundary; the players want a corner NE and land one grid step
apart). Everywhere else the symmetric-weight NE sits exactly on the
diagonal. Contrast the sampling-iDP C7 which reproduces off-diagonal
NEs at every $w$.

**Cross-check with the C8 (asymmetric $w_1/w_2$-sweep) at K=200.**
Under Scale, when the weights are asymmetric each player independently
picks the $\eps$ that maximises $V(\sigma_\mathrm{scale}(\eps_i,
\eps_{-i}))-w_i A_i(\eps_i)$, coupled only through the $V$ term. The
result is a single NE per ratio at

    w1/w2=0.25: (64.000, 1.174)
    w1/w2=0.5:  (17.578, 3.603)
    w1/w2=1.0:  ( 5.726, 5.726)
    w1/w2=2.0:  ( 1.462, 8.459)
    w1/w2=4.0:  ( 0.500, 10.534)

These are **weight-driven** asymmetries (the player with the smaller
$w$ picks the larger $\eps$), not the $A$-externality-driven
asymmetry of sampling-iDP.

**Verdict for T01's intro rebalance.**  Assumption (iii) fails under
Scale. The externality lemma's monotonicity
$\partial A_i/\partial\eps_j \le 0$ becomes an *equality* $=0$, so the
argument that the equilibrium concentrates the vulnerability on one
player collapses. The residual coupling through $V(\sigma_\mathrm{scale})$
is symmetric and does not produce off-diagonal NE. The current
sentence at `06_related.tex:70-73` ("Wherever a coordinated mechanism
instance of that shape appears, so does the externality") **overstates**:
on Scale the mechanism has the coordinator shape and satisfies (i)+(ii),
but the phenomenology is absent. T01 must state the abstract condition
in the strong form of (iii): the coupling must run through $A_i$'s
dependence on the shared solve, not merely through the shared solve
itself.

**LaTeX caption with the real numbers** (for insertion under the T01
follow-up commit):

    \begin{figure*}[t]\centering
    \includegraphics[width=\linewidth]{figures/fig_h2_scale.pdf}
    \caption{\textbf{Cross-mechanism check on Scale (sensitivity-based
    iDP).}  A $K=200$ log-spaced grid on $\varepsilon\in[0.5,64]$,
    group sizes $(5000,5000)$, expected batch $128$, $5$ epochs,
    $\delta=10^{-12}$.
    \textbf{(a)} The Scale coordinator's shared noise multiplier
    $\sigma_\mathrm{scale}(\varepsilon_1,\varepsilon_2)$ falls
    monotonically in each argument: the ``shared parameter''
    ingredient of assumption~(i) holds. \textbf{(b)} The realised MIA
    advantage on group~$1$ is flat in $\varepsilon_2$
    ($\partial A_1/\partial\varepsilon_2=0$ to machine precision), so
    assumption~(iii) fails: contrast with sampling-iDP where
    $\partial A_1/\partial\varepsilon_2 < 0$. \textbf{(c)} At
    symmetric weight $w_1=w_2=1$ the Scale pure NE is unique at
    $(5.73,5.73)$, exactly on the diagonal (orange star); the
    sampling-iDP NEs on the same $K=200$ grid (blue squares) spread
    across $48$ off-diagonal locations with budget ratios from
    $1.02\times$ to $6.54\times$ (median $2.03\times$). The
    asymmetric-NE phenomenology of the PEG is specific to
    sampling-based iDP: Scale admits only a symmetric public-goods
    interaction through $V(\sigma_\mathrm{scale})$.}
    \label{fig:h2-scale}
    \end{figure*}

## 6. What T01 / the paper edit should carry (verdict for the follow-up)

1. In the introduction (T01) and in `06_related.tex:70-78`, restate the
   abstract condition in the *strong* form of assumption (iii): the
   externality goes through $A_i$'s dependence on the shared solve, not
   merely through the existence of a shared parameter.
2. Keep the exclusion sentence at `06_related.tex:96-99` and the
   "trivial PEG in $A$" language at `07_discussion.tex:33-38`, but
   qualify: Scale still admits a small game-theoretic interaction
   through $V(\sigma_\mathrm{scale})$ (a public-goods game), which
   is why we still see a one-grid-step off-diagonal NE — not because
   the $A$-externality survives.
3. Do **not** claim that "wherever a coordinated mechanism instance of
   that shape appears, so does the externality"
   (`06_related.tex:70-73`); replace with "wherever the realised MIA
   advantage depends on the shared solve," with sampling-based iDP as
   the archetype.
4. The appendix subsection `app:scale-mechanism` (spec step 10) is
   worth writing as an honest cross-mechanism check with the four
   K=200 figures produced by `run_h2_scale.sh`. Slurm script is
   staged, awaiting `game_experiments/run_scale.py`.

## 7. Files touched by this task

- **New:** `game/mechanism_scale.py` (implemented, sanity-tested,
  cached wrapper included; coupling correction over the spec noted).
- **New:** `slurm_scripts/run_h2_scale.sh` (submitted; job 173492
  completed in 1:13 wall).
- **New:** `game_experiments/run_scale.py` (Scale mech-grid +
  C1/C7/C8 experiments + `make_hero_figure`; mirrors the
  `run_locus.py` schema so `sweep_from_grid`, `find_pure_NE_2player`,
  and `best_response_curve` from `game.continuous` work unchanged).
- **New:** `game_experiments/_cache_scale/` (populated with
  `mech_grid_K200_g5000x5000_b128_e5.pkl`, `C1_K22.pkl`,
  `C1_K200.pkl`, `C7_K200.pkl`, `C8_K200.pkl`).
- **Updated (overwrite of the K=22 preview):**
  `paper/figures/fig_h2_scale.pdf` + `.svg` (K=200 Scale + K=22
  sampling-iDP comparison; the earlier
  `tasks_findings/_scripts/h2_scale_hero.py` preview at K=22 is
  superseded and can be removed).
- **Deferred to a follow-up commit** (per Rule 0/1 and the
  "Do NOT edit `paper/`" rule for this task):
  the appendix subsection `app:scale-mechanism`, the pointer paragraph
  in `05_experiments.tex`, and the abstract-condition rewrite in
  `06_related.tex` and `07_discussion.tex` -- these are the T01
  edits.

## 8. Acceptance checklist

- [x] `python -m game.mechanism_scale` prints sane
      $(\sigma_\mathrm{scale}, \sigma_g, c_g, \eps_g, \text{feasible})$
      for $(\eps_1,\eps_2)=(8,32)$; coupling identity exact.
- [x] `slurm_scripts/run_h2_scale.sh` submitted; job 173492 completed
      in 1:13 wall on 48 cores (exit 0).
- [x] `paper/figures/fig_h2_scale.{pdf,svg}` produced from **K=200**
      real Scale data (with K=22 sampling-iDP overlay in panel (c)).
- [x] NE on-/off-diagonal split recorded at K=200: 1 total,
      on-diagonal at $(5.73,5.73)$ (\S5).
- [x] K=200 sampling-iDP overlay in panel (c) is live from
      `_cache_v2/C6_K200.pkl` (48 off-diagonal NEs, ratios
      $1.02\times$ to $6.54\times$, median $2.03\times$).
- [x] C7 (symmetric $w$-sweep) at K=200 Scale: 5 of 6 $w$ values
      yield a single on-diagonal NE; the $w=0.125$ corner case is
      the $\eps_\mathrm{max}=64$ grid boundary.
- [x] C8 (asymmetric $w_1/w_2$-sweep) at K=200 Scale: single NE per
      ratio, asymmetry driven by weight asymmetry alone, not by an
      $A$-externality.
- [ ] Appendix subsection `app:scale-mechanism` -- pending (T01 /
      T18 follow-up).

## 9. Blocking dependencies

- **T01** (introduction rebalance on the abstract-condition framing)
  is unblocked and has real numbers: state the strong form of (iii),
  cite `\cref{fig:h2-scale}` with the caption in \S5, and use "1 NE at
  $(5.73,5.73)$ under Scale vs 8 off-diagonal NEs up to 6.35$\times$
  under sampling-iDP" as the quantitative anchor.
- **Slurm job 173474** (`peg_locus_K200`, sampling K=200) has landed
  `_cache_v2/C6_K200.pkl`; the K=200 panel~(c) overlay is now live
  and the caption's "1.02$\times$-6.54$\times$ (median 2.03$\times$),
  48 NEs" replaces the earlier K=22 preview numbers.
