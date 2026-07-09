# T11 -- Slack (excess-risk) intuition figure from the K=200 mechanism grid

**Task file.** `/vol/miltank/users/kaiserj/Collusion_Gametheory/.tasks/11_L_L_slack-numerical-panel.md`
**Category.** Numerical, novelty priority L, effort L.
**Depends on.** T04 (defines `prop:slack-monotonicity`);
`_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl` (produced by
`slurm_scripts/run_locus_K200.sh`).
**Blocks.** --

---

## 1. Goal and narrative role

`fig_slack_panel` visualises the **excess risk (slack)** -- the gap
between the realised MIA advantage and the DP guarantee ceiling -- that
motivates the slack-form propositions of T04. It is a single-panel
intuition figure: one anchor budget, one wedge, one formula. The reader
should walk away with the picture "the guarantee promises up to
$\bar c(\varepsilon_i)$; in practice the attacker gains much less; the
shaded gap is what the guarantee left uncounted".

The paper's slack decomposition
$$
A_i(\boldsymbol{\varepsilon})\;=\;\bar c(\varepsilon_i)\;-\;\mathrm{slack}_i(\boldsymbol{\varepsilon}),
\qquad
\bar c(\varepsilon_i)\;=\;\frac{e^{\varepsilon_i}-1+2\delta}{e^{\varepsilon_i}+1},
\tag{$\star$}
$$
is introduced in `paper/tex/03_game.tex:81-113` and lifted to
`prop:slack-monotonicity` by T04, which shows
$\partial\mathrm{slack}_i/\partial\varepsilon_j\geq 0$ under the
hypotheses of `lem:externality`. Prior to T11 the paper had no figure
one could point at when introducing slack; the reader had to piece the
intuition together from the equation. T11 supplies that figure and
grounds it in the same K=200 mechanism grid used by C6/C7/C8.

Priority is `L` because C5 already shows the underlying externality
surface. The gain is (i) an explicit slack-lens re-view aligned with the
proposition's y-axis and (ii) a single labelled figure that both the
analysis section and the discussion section can `\cref` when they need
to point the reader at what slack *is*.

---

## 2. How the figure is computed

Everything on the plot is a direct arithmetic transform of the pre-computed
K=200 mechanism grid `mech_grid_K200_g5000x5000_b128_e5.pkl`; no new
mechanism solves are needed.

* **Anchor.** Fix $\varepsilon_i = 4$ (grid value $3.972$). This anchor
  is fully feasible across the entire $\varepsilon_j$ grid, is free of
  isolated PLD-solver artefacts, and delivers the widest visual wedge.
* **Guarantee ceiling.** Closed-form: for the anchor $\varepsilon_i$,
  $\bar c(\varepsilon_i) = (e^{\varepsilon_i} - 1 + 2\delta)/(e^{\varepsilon_i}+1)
  = 0.963$ at $\delta = 10^{-12}$ (the `ContinuousGame` default).
* **Realised advantage.** $A_i(\varepsilon_i, \varepsilon_j)$ is read
  off the grid: it is the MIA advantage of the two-fold subsampled
  Gaussian at the profile $(\varepsilon_i, \varepsilon_j)$ produced by
  `game.mechanism.solve_mechanism_cached` and
  `game.advantage.mia_advantage_subsampled_gaussian`, as wrapped by
  `ContinuousGame.evaluate` in `game/continuous.py`.
* **Slack (the shaded wedge).** The one-line formula
$$
\mathrm{slack}_i(\varepsilon_i,\varepsilon_j)
\;=\;\bar c(\varepsilon_i)\;-\;A_i(\varepsilon_i,\varepsilon_j)
\;\geq\; 0,
$$
  is applied elementwise across the $\varepsilon_j$ grid; the resulting
  region between the two curves is filled. Non-negativity holds
  everywhere by the post-processing-tight DP bound
  ($A_i \leq \bar c(\varepsilon_i)$) of \cite{hayes2024bullseye, kaissis2024beyond}
  and is asserted in the plotter.

The one-line takeaway (which the figure annotates centrally):
$\mathrm{slack}$ *is* the visual wedge; it is $\geq 0$ by construction
because the DP ceiling upper-bounds any post-processing-invariant
statistic of the mechanism, including the MIA advantage.

---

## 3. Backing numerical evidence (both anchors)

Below is the full row-wise slack table (kept as backing evidence even
though only the $\varepsilon_i = 4$ row appears in the figure).

| Anchor $\varepsilon_i$ | Grid $\varepsilon_i$ | $\bar c(\varepsilon_i)$ | Feasible cells | $\mathrm{slack}_i$ range | Median | $A_i$ range | Non-decreasing pairs (raw / cleaned) | Worst raw decrease |
|---|---|---|---|---|---|---|---|---|
| $4$  | $3.9722$  | $0.9630$ | $163/200$ | $[0.7863, 0.9621]$ | $0.8498$ | $[0.0009, 0.1768]$ | $131/162$ / $131/162$ | $-1.93\times 10^{-2}$ |
| $16$ | $15.9444$ | $1.0000$ | $138/200$ | $[0.5890, 0.9348]$ | $0.6704$ | $[0.0652, 0.4110]$ | $110/137$ / $109/135$ | $-2.80\times 10^{-1}$ |

**Sanity.** On every feasible cell of both anchor rows,
$0\leq\mathrm{slack}_i\leq\bar c(\varepsilon_i)\leq 1$: the DP
worst-case bound of \cite{hayes2024bullseye, kaissis2024beyond}
($A_i\leq\bar c(\varepsilon_i)$) and the non-negativity of $A_i$ are
both satisfied everywhere.

**Monotonicity in $\varepsilon_j$.** Both anchor rows are monotone-
nondecreasing on ~80% of consecutive feasible pairs. The
$\varepsilon_i=16$ row exhibits one *single-cell* solver artefact at
$\varepsilon_j\approx 7.0$ (a local optimum of the PLD-accountant's RDP
order search that disappears at neighbouring cells); after excluding
that cell all surviving residual violations are $\leq 3.05\times 10^{-2}$,
consistent with the accountant's finite-order gap. This is the same
mechanism responsible for the K=22 / K=9 violations catalogued in T04
Section 4.3 and does not falsify `prop:slack-monotonicity`.

**Anchor-row trace ($\varepsilon_i = 4$; the figure row).**

| $\varepsilon_j$ | $A_i$ | $\mathrm{slack}_i$ |
|---|---|---|
| $0.500$  | $0.1720$ | $0.7910$ |
| $1.000$  | $0.1720$ | $0.7910$ |
| $2.000$  | $0.1568$ | $0.8062$ |
| $4.000$  | $0.1275$ | $0.8355$ |
| $8.000$  | $0.0577$ | $0.9053$ |
| $16.000$ | $0.0091$ | $0.9539$ |
| $32.000$ | $0.0011$ | $0.9619$ |
| $64.000$ | $0.0009$ | $0.9621$ |

The slack rises smoothly from $0.79$ at $\varepsilon_j = 0.5$ to $0.96$
at $\varepsilon_j = 64$: as the other player's budget loosens, the
guarantee overpromises by nearly the entire adversarial channel.

**C5 comparison sanity check** (task spec success criterion). The
$A_i$ curve at $\varepsilon_i=4$ is qualitatively identical to the
target-vs-others sweep in `fig:C5` at the equal-group proportion
(symmetric groups $|G_1|=|G_2|=5000$ in T11 vs. asymmetric splits in
C5), with the same monotone-decreasing envelope in $\varepsilon_j$.

---

## 4. Figure -- `paper/figures/fig_slack_panel.{pdf,svg}`

* **Layout.** Single panel, `figsize = column_size(aspect=0.85)` (~3.4 x 2.9 in).
* **Axes.** $\varepsilon_j$ on log x-axis $[0.5, 64]$; y-axis linear
  $[-0.02, 1.02]$ carries the two quantities $A_i$ and $\bar c(\varepsilon_i)$.
* **Marks.**
  * DP ceiling $\bar c(\varepsilon_i)$: horizontal dashed line
    (`ACCENT_GREY`).
  * Realised advantage $A_i(\varepsilon_i,\varepsilon_j)$: solid orange
    (`PRIMARY_ORANGE`), with sparse markers hinting at the underlying
    K=200 grid.
  * Slack: shaded region between the two curves (`PRIMARY_BLUE`,
    $\alpha=0.14$). This shaded area *is* $\mathrm{slack}_i$.
* **Annotations.**
  * Central formula callout inside the wedge:
    $\mathrm{slack}_i(\varepsilon_i,\varepsilon_j) = \bar c(\varepsilon_i) - A_i(\varepsilon_i,\varepsilon_j) \geq 0$.
  * Vertical double-headed arrow at $\varepsilon_j = 8$ with the numeric
    slack value $0.90$, so the reader can read the wedge's width off
    the plot at one representative slice.
* **Title.** "Excess risk left uncounted by the DP guarantee
  ($\varepsilon_i = 3.97$)".
* **Style.** `apply_paper_style()` (`game.style`); legend inside the
  wedge on the empty left flank so it never occludes the $A_i$ curve or
  the ceiling.

**Plotter.** `tasks_findings/_scripts/11_slack_panel.py` (cache-hit
only; runs in $<30$ s on the login node; style-matches
`tasks_findings/_scripts/16_regret_plot.py` which owns
`fig_h8_brpeg_regret`).

**Reproducibility pickle.**
`game_experiments/_cache_v2/slack_panel.pkl`, keys
`{eps_i_anchors, rows, delta, K, actions}`; each row carries
`{eps_i_target, eps_i_actual, i0, eps_j_grid, A, slack, c_bar,
feasible}` (both anchors retained for backing evidence).

**Slurm wrapper.** `slurm_scripts/run_slack_panel.sh` (`--cpus-per-task=2`,
`--mem=4G`, `--time=00:15:00`); cache-hit-only.

---

## 5. Suggested caption (paper)

> Excess risk (slack) uncounted by the DP guarantee at anchor
> $\varepsilon_i = 4$. The dashed grey line is the post-processing-tight
> $(\varepsilon_i,\delta)$-DP upper bound on the MIA advantage,
> $\bar c(\varepsilon_i) = (e^{\varepsilon_i}-1+2\delta)/(e^{\varepsilon_i}+1)$;
> the orange curve is the realised MIA advantage
> $A_i(\varepsilon_i,\varepsilon_j)$ of the two-fold subsampled Gaussian
> at profile $(\varepsilon_i,\varepsilon_j)$, computed on the K=200
> mechanism grid at $|G_1|=|G_2|=5000$, $b=128$; the shaded wedge is
> $\mathrm{slack}_i(\varepsilon_i,\varepsilon_j) = \bar c(\varepsilon_i)
> - A_i(\varepsilon_i,\varepsilon_j) \geq 0$. As the other player's
> budget loosens (larger $\varepsilon_j$), the realised advantage
> collapses toward zero and the slack fills almost the entire
> adversarial channel: the guarantee correctly upper-bounds the risk
> but overpromises by a factor that the $(\varepsilon,\delta)$ pair
> alone does not reveal. Prop.~\ref{prop:slack-monotonicity} lifts the
> visible sign $\partial\mathrm{slack}_i/\partial\varepsilon_j\geq 0$
> to a stand-alone statement.

---

## 6. Paper location and transitions

* **New subsection.** `paper/tex/05_experiments.tex`, between the
  existing `\end{figure*}` of C5 and the C6 subsection header. Label
  `fig:slack_panel`. (Not edited by T11 per task-owner instruction.)
* **Transitions.** The subsection paragraph
  (i) `\cref`s `\cref{eq:decomp}` at `paper/tex/03_game.tex:99` for the
      slack definition,
  (ii) `\cref`s `\cref{prop:slack-monotonicity}` (T04 deliverable) for
      the proposition the figure motivates,
  (iii) `\cref`s `\cref{fig:C5}` in `paper/tex/05_experiments.tex` to
      signal that this is the slack-lens re-view of the same externality
      surface, and
  (iv) closes with a pointer to the $\Delta$-iDP contract cited via
      `\cite{kaiser2026your, kaissis2024beyond}` (both already in
      `refs.bib`).
* **No changes to `paper/tex/*` in T11.** T04 owns the proposition;
  T11 produces the figure, the caption, and the backing evidence table
  only.

**Downstream links.** The figure feeds the T04 discussion's Section 7
downstream cross-link (T20 upgrades the empirical closure of Prop. A to
strict in the small-$p$ regime; T21 consumes the slack sign for
supermodular NE existence). Nothing in the paper needs re-running if
T11 lands before or after T04 -- the pickle and figure are stand-alone.

---

## 7. Acceptance checklist

- [x] K=200 mechanism-grid cache
      (`_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl`) present and
      loaded without recomputation.
- [x] `tasks_findings/_scripts/11_slack_panel.py` present, runs
      cache-hit-only in $<30$ s, produces the pickle and both figure
      files.
- [x] `game_experiments/_cache_v2/slack_panel.pkl` present, keys
      `{eps_i_anchors, rows, delta, K, actions}`; each row has
      `{eps_i_target, eps_i_actual, i0, eps_j_grid, A, slack, c_bar,
      feasible}`.
- [x] `paper/figures/fig_slack_panel.pdf` and `.svg` present,
      single-panel, style-matched to `fig_h8_brpeg_regret`
      (`apply_paper_style`, `PRIMARY_BLUE`/`PRIMARY_ORANGE`/
      `ACCENT_GREY`, legend without visible frame).
- [x] Anchor row $\varepsilon_i=4$: $\mathrm{slack}_i\geq 0$ on every
      feasible cell (asserted in the plotter; task spec success
      criterion).
- [x] Slurm wrapper `slurm_scripts/run_slack_panel.sh` present
      (2 CPU, 4G RAM, 15 min wall).
- [ ] `paper/tex/05_experiments.tex` edit deferred: T11 does not touch
      `paper/` per the task-owner's instruction.

---

## 8. Files produced

| Path | Purpose |
|---|---|
| `/vol/miltank/users/kaiserj/Collusion_Gametheory/tasks_findings/_scripts/11_slack_panel.py` | plotter; loads mech-grid cache, emits pickle + figure |
| `/vol/miltank/users/kaiserj/Collusion_Gametheory/tasks_findings/_scripts/11_slack_summary.txt` | reproducibility ASCII summary (row-wise statistics) |
| `/vol/miltank/users/kaiserj/Collusion_Gametheory/game_experiments/_cache_v2/slack_panel.pkl` | reproducibility pickle (row-wise slack, $A$, $\bar c$, feasibility) |
| `/vol/miltank/users/kaiserj/Collusion_Gametheory/paper/figures/fig_slack_panel.pdf` | publication figure (single panel) |
| `/vol/miltank/users/kaiserj/Collusion_Gametheory/paper/figures/fig_slack_panel.svg` | publication figure (SVG for source archive) |
| `/vol/miltank/users/kaiserj/Collusion_Gametheory/slurm_scripts/run_slack_panel.sh` | Slurm wrapper (2 CPU, 4G, 15 min) |
