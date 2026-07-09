# T13 -- H4 aware-target coalition Stackelberg

**Novelty priority:** M  **Effort:** L (real: Slurm job on 24 CPU cores completed in well under the 2h wall budget)
**Status:** COMPLETE (figure-rewrite pass, 2026-07-02). Cache
`_cache_v2/H4.pkl` produced by Slurm job on 2026-07-01; figure
`paper/figures/fig_h4_aware_target.{pdf,svg}` regenerated 2026-07-02
after user feedback ("too many floats on the axes, no titles, not
self-contained") as a single log-$y$ panel with no titles and no
floating annotations -- all descriptive content moved into the LaTeX
caption below. Findings numbers unchanged from the 2026-07-01 finalizer
pass. No files under `paper/tex/` touched (per subagent scope; paper
edits are the next follow-up task's remit).

**Verdict: NEGATIVE branch.** Aware-target best response *collapses* the
C4 attack. $A_t^{\text{aware}}(k) \in [0.014, 0.041]$ across
$k \in \{1,2,3,4\}$, i.e.\ **17.9x to 34.8x lower** than the passive-target
baseline $A_t^{\text{passive}}(k) \in [0.483, 0.728]$. The aware curve is
also **not monotone in $k$** (k=3 dips below k=2). The maximum aware
$A_t$ across all $k$ (0.041 at $k=4$) is $\sim$10.9x below even the
$k{=}0$ no-coalition baseline $A_t=0.442$. C4 as written is against a
naive target.

---

## 1 Goal and narrative role

The paper's C4 subsection (`paper/tex/05_experiments.tex:127-170`) reports a
five-player coalition Stackelberg attack against a target pinned at
$\varepsilon_t = 32$: the target's realised MIA advantage climbs from
$0.442$ ($k=0$) to $0.728$ ($k=4$), a $65\%$ inflation. `NOVELTY_REVIEW.md:22`
flags this as reviewer risk N2 -- "a rerun of Paper B's Fig. 5/6", i.e.\ the
$62\%$ budget-manipulation attack of the companion paper (Kaiser et al.\
SaTML 2026, `paper/refs.bib:18`, key `kaiser2026your`; companion PDF
`2601.12922v1.pdf` Section V.B). The reviewer objection is that C4 as
written is a *relabelling* of an existing empirical attack, not a genuine
game-theoretic contribution.

H4 is the differentiator prescribed in `NOVELTY_ACTION_PLAN.md:143` (Path 2)
and `EXPERIMENT_IDEAS.md:107-118`: rerun C4 with the target
**best-responding** instead of pinned at $32$. This turns the target from
a passive victim into a Stackelberg follower and asks whether the same
attack survives.

**The literature term "aware target"** is not a canonical label; it maps
onto the standard Stackelberg-follower rule used in strategic
classification and adversarial-ML privacy games (leader commits, follower
best-responds). Closest matches in the recent literature:

- Strategic classification treats the interaction as a Stackelberg game
  where the classifier is the leader and strategic agents are followers
  who best-respond ("Who Leads and Who Follows in Strategic
  Classification?", Zrnic et al.\ 2021, arXiv:2106.12529).
- Adversarial machine unlearning models unlearning as a Stackelberg game
  between a defender and a membership-inference attacker (Liu et al.,
  OpenReview iQIQT88prm, 2024).
- Bayesian Stackelberg privacy games explicitly compute the adversary's
  best response to a committed protection mechanism (Shokri et al.,
  "Privacy Games: Optimal User-Centric Data Obfuscation", arXiv:1402.3426).

In our PEG the roles are reversed compared to the last of these: the
**coalition is the leader** (commits to $\varepsilon_C$) and the **target
is the follower** (best-responds by choosing $\varepsilon_t$ to maximise
its own utility $u_t = V(\sigma) - w_t A_t$). H4 answers the question the
companion paper cannot, because in its threat model the target is fixed.

**Two outcome branches** (both planned, per `EXPERIMENT_IDEAS.md:115-118`):

- *Positive*: aware-target $A_t^\star(k)$ still monotone in $k$ and
  significantly above the $k=0$ baseline. Retains the "recovers the
  empirical attack" framing; C4 is upgraded from replication to
  equilibrium-level recovery.
- *Negative*: aware-target $A_t^\star(k)$ collapses. C4 is honestly
  softened to "against a naive target"; the paper still gains from the
  H4 figure, because it now documents the boundary of the attack.

Either way, the paper's C4 defensibility improves.

---

## 2 Precise protocol

### 2.1 Regime

Mirrors the passive-target C4 configuration in
`game_experiments/run_continuous.py:247-305` exactly, changing only the
target's decision rule.

| symbol | value | source |
|---|---|---|
| $n$ | $5$ symmetric aware players | `run_continuous.py:254`, `game/utility.py:61` |
| target index | $0$ | `run_continuous.py:255` |
| $|G_i|$ | $5000$ per player | `run_continuous.py:70-75` |
| batch $E_b$ | $128$ | `run_continuous.py:59` |
| epochs | $5$ (steps $= 5 \cdot 5000 \cdot 5 / 128 = 976$) | `run_continuous.py:75` |
| $\delta$ | $10^{-12}$ | `run_continuous.py:50` |
| $w_i$ | $1$ | `game/utility.py:39` |
| $V(\sigma)$ | $1/(1+\sigma^2)$ | `game/utility.py:42-53` |
| $A_i$ | subsampled-Gaussian PLD advantage | `game/advantage.py:mia_advantage_subsampled_gaussian` |
| eps grid | $14$ log-spaced points in $[0.5, 64]$ | `run_continuous.py:257` |
| non-coalition non-target $\varepsilon$ | $32$ (fixed) | `run_continuous.py:255` |

### 2.2 Aware-target rule

For each coalition size $k \in \{1, 2, 3, 4\}$ and each coalition action
$\varepsilon_C$ on the grid, the target picks
$\varepsilon_t^\star(\varepsilon_C, k) = \arg\max_{\varepsilon_t} u_t(\varepsilon_t, \varepsilon_C; k)$
on the same 14-point grid. The coalition then picks
$\varepsilon_C^\star(k) = \arg\max_{\varepsilon_C} A_t(\varepsilon_t^\star(\varepsilon_C, k), \varepsilon_C; k)$.
This is the Stackelberg game with **$\lambda \to \infty$** in the coalition
objective of the paper's Definition 5: the coalition's leader objective is
the target's realised MIA advantage.

Anchor code: `game/game.py:169-221` implements the same idea for the
discrete action grid; `game/solvers.py:103-138` `coalition_attack_sweep`
already exposes a `non_coalition_action=None` branch that best-responds
target AND non-coalition non-target players. We do **not** reuse it: H4
mirrors C4 verbatim (fixed non-coalition non-target at $32$, only the
target best-responds), so the comparison isolates the aware-target rule
as the single moved variable. Code lives in
`game_experiments/run_h4.py`.

### 2.3 Metric and reported quantities

For each $k$:

- $A_t^{\text{passive}}(k)$ from `_cache_v2/C4.pkl` (already produced).
- $A_t^{\text{aware}}(k)$ from H4 sweep (this task).
- $\varepsilon_t^\star(k)$: target's Stackelberg-follower budget.
- $\varepsilon_C^\star(k)$: coalition's Stackelberg-leader budget.
- $\sigma$ and $u_t$ at the equilibrium (auxiliary).

Sweep size: $4 \times 14 \times 14 = 784$ mechanism solves plus multiset
dedup via `solve_mechanism_cached` (`game/mechanism.py:163-199`).

### 2.4 Acceptance conditions on the raw data

- Cache file `_cache_v2/H4.pkl` contains `per_k` with `feasible=True` for
  every $k \in \{1, 2, 3, 4\}$.
- Passive-target curve on the plot numerically matches the C4 headline
  numbers $A_t(0)=0.442$, $A_t(4)=0.728$ within $0.01$ (reload from
  `_cache_v2/C4.pkl`; no recompute).
- Coalition equilibrium action $\varepsilon_C^\star(k)$ prefers the low
  end of the grid (otherwise the coalition is not attacking).

---

## 3 Realised results (from `_cache_v2/H4.pkl`)

Slurm run: cache written 16:13 (35 KB); plot regenerated 16:14. All
$k \in \{1,2,3,4\}$ report `feasible=True`.

### 3.1 Aware-target Stackelberg equilibria

| $k$ | $\varepsilon_C^\star$ | $\varepsilon_t^\star$ | $A_t^\star$ (aware) | $A_t$ (passive C4) | drop factor | $\sigma$ | $u_t$ | $A_C$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.06 |  9.90 | **0.0139** | 0.4828 | 34.8x | 0.433 | 0.828 | 0.657 |
| 2 | 30.34 |  9.90 | **0.0210** | 0.5310 | 25.2x | 0.450 | 0.810 | 0.460 |
| 3 | 64.00 | 14.38 | **0.0187** | 0.6074 | 32.4x | 0.371 | 0.860 | 0.748 |
| 4 |  0.50 |  0.73 | **0.0406** | 0.7277 | 17.9x | 2.400 | 0.107 | 0.025 |

Passive baseline (from `_cache_v2/C4.pkl`, no recompute):
$A_t(0)=0.4420$, $(0.4828, 0.5310, 0.6074, 0.7277)$ for $k=1..4$.
Matches paper prose to three decimals (acceptance condition (2) satisfied
to $<10^{-3}$, well inside the 0.01 tolerance).

### 3.2 Monotonicity

Aware curve is **not monotone in $k$**: 0.0139, 0.0210, 0.0187, 0.0406.
Kendall $\tau = 0.667$ ($p=0.333$, not significant at $n=4$). The passive
curve is strictly monotone ($\tau = 1.0$).

### 3.3 Regime interpretation

The equilibrium at $k=4$ is qualitatively different from $k=1..3$:
- **$k \in \{1,2,3\}$:** coalition plays a *permissive* budget
  ($\varepsilon_C^\star \in [30, 64]$) and the target retreats to
  moderate $\varepsilon_t^\star \in [9.9, 14.4]$; $\sigma$ stays low
  ($0.37$-$0.45$) and $A_t^\star$ is a fraction of a percent.
- **$k = 4$:** the coalition's leader problem has **two competing local
  optima**. The Stackelberg-argmax is a *poisoning corner* at the grid
  floor $\varepsilon_C^\star = 0.5$; the target follows to
  $\varepsilon_t^\star = 0.73$; $\sigma = 2.4$; $u_t = 0.107$;
  $A_t^\star = 0.041$ (this is the highest aware value but still
  $17.9\times$ below passive and $10.9\times$ below the $k{=}0$
  baseline). The *interior* local optimum sits at $\varepsilon_C = 30.3$,
  $\varepsilon_t^\star = 14.4$, $A_t = 0.017$, $A_C = 0.254$, which
  continues the $k{=}1{,}2{,}3$ pattern smoothly and would be the
  attractor an unconstrained coalition would rationally pick (the corner
  gives them only $A_C = 0.025$). Both are $\gg 15\times$ below the
  passive curve, and the collapse story is robust to which one you
  believe is "the" equilibrium. See Section 5 for the diagnosis and the
  figure treatment.

### 3.4 Cache provenance

- `_cache_v2/C4.pkl` **exists** (verified). Contains rows for
  $k=0..4$ with `A_target` values matching the paper prose exactly
  (0.442, 0.483, 0.531, 0.607, 0.728). The H4 figure reloads it and
  overlays as the "passive target" curve.
- `_cache_v2/H4.pkl` **produced** by Slurm run (35 KB, 16:13).
  Idempotent plot pass at 16:14 skipped the sweep and wrote the figure.
- Slurm jobs `173474` (K=200 locus) and `173491` (T19) were unrelated
  and left running; H4 got its own job.
- Realised wall: sweep completed well inside the 2h Slurm cap
  (`H4.pkl` timestamped 16:13). PLD-LRU dedup on the 14-point coalition
  grid recycled most of the 784 evaluations.

---

## 4 Slurm script (staged, not submitted)

**File:** `slurm_scripts/run_h4_aware_target.sh` (mirrors
`run_locus_K200.sh`). Body:

```bash
#SBATCH --job-name=peg_h4_aware
#SBATCH --output=./out/peg_h4_aware-%A.out
#SBATCH --error=./out/peg_h4_aware-%A.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=12G

export PYTHONUNBUFFERED=true
export PEG_N_WORKERS=24
source ./.venv/bin/activate
python -m game_experiments.run_h4
```

Budget rationale:

- 24 CPUs $\times$ $\sim$$8$ minutes each $\times$ 33 fresh profiles per
  worker (784 / 24) $\approx$ 15-30 min wall. `--time=02:00:00` is a
  $4$-$8\times$ cushion consistent with C6/C7/C8's ratio.
- Memory: PLD LRU caches at $n{=}5$ are lighter than the $K{=}200$ 2-player
  case ($24$ G there), so $12$ G is enough at $24$ workers.
- No GPU (`CLAUDE.md` Rule 3: mechanism-solver work is CPU-only).

**Not submitted** per task prompt.

---

## 5 Publication-quality figure

**File:** `paper/figures/fig_h4_aware_target.{pdf,svg}` (target path).
Rewritten 2026-07-02 (second pass): dropped panel titles, dropped
floating axis annotations (ratios, poisoning-corner arrow, grid-floor
reference, $k{=}0$ baseline line), collapsed to a single log-$y$ panel.
All descriptive content moved into the LaTeX caption so the axes read as
a first-look self-contained comparison.

**Layout** (encoded in `game_experiments/run_h4.py:plot`, style-matched
to `fig_h8_brpeg_regret.pdf` at the mark-and-line level: paper-style
grid, thin lines, `fontsize=7` sans legend, LaTeX-native math).

- **Single panel, log-$y$.** $x$: coalition size $k \in \{0,1,2,3,4\}$;
  $y$ log-scaled $A_t^\star(k) \in [8\!\times\!10^{-3}, 1]$.
- Orange squares with dashed line: passive target pinned at
  $\varepsilon_t{=}32$ (loaded from `_cache_v2/C4.pkl` for $k{=}0..4$).
- Blue circles with solid line: aware target
  ($\varepsilon_t^\star \in \arg\max u_t$, from `_cache_v2/H4.pkl`) for
  $k{=}1..4$.
- Blue hollow circle at $k{=}4$: the aware coalition's *interior*
  local optimum, obtained by excluding the two grid-floor coalition
  budgets and re-taking the target-BR argmax over $A_t$
  ($A_t{=}0.017$). No axis label; explained in caption.
- Legend at `center right` (only mostly-empty region between the two
  curves). No titles. No text annotations. No auxiliary reference
  lines.

**Diagnosis of the $k{=}4$ corner** (verified from `_cache_v2/H4.pkl`,
row-by-row inspection; unchanged from previous pass). The Stackelberg
leader has two competing local optima at $k{=}4$: (i) the *permissive*
branch at $\varepsilon_C{=}30.3, \varepsilon_t^\star{=}14.4$,
$A_t{=}0.017$, $A_C{=}0.254$, continuous with the $k{=}1{,}2{,}3$ optima;
(ii) the *poisoning corner* at the grid floor
$\varepsilon_C{=}0.5, \varepsilon_t^\star{=}0.73$, $\sigma{=}2.4$,
$A_t{=}0.041$, $A_C{=}0.025$. The corner wins on $A_t$ by a factor
of ${\sim}2.4$, but is a boundary artefact: it sits at the smallest
coalition action on the $14$-point $[0.5, 64]$ grid and gives the
coalition itself only $A_C{=}0.025$, so no rational coalition would run
it in an unrestricted game. Both points are shown on the figure (filled
$=$ boundary Stackelberg argmax, hollow $=$ interior local optimum);
the caption spells out the interpretation.

Palette matches the rest of the paper (`PRIMARY_ORANGE`, `PRIMARY_BLUE`
from `game/style.py`).

**Data provenance guard.** `run_h4.py:plot` refuses to draw the aware
curve if `_cache_v2/H4.pkl` contains no feasible rows, and reloads the
passive curve from `_cache_v2/C4.pkl` at plot time. There is no
synthetic-data fallback.

**Caption (fenced LaTeX for direct paste into a later T13 follow-up).**
Absorbs every annotation removed from the axes: the ratio scale
($18{-}35\times$), the $k{=}0$ no-coalition baseline value, the
target-follower and coalition-leader Stackelberg budgets that used to
sit in a second panel, and the poisoning-corner / interior-optimum
split at $k{=}4$.

```latex
\caption{\textbf{H4 -- The C4 attack collapses against an aware target.}
Target's realised MIA advantage $A_t^\star$ vs coalition size $k$
(log-$y$; larger is a worse attack for the target). \emph{Orange
squares (dashed):} passive target pinned at $\varepsilon_t{=}32$ as in
\cref{fig:C4}; $A_t$ climbs monotonically from $0.442$ at $k{=}0$ (no
coalition) to $0.728$ at $k{=}4$, recovering the empirical
budget-manipulation attack of \citet{kaiser2026your}. \emph{Blue
circles (solid):} aware target best-responding on the same $14$-point
log-spaced grid $\varepsilon \in [0.5, 64]$; $A_t^\star \in
\{0.014, 0.021, 0.019, 0.041\}$ for $k \in \{1,2,3,4\}$, i.e.\
$17.9\times$ to $34.8\times$ below the passive curve and $\sim$$11\times$
below even the $k{=}0$ baseline. The aware coalition's Stackelberg
budgets are $\varepsilon_C^\star \in [30, 64]$ for $k \le 3$ with the
target retreating to $\varepsilon_t^\star \in [9.9, 14.4]$. At $k{=}4$
the leader problem has two local optima: the filled blue marker is the
Stackelberg argmax, a boundary \emph{poisoning corner} at the grid floor
($\varepsilon_C{=}0.5$, $\varepsilon_t^\star{=}0.73$, $\sigma{=}2.4$)
that yields the coalition itself only $A_C{=}0.025$; the hollow blue
marker is the \emph{interior} local optimum
($\varepsilon_C{=}30.3$, $A_t{=}0.017$), which is continuous with
$k{\le}3$ and is what a rational coalition would play. Under either
$k{=}4$ interpretation the collapse is $\ge 17\times$. Scenario
otherwise matches \cref{fig:C4}: $n{=}5$ symmetric aware players,
$|G_i|{=}5000$, $E_b{=}128$, $5$ epochs, $\delta{=}10^{-12}$. Data:
\path{game_experiments/_cache_v2/H4.pkl} (aware curve),
\path{game_experiments/_cache_v2/C4.pkl} (passive curve, reloaded).}
\label{fig:H4}
```

---

## 6 Paper insertion plan

Per `NOVELTY_ACTION_PLAN.md:143-154` and the two outcome branches:

### 6.1 Location

New labelled subsection `\subsection{C4b --- Does the attack survive an
aware target?}` with `\label{sec:C4b}`, inserted at
`paper/tex/05_experiments.tex:171` (immediately after the C4 figure's
`\end{figure}` on line 170, before the C5 header on line 173).

`\includegraphics[width=\linewidth]{figures/fig_h4_aware_target.pdf}`
referenced by `\cref{fig:H4}`.

### 6.2 Caption edit for C4 (`paper/tex/05_experiments.tex:165-169`)

Retitle to open with (per `NOVELTY_ACTION_PLAN.md:150` Path 1):

> Recovery of the empirical attack of \citet{kaiser2026your} as a
> coalition Stackelberg best response against a passive target.

### 6.3 Selected branch: NEGATIVE

Rule from acceptance checklist (4) reads: *positive iff*
$A_t^{\text{aware}}(k)$ monotone non-decreasing in $k \in \{1,2,3,4\}$
AND $A_t^{\text{aware}}(4) > A_t^{\text{aware}}(1) + 0.05$.
Realised: not monotone ($k{=}3 < k{=}2$) AND
$A_t^{\text{aware}}(4) - A_t^{\text{aware}}(1) = 0.041 - 0.014 = 0.027 < 0.05$.
Both sub-conditions fail. Positive branch is not admissible; take
negative branch.

### 6.4 Negative-branch wording (replaces
`paper/tex/05_experiments.tex:152-160`; numbers filled from H4 cache)

> The strategic content of this experiment is the recovery of an
> empirical phenomenon as the rational equilibrium of the game we have
> defined, \emph{against a naive target}. The budget-manipulation attack
> of \citet{kaiser2026your}, reported there as a $62\%$ success rate
> across diverse datasets, is what a coalition would naturally do
> against a passive victim -- it is the coalition's best response in our
> PEG when the target does not respond. \cref{fig:H4} shows that if the
> target itself best-responds on the same $14$-point grid, its optimal
> budget shifts to $\varepsilon_t^\star \in [0.73, 14.4]$ and
> $A_t^\star$ falls to at most $0.041$ (at $k{=}4$) -- 18-35$\times$
> below the passive-target curve and $\sim$11$\times$ below even the
> $k{=}0$ no-coalition baseline. The aware-target extension neutralises
> the attack. The C4 subsection is therefore honest to read as the
> equilibrium outcome under a passive victim, not as a robust
> game-theoretic strengthening of the empirical attack.

### 6.5 Acceptance checklist -- finalizer status

| # | Item | Status |
|---|---|---|
| 1 | Cache `_cache_v2/H4.pkl` exists; `feasible=True` for $k \in \{1,2,3,4\}$ | DONE (verified, 4/4 feasible) |
| 2 | Passive curve matches $(0.442, 0.483, 0.531, 0.607, 0.728)$ within $0.01$ | DONE (matches to $<10^{-3}$) |
| 3 | `paper/figures/fig_h4_aware_target.{pdf,svg}` produced | DONE (20 KB PDF, 57 KB SVG, 16:14) |
| 4 | Outcome branch chosen by the monotonicity+gap rule | DONE (negative branch, both sub-conditions fail; see 6.3) |
| 5 | LaTeX edits per Section 6.2 + 6.4 | TODO (out of finalizer scope; follow-up T13-paper task) |
| 6 | `pdflatex -interaction=nonstopmode main.tex` clean | TODO (follow-up) |
| 7 | `\citet{kaiser2026your}` in C4 caption and C4b prose | TODO (follow-up) |
| 8 | No orphaned "aware target" references outside `sec:C4b` and fig caption | TODO (follow-up) |

---

## 7 Files touched

- `game_experiments/run_h4.py` -- experiment script (unchanged sweep),
  `plot()` rewritten 2026-07-02 to a single log-$y$ panel with no
  titles and no floating annotations.
- `slurm_scripts/run_h4_aware_target.sh` -- Slurm script (unchanged
  from previous pass).
- `game_experiments/_cache_v2/H4.pkl` -- produced by Slurm job (35 KB,
  2026-07-01 16:13; unchanged in this pass).
- `paper/figures/fig_h4_aware_target.{pdf,svg}` -- overwritten
  2026-07-02 from the same cache with the single-panel layout.
- `tasks_findings/_scripts/13_h4_plot.py` -- standalone copy of the
  new plotting code (produces the same PDF/SVG from the pickles).
- `tasks_findings/13_h4_aware_target.md` -- this file (Section 5 and
  caption rewritten for the single-panel layout).
- **Not touched:** `paper/tex/**` (per finalizer scope; paper prose
  edits are the next follow-up task's remit), `_cache_v2/C4.pkl`
  (reused, not modified), `_cache_v2/H4.pkl` (reused, not modified).
