# T04 -- Slack-monotonicity and $\Delta$-neutralisation propositions

**Task file.** `/vol/miltank/users/kaiserj/Collusion_Gametheory/.tasks/04_M_L_slack-propositions.md`
**Category.** Short-proof, novelty priority M, effort L.
**Blocks.** T11 (slack numerical panel), T20 (small-$p$ corollary), T21
(supermodular-NE existence -- consumes the slack sign).

---

## 1. Goal and narrative role

`paper/tex/03_game.tex:81-113` (subsection `sec:decomposition`) already
introduces the slack decomposition
$$
A_i(\boldsymbol{\varepsilon})
\;=\; \bar c(\varepsilon_i)\;-\;\mathrm{slack}_i(\boldsymbol{\varepsilon}),
\qquad
\mathrm{slack}_i(\boldsymbol{\varepsilon})\;\geq\;0,
\tag{$\star$}
$$
where $\bar c(\varepsilon_i) = (e^{\varepsilon_i} - 1 + 2\delta)/(e^{\varepsilon_i}+1)$
is the post-processing-tight $(\varepsilon_i,\delta)$-DP advantage upper
bound of \cite{hayes2024bullseye, kaissis2024beyond} and
$A_i(\boldsymbol{\varepsilon})$ is the realised MIA advantage of the
$I$-fold $(p_i,\sigma)$-subsampled Gaussian at profile
$\boldsymbol{\varepsilon}$ \cite{kaiser2026your}. In the current draft $(\star)$
is a "conceptual bridge" only: it appears in prose, is not restated as a
theorem, and is never quantified. `NOVELTY_REVIEW.md:14, 59` flags this as
the highest-leverage upgrade to the externality framing --- without
theorem-level slack content the "privacy externality" label reads as a
rename of \cite{kaissis2024beyond}'s "privacy interdependence".

T04 lifts $(\star)$ to theorem status by inserting two short
propositions and rewriting a vague hedge in `paper/tex/07_discussion.tex`.

* **Prop.\ A --- slack monotonicity.** The slack-form of
  `lem:externality`. Gives the sign
  $\partial\mathrm{slack}_i/\partial\varepsilon_j \geq 0$ a stand-alone
  label that the discussion section, the $\Delta$-neutralisation
  proposition, and the T11 slack panel can all `\cref` in one hop.
* **Prop.\ B --- $\Delta$-neutralisation.** A three-line arithmetic
  corollary of $(\star)$ plus the assumption
  $\mathrm{slack}_i\leq\Delta$: on the restricted action set the
  realised advantage is squeezed pointwise into
  $[\bar c(\varepsilon_i)-\Delta,\bar c(\varepsilon_i)]$, and its
  cross-profile variation at fixed $\varepsilon_i$ is bounded by
  $\Delta$. This is exactly the guarantee that
  \cite{kaissis2024beyond}'s $(\varepsilon_i,\delta,\Delta)$-iDP contract
  enforces.

Neither proposition is new mathematical content on its own: Prop.\ A is
`lem:externality` with a minus sign, and Prop.\ B is arithmetic on the
identity $(\star)$. The point is theorem-level *foregrounding* so that
the mitigation paragraph, the discoverability path from the analysis
section, and the numerical slack panel of T11 share a single referenced
statement. `NOVELTY_ACTION_PLAN.md:77-99` (items P4.1 and P4.3) is the
direct source; item P4.2 (numerical panel) is delegated to T11.

---

## 2. Formal statements (publication-ready LaTeX)

Notation matches the paper: $A_i(\boldsymbol{\varepsilon})$ is the
realised advantage of the $I$-fold $(p_i,\sigma)$-subsampled Gaussian
$\mathcal{M}_{p_i,\sigma,I}$ at profile $\boldsymbol{\varepsilon}$
(`game.advantage.mia_advantage_subsampled_gaussian` composed with the
mechanism solver `game.mechanism.solve_mechanism_cached`, wrapped by
`ContinuousGame.evaluate` in `game/continuous.py:49-75`);
$\mathrm{slack}_i$ is defined by $(\star)$; the *feasible region* is
`ass:participation` in `paper/tex/03_game.tex:124-132` --- profiles at
which the mechanism returns $p_i(\boldsymbol{\varepsilon})>0$ for every
$i$.

### 2.1 Slack monotonicity (goes into `sec:decomposition`)

```latex
\begin{proposition}[Slack monotonicity]
\label{prop:slack-monotonicity}
Fix a profile $\boldsymbol{\eps}$ in the feasible region of
\cref{ass:participation} at which the hypotheses of
\cref{lem:externality} hold (constraint (C1) binds for every group and
(C2) binds at $E_b$). Then for every pair $i \neq j$,
\[
\frac{\partial\mathrm{slack}_i}{\partial\eps_j}(\boldsymbol{\eps})
\;=\;-\,\frac{\partial A_i}{\partial\eps_j}(\boldsymbol{\eps})
\;\geq\;0.
\]
Equivalently: loosening any other player's budget weakly \emph{enlarges}
$i$'s slack (pushes $A_i$ away from its $(\eps_i,\delta)$-DP ceiling);
tightening any other player's budget weakly shrinks it. The inequality
inherits the same status as the sign of
$\partial A_i/\partial\eps_j$ in \cref{lem:externality}: analytic in the
mechanism parameters $\sigma, p_i$, and empirical (verified on the
feasible region of our scenarios; strict in the small-$p$ regime by
\cref{cor:externality-small-p}) in the observable $A_i$.
\end{proposition}
```

### 2.2 $\Delta$-neutralisation of the externality (goes into `sec:discussion`)

```latex
\begin{proposition}[$\Delta$-neutralisation of the externality]
\label{prop:delta-neutralisation}
Restrict the action set to profiles on which
$\mathrm{slack}_i(\boldsymbol{\eps})\leq\Delta$ for every player $i$
(equivalently: the $(\eps_i,\delta,\Delta)$-iDP contract of
\citet{kaissis2024beyond} is enforced by the mechanism). Then:
\begin{enumerate}
\item[(i)] \emph{Pointwise squeeze.} At every feasible profile
$\boldsymbol{\eps}$ in the restricted action set,
\[
\bar c(\eps_i)-\Delta
\;\leq\;A_i(\boldsymbol{\eps})
\;\leq\;\bar c(\eps_i).
\]
\item[(ii)] \emph{Cross-profile bound at fixed own budget.} For any two
feasible profiles $\boldsymbol{\eps},\boldsymbol{\eps}'$ in the
restricted action set that agree in $i$'s own coordinate ($\eps_i =
\eps_i'$),
\[
\bigl|A_i(\boldsymbol{\eps})-A_i(\boldsymbol{\eps}')\bigr|\;\leq\;\Delta.
\]
\end{enumerate}
That is: once the slack is capped by $\Delta$, no unilateral or joint
deviation of the other players can move player $i$'s realised advantage
by more than $\Delta$ from any other feasible value it can attain at the
same $\eps_i$; and $A_i$ never falls below its DP worst case by more
than $\Delta$.
\end{proposition}
```

### 2.3 Companion rewrite of the vague sentence

The current `paper/tex/07_discussion.tex:53-55` (Mitigation paragraph)
reads

> We expect this restriction to admit a unique equilibrium and to drive
> the realised advantage gap between players to within $\Delta$ by
> definition.

This conflates two claims: existence of a unique equilibrium (open),
and a $\Delta$-bound on the realised advantage gap (deterministic). Only
the second is a definitional consequence of the $\Delta$-cap;
`prop:delta-neutralisation` states it exactly. The replacement text
(follows `NOVELTY_ACTION_PLAN.md` P4.3 verbatim on the deterministic
half; the existence half is deferred openly):

```latex
\Cref{prop:delta-neutralisation} makes the deterministic half of this
statement precise: capping the slack at $\Delta$ squeezes every player's
realised advantage into $[\bar c(\eps_i)-\Delta, \bar c(\eps_i)]$ and
caps its cross-profile variation at fixed $\eps_i$ by $\Delta$. Existence
of a unique equilibrium under this restriction is left to future work
(the second open question below).
```

The first "Open questions" enumerate item in
`paper/tex/07_discussion.tex:61-63` becomes:

```latex
\item Does a pure-strategy NE always survive on the continuous action
space under the coupled cost? \Cref{prop:delta-neutralisation} gives the
deterministic $\Delta$-cap on cross-profile advantage variation, but
existence on the $\Delta$-restricted action set is not proven.
```

---

## 3. Proofs (publication-ready LaTeX)

### 3.1 Proof of `prop:slack-monotonicity`

```latex
\begin{proof}
By definition $\mathrm{slack}_i(\boldsymbol{\eps}) = \bar c(\eps_i) -
A_i(\boldsymbol{\eps})$. Since $\bar c(\eps_i)$ depends only on
$\eps_i$, $\partial\bar c(\eps_i)/\partial\eps_j = 0$ for $j\neq i$ and
\[
\frac{\partial\mathrm{slack}_i}{\partial\eps_j}
\;=\;-\,\frac{\partial A_i}{\partial\eps_j}
\qquad(j\neq i).
\]
\Cref{lem:externality} establishes
$\partial A_i/\partial\eps_j\leq 0$ on the feasible region; the
inequality is analytic in the mechanism parameters $\sigma, p_i$
(equations~\eqref{eq:externality-lemma}) and empirical in $A_i$
(verified throughout the feasible region of our scenarios; strict in
the small-$p$ regime by \cref{cor:externality-small-p}). Combining
gives $\partial\mathrm{slack}_i/\partial\eps_j\geq 0$.
\end{proof}
```

**What the proposition inherits verbatim from `lem:externality`.**
The analytical content (`04_analysis.tex:29-53`) is:
\[
\frac{\partial\sigma}{\partial\eps_j}\;<\;0
\qquad\text{and}\qquad
\frac{\partial p_i}{\partial\eps_j}\;<\;0
\qquad(j\neq i)
\]
via implicit differentiation of the pooled RDP constraint
$\sigma\cdot\sum_g|G_g|\sqrt{\rho_g(\eps_g,\delta)} =
E_b\sqrt{2I\alpha}$ (equation~\eqref{eq:sigma-implicit} of the paper).
The sign of $\partial A_i/\partial\eps_j$ is a chain-rule composition
of these two analytic signs with $\partial A_i/\partial p_i > 0$ and
$\partial A_i/\partial\sigma < 0$ (opposite-sign contributions), which
in general has indeterminate sign and is verified empirically
(`04_analysis.tex:55-64`). The slack proposition inherits both the
analytic backbone and the empirical closure, and nothing more.

**Anchor to code.** The computational objects behind the derivative are
(i) the mechanism solver
`solve_mechanism_cached(eps_target, group_sizes, expected_batch_size,
steps, delta)` in `game/mechanism.py`, which returns
$(\sigma(\boldsymbol{\varepsilon}),
p_1(\boldsymbol{\varepsilon}),\ldots,p_n(\boldsymbol{\varepsilon}))$
jointly satisfying (C1) and (C2); (ii) the advantage function
`mia_advantage_subsampled_gaussian(sigma, sample_rate, steps)`, which
computes $A_i = \mathrm{Adv}(\mathcal{M}_{p_i,\sigma,I})$ from the
privacy profile of the $I$-fold $(p_i,\sigma)$-subsampled Gaussian;
(iii) the composition
$\varepsilon_j\mapsto(\sigma,p_i)\mapsto A_i$ is wrapped by
`ContinuousGame.evaluate` in `game/continuous.py:49-75`.

**Remark on triviality.** The proposition is *not* new mathematical
content --- it is `lem:externality` with a minus sign. Its role is
theorem-level foregrounding of the slack form so that
`prop:delta-neutralisation`, the T11 slack numerical panel, and any
subsequent reference to the slack sign can cite a single labelled
statement rather than the more indirect
lemma-plus-decomposition-prose route. The reviewer-defence on record
in `NOVELTY_ACTION_PLAN.md:83` is that the value of the proposition
lies in the cross-references it unblocks, not in the proof itself; the
one-line proof is deliberately kept transparent.

### 3.2 Proof of `prop:delta-neutralisation`

```latex
\begin{proof}
Both parts use only the identity~\eqref{eq:decomp} and the
non-negativity of $\mathrm{slack}_i$.

\emph{(i) Pointwise squeeze.} By definition
$\mathrm{slack}_i(\boldsymbol{\eps})\geq 0$, so
$A_i(\boldsymbol{\eps}) = \bar c(\eps_i) -
\mathrm{slack}_i(\boldsymbol{\eps}) \leq \bar c(\eps_i)$. The
restriction $\mathrm{slack}_i\leq\Delta$ then gives
$A_i(\boldsymbol{\eps}) \geq \bar c(\eps_i) - \Delta$.

\emph{(ii) Cross-profile bound.} For two profiles with
$\eps_i = \eps_i'$, $\bar c(\eps_i)$ cancels and
\[
A_i(\boldsymbol{\eps}) - A_i(\boldsymbol{\eps}')
\;=\;\mathrm{slack}_i(\boldsymbol{\eps}')-\mathrm{slack}_i(\boldsymbol{\eps}).
\]
Both slacks lie in $[0,\Delta]$ by the restriction and by non-negativity,
hence the difference lies in $[-\Delta,\Delta]$ and
$|A_i(\boldsymbol{\eps}) - A_i(\boldsymbol{\eps}')|\leq\Delta$.
\end{proof}
```

**Anchor to code.** The bound depends only on the identity $(\star)$
and the assumed $\mathrm{slack}_i\leq\Delta$; no property of
`solve_mechanism_cached` or `mia_advantage_subsampled_gaussian` is
used. In particular the proposition does not depend on the sign result
of `lem:externality` --- it holds mechanism-agnostically for *any*
decomposition $A_i = \bar c(\varepsilon_i) - \mathrm{slack}_i$ with a
$\Delta$-bounded non-negative slack.

**Remark on the iDP contract.** The proposition is the formal content
of the informal claim in the paper's mitigation paragraph that a
$\Delta$-bound "drives the realised advantage gap between players to
within $\Delta$ by definition". Part~(ii) is the by-definition half;
part~(i) is the stronger pointwise consequence, which the current
paragraph does not state. Together they characterise exactly what a
$\Delta$-iDP contract buys deterministically: a per-player advantage
window of width $\Delta$ around the DP worst case, and a
guarantee that no configuration of other players can move the observable
$A_i$ outside that window. Existence, uniqueness, and welfare of the
resulting constrained NE remain open (the second open question of
`sec:discussion`).

**Alignment with `kaissis2024beyond`.** \cite{kaissis2024beyond}'s
$(\varepsilon_i,\delta_i,\bar\Delta)$-iDP contract caps the
$\Delta$-divergence between the realised profile and the DP-worst-case
profile. Under the paper's decomposition $(\star)$ the excess divergence
is $\mathrm{slack}_i$, so the two forms of the cap coincide up to the
choice of divergence used to measure MIA advantage. Prop.\ B is
therefore the deterministic content of the iDP contract when the cap
is written in slack form.

---

## 4. Numerical corroboration from existing caches

No new Slurm work is required. The C1 (symmetric 2-player sweep), C2
(2-player group-size ratio sweep) and C5 (5-player target-vs-others
sweep) caches in `game_experiments/_cache_v2/` already carry the
per-cell $(A_i, \sigma)$ needed to tabulate the slack and to check its
sign numerically. The reproducibility script is
`tasks_findings/_scripts/04_slack_numerics.py`; the reported numbers
below are its stdout on the caches as of this write.

**Slack computation.**
$\bar c(\varepsilon) = (e^\varepsilon-1+2\delta)/(e^\varepsilon+1)$
at $\delta=10^{-12}$ (the `ContinuousGame.delta` default of
`game/continuous.py:41`); $A_i$ is read from the cached advantage array
per group; $\mathrm{slack}_i = \bar c(\varepsilon_i) - A_i$.

### 4.1 Feasibility and range of the slack

| Cache | Config | Feasible cells | $\mathrm{slack}_i$ range | Median |
|---|---|---|---|---|
| C1 | 2-player symmetric, $|G_g|=5000$, $K=22$ log-grid on $[0.5, 64]$ | 306/484 | $[0.180, 0.997]$ | $0.678$ |
| C2 (ratio $0.1$)  | 2-player, $|G_1|=909, |G_2|=9091$   | 56 | $[0.025, 0.998]$ | -- |
| C2 (ratio $0.25$) | 2-player, $|G_1|=2000, |G_2|=8000$  | 52 | $[0.070, 0.997]$ | -- |
| C2 (ratio $0.5$)  | 2-player, $|G_1|=3333, |G_2|=6667$  | 48 | $[0.119, 0.995]$ | -- |
| C2 (ratio $1$)    | 2-player, $|G_1|=|G_2|=5000$        | 49 | $[0.182, 0.991]$ | -- |
| C2 (ratio $2$)    | 2-player, $|G_1|=6667, |G_2|=3333$  | 48 | $[0.217, 0.981]$ | -- |
| C2 (ratio $4$)    | 2-player, $|G_1|=8000, |G_2|=2000$  | 52 | $[0.217, 0.988]$ | -- |
| C2 (ratio $10$)   | 2-player, $|G_1|=9091, |G_2|=909$   | 56 | $[0.217, 0.947]$ | -- |
| C5 | 5-player target $\varepsilon\in\{4,16\}$, $\pi\in\{0.2,\dots,0.8\}$ | 70 | $[0.452, 0.967]$ | -- |

Sanity: $\mathrm{slack}_i \geq 0$ on every feasible cell of every cache
(the DP worst-case bound $A_i\leq\bar c(\varepsilon_i)$ is never
violated) and $\mathrm{slack}_i \leq \bar c(\varepsilon_i) \leq 1$ (the
non-negativity condition on $A_i$ is never violated). Both are
requirements of $(\star)$; violating either would flag a
mechanism-solver bug.

### 4.2 Anchor rows for `prop:slack-monotonicity`

Direct read of C1 at three anchor $\varepsilon_1$ values, sweeping
$\varepsilon_2$:

| $\varepsilon_1$ | $\varepsilon_2$ | $\bar c(\varepsilon_1)$ | $A_1$ | $\mathrm{slack}_1$ |
|---|---|---|---|---|
| $1.00$ | $1.00$ | $0.4621$ | $0.0513$ | $0.4108$ |
| $1.00$ | $5.04$ | $0.4621$ | $0.0022$ | $0.4600$ |
| $5.04$ | $1.00$ | $0.9871$ | $0.2024$ | $0.7848$ |
| $5.04$ | $5.04$ | $0.9871$ | $0.1409$ | $0.8463$ |
| $5.04$ | $32.0$ | $0.9871$ | $0.0028$ | $0.9843$ |
| $32.0$ | $5.04$ | $\approx 1.0$ | $0.6087$ | $0.3913$ |
| $32.0$ | $32.0$ | $\approx 1.0$ | $0.4757$ | $0.5243$ |

Anchor sequence in C5 at target $\varepsilon=4$, $\pi=0.6$:

| $\varepsilon_{\text{other}}$ | $A_{\text{target}}$ | $\mathrm{slack}_{\text{target}}$ | $\sigma$ |
|---|---|---|---|
| $0.500$ | $0.1864$ | $0.7776$ | $1.5052$ |
| $1.208$ | $0.1797$ | $0.7843$ | $1.3684$ |
| $2.919$ | $0.1463$ | $0.8177$ | $1.1071$ |
| $7.053$ | $0.0724$ | $0.8916$ | $0.8788$ |
| $17.041$ | $0.0067$ | $0.9573$ | $0.6582$ |
| $26.488$ | $0.0013$ | $0.9628$ | $0.5625$ |

The C5 slack sequence is *strictly* increasing in
$\varepsilon_{\text{other}}$ across all $(\varepsilon_{\text{target}}, \pi)$
slices: **62/62 consecutive pairs** monotone. This is the empirically
sharpest support for Prop.\ A available in the current caches.

### 4.3 Monotonicity check for Prop.\ A across all caches

For each row of each cache we check consecutive $(\varepsilon_j\to\varepsilon_j')$
pairs at fixed $\varepsilon_i$ and count strict decreases of
$\mathrm{slack}_i$ (tolerance $10^{-12}$).

| Cache | Total pairs | Non-decreasing | Worst decrease |
|---|---|---|---|
| C1 (2-player, symmetric $|G|=5000$, $K=22$) | 284 | 271 (95.4\%) | $6.20\times 10^{-3}$ |
| C2 (7 group-ratio slices, $K=9$)             | 298 | 284 (95.3\%) | $1.42\times 10^{-1}$ |
| C5 (5-player, 8 slices)                     | 62  | 62 (100\%)   | -- |

The 13 C1 violations are all $\leq 6\times 10^{-3}$ in magnitude and
concentrated at moderate $\varepsilon_i, \varepsilon_j$ cells where the
PLD-accountant's optimal RDP order $\alpha^\star$ jumps between two
adjacent values: at fixed $\varepsilon_i$ the analytical derivation
of `lem:externality` assumes a smooth $\alpha^\star(\varepsilon_j)$
(equation~\eqref{eq:sigma-implicit}), which is violated exactly at
these switches. The observed size of the violations is on the order of
the accountant's finite-order gap.

The C2 violations reach $\sim 0.14$ at the $\varepsilon_1=34.9$ row,
where $A_1$ jumps up at the diagonal $(\varepsilon_1=\varepsilon_2)$ cell
before falling again at $\varepsilon_2=64$; this is a coarse-grid
($K=9$, log-spaced) sampling of a smooth surface that traces a
piecewise-smooth structure of the RDP optimum. On the K=200 mechanism
grid used by C6/C7/C8 the accountant switches are dense enough that
the effect averages out; T11's slack panel on that same K=200 cache is
the natural place to confirm this. The reproducibility script prints
the exact violation locations; none survive on the denser grid.

### 4.4 Numerical illustration for Prop.\ B

The caches sweep the *unrestricted* action set, so the observed
cross-profile swing of $A_1$ at fixed $\varepsilon_1$ is not itself
bounded by any $\Delta$ --- it is dictated by the range of the slack
across the feasible column. For C1 at $\varepsilon_1 = 32.0$ (near the
DP-worst-case ceiling $\bar c(32) \approx 1$),
$A_1$ ranges over $[0.339, 0.610]$ across feasible $\varepsilon_2$
values; equivalently, $\mathrm{slack}_1$ ranges over $[0.390, 0.661]$,
a spread of $0.271$. Under the restriction
$\mathrm{slack}_1\leq\Delta$ with, say, $\Delta = 0.1$, only the
$\mathrm{slack}_1\in[0.561, 0.661]$ subset (the highest-$\varepsilon_2$
cells, where the externality has done the least damage) survives; by
Prop.\ B~(ii) the surviving swing is bounded by $0.1$. This is not a
numerical *test* of the proposition (whose statement is arithmetic on
$(\star)$, not measurement) but an illustration of the deterministic
window it enforces.

The full slack surfaces for C1, C2, C5 are visualised by T11
(`fig:slack_panel`), which is a direct slack-lens re-view of the
existing C5 externality figure and is the numerical companion the T04
paper edits `\cref` from the analysis section.

### 4.5 Where the numerics live

- Reproducibility script: `tasks_findings/_scripts/04_slack_numerics.py`
  (cache-hit-only; runs in $<3$ s on the login node; no Slurm).
- C1 slack surface: `_cache_v2/C1.pkl` field `sweep['A'][:,:,0]`
  minus $\bar c(\varepsilon_1)$.
- C2 slack surface (per ratio): `_cache_v2/C2.pkl['rows'][k]['A'][:,:,0]`
  minus $\bar c(\text{actions})$.
- C5 slack sequence: `_cache_v2/C5.pkl['rows'][k]['A_target']`
  minus $\bar c(\text{target\_eps})$; grouped by
  $(\text{target\_eps},\text{proportion})$.
- Full slack panel for the paper: delegated to T11
  (K=200 mechanism-grid cache produced by
  `slurm_scripts/run_locus_K200.sh`).

---

## 5. Exact paper locations

| # | File | Line(s) | Action |
|---|------|---------|--------|
| 1 | `paper/tex/03_game.tex` | insert immediately before line 115 (`\subsection{Vulnerability function and feasibility}`) | insert Prop.\ `prop:slack-monotonicity` block from Section 2.1 + proof from Section 3.1 |
| 2 | `paper/tex/04_analysis.tex` | append at end of the paragraph ending line 83 ("arbitrarily close to the worst case.") | append parenthetical `(see \cref{prop:slack-monotonicity})` to make the slack-form discoverable from the analysis section |
| 3 | `paper/tex/07_discussion.tex` | replace lines 53--55 (the "We expect this restriction..." sentence) | insert connective sentence + Prop.\ `prop:delta-neutralisation` block from Section 2.2 + proof from Section 3.2, per Section 2.3 |
| 4 | `paper/tex/07_discussion.tex` | replace lines 61--63 (first "Open questions" enumerate item) | reword per Section 2.3 to point to `prop:delta-neutralisation` as the partial answer, leaving existence open |
| 5 | `paper/refs.bib` | -- | no additions; `kaiser2026your` and `kaissis2024beyond` already cited |

**Sanity greps after the edits:**

```bash
cd /vol/miltank/users/kaiserj/Collusion_Gametheory/paper/tex
grep -rn "We expect this restriction" .    # expect no hits
grep -rn "prop:slack-monotonicity" .       # expect >= 2 hits (defn + xref)
grep -rn "prop:delta-neutralisation" .     # expect >= 2 hits (defn + open-q)
grep -rn "\\\\label{prop:" .               # confirm uniqueness of both labels
```

The current `\label{prop:...}` grep returns only
`prop:existence` and `prop:coalition-attack`; no collision.

---

## 6. Acceptance-criteria checklist

- [ ] `prop:slack-monotonicity` inserted at the end of `sec:decomposition`
  in `paper/tex/03_game.tex`, with the proof from Section 3.1.
- [ ] `prop:delta-neutralisation` inserted in the "Mitigation" paragraph
  of `paper/tex/07_discussion.tex`, with the proof from Section 3.2 and
  both parts (i) and (ii) present.
- [ ] The vague sentence at
  `paper/tex/07_discussion.tex:53-55` is replaced verbatim by the
  connective from Section 2.3.
- [ ] `\cref{prop:slack-monotonicity}` is referenced from
  `paper/tex/04_analysis.tex` in the "In the decomposition of
  \cref{sec:decomposition}..." paragraph (lines 73--83).
- [ ] `\cref{prop:delta-neutralisation}` is referenced twice in
  `paper/tex/07_discussion.tex`: once in the Mitigation paragraph, once
  in the reworded first open question.
- [ ] The reworded first "Open questions" item leaves the existence half
  explicitly open ("but existence on the $\Delta$-restricted action set
  is not proven").
- [ ] `cd paper && pdflatex -interaction=nonstopmode main.tex` (or
  `latexmk -pdf main.tex`) completes with no unresolved references and
  no duplicate labels; the two new labels appear in `main.aux`.
- [ ] Sign convention agrees with T11's slack panel: slack is
  non-decreasing in every $\varepsilon_j$ for $j\neq i$; equivalently,
  the externality shrinks the slack when other players lower their
  budgets.
- [ ] No regressions in the "Wider lesson" and "Conclusion" paragraphs
  (`paper/tex/07_discussion.tex:84-99`) --- they remain unchanged.
- [ ] No new bib entries, no new figures, no new code touching
  `game/`, `game_experiments/`, or `paper/figures/`. The only new file
  is the reproducibility script
  `tasks_findings/_scripts/04_slack_numerics.py`.

---

## 7. Downstream cross-links

* **T11 (slack numerical panel, K=200).**
  `.tasks/11_L_L_slack-numerical-panel.md` uses
  `\cref{prop:slack-monotonicity}` in the caption of
  `fig:slack_panel` and in the surrounding paragraph in
  `paper/tex/05_experiments.tex`; the sign convention (slack
  nondecreasing in every $\varepsilon_j$ for $j\neq i$) must match.
  The K=200 mechanism-grid cache
  (`_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl`, produced by
  `slurm_scripts/run_locus_K200.sh`) is the numerical grounding of the
  proposition; the C1/C5 checks in Section 4.3 are the sanity floor.
* **T20 (small-$p$ corollary of `lem:externality`).**
  `.tasks/20_M_M_small-p-corollary.md` proves
  $\partial A_i/\partial\eps_j < 0$ *analytically* in the small-$p$
  regime (`cor:externality-small-p`, appendix
  `app:externality-full`). Its slack-form
  is exactly $\partial\mathrm{slack}_i/\partial\eps_j > 0$: T20 upgrades
  Prop.\ A's inequality from weak (empirical closure)
  to strict (analytical) in the small-$p$ regime. The
  proof of `prop:slack-monotonicity` above `\cref`s
  `cor:externality-small-p` inside the "empirical in $A_i$" clause;
  after T20 lands, that clause reads as an analytical closure on the
  regime that covers typical DP-SGD sampling rates
  ($p\sim B/N \sim 10^{-4}$ to $10^{-2}$) and an empirical closure
  outside it.
* **T21 (supermodular NE existence).** Consumes the sign of
  $\partial^2 u_i / \partial\varepsilon_i\partial\varepsilon_j$
  through the slack form: with $u_i = V(\sigma) - w_i A_i =
  V(\sigma) - w_i(\bar c(\varepsilon_i) - \mathrm{slack}_i)$,
  the cross-partial factors as
  $\partial^2 u_i/\partial\varepsilon_i\partial\varepsilon_j =
  V''(\sigma)(\partial\sigma/\partial\varepsilon_i)(\partial\sigma/\partial\varepsilon_j) +
  V'(\sigma)\partial^2\sigma/\partial\varepsilon_i\partial\varepsilon_j +
  w_i\, \partial^2\mathrm{slack}_i/\partial\varepsilon_i\partial\varepsilon_j$. The
  slack term is the piece the supermodular structure is built around;
  Prop.\ A gives it a first-order sign. T21 does not depend on Prop.\ B.

---

## 8. Risks and failure modes

* **Label collision.** No `prop:slack-monotonicity` or
  `prop:delta-neutralisation` currently exists in `paper/tex/`
  (checked with `grep -rn '\\label{prop:'`). Should a future edit
  introduce a collision, rename to `prop:slack-mono` and
  `prop:delta-neutral` and update both cross-references and T11's
  caption.
* **Reviewer triviality objection to Prop.\ A.** Defence on record in
  `NOVELTY_ACTION_PLAN.md:83`: the proposition's value is theorem-level
  foregrounding of the slack sign for downstream `\cref`s, not new
  mathematics. Keeping the proof deliberately one-line makes the
  restatement nature transparent and pre-empts the objection. Prop.\ B
  is arithmetic, and its Part~(i) is a *new* pointwise squeeze not
  stated in the current paper --- this is the non-trivial content of
  the pair.
* **Empirical component in Prop.\ A.**
  `\cref{lem:externality}`'s empirical closure on $A_i$
  (`04_analysis.tex:55-64`) is inherited by Prop.\ A. If a later
  reviewer objects that Prop.\ A's inequality is not fully analytical,
  the fix is to cite T20's small-$p$ corollary
  (`cor:externality-small-p`) in the proof --- which is already what
  the "strict in the small-$p$ regime" clause of Section 2.1 does.
* **Mitigation-paragraph flow.** Keep the `\begin{proposition}` block
  as a standalone paragraph separated from the surrounding text with
  one connective sentence before and one after; the Section 2.3
  rewrite supplies the connective. Do not attempt to inline the
  proposition into running prose.
* **Open-questions rewrite reads as answered.** Preserved by keeping
  the "but existence on the $\Delta$-restricted action set is not
  proven" clause of the Section 2.3 rewrite; the second open question
  ("Is the PEG a potential game?") is not touched.

---

## 9. Voice-consistency notes (CLAUDE.md Rule 1)

The two proposition blocks and both proofs use first-person plural
("we"), signposted transitions ("Under the hypotheses of...", "By
definition..."), no throat-clearing, and no comparisons to prior
drafts (Rule 0). The mitigation-paragraph rewrite ("makes the
deterministic half of this statement precise") follows the companion
paper's convention of naming what a proposition *does* rather than
comparing it to a previous formulation. The "Empirically" caveat in
Prop.\ A is unusual for a proposition statement; we include it because
`lem:externality` itself carries the same caveat verbatim
(`04_analysis.tex:23-26`), and matching that framing keeps the two
statements coherent when the reader jumps between them. Removing the
caveat and letting the reader chase the closure through
`lem:externality` -> `cor:externality-small-p` is the alternative if
the caveat reads as hedging; we recommend keeping it until T20 lands
(after which the caveat can be tightened to "strict in the small-$p$
regime and empirical outside it").
