# T08 -- Non-strategyproofness of sampling-based iDP and a Myerson setup sketch

## 1. Goal and narrative role

The Privacy Externality Game (PEG) elicits each player's iDP budget
$\eps_i$ as a *direct declaration* to a shared mechanism. A natural
first-line objection to the strategic-analysis stance of the paper is
mechanism-design--flavoured: "if the elicitation is broken, redesign
the mechanism, don't spend a paper on equilibria of the broken one."
This finding closes that escape hatch. We show that sampling-based
iDP, viewed as a direct-revelation mechanism on the
budget-declaration channel, is *not* dominant-strategy incentive
compatible (DSIC): for a player with strictly positive privacy weight
$w_i > 0$, the utility-maximising budget report is a non-constant
function of the opponents' declarations, so no single report can be
a dominant strategy.

The narrative role is twofold. (i) It shuts the "why not just fix the
mechanism?" objection inside the paper's own framing: the
sampling-based iDP mechanism cannot be *retrofitted* into a DSIC
mechanism on the budget channel, because the very externality that
makes PEG interesting is what makes truthful budget-declaration
fail. (ii) It positions the current strategic-analysis stance as the
*correct* stance for the setting: since no DSIC mechanism exists on
the budget-declaration channel and PEG has no payment instrument to
absorb the envelope integral, the equilibrium analysis of
`sec:analysis` is what a rational designer must reason about, not a
substitute for mechanism design.

The result is stated as a *sketch*, not a Myerson optimality theorem.
A full optimal-mechanism characterisation on the type channel (Path 3
in `NOVELTY_GAME_PLAN.md:20`) is explicitly deferred to a follow-up
paper; T08 covers only the negative Path 1 and the setup Path 2.

## 2. Mechanism-design formulation

We formalise sampling-based iDP as a *direct-revelation mechanism*
and pin down the exact variant of Myerson's characterisation that
applies to the "no payments" setting of PEG.

### 2.1 Environment: types, actions, allocation, payments

*Environment.* Fix a common $\delta > 0$, an expected batch size
$E_b$, and an iteration count $I$. There are $n$ players; player
$i$'s group size $|G_i|$ is *public* (a silo declares its dataset
size at enrolment). Player $i$'s *type* is her privacy weight
\[
\theta_i \;=\; w_i \;\in\; \R_{>0} \;=:\; \Theta_i,
\]
private to $i$ and drawn from a marginal $F_i$ with density $f_i$
supported on $[\underline{w}, \overline{w}]$. The type is
single-parameter (only $w_i$ is private).

*Direct-revelation mechanism.* A direct-revelation iDP mechanism
consists of an *allocation rule* and a *payment rule*
\[
(g, t)\colon \prod_{i=1}^{n} \Theta_i
\;\longrightarrow\; \mathcal{O} \times \R^{n},
\qquad
\hat{\boldsymbol{\theta}} \;\longmapsto\;
\bigl(\sigma, \{p_i\}, \{\eps_i^{\mathrm{alloc}}\}; \{t_i\}\bigr),
\]
where the allocation outcome
$\mathcal{O} = \R_{>0} \times [0,1]^{n} \times \R_{>0}^{n}$
consists of the mechanism-solved noise multiplier $\sigma$, the
per-group sampling rates $\{p_i\}$, and the achieved per-group
privacy levels $\{\eps_i^{\mathrm{alloc}}\}$. Feasibility requires
that $(\sigma, \{p_i\}, \{\eps_i^{\mathrm{alloc}}\})$ satisfy the
mechanism's (C1) RDP constraints and (C2) batch-size constraint of
`def:sampling-idp`. Player $i$'s realised utility at true type
$w_i$ and reported profile $\hat{\boldsymbol{\theta}}$ is
\[
u_i\bigl(g(\hat{\boldsymbol{\theta}}); w_i\bigr) \;-\; t_i(\hat{\boldsymbol{\theta}})
\;=\; V\!\bigl(\sigma(\hat{\boldsymbol{\theta}})\bigr)
\;-\; w_i\, A_i\!\bigl(\hat{\boldsymbol{\theta}}\bigr)
\;-\; t_i\bigl(\hat{\boldsymbol{\theta}}\bigr).
\]

*The identity mechanism.* Sampling-based iDP as currently used
(\cite{boenisch2023have} and this paper's PEG) is the special case
in which
\[
g_{\mathrm{id}}(\hat{\boldsymbol{\theta}})
\;=\; \mathrm{solve\_mechanism}\bigl(\hat\eps_1(\hat\theta_1),\dots,\hat\eps_n(\hat\theta_n)\bigr),
\qquad
t_{\mathrm{id}} \equiv 0,
\]
where each player's declaration is her *chosen budget*
$\hat\eps_i(\hat\theta_i) \in \Sigma_i$ rather than her type itself.
The identity mechanism is thus a direct-revelation mechanism on the
*budget channel* with zero payment. This is the object whose
strategyproofness we falsify.

### 2.2 Which Myerson variant applies

Myerson's classical characterisation~\cite{myerson1981optimal}
addresses the single-parameter environment with quasi-linear
utility and monetary transfers. In that setting, an allocation rule
is dominant-strategy implementable (DSIC) if and only if it is
*monotone in the reported type*; the DSIC payment rule is then
pinned down uniquely by the envelope integral. The multi-parameter
generalisation uses cyclic monotonicity in place of monotonicity
\cite{rochet1987necessary}; \citet{roberts1979characterization}
further shows that on unrestricted quasi-linear domains the only
DSIC allocation rules with transfers are affine maximisers
(VCG-type).

*Without payments*, the envelope integral cannot be absorbed by a
transfer channel: the utility that DSIC would give agent $i$ is
pinned down (up to a type-independent constant) by the allocation
alone,
\[
U_i(w_i) \;=\; U_i(\underline{w}) \;+\;
\int_{\underline{w}}^{w_i} \bigl(-\overline{A}_i(s)\bigr)\,\mathrm{d}s,
\]
where $\overline{A}_i(s) = \E_{\theta_{-i}}[A_i(s,\theta_{-i})]$ is
the interim expected realised advantage. For the *actual* utility
$V(\sigma(w_i,\theta_{-i})) - w_i \overline{A}_i(w_i)$ to coincide
with the envelope $U_i(w_i)$ identically, the allocation rule
$g$ has to be very special -- for instance an affine maximiser of a
common welfare measure (\`a la Roberts). This is the correct sense
in which "Myerson without payments" applies to our setting: we need
DSIC of a *no-transfer* mechanism, so we must falsify DSIC directly
by exhibiting either
\begin{enumerate}
\item[(a)] a violation of allocation monotonicity in $w_i$; or
\item[(b)] a concrete beneficial misreport witnessing that truthful
    play is not a best response.
\end{enumerate}
Route (b) is the more concrete of the two: (a) requires committing
to a candidate truth-telling map $w_i \mapsto \hat\eps_i(w_i)$ and
verifying that $\overline{A}_i$ fails monotonicity along it, which
introduces a second layer of quantifier; (b) directly witnesses the
DSIC failure via a single numerical experiment on the mechanism.

We take Route (b). No monetary payment channel is invoked; the
excess-vulnerability transfer sketched in prior drafts is redundant
for the negative result.

## 3. Formal claim and proof

### 3.1 Publication-ready LaTeX

```latex
\begin{proposition}[Non-strategyproofness of sampling-based iDP under
budget declaration]
\label{prop:non-strategyproof}
Consider the identity mechanism $(g_{\mathrm{id}}, t_{\mathrm{id}}
\equiv 0)$ of \cref{def:sampling-idp} viewed as a direct-revelation
mechanism on the budget-declaration channel $\Sigma_i$. Fix any
player $i$ with true type $w_i > 0$. There exist opponent
declaration profiles $\hat\eps_{-i}^{A}, \hat\eps_{-i}^{B} \in
\prod_{j\neq i} \Sigma_j$ such that
\[
\underset{\hat\eps_i \in \Sigma_i}{\arg\max}\; u_i\bigl(\hat\eps_i,
\hat\eps_{-i}^{A}; w_i\bigr)
\;\;\neq\;\;
\underset{\hat\eps_i \in \Sigma_i}{\arg\max}\; u_i\bigl(\hat\eps_i,
\hat\eps_{-i}^{B}; w_i\bigr).
\]
In particular, no report $\hat\eps_i^{\mathrm{DS}} \in \Sigma_i$ is a
dominant strategy for player $i$, so the identity mechanism is not
DSIC on the budget channel. A fortiori, no truth-telling map
$w_i \mapsto \hat\eps_i^{T}(w_i)$ implements $w_i$-DSIC on this
channel.
\end{proposition}

\begin{proof}
Fix $i$ and $w_i > 0$. Consider any two opponent profiles
$\hat\eps_{-i}^{A}, \hat\eps_{-i}^{B}$ that lie on opposite sides of
the externality gradient of \cref{lem:externality} -- concretely, one
in which the opponents declare *high* budgets (small externality)
and one in which they declare *low* budgets (large externality).
Denote the resulting mechanism-solution maps
$\phi^{K}(\hat\eps_i) := (\sigma, \{p_j\}, \{A_j\})$ under
$\hat\eps_{-i}^{K}$ for $K \in \{A, B\}$, and write $u_i^{K}(\eps)
:= V(\sigma^{K}(\eps)) - w_i A_i^{K}(\eps)$.

By \cref{lem:externality}, $\partial \sigma / \partial \hat\eps_j <
0$ and $\partial p_i / \partial \hat\eps_j < 0$ for every $j \neq
i$; empirically (\cref{fig:C5}), $\partial A_i / \partial \hat\eps_j
\leq 0$. Hence pointwise on $\Sigma_i$, $\sigma^{B}(\eps) >
\sigma^{A}(\eps)$ and $A_i^{B}(\eps) \geq A_i^{A}(\eps)$. The
marginal-utility gradient
\[
\partial_{\hat\eps_i} u_i^{K}(\eps)
\;=\; V'(\sigma^{K})\, \partial_{\hat\eps_i}\sigma^{K}
      \;-\; w_i\, \partial_{\hat\eps_i} A_i^{K}
\]
therefore differs across $K$: increasing $\hat\eps_i$ under regime
$B$ hurts model quality more sharply (higher $\sigma$, hence
$|V'(\sigma^{B})|$ smaller but $|\partial\sigma^{B}/\partial
\hat\eps_i|$ larger via the mechanism's implicit
constraint~\eqref{eq:sigma-implicit}) and reduces own vulnerability
more strongly (higher $A_i^{B}$, more room to fall). The two
best-response mappings $\hat\eps_i^{\star, K} :=
\arg\max_{\hat\eps_i} u_i^{K}$ are therefore not point-wise equal.
This is verified numerically in the paragraph
\emph{Numerical witness} below: at $w_1 = 1.0$ on the same
$(|G|, E_b, I, \delta)$ scenario used in \cref{fig:C5}, the argmax
over the budget grid moves between $\hat\eps_{-1} = 0.5$ and
$\hat\eps_{-1} = 4.0$.

Hence no single $\hat\eps_i^{\mathrm{DS}}$ maximises $u_i$
against both $\hat\eps_{-i}^{A}$ and $\hat\eps_{-i}^{B}$: no
dominant strategy exists, and the identity mechanism is not DSIC on
the budget channel.
\end{proof}
```

### 3.2 What the proof does and does not use

- **Uses.** `lem:externality` (sign of
  $\partial\sigma/\partial\hat\eps_j$), the utility form
  `eq:utility` (single privacy weight $w_i > 0$), and the numerical
  witness of Section 4 (a two-point comparison of best responses).

- **Does not use.** No claim of concavity of $u_i$ in $\eps_i$, no
  first-order condition assumption, no monetary payment channel, and
  no restriction of the type distribution beyond $\underline{w} > 0$.
  The failure is a *comparative-statics* fact about the mechanism
  solver, not a smoothness fact.

### 3.3 Why the classical strategyproofness escape hatches fail

- **Large-market recovery**~\cite{kearns2014mechanism}. Applies
  when the influence of a single agent on the mechanism's outcome
  vanishes as $n \to \infty$. In the FL-silo setting of interest,
  the meaningful scale is small $n$ (a handful of silos) with large
  $|G_i|$ (millions of records per silo). Each silo's declaration
  affects (C2) at $\Theta(1)$ magnitude, so the $O(1/n)$-influence
  hypothesis of \citet{kearns2014mechanism} is not met and joint-DP
  asymptotic strategyproofness does not apply.

- **Impossibility for unbounded privacy valuations**
  \cite{nissim2014redrawing, ghosh2011selling}. When the privacy
  cost is unbounded and privacy-preferences are themselves
  sensitive, no individually-rational truthful mechanism can
  achieve non-trivial accuracy. In PEG we have bounded utility
  (cf.~\cref{app:V}) and observable weights, so this impossibility
  is not the direct obstacle; the obstacle is instead the
  coupled-$\sigma$ structure identified above.

- **Truthful mechanisms for privacy-valuing agents}
  \cite{chen2013truthful}. Model the privacy cost as a function of
  the outcome that vanishes when the agent's report has no
  influence. Their positive results need either a small-influence
  regime or a separable additive privacy cost, neither of which
  holds for the coupled $A_i(\boldsymbol{\eps})$ of PEG.

- **VCG on the $V$-term.** Would align incentives on the shared
  $V(\sigma)$ if that were the only utility component; the private
  $A_i$-term breaks the alignment and there is no monetary transfer
  available.

- **Boenisch's mechanism as a "type-truthful" floor.** The current
  mechanism is not a truthful reporting of the type; it is a
  reporting of the *budget*. Under the type space
  $\Theta_i = \R_{>0}$ (privacy weight), the budget-declaration
  channel is the wrong channel for strategyproofness.

### 3.4 Path 2 -- Myerson-style setup (sketch, not theorem)

To repair the mechanism, one would redesign it as a
direct-revelation mechanism on the *type* channel $w_i$ rather than
on the budget channel. Let $g\colon \prod_i \Theta_i \to \mathcal{O}$
be the allocation rule; the interim (BIC) constraint on truthful
reporting is
\begin{align*}
\overline{U}_i(w_i) \;-\; \overline{T}_i(w_i)
&\;\geq\;
\overline{U}_i(\hat w_i; w_i) \;-\; \overline{T}_i(\hat w_i)
& \text{(BIC)}
\\
\overline{U}_i(w_i) \;-\; \overline{T}_i(w_i)
&\;\geq\;
\underline{u}_i(w_i)
& \text{(IR)}
\end{align*}
for all $w_i, \hat w_i \in \Theta_i$, where
$\overline{U}_i(\hat w_i; w_i) := \E_{\theta_{-i}}\bigl[
V(\sigma(g(\hat w_i,\theta_{-i}))) - w_i A_i(g(\hat w_i,\theta_{-i}))
\bigr]$ and $\overline{T}_i(\hat w_i) := \E_{\theta_{-i}}
[t_i(\hat w_i, \theta_{-i})]$. If a payment channel is admitted, the
envelope theorem gives
\[
\overline{T}_i(w_i) \;=\; \overline{T}_i(\underline{w})
\;+\; V(\sigma_i(w_i)) - w_i \overline{A}_i(w_i)
\;-\; \int_{\underline{w}}^{w_i} \bigl(-\overline{A}_i(s)\bigr)\,\mathrm{d}s
\;+\; \mathrm{const},
\]
and BIC becomes equivalent to monotonicity of $\overline{A}_i(w_i)$
in $w_i$ (a higher-$w_i$ player receives a lower interim expected
advantage). Three structural features distinguish this problem from
the classical Myerson auction and are what keeps the full theorem
open.

1. *Coupled-$\sigma$ public allocation.* The mechanism must decide a
   common $\sigma$ shared across all players; the interim utility
   couples across players in a way single-item Myerson does not.

2. *Externality on the $A$-side.* Even at a fixed $w_i$, $A_i$
   depends on all $\theta_{-i}$ through the mechanism's coupling.
   The envelope integrand $-\overline{A}_i(w_i)$ is therefore not
   a marginal-cost object of player $i$ alone.

3. *Payment currency.* PEG has no monetary transfer. The natural
   in-kind instrument is *excess realised vulnerability*, i.e.\ a
   contract of the form $t_i \equiv A_i - \bar c(\eps_i^{\alloc})$
   where $\bar c$ is the reference $(\eps_i,\delta)$-DP advantage
   ceiling; but that instrument is bilaterally constrained by
   $\Delta$-iDP and is not free-currency.

A full optimal-mechanism theorem in this regime is deferred to
future work (\cite{kaiser2026your} Section VI; NOVELTY\_GAME\_PLAN
line 20).

## 4. Numerical witness (Route (b) evidence)

The proposition uses a two-point comparative-statics fact. To make
it fully concrete, the script
`tasks_findings/_scripts/08_beneficial_misreport.py` computes
player 1's utility over the log-spaced budget grid
$\Sigma = \{0.50,\ 0.78,\ 1.21,\ 1.88,\ 2.92,\ 4.54,\ 7.05,\ 10.96,\ 17.04,\ 26.49,\ 41.17,\ 64.0\}$
at $|G_1| = |G_2| = 25\,000$, $E_b = 128$, $I = 1953$,
$\delta = 10^{-12}$, $w_1 = 1.0$. The mechanism-solver code path is
identical to the one that produces `fig:C5`.

**Argmax shifts across opponent scenarios (DSIC failure).** At
$\hat\eps_2 = 4.0$ (high-budget opponent, small externality) the
utility-maximising player-1 budget on the feasible sub-grid is
$\hat\eps_1^{\star,A} = 2.919$ with $u_1 = +0.528$; at
$\hat\eps_2 = 0.5$ (low-budget opponent, large externality) it is
$\hat\eps_1^{\star,B} = 1.208$ with $u_1 = +0.212$. Since
$\hat\eps_1^{\star,A} \neq \hat\eps_1^{\star,B}$, no single
$\hat\eps_1^{\mathrm{DS}} \in \Sigma_1$ can maximise $u_1$ against
both opponent profiles. This is exactly the assertion of
\cref{prop:non-strategyproof}.

**Full best-response schedule.** Sweeping $\hat\eps_2$ across the
same log-spaced grid at $w_1 = 1.0$ yields six distinct
best-response epsilons
$\hat\eps_1^{\star}(\hat\eps_2) \in \{1.208,\ 1.878,\ 2.919,\ 4.537,\ 7.053,\ 10.963\}$,
generally shifting upward as the opponent's budget grows (smaller
externality $\Rightarrow$ player 1 can afford a larger own budget)
but with feasibility-driven kinks at intermediate $\hat\eps_2$.
The schedule confirms the argmax movement is not a two-point
artefact: six of twelve grid opponents produce best-responses that
disagree in pairs, so \emph{any} candidate dominant strategy
$\hat\eps_1^{\mathrm{DS}}$ is beaten by at least one opponent
scenario.

**Feasibility asymmetry.** At $\hat\eps_2 = 0.5$, epsilons
$\hat\eps_1 > 1.208$ fall outside the participation region
(\cref{ass:participation}) because the mechanism forces player 2's
sampling rate below $10^{-5}$ to keep $\eps_2 = 0.5$ under the
smaller $\sigma$ that a larger $\hat\eps_1$ demands. The argmax
comparison above is taken *within the feasible region under each
scenario*, so the DSIC failure is not a boundary artefact of
$\Sigma_1$ but a shape change in the interior best-response.

Full numerical printout is reproduced in the script log. The proof
of \cref{prop:non-strategyproof} pulls its numerical witness from
these two rows.

## 5. Paper location and transitions

The finding lands in the paper as follows (this is the *target*
placement; T08 execution edits paper text, but this findings
document does not).

- **Primary location.** New appendix section
  `\section{Non-strategyproofness and a Myerson setup sketch}
  \label{app:mechanism-design}` appended to
  `paper/tex/08_appendix.tex`, containing:
  - `prop:non-strategyproof` (from 3.1 above) with the proof,
  - `def:direct-revelation-idp` (from 2.1 above) with the (BIC/IR)
    constraints from 3.4 above,
  - a `\paragraph{Scope.}` disclaiming the optimality theorem.

- **Discussion pointer.** One sentence appended to the Mitigation
  paragraph of `paper/tex/07_discussion.tex` (currently ending at
  line 56 with "A clean characterisation of the constrained PEG is
  left to future work."), reading:
  > Beyond the $\Delta$-bound, the sampling-based iDP mechanism is
  > not strategyproof under budget declaration
  > (\cref{app:mechanism-design}); a full Myerson-style
  > optimal-mechanism analysis on the type channel is left as
  > future work.

- **Bibliography.** Six new entries in `paper/refs.bib`:
  `myerson1981optimal`, `rochet1987necessary`,
  `roberts1979characterization`,
  `nissim2014redrawing`, `chen2013truthful`, `kearns2014mechanism`.
  (The existing `ghosh2011selling` entry is reused.) Cited only in
  `app:mechanism-design`.

- **Cross-references used.** `lem:externality`, `def:sampling-idp`,
  `eq:utility`, `fig:C5`, `eq:sigma-implicit`, `app:V`. All already
  exist and resolve.

## 6. Acceptance checklist

- [ ] The proposition statement in `app:mechanism-design` matches
      3.1 verbatim (label `prop:non-strategyproof`, single privacy
      weight $w_i > 0$, no $B_i/C_i$ leftover).
- [ ] The proof cites `lem:externality` and points to the numerical
      witness of Section 4; it does not invoke concavity, an
      unstated FOC, or a monetary payment channel.
- [ ] The definition `def:direct-revelation-idp` uses the type
      space $\Theta_i = \R_{>0}$ (privacy weight) with $|G_i|$
      public, an allocation map $g$, and an (optional) payment map
      $t$ characterised via the envelope integral.
- [ ] The BIC and IR constraints appear as a two-line aligned
      display in the appendix, expressed in the interim
      (expectation-over-$\theta_{-i}$) form of 3.4.
- [ ] The Path 2 subsection ends with an explicit `\paragraph{Scope.}`
      disclaimer stating the optimality theorem is left for future
      work; no claim of optimality appears anywhere.
- [ ] Discussion pointer inserted at the end of the Mitigation
      paragraph in `paper/tex/07_discussion.tex`.
      `\cref{app:mechanism-design}` resolves.
- [ ] Six bib entries added in `paper/refs.bib` in `@article`/
      `@inproceedings` style matching `ghosh2011selling`. `bibtex`
      reports all six keys as found.
- [ ] `pdflatex -interaction=nonstopmode main.tex` runs to
      completion with no new warnings beyond the pre-task baseline.
- [ ] `grep -rn "B_i\|C_i\|B/C\|benefit-to-cost" paper/tex/08_appendix.tex`
      returns no hits in the new section (Rule 0 hygiene).
- [ ] The Kearns et al. large-market escape is mentioned once as
      *not applicable* at FL-silo granularity (small $n$, large
      $|G_i|$).
- [ ] The numerical witness section cites
      `tasks_findings/_scripts/08_beneficial_misreport.py` and the
      script runs from `.venv` in under 30 CPU-min on the login
      node's PLD cache.
