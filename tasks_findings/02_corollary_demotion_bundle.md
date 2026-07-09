# Task 02 findings -- Corollary/scaffold demotion bundle

**Task file:** `.tasks/02_M_L_corollary-demotion-bundle.md`
**Novelty priority:** M
**Effort:** L
**Category:** paper-rewrite (no code, no experiments)
**Deliverable form:** publication-ready LaTeX drop-ins in this document; the paper
files are *not* modified here.

---

## 1. Goal and narrative role

Three cheap, coordinated edits jointly defuse the three top-likelihood reviewer
objections identified in the novelty review, without altering the truth value of
any claim in the paper. Each is a *reframing* rather than a *retraction*:

- **Prop. 4 -> Corollary.** `NOVELTY_REVIEW.md:21` predicts a reviewer will call
  the coalition-attack result "Paper B in Stackelberg clothing"; `NOVELTY_REVIEW.md:57`
  ranks this as the highest-likelihood objection at any venue. `NOVELTY_ACTION_PLAN.md:128-130`
  (items N1.1 / N1.2) prescribes restyling the proposition as a `Corollary` of the
  externality lemma, plus a one-sentence admission that the result aligns the
  empirical attack of \cite{kaiser2026your} with the coalition's equilibrium best
  response and does not claim to add empirical content.
- **Prop. 2 -> paragraph.** `NOVELTY_REVIEW.md:25` and `NOVELTY_REVIEW.md:61` predict
  a reviewer will call the finite-game NE-existence proposition trivial (Nash 1950
  on a finite discretised game). `NOVELTY_ACTION_PLAN.md:187` (item N5.1) prescribes
  demoting it to a paragraph-level computational-scaffold statement, with the
  "existence on the continuous action space is not claimed" clause pulled up into
  the same paragraph.
- **C4 caption retitle.** `NOVELTY_REVIEW.md:22` predicts a reviewer will call
  Figure C4 a "relabelled replication" of Paper B's Section V.B.
  `NOVELTY_ACTION_PLAN.md:142` and `NOVELTY_ACTION_PLAN.md:150` (item N2, Path 1)
  prescribe a caption retitle that leads with an explicit acknowledgement of the
  companion mapping.

The paper's C1 headline (foregrounded separately by task 01) already carries the
load-bearing novelty; these three edits shave off the derivative-looking framing
that would otherwise draw reviewer fire. None of the eight NEs of C1, the $A_t$
inflation numbers of C4, or the externality-lemma proof are touched.

The corollary reframing is honest, not defensive: the coalition result *is* a
direct consequence of `\cref{lem:externality}` plus `\cref{def:coalition}` in the
limit $\lambda \to \infty$. Admitting the derivation is the intended posture
(`NOVELTY_ACTION_PLAN.md:128`).

Style-check anchors (Rule 1, CLAUDE.md): the drop-ins use first-person plural
sparingly, signposted transitions that name a mechanism ("\Cref{lem:externality}
applied to the coalition variant of \cref{def:coalition} pins..."), no
throat-clearing, and no comparisons to any prior draft (Rule 0: "overwrite,
don't append"). No sentence of the form "we now demote...", "previously a
proposition...", "an earlier draft..." appears; the corollary is presented as
if it were always a corollary. The corollary statement body is tightened for
precision (see Edit 1 diff summary) rather than left verbatim, because the
verbatim form under-specified what stays fixed under the monotonicity in $k$
and gave a strictly-necessary rather than a sufficient feasibility
condition — both fixable in one clean rewrite without any change to the
result's truth value or the C4 numbers.

---

## 2. Edit 1 -- Prop. 4 -> Corollary (`paper/tex/04_analysis.tex`)

### 2.1 Subsection heading and lead-in (currently lines 163-174)

**Current text (`paper/tex/04_analysis.tex:163-174`):**

```latex
\subsection{Coalition Stackelberg and the attack recovery}

\begin{proposition}[Coalition best response recovers the attack]
\label{prop:coalition-attack}
Fix a target $t$ and a coalition $C$ of size $k$, with $\lambda \to \infty$
in Definition~\ref{def:coalition} (the coalition cares only about the
target's vulnerability). The coalition's best response is $\eps_i =
\eps^{\star}_k$ for every $i \in C$, where $\eps^{\star}_k$ is the smallest
admissible budget that keeps the coalition's mechanism solution feasible
(i.e., $p_i > 0$ for every $i \in C$). Furthermore, the target's MIA
advantage $A_t$ is non-decreasing in $k$.
\end{proposition}
```

**Rewrite (drop-in for lines 163-174):**

```latex
\subsection{Coalition Stackelberg recovers the attack}

\Cref{lem:externality} applied to the coalition variant of
\cref{def:coalition} pins the malice-maximising coalition's Stackelberg
action down to a single point: the participation-preserving minimum
budget. We record this as a corollary.

\begin{corollary}[Coalition best response recovers the attack]
\label{cor:coalition-attack}
Fix a target $t \in N$, a coalition $C \subseteq N \setminus \{t\}$ of
size $k$, and any profile $\boldsymbol{\eps}_{-C} \in \prod_{j \notin C}
\Sigma_j$ for the non-coalition players. Let $\lambda \to \infty$ in
\cref{def:coalition}, so the coalition maximises $A_t$ alone. Then the
coalition's Stackelberg best response is the coalition-symmetric profile
$\eps_i = \eps^{\star}_k(\boldsymbol{\eps}_{-C})$ for every $i \in C$,
where $\eps^{\star}_k(\boldsymbol{\eps}_{-C})$ is the infimum of the
coalition-symmetric budgets at which the induced profile is feasible in
the sense of \cref{ass:participation}. Furthermore, holding
$\boldsymbol{\eps}_{-C}$ fixed, the induced target advantage
$A_t(\eps^{\star}_k, \boldsymbol{\eps}_{-C})$ is non-decreasing in $k$.
\end{corollary}
```

Diff summary:
- Subsection title: `Coalition Stackelberg and the attack recovery` ->
  `Coalition Stackelberg recovers the attack`. This mirrors the intro
  contribution bullet (`paper/tex/01_intro.tex:89`) rather than
  telegraphing the demotion in the heading. Rule 0 asks us to present the
  corollary as if it were always a corollary; naming the phenomenon,
  not the theorem-environment, is the companion paper's habit
  (`Monotonicity of the externality`, `Existence and structure of pure
  Nash equilibria`, `An aside on awareness` — none of them name a
  theorem-environment in the heading).
- Insert a **two-sentence** lead paragraph that names the two ingredients
  (`\cref{lem:externality}`, `\cref{def:coalition}`) and states the
  conclusion positively. The previous draft's second sentence ("We state
  this explicitly as a corollary; it is a derivative of the externality
  sign, not a headline theorem") was throat-clearing meta-narration in
  Rule-1 terms and negated its own subject — replaced by a single clean
  clause "We record this as a corollary", which does the same demotion
  work without commenting on it.
- `\begin{proposition}` -> `\begin{corollary}`,
  `\end{proposition}` -> `\end{corollary}`.
- `\label{prop:coalition-attack}` -> `\label{cor:coalition-attack}`.
- Statement body: **tightened for precision** in three places (was
  "unchanged verbatim" in the earlier draft; the tightening does not
  alter the truth value of any claim and is required for the corollary
  to correctly cover the C4 experimental setup, in which non-coalition
  players are held at a passive baseline $\eps = 32$ rather than at
  their Stackelberg follower best response):
  1. `Fix a target $t$ and a coalition $C$ of size $k$` ->
     `Fix a target $t \in N$, a coalition $C \subseteq N \setminus \{t\}$
     of size $k$, and any profile $\boldsymbol{\eps}_{-C}$ for the
     non-coalition players`. The set inclusion $C \subseteq N \setminus
     \{t\}$ is inherited verbatim from `\cref{def:coalition}` at
     `paper/tex/03_game.tex:138`, so no game-definitional drift; the
     explicit quantifier over $\boldsymbol{\eps}_{-C}$ makes the
     coalition's best response a function of the followers' profile
     rather than a scalar, which is what the proof actually derives.
  2. `keeps the coalition's mechanism solution feasible (i.e., $p_i > 0$
     for every $i \in C$)` -> `the induced profile is feasible in the
     sense of \cref{ass:participation}`. The previous parenthetical
     specified $p_i > 0$ only for coalition members, but
     \cref{ass:participation} (at `paper/tex/03_game.tex:124-132`)
     requires $p_i > 0$ for every player; referring to the assumption
     is both more general and eliminates the risk of a reviewer
     objecting that the coalition can drive a follower's $p$ to zero.
  3. `Furthermore, the target's MIA advantage $A_t$ is non-decreasing
     in $k$` -> `Furthermore, holding $\boldsymbol{\eps}_{-C}$ fixed,
     the induced target advantage $A_t(\eps^{\star}_k,
     \boldsymbol{\eps}_{-C})$ is non-decreasing in $k$`. The
     monotonicity in $k$ is measured with the followers' profile held
     constant (this is what the proof at lines 176-187 actually shows,
     and what C4 measures). Without this qualifier a reader could
     reasonably ask "monotone in $k$ under whose follower response?" —
     the tightened version is unambiguous.
- The proof block (lines 176-187) is compatible with all three
  tightenings without modification: (i) the "$\boldsymbol{\eps}_{-C}$
  fixed" reading is implicit in "By \cref{lem:externality}, $\partial
  A_t / \partial \eps_i \leq 0$ for every $i \in C$ on the feasible
  region"; (ii) the participation qualifier is exactly what the sketch
  describes ("at extremely low $\eps_i$ the mechanism may set $p_i = 0$
  (the colluder effectively drops out ...)"); (iii) the "adding a member
  to $C$" monotonicity argument implicitly fixes
  $\boldsymbol{\eps}_{-C}$ outside the enlargement.

### 2.2 Proof block (lines 176-187)

**Unchanged.** The proof block is:

```latex
\begin{proof}[Proof sketch]
By \cref{lem:externality}, $\partial A_t / \partial \eps_i \leq 0$ for every
$i \in C$ on the feasible region. Setting every $\eps_i$ as low as the
feasibility region admits therefore maximises $A_t$. The
participation-preservation qualifier is necessary because at extremely low
$\eps_i$ the mechanism may set $p_i = 0$ (the colluder effectively drops
out, removing its constraint from (C2) and so its externality on the
target); the coalition's optimum is thus the infimum of feasible
$\eps_i$. Monotonicity in $k$ is the same lemma applied to adding a member
to $C$: doing so strictly enlarges the set of budgets that pull $\sigma$
downward, raising $A_t$.
\end{proof}
```

This proof already derives the result from `\cref{lem:externality}`, so it is
consistent with the corollary framing without modification. It also is the
honest defence against a reviewer who objects that "corollary" undersells the
monotonicity-in-$k$ step (task file, failure-modes section): the monotonicity is
the same lemma re-applied on an enlarged coalition set.

### 2.3 Follow-up paragraph (lines 189-194)

**Current text (`paper/tex/04_analysis.tex:189-194`):**

```latex
Proposition~\ref{prop:coalition-attack} establishes that the budget-
manipulation attack of \citet{kaiser2026your} is precisely the
\emph{rational} best response of any coordinated subset of data holders,
regardless of malicious intent. Whether the coalition is adversarial is
irrelevant: any group of aware players whose joint utility weights their
target's harm heavily will pick the same action vector.
```

**Rewrite (drop-in for lines 189-194):**

```latex
\Cref{cor:coalition-attack} identifies the budget-manipulation attack
of \citet{kaiser2026your} as the coalition's equilibrium best response:
its contribution is the game-theoretic mapping, not an empirical
strengthening of that work. Whether the coalition is adversarial is
irrelevant, since any group of aware players whose joint utility
weights their target's harm heavily picks the same action vector as a
rational best response.
```

Diff summary:
- `Proposition~\ref{prop:coalition-attack} establishes that the
  budget-manipulation attack ... is precisely the \emph{rational} best
  response` -> `\Cref{cor:coalition-attack} identifies the
  budget-manipulation attack of \citet{kaiser2026your} as the
  coalition's equilibrium best response`. The previous draft's `aligns
  ... with` was passive and hedged; `identifies ... as` is the
  companion paper's active voice for a "this = that" claim (compare
  `paper/tex/04_analysis.tex:67`: `\Cref{lem:externality} formalises
  the empirical observation of \citet{kaiser2026your}` and
  `paper/tex/04_analysis.tex:116`: `\Cref{lem:externality} says
  lowering another player's budget...`). Also promotes to `\Cref`
  (explicit capitalising form) at sentence start. Technically
  redundant given `\usepackage[capitalise]{cleveref}` in
  `paper/main.tex:15` — the `capitalise` option makes `\cref` render
  "Corollary" regardless of position — but the companion voice
  convention in `paper/tex/04_analysis.tex` uses `\Cref` at
  sentence-initial positions (lines 67, 116, 129), so the drop-in
  matches that convention for consistency.
- Replaces the previous draft's `it does not claim to add empirical
  content over that work. Its role in the game-theoretic framing is
  that ...` with the tighter `its contribution is the game-theoretic
  mapping, not an empirical strengthening of that work. Whether ...`.
  Same admission (Task 02 deliverable N1.2), positive framing (Rule 1),
  and without the throat-clearing `Its role in the game-theoretic
  framing is that whether` construction.
- The substantive game-theoretic content (adversariality is irrelevant,
  aware players pick the same action) is preserved verbatim, only
  cleaned syntactically (`is that whether X is Y` -> `Whether X is Y,
  since ...`).

---

## 3. Edit 2 -- Prop. 2 -> paragraph scaffold (`paper/tex/04_analysis.tex`)

### 3.1 Existence block (currently lines 95-114)

**Current text (`paper/tex/04_analysis.tex:95-114`):**

```latex
\subsection{Existence and structure of pure Nash equilibria}

\begin{proposition}[NE existence on the discretised game]
\label{prop:existence}
The discretised PEG on the action grid $\mathcal{A}_K =
\{\eps^{\min},\dots,\eps^{\max}\}\subset[\eps^{\min},\eps^{\max}]$
with utilities~\eqref{eq:utility} is a finite normal-form game restricted to
the (closed) feasible set. By the Nash existence
theorem~\cite{nash1950equilibrium}, every such game admits at least one
mixed-strategy NE. Pure NE may or may not exist, depending on
$(w_i, |G_i|, V, K)$.
\end{proposition}

We do not claim existence on the continuous action space without further
quasi-concavity assumptions; in particular, our utility~\eqref{eq:utility}
is not generally concave in $\eps_i$ because $A_i$ is the trade-off
function of a subsampled Gaussian, whose dependence on $\eps_i$ through
$(\sigma, p_i)$ is non-trivial. We instead report pure NE found by exhaustive
search on a dense log-spaced grid and verify that they survive grid
refinement.
```

**Rewrite (drop-in for lines 95-114):**

```latex
\subsection{Existence and structure of pure Nash equilibria}

\paragraph{Scaffolding: existence on the discretised game.}
The discretised PEG on the log-spaced action grid $\mathcal{A}_K
\subset [\eps^{\min},\eps^{\max}]$ (\cref{sec:experiments}) with
utilities~\eqref{eq:utility} is a finite normal-form game restricted to
the (closed) feasible region of \cref{ass:participation}, so a
mixed-strategy NE exists by~\cite{nash1950equilibrium}. This grounds
the numerical search: the exhaustive-enumeration procedure of
\cref{sec:experiments} has at least one fixed point to find, and
restricting attention to $\mathcal{A}_K$ is warranted. Pure NE on
$\mathcal{A}_K$ need not exist and depend on $(w_i, |G_i|, V, K)$; we
enumerate them by brute force and verify stability under grid
refinement (\cref{sec:C6}). Existence on the continuous action space is
\emph{not} claimed---our utility~\eqref{eq:utility} is not generally
concave in $\eps_i$ because $A_i$ is the tight trade-off function of an
$I$-fold subsampled Gaussian, whose dependence on $\eps_i$ through
$(\sigma, p_i)$ has no closed form, so the standard quasi-concavity
route to pure-NE existence for continuous games does not apply.
```

Diff summary:
- Subsection title unchanged (`Existence and structure of pure Nash
  equilibria`).
- `\begin{proposition}[NE existence on the discretised game]
  \label{prop:existence} ... \end{proposition}` -> `\paragraph{Scaffolding:
  existence on the discretised game.}` paragraph. The heading
  `Scaffolding:...` leads with what the paragraph *is*, not with what
  it *isn't* — the earlier draft `(computational scaffold)` as a
  parenthetical inside the heading was noisier; the colon-led title
  ("Scaffolding: X") matches other companion-style headings such as
  `Minority and majority intuition.` at
  `paper/tex/04_analysis.tex:85`.
- The `\label{prop:existence}` is dropped. Grep against the current
  tree confirms only the label definition itself and stale
  `paper/main.aux` entries reference it; no `\cref`/`\ref` call site
  survives (see Section 5 grep output), so no downstream file needs
  updating and the stale `main.aux` entry is regenerated on the next
  compile.
- The Nash 1950 citation is kept as scaffolding. The earlier draft's
  `This is a computational scaffold rather than a novelty claim`
  self-narrates the demotion (Rule 1 throat-clearing) and Rule 0's
  ``overwrite, don't append'' principle: the final paper doesn't need
  to disclaim what it is not. Replaced by `This grounds the numerical
  search`, which is positive and states the role directly.
- The "existence on the continuous action space is not claimed" clause
  is promoted from the (now-removed) second paragraph into the same
  paragraph. Since neither Debreu (1952), Fan, nor Glicksberg is in
  `refs.bib`, and Task 02 explicitly requires no new bib entries, the
  earlier draft's `Fan--Glicksberg does not apply off the shelf` is
  softened to `the standard quasi-concavity route to pure-NE existence
  for continuous games does not apply`. This is technically identical
  (Debreu--Fan--Glicksberg is precisely the "quasi-concavity route")
  but does not commit the paper to an unattributed proper-name
  citation. If a reviewer asks, the honest one-line answer is
  "Debreu (1952)".
- `is not generally concave in $\eps_i$ because $A_i$ is the
  trade-off function of a subsampled Gaussian, whose dependence on
  $\eps_i$ through $(\sigma, p_i)$ is non-trivial` -> `is not
  generally concave in $\eps_i$ because $A_i$ is the tight trade-off
  function of an $I$-fold subsampled Gaussian, whose dependence on
  $\eps_i$ through $(\sigma, p_i)$ has no closed form`. Matches the
  vocabulary of `paper/tex/05_experiments.tex:27-28` ("tight
  analytical MIA advantage of the $I$-fold $(p_i, \sigma)$-subsampled
  Gaussian") and `paper/tex/04_analysis.tex:158` ("the dependence of
  $A_i$ on $(\sigma, p_i)$ has no closed form"). `non-trivial` was
  vague; `has no closed form` is the actual technical claim that
  forecloses the quasi-concavity check.
- The forward reference `(\cref{sec:C6})` is added to point pure-NE
  stability under grid refinement at the C6 experiment that actually
  verifies it. This is inside the paragraph, so it lands with the
  claim it justifies.
- The subsequent paragraphs (`\paragraph{What the externality lemma
  implies for NE structure.}` etc., currently starting at line 116)
  are unchanged and follow directly.

### 3.2 Downstream `prop:existence` references

`grep -rn "prop:existence" paper/tex/` returns only the label definition itself
(inside the block being deleted). No `\cref`/`\ref` calls to `prop:existence`
survive the edit, so no downstream file needs updating for this label.

---

## 4. Edit 3 -- C4 caption retitle (`paper/tex/05_experiments.tex`)

### 4.1 Caption block (currently lines 162-170)

**Current text (`paper/tex/05_experiments.tex:162-170`):**

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{figures/fig_C4_coalition.pdf}
\caption{\textbf{C4.} Target's realised MIA advantage as a function of
coalition size $k$, with the coalition playing its target-maximising
joint action. Annotations show the optimal coalition budget $\eps^\star$
at each $k$.}
\label{fig:C4}
\end{figure}
```

**Rewrite (drop-in for lines 162-170):**

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=\linewidth]{figures/fig_C4_coalition.pdf}
\caption{\textbf{C4. Recovery of the empirical attack of
\citet{kaiser2026your} as coalition Stackelberg best response.}
Target's realised MIA advantage as a function of coalition size $k$,
with the coalition playing its target-maximising joint action.
Annotations show the optimal coalition budget $\eps^\star$ at each $k$.}
\label{fig:C4}
\end{figure}
```

Diff summary:
- Bold caption lead: `\textbf{C4.}` -> `\textbf{C4. Recovery of the empirical attack of \citet{kaiser2026your} as coalition Stackelberg best response.}`.
- Body of the caption is unchanged.
- `\label{fig:C4}`, figure path, and figure environment options are all unchanged.

This makes the companion-paper mapping *explicit at the caption level*
(satisfying `NOVELTY_ACTION_PLAN.md:150` Path 1); a reviewer skimming the figure
will read the acknowledgement before the phenomenon.

---

## 5. Edit 4 -- Rename all `\ref{prop:coalition-attack}` -> `\ref{cor:coalition-attack}`

The task file lists three downstream references. `grep -rn "prop:coalition-attack" paper/tex/` at
the time of writing returns:

```
paper/tex/01_intro.tex:93:  (Proposition~\ref{prop:coalition-attack}). This exactly recovers the
paper/tex/07_discussion.tex:12:and \cref{prop:coalition-attack} shows this is the rational best response of
paper/tex/05_experiments.tex:149:\cref{prop:coalition-attack} relaxes as more colluders join because the
paper/tex/04_analysis.tex:166:\label{prop:coalition-attack}
paper/tex/04_analysis.tex:189:Proposition~\ref{prop:coalition-attack} establishes that the budget-
```

The two `paper/tex/04_analysis.tex` hits are handled by Edit 1 (label rename +
paragraph rewrite). The three remaining downstream hits need the following exact
substitutions:

### 5.1 `paper/tex/01_intro.tex:93`

**Current text (`paper/tex/01_intro.tex:89-96`), with the target sentence at line 93:**

```latex
\item \textbf{Coalition Stackelberg recovers the attack.} A coalition of
  data holders maximising the target's realised MIA advantage picks the
  participation-preserving minimum $\eps^{\star}$, and the target's $A_t$ is
  monotone-increasing in coalition size $k$
  (Proposition~\ref{prop:coalition-attack}). This exactly recovers the
  budget-manipulation attack of \citet{kaiser2026your} as the rational best
  response of any coordinated set of data holders. In our $n=5$ canonical
  setting the target's $A_t$ exceeds $0.7$ at $k=4$.
```

**Rewrite (drop-in, line 93 changed):**

```latex
\item \textbf{Coalition Stackelberg recovers the attack.} A coalition of
  data holders maximising the target's realised MIA advantage picks the
  participation-preserving minimum $\eps^{\star}$, and the target's $A_t$ is
  monotone-increasing in coalition size $k$
  (Corollary~\ref{cor:coalition-attack}). This exactly recovers the
  budget-manipulation attack of \citet{kaiser2026your} as the rational best
  response of any coordinated set of data holders. In our $n=5$ canonical
  setting the target's $A_t$ exceeds $0.7$ at $k=4$.
```

Diff summary: `(Proposition~\ref{prop:coalition-attack})` -> `(Corollary~\ref{cor:coalition-attack})`. The bullet's opening `\textbf{Coalition Stackelberg recovers the attack.}` stays as-is: it names the phenomenon, not the theorem-environment, so it survives the demotion without semantic loss.

### 5.2 `paper/tex/05_experiments.tex:149`

**Current text (surrounding lines 146-150):**

```latex
$k \in \{1, 2, 3\}$ the coalition picks $\eps_C \approx 6.8$, but at
$k = 4$ it can afford to drop to $\eps_C \approx 3.2$ while still
keeping every $p_i > 0$. The participation-preserving minimum of
\cref{prop:coalition-attack} relaxes as more colluders join because the
collective constraint slackens.
```

**Rewrite (drop-in for line 149):**

```latex
$k \in \{1, 2, 3\}$ the coalition picks $\eps_C \approx 6.8$, but at
$k = 4$ it can afford to drop to $\eps_C \approx 3.2$ while still
keeping every $p_i > 0$. The participation-preserving minimum of
\cref{cor:coalition-attack} relaxes as more colluders join because the
collective constraint slackens.
```

Diff summary: `\cref{prop:coalition-attack}` -> `\cref{cor:coalition-attack}`. `cleveref` will render the result as `Corollary N` automatically because the label is now attached to a `corollary` environment.

### 5.3 `paper/tex/07_discussion.tex:12`

**Current text (surrounding lines 8-14):**

```latex
higher realised MIA advantage and is compensated by a higher model-utility
benefit; the other absorbs the opposite trade. The budget-manipulation
attack of \citet{kaiser2026your} is precisely the asymmetric equilibrium
played by a coalition that weights the target's harm in its joint utility,
and \cref{prop:coalition-attack} shows this is the rational best response of
\emph{any} coordinated subset of data holders.
```

**Rewrite (drop-in for line 12):**

```latex
higher realised MIA advantage and is compensated by a higher model-utility
benefit; the other absorbs the opposite trade. The budget-manipulation
attack of \citet{kaiser2026your} is precisely the asymmetric equilibrium
played by a coalition that weights the target's harm in its joint utility,
and \cref{cor:coalition-attack} shows this is the rational best response of
\emph{any} coordinated subset of data holders.
```

Diff summary: `\cref{prop:coalition-attack}` -> `\cref{cor:coalition-attack}`.
No prose around this reference needs to change: the sentence says "shows this
is the rational best response", which is equally true whether the labelled
statement is a proposition or a corollary.

### 5.4 Global grep after edits

After Edits 1-4 land, both of these greps must return empty:

```bash
grep -rn "prop:coalition-attack" paper/tex/
grep -rn "\\\\ref{prop:existence}\|\\\\cref{prop:existence}" paper/tex/
```

(The second grep is scoped to `\ref`/`\cref` calls because the *label*
`prop:existence` also vanishes with the block being removed, but a stale hit in
`main.aux` from a previous build is harmless -- it will be regenerated on the
next `pdflatex` pass.)

---

## 6. Preamble and main-file check

- `paper/main.tex:22` already contains
  `\newtheorem{corollary}[theorem]{Corollary}`, sharing the counter with
  `theorem` (line 19), `lemma` (line 20), and `proposition` (line 21). No
  preamble change is needed and the corollary will pick up the next number in
  the shared sequence, which is the same behaviour the old proposition had. No
  hand-numbered references in the prose need to shift (all references use
  `\ref`/`\cref`).
- No new bib entries. `\citet{kaiser2026your}` used in the new caption already
  resolves via its many existing calls (`paper/tex/01_intro.tex:44`,
  `paper/tex/03_game.tex:20,74,103,112,152`, etc.).

---

## 7. Compile procedure and expected outcome

From `paper/`, remove the stale build artefacts first (they carry the
old `prop:existence` and `prop:coalition-attack` label entries) and
then compile:

```bash
rm -f paper/main.aux paper/main.bbl paper/main.blg paper/main.out
cd paper && pdflatex -interaction=nonstopmode main.tex \
  && bibtex main \
  && pdflatex -interaction=nonstopmode main.tex \
  && pdflatex -interaction=nonstopmode main.tex
```

Expected:

- No new warnings beyond the pre-existing `hyperref` PDF-string
  warnings on math-in-headings (CLAUDE.md Rule 4).
- No `undefined reference` warnings.
- No `multiply-defined labels` errors.
- The rendered PDF shows `Corollary N` at the coalition result, with
  `N` drawn from the shared `theorem`/`lemma`/`proposition`/
  `corollary` counter (`paper/main.tex:19-22`). The pre-edit counter
  values are `Lemma 7` (externality), `Proposition 8` (NE existence),
  `Proposition 9` (coalition attack); post-edit they become `Lemma 7`,
  `Corollary 8` (coalition attack). Prop. 8's slot is freed by Edit 2,
  so every subsequent theorem-counter user in
  `paper/tex/09_bayesian.tex` and `paper/tex/08_appendix.tex` may
  renumber by $-1$. `cleveref` auto-resolves the numeric values
  everywhere.
- Post-compile spot check via `pdftotext`:
  ```bash
  pdftotext paper/main.pdf - | grep -nE "Corollary [0-9]+|Scaffolding: existence|Recovery of the empirical"
  ```
  should return four hits: the corollary environment header, at least
  two `\cref{cor:coalition-attack}` cross-references (from
  `01_intro.tex:93`, `05_experiments.tex:149`, `07_discussion.tex:12`),
  the `Scaffolding: existence on the discretised game.` paragraph
  heading, and the C4 caption lead.
- The previously-numbered `Proposition N (NE existence...)` block is
  gone, replaced by the inline paragraph heading `Scaffolding:
  existence on the discretised game.`

---

## 8. Acceptance-criteria checklist

Mapped one-to-one from the `Success criteria` section of the task file,
extended with three criteria that the task file omits but that are
necessary for the demoted result to render and reference cleanly
(items marked *added*).

- [ ] **`pdflatex -interaction=nonstopmode main.tex` (or the sequence in
  Section 7) finishes without errors or undefined references.**
  Ensured by keeping every `\ref`/`\cref` target renamed in lock-step
  (Edit 1 renames the label; Edits 5.1-5.3 rename every downstream
  call). The theorem counter is shared across
  `theorem`/`lemma`/`proposition`/`corollary`, so numbering renumbers
  automatically.
- [ ] **`grep -rn "prop:coalition-attack" paper/` returns empty.** After
  Edits 1 (label rename) and 5.1-5.3 (three downstream call sites), no
  source file contains the string. The pre-edit grep in Section 5 lists
  exactly five hits; the drop-ins above address all five
  (`04_analysis.tex:166`, `04_analysis.tex:189`, `01_intro.tex:93`,
  `05_experiments.tex:149`, `07_discussion.tex:12`). The `paper/main.aux`
  hits are stale build artefacts and regenerate on the next compile;
  a `rm paper/main.aux paper/main.bbl paper/main.blg` before the first
  compile removes them cleanly.
- [ ] **`grep -rn "prop:existence" paper/tex/` returns empty.** Verified
  pre-edit via grep; the label is defined once and never referenced.
  Edit 2 removes the label along with the block.
- [ ] **Rendered PDF shows a `Corollary N` environment for the coalition
  result.** Guaranteed by the `\begin{corollary}` change and the
  pre-existing preamble `\newtheorem{corollary}[theorem]{Corollary}`
  (`paper/main.tex:22`). Verify by
  `pdftotext paper/main.pdf - | grep -n "Corollary [0-9]"` — the
  coalition-attack corollary should appear as the first `Corollary N`
  hit inside `\cref{sec:analysis}`.
- [ ] **Rendered PDF shows the paragraph-level "Scaffolding: existence
  on the discretised game." heading for the Nash-1950 statement.**
  Guaranteed by replacing the `\begin{proposition}` block with
  `\paragraph{...}` in Edit 2. Verify by
  `pdftotext paper/main.pdf - | grep -n "Scaffolding: existence"`.
- [ ] **C4 caption leads with "Recovery of the empirical attack of
  \citet{kaiser2026your} as coalition Stackelberg best response."**
  Guaranteed by the caption rewrite in Edit 3. Verify by
  `pdftotext paper/main.pdf - | grep -n "Recovery of the empirical"`.
- [ ] **Narrative coherence around the corollary.** The
  ``\Cref{cor:coalition-attack} identifies ... as the coalition's
  equilibrium best response'' rewrite (Edit 1.3) is a semantic drop-in
  for the previous ``Proposition ... establishes that the
  budget-manipulation attack ... is precisely the rational best
  response'' sentence: same antecedent (the citation), same downstream
  claim (``any group of aware players ... picks the same action
  vector''), positive framing (``identifies'' rather than ``aligns
  with'').
- [ ] **No numerical claim changes.** The eight NEs of C1, the $A_t$
  inflation numbers of C4 ($0.442, 0.483, 0.531, 0.607, 0.728$), the
  optimal coalition budgets ($\eps_C \approx 6.8$ at $k \in \{1,2,3\}$;
  $\eps_C \approx 3.2$ at $k=4$), and the externality-lemma proof are
  all untouched. No experiment needs re-running; no cache in
  `game_experiments/_cache_v2/` needs deleting; no figure in
  `paper/figures/` needs regenerating.
- [ ] **(*added*) Sentence-initial reference uses `\Cref` (companion-voice
  convention).** The rewrite of the follow-up paragraph (Edit 1.3)
  begins the sentence with `\Cref{cor:coalition-attack}` (capital-C).
  With `\usepackage[capitalise]{cleveref}` at `paper/main.tex:15`,
  `\cref` would render "Corollary" too, so this criterion is a style
  match to the sentence-initial `\Cref` calls at
  `paper/tex/04_analysis.tex:67, 116, 129` rather than a technical
  requirement. No other rewritten sentence in Edits 1-4 begins with a
  `\cref` call, so this is the only site to check. Verify by
  `grep -nE '(^|\. )\\cref' paper/tex/04_analysis.tex` — the result
  should not include the rewritten follow-up paragraph.
- [ ] **(*added*) Corollary label lookup via `\cref` renders "Corollary
  N" in the intro, experiments, and discussion.** `cleveref` reads the
  environment type from the `\label` placement, so
  `\cref{cor:coalition-attack}` will render as `Corollary N` (or
  `corollary~N` in mid-sentence, given `capitalise` is loaded and
  \cref{...} handles the initial capitalisation contextually). Verify
  by
  `pdftotext paper/main.pdf - | grep -n "Corollary [0-9]"` — three
  hits are expected (intro bullet, experiments C4 paragraph,
  discussion) beyond the corollary environment header itself.
- [ ] **(*added*) The intro contribution bullet's opening
  (`\textbf{Coalition Stackelberg recovers the attack.}` at
  `paper/tex/01_intro.tex:89`) stays semantically compatible with the
  corollary demotion.** The bullet's title names the phenomenon
  (``Coalition Stackelberg recovers the attack''), not the
  theorem-environment; its final sentence names an experimental
  fact (``In our $n=5$ canonical setting the target's $A_t$ exceeds
  $0.7$ at $k=4$'') sourced from C4, unchanged. Only the parenthetical
  `(Proposition~\ref{prop:coalition-attack})` at line 93 flips to
  `(Corollary~\ref{cor:coalition-attack})`. No re-titling of the
  bullet, no softening of the bullet's second sentence (``This
  exactly recovers the budget-manipulation attack of
  \citet{kaiser2026your} as the rational best response of any
  coordinated set of data holders.'') is needed: ``recovers ... as
  the rational best response'' is equally the corollary's content as
  it was the proposition's.

---

## 9. Rule 0 / Rule 1 style check

- **Rule 0 (overwrite, don't append).** The drop-ins do not narrate the
  demotion. No sentence of the form "previously a proposition", "we now
  demote", "in an earlier draft", or "the two-weight formulation
  collapses to" appears. The corollary is presented as if it were
  always a corollary; the paragraph scaffold is presented as if the
  existence question were always a scaffolding question. The subsection
  title `Coalition Stackelberg recovers the attack` names the
  phenomenon directly, exactly as the intro bullet
  (`paper/tex/01_intro.tex:89`) does, so the two references now
  agree — an earlier revision of these drop-ins had the heading
  `Coalition Stackelberg as a corollary of the externality lemma`,
  which telegraphed the theorem environment in the title (unlike every
  other subsection heading in `paper/tex/04_analysis.tex`) and was
  therefore replaced.
- **Rule 1 (companion-paper voice).** Verified against the four
  companion-voice anchors in `paper/tex/01_intro.tex`,
  `paper/tex/03_game.tex`, `paper/tex/04_analysis.tex`, and
  `paper/tex/05_experiments.tex`:
  * *First-person plural, used sparingly for our own moves*:
    `We record this as a corollary`, `we enumerate them by brute
    force`, `we verify stability under grid refinement`. Matches
    `paper/tex/04_analysis.tex:4` (`We characterise the equilibria
    ...`), `paper/tex/04_analysis.tex:112` (`We instead report pure
    NE...`).
  * *Signposted transitions that name a mechanism, not a genre*:
    `\Cref{lem:externality} applied to the coalition variant of
    \cref{def:coalition} pins the malice-maximising coalition's
    Stackelberg action down to a single point`. Matches
    `paper/tex/04_analysis.tex:67` (`\Cref{lem:externality}
    formalises the empirical observation of \citet{kaiser2026your}`).
    The earlier draft's opening (`As a direct consequence of
    \cref{lem:externality} and the coalition variant of
    \cref{def:coalition}, the malice-maximising coalition's
    Stackelberg best response is the participation-preserving
    minimum budget. We state this explicitly as a corollary; it is a
    derivative of the externality sign, not a headline theorem.`) had
    two problems: (a) the second sentence self-narrated the demotion
    ("We state this explicitly as a corollary"), and (b) the "not a
    headline theorem" clause was a comparative negation of a claim
    the paper never makes. Both are excised.
  * *Positive framing over negations*: `identifies ... as the
    coalition's equilibrium best response` (Edit 1.3) rather than
    the earlier draft's `aligns the empirical budget-manipulation
    attack ... with the coalition's equilibrium best response; it
    does not claim to add empirical content over that work`. The
    replacement carries the same admission (task deliverable N1.2)
    in a positive frame (`its contribution is the game-theoretic
    mapping, not an empirical strengthening of that work`).
  * *No throat-clearing*: the earlier draft's `This is a
    computational scaffold rather than a novelty claim` in the
    existence paragraph (self-negation) was replaced by `This
    grounds the numerical search` (positive frame). The earlier
    draft's `computational scaffold` inline parenthetical inside the
    `\paragraph{...}` heading was replaced by `Scaffolding:` prefix
    in the heading, matching the colon-headline pattern of
    `Minority and majority intuition.` at
    `paper/tex/04_analysis.tex:85`.
  * *Caption pattern*: the C4 caption retitle uses
    `\textbf{C4. Recovery of ... best response.}` matching the
    `\textbf{CN. Descriptor.}` shape of the surrounding C1, C2, C5,
    C6, C7, C8 captions.
- **Rule 6 (nothing extraneous).** No TODOs, no XXX markers, no author
  names, no acknowledgements. The `\author{Anonymous Submission}` line
  in `main.tex` is untouched.

---

## 10. Files that need changes (summary index)

| File | Lines pre-edit | Change |
|---|---|---|
| `paper/tex/04_analysis.tex` | 95-114 | Prop. 2 block -> `\paragraph{...}` scaffold (Edit 2) |
| `paper/tex/04_analysis.tex` | 163-174 | Subsection title + Prop. 4 -> Corollary lead-in and environment (Edit 1.1) |
| `paper/tex/04_analysis.tex` | 189-194 | "Proposition ... establishes" -> "Corollary ... aligns" (Edit 1.3) |
| `paper/tex/05_experiments.tex` | 162-170 | C4 caption retitle (Edit 3) |
| `paper/tex/05_experiments.tex` | 149 | `\cref{prop:coalition-attack}` -> `\cref{cor:coalition-attack}` (Edit 5.2) |
| `paper/tex/01_intro.tex` | 93 | `(Proposition~\ref{prop:coalition-attack})` -> `(Corollary~\ref{cor:coalition-attack})` (Edit 5.1) |
| `paper/tex/07_discussion.tex` | 12 | `\cref{prop:coalition-attack}` -> `\cref{cor:coalition-attack}` (Edit 5.3) |
| `paper/main.tex` | -- | No change (`corollary` env already at line 22) |

No code files, no bib entries, no figures, no experiment reruns are involved.
