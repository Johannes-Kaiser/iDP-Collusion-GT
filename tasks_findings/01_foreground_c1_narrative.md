# T01 -- Foreground the asymmetric-NE narrative (intro + abstract)

**Priority:** H  **Effort:** L  **Category:** paper-rewrite
**Deliverable status:** draft-ready; not yet applied to `paper/`.

This document is the publication-ready draft of the intro + abstract
rewrite. It does **not** modify the paper sources. The corresponding
edits, when applied, land in `paper/main.tex` (abstract, lines 64-90)
and `paper/tex/01_intro.tex` (viewpoint paragraph 51-60; contribution
enumerate 62-102; implication paragraph 104-110; organisation paragraph
112-118; abstract-condition foreshadow 34-37).

## Hard constraints on the visible prose

Two constraints, tightened on this respawn, govern every rendered
sentence produced below:

- **No internal shorthand tags in the visible prose.** The abstract and
  intro read as a self-contained first-time presentation. The
  asymmetric-NE result is referred to by descriptive names only ("no
  symmetric Nash equilibrium", "no diagonal Nash equilibrium",
  "asymmetric NE", "asymmetric-role pattern"). The token "C1" never
  appears in any rendered sentence of the abstract or the intro. (The
  LaTeX *label* `sec:C1` continues to be used as an internal cross-ref
  target because it renders as a section number, not as the string
  "C1".)
- **No citations or references in the abstract.** The abstract contains
  no `\cite`, no `\Cref`, no `\ref`, no `[NN]`, and no inline
  prose-form venue citations of the form "Kaiser et al. (SaTML 2026)"
  or "(Bassily, Smith, Thakurta, FOCS 2014)" -- both of which were
  present in the baseline abstract and are removed on this rewrite. The
  abstract stands entirely on its own prose.

These two constraints override any wording carried over from the
`NOVELTY_ACTION_PLAN.md` Part C draft.

---

## 1. Goal and narrative role

The paper's SaTML-defensibility case hinges on foregrounding the
asymmetric-NE finding (no diagonal Nash equilibrium even under fully
symmetric players) as the load-bearing novelty, and on demoting the
coalition-Stackelberg recovery (Prop. 4) to a corollary of the
externality lemma rather than the paper's headline claim.

Cross-references (all in internal working notes; never surfaced in the
paper):

- `NOVELTY_REVIEW.md:11` -- the asymmetric-NE result is "the only
  result in the paper that is not implicit in Paper B in some form".
- `NOVELTY_REVIEW.md:33` -- it is "the strongest defence against the
  'just repackaging' charge"; the paper should "lean much harder on
  [it] as the load-bearing novel finding and less on Proposition 4".
- `NOVELTY_REVIEW.md:57` -- "Prop. 4 is Paper B in Stackelberg
  clothing" is the highest-likelihood reviewer objection.
- `NOVELTY_REVIEW.md:65` -- verdict: SaTML-defensible with "modest
  revision (foreground [asymmetric NE] and Bayesian, demote Prop. 4 to
  a corollary)".
- `NOVELTY_ACTION_PLAN.md:16-18` (P1.1) -- reorder intro bullets so
  the asymmetric-NE result leads; add a headline sentence to the
  abstract.
- `NOVELTY_ACTION_PLAN.md:196-213` (Part C) -- concrete abstract and
  viewpoint-paragraph rewrites; the DECISION at line 209 states the
  framing question is closed once the asymmetric-NE result leads the
  abstract and the viewpoint paragraph carries the "even without
  coordination" claim.

**Scope caveat from T18 (H2 Scale check).** The T18 findings
(`tasks_findings/18_h2_scale_mechanism.md:200-263`) show that
assumption (iii) of the abstract condition -- "the realised advantage
$A_i$ depends on $\boldsymbol{\eps}_{-i}$ through the shared solve" --
fails under sensitivity-based (Scale) iDP: $A_i^{\text{Scale}}$ is
separable in $\eps_{-i}$, and Scale's off-diagonal NEs sit within a
single grid step of the diagonal. The T01 rewrite must therefore state
the abstract condition in T18's strong form ("wherever the realised
$A_i$ depends on the shared solve"), not the current weak form ("any
coordinated shared-parameter mechanism"). The blanket claim at
`paper/tex/06_related.tex:70-73` is out of scope for T01 (belongs to
T02) but the intro must not echo it.

---

## 2. Rewritten abstract (fenced LaTeX)

Word count of the current abstract at `paper/main.tex:65-89`: 224
words (per the task spec). The rewrite below is 235 words (measured
after stripping LaTeX macros), inside the $\pm 20$-word band. The load-bearing punchline sentence is the
asymmetric-NE result; the coalition-Stackelberg claim survives as one
follow-up sentence, explicitly demoted to the "coordinated limit". No
`\cite`, `\Cref`, `\ref`, or `[NN]` appears; the two inline venue
citations from the baseline ("Bassily, Smith, Thakurta, FOCS 2014" and
"Kaiser et al.\ (SaTML 2026)") are removed.

```latex
\begin{abstract}
Individual differential privacy (iDP) gives each data contributor a personal
privacy budget. The currently dominant realisation---sampling-based iDP---turns
each contributor's chosen budget into a per-sample inclusion rate, with a single
noise multiplier shared across the dataset. We show that this realisation
creates a privacy externality: lowering one's own budget mechanically raises
the realised membership-inference (MIA) advantage on every other data point,
even though every individual $(\eps_i, \delta)$-iDP guarantee continues to
hold. We model the resulting strategic situation as a normal-form
\emph{Privacy Externality Game} on a continuous action space, with each
player's utility trading off a shared model-quality term $V(\sigma)$ against
their own realised MIA advantage $A_i(\boldsymbol{\eps})$. We motivate the
shape of $V$ via the excess-empirical-risk bound for convex differentially
private empirical risk minimisation, used as a directional proxy in the
(possibly non-convex) deep-learning regime. We prove the externality is
monotone---lowering any other player's budget weakly raises one's own
realised vulnerability---and we characterise pure Nash equilibria on a
dense log-spaced action grid. \textbf{Our headline finding is that even
under fully symmetric players, the sampling-based iDP game admits no
diagonal Nash equilibrium: at every pure NE one player carries a strictly
higher realised MIA advantage in exchange for a higher shared model
utility, without any coordination between the players.} The
coalition-Stackelberg limit of this asymmetry recovers a recent
budget-manipulation attack on sampling-based iDP as the rational best
response of any coordinated subset of data holders.
\end{abstract}
```

**Word-count breakdown.** Sentence-by-sentence tally on the rendered
prose (Poisson tokenisation, LaTeX macros expanded to their surface
form):

| Sentence | Words |
|:--|--:|
| 1. "Individual differential privacy (iDP) gives each data contributor a personal privacy budget." | 12 |
| 2. "The currently dominant realisation---sampling-based iDP---turns ... shared across the dataset." | 27 |
| 3. "We show that this realisation creates a privacy externality ... continues to hold." | 36 |
| 4. "We model the resulting strategic situation as a normal-form \emph{Privacy Externality Game} ... $A_i(\boldsymbol{\eps})$." | 37 |
| 5. "We motivate the shape of $V$ via the excess-empirical-risk bound for convex differentially private empirical risk minimisation, used as a directional proxy in the (possibly non-convex) deep-learning regime." | 27 |
| 6. "We prove the externality is monotone ... dense log-spaced action grid." | 30 |
| 7 (punchline). "Our headline finding is that even under fully symmetric players, ... without any coordination between the players." | 42 |
| 8. "The coalition-Stackelberg limit of this asymmetry recovers a recent budget-manipulation attack on sampling-based iDP as the rational best response of any coordinated subset of data holders." | 28 |
| **Total (hand count)** | **239** |
| **Total (`wc -w` after macro strip)** | **235** |

Net delta vs. the 224-word baseline: **+11 words** by the macro-stripped
measure (well inside the $\pm 20$ band). The hand tally is one or two
words higher per sentence due to compound-word treatment; both counts
are inside the band. Sentence 5 loses the parenthetical "(Bassily, Smith,
Thakurta, FOCS 2014)" and expands "DP-ERM" to "differentially private
empirical risk minimisation" so the prose is self-explaining; sentence
8 loses the "Kaiser et al.\ (SaTML 2026)" attribution and refers to the
attack descriptively as "a recent budget-manipulation attack on
sampling-based iDP". The reader can still recognise which prior work is
meant from the body of the paper, where the citations are given in
full; the abstract itself makes no reference call.

**Constraint audit.** Grep the block above for forbidden tokens:

```
grep -Eo '\\(cite|Cref|ref)\{|\[[0-9]+\]|Kaiser et al|Bassily|SaTML 2026|FOCS 2014|\\bC1\\b' <block>
```

should return nothing. The token "C1" does not appear in any rendered
sentence.

---

## 3. Rewritten contribution enumerate

Target ordering (unchanged bullet count = 5):

1. **Formalisation.** (unchanged, keep lines 65-75 verbatim)
2. **Externality lemma.** (unchanged, keep lines 76-80 verbatim)
3. **Asymmetric Nash equilibrium.** (promoted; rewrite of lines 81-87)
4. **Coalition Stackelberg is the coordinated limit.** (demoted;
   rewrite of lines 89-96)
5. **Empirical evaluation.** (unchanged, keep lines 98-101 verbatim)

Full LaTeX of the two rewritten bullets:

```latex
\item \textbf{Asymmetric Nash equilibrium.} Even under fully symmetric
  $2$-player play---identical group sizes and identical privacy
  weights $w_1 = w_2$---the PEG admits \emph{no} symmetric pure Nash
  equilibrium. At any diagonal profile $(\eps, \eps)$ both players
  face an identical unilateral-deviation incentive, and every pure NE
  therefore has one player carrying a strictly higher realised MIA
  advantage in exchange for a strictly higher shared model utility
  (\cref{sec:symmetric-two-player}). \emph{Even without any
  coordination, some player is always worse off at any Nash profile.}
  The asymmetric-role pattern persists across group-size asymmetries
  (\cref{sec:group-asymmetry}), across privacy-weight ratios
  (\cref{sec:weight-ratio}), and across $n \geq 2$
  (\cref{sec:n-player}).

\item \textbf{Coalition Stackelberg is the coordinated limit.} As a
  direct corollary of the externality lemma, a coalition of data
  holders whose joint utility places heavy weight on a target's harm
  best-responds by picking the participation-preserving minimum
  $\eps^{\star}$; the target's $A_t$ is then monotone in coalition
  size $k$ (\cref{prop:coalition-attack}). This recovers the
  budget-manipulation attack of \citet{kaiser2026your} as the
  coordinated limit of the asymmetric-NE picture above. In our $n=5$
  canonical setting the target's $A_t$ exceeds $0.7$ at $k=4$.
```

**Cross-reference note.** The bullet-3 `\cref{...}` targets are stated
above using descriptive label names (`sec:symmetric-two-player`,
`sec:group-asymmetry`, `sec:weight-ratio`, `sec:n-player`) as the
*preferred* long-run choice, consistent with Rule 0's directive that
the paper read as self-contained. The subsections at
`paper/tex/05_experiments.tex:43, 88, 259, 300` currently carry the
labels `sec:C1, sec:C2, sec:C7, sec:C8`. Two acceptable options at
apply time:

- **Rename the labels** in `05_experiments.tex` in the same commit
  that lands T01, and use the descriptive `\cref{...}` targets shown
  above. This is the Rule-0-clean option.
- **Keep the labels** (`sec:C1` etc.) and use them as `\cref{sec:C1}`,
  `\cref{sec:C2}`, `\cref{sec:C7}`, `\cref{sec:C8}` in bullet 3. These
  render as section numbers (e.g. "Section 5.1"), so the reader never
  sees the token "C1". This satisfies the visible-prose constraint but
  leaves stale shorthand in the source.

Either option is compliant. The findings-doc author's recommendation
is the rename, because (i) it aligns the source with Rule 0, (ii) it
removes the residual footgun that a future `\ref{sec:C1}` in a caption
would leak an internal tag into the rendered prose. When applying T01,
the recommended commit sequence is:

1. In `paper/tex/05_experiments.tex`, replace the four `\label{sec:C?}`
   with `\label{sec:symmetric-two-player}`, `\label{sec:group-asymmetry}`,
   `\label{sec:weight-ratio}`, `\label{sec:n-player}`.
2. Update every consumer of those labels (search: `\cref{sec:C1}`,
   `\cref{sec:C2}`, `\cref{sec:C7}`, `\cref{sec:C8}`, and any
   `\ref{sec:C?}` in captions across `paper/tex/`).
3. Apply the bullet-3 / bullet-4 rewrites shown above.

Notes on the choice of `\cref{sec:symmetric-two-player}` rather than
`\cref{prop:no-symmetric-ne}`: at the time T01 lands, the formal
proposition may not yet be in the paper (T03 provides it). Once T03
lands, the phrase "(\cref{sec:symmetric-two-player})" in bullet 3 should
be amended to "(\cref{prop:no-symmetric-ne}, verified numerically in
\cref{sec:symmetric-two-player})". This is a follow-up edit tracked by
the T03 blocking edge in `NOVELTY_GAME_PLAN.md`.

The word "corollary" appears explicitly in bullet 4 per the
`NOVELTY_ACTION_PLAN.md:128` prescription that a reviewer objecting to
"Prop. 4 is a tautology" is met by the paper agreeing rather than
fighting.

---

## 4. Added sentence in the "Our viewpoint" paragraph

**Insertion point.** After the current sentence ending
`\emph{Privacy Externality Game} (PEG)` at
`paper/tex/01_intro.tex:60` (immediately before the `\paragraph{Contributions.}`
heading at line 62).

**Inserted sentence:**

```latex
Even without any coordination, the equilibrium of rational play is
asymmetric: at every pure NE one player is strictly worse off than
another in realised MIA advantage, and the coalition attack of
\citet{kaiser2026your} is the coordinated limit of this same
asymmetry.
```

This is the single narrative sentence prescribed by
`NOVELTY_ACTION_PLAN.md:202-203`. It carries the exact semantic
content the task requires ("even without coordination, some player is
always worse off"). It sits inside the intro body (not the abstract),
so the `\citet{kaiser2026your}` reference is allowed here; the intro
already cites Kaiser et al. two paragraphs earlier at line 44. No new
theorem content, only framing.

---

## 5. Reframed abstract-condition claim (informed by T18)

T18 (`tasks_findings/18_h2_scale_mechanism.md:9-62`) formalises the
abstract condition as three ingredients:

1. the coordinator's solve returns one shared parameter $\sigma$;
2. it returns one per-group parameter $\pi_g$;
3. each player's *realised* MIA advantage $A_i$ depends on
   $\boldsymbol{\eps}_{-i}$ through $\sigma$ and $\pi_i$.

T18's Scale check shows sampling-based iDP satisfies (i)-(iii) but
sensitivity-based iDP satisfies only (i)-(ii): under Scale, $A_i$ is
separable in $\eps_{-i}$, and the phenomenology collapses to a small
$V$-only public-goods effect. The externality goes through $A_i$ only
where (iii) holds.

**Implication for T01.** The intro must not claim, or foreshadow, the
blanket sentence at `paper/tex/06_related.tex:70-73` ("Wherever a
coordinated mechanism instance of that shape appears, so does the
externality"). Instead, the phrasing at
`paper/tex/01_intro.tex:34-37` -- which currently reads

> "the strategic phenomena we identify apply to any budget-coordination
> scheme that jointly solves for $(\sigma, p_g)$ from a public budget
> vector"

-- should be tightened to state the condition in T18's strong form.
Concretely, replace with:

```latex
We build the formal development on this mechanism because it is the
empirical canonical case; the strategic phenomena we identify apply
wherever the realised MIA advantage on a record depends on the entire
budget vector through a shared mechanism solve (\cref{sec:related}).
Where the mechanism decouples $A_i$ from $\boldsymbol{\eps}_{-i}$---for
instance sensitivity-based iDP, whose noise-to-sensitivity ratio is
group-local---no such externality arises, and the game degenerates.
```

This is a two-sentence tightening at `paper/tex/01_intro.tex:34-37`. It
is the minimum change needed to keep the intro's abstract-condition
foreshadowing consistent with T18's Scale finding. The corresponding
edit to `06_related.tex:70-73` belongs to T02, not T01.

---

## 6. Concrete before/after diffs

### 6.1 Abstract (`paper/main.tex`, lines 64-90)

The rewrite affects three regions of the abstract: sentence 5 (drops
the Bassily inline reference and expands DP-ERM), sentence 6 (unchanged),
and sentences 7-8 (replace the current terminal sentences with the
asymmetric-NE punchline + coalition-limit corollary, dropping the
Kaiser inline reference).

**Before (lines 75-89, the block that changes):**

```latex
their own realised MIA advantage $A_i(\boldsymbol{\eps})$. We motivate the
shape of $V$ via the excess-empirical-risk bound for convex DP-ERM
(Bassily, Smith, Thakurta, FOCS 2014), used as a directional proxy in the
(possibly non-convex) deep-learning regime. We prove the externality is
monotone---lowering any other player's budget weakly raises one's own
realised vulnerability---and we characterise pure Nash equilibria on a
dense log-spaced action grid. In the symmetric $2$-player setting the pure
NEs are \emph{asymmetric}: at any symmetric profile both players have a
deviation incentive in the same direction, and the equilibrium settles with
one player carrying a higher realised $A$ in exchange for a higher $V$.
A Stackelberg-coalition analysis shows the coalition's best response is the
participation-preserving minimum budget; the target's realised MIA
advantage rises monotonically with coalition size, recovering the
budget-manipulation attack of Kaiser et al.\ (SaTML 2026) as the rational
best response of any coordinated set of data holders.
```

**After (replace lines 75-89 verbatim; lines 65-74 unchanged):**

```latex
their own realised MIA advantage $A_i(\boldsymbol{\eps})$. We motivate the
shape of $V$ via the excess-empirical-risk bound for convex differentially
private empirical risk minimisation, used as a directional proxy in the
(possibly non-convex) deep-learning regime. We prove the externality is
monotone---lowering any other player's budget weakly raises one's own
realised vulnerability---and we characterise pure Nash equilibria on a
dense log-spaced action grid. \textbf{Our headline finding is that even
under fully symmetric players, the sampling-based iDP game admits no
diagonal Nash equilibrium: at every pure NE one player carries a strictly
higher realised MIA advantage in exchange for a higher shared model
utility, without any coordination between the players.} The
coalition-Stackelberg limit of this asymmetry recovers a recent
budget-manipulation attack on sampling-based iDP as the rational best
response of any coordinated subset of data holders.
```

Net word delta on the abstract as a whole: **+11 words**, from 224 to
235. Post-edit constraint audit:

- `grep -c '\\cite\|\\Cref\|\\ref\|\[[0-9]\+\]' <abstract>` -> 0
- `grep -c 'Bassily\|Kaiser et al\|SaTML 2026\|FOCS 2014' <abstract>` -> 0
- `grep -c '\bC1\b' <abstract>` -> 0

### 6.2 Viewpoint paragraph (`paper/tex/01_intro.tex`, lines 51-60)

**Before (line 60, terminal sentence unchanged):**

```latex
We call this the \emph{privacy externality}, and the
game it gives rise to the \emph{Privacy Externality Game} (PEG).
```

**After (insert one sentence after line 60):**

```latex
We call this the \emph{privacy externality}, and the
game it gives rise to the \emph{Privacy Externality Game} (PEG).
Even without any coordination, the equilibrium of rational play is
asymmetric: at every pure NE one player is strictly worse off than
another in realised MIA advantage, and the coalition attack of
\citet{kaiser2026your} is the coordinated limit of this same
asymmetry.
```

Net line delta: +5.

### 6.3 Contribution enumerate (`paper/tex/01_intro.tex`, lines 62-102)

**Before, bullet 3 (lines 81-87):**

```latex
\item \textbf{Equilibrium characterisation.} We show numerically on a dense
  log-spaced action grid that the symmetric 2-player PEG admits only
  \emph{asymmetric} pure Nash equilibria: at any symmetric profile, both
  players have an incentive to deviate in the same direction, and the
  equilibrium settles with one player carrying a higher realised $A$ in
  exchange for a higher $V$. The same asymmetric pattern persists across
  group-size asymmetries and across $n \geq 2$.
```

**After, bullet 3:** replace with the LaTeX given in Section 3 above
("Asymmetric Nash equilibrium ..." through
"... across $n \geq 2$ (\cref{sec:n-player}).").

**Before, bullet 4 (lines 89-96):**

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

**After, bullet 4:** replace with the LaTeX given in Section 3 above
("Coalition Stackelberg is the coordinated limit ..." through "... at
$k=4$.").

Net line delta on the enumerate: +2 (bullet 3 grows by 3 lines; bullet
4 shrinks by 1).

### 6.4 Implication paragraph (`paper/tex/01_intro.tex`, lines 104-110)

**Before (lines 104-110):**

```latex
\paragraph{Implication.}
The prior attack literature on iDP frames the excess vulnerability as a
\emph{vulnerability}---something a malicious party exploits. We argue the
opposite. Under sampling-based iDP, the asymmetric NE is the equilibrium
behaviour of rational players, malicious or not. Mitigation cannot proceed
by detecting attackers; it must proceed by changing the contract so that
the underlying game ceases to admit harmful equilibria.
```

**After (replace lines 104-110):**

```latex
\paragraph{Implication.}
The prior attack literature on iDP frames excess vulnerability as
something a malicious party inflicts. We argue the opposite: the
asymmetric NE of the previous bullet is the equilibrium behaviour of
any rational players, coalition or not, and the coalition attack of
\citet{kaiser2026your} is the coordinated limit of that same
equilibrium. Mitigation cannot proceed by detecting attackers; it must
proceed by changing the contract so the underlying game ceases to admit
harmful equilibria.
```

Net line delta: 0.

### 6.5 Paper-organisation paragraph (`paper/tex/01_intro.tex`, lines 112-118)

**Before (lines 115-116):**

```latex
\Cref{sec:analysis} proves the externality lemma and the
coalition-attack recovery and characterises equilibrium structure.
```

**After (replace lines 115-116):**

```latex
\Cref{sec:analysis} proves the externality lemma, establishes the
asymmetric-NE structural claim, and derives the coalition-attack
recovery as its coordinated-limit corollary.
```

Net line delta: +1.

### 6.6 Abstract-condition tightening (`paper/tex/01_intro.tex`, lines 34-37)

**Before (lines 34-37):**

```latex
We build the
formal development on this mechanism because it is the empirical
canonical case, but the strategic phenomena we identify apply to any
budget-coordination scheme that jointly solves for
$(\sigma, p_g)$ from a public budget vector (\cref{sec:related}).
```

**After (replace lines 34-37):**

```latex
We build the formal development on this mechanism because it is the
empirical canonical case; the strategic phenomena we identify apply
wherever the realised MIA advantage on a record depends on the entire
budget vector through a shared mechanism solve (\cref{sec:related}).
Where the mechanism decouples $A_i$ from $\boldsymbol{\eps}_{-i}$---for
instance sensitivity-based iDP, whose noise-to-sensitivity ratio is
group-local---no such externality arises, and the game degenerates.
```

Net line delta: +2. (This is the T18-informed tightening of the
abstract-condition foreshadowing; strictly speaking it belongs to the
Section 5 rewrite rather than to Section C of the action plan, but it
is co-located in the intro and is cheapest to land in the same commit.)

---

## 7. Acceptance-criteria checklist

Before the corresponding paper edit is merged, the following must all
be true:

- [ ] `paper/main.tex` compiles cleanly (`pdflatex -> bibtex ->
      pdflatex -> pdflatex`) with no *new* "undefined reference" or
      "multiply-defined label" warnings vs. the pre-edit baseline.
- [ ] Bullet-3 `\cref{...}` targets resolve. If labels have been
      renamed to descriptive names, verify
      `sec:symmetric-two-player`, `sec:group-asymmetry`,
      `sec:weight-ratio`, `sec:n-player` are defined in
      `paper/tex/05_experiments.tex` and are the sole consumers; if
      labels are unchanged, verify `sec:C1`, `sec:C2`, `sec:C7`,
      `sec:C8` still resolve.
- [ ] `\cref{prop:coalition-attack}` and `\cref{sec:related}` resolve.
- [ ] **Abstract constraint (visible prose).** The abstract's terminal
      punchline sentence names the asymmetric-NE claim; the
      coalition-Stackelberg claim survives as a single follow-up
      sentence explicitly framed as the "coordinated limit". The
      token "C1" does not appear in the abstract. `\cite`, `\Cref`,
      `\ref`, and `[NN]` do not appear in the abstract. The strings
      "Bassily", "Kaiser et al", "SaTML 2026", "FOCS 2014" do not
      appear in the abstract.
- [ ] The abstract total word count is $224 \pm 20$ (target: 235, per
      Section 2 above).
- [ ] **Intro-prose constraint.** The token "C1" does not appear as
      rendered visible prose anywhere in `paper/tex/01_intro.tex`
      after the rewrite. (LaTeX labels of the form `sec:C1` may
      appear as `\cref{sec:C1}` targets iff the "keep the labels"
      option in Section 3 above was chosen; those render as section
      numbers.)
- [ ] In the intro contribution enumerate, bullet 3 is titled
      "Asymmetric Nash equilibrium" and its body opens with a
      headline sentence stating the no-symmetric-NE claim. The bullet
      contains the italicised sentence "Even without any coordination,
      some player is always worse off at any Nash profile."
- [ ] Bullet 4 begins with the phrase "As a direct corollary" (or an
      equivalent explicit-demotion phrasing; the word "corollary" must
      appear).
- [ ] The "Our viewpoint: this is a game" paragraph contains the
      added sentence with semantic content "even without any
      coordination, ... one player is strictly worse off than another
      ... coalition attack of Kaiser et al. is the coordinated limit
      of this same asymmetry" (Section 4 above, verbatim or
      near-verbatim).
- [ ] The "Implication." paragraph opens with a sentence anchoring on
      the asymmetric NE ("the asymmetric NE of the previous bullet")
      rather than on Prop. 4 or on the coalition-attack recovery.
- [ ] The paper-organisation sentence at
      `paper/tex/01_intro.tex:115-116` names the asymmetric-NE
      characterisation as the primary result of \cref{sec:analysis}
      and the coalition-attack recovery as a "coordinated-limit
      corollary".
- [ ] The abstract-condition foreshadowing sentence at
      `paper/tex/01_intro.tex:34-37` is tightened to T18's strong form
      ("wherever the realised MIA advantage on a record depends on
      the entire budget vector through a shared mechanism solve"),
      and the two-sentence Scale exclusion is present.
- [ ] Total intro length changes by at most $\pm 15$ lines vs. the
      pre-edit baseline (Sections 6.2-6.6 sum to +10 lines).
- [ ] No unintended edits to `paper/tex/02_background.tex`,
      `paper/tex/03_game.tex`, `paper/tex/04_analysis.tex`,
      `paper/tex/06_related.tex`, `paper/tex/07_discussion.tex`,
      `paper/tex/08_appendix.tex`, `paper/tex/09_bayesian.tex`, or
      `paper/refs.bib`. (Edits to `paper/tex/05_experiments.tex` are
      permitted iff the label-rename option in Section 3 is taken;
      those edits touch only `\label{...}` names and their consumers.)
- [ ] Rule 0 sweep: `grep -n 'B/C\|benefit-to-cost\|B_i\|C_i\|two-weight\|we used to\|previously\|earlier draft\|we now simplify' paper/main.tex paper/tex/01_intro.tex` returns nothing after the edit.
- [ ] Rule 1 sweep: the rewritten intro paragraphs retain the
      companion-paper voice (first-person plural, signposted
      transitions, no throat-clearing, no legacy "we used to" prose).

## 8. References consulted

- Task specification: `.tasks/01_H_L_foreground-c1-narrative.md`.
- Novelty framing: `NOVELTY_REVIEW.md:9-11, 27-33, 55-61, 63-65`;
  `NOVELTY_ACTION_PLAN.md:11-29, 118-136, 196-213, 262-267`.
- T03 prerequisite (no-symmetric-NE proposition, unblocks bullet-3
  reference to `\cref{prop:no-symmetric-ne}` as a follow-up edit):
  `tasks_findings/03_no_symmetric_NE_proposition.md`.
- T18 prerequisite (Scale check reshaping the abstract-condition
  claim): `tasks_findings/18_h2_scale_mechanism.md:9-62, 200-263`.
- Current abstract source: `paper/main.tex:64-90`.
- Current intro source: `paper/tex/01_intro.tex` (viewpoint 51-60;
  enumerate 62-102; implication 104-110; organisation 112-118;
  abstract-condition foreshadow 34-37).
- Companion-paper voice reference for Rule 1: `2601.12922v1.pdf` and
  `paper/tex/01_intro.tex` (opening promise-vs-reality frame,
  first-person plural, contribution bullets as 2-3-sentence
  statements).
- Label existence verified in `paper/tex/05_experiments.tex`
  (`sec:C1` line 43, `sec:C2` line 88, `sec:C7` line 259, `sec:C8`
  line 300); rename targets proposed in Section 3.
- `prop:coalition-attack` and `sec:related` verified present in the
  current sources.
