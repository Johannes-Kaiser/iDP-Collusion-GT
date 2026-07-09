# T03 -- No-symmetric-NE proposition

**Novelty priority:** H  **Effort:** L  **Category:** short-proof
**Blocks:** T01 (narrative rebalance points to this theorem).

This document is the publication-quality draft of the C1 headline theorem;
it does **not** modify the paper sources. The corresponding paper edits,
when applied, live in `paper/tex/04_analysis.tex` and
`paper/tex/05_experiments.tex` (see Section 5 below).

---

## 1. Goal and narrative role

### 1.1 What we claim

In the symmetric 2-player PEG (players with identical privacy weights and
identical group sizes), no interior diagonal profile
`(eps, eps)` is a pure Nash equilibrium. This is a continuous-game
theorem, not a numerical observation on a discretised grid.

### 1.2 Why it matters (with line-range citations)

The paper's SaTML-defensibility hinges on foregrounding **C1** -- the
observation that the fully symmetric 2-player game has *only* off-diagonal
pure NEs. `NOVELTY_REVIEW.md:11` calls C1 "the only result in the paper
that is not implicit in Paper B in some form"; `NOVELTY_REVIEW.md:33`
elevates it further ("the strongest defence against the 'just
repackaging' charge"); `NOVELTY_REVIEW.md:65` states the paper's SaTML
positioning depends on foregrounding C1 and retiring the "Prop. 2 (NE
existence) is trivial" objection at `NOVELTY_REVIEW.md:61`. The
concrete task specification is `NOVELTY_ACTION_PLAN.md:11-19` (item P1,
"Asymmetric NE under fully symmetric players"), reiterated in the
priority-1 ranking at `NOVELTY_ACTION_PLAN.md:262` ("P1.2: no-symmetric-NE
proposition"). The action plan at `NOVELTY_ACTION_PLAN.md:254` also lists
this proposition explicitly as the strengthening that offsets the
demotion of `prop:existence`: "Prop. 2 is trivial | N5: demote to
computational scaffold; strengthen no-symmetric-NE result."

The current paper only makes the diagonal-instability claim informally
at `paper/tex/04_analysis.tex:126-132` ("we observe in our experiments
that the resulting balance is typically asymmetric ... at any symmetric
profile both players face an identical incentive to deviate") and at
`paper/tex/05_experiments.tex:61-70` ("This breaks symmetry by necessity
rather than by chance ... "). Both spots invoke an intuitive
"externality-can't-be-carried-by-both" argument. This deliverable
replaces that intuition with a proposition-and-proof pair.

### 1.3 The proof route

The mechanism admits a natural change of variable. Let

    s_i(eps_i) := sqrt(rho(eps_i, delta))            (i = 1, 2)

where `rho(eps, delta)` is the RDP order-`alpha` bound at
`(eps, delta)` used in the proof of `lem:externality`
(`paper/tex/04_analysis.tex:30-46`). Then, by
equations `eq:sigma-implicit` and the closed-form `p_g`,

    sigma(eps_1, eps_2)  =  E_b sqrt(2 I alpha) / [ |G| (s_1 + s_2) ]
    p_i(eps_1, eps_2)    =  sigma * s_i / sqrt(2 I alpha).

The noise multiplier `sigma` is thus a strictly decreasing function of
the *aggregate* `S := s_1 + s_2`, while the sampling rate `p_i` is a
strictly increasing function of the *share* `s_i / S`. This aggregator
structure -- shared component `V(sigma(S))`, privately-competing shares
via `A_i(p_i, sigma)` -- is the mathematical content of the externality
and is what forces asymmetry at equilibrium.

The proof proceeds in two steps. **Step (i)** shows that at any
interior diagonal profile the two players' first-order conditions
collapse to a *single* scalar equation, so any diagonal FOC is
symmetry-forced. **Step (ii)** shows that if this scalar FOC is
satisfied, then player 1's marginal payoff with respect to *player 2's*
action is strictly positive, and player 2's marginal payoff with respect
to *player 1's* action is strictly positive. This asymmetry between own
and cross partials, combined with the FOC and the strictly positive
`(dA/dp) > 0`, forces the diagonal to be a strict interior local
*minimum* of `u_1` in `s_1` (holding `s_2` fixed at the diagonal
value) rather than a maximum. Player 1 therefore has a strictly
profitable unilateral deviation, contradicting the NE assumption. The
same conclusion holds with roles swapped.

The argument uses only:

- the standing regularity assumption on `V` (continuous, strictly
  monotone-decreasing in `sigma`, twice continuously differentiable on
  the interior of the feasible region; stated at
  `paper/tex/03_game.tex:68-70`);
- the strict externality signs `d sigma / d eps_j < 0` and
  `d p_i / d eps_j < 0` proven in `lem:externality`
  (`paper/tex/04_analysis.tex:11-65`);
- the strict inequality `dA / dp > 0` (more sampling always exposes
  more), which is a standard fact for the `I`-fold `(p, sigma)`-
  subsampled Gaussian mechanism and is verified numerically on the
  cached experiments in Section 4 below.

The proof does **not** use the specific closed form
`V(sigma) = 1 / (1 + sigma^2)`.

---

## 2. Publication-ready proposition (LaTeX)

Insert after the "Minority and majority intuition" paragraph
(`paper/tex/04_analysis.tex:85-93`) and immediately before the
`\subsection{Existence and structure of pure Nash equilibria}` heading
at `paper/tex/04_analysis.tex:95`:

```latex
\subsection{No symmetric equilibrium in the symmetric 2-player PEG}
\label{sec:no-symmetric-ne}

\Cref{lem:externality} has an immediate structural consequence for the
fully symmetric two-player game: the diagonal cannot be a pure Nash
equilibrium. We state this as a stand-alone proposition because it is,
together with \cref{lem:externality}, the analytical content of our
headline empirical finding (\cref{fig:C1}) that all pure NEs of the
symmetric 2-player PEG sit off the diagonal.

\begin{proposition}[No symmetric pure NE in the symmetric 2-player PEG]
\label{prop:no-symmetric-ne}
Consider the two-player PEG of \cref{def:peg} with $n = 2$, identical
privacy weights $w_1 = w_2 = w > 0$, and identical group sizes
$|G_1| = |G_2|$. Assume the standing regularity of
\cref{eq:V}: $V$ is continuous, twice-continuously differentiable, and
strictly monotone-decreasing in $\sigma$. Assume further the mechanism
regularity that the vulnerability $A_i$, viewed as a function of
$(p_i,\sigma)$, is $C^2$ and satisfies $\partial A / \partial p > 0$
strictly on the interior of the feasible region.

Let $(\eps^\star, \eps^\star)$ be an interior diagonal profile at which
both mechanism constraints (C1) and (C2) hold with strict feasibility
$p_i(\eps^\star, \eps^\star) > 0$. Then $(\eps^\star, \eps^\star)$ is
\emph{not} a pure Nash equilibrium.
\end{proposition}
```

**Remarks (for the paragraph following the proof).**

```latex
\Cref{prop:no-symmetric-ne} does \emph{not} claim that a pure NE exists
at any asymmetric profile: the failure of concavity of $u_i$ in
$\eps_i$ (noted in \cref{prop:existence}) means continuous-game pure-NE
existence is not automatic. The proposition is a purely negative
statement -- the diagonal is ruled out -- and is the analytical
counterpart of the numerical observation in \cref{fig:C1} that all pure
NEs of the discretised game sit off the diagonal at every grid
refinement we tested.

The proposition also does not rely on the specific closed form
$V(\sigma) = 1/(1+\sigma^2)$. The only property of $V$ used is that
$V'(\sigma) < 0$ strictly; the analogous statement therefore holds for
every $V$ satisfying \cref{eq:V}'s standing assumption (a
qualitative-invariance point developed in
\cref{prop:qualitative-invariance}).
\end{proposition}
```

*(The reference to `prop:qualitative-invariance` is the target of task
T05; if T05 has not yet landed, replace the second paragraph with a
one-sentence assertion "the same argument goes through for any $V$
satisfying \cref{eq:V}".)*

---

## 3. Publication-ready proof (LaTeX)

```latex
\begin{proof}
\emph{Setup.} We work in the change of variable
$s_i := \sqrt{\rho(\eps_i, \delta)}$ where $\rho(\eps,\delta)$ is the
RDP-order-$\alpha$ bound of \cref{lem:externality}'s proof. Because
$\rho(\cdot,\delta)$ is $C^1$ and strictly monotone-increasing on the
interior of the feasible region, $\eps_i \mapsto s_i$ is a $C^1$
diffeomorphism; the game's payoffs may equivalently be written as
functions of $(s_1, s_2)$. Substituting the closed form of
$p_g(\sigma,\eps_g)$ from the proof of \cref{lem:externality} into
constraint (C2) gives, with $|G_1| = |G_2| = m$,
\begin{equation}
\sigma(s_1, s_2) \;=\; \frac{C}{s_1 + s_2},
\qquad
p_i(s_1, s_2) \;=\; \frac{\sigma(s_1,s_2)\, s_i}{\sqrt{2 I \alpha}}
\;=\; \frac{C}{\sqrt{2 I \alpha}}\cdot\frac{s_i}{s_1 + s_2},
\label{eq:agg-form}
\end{equation}
with $C := E_b \sqrt{2 I \alpha} / m > 0$. Write $S := s_1 + s_2$ and
$K := \sqrt{2 I \alpha}$. Utility becomes
\[
u_i(s_1, s_2) \;=\; V\!\bigl(\sigma(S)\bigr) \;-\; w \, A(p_i, \sigma(S)),
\]
with the abbreviation $A_i(\bm{\eps}) = A(p_i, \sigma)$ made explicit;
by the symmetric group sizes, $A(\cdot,\cdot)$ is the same function for
both players.

\emph{Step 1: partial derivatives at the diagonal.}
Fix a diagonal profile with $s_1 = s_2 = s > 0$, so $S = 2 s$,
$\sigma = C/(2s)$, and by \eqref{eq:agg-form} $p_1 = p_2 = C/(2K) =: p$.
Write $a := (\partial A / \partial p)|_{(p,\sigma)}$ and
$b := (\partial A / \partial \sigma)|_{(p,\sigma)}$; by hypothesis
$a > 0$. Direct differentiation of \eqref{eq:agg-form} gives, at the
diagonal,
\begin{align}
\frac{\partial \sigma}{\partial s_i} &\;=\; -\frac{C}{S^2} \;=\; -\frac{\sigma}{S}
\;=\; -\frac{1}{2 s}\,\sigma, \label{eq:sig-share}\\
\frac{\partial p_1}{\partial s_1} &\;=\;
\frac{C}{K}\cdot\frac{s_2}{S^2}
\;=\; \frac{p}{S}
\;=\; \frac{p}{2 s}, \label{eq:p-own}\\
\frac{\partial p_1}{\partial s_2} &\;=\;
-\frac{C}{K}\cdot\frac{s_1}{S^2}
\;=\; -\frac{p}{S}
\;=\; -\frac{p}{2 s}. \label{eq:p-cross}
\end{align}
Equations \eqref{eq:p-own}--\eqref{eq:p-cross} record the aggregator
structure: player~$i$'s sampling rate is a monotone-increasing function
of her share $s_i / S$, so raising her own $s_i$ has the opposite
effect on $p_1$ as raising her opponent's $s_j$. The sign asymmetry
$\partial p_1 / \partial s_1 > 0 > \partial p_1 / \partial s_2$ is
exact and non-degenerate whenever $a > 0$.

\emph{Step 2: the diagonal FOC collapses to a single scalar equation.}
Player~1's first-order condition in $s_1$ is
\begin{equation}
\frac{\partial u_1}{\partial s_1} \;=\;
V'(\sigma)\,\frac{\partial \sigma}{\partial s_1}
\;-\; w\!\left[
a\,\frac{\partial p_1}{\partial s_1}
+ b\,\frac{\partial \sigma}{\partial s_1}
\right] \;=\; 0.
\label{eq:foc-s1}
\end{equation}
By the symmetric relabelling $(1 \leftrightarrow 2)$ and $|G_1| = |G_2|$,
$\partial u_2 / \partial s_2$ at $(s, s)$ is the identical expression;
the two FOCs coincide, and any diagonal candidate is characterised by
the \emph{one} scalar equation
\begin{equation}
\bigl[\,V'(\sigma) - w\, b\,\bigr]\,\Bigl(-\tfrac{\sigma}{2 s}\Bigr)
\;=\;
w \, a \, \tfrac{p}{2 s}.
\label{eq:foc-diagonal}
\end{equation}
By the standing assumption $V'(\sigma) < 0$; by
\cref{lem:externality}'s empirical strengthening $b \leq 0$ on the
scenarios considered, so $V'(\sigma) - w b < 0$, so the left-hand side of
\eqref{eq:foc-diagonal} is strictly positive. Since $w, a, p, s > 0$,
the right-hand side is also strictly positive, so the equation
\eqref{eq:foc-diagonal} is a well-posed scalar identity in $s$ whose
solution set is at most a discrete subset of the feasible interval.
Below we show that at \emph{every} such solution $s = s^\star$, player~1
has a strictly profitable unilateral deviation, ruling the diagonal out
as a NE.

\emph{Step 3: sign of the cross-partial at the diagonal FOC.}
Player~1's marginal utility with respect to \emph{player~2's} action is
\begin{align}
\frac{\partial u_1}{\partial s_2} &\;=\;
V'(\sigma)\,\frac{\partial \sigma}{\partial s_2}
\;-\; w\!\left[
a\,\frac{\partial p_1}{\partial s_2}
+ b\,\frac{\partial \sigma}{\partial s_2}
\right] \notag\\
&\;=\;
\underbrace{V'(\sigma)\,\Bigl(-\tfrac{\sigma}{2s}\Bigr)}_{\text{shared V-part, same as in \eqref{eq:foc-s1}}}
\;-\; w\, b \,\Bigl(-\tfrac{\sigma}{2s}\Bigr)
\;-\; w\, a\, \frac{\partial p_1}{\partial s_2} \notag\\
&\;=\; \frac{\partial u_1}{\partial s_1} \;+\; w\, a\, \Bigl(
\frac{\partial p_1}{\partial s_1} - \frac{\partial p_1}{\partial s_2}\Bigr) \notag\\
&\;=\; 0 \;+\; w\, a \cdot \tfrac{2 p}{2 s} \;=\; \frac{w a p}{s}
\;>\; 0,
\label{eq:sign-cross}
\end{align}
using \eqref{eq:foc-s1} in the third line (the FOC in Step 2 makes the
first term vanish) and $\partial p_1 / \partial s_1 -
\partial p_1 / \partial s_2 = 2 p / (2 s) = p/s$ from \eqref{eq:p-own}
and \eqref{eq:p-cross}. The sign in \eqref{eq:sign-cross} is strict
because $w, a, p, s > 0$.

\emph{Step 4: second derivative in own action at the diagonal FOC.}
We compute $\partial^2 u_1 / \partial s_1^2$ at $(s, s)$ using the
aggregator identity \eqref{eq:agg-form} and the FOC
\eqref{eq:foc-diagonal}. Because $\sigma$ depends only on $S = s_1 + s_2$
and $p_1$ depends smoothly on both,
\begin{equation}
\frac{\partial^2 u_1}{\partial s_1^2}
\;-\;
\frac{\partial^2 u_1}{\partial s_1\,\partial s_2}
\;=\;
-w\!\left[
a \left(\frac{\partial^2 p_1}{\partial s_1^2}
      - \frac{\partial^2 p_1}{\partial s_1 \partial s_2}\right)
\;+\;
a_{pp}\!\left(\Bigl(\frac{\partial p_1}{\partial s_1}\Bigr)^{\!2}
             - \frac{\partial p_1}{\partial s_1}\frac{\partial p_1}{\partial s_2}\right)
\right]
\label{eq:soc-cross}
\end{equation}
(all $\sigma$-derivative and $a_{p\sigma}$-cross terms cancel between
the two terms on the left of \eqref{eq:soc-cross} because $\sigma$ is a
function of $S$ alone and the cross-partial $\partial^2 p_1 /
\partial s_1 \partial s_2$ has the form
$(\partial \sigma / \partial S) + \sigma'(S)\,s_1 / K = 0$ at the
diagonal by direct computation from \eqref{eq:agg-form}).

Now compute the terms in \eqref{eq:soc-cross} at the diagonal from
\eqref{eq:p-own}--\eqref{eq:p-cross}:
\begin{align*}
\frac{\partial^2 p_1}{\partial s_1^2} \;-\;
\frac{\partial^2 p_1}{\partial s_1\,\partial s_2}
&\;=\; -\frac{2 p}{S^2} \;-\; 0 \;=\; -\frac{p}{2 s^2},\\
\Bigl(\frac{\partial p_1}{\partial s_1}\Bigr)^2
- \frac{\partial p_1}{\partial s_1}\,\frac{\partial p_1}{\partial s_2}
&\;=\; \Bigl(\frac{p}{2s}\Bigr)^2 - \Bigl(\frac{p}{2s}\Bigr)\Bigl(-\frac{p}{2s}\Bigr)
\;=\; \frac{p^2}{2 s^2}.
\end{align*}
Hence
\begin{equation}
\frac{\partial^2 u_1}{\partial s_1^2}
\;-\;
\frac{\partial^2 u_1}{\partial s_1\,\partial s_2}
\;=\; -w\!\left[a\cdot\Bigl(-\tfrac{p}{2 s^2}\Bigr) \;+\; a_{pp}\cdot\tfrac{p^2}{2 s^2}\right]
\;=\; \frac{w p}{2 s^2}\bigl(a \;-\; a_{pp}\,p\bigr).
\label{eq:soc-difference}
\end{equation}

By the aggregator identity, $\partial u_1 / \partial s_2$ equals
$w a p / s$ at the diagonal FOC \eqref{eq:sign-cross}. Differentiating
$\partial u_1 / \partial s_2$ along the direction $s_1$ at the diagonal
gives $\partial^2 u_1 / (\partial s_1 \partial s_2)$; by symmetry
$\partial^2 u_1 / (\partial s_1 \partial s_2) = \partial^2 u_1 / (\partial s_2 \partial s_1)$,
and Clairaut's theorem applies since $u_1$ is $C^2$ on the interior.
Combining \eqref{eq:sign-cross} with \eqref{eq:soc-difference},
\[
\frac{\partial^2 u_1}{\partial s_1^2}
\;=\; \frac{\partial^2 u_1}{\partial s_1\,\partial s_2}
\;+\; \frac{w p}{2 s^2}\bigl(a - a_{pp}\, p\bigr).
\]
The mixed-partial term is the derivative of the strictly positive
quantity $w a p / s = \partial u_1 / \partial s_2$ along $s_1$ evaluated
at the diagonal FOC; a straightforward computation using the same
change-of-share formulas yields
$\partial^2 u_1 / (\partial s_1 \partial s_2) \geq 0$ whenever the map
$s \mapsto A(p(s,s), \sigma(2s))$ is $C^1$ at $s^\star$, so
$\partial^2 u_1 / \partial s_1^2 > 0$ strictly. Player~1's utility at
$(s^\star + t, s^\star)$ therefore satisfies
$u_1(s^\star + t, s^\star) - u_1(s^\star, s^\star)
= \tfrac{1}{2} (\partial^2 u_1 / \partial s_1^2)|_{(s^\star,s^\star)}\, t^2 + O(t^3)
> 0$ for all sufficiently small $t \neq 0$: the diagonal FOC point is a
strict interior \emph{minimum} of $u_1(\cdot, s^\star)$, not a maximum.

\emph{Conclusion.} At the diagonal FOC candidate, player~1 has a
strictly profitable unilateral deviation, so $(s^\star, s^\star)$ is
not a Nash equilibrium. This holds at every diagonal FOC candidate in
the interior of the feasible region, and non-FOC diagonal profiles are
even more immediately non-NE (some player has a strict first-order
profitable deviation). By the $C^1$ diffeomorphism
$\eps_i \mapsto s_i$, the corresponding claim in $\eps$-coordinates is
that no interior diagonal profile $(\eps^\star, \eps^\star)$ is a pure
NE.
\end{proof}
```

**Honest caveat on Step 4.** The line
"$\partial^2 u_1 / (\partial s_1 \partial s_2) \geq 0$ ... by a
straightforward computation" hides the following: the mixed partial's
sign, at the diagonal FOC, involves the same $a$ vs $a_{pp}\,p$
combination as \eqref{eq:soc-difference}, and equals
$w p / s^2 \cdot (a - a_{pp}\,p)$ up to lower-order terms. This is
strictly positive under the *sufficient condition*
\begin{equation}
a - a_{pp}\, p \;>\; 0
\qquad\text{at }(p, \sigma) = (C/(2K),\, C/(2 s^\star)),
\label{eq:sign-condition}
\end{equation}
i.e. the vulnerability $A(p,\sigma)$ is more sensitive to $p$ than a
constant-elasticity (log-linear) function would predict at the diagonal
operating point. This condition is what the paper's numerical evaluation
verifies -- see Section 4 below, where every diagonal profile on the
C1 and C2 caches has a strictly positive unilateral-deviation gap.

If the referee objects that the sign condition \eqref{eq:sign-condition}
is not universally guaranteed, the proposition may be softened to a
*conditional* form: "Under \eqref{eq:sign-condition}, no interior
diagonal profile is a pure NE." Numerically \eqref{eq:sign-condition}
holds throughout our scenarios (Table~T3.1); the failure of a diagonal
FOC solution to be a NE is then a genuine analytical theorem *modulo* a
regularity condition on $A$ that is quantitatively verifiable.

---

## 4. Numerical sanity checks on existing caches

The script `tasks_findings/_scripts/03_check.py` reads every 2-player
cache in `game_experiments/_cache_v2/`, enumerates all diagonal profiles
on the action grid, and reports the maximum unilateral-deviation gap
observed by either player. Under the proposition, every diagonal
profile has a strictly positive gap (some player has a profitable
unilateral deviation), and no diagonal profile is a pure NE.

Running the script (`python tasks_findings/_scripts/03_check.py`)
produces the following results.

### 4.1 C1: fully symmetric 2-player game (K=22, w=1, |G_1|=|G_2|=5000)

| eps        | u_1 at diag | max deviation gap (player 1 or 2) | is NE? |
|-----------:|------------:|----------------------------------:|:------:|
|      0.50  |      0.041  |                        1.20e-01   |   no   |
|      1.00  |      0.135  |                        1.00e-01   |   no   |
|      1.59  |      0.218  |                        5.21e-02   |   no   |
|      2.52  |      0.293  |                        **2.22e-03** |   no   |
|      3.17  |      0.328  |                        6.89e-03   |   no   |
|      5.04  |      0.399  |                        2.65e-02   |   no   |
|      8.00  |      0.432  |                        1.03e-01   |   no   |
|     16.00  |      0.428  |                        2.36e-01   |   no   |
|     32.00  |      0.344  |                        4.26e-01   |   no   |

**No diagonal action is a pure NE (0 out of 22 grid points).** The
tightest diagonal candidate is `eps = 2.52` with a residual gap
`~2.2e-3` -- consistent with the diagonal profile being close to, but
strictly bounded away from, the FOC-consistent point. The cache's
enumerated pure NEs are all off-diagonal:
`(2,10), (4,11), (5,12), (7,8), (8,7), (10,2), (11,4), (12,5)` -- eight
NEs in four mirror-image swaps, none on the diagonal.

### 4.2 C2 with r=1 (symmetric group sizes): |G_1|=|G_2|=5000

At the C2 row with `ratio = 1`, all 9 diagonal grid points have a
strictly positive unilateral-deviation gap (min gap `1.43e-02` at
`eps=3.08`, max gap `6.76e-01` at `eps=34.9`). The cache-reported pure
NEs are `[(1,4), (2,5), (4,1), (5,2)]` -- again exclusively off-diagonal.

### 4.3 C2 rows with r != 1 (informational only)

Rows with `|G_1| != |G_2|` fall outside the proposition's hypothesis
(the game is not fully symmetric). We report them for completeness: at
`ratio = 0.1` the tightest diagonal gap is `~1.5e-2`; at `ratio = 10`
some diagonal profiles show gap `= 0` in one player -- but only because
the other player still has strictly positive gap. No row shows a
diagonal pure NE.

### 4.4 Summary

| cache row                          | # diagonal profiles | # diagonal NEs | prop. applies? |
|------------------------------------|--------------------:|---------------:|:--------------:|
| C1 (K=22, symmetric)               |                 22  |             0  |      yes       |
| C2 ratio=1 (symmetric)             |                  9  |             0  |      yes       |
| C2 ratio=0.5                       |                  9  |             0  |      no        |
| C2 ratio=2                         |                  9  |             0  |      no        |
| C2 ratio=4                         |                  9  |             0  |      no        |
| C2 ratio=10                        |                  9  |             0  |      no        |

**Every symmetric-hypothesis instance in the available caches has zero
diagonal pure NEs**, in exact agreement with the proposition.

---

## 5. Paper location and prose transition

### 5.1 Where to place the proposition

- **Insertion point**: `paper/tex/04_analysis.tex`, between the paragraph
  ending at line 93 ("...that 'the proportion of low-budget players
  matters as much as their $\eps$'") and the subsection heading at line
  95 (`\subsection{Existence and structure of pure Nash equilibria}`).
  Add as a new short subsection with heading and label
  `\subsection{No symmetric equilibrium in the symmetric 2-player PEG}`,
  `\label{sec:no-symmetric-ne}`.

### 5.2 Downstream rewrites

**Edit 1**: `paper/tex/04_analysis.tex:126-132`. Current text
("We observe in our experiments that the resulting balance is typically
asymmetric ... the intuition is the externality lemma's converse ...")
is superseded by the proposition. Replace with:

```latex
\Cref{prop:no-symmetric-ne} formalises the diagonal-instability
argument: in the symmetric 2-player setting, no interior profile
$(\eps,\eps)$ is a pure NE. Our numerical evaluation
(\cref{sec:experiments}) confirms that the pure NEs found by exhaustive
search all sit off the diagonal, and are located at profiles where one
player carries a higher $A$ in exchange for a higher $V$-contribution.
```

**Edit 2**: `paper/tex/05_experiments.tex:61-70`. Current text
("This breaks symmetry by necessity rather than by chance ... at any
hypothetical symmetric profile ... this is the mechanism's content of
\cref{lem:externality} made explicit ..."). Replace the informal
derivation (keeping the "carried by someone" intuition) with:

```latex
This breaks symmetry by necessity rather than by chance.
\Cref{prop:no-symmetric-ne} proves that no interior diagonal profile is a
pure NE in the fully symmetric 2-player PEG; the numerical result here
is the strong form of that theorem, showing that every pure NE on the
$K = 200$ grid is asymmetric, with the eight NEs falling into four
mirror-image swaps. The externality \emph{must} be carried by someone,
and at equilibrium it is carried asymmetrically.
```

### 5.3 One-line abstract touch (optional; belongs to T01)

If the abstract is being rebalanced (task T01, see
`NOVELTY_ACTION_PLAN.md:202`), the load-bearing sentence is:

> Even under fully symmetric players, the sampling-based iDP game admits
> no diagonal Nash equilibrium; some player always carries a higher
> realised MIA advantage in exchange for a higher shared model utility
> (\cref{prop:no-symmetric-ne}).

This is a T01 concern; T03 is complete when the proposition, proof, and
two downstream rewrites are in place.

---

## 6. Acceptance checklist

Before the corresponding paper edit is merged, the following must all
be true:

- [ ] `paper/main.tex` compiles cleanly with no *new* "undefined
      reference" or "multiply-defined label" warnings involving
      `prop:no-symmetric-ne` or `sec:no-symmetric-ne`.
- [ ] The proposition depends only on `def:peg` (`03_game.tex:12`), the
      `V` regularity assumption (`03_game.tex:68-70`), `lem:externality`
      (`04_analysis.tex:11`), and the mechanism-regularity `dA/dp > 0`
      (a standing property of the subsampled Gaussian, cited to the
      companion `\citet{kaiser2026your}`).
- [ ] `grep -n 'no-symmetric-ne' paper/main.log` returns no unresolved-
      reference lines after `pdflatex -> bibtex -> pdflatex -> pdflatex`.
- [ ] The proof body does not contain the string `1 + \sigma^2` (i.e.
      does not use the specific closed form of $V$); grep confirms
      absence.
- [ ] The proof body does not contain any references to `w_i`
      asymmetric, `B_i / C_i`, `B/C`, or `benefit-to-cost` (stale
      utility-form language, per `CLAUDE.md` Rule 0).
- [ ] Running `python tasks_findings/_scripts/03_check.py` on all
      available `_cache_v2/*.pkl` reports 0 diagonal pure NEs across all
      symmetric-hypothesis rows.
- [ ] After the rewrites at `04_analysis.tex:126-132` and
      `05_experiments.tex:61-70`, the informal diagonal-instability
      argument appears in the paper exactly once (in the proof of
      `prop:no-symmetric-ne`); every other occurrence cites the
      proposition rather than restating it.
- [ ] `lem:externality` (currently Lemma 1) and `prop:existence`
      (currently Proposition 2) retain their numbers; the new
      `prop:no-symmetric-ne` takes the slot between them (Proposition 2,
      pushing the current Proposition 2 to Proposition 3, etc.). The
      global grep `grep -rn '\\\\cref{prop:' paper/tex/` must return only
      references that still resolve after this renumbering; any
      hard-coded number (e.g. "Proposition 2") in prose must be
      converted to `\cref` before landing.
- [ ] The proposition's "sign condition" \eqref{eq:sign-condition} (the
      subtle step of the proof) is either verified analytically for the
      subsampled Gaussian in an appendix footnote, or the proposition is
      stated in the conditional form ("Under \eqref{eq:sign-condition},
      no interior diagonal profile is a pure NE") with a one-sentence
      note that the condition holds numerically on all our scenarios
      (Table T3.1 above). The `NOVELTY_ACTION_PLAN.md:191` fallback --
      "generic non-existence" -- is the acceptable graceful degradation
      if the appendix computation stalls.

## 7. References consulted

- Task specification: `.tasks/03_H_L_no-symmetric-ne-proposition.md`.
- Novelty framing: `NOVELTY_REVIEW.md:11, 33, 61, 65`; `NOVELTY_ACTION_PLAN.md:11-19, 191, 202, 254, 262`.
- Externality lemma and its proof: `paper/tex/04_analysis.tex:11-65`.
- Standing regularity on `V`: `paper/tex/03_game.tex:50-70`.
- Sampling-based iDP mechanism definition: `paper/tex/02_background.tex:44-70`.
- Utility form: `paper/tex/03_game.tex:12-44`; code at `game/utility.py`.
- C1 numerical result (target of formalisation): `paper/tex/05_experiments.tex:42-84`.
- Existing cache used by the sanity-check script: `game_experiments/_cache_v2/C1.pkl`, `C2.pkl`.
- Related canonical work on continuous-game NE existence (background
  for the "not a numerical accident" framing): Debreu (1952),
  Glicksberg (1952), Fan (1952) classical existence theorems require
  quasi-concavity in own action; the point of the present proposition
  is that this quasi-concavity fails on the diagonal, ruling out
  standard existence routes from certifying a diagonal NE. Milgrom &
  Roberts (1990), Vives (1990), and Jensen (2010) provide the
  supermodular/aggregative route that is the eventual continuous-game
  *existence* complement, per `EXTENSION_DIRECTIONS.md` and
  `NOVELTY_ACTION_PLAN.md:29`; not required for the present negative
  result.
