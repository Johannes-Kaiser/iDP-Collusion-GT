# T09 -- BR-PEG sharpen and promote out of appendix

**Novelty priority:** H   **Effort:** L   **Category:** short-proof + repositioning

Publication-quality draft for insertion into `paper/tex/07_discussion.tex`
(new subsection `sec:br-peg`), with the appendix source
`paper/tex/09_bayesian.tex` retired. This document does *not* modify the
paper sources; it is the drop-in text plus supporting artefacts.

Anchors:

- Cache: `game_experiments/_cache_v2/bayesian_demo.pkl`.
- Verification script: `tasks_findings/_scripts/09_check.py` (run against
  `.venv`; output at `tasks_findings/_scripts/09_regret_curve.txt`).
- Code: `game/bayesian.py` (H's BR at line 71, posterior-greedy at line
  123, posterior update at line 164, simulator at line 199).

---

## 1. Goal and narrative role

### 1.1 What we claim

In the Bayesian Repeated Privacy-Externality Game (BR-PEG) -- H holds a
private type `theta_H = (w_H, n_H)` drawn from a finite known set
`Theta`, M plays a leader who commits to `(eps_M, n_M)` each round, H
myopically best-responds, and M updates a posterior over `theta_H` from
the pair `(eps_H, u_H)` -- the posterior-greedy budget of M converges to
an **interior** point of the feasible action grid, strictly above the
participation-preserving minimum. Concretely, for the demo instance of
`game_experiments/run_bayesian.py`,

```
eps_min (grid floor) = 0.5,     eps_min (naive-attack anchor) = 1.0
eps_M^greedy(t)       -> 3.084   for all t >= 1
A_H at eps_M^greedy   -> 0.4540  (= oracle value; posterior collapses at t = 1)
eta := eps_M^greedy - eps_min = 2.584  (vs. grid floor) or 2.084 (vs. naive)
```

(from `09_check.py`, `bayesian_demo.pkl`).

### 1.2 Why it matters and how it differs from Stackelberg-learning

The security-conference finding is that the *optimal adversarial budget is
not the minimum feasible one*. The naive picture in the one-shot literature
-- an attacker declares `eps_M -> 0` to force `sigma` down -- is wrong under
sampling-based iDP once H is free to move: at very low `eps_M` the
mechanism's sampling floor collapses H's participation rate, H's best
response is a small `eps_H`, and H therefore leaks little. M's gain comes
from raising `eps_M` far enough that H's best response is a *large*
`eps_H`, which the mechanism then executes at a fine sampling rate that
exposes H. Bayesian identification of `theta_H` is what unlocks this
manoeuvre: M cannot optimise a composed map through H's best response
without knowing H's weight.

Positioning against the neighbouring literature:

- **Stackelberg-learning template (Letchford-Munro-Parkes 2009 [1],
  Blum-Haghtalab-Procaccia 2014 [2], Balcan-Blum-Haghtalab-Procaccia 2015
  [3]).** These works give no-regret / query-complexity guarantees for a
  leader who commits to a mixed strategy each round against an unknown-
  type follower on a *finite* action set. BR-PEG is the same skeleton --
  leader commits, follower best-responds, leader updates -- but the
  follower's action is continuous (`eps_H` on a compact interval) and the
  per-round payoff is delivered by a differential-privacy accountant
  rather than a tabular game. The mechanism induces a coupling between
  the two players' budgets that the tabular template does not model;
  Proposition~\ref{prop:bayesian-not-min} identifies the qualitative
  consequence of that coupling.
- **One-sided incomplete-information repeated games (Aumann-Maschler
  1995 [4]).** Their central problem is the informed player's optimal
  revelation policy. In BR-PEG the informed player is myopic, so the
  Aumann-Maschler splitting disappears and the game reduces to
  Stackelberg-learning. We keep the citation to signal where the fully
  two-sided extension lives.
- **Rational learning (Kalai-Lehrer 1993 [5]).** Convergence to Nash
  under an absolute-continuity ("grain of truth") condition. BR-PEG
  satisfies this trivially -- H's type is drawn from a finite `Theta`
  with M's prior full-support on `Theta` -- so posterior consistency is
  automatic; the interesting content is what M does *once* consistency
  kicks in.
- **No-regret in Bayesian games (Kalai-Vempala; Hartline-Syrgkanis-
  Tardos 2015 [6]).** These bound external regret against the best
  fixed action in hindsight. Our claim is stronger and different: we
  identify the fixed action in hindsight for our specific game and show
  it is bounded away from the naive one. Regret rates are follow-up
  work (T16).

The differentiator against the whole line is therefore: BR-PEG is not
an algorithm-design contribution ("here is a no-regret learner"); it is
a game-theoretic identification of an interior adversarial optimum in a
mechanism-based Stackelberg game whose leader-follower coupling is set
by DP-accounting rather than by a tabular payoff matrix. This is the
promise-vs-reality frame the paper argues elsewhere: the naive picture
is `eps_M^attack = eps_min`; the reality is `eps_M^attack = eps_min +
eta` with `eta` on the order of the grid width.

---

## 2. Diagnosis of the current appendix's soft claims

The appendix (`paper/tex/09_bayesian.tex`) reads as an illustrative
sketch, not a numbered claim. Specific soft spots:

**Line 85-96 -- the buried punchline.** "The Bayesian-greedy attack
identifies the true type in one round and then plays `eps_M ~ 3.1` for
the remaining rounds, eliciting `eps_H ~ 19` from H and delivering a
per-round `A_H ~ 0.45` --- roughly twice the naive attack and around
fifty times the benign baseline. The optimal adversarial budget is
therefore *not* the minimum". This is the paper's most novel result
per `NOVELTY_REVIEW.md` (sec. 2 item 5, sec. 6 item 5), and it appears
as a paragraph inside an unnumbered `\paragraph{Simulation.}` block, with
no cross-referable label. Reviewers cannot point to it.

**Line 4-13 -- the framing.** "This appendix sketches how the PEG
specialises to a standard template in that line of work". The word
"sketches" invites the reviewer objection "then why is this in the
paper at all?". The BR-PEG is not a specialisation of Stackelberg-
learning -- it is an instance whose leader-follower coupling produces
a non-tabular phenomenon (`eps_M^greedy` bounded away from `eps_min`)
that the tabular template does not predict.

**Line 33-44 -- the "differences from the standard template are minor"
sentence.** This undersells the contribution. The continuous follower
action and the DP-mechanism coupling are precisely what force the
interior optimum. They should be flagged as the load-bearing
differentiators, not as minor implementation choices.

**Line 62-70 -- reduction to Stackelberg-learning.** "The framework is a
one-sided incomplete-information repeated game... but with the informed
player myopic, which reduces it to the Stackelberg-learning instance of
Letchford et al.". This is correct but sells the game as *the same*
object as Stackelberg-learning. See the previous point.

**Line 74 -- the type comment.** The appendix says H's true type is
`w_H = 0.5` (utility-preferring), but the cache
`_cache_v2/bayesian_demo.pkl` was written under the legacy two-weight
formulation with `(B, C) = (2, 1)`, i.e. effective `w = B/C = 2`
(privacy-preferring). Under the *current* single-weight utility of
`game/utility.py` the H's BR and M's posterior-greedy argmax depend only
on `w = B/C`, so the trajectory in `fig_bayesian.pdf` is what a fresh
run with `w_true = 2.0` would produce. The prose and the figure are
therefore *inconsistent* on the sign of the type: the promoted subsection
must either regenerate the figure under `w_true = 0.5` (T16 does this as
part of the regret sweep) or align the prose to `w_true = 2.0`. This
document adopts the second option because it is the cheaper fix and the
qualitative claim (the interior optimum) is unchanged.

**Line 112-124 -- the "what this leaves open" hedge.** The claim
"framework slots into either extension with no further modification" is
correct but reads as a defensive coda. In the main body it should
become a scope statement, one line, at the end of the subsection.

---

## 3. Sharpened formal statement (publication-ready LaTeX)

Insertion target: new `\subsection{Repeated Bayesian variant (BR-PEG)}`
in `paper/tex/07_discussion.tex`, between the "Limitations of this work"
paragraph and the "The role of the model-utility function `V`"
paragraph. See section 6 for the promotion plan.

```latex
\subsection{Repeated Bayesian variant (BR-PEG)}
\label{sec:br-peg}

The one-shot analysis leaves the strategic content of repetition and
incomplete information untouched. We formalise the natural extension --
the \emph{Bayesian Repeated PEG} -- and identify the qualitative
consequence of the mechanism's leader--follower coupling: the optimal
adversarial budget is bounded away from the minimum feasible one.

\paragraph{The game.}
Two players, $H$ (honest) and $M$ (malicious). At $t=0$, $H$ draws a
private type $\theta_H = (w_H, n_H)$ from a finite set $\Theta$ known
to $M$. Per round $t \in \{1,\dots,T\}$:
\begin{enumerate}
\item $M$ observes the history $\mathcal{H}_{t-1}$ and picks
$(\eps_M^t, n_M^t)$;
\item $H$ observes $(\eps_M^t, n_M^t)$ and myopically best-responds
with $\eps_H^t \in [\eps^{\min}, \eps^{\max}]$ that maximises the
one-shot utility $u_H(\eps_H; \eps_M^t, \theta_H) = V(\sigma) -
w_H A_H$, given by the sampling-based iDP mechanism of
\cref{def:sampling-idp};
\item the mechanism returns $(\sigma^t, p_H^t, p_M^t, A_H^t, A_M^t)$;
\item $(\eps_H^t, u_H^t)$ becomes public, and $M$ updates a posterior
$\pi^t \in \Delta(\Theta)$ over $\theta_H$ by
\begin{equation}
\pi^t(\theta) \;\propto\; \pi^{t-1}(\theta)\,
L\bigl(\eps_H^t, u_H^t \,\big|\, \eps_M^t, \theta\bigr),
\label{eq:posterior-update}
\end{equation}
where $L$ is the observation likelihood induced by $H$'s best-response
map.
\end{enumerate}
$M$'s total score is $\sum_t A_H^t$. Positioned in the literature, the
game is the Stackelberg-learning template of
\citet{letchford2009learning, blum2014learning, balcan2015commitment}
with the tabular follower payoff replaced by the DP-accountant coupling
of \cref{def:sampling-idp}, and it is the myopic-informed special case
of the one-sided incomplete-information repeated game of
\citet{aumann1995repeated}.

\paragraph{Information structure.}
$M$ knows $\Theta$, $V$, and the mechanism; $H$ knows her own type.
Both observe $(\eps_H^t, u_H^t)$ after each round; only $M$ maintains a
non-degenerate belief. The prior $\pi^0$ has full support on $\Theta$
so that Kalai--Lehrer's absolute-continuity condition holds trivially;
posterior consistency is therefore automatic and the interesting
content is what $M$ does once consistency kicks in.

\begin{proposition}[Bayesian-greedy attack is not min-budget]
\label{prop:bayesian-not-min}
Consider the BR-PEG with $H$ myopic and $M$ posterior-greedy on a finite
action grid $\mathcal{A} \subset [\eps^{\min}, \eps^{\max}]$. Fix a
type $\theta_H^\star = (w_H^\star, n_H^\star) \in \Theta$ and let
\[
G(\eps_M; \theta) \;:=\; A_H\!\bigl(\eps_M,\, \eps_H^{\mathrm{BR}}(\eps_M; \theta)\bigr)
\]
be the composed map from $M$'s action to the elicited advantage, where
$\eps_H^{\mathrm{BR}}(\eps_M; \theta) \in \arg\max_{\eps_H \in
\mathcal{A}} u_H(\eps_H; \eps_M, \theta)$. Assume
\begin{enumerate}
\item[\normalfont(I)] \emph{Identifiability.} The likelihood
in~\eqref{eq:posterior-update} is
$\theta_H^\star$-identifying on the equilibrium path in the sense of
Kalai--Lehrer: there exists a full-support prior $\pi^0$ and a
$M$-policy $\pi^0$-a.s. such that $\pi^t(\theta_H^\star) \to 1$;
\item[\normalfont(A)] \emph{Interior argmax.} $G(\cdot; \theta_H^\star)$
attains its maximum on $\mathcal{A}$ at
$\eps_M^\star(\theta_H^\star) \in \mathcal{A}$ with
$\eps_M^\star(\theta_H^\star) > \eps^{\min}$.
\end{enumerate}
Then the posterior-greedy action
$\eps_M^{\mathrm{greedy}}(t) \in \arg\max_{\eps_M \in \mathcal{A}}
\mathbb{E}_{\theta \sim \pi^t}\!\bigl[G(\eps_M; \theta)\bigr]$
satisfies
\begin{equation}
\eps_M^{\mathrm{greedy}}(t) \;=\; \eps_M^\star(\theta_H^\star)
\qquad \text{eventually, } \pi^0\text{-a.s.,}
\label{eq:eps-M-limit}
\end{equation}
and, setting $\eta := \eps_M^\star(\theta_H^\star) - \eps^{\min} > 0$,
\begin{equation}
\eps_M^{\mathrm{greedy}}(t) \;\geq\; \eps^{\min} + \eta
\qquad \text{for all sufficiently large } t.
\label{eq:eta-gap}
\end{equation}
Equivalently, $M$'s posterior-greedy action is bounded away from the
participation-preserving minimum budget.
\end{proposition}
```

Notes on the statement:

- Assumption (I) is Kalai-Lehrer merging on the equilibrium path; it is
  satisfied automatically when `Theta` is finite, `pi_0` has full
  support, and the likelihood `L` in
  `game/bayesian.py:posterior_update` separates types via the log-
  Gaussian metric on `(eps_H, u_H)` with `sig = cfg.posterior_noise`
  (line 183). We do not attempt to state the metric-separation
  condition in the paper; it is implicit in "identifying".
- Assumption (A) is what the numerical instance provides. The
  externality lemma of \cref{lem:externality} plus the sign of
  `deps_H^BR / deps_M` on the interior determines whether
  `G(.; theta*)` has an interior max, but the sign is *not*
  monotone across the whole feasible region (see section 3.1 below);
  we therefore state (A) as an assumption verified numerically by the
  demo, not as a corollary of the externality lemma.
- `eta > 0` is a per-type quantity depending on `theta_H^star` and on
  `V`. This is the honest statement: T16's regret sweep in
  `EXPERIMENT_IDEAS.md:152` maps `eta(theta)` over `Theta`.

### 3.1 Why we do *not* state monotonicity as a lemma

The proof sketch in `.tasks/09_H_L_br-peg-sharpen-and-promote.md`
lines 106-127 argues the following:

> As M raises `eps_M`, the induced `sigma` rises, so H's participation
> rate `p_H` rises, and H's best response accepts a higher `eps_H`
> because the model-utility benefit `V(sigma)` dominates the marginal
> `w_H A_H` cost.

This is directionally correct on a *sub-interval* of `[eps^min,
eps^max]` but not uniformly. At very low `eps_M`, (C2) binds hard, `p_H`
is at its floor, and `V(sigma) - w_H A_H` is dominated by the
`w_H A_H` term; H's BR sits at the grid minimum. At intermediate `eps_M`
the mechanism releases the constraint on `p_H`, `V(sigma)` rises fast,
and H's BR jumps to a large `eps_H`. At high `eps_M`, both `sigma` and
`p_H` are at ceiling, `V(sigma) ~ 1`, and H's BR flattens or reverses
because the marginal `V` gain vanishes while `A_H` continues to rise
with `eps_H`. `game/bayesian.py:H_best_response` computes the BR by
grid argmax precisely because there is no closed-form monotone
characterisation.

We therefore treat (A) as an assumption verified numerically for the
concrete configuration `(V, Theta, expected_batch, steps, delta)` of
\cref{sec:experiments}. This is a genuine restriction; it is why the
proposition says "there exists `eta > 0`" and not "for every
`theta_H^star`, `eta(theta) > eta_0`".

---

## 4. Complete proof (publication-ready LaTeX)

```latex
\begin{proof}[Proof of \cref{prop:bayesian-not-min}]
Write $\bar G(\eps_M; \pi) := \mathbb{E}_{\theta \sim \pi} G(\eps_M;
\theta)$ for the posterior-averaged score. The map $\eps_M \mapsto
\bar G(\eps_M; \pi)$ is a finite mixture of the per-type scores, hence
continuous in $\pi$ under the total-variation topology and continuous
in $\eps_M$ on the finite grid $\mathcal{A}$.

\emph{Step 1 (posterior collapses on the truth).}  By identifiability
(I), $\pi^t \to \delta_{\theta_H^\star}$ almost surely; equivalently,
$\|\pi^t - \delta_{\theta_H^\star}\|_{\mathrm{TV}} \to 0$. This is the
Kalai--Lehrer merging conclusion~\cite{kalai1993rational} specialised
to a finite type space with a full-support prior and a type-separating
likelihood; the equilibrium-path guarantee is delivered by the fact
that $M$'s posterior-greedy action is itself finite-valued and
therefore any $\theta \neq \theta_H^\star$ that produces a distinct
best response at $M$'s chosen $\eps_M^t$ contributes a strictly
positive Kullback--Leibler divergence to the log-likelihood, which
accumulates over rounds.

\emph{Step 2 (score converges uniformly to the truth-score).}  Because
$G(\eps_M; \theta)$ is uniformly bounded ($0 \leq A_H \leq 1$), for
every $\eps_M \in \mathcal{A}$,
\[
\bigl|\bar G(\eps_M; \pi^t) - G(\eps_M; \theta_H^\star)\bigr|
\;\leq\; \sum_{\theta \neq \theta_H^\star} \pi^t(\theta) \cdot 1
\;=\; 1 - \pi^t(\theta_H^\star) \;\longrightarrow\; 0.
\]
Since $\mathcal{A}$ is finite, the convergence is uniform in $\eps_M$.

\emph{Step 3 (argmax converges to the truth-argmax).}  Let
$\eps_M^\star := \eps_M^\star(\theta_H^\star)$ and
$\Delta := G(\eps_M^\star; \theta_H^\star) - \max_{\eps_M \in
\mathcal{A} \setminus \{\eps_M^\star\}} G(\eps_M; \theta_H^\star)$. By
(A), $\Delta > 0$: the truth-argmax is the unique interior maximiser
on the grid.\footnote{If the truth-argmax is degenerate ($\Delta = 0$),
replace $\eps_M^\star$ by the set of maximisers and the conclusion
holds set-valued.} Choose $t_0$ such that $1 - \pi^t(\theta_H^\star) <
\Delta / 2$ for all $t \geq t_0$; such $t_0$ exists by Step 1. For
$t \geq t_0$ and any $\eps_M \neq \eps_M^\star$,
\begin{align*}
\bar G(\eps_M^\star; \pi^t) - \bar G(\eps_M; \pi^t)
&\;\geq\; G(\eps_M^\star; \theta_H^\star) - G(\eps_M; \theta_H^\star)
- 2(1 - \pi^t(\theta_H^\star)) \\
&\;\geq\; \Delta - \Delta \;>\; 0
\quad\text{when } G(\eps_M; \theta_H^\star) \leq G(\eps_M^\star;
\theta_H^\star) - \Delta,
\end{align*}
which holds by definition of $\Delta$. Hence $\eps_M^{\mathrm{greedy}}
(t) = \eps_M^\star$ for all $t \geq t_0$, which is
\eqref{eq:eps-M-limit}.

\emph{Step 4 (separation from $\eps^{\min}$).} By (A), $\eta :=
\eps_M^\star - \eps^{\min} > 0$; substituting into \eqref{eq:eps-M-limit}
gives \eqref{eq:eta-gap}. \qedhere
\end{proof}
```

### 4.1 Anchoring to the code

The proof's ingredients correspond exactly to the following pieces of
`game/bayesian.py`:

| Proof object | Code |
|---|---|
| `G(eps_M; theta)` | inner loop of `strategy_posterior_greedy` (`bayesian.py:131-149`), reduced to a single `theta` |
| `bar G(eps_M; pi)` | full inner loop `bayesian.py:131-155` (mixture over posterior mass) |
| `eps_M^greedy(t)` | outer argmax `bayesian.py:150-154` |
| Posterior update (Step 1) | `posterior_update` (`bayesian.py:164-192`) with log-Gaussian likelihood, `sig = cfg.posterior_noise` (line 183) |
| Kalai-Lehrer identifiability | full-support `initial_posterior` in `simulate` (`bayesian.py:203-207`) plus finite `Theta` in `GameConfig` (`bayesian.py:105`) |
| Truth-argmax `eps_M^star` | numerical fact reported by `09_check.py`: `eps_M^star = 3.084` |

The proof does not require any monotonicity of `eps_H^BR` in `eps_M`;
that was the load-bearing hole in the task-spec sketch and is what
motivated Section~3.1's honest downgrade of the monotonicity argument
to a numerically-verified assumption.

### 4.2 Fallback statement (if reviewers push back on (A))

If a reviewer objects that (A) is verified only for one type, restate
the proposition as a numerical certificate over the K=200 mechanism
grid across all `theta_H in Theta_demo` (analogous to T12's numerical
externality certificate). The prose becomes:

> On the mechanism grid of `\cref{sec:experiments}` and the type set
> `Theta_demo = {w_H in {0.5, 1, 2}, n_H = 5000}`, the posterior-greedy
> argmax `eps_M^greedy` under a Dirac-at-`theta_H^star` posterior lies
> strictly above the grid minimum for every `theta_H^star in
> Theta_demo`.

This is honest, still defeats the "trivially `eps_M = eps_min`"
objection, and matches T12's style.

---

## 5. Numerical corroboration

### 5.1 Script

`tasks_findings/_scripts/09_check.py` loads
`_cache_v2/bayesian_demo.pkl`, computes the anchors used in the
proposition and its proof, and writes an ASCII regret table to
`_scripts/09_regret_curve.txt`. Reproduce with

```
/vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \
    tasks_findings/_scripts/09_check.py
```

### 5.2 Anchors extracted from the cache

```
true H type:                w = 2.0  (from legacy B=2.0, C=1.0)
M grid floor:               eps_min = 0.5
eps_M^greedy(T):            3.084
eps_H^BR at eps_M^star:     19.027
A_H at eps_M^star:          0.4540
eta vs. grid floor 0.5:     2.584
eta vs. naive min 1.0:      2.084
identifiability round t*:   1  (pi^t(true) >= 0.99 at t=1)
cumulative regret over T=15:+0.138  (all incurred at t=0)
```

The proposition's proof calls out `eps_M^star = 3.084` and `eta = 2.084`
as the numerical realisation; the paper prose in section~6 uses those
values verbatim.

### 5.3 Regret vs rounds

Oracle baseline: `A_H^oracle = A_H` under a Dirac-at-`theta_H^star`
posterior, which by inspection of the cache equals `A_H` at any
`t >= 1` where the posterior has collapsed (`p_true = 1.000`).
Per-round regret is `regret_t = A_H^oracle - A_H^bayesian(t)`.

```
t     A_H_bayes  A_H_naive  A_H_high   regret     cum_regret  p_true
 0    0.316      0.219      0.008      +0.138     +0.138      0.745
 1    0.454      0.219      0.008      +0.000     +0.138      1.000
 2    0.454      0.219      0.008      +0.000     +0.138      1.000
...                                                (constant tail)
14    0.454      0.219      0.008      +0.000     +0.138      1.000
```

Full table at `tasks_findings/_scripts/09_regret_curve.txt`.

Reading:

- **Instantaneous regret** decays to zero after one round. This is
  visually a step function, not a `1/sqrt(t)` decay, because the
  finite-type Kalai-Lehrer setup allows one-shot identification when
  the log-Gaussian likelihood cleanly separates the three types.
- **Cumulative regret** saturates at `+0.138 = A_H^oracle -
  A_H^bayesian(0)`, a constant. This is the sharpest possible statement
  compatible with `A_H in [0, 1]` and one-round identifiability. It is
  the identification-of-Bayesian-Nash-equilibrium regret, not a
  `O(sqrt(T))` regret bound; the two are conceptually distinct.
- **Comparison to the naive attack** (`eps_M = 1.0`, constant): the
  Bayesian-greedy attack outperforms it by a factor of `0.454 / 0.219 =
  2.08` per round from `t = 1` onwards, so the *advantage* over the
  Stackelberg-learning-tabular literature's benchmark strategy is
  `Omega(T)`, not sublinear.

The paper prose in the promoted subsection reports one line: "the
posterior collapses at `t = 1`, and from then on the Bayesian-greedy
attack plays `eps_M^star \approx 3.08` and elicits `A_H \approx 0.45`
--- twice the naive attack and fifty times the benign baseline." T16
will produce the `eta`-vs-`theta` sweep and a genuine
identification-rounds curve for the reviewer figure.

### 5.4 Stale-cache flag

The cache was written with the legacy `ThetaH(B, C, n)` schema, so
`theta_H_true.B = 2.0, C = 1.0` corresponds to `w = 2.0` (privacy-
preferring). The current `run_bayesian.py` (post-refactor) sets
`theta_H_true = types[2]` with `w = 0.5` (utility-preferring). The
trajectory in `fig_bayesian.pdf` is the `w = 2.0` trajectory; the
appendix prose currently states `w = 0.5`. The promoted subsection
below aligns the prose with the figure (`w_H^\star = 2.0`); T16 will
regenerate the figure under `w_H^\star = 0.5` if that is the preferred
canonical choice.

---

## 6. Promotion plan (what goes where)

**Do not edit `paper/` in this task.** Deliverables for T09-executor
(a follow-up implementation task) or for direct hand-off:

### 6.1 New subsection in `paper/tex/07_discussion.tex`

Insert *after* the "Limitations of this work" paragraph
(`07_discussion.tex:15-38`) and *before* the "The role of the model-
utility function `V`" paragraph (line 40). Content:

```latex
\subsection{Repeated Bayesian variant (BR-PEG)}
\label{sec:br-peg}

The one-shot analysis leaves the strategic content of repetition and
incomplete information untouched. We formalise the natural extension --
the \emph{Bayesian Repeated PEG} -- and identify the qualitative
consequence of the mechanism's leader--follower coupling: the optimal
adversarial budget is bounded away from the minimum feasible one.

[GAME PARAGRAPH from Section 3 above]

[INFORMATION-STRUCTURE PARAGRAPH from Section 3 above]

\begin{proposition}[Bayesian-greedy attack is not min-budget]
\label{prop:bayesian-not-min}
[STATEMENT from Section 3 above]
\end{proposition}

\begin{proof}
[PROOF from Section 4 above]
\end{proof}

\paragraph{Numerical realisation.}
We run one instance to fix intuition. $H$'s true type is $w_H^\star =
2.0$ (privacy-preferring); $M$'s prior is uniform over $\Theta = \{w_H
\in \{0.5, 1, 2\}\}$ with $n_H = n_M = 5000$; other parameters match
\cref{sec:experiments}. Three $M$ strategies are compared: benign
constant $\eps_M = 64$, naive-minimum constant $\eps_M = 1$, and
posterior-greedy Bayesian. \Cref{fig:bayesian} shows the outcome.
The posterior collapses to $\theta_H^\star$ within one round, after
which $\eps_M^{\mathrm{greedy}} = 3.08$ and $A_H = 0.454$
--- roughly twice the naive attack, and fifty times the benign
baseline. The truth-argmax is interior to the grid, so the assumption
(A) of \cref{prop:bayesian-not-min} is verified; the separation is
$\eta \approx 2.08$ against the naive-attack anchor $\eps^{\min} = 1$.

\begin{figure}[!t]
\centering
\includegraphics[width=0.95\linewidth]{figures/fig_bayesian.pdf}
\caption{\textbf{BR-PEG demo.} Left: per-round realised MIA advantage
$A_H$ of the honest player under three $M$ policies. Right: $M$'s
posterior over $H$'s type under the Bayesian-greedy policy; the
posterior collapses to the true type within one round.}
\label{fig:bayesian}
\end{figure}

\paragraph{Scope.} Two extensions sit outside the scope of this paper.
First, the strategising-$H$ case, in which $H$ chooses her action
sequence to manage what she reveals: this is a two-sided incomplete-
information game in the sense of \citet{aumann1995repeated} and the
iDP-adapted regret analysis is a follow-up programme. Second, a
regret rate in the style of \citet{balcan2015commitment,
blum2014learning} that measures the number of rounds until
$\eps_M^{\mathrm{greedy}}(t) = \eps_M^\star$; the finite-type
Kalai--Lehrer setup used here delivers one-round identification and
so provides only a constant-regret upper bound. Both extensions fit
the framework above without further modification.
```

### 6.2 Cross-reference rewrite in `paper/tex/07_discussion.tex`

Line 20 currently reads

> `\cref{app:bayesian} sketches such a game --- the \emph{Bayesian
> Repeated PEG} --- as a direct instance of the Stackelberg-learning
> template of \citet{letchford2009learning, blum2014learning,
> balcan2015commitment} applied to the iDP mechanism, together with a
> small simulation. We treat it as a brief extension here and do not
> pursue regret bounds or the fully two-sided repeated game of
> \citet{aumann1995repeated}.`

Rewrite to

> `\Cref{sec:br-peg} formalises such a game as the \emph{Bayesian
> Repeated PEG} (BR-PEG), states a sharpened attack characterisation
> (\cref{prop:bayesian-not-min}) that identifies the optimal
> adversarial budget as strictly interior to the feasible action grid,
> and reports a small simulation. We treat the fully two-sided
> repeated game of \citet{aumann1995repeated} and quantitative regret
> rates in the style of \citet{balcan2015commitment} as follow-up work.`

### 6.3 `paper/main.tex`

Delete the line `\input{tex/09_bayesian}` (currently line 109). Keep
`\input{tex/08_appendix}` in place.

### 6.4 `paper/tex/09_bayesian.tex`

Delete. All material has moved to `sec:br-peg`.

### 6.5 Column-width change on the figure

Change `figure*` to `figure` (single-column IEEEtran; discussion is in
one column). Caption fits after the change (verified by inspection of
`fig_bayesian.pdf`'s aspect ratio).

### 6.6 Regression sweeps

- `grep -rn "app:bayesian" paper/` -- should return no hits after the
  rewrite.
- `grep -rn "fig:bayesian" paper/` -- should still resolve to a single
  hit in the new `sec:br-peg`.
- `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex
  main.tex` -- warning-clean per CLAUDE.md Rule 4.

---

## 7. Acceptance checklist

Post-implementation (T09-executor), verify:

- [ ] `paper/main.tex` no longer inputs `tex/09_bayesian.tex`; the file
  is deleted.
- [ ] `paper/tex/07_discussion.tex` contains `\subsection{Repeated
  Bayesian variant (BR-PEG)}` with `\label{sec:br-peg}` between the
  Limitations paragraph and the `V`-role paragraph.
- [ ] `\begin{proposition}[Bayesian-greedy attack is not min-budget]
  \label{prop:bayesian-not-min}` renders with an auto-numbered
  `Proposition N.` label sharing the `theorem` counter
  (`paper/main.tex:19-27`).
- [ ] `\cref{prop:bayesian-not-min}` resolves in the rewritten
  cross-reference sentence at former line 20 of
  `paper/tex/07_discussion.tex`.
- [ ] Figure `\ref{fig:bayesian}` renders in the main body, single
  column, with the caption unchanged from Section 6.1.
- [ ] Numerical anchors in the promoted prose match
  `09_check.py`'s output to two significant figures:
  `eps_M^star ~ 3.08`, `A_H ~ 0.454`, `eta ~ 2.08` (against
  `eps^{\min} = 1`, the naive-attack anchor).
- [ ] `grep -rn "app:bayesian" paper/` returns nothing.
- [ ] `grep -rn "fig:bayesian" paper/` returns exactly one hit
  (in `07_discussion.tex`).
- [ ] `pdflatex main.tex && bibtex main && pdflatex main.tex &&
  pdflatex main.tex` produces `paper/main.pdf` with no `Reference ...
  undefined` warnings and no new hyperref warnings beyond the
  pre-existing math-in-heading ones (CLAUDE.md Rule 4).
- [ ] `2601.12922v1.pdf` voice check on the promoted subsection: the
  opening promise-vs-reality frame is present ("the naive picture is
  `eps_M^attack = eps_min`; we show the reality is `eps_M^attack =
  eps_min + eta`"), first-person plural throughout, no
  throat-clearing, one metaphor at most (CLAUDE.md Rule 1).
- [ ] `NOVELTY_REVIEW.md` sec 2 item 5 and sec 6 item 5 are addressed:
  the "most novel result" is a numbered proposition in the main body,
  not a paragraph in an appendix.

Follow-up unblocked by T09: T16 (`.tasks/16_M_L_h8-brpeg-regret-curve.md`)
takes `prop:bayesian-not-min` as its formal statement, sweeps
`eta(theta)` across `Theta`, and produces the regret-vs-rounds figure
under a richer type set that gives multi-round (not one-round)
identification.

---

## References cited above

[1] Letchford, Conitzer, Munagala. "Learning and approximating the
optimal strategy to commit to." SAGT 2009. Cited in
`paper/tex/09_bayesian.tex:22`.

[2] Blum, Haghtalab, Procaccia. "Learning optimal commitment to
overcome insecurity." NeurIPS 2014. Cited at same line.

[3] Balcan, Blum, Haghtalab, Procaccia. "Commitment without regrets:
online learning in Stackelberg security games." EC 2015. Cited at
same line.

[4] Aumann, Maschler. *Repeated Games with Incomplete Information.*
MIT Press 1995. Cited in `paper/tex/09_bayesian.tex:16`.

[5] Kalai, Lehrer. "Rational learning leads to Nash equilibrium."
*Econometrica* 1993. Cited in `paper/tex/09_bayesian.tex:24`.

[6] Hartline, Syrgkanis, Tardos. "No-regret learning in Bayesian
games." NeurIPS 2015. Not currently cited; only mentioned in this
document's positioning discussion and not needed as a new bib entry.
