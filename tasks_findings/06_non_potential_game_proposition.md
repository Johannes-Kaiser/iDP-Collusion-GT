# T06 - Non-potential-game proposition

**Novelty priority:** M   **Effort:** L   **Category:** short-proof

Publication-quality draft of Proposition `prop:non-potential`, its two
supporting lemmas, and the numerical corroboration. Insertion targets:
`paper/tex/04_analysis.tex` (proposition), `paper/tex/08_appendix.tex`
(lemmas + proofs + numerical certificate). *This note is the specification
only; it does not modify the paper sources.*

Reproducibility scripts live in `tasks_findings/_scripts/`:

- `t06_symbolic_mixed_partial.py` -- sympy verification of every algebraic
  step below, including a three-player sanity check.
- `t06_numerical_cross_check.py` -- finite-difference Monderer-Shapley
  falsification on `game_experiments/_cache_v2/C1.pkl`.
- `t06_improvement_cycle.py` -- explicit 4-cycle of strict unilateral
  improvements on the same cache; rules out an *ordinal* potential.

Run:

    .venv/bin/python tasks_findings/_scripts/t06_symbolic_mixed_partial.py
    .venv/bin/python tasks_findings/_scripts/t06_numerical_cross_check.py
    .venv/bin/python tasks_findings/_scripts/t06_improvement_cycle.py

All three complete in seconds on the login node (no compute claim
required).

---

## 1. Narrative role in the paper

The proposition serves a single expository function: it forecloses the
reviewer question "why do you search Nash equilibria on a discrete grid
instead of minimising a scalar potential, as Pejo et al.\
(2019) did for the Price-of-Privacy?" The answer is that the PEG's shared
mechanism state $(\sigma, p_i)$ couples the players' costs, and the
coupling breaks not just exact potentials but weighted and ordinal
potentials as well. Once this is stated as a proposition rather than as
prose, the paper can pivot cleanly from "no potential is available" to
the smoothness / supermodular / discretised-search route it actually
uses (`prop:existence`, the C1-C7 numerical NE catalogue). The route
matters: the companion paper's abstract-condition treatment of iDP
(`2601.12922v1.pdf`, sec. 3) similarly opens with a "what breaks" claim
before switching to what does work; the T06 proposition is the
game-theoretic analogue for the PEG.

Three concrete places in the manuscript pivot on this proposition:

1. `paper/tex/03_game.tex:110` -- currently asserts informally that PoP's
   separable cost made it a potential game, and the PEG's coupled cost
   does not.
2. `paper/tex/04_analysis.tex` (after line 161) -- the natural home for
   the proposition, immediately after the FOC discussion and before the
   coalition-Stackelberg subsection.
3. `paper/tex/07_discussion.tex:64-68` -- currently poses "Is the PEG a
   potential game?" as an open question. The T06 rewrite replaces this
   with a forward reference and a narrower residual question about
   best-reply potentials.

## 2. What the proposition rules out and what it does not

Following Monderer & Shapley (1996), we distinguish four increasingly
weak notions of potential structure for a strategic-form game with
continuously differentiable payoffs on a smooth action space:

| Notion             | Definition                                                                                                | Sufficient obstruction                        |
|--------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| Exact potential    | $\exists\,\Phi$ with $u_i(s) - u_i(s') = \Phi(s) - \Phi(s')$ for every unilateral deviation.              | $\partial^2 u_i/\partial\varepsilon_i\partial\varepsilon_j \neq \partial^2 u_j/\partial\varepsilon_i\partial\varepsilon_j$ at some interior profile.       |
| Weighted potential | Same as above with $u_i - u_i' = \eta_i(\Phi - \Phi')$ for some fixed positive weights $\{\eta_i\}$.       | $\eta_j\partial^2 u_i/\dots \neq \eta_i\partial^2 u_j/\dots$ for every choice of positive $\{\eta_i\}$ at some profile. |
| Ordinal potential  | $\exists\,\Phi$ with $\mathrm{sign}(u_i - u_i') = \mathrm{sign}(\Phi - \Phi')$ for every deviation.       | Existence of a *strict improvement cycle*; equivalent to the failure of Monderer-Shapley's finite-improvement property. |
| Best-reply potential | $\exists\,\Phi$ that preserves the best-reply correspondence (Voorneveld 2000).                        | Not addressed here; residual open question.   |

The T06 proposition rules out exact, weighted, and ordinal. It leaves
open the best-reply potential question, which the discussion rewrite
carries as the sole residual.

## 3. Proposition (LaTeX-ready)

Insertion point: `paper/tex/04_analysis.tex`, end of the "Existence and
structure of pure Nash equilibria" subsection, immediately before the
`\subsection{Coalition Stackelberg and the attack recovery}` header.

```latex
\paragraph{Non-potential structure.}
The Pejo et al.\ Price-of-Privacy game~\cite{pejo2019together} was a
potential game because its privacy cost was separable across players,
reducing equilibrium computation to minimising a single scalar
potential. The PEG's cost, coupled through the mechanism's shared
$(\sigma, p_i)$, breaks that reduction.

\begin{proposition}[Non-potential game]
\label{prop:non-potential}
Under the utility form \eqref{eq:utility} and the mechanism of
\cref{def:sampling-idp}, the PEG is neither an exact, weighted, nor
ordinal potential game in the sense of \citet{monderer1996potential}:
\begin{enumerate}
  \item[\rm(i)] there exist no positive weights $\{\eta_i\}_{i=1}^n$
        and no twice-differentiable function
        $\Phi:\prod_i[\varepsilon^{\min},\varepsilon^{\max}]\to\mathbb R$
        such that
        $u_i(\varepsilon_i,\boldsymbol\varepsilon_{-i}) -
         u_i(\varepsilon_i',\boldsymbol\varepsilon_{-i}) =
         \eta_i\bigl(\Phi(\varepsilon_i,\boldsymbol\varepsilon_{-i}) -
         \Phi(\varepsilon_i',\boldsymbol\varepsilon_{-i})\bigr)$
        on the interior of the feasible region;
  \item[\rm(ii)] and there exists a length-four cycle of strict
        unilateral improvements
        $\varepsilon^{(0)}\to\varepsilon^{(1)}\to
         \varepsilon^{(2)}\to\varepsilon^{(3)}\to\varepsilon^{(0)}$
        on the discretised feasible grid, precluding any ordinal
        potential (Monderer-Shapley, Lemma~2.5).
\end{enumerate}
\end{proposition}

The proof is in \cref{app:non-potential} and combines a closed-form
Monderer-Shapley cross-partial calculation (rules out (i)) with an
explicit improvement cycle extracted from the C1 utility grid (rules
out (ii)). \Cref{prop:non-potential} explains why our equilibrium
analysis proceeds by discretised NE search (\cref{prop:existence})
rather than by minimising a closed-form potential: neither
Monderer-Shapley best-response dynamics nor the finite-improvement
property is available.
```

Note the strategic framing: the proposition motivates the *route*
(discretised NE search), not the *finding*. This mirrors the companion
paper's habit of introducing each theorem with the machinery it
unlocks (`2601.12922v1.pdf`, sec. 3.2).

## 4. Proofs (LaTeX-ready)

Insertion point: `paper/tex/08_appendix.tex`, new section
`\section{Non-potential structure of the PEG}\label{app:non-potential}`,
appended after `app:sensitivity`.

### 4.1 Notation and setup

Following the externality lemma of `paper/tex/04_analysis.tex:29-53`,
the mechanism satisfies the implicit relation

    sigma(eps) * sum_g |G_g| * r_g(eps_g)  =  E_b * sqrt(2 I alpha),
                                                           (eq:sigma-implicit)

where $r_g(x) := \sqrt{\rho_g(x, \delta)}$ is the Renyi-DP profile of
group $g$ evaluated at budget $x$, and $E_b\sqrt{2I\alpha}$ is the
mechanism-independent normalisation constant. Every per-group sample
rate is then $p_g = \sigma\, r_g(\varepsilon_g)/\sqrt{2I\alpha}$. Utilities
are $u_i(\boldsymbol\varepsilon) = V(\sigma(\boldsymbol\varepsilon)) - w_i\, A_i(\sigma, p_i)$
with $V(x) = 1/(1+x^2)$ and $A_i$ the MIA advantage on group $i$.

Write $R := \sum_g |G_g|\, r_g$, $\sigma_j := \partial\sigma/\partial\varepsilon_j$,
$\sigma_{ij} := \partial^2\sigma/(\partial\varepsilon_i\partial\varepsilon_j)$,
and analogously $p_{k,j}$, $p_{k,ij}$ for the sample rate of group $k$.
Fix a symmetric reference profile $\varepsilon_1=\dots=\varepsilon_n=\varepsilon^\star$
in the interior of the feasible region and write $r := r_g(\varepsilon^\star)$,
$r' := r_g'(\varepsilon^\star)$ (the functional form of $r_g$ makes both
group-independent when budgets coincide).

### 4.2 Lemma: shared-utility cross-partial

```latex
\begin{lemma}[Shared-utility cross-partial]
\label{lem:cross-partial-V}
At any interior profile,
\[
\frac{\partial^2 V(\sigma(\boldsymbol\varepsilon))}
     {\partial\varepsilon_i\,\partial\varepsilon_j}
 = V''(\sigma)\,\sigma_i\sigma_j + V'(\sigma)\,\sigma_{ij},
\]
and this expression is symmetric under $i\leftrightarrow j$. In
particular, the $V$-contribution appears identically on both sides of the
Monderer-Shapley symmetry condition and cancels.
\end{lemma}
```

**Proof.** Implicit differentiation of `eq:sigma-implicit` gives
$\sigma_j = -\sigma\, |G_j|\, r_j'/R$. Differentiating again in
$\varepsilon_i$ (with $i\neq j$) and substituting $\sigma_i = -\sigma\,|G_i|\,r_i'/R$
into the resulting expression yields
$\sigma_{ij} = 2\sigma\, |G_i|\, |G_j|\, r_i'\, r_j'/R^2$.
Both $\sigma_i\sigma_j$ and $\sigma_{ij}$ are symmetric under
$i\leftrightarrow j$; multiplication by the scalar $V''(\sigma)$ and
$V'(\sigma)$ preserves the symmetry.  qed.

*Sympy verification:* `t06_symbolic_mixed_partial.py` derives
$\sigma_{12}$ from the implicit constraint and confirms

    sigma_{12} |_sym  =  2 * sig * G1 * G2 * rp^2 / (rr^2 * (G1 + G2)^2),

symmetric in $(G_1, G_2)$. A three-player run gives the same formula
with $N := G_1+G_2+G_3$ in place of $G_1+G_2$.

### 4.3 Lemma: privacy-cost cross-partial and its asymmetry

```latex
\begin{lemma}[Privacy-cost cross-partial]
\label{lem:cross-partial-A}
Using the shorthand $A_{i,\sigma} := \partial A_i/\partial\sigma$,
$A_{i,p} := \partial A_i/\partial p_i$ (and analogously for higher
orders), the cross-partial of $w_i\, A_i(\sigma, p_i)$ decomposes as
\[
\frac{\partial^2 [w_i A_i]}
     {\partial\varepsilon_i\,\partial\varepsilon_j}
= w_i\Bigl[
     A_{i,\sigma\sigma}\,\sigma_i\sigma_j
   + A_{i,pp}\,p_{i,i}\,p_{i,j}
   + A_{i,\sigma p}\bigl(\sigma_i p_{i,j} + \sigma_j p_{i,i}\bigr)
   + A_{i,\sigma}\,\sigma_{ij}
   + A_{i,p}\,p_{i,ij}
\Bigr].
\]
At the symmetric reference profile,
\[
p_{i,ij}\bigl|_{\rm sym}
 = \frac{\sigma\,(r')^{2}\,|G_j|\,\bigl(2|G_i| - N\bigr)}
        {r\, N^{2}\,\sqrt{2 I \alpha}},
\qquad N := \sum_g |G_g|,
\]
which fails the exchange symmetry $i\leftrightarrow j$ whenever
$|G_i|\neq |G_j|$.
\end{lemma>
```

**Proof.** For $i\neq j$,
$p_{i,j} = \sigma_j\, r_i/\sqrt{2I\alpha}$,
$p_{i,i} = (\sigma_i\, r_i + \sigma\, r_i')/\sqrt{2I\alpha}$, and
$p_{i,ij} = (\sigma_{ij}\, r_i + \sigma_j\, r_i')/\sqrt{2I\alpha}$
because $r_i(\varepsilon_i)$ has no $\varepsilon_j$-dependence.
Substituting $\sigma_{ij} = 2\sigma|G_i||G_j|(r')^2/(r N)^2$ and
$\sigma_j = -\sigma|G_j|r'/(rN)$ at the symmetric profile,

    p_{i,ij}|sym  =  (2 sigma |G_i| |G_j| (r')^2 / (r N)^2 * r
                     +  (- sigma |G_j| r' / (r N)) * r') / sqrt(2 I alpha)
                 =  sigma (r')^2 |G_j| (2|G_i| - N) / (r N^2 sqrt(2 I alpha)).

For the swapped pair $p_{j,ij}|_{\rm sym}$ the same computation gives
$\sigma(r')^2|G_i|(2|G_j|-N)/(rN^2\sqrt{2I\alpha})$. The two expressions
coincide iff $|G_i|(2|G_j|-N) = |G_j|(2|G_i|-N)$, i.e., iff
$|G_i| = |G_j|$.  qed.

*Sympy verification:* `t06_symbolic_mixed_partial.py` prints

    p_{1,12}|sym  =  G2 * rp^2 * sig * (G1 - G2) / (S * rr * (G1 + G2)^2),
    p_{2,12}|sym  = -G1 * rp^2 * sig * (G1 - G2) / (S * rr * (G1 + G2)^2),

confirming $2G_i - N = G_i - G_j$ at $N=G_1+G_2$ and the
$|G_i|\leftrightarrow |G_j|$ asymmetry.

### 4.4 Proof of Proposition part (i): no exact / weighted potential

Assume for contradiction that positive weights $\{\eta_i\}$ and $\Phi$
satisfy the weighted-potential identity. The Monderer-Shapley
cross-partial condition is then

    eta_j * d^2 u_i / (d eps_i d eps_j)  =  eta_i * d^2 u_j / (d eps_i d eps_j)   (MS-sym)

everywhere on the interior of the feasible region.  By
`lem:cross-partial-V`, the $V$-contribution cancels between the two
sides, leaving

    eta_j * w_i * d^2 A_i / (d eps_i d eps_j)  =  eta_i * w_j * d^2 A_j / (d eps_i d eps_j).   (*)

Under mechanism-exchange symmetry at the symmetric profile
$\varepsilon^\star$, the partial derivatives $A_{i,\sigma}$,
$A_{i,\sigma\sigma}$, $A_{i,pp}$, $A_{i,\sigma p}$, $A_{i,p}$ take a
group-independent value (write them $a_\sigma$, $a_{\sigma\sigma}$,
$a_{pp}$, $a_{\sigma p}$, $a_p$). Substituting the expressions of
`lem:cross-partial-A` into $(*)$ and collecting terms of order
$a_p$ -- the only order at which
$p_{i,ij}$ differs from $p_{j,ij}$ -- reduces the identity, at every
symmetric $\varepsilon^\star$, to

    eta_j * w_i * G_j * (2 G_i - N)  =  eta_i * w_j * G_i * (2 G_j - N).      (**)

Sympy factorisation of $(**)$ under the $\eta_1=\eta_2=1$ exact-potential
assumption gives coefficient
$-(G_1-G_2)(G_1 w_2 + G_2 w_1)$, which is nonzero whenever
$|G_1|\neq|G_2|$: rules out exact potentials on any heterogeneous-group
configuration. For weighted potentials, the residual left by cancelling
(**) also carries the $V$-part $(\eta_2-\eta_1)\,d^2 V$; because
$\sigma_i\sigma_j$ and $\sigma_{ij}$ are nonconstant on the feasible
interval as $\varepsilon^\star$ varies, no positive $(\eta_1,\eta_2)$
with $\eta_1\neq\eta_2$ can zero this residual uniformly. In the fully
symmetric restriction $|G_1|=|G_2|,\, w_1=w_2$ the algebra at symmetric
profiles alone does *not* rule out exact potential; part (ii) closes
that gap.  qed.

### 4.5 Proof of Proposition part (ii): no ordinal potential

Monderer-Shapley (1996, Lemma~2.5) shows that a game admits an ordinal
potential iff it has no strict improvement cycle. It therefore suffices
to exhibit one on any restriction of the PEG. On the C1 discretisation
(fully symmetric two-player configuration, $w_1=w_2=1$,
$|G_1|=|G_2|=5000$, 22-point log-spaced budget grid on
$[0.5,64]$), the search of `t06_improvement_cycle.py` returns the cycle

    (eps_1, eps_2) = (1.587, 5.040)
        --[player 1, +7.3e-7]-->  (2.000, 5.040)
        --[player 2, +1.5e-3]-->  (2.000, 4.000)
        --[player 1, +3.0e-3]-->  (1.587, 4.000)
        --[player 2, +2.8e-3]-->  (1.587, 5.040),

with all four deviations strictly improving for the deviating player.
Any ordinal potential $\Phi$ would strictly increase at each step,
yielding $\Phi(1.587,5.040) > \Phi(1.587,5.040)$, a contradiction.
qed.

Under the fully symmetric restriction where part (i)'s algebra at
symmetric profiles does not decide, the improvement cycle above closes
the gap; the two arguments together rule out the whole hierarchy of
cardinal and ordinal potentials.

## 5. Weighted-potential edge case

The natural weakening "is there a weighted potential on some
subregion?" is delimited by the algebraic condition
$|G_1|(2|G_2|-N) = |G_2|(2|G_1|-N)$ (i.e., $|G_1|=|G_2|$) together with
the vanishing of the $V$-mismatch $(\eta_2-\eta_1)\,d^2V$. The
two conditions coincide on the two-player fully symmetric case
$|G_1|=|G_2|$, $w_1=w_2$, $\eta_1=\eta_2$. Even on that submanifold,
part (ii)'s improvement cycle disqualifies an ordinal potential.
The residual, unresolved question is whether a *best-reply* potential
(Voorneveld 2000) exists on a sub-cone of budget space at which
the best-reply correspondence is monotone; the discussion rewrite
carries this question as the sole remaining open item.

## 6. Numerical corroboration

Insertion point: end of `app:non-potential` (optional; keeps the
appendix short if desired).

**Setup.** The C1 cache stores a $22\times 22\times 2$ utility array for
the fully symmetric two-player scenario. We compute the finite-
difference mixed partial

    D u_k (i, j) := (U[i+1,j+1,k] - U[i+1,j-1,k]
                     - U[i-1,j+1,k] + U[i-1,j-1,k])
                    / ((eps[i+1]-eps[i-1]) * (eps[j+1]-eps[j-1]))

for $k\in\{0,1\}$ (players 1 and 2). Under any weighted potential the
ratio $D u_1(i,j)/D u_2(i,j)$ would equal the constant $\eta_2/\eta_1$
at every interior feasible cell.

**Result (from `t06_numerical_cross_check.py`, on 222 interior
feasible cells):**

| $(\varepsilon_1, \varepsilon_2)$ | $D^2 u_1 / (\partial\varepsilon_1\partial\varepsilon_2)$ | $D^2 u_2 / (\partial\varepsilon_1\partial\varepsilon_2)$ | ratio |
|---|---|---|---|
| $(20.16,\, 10.08)$ | $-6.264 \times 10^{-4}$ | $-6.353 \times 10^{-6}$ | $+98.60$ |
| $(4.00,\, 1.59)$  | $-1.184 \times 10^{-2}$ | $+1.362 \times 10^{-4}$ | $-86.97$ |
| $(10.08,\, 40.32)$ | $+7.082 \times 10^{-4}$ | $+1.755 \times 10^{-5}$ | $+40.36$ |
| $(3.17,\, 6.35)$  | $+3.185 \times 10^{-3}$ | $+1.047 \times 10^{-4}$ | $+30.40$ |
| $(16.00,\, 8.00)$  | $-9.275 \times 10^{-4}$ | $+3.424 \times 10^{-5}$ | $-27.09$ |
| $(10.08,\, 4.00)$  | $-2.996 \times 10^{-3}$ | $+1.876 \times 10^{-4}$ | $-15.97$ |
| $(5.04,\, 10.08)$  | $+1.365 \times 10^{-3}$ | $+1.366 \times 10^{-4}$ | $+9.99$ |
| $(40.32,\, 16.00)$ | $-3.909 \times 10^{-4}$ | $-4.596 \times 10^{-5}$ | $+8.51$ |

Diagonal reference ($\varepsilon_1 = \varepsilon_2$, mechanism exchange
symmetry recovers $D^2 u_1 = D^2 u_2$ up to floating point):

| $\varepsilon$ | $D^2 u_1$ | $D^2 u_2$ | ratio |
|---|---|---|---|
| $0.794$ | $-8.788 \times 10^{-3}$ | $-8.788 \times 10^{-3}$ | $+1.0000$ |
| $2.000$ | $-3.162 \times 10^{-3}$ | $-3.162 \times 10^{-3}$ | $+1.0000$ |
| $5.040$ | $-3.837 \times 10^{-3}$ | $-3.837 \times 10^{-3}$ | $+1.0000$ |
| $12.699$ | $-1.273 \times 10^{-3}$ | $-1.273 \times 10^{-3}$ | $+1.0000$ |
| $32.000$ | $-1.365 \times 10^{-4}$ | $-1.365 \times 10^{-4}$ | $+1.0000$ |

**Interpretation.** Over the 202 off-diagonal feasible cells, 134 have a
positive ratio and 68 have a negative ratio; the extremes are $-86.97$
and $+98.60$. No pair of *positive* weights $(\eta_1,\eta_2)$ can make
$\eta_2/\eta_1$ take both signs simultaneously; the constant-ratio
requirement is falsified. Combined with the improvement cycle of
section 4.5, this is a self-contained numerical certificate that the
C1 discretisation admits no potential structure of any of the three
canonical kinds.

## 7. Paper-side edits (unchanged from prior plan; not executed here)

1. Insert `prop:non-potential` at `paper/tex/04_analysis.tex` end of the
   pure-NE subsection.
2. Insert `app:non-potential` at `paper/tex/08_appendix.tex` after
   `app:sensitivity`; two lemmas, then the proof of the proposition, then
   the numerical certificate (optional inline).
3. Rewrite `paper/tex/07_discussion.tex:64-68`:
   ```latex
   \item \Cref{prop:non-potential} rules out exact, weighted, and
   ordinal potentials for the PEG. Whether a \emph{best-reply
   potential} \citep{voorneveld2000best} exists on a sub-cone of budget
   space at which the best-reply correspondence is monotone is left
   open; a positive answer would recover the Monderer-Shapley
   best-response-dynamics toolkit.
   ```
   If `voorneveld2000best` is not added to `refs.bib`, drop the citation
   and shorten the second sentence.
4. Extend the parenthetical at `paper/tex/03_game.tex:110`:
   ```latex
   PoP~\cite{pejo2019together} (whose cost $c(p_i)$ was separable, hence
   a potential game; the PEG's coupled cost breaks that structure --
   \cref{prop:non-potential})
   ```
5. Sweep `grep -n "potential game\|monderer" paper/tex/*.tex` and update
   any remaining prose. Do not disturb the fictitious-play citation at
   `paper/tex/03_game.tex:167`.

## 8. Acceptance checklist

For the paper-side task (T06 execution), not this document.

- [ ] `paper/main.pdf` compiles cleanly (`pdflatex -interaction=nonstopmode`),
      no new warnings on the labels `prop:non-potential`,
      `app:non-potential`, `lem:cross-partial-V`, `lem:cross-partial-A`.
- [ ] `prop:non-potential` renders in `sec:analysis` immediately after
      the FOC discussion and before the coalition subsection.
- [ ] `app:non-potential` renders in the appendix after `app:sensitivity`,
      contains both lemmas, the exact / weighted proof, and the
      improvement-cycle certificate.
- [ ] `paper/tex/07_discussion.tex` no longer poses the potential-game
      question as open; the bullet points at `prop:non-potential`.
- [ ] `paper/tex/03_game.tex:110` parenthetical cross-references
      `prop:non-potential`.
- [ ] No regression: `lem:externality`, `prop:existence`,
      `prop:coalition-attack` still typeset and cross-reference correctly.
- [ ] The three T06 scripts under `tasks_findings/_scripts/` complete
      without assertion errors on the current `.venv`.
- [ ] `CLAUDE.md` Rule 0 respected: no "we previously formulated..."
      prose introduced anywhere in the paper by this task.
- [ ] `CLAUDE.md` Rule 3 respected: no compute run outside Slurm; the
      three scripts read the existing C1 cache and are login-node safe.

## 9. Anchors in the codebase

- `game/utility.py:56-57` -- definition of $u_i = V(\sigma) - w_i A_i$.
- `game/utility.py:42-53` -- $V(\sigma) = 1/(1+\sigma^2)$.
- `game/continuous.py:203-260` -- pipeline that produced
  `_cache_v2/C1.pkl`.
- `game_experiments/run_continuous.py:56-67` -- `make_2player`, the
  fully symmetric two-player scenario used by C1.
- `game_experiments/_cache_v2/C1.pkl` -- source of the numerical
  certificates in sections 4.5 and 6.
- `paper/tex/04_analysis.tex:29-65` -- externality-lemma proof;
  source of `eq:sigma-implicit`.
- `paper/tex/03_game.tex:110` -- current informal reference.
- `paper/tex/07_discussion.tex:64-68` -- current open-question phrasing.
- `tasks_findings/_scripts/t06_symbolic_mixed_partial.py` -- sympy
  verification of `lem:cross-partial-V`, `lem:cross-partial-A`, and the
  MS-residual reduction, including a three-player sanity check.
- `tasks_findings/_scripts/t06_numerical_cross_check.py` -- finite-
  difference MS-sym falsification on the C1 cache.
- `tasks_findings/_scripts/t06_improvement_cycle.py` -- explicit
  4-cycle of strict unilateral improvements on the C1 cache.
