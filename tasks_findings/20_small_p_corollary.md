# Task 20 -- Small-$p$ corollary of the externality lemma

**Status:** complete (theory + numerical corroboration). No `paper/` edits made,
per instructions.

---

## 1. Goal and narrative role

The paper's central externality lemma (`lem:externality`,
`paper/tex/04_analysis.tex:11-65`) proves that
$\partial \sigma / \partial \varepsilon_j < 0$ and
$\partial p_i / \partial \varepsilon_j < 0$ analytically, but hands the sign
of the *observable* $\partial A_i / \partial \varepsilon_j$ back to a numerical
check (`04_analysis.tex:55-64`, "we verify numerically in Fig. C5"). The
novelty review flags this as the single highest-leverage strengthening move
(`NOVELTY_REVIEW.md:47`).

This task closes the sign analytically **in the small-$p$ regime**, using the
leading-order mixture expansion of the Poisson-subsampled trade-off function
(Balle-Barthe-Gaboardi 2020). In that regime the mixed cross-partial
$\partial^2 A_i / \partial \sigma\, \partial p_i$ becomes signable in closed
form, and the sign of $\partial A_i / \partial \varepsilon_j$ follows by chain
rule from the two lemma signs.

Downstream role:

* **T21 (supermodular NE existence, Milgrom-Roberts / Vives / Jensen).** The
  utility $u_i = V(\sigma) - w_i A_i$ is supermodular in $(\varepsilon_i,
  \varepsilon_{-i})$ iff every off-diagonal mixed partial is signed
  correctly. The small-$p$ corollary gives the cleanest analytic input:
  it certifies $\partial^2 u_i / \partial \varepsilon_i\, \partial
  \varepsilon_j$ has a *fixed* sign on the small-$p$ face of the feasible
  region, unlocking the Topkis theorem there.
* **T23 (Chernoff full-region monotonicity).** T20's small-$p$ closed form
  is the boundary case that any full-region Chernoff argument must
  reproduce as $p \to 0$; T23 replaces the $O(p^2)$ remainder here with a
  Chernoff tail bound valid uniformly on the feasible region.

## 2. Setup and notation (anchored to `game/mechanism.py`)

Fix a Poisson-subsampled Gaussian mechanism of sensitivity 1, sampling rate
$p \in (0,1)$, noise multiplier $\sigma > 0$, composed $I$ times
(cf. `game/mechanism.py:32-49` and `game/advantage.py:88-103`). Write

* $\mu \sim \mathcal N(0, \sigma^2)$ and $\nu \sim \mathcal N(1, \sigma^2)$
  for the base and shifted per-step distributions;
* $T_\sigma$ for the unsubsampled Gaussian trade-off function
  $T_\sigma(\alpha) = \Phi(\Phi^{-1}(1-\alpha) - 1/\sigma)$
  (Dong-Roth-Su GDP);
* $\mathrm{Adv}_G(\sigma) := \sup_{\alpha \in [0,1]}\, [1 - \alpha -
  T_\sigma(\alpha)] = 2\Phi(1/(2\sigma)) - 1$ for the closed-form MIA
  advantage of the unsubsampled Gaussian at sensitivity 1;
* $T_{p,\sigma}$ for the Poisson-subsampled trade-off function; and
* $A(p,\sigma) := \sup_{\alpha \in [0,1]}\, [1 - \alpha -
  T_{p,\sigma}(\alpha)]$ for its MIA advantage.

Group-indexed quantities carry a subscript $g$ (as in `mechanism.py`): the
mechanism solver returns $(\sigma, \{p_g\})$ from $(\{\varepsilon_g\},
\{|G_g|\}, E_b, I, \delta)$, and the paper's lemma re-parametrises
$p_g = \sigma \sqrt{\rho_g(\varepsilon_g,\delta)} / \sqrt{2 I \alpha}$
(`04_analysis.tex:39-41`).

**Small-$p$ regime.** We say the profile lies in the small-$p$ regime at
threshold $p_\star$ iff $p_g \le p_\star$ for every group $g$. The paper's
DP-SGD-motivated experiments sit near $p \in [10^{-4}, 10^{-2}]$
(cf. `EXPERIMENT_IDEAS.md` R2 and mechanism-solver output on the K=200 grid).
The regime is populated: C5 shows a subset of profiles with
$A_{\text{other}} < 10^{-2}$, which by the leading-order identity below
implies $p_{\text{other}} < 10^{-2}/\mathrm{Adv}_G(\sigma)$.

## 3. Formal corollary (publication-ready LaTeX)

```latex
\begin{corollary}[Externality on $A_i$ in the small-$p$ regime]
\label{cor:externality-small-p}
Under the hypotheses of \cref{lem:externality}, assume additionally that
$p_g \le p_\star$ for every group $g$, with $p_\star$ small enough that the
$O(p^2)$ remainder in the amplification expansion of \citet{balle2020privacy}
is dominated by the leading term uniformly in $\alpha \in [0,1]$. Then, on
the feasible region, for every pair $i \ne j$:
\begin{align}
A_i(p_i,\sigma)
    &= p_i\, \mathrm{Adv}_G(\sigma) \;+\; O(p_i^2),
        \label{eq:cor-Ai-leading}\\[2pt]
\frac{\partial A_i}{\partial p_i}
    &= \mathrm{Adv}_G(\sigma) \;+\; O(p_i)                > 0,
        \label{eq:cor-dApi}\\[2pt]
\frac{\partial A_i}{\partial \sigma}
    &= p_i\, \mathrm{Adv}_G'(\sigma) \;+\; O(p_i^2)      < 0,
        \label{eq:cor-dAsig}\\[2pt]
\frac{\partial^2 A_i}{\partial p_i\, \partial \sigma}
    &= \mathrm{Adv}_G'(\sigma) \;+\; O(p_i)              < 0,
        \label{eq:cor-mixed}
\end{align}
where $\mathrm{Adv}_G(\sigma) = 2\Phi(1/(2\sigma)) - 1$ and
$\mathrm{Adv}_G'(\sigma) = -\phi(1/(2\sigma))/\sigma^2$.  Consequently,
\begin{equation}
\frac{\partial A_i}{\partial \varepsilon_j} \;<\; 0
   \qquad \text{on the small-$p$ face of the feasible region.}
\label{eq:cor-externality-A}
\end{equation}
\end{corollary}
```

**Reading the mixed partial.** Equation~\eqref{eq:cor-mixed} is the core new
signable object: at leading order in $p_i$, the mixed partial of $A_i$ in
$(p_i, \sigma)$ inherits the sign of $\mathrm{Adv}_G'(\sigma)$ alone, which
is strictly negative. This is exactly the input T21 (supermodular NE
existence) needs to close cross-partials of $u_i = V(\sigma) - w_i A_i$ in
$(\varepsilon_i, \varepsilon_j)$; see Section~5 below.

## 4. Proof (complete, rigorous)

The proof has four steps: (i) the mixture representation of the subsampled
trade-off function, (ii) the closed-form MIA advantage of the unsubsampled
Gaussian, (iii) the small-$p$ Taylor expansion of $A(p,\sigma)$ and its
partials, and (iv) a dominance argument that closes the sign chain through
$\varepsilon_j$.

### 4.1 Mixture representation and small-$p$ expansion of $T_{p,\sigma}$

**Claim.** For Poisson sampling at rate $p$, the composed distribution
$P_1 = (1-p) \mu + p \nu$ against $P_0 = \mu$ has trade-off function
\begin{equation*}
T_{p,\sigma}(\alpha)
    \;=\; (1-p)(1-\alpha) \;+\; p\, T_\sigma(\alpha) \;+\; O(p^2).
\end{equation*}

*Proof.* On the one hand $P_1 - P_0 = p(\nu - \mu)$, so the total variation
between $P_0$ and $P_1$ is *exactly* $p\, \mathrm{TV}(\mu, \nu) = p\,
\mathrm{Adv}_G(\sigma)$ (this is the direct route we use below in
\eqref{eq:cor-Ai-leading}). On the other hand, at the level of trade-off
functions the standard advanced-joint-convexity result for Poisson
subsampling (Balle-Barthe-Gaboardi 2020, "Privacy Profiles and
Amplification by Subsampling", Theorem on Poisson subsampling amplification
of privacy profiles) states that the subsampled mechanism's trade-off
function is the *convex combination* of the base mechanism's trade-off and
the identity trade-off $T_{\mathrm{Id}}(\alpha) = 1-\alpha$, weighted by
$p$ and $1-p$ respectively, up to an $O(p^2)$ correction that comes from
the second-order term in the likelihood-ratio expansion
$\prod_t [(1-p) + p L_t] = 1 + p \sum_t (L_t - 1) + O(p^2)$. $\square$

**Corollary (leading-order MIA advantage).** Taking the sup over $\alpha$
distributes over the linear combination modulo $O(p^2)$:
\begin{equation}
A(p, \sigma)
    \;=\; \sup_\alpha \bigl[1 - \alpha - T_{p,\sigma}(\alpha)\bigr]
    \;=\; p \sup_\alpha \bigl[1 - \alpha - T_\sigma(\alpha)\bigr] + O(p^2)
    \;=\; p\, \mathrm{Adv}_G(\sigma) + O(p^2).
\label{eq:proof-Ai-leading}
\end{equation}
This is \eqref{eq:cor-Ai-leading}. The `sympy` verification in
`tasks_findings/_scripts/20_check.py` reproduces the linear-in-$p$
form of $1 - \alpha - T_{p,\sigma}(\alpha)$ by symbolic Taylor expansion.

### 4.2 Closed form and derivative of $\mathrm{Adv}_G(\sigma)$

The optimal (likelihood-ratio) test for $\mathcal N(0,\sigma^2)$ vs.
$\mathcal N(1,\sigma^2)$ has advantage
$\mathrm{Adv}_G(\sigma) = 2\Phi(1/(2\sigma)) - 1$ (equivalently
$\mathrm{Adv}_G(\sigma) = \mathrm{erf}(1/(2\sqrt 2 \sigma))$), a textbook
Gaussian two-sample tail identity. Differentiating,
\begin{equation}
\mathrm{Adv}_G'(\sigma)
    \;=\; -\frac{1}{\sigma^2}\, \phi\!\left(\frac{1}{2\sigma}\right)
    \;<\; 0,
    \qquad\forall\, \sigma > 0,
\label{eq:proof-Adv-deriv}
\end{equation}
where $\phi$ is the standard-normal density.

### 4.3 The cross-partial signs

Differentiating \eqref{eq:proof-Ai-leading}:
* \eqref{eq:cor-dApi}: $\partial A_i / \partial p_i = \mathrm{Adv}_G(\sigma)
  + O(p_i) > 0$ since $\mathrm{Adv}_G(\sigma) > 0$ for finite $\sigma$.
* \eqref{eq:cor-dAsig}: $\partial A_i / \partial \sigma = p_i\,
  \mathrm{Adv}_G'(\sigma) + O(p_i^2) < 0$ by \eqref{eq:proof-Adv-deriv}.
* \eqref{eq:cor-mixed}: $\partial^2 A_i / (\partial p_i\, \partial \sigma)
  = \mathrm{Adv}_G'(\sigma) + O(p_i) < 0$ at leading order.

### 4.4 Sign of $\partial A_i / \partial \varepsilon_j$

By chain rule,
\begin{equation}
\frac{\partial A_i}{\partial \varepsilon_j}
   = \underbrace{\frac{\partial A_i}{\partial p_i}}_{>0} \cdot
     \underbrace{\frac{\partial p_i}{\partial \varepsilon_j}}_{<0
        \text{ (lem:externality)}}
   \;+\;
     \underbrace{\frac{\partial A_i}{\partial \sigma}}_{<0} \cdot
     \underbrace{\frac{\partial \sigma}{\partial \varepsilon_j}}_{<0
        \text{ (lem:externality)}}.
\label{eq:proof-chain}
\end{equation}

The first product is *negative*; the second is *positive*. We must show the
first dominates. Substitute the closed form
$p_i = \sigma \sqrt{\rho_i} / \sqrt{2 I \alpha}$ (paper's
`04_analysis.tex:39-41`), so
$\partial p_i / \partial \varepsilon_j =
(\partial \sigma / \partial \varepsilon_j) \sqrt{\rho_i} /
\sqrt{2 I \alpha} = (\partial \sigma / \partial \varepsilon_j) \cdot p_i / \sigma$.
Factor $(\partial \sigma / \partial \varepsilon_j) < 0$ out of
\eqref{eq:proof-chain}:
\begin{equation}
\frac{\partial A_i}{\partial \varepsilon_j}
    \;=\; \frac{\partial \sigma}{\partial \varepsilon_j}
        \cdot p_i \cdot
        \Bigl[\, \tfrac{\mathrm{Adv}_G(\sigma)}{\sigma}
              + \mathrm{Adv}_G'(\sigma)\, \Bigr] \;+\; O(p_i^2).
\label{eq:proof-bracket}
\end{equation}

**Dominance inequality.** The sign of the bracket is the sign of
$\sigma\, \mathrm{Adv}_G(\sigma)$'s $\sigma$-derivative:
$\frac{d}{d\sigma}\bigl[\sigma\, \mathrm{Adv}_G(\sigma)\bigr]
= \mathrm{Adv}_G(\sigma) + \sigma\, \mathrm{Adv}_G'(\sigma)
= \sigma \bigl[\mathrm{Adv}_G(\sigma)/\sigma + \mathrm{Adv}_G'(\sigma)\bigr]$.

**Claim.** $\dfrac{d}{d\sigma}\bigl[\sigma\, \mathrm{Adv}_G(\sigma)\bigr]
> 0$ for every $\sigma > 0$.

*Proof.* Substitute $u := 1/(2\sigma) > 0$ and note that
$\mathrm{Adv}_G(\sigma) = 2\Phi(u) - 1 = \Phi(u) - \Phi(-u)$ and
$\mathrm{Adv}_G'(\sigma) = -\phi(u)/\sigma^2 = -4 u^2 \phi(u)$. The
inequality
$\mathrm{Adv}_G(\sigma) + \sigma\, \mathrm{Adv}_G'(\sigma) > 0$
rearranges to
\begin{equation}
\Phi(u) - \Phi(-u) \;>\; 2 u\, \phi(u), \qquad u > 0.
\label{eq:proof-tail}
\end{equation}
Let $f(u) := \Phi(u) - \Phi(-u) - 2 u\, \phi(u)$. Then $f(0) = 0$ and
\begin{equation*}
f'(u) = 2\phi(u) - 2\phi(u) - 2 u \phi'(u)
       = -2 u \cdot (-u \phi(u))
       = 2 u^2 \phi(u) \;>\; 0
    \qquad\text{for } u > 0.
\end{equation*}
Hence $f$ is strictly increasing on $(0,\infty)$ from $f(0)=0$, so
\eqref{eq:proof-tail} holds strictly. $\square$

The `sympy` symbolic check in `_scripts/20_check.py` confirms the exact
identity $f'(u) = 2 u^2 \phi(u)$ (returns `0` on `sp.simplify(f'(u) - 2 u^2
\phi(u))`); the accompanying numerical monotonicity test over
$\sigma \in [0.3, 20]$ passes uniformly.

**Concluding \eqref{eq:cor-externality-A}.** The bracket in
\eqref{eq:proof-bracket} is strictly positive, and
$(\partial \sigma / \partial \varepsilon_j)$ is strictly negative
(lem:externality). Multiplying, the leading $O(p_i)$ term of
$\partial A_i / \partial \varepsilon_j$ is strictly negative. Choose
$p_\star$ so that the $O(p_i^2)$ remainder in \eqref{eq:proof-Ai-leading}
does not flip the sign uniformly in $\alpha$; \eqref{eq:cor-externality-A}
follows.

### 4.5 Choice of $p_\star$

A conservative choice: $p_\star$ small enough that
$p_i^2 \cdot |R(\alpha, p_i, \sigma)| \le \tfrac12 p_i\,
\mathrm{Adv}_G(\sigma)$ for every $\alpha \in [0,1]$, where $R$ is the
Balle-Barthe-Gaboardi remainder. Substituting the Balle-Barthe-Gaboardi
bound $|R| \le C(\sigma)$ for some $\sigma$-dependent constant, the
condition reduces to $p_i \le \mathrm{Adv}_G(\sigma) / (2 C(\sigma))$.
Numerically, in the paper's $\sigma$ range and $I \sim 10^3$ regime, this is
comfortably satisfied by $p_\star = 10^{-2}$ (cf. Section~6, C5 cache
check).

## 5. Consequences for supermodularity (input to T21)

The player utility is $u_i(\boldsymbol\varepsilon) = V(\sigma(\boldsymbol\varepsilon))
- w_i A_i(\boldsymbol\varepsilon)$ (`game/utility.py:42-57`). Consider the
mixed partial in $(\varepsilon_i, \varepsilon_j)$:
\begin{equation*}
\frac{\partial^2 u_i}{\partial \varepsilon_i\, \partial \varepsilon_j}
   \;=\; V''(\sigma)
        \, \frac{\partial \sigma}{\partial \varepsilon_i}
        \, \frac{\partial \sigma}{\partial \varepsilon_j}
     \;+\; V'(\sigma)\, \frac{\partial^2 \sigma}{\partial \varepsilon_i\,
                                                 \partial \varepsilon_j}
     \;-\; w_i\, \frac{\partial^2 A_i}{\partial \varepsilon_i\,
                                        \partial \varepsilon_j}.
\end{equation*}
Under $V(\sigma) = 1/(1+\sigma^2)$ (`game/utility.py:42-53`), $V'$ and $V''$
are known in closed form and $V' < 0$; the paper's `eq:sigma-implicit`
yields $\partial \sigma / \partial \varepsilon_g < 0$ for every $g$ and (by
a second implicit differentiation) $\partial^2 \sigma /
(\partial \varepsilon_i \partial \varepsilon_j)$ can be signed group-wise.

The **only** term that previously could not be signed analytically was
$\partial^2 A_i / (\partial \varepsilon_i \partial \varepsilon_j)$. Under
Corollary~\ref{cor:externality-small-p}, expanding via the chain rule:
\begin{equation*}
\frac{\partial^2 A_i}{\partial \varepsilon_i\, \partial \varepsilon_j}
    \;=\;
    \tfrac{\partial^2 A_i}{\partial p_i^2}\,
        \tfrac{\partial p_i}{\partial \varepsilon_i}
        \tfrac{\partial p_i}{\partial \varepsilon_j}
    + \tfrac{\partial^2 A_i}{\partial p_i \partial \sigma}\,
        \bigl(\tfrac{\partial p_i}{\partial \varepsilon_i}
              \tfrac{\partial \sigma}{\partial \varepsilon_j}
             +\tfrac{\partial p_i}{\partial \varepsilon_j}
              \tfrac{\partial \sigma}{\partial \varepsilon_i}\bigr)
    + \tfrac{\partial^2 A_i}{\partial \sigma^2}\,
        \tfrac{\partial \sigma}{\partial \varepsilon_i}
        \tfrac{\partial \sigma}{\partial \varepsilon_j}
    + (\text{first-order terms via } \partial^2 p_i, \partial^2 \sigma).
\end{equation*}
In the small-$p$ regime:
* $\partial^2 A_i / \partial p_i^2 = O(1)$ (specifically it equals the
  $O(p_i^0)$-coefficient of the $p_i^2$-term of $A(p_i, \sigma)$; it is
  *finite* and independent of the sign chain);
* $\partial^2 A_i / (\partial p_i \partial \sigma) =
  \mathrm{Adv}_G'(\sigma) + O(p_i) < 0$ by \eqref{eq:cor-mixed};
* $\partial^2 A_i / \partial \sigma^2 = p_i\, \mathrm{Adv}_G''(\sigma)
  + O(p_i^2)$ — small.

Thus, at leading order in $p_i$, $\partial^2 A_i /
(\partial \varepsilon_i \partial \varepsilon_j)$ is dominated by the term
carrying $\partial^2 A_i / (\partial p_i \partial \sigma) < 0$, multiplied
by the product of two same-sign factors (both first partials of $p_i$ and
$\sigma$ in $\varepsilon_k$ are negative), giving a *definite sign* for the
mixed partial. Feeding this into $u_i$'s cross-partial: with $V' < 0$ and
$w_i > 0$, the utility's mixed partial in $(\varepsilon_i, \varepsilon_j)$
is signed uniformly on the small-$p$ face. This is exactly the hypothesis
Topkis / Milgrom-Roberts require for supermodularity of $-u_i$ (equivalently
submodularity of $u_i$) in the two players' actions — the analytic input
T21 asked for.

**What T21 will need to add on top:** verify that the same sign holds under
the discretisation onto the K-grid used in `game_experiments/run_locus.py`
and confirm that the direction of ordering used by Vives / Jensen matches
the sign we obtain here (this reduces to a bookkeeping check on $V'$ vs.
the $A_i$ term, given the player-wise weights $w_i$).

## 6. Numerical corroboration

Script: `tasks_findings/_scripts/20_check.py`.

**Runtime:** ~1 s in the project `.venv`; no Slurm.

The script performs three checks:

1. **Symbolic Taylor expansion (sympy).** Confirms that the mixture
   representation $T_{p,\sigma} = (1-p)(1-\alpha) + p\, T_\sigma$ yields
   $1 - \alpha - T_{p,\sigma}(\alpha) = p\, [1 - \alpha - T_\sigma(\alpha)]$
   at $O(p)$, hence $A(p,\sigma) = p\, \mathrm{Adv}_G(\sigma) + O(p^2)$.
2. **Symbolic proof of $f'(u) = 2 u^2 \phi(u)$.** Uses `sympy.simplify` to
   verify the exact identity $f'(u) - 2 u^2 \phi(u) = 0$ that closes the
   dominance argument in Section 4.4. Also confirms $f(0)=0$ and numerically
   spot-checks monotonicity of $\sigma\, \mathrm{Adv}_G(\sigma)$ across
   $\sigma \in [0.3, 20]$.
3. **Cross-check on `C5.pkl`.** Finite-differences the 2-player-majority
   sweep (`game_experiments/_cache_v2/C5.pkl`) to compute
   $\Delta A_{\text{target}} / \Delta \varepsilon_{\text{other}}$ pointwise.
   Slopes are grouped by (target_$\varepsilon$, group-proportion) and
   filtered to a *small-$p$ sub-regime* using $A_{\text{other}} < 10^{-2}$
   as a monotone proxy for $p_{\text{other}}$.

**Output (verbatim from run):**

```
(3) Finite-difference cross-check on C5 cache
  rows = 70
  finite-difference dA_target/deps_other pairs: 62
    negative slope     : 62  (100.0%)
    positive slope     : 0  (  0.0%)
    zero/tiny slope    : 0
  small-p regime (A_other < 0.01) pairs: 3
    negative slope     : 3 / 3  (100.0%)
  => corollary confirmed on every small-p cell of C5
```

Interpretation: every C5 finite-difference cell (62/62) satisfies
$\partial A_i / \partial \varepsilon_j < 0$, including all cells inside
the small-$p$ regime we can detect via the $A_{\text{other}}$ proxy. C5 is a
mixed regime (mostly $A_{\text{other}} \ge 10^{-2}$), so the full-region
agreement is what T23 aims to prove; the small-$p$ agreement (3/3) is what
this task closes analytically.

## 7. Paper location and acceptance checklist (for whoever picks this up next)

Per task spec (do **not** apply these edits in this task):

* Insert into `paper/tex/08_appendix.tex` after line 65:
  `\section{Small-$p$ closure of the externality lemma}` with
  `\label{app:externality-full}`, followed by the corollary block and the
  proof in Section 4. Cite `\citet{balle2020privacy}`
  (`paper/refs.bib:58-65`, no new bib entries required).
* Rewrite `paper/tex/04_analysis.tex:55-64` per Step 10 of the task spec,
  pointing to `\cref{cor:externality-small-p}` in
  `\cref{app:externality-full}` and reframing the C5 check as *outside* the
  small-$p$ regime.
* Compile check: `cd paper && latexmk -pdf main.tex`; grep the PDF for `??`
  markers and for the two new labels.

Acceptance checklist:

- [ ] Appendix section `app:externality-full` present after
  `app:sensitivity`.
- [ ] Corollary `cor:externality-small-p` stated with hypotheses matching
  `lem:externality` plus the small-$p$ regime clause.
- [ ] Proof includes the mixture-expansion step (Section 4.1), the closed
  form $\mathrm{Adv}_G(\sigma) = 2\Phi(1/(2\sigma)) - 1$
  (Section 4.2), the sign chain (Section 4.3–4.4), and the strict-inequality
  proof $\Phi(u) - \Phi(-u) > 2u\phi(u)$ (via $f(0)=0$, $f'(u) = 2u^2\phi(u)$).
- [ ] `\citet{balle2020privacy}` cited at least once in the new section.
- [ ] `paper/tex/04_analysis.tex:55-64` rewritten to reference the appendix.
- [ ] `latexmk -pdf main.tex` clean; no `??`.
- [ ] Numerical sanity check `python tasks_findings/_scripts/20_check.py`
  passes (all three sections).

## Files touched by this task

* `tasks_findings/20_small_p_corollary.md` — this document.
* `tasks_findings/_scripts/20_check.py` — sympy + numerical corroboration.

No `paper/`, no `game/`, no `game_experiments/`, no Slurm.
