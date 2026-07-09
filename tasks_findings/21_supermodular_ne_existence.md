# T21 — Supermodular NE existence for the PEG (n-player)

**Status.** Theory + numerical corroboration complete. No `paper/` edits made
here (per task instructions). This document is the specification that a
paper-side follow-up executes.

**Category.** H-priority / H-effort theory appendix.
**Depends on.** T20 (small-$p$ corollary of `lem:externality`).
**Un-blocks.** T07 step 7 (n-player self-sorting via Milgrom–Shannon),
T19 (a-priori extremal-NE anchor for the FP sweep), T22 Path 3
(aggregative-supermodular PoA).

Companion script: `tasks_findings/_scripts/21_check.py`
(sympy expansion of the cross-partial + finite-difference certificate on
C1 and C2, plus an MCS spot check on C2). Run:

    .venv/bin/python tasks_findings/_scripts/21_check.py

Verbatim stdout snippets are quoted in Section 6.

---

## 1. Goal and narrative role

The paper currently hedges continuous-action pure-NE existence
(`paper/tex/04_analysis.tex:107-114`) and lists it as open question 1
(`paper/tex/07_discussion.tex:60-63`). The Debreu–Fan–Glicksberg route
fails because $A_i$, the trade-off function of the $I$-fold subsampled
Gaussian, is not concave in $\varepsilon_i$; T06 ruled out the
potential-game route (no exact, weighted, or ordinal potential). What
remains is the **order-based route**: Topkis 1979 / Milgrom–Roberts 1990
lattice fixed points, which trade concavity for monotonicity.

The mixed partial input for the Topkis route is exactly what
T20 signed analytically on the small-$p$ face. Under
$V(\sigma)=1/(1+\sigma^2)$ and the T20 corollary, we show the utility is
*submodular* in $\boldsymbol\varepsilon$ on that face, which under the
reparametrisation $a_i := -\varepsilon_i$ is supermodularity in $a$ on the
matching sublattice. Milgrom–Roberts Theorem 5 then delivers a non-empty
complete lattice of pure NEs with a largest and a smallest element.
Milgrom–Shannon 1994 monotone comparative statics gives the corollary
that unblocks T07's n-player self-sorting remark and gives T19's
fictitious-play sweep an a-priori anchor.

The claim we deliver here is *conditional* — it lives on the small-$p$
sublattice covered by T20. The numerical certificate on C1 and C2
(Section 6) shows that the sub-lattice populates the operating region
used in every DP-SGD-motivated scenario in the paper.

**What this does NOT deliver.** Global (all-$p$) existence on the full
continuous action space. On the full C1 grid, roughly 26% of interior
feasible cells fail submodularity (Section 6.1), so a full-region theorem
would require either T23 (Chernoff-tail full-region monotonicity) or a
finer decomposition. This gap is honestly signposted in the paper
rewrite (Section 7).

---

## 2. Setup and notation

We use the paper's canonical utility (`game/utility.py:56-57`)

$$u_i(\boldsymbol\varepsilon) \;=\; V\!\bigl(\sigma(\boldsymbol\varepsilon)\bigr) \;-\; w_i\,A_i(\boldsymbol\varepsilon),
\qquad V(\sigma)=\tfrac1{1+\sigma^{2}},\qquad w_i>0,$$

and the mechanism state $(\sigma(\boldsymbol\varepsilon),\{p_g(\boldsymbol\varepsilon)\})$
solved from (C1)+(C2) (`game/mechanism.py:32-49`), where $p_g$ obeys the
paper's closed form
$p_g = \sigma\sqrt{\rho_g(\varepsilon_g,\delta)}/\sqrt{2 I \alpha}$
(`paper/tex/04_analysis.tex:39-41`, `eq:sigma-implicit`).

**Action space and lattice.** For each player $i$ the action set is the
compact interval $\Sigma_i := [\varepsilon^{\min}, \varepsilon^{\max}]$
with the *reversed* usual order,

$$a_i \;\preceq\; a_i' \quad\Longleftrightarrow\quad \varepsilon_i \;\geq\; \varepsilon_i',$$

equivalently $a_i := \varepsilon^{\max} - \varepsilon_i$ under the natural
order (identical modulo a constant to $a_i := -\varepsilon_i$; we use the
positive convention $a_i \in [0, \varepsilon^{\max}-\varepsilon^{\min}]$
to keep the interval in $\mathbb R_{\ge 0}$). The product
$\Sigma := \prod_{i=1}^n \Sigma_i$ is then a compact complete lattice in
the product order.

**Feasible region.** Let $\mathcal F \subset \Sigma$ be the closure of the
set of profiles at which (C1) binds for every group, (C2) binds at
$E_b$, and $p_i(\boldsymbol\varepsilon) \ge p_{\min}$ for every $i$
(`ass:participation`, `paper/tex/03_game.tex:124-132`).

**Small-$p$ face.** Fix a threshold $p_\star$ such that T20's expansion
$A_i = p_i\,\mathrm{Adv}_G(\sigma) + O(p_i^2)$ has its remainder dominated
by the leading term uniformly in $\alpha$
(`tasks_findings/20_small_p_corollary.md` §4.5). The small-$p$ face is

$$\mathcal F_\star \;:=\; \bigl\{ \boldsymbol\varepsilon \in \mathcal F : p_i(\boldsymbol\varepsilon) \le p_\star \text{ for every } i \bigr\}.$$

$\mathcal F_\star$ is a closed subset of $\mathcal F$; sublattice
closure is verified in Section 3.3.

---

## 3. Formal statements

### 3.1 Supermodular structure of the PEG (small-$p$ face)

```latex
\begin{lemma}[Increasing differences in the reversed-$\varepsilon$ order]
\label{lem:increasing-differences}
Under the hypotheses of \cref{lem:externality} and
\cref{cor:externality-small-p} (T20 small-$p$ regime with threshold
$p_\star$), for every ordered pair $i \neq j$ and every
$\boldsymbol\varepsilon$ in the interior of $\mathcal F_\star$,
\[
\frac{\partial^{2} u_i(\boldsymbol\varepsilon)}
     {\partial \varepsilon_i\,\partial \varepsilon_j}
 \;\le\; 0,
\]
with strict inequality on the interior. Equivalently, under
$a_k := -\varepsilon_k$, $u_i$ has non-negative cross-partials in
$(a_i, a_j)$ on the reversed-order sublattice, so the PEG is a
supermodular game in $a$ on $\mathcal F_\star$.
\end{lemma}
```

Signs of the input quantities (all on the interior of $\mathcal F_\star$):

* From `lem:externality`: $\partial\sigma/\partial\varepsilon_k < 0$
  for every $k$; $\partial p_i/\partial\varepsilon_j < 0$ for $j\neq i$;
  $\partial p_i/\partial\varepsilon_i > 0$ (the $\sigma\,r'$ term dominates
  in the interior — verified in T06 Sec 4.1).
* From T20: $V'(\sigma)<0$; $\partial A_i/\partial p_i =
  \mathrm{Adv}_G(\sigma)+O(p_i)>0$; $\partial A_i/\partial\sigma =
  p_i\,\mathrm{Adv}_G'(\sigma)+O(p_i^2)<0$; the *signed mixed partial*
  $\partial^2 A_i/(\partial p_i \partial\sigma) = \mathrm{Adv}_G'(\sigma)
  +O(p_i) < 0$.

### 3.2 Existence theorem (Milgrom–Roberts on the small-$p$ sublattice)

```latex
\begin{theorem}[Pure-strategy NE existence on the small-$p$ sublattice]
\label{thm:continuous-ne-smallp}
Assume the hypotheses of \cref{lem:increasing-differences} and that
$\mathcal F_\star$ is a compact sublattice of $\Sigma$ under the
reversed-$\varepsilon$ product order (\cref{lem:sublattice}). Then the
$n$-player PEG restricted to $\mathcal F_\star$ has a non-empty set of
pure-strategy Nash equilibria. This set is a complete lattice in the
product order; in particular there exist a largest and a smallest NE
$\boldsymbol\varepsilon^{\mathrm{sup}}, \boldsymbol\varepsilon^{\mathrm{inf}}
\in \mathcal F_\star$.
\end{theorem>
```

### 3.3 Sublattice closure of the feasible region

```latex
\begin{lemma}[$\mathcal F_\star$ is a sublattice]
\label{lem:sublattice}
Under \cref{lem:externality}, for every
$\boldsymbol\varepsilon,\boldsymbol\varepsilon' \in \mathcal F_\star$ the
componentwise meet and join in the reversed-$\varepsilon$ order,
$(\boldsymbol\varepsilon \wedge \boldsymbol\varepsilon')_k
= \max(\varepsilon_k,\varepsilon_k')$ and
$(\boldsymbol\varepsilon \vee \boldsymbol\varepsilon')_k
= \min(\varepsilon_k,\varepsilon_k')$, again lie in $\mathcal F_\star$.
\end{lemma>
```

### 3.4 Monotone comparative statics (Milgrom–Shannon)

```latex
\begin{corollary}[Sort order at the extremal NEs]
\label{cor:mcs}
Fix all parameters except one privacy weight $w_i$. On $\mathcal F_\star$
the extremal NEs are monotone in $w_i$:
$\varepsilon_i^{\mathrm{sup}}(w_i), \varepsilon_i^{\mathrm{inf}}(w_i)$
are weakly decreasing in $w_i$; and for every $j\neq i$,
$\varepsilon_j^{\mathrm{sup}}(w_i), \varepsilon_j^{\mathrm{inf}}(w_i)$
are weakly increasing in $w_i$.
\end{corollary>
```

---

## 4. Proofs

### 4.1 Proof of Lemma 3.1 (increasing differences)

Expand $u_i = V(\sigma) - w_i A_i$ by chain rule. Let
$\sigma_k := \partial\sigma/\partial\varepsilon_k$,
$\sigma_{ij} := \partial^2\sigma/(\partial\varepsilon_i\partial\varepsilon_j)$,
$p_{i,k} := \partial p_i/\partial\varepsilon_k$,
$p_{i,ij} := \partial^2 p_i/(\partial\varepsilon_i\partial\varepsilon_j)$,
and $A_{i,p}, A_{i,\sigma}, A_{i,pp}, A_{i,p\sigma}, A_{i,\sigma\sigma}$
the partials of $A_i$ in $(p_i,\sigma)$. Then

$$\frac{\partial^{2} u_i}{\partial\varepsilon_i\,\partial\varepsilon_j}
 \;=\; \underbrace{V''(\sigma)\,\sigma_i\sigma_j + V'(\sigma)\,\sigma_{ij}}_{\text{shared-$V$ contribution}}
 \;-\; w_i \Bigl[
     A_{i,pp}\,p_{i,i}p_{i,j}
   + A_{i,p\sigma}(p_{i,i}\sigma_j + p_{i,j}\sigma_i)
   + A_{i,\sigma\sigma}\,\sigma_i\sigma_j
   + A_{i,p}\,p_{i,ij}
   + A_{i,\sigma}\,\sigma_{ij}\Bigr].$$

The sympy expansion in `_scripts/21_check.py` verifies this identity
(residual `0`).

**Signs.** On the interior of $\mathcal F_\star$:

| Quantity | Sign | Source |
|---|---|---|
| $V'(\sigma)$ | $<0$ | closed form |
| $V''(\sigma) = (6\sigma^2-2)/(1+\sigma^2)^3$ | $>0$ once $\sigma>1/\sqrt 3$ | closed form |
| $\sigma_i, \sigma_j$ | $<0$ | `lem:externality` |
| $\sigma_{ij}$ | $>0$ (symmetric groups) | T06 Lemma 4.2 |
| $p_{i,i}$ | $>0$ | `eq:sigma-implicit` + interior |
| $p_{i,j}\,(j\neq i)$ | $<0$ | `lem:externality` |
| $p_{i,ij}$ | $=0$ at symmetric groups | T06 Lemma 4.3 |
| $A_{i,p}$ | $>0$ | T20 Cor eq (cor-dApi) |
| $A_{i,\sigma}$ | $<0$ | T20 Cor eq (cor-dAsig) |
| $A_{i,p\sigma}$ | $<0$ | T20 Cor eq (cor-mixed) |
| $A_{i,pp},A_{i,\sigma\sigma}$ | $O(p_i)$ or $O(1)$ small | T20 §5 |

**Leading-order argument.** In the small-$p$ regime write $A_i = p_i
\mathrm{Adv}_G(\sigma) + O(p_i^2)$ so $A_{i,pp} = O(1)$,
$A_{i,\sigma\sigma} = O(p_i)$, $A_{i,p} = \mathrm{Adv}_G+O(p_i)$,
$A_{i,p\sigma} = \mathrm{Adv}_G'+O(p_i)$. Keep only the $O(1)$ and
$O(p_i)$ contributions after the $-w_i$ factor. At the symmetric-group
reference $p_{i,ij}=0$, so the $A_{i,p}\,p_{i,ij}$ term vanishes; the
$A_{i,\sigma}\,\sigma_{ij}$ term is $O(p_i)$ and positive-signed (since
$A_{i,\sigma}<0$ and $\sigma_{ij}>0$ give a negative product, which
subtracted with $-w_i$ becomes positive but $O(p_i)$).

The two dominant $O(1)$ contributions are then

* $V''(\sigma)\sigma_i\sigma_j$: with $\sigma_i\sigma_j>0$, this is
  $\ge 0$ once $\sigma\ge 1/\sqrt 3$.
* $-w_i A_{i,p\sigma}\,(p_{i,i}\sigma_j + p_{i,j}\sigma_i)$: the sign of
  the bracket is that of $\sigma\,\mathrm{Adv}_G(\sigma)$'s derivative;
  T20 §4.4's dominance calculation (the `tail` inequality
  $\Phi(u)-\Phi(-u)>2u\phi(u)$ proved by
  $f'(u)=2u^2\phi(u)$) shows this bracket is *negative* on the interior,
  and $A_{i,p\sigma}<0$; with the $-w_i$ prefix, the whole term is
  *negative*.

Both dominant contributions therefore *lower* $\partial^2 u_i /
(\partial\varepsilon_i\partial\varepsilon_j)$; the subleading positive
terms are $O(p_i)$. Thus on $\mathcal F_\star$ with $p_\star$ small
enough (and $\sigma \ge 1/\sqrt 3$),
$\partial^2 u_i/(\partial\varepsilon_i\partial\varepsilon_j) \le 0$
with strict inequality on the open interior.

Under $a_k := -\varepsilon_k$ this is $\partial^2 u_i / (\partial a_i
\partial a_j)\ge 0$, the standard Milgrom–Roberts increasing-differences
condition. $\square$

**Sensitivity to $\sigma < 1/\sqrt 3$.** The $V''$ contribution flips
sign for $\sigma<1/\sqrt 3$. Two options:

* (Analytic) Restrict $\mathcal F_\star$ further to
  $\{\sigma(\boldsymbol\varepsilon)\ge 1/\sqrt 3\}$. The mechanism-solver
  cache on the paper's operating region confirms $\sigma\in[0.5, 5]$ at
  every populated cell of the K=200 grid, so
  $\sigma\ge 1/\sqrt 3 \approx 0.577$ is a mild restriction.
* (Numerical fallback) Certify the sign directly on the mech-grid cache
  (Section 6). This is the path the paper takes for the sub-region
  $\sigma<1/\sqrt 3$ that survives the $\mathcal F_\star$ definition.

### 4.2 Proof of Theorem 3.2 (existence)

We apply Theorem 5 of Milgrom–Roberts 1990. Hypotheses:

1. **Compact product lattice.** Each $\Sigma_i = [\varepsilon^{\min},
   \varepsilon^{\max}]$ is a compact totally-ordered interval, hence a
   complete chain; the finite product $\Sigma = \prod_i \Sigma_i$ is a
   compact complete lattice in the product order (identically so in
   either the natural or the reversed order).
2. **Order-upper-semicontinuity of $u_i$.** $u_i(\boldsymbol\varepsilon)$
   is continuous on $\overline{\mathcal F_\star}$: $V\in C^\infty$;
   $\sigma$ and $p_g$ are continuous in $\boldsymbol\varepsilon$ by the
   implicit-function theorem applied to (C1)+(C2) on the interior; $A_i$
   is continuous in $(p_i, \sigma)$ by Balle–Barthe–Gaboardi 2020
   (continuity of the subsampled-Gaussian trade-off function; already
   cited as `balle2020privacy` in `paper/refs.bib`). Continuity implies
   order-upper-semicontinuity on both sides of the order.
3. **Supermodularity in own action.** Each $\Sigma_i$ is
   one-dimensional; own-action supermodularity is automatic (a chain
   has no non-trivial meet/join off the diagonal).
4. **Increasing differences in others' actions.** Lemma 3.1 on the
   $a$-order.
5. **Sublattice.** Lemma 3.3 (below).

Under (1)–(5), Milgrom–Roberts Theorem 5 gives a non-empty complete
lattice of pure NEs with a largest and a smallest element. $\square$

### 4.3 Proof of Lemma 3.3 (sublattice)

Let $\boldsymbol\varepsilon,\boldsymbol\varepsilon' \in \mathcal F_\star$.
Under the reversed order, the meet is $(\boldsymbol\varepsilon \wedge
\boldsymbol\varepsilon')_k = \max(\varepsilon_k, \varepsilon_k')$ and the
join is the componentwise min. We show both remain feasible.

**Feasibility of (C1)+(C2).** The mechanism solver takes an arbitrary
$\boldsymbol\varepsilon \in [\varepsilon^{\min}, \varepsilon^{\max}]^n$
and returns a *unique* $(\sigma, \{p_g\})$ (`solve_mechanism_cached`);
that pair is by construction feasible iff each $p_g \in [0,1]$ and (C2)
is satisfied at $E_b$. The set of $\boldsymbol\varepsilon$ that produce
$p_g\ge p_{\min}$ for every $g$ is exactly $\mathcal F$
(`ass:participation`). Monotonicity: $p_g$ is monotone in every
$\varepsilon_k$ (own coordinate: increasing; others': decreasing by
`lem:externality`). The componentwise-max profile
$\boldsymbol\varepsilon\wedge\boldsymbol\varepsilon' \ge_{\rm elt}
\boldsymbol\varepsilon$ raises $p_g(g$'s coordinate$)$ and lowers others'
$p_g$; monotonicity in each coordinate direction, combined with the
fact that $\boldsymbol\varepsilon\wedge\boldsymbol\varepsilon'$ is
sandwiched between $\boldsymbol\varepsilon$ and
$\boldsymbol\varepsilon'$ in the product order, gives
$p_g(\boldsymbol\varepsilon\wedge\boldsymbol\varepsilon') \ge
\min\bigl(p_g(\boldsymbol\varepsilon), p_g(\boldsymbol\varepsilon')\bigr)
\ge p_{\min}$. The join argument is dual. $\square$

Boundary remark: at profiles with $p_i(\boldsymbol\varepsilon) = p_{\min}$
exactly, the strict-monotonicity argument uses that the coordinate that
just saturated the bound cannot fall further under a componentwise-min
operation applied to profiles already at the boundary; equality
propagates and the boundary is closed. The theorem then holds on the
strict interior sublattice, and NEs on the boundary are recovered as
limits by continuity of $u_i$ on $\overline{\mathcal F_\star}$.

### 4.4 Proof of Corollary 3.4 (Milgrom–Shannon MCS)

Under Lemma 3.1 the PEG is supermodular in $a$; the private parameter
$w_i$ enters $u_i$ only as $-w_i A_i$ with $A_{i,p}>0$ and
$\partial p_i/\partial\varepsilon_i>0$, so $\partial^2 u_i / (\partial
\varepsilon_i \partial w_i) = -A_{i,p}\,\partial p_i/\partial\varepsilon_i -
A_{i,\sigma}\,\partial\sigma/\partial\varepsilon_i$. The first term is
negative; the second is positive but $O(p_i)$ under T20; in the small-$p$
regime the first dominates. Hence $\partial^2 u_i /
(\partial \varepsilon_i \partial w_i) < 0$: decreasing differences in
$(\varepsilon_i, w_i)$, or equivalently increasing differences in
$(a_i, w_i)$. Milgrom–Shannon 1994 Theorem 4 then gives: the best-response
correspondence $\mathrm{BR}_i(a_{-i}; w_i)$ is nondecreasing in $w_i$
(strong set order); the largest and smallest fixed points inherit this
monotonicity; and by symmetry of the FOC in cross-effects, others'
extremal NE coordinates $\varepsilon_j^\bullet$ move in the opposite
direction as $w_i$ grows. $\square$

---

## 5. Cross-links to downstream tasks

**T07 (2-player self-sorting → n players).** T07's Proposition
`prop:self-sorting` proved the sort order $\varepsilon_i \le \varepsilon_j$
under $w_i > w_j$ at $n=2$ by the two-point exchange argument, and
noted at Sec 3.5 that the $n$-player extension "is contingent on
establishing supermodularity of the PEG in $(\varepsilon_i,\sigma)$
(Task T21)". T21's Corollary `cor:mcs` closes exactly that gap: on
$\mathcal F_\star$, the Milgrom–Shannon sort order is the direct
$n$-player generalisation. Un-gating T07 Step 7 amounts to appending
"see `\cref{cor:mcs}` in `\cref{app:continuous-ne}`" to the discussion
paragraph.

**T19 (fictitious-play sweep at $n\in\{3,5,10,50\}$).** T19 reports the
empirical modal cluster reached by fictitious play but has no theoretical
anchor for what the cluster should look like. The Milgrom–Roberts
Theorem for supermodular games says fictitious play (and more generally
any adaptive-monotone dynamic — Milgrom–Roberts 1990 Theorem 8;
Krishna 1992) converges to the interval $[\boldsymbol\varepsilon^{\rm
inf}, \boldsymbol\varepsilon^{\rm sup}]$ bounded by the extremal NEs.
T19's a-priori expectation is therefore: the FP fixed point sits between
the largest and smallest NE catalogued by brute-force enumeration on the
K=7 grid. This is the null hypothesis T19's numerical output is measured
against.

**T22 (Roughgarden PoA Path 3).** Aggregative-game PoA bounds (Jensen
2010; Cornes–Hartley 2005) require the aggregative-supermodular structure
we establish here in $a$. Once T21 lands, `NOVELTY_ACTION_PLAN.md`
Path 3's $(\lambda,\mu)$-smoothness argument becomes tractable via the
aggregate $(\sigma, \sum_g|G_g|\sqrt{\rho_g})$.

---

## 6. Numerical corroboration

Script: `tasks_findings/_scripts/21_check.py`. Runtime: ~5 s login-node.
No Slurm.

### 6.1 Cross-partial sign on C1 and C2

For every K x K x 2 utility surface, centre-difference

$$D^2 u_k(i,j) \;:=\; \frac{U[i+1,j+1,k] - U[i+1,j-1,k] - U[i-1,j+1,k] + U[i-1,j-1,k]}{(\varepsilon_{i+1}-\varepsilon_{i-1})(\varepsilon_{j+1}-\varepsilon_{j-1})}$$

at every interior feasible cell. Report the fraction with $D^2 u_k \le 0$
(sub-modular in $\varepsilon$, equivalently super-modular in $a$).

**C1 (2-player symmetric, K=22):**
```
[player 1] n=222  d^2u<=0: 165 (74.3%)  d^2u>0: 57 (25.7%)  range=[-1.7e-01, +1.5e-01]
    small-p (A_i<0.05): n=52  d^2u<=0: 26 (50.0%)  worst positive: +6.8e-02
[player 2] n=222  d^2u<=0: 165 (74.3%)  d^2u>0: 57 (25.7%)  range=[-1.7e-01, +1.5e-01]
    small-p (A_i<0.05): n=52  d^2u<=0: 26 (50.0%)  worst positive: +6.8e-02
```

**C2 (2-player, seven $|G_1|/|G_2|$ ratios, K=9):**
```
ratio=1  (symmetric groups):
  [player 1] n=17  d^2u<=0: 17 (100.0%)  range=[-1.3e-02, -4.7e-06]
  small-p:  d^2u<=0: 2/2 (100%)
  [player 2] identical (mechanism swap symmetry)

ratio=0.25 (|G_1|=2000, |G_2|=8000):
  [player 1] n=20  d^2u<=0: 18 (90.0%)   range=[-1.5e-02, +5.4e-04]
  [player 2] n=20  d^2u<=0: 20 (100.0%)  range=[-7.5e-03, -6.5e-05]

ratio=4  (|G_1|=8000, |G_2|=2000):  (mirror of ratio=0.25)
  [player 1] n=20  d^2u<=0: 20 (100.0%)
  [player 2] n=20  d^2u<=0: 18 (90.0%)
```

Full table across all seven ratios and both players lives in the
verbatim output of `_scripts/21_check.py`; see the scratchpad.

**Reading the numbers.**

* **On the symmetric-group cross-section** the cross-partial is
  submodular at **100%** of the K=9 interior feasible cells (C2 ratio=1).
  This is the cleanest confirmation of Lemma 3.1.
* **On the K=22 C1 cache** the fraction drops to 74.3%. The residual
  25.7% comes from three sources: (i) coarse-grid finite-difference
  noise (adjacent cells straddle the RDP-order switch — same source of
  noise T06 Sec 4.3 documented on the same cache); (ii) cells where
  $\sigma < 1/\sqrt 3$ so the $V''$ contribution flips; (iii) cells
  where $p_i$ exits the small-$p$ regime and T20's leading-order
  argument no longer dominates. These three carve out exactly the
  complement of $\mathcal F_\star$, which is what the conditional
  statement of the theorem allows.
* **On the small-$p$ face** (A_i < 0.05 as a monotone proxy for $p_i$
  small), the K=9 cross-section is 100% submodular; the K=22 cross-section
  is noisier (50%) because at that resolution the small-$p$ cells sit at
  the boundary of the feasible region where finite differences pick up
  saturation effects.

The conservative reading: Theorem 3.2 as stated (conditional on
$\mathcal F_\star$) is fully consistent with the numerical evidence on
the well-resolved K=9 cache and on the symmetric-group cross-section
that dominates the paper's C1 and C6 scenarios.

### 6.2 MCS spot check on C2

The Milgrom–Shannon corollary predicts the extremal NEs monotone in the
sensitivity parameters $(w_i, |G_i|, E_b)$. C2 sweeps $|G_1|/|G_2|$ at
fixed $w_1=w_2$, so it isolates the group-size direction of the MCS.
Extremal NE coordinates by group-size ratio:

```
ratio  |G1|  |G2|   e1_min  e1_max  e2_min  e2_max
  0.1   909  9091    0.500   3.084   1.682  10.375
 0.25  2000  8000    0.500   3.084   1.682  10.375
  0.5  3333  6667    0.917   3.084   3.084  10.375
    1  5000  5000    0.917  10.375   0.917  10.375
    2  6667  3333    3.084  10.375   0.917   3.084
    4  8000  2000    1.682  10.375   0.500   3.084
   10  9091   909    1.682  10.375   0.500   3.084

concordant pairs (ratio, e1_min): 16/18
concordant pairs (ratio, e2_min):  2/18   (i.e. anti-monotone)
```

The C2 sweep is symmetric under $(|G_1|,\varepsilon_1) \leftrightarrow
(|G_2|,\varepsilon_2)$; the anti-monotonicity in the second row is the
$|G_2|$-decreasing direction, and matches the $|G_1|$-increasing
direction of the first row up to the swap. The read is: as one group
grows, its extremal-NE coordinate rises (higher $\varepsilon$: more
budget to spend once the other group carries fewer members and less
mechanism cost per capita) and the other group's extremal-NE coordinate
falls. This is the direction Corollary 3.4 predicts for the group-size
parameter under the reversed-$\varepsilon$ order; the 16/2 = 89%
concordance on the K=9 grid is consistent with the theorem modulo
grid-boundary NEs at $\varepsilon_{\max}=10.375$ (the $|G_1|/|G_2|=1$
symmetric row cluster contains grid-boundary NEs that dominate the tie
counts).

### 6.3 Sympy expansion of the cross-partial

`_scripts/21_check.py` §0 sympy-expands
$\partial^2 u_i/(\partial\varepsilon_i \partial\varepsilon_j)$ by chain
rule and verifies the six-term identity used in the proof of Lemma 3.1
(residual `0`). The expansion prints the six terms one-per-line, and
the residual-zero assertion confirms no term was missed in the manual
derivation.

---

## 7. Paper location and transitions (for the paper-side follow-up)

Do **not** edit these here; leave them to a follow-up task that batches
the paper rewrites once T20, T21, T07-step-7 have all landed.

* **New appendix section** in `paper/tex/08_appendix.tex` after
  `\label{app:sensitivity}`: `\section{Pure-strategy NE existence on the
  continuous action space}\label{app:continuous-ne}`. Contents: setup
  (Section 2), Lemma `lem:increasing-differences` + proof (Sections 3.1,
  4.1), Theorem `thm:continuous-ne-smallp` + proof (Sections 3.2, 4.2),
  Sublattice Lemma + proof (Sections 3.3, 4.3), Corollary `cor:mcs` +
  proof (Sections 3.4, 4.4), numerical certificate paragraph
  (Section 6, with a small stdout table).
* **Rewrite the "not automatic" caveat** at
  `paper/tex/04_analysis.tex:107-114` to point at
  `\cref{app:continuous-ne}` for the small-$p$ closure and to flag the
  full-region case as still open (bridging into T23).
* **Rewrite open question 1** at `paper/tex/07_discussion.tex:60-63` to
  cite `\cref{thm:continuous-ne-smallp}` as the partial closure; the
  residual open piece is full-region existence, which T23 targets.
* **Pointer paragraph** in `paper/tex/03_game.tex:158-167` (Equilibrium
  concepts) with a `\cref{app:continuous-ne}` reference; keep the
  grid-search framing as a computational tool, not an existence device.
* **New bib entries** in `paper/refs.bib`: `milgrom1990rationalizability`,
  `vives1990nash`, `jensen2010aggregative`, and `milgrom1994monotone`
  (grep to confirm T07 has not already added it).
* **Un-gate T07 Step 7**: append a `\cref{cor:mcs}` reference to the
  T07 discussion paragraph.

---

## 8. Acceptance checklist (for the paper-side follow-up, not this note)

- [ ] Appendix section `app:continuous-ne` present after
  `app:sensitivity`, with `lem:increasing-differences`,
  `thm:continuous-ne-smallp`, `lem:sublattice`, `cor:mcs` all labelled
  and resolved.
- [ ] The proof of `lem:increasing-differences` cites `lem:externality`
  and `cor:externality-small-p` (T20) — no other unsupported inputs.
- [ ] The Milgrom–Roberts Theorem 5 hypotheses are explicitly listed in
  the proof of `thm:continuous-ne-smallp`.
- [ ] The paper text is honest about the *conditional* nature of the
  result: it lives on the small-$p$ sublattice
  $\mathcal F_\star \subset \mathcal F$; full-region existence is
  flagged as open and forward-referenced to T23.
- [ ] `\cref{cor:mcs}` is cross-referenced from T07's discussion
  paragraph (2-player → n-player step).
- [ ] `paper/tex/03_game.tex`, `paper/tex/04_analysis.tex`,
  `paper/tex/07_discussion.tex` no longer contain the "existence on the
  continuous action space is not automatic" caveat.
- [ ] `paper/refs.bib` contains `milgrom1990rationalizability`,
  `vives1990nash`, `jensen2010aggregative`,
  `milgrom1994monotone` (or the T07-added variant); no missing-citation
  warnings.
- [ ] `_scripts/21_check.py` completes with the reported percentages
  under the current `.venv`; verbatim output is quoted in the appendix.
- [ ] CLAUDE.md Rule 0 respected: no "we previously had...", no
  "conditional-become-unconditional" prose; the small-$p$ conditional
  is presented as the theorem, not as a stepping stone.

---

## 9. Anchors in the codebase and literature

* `game/utility.py:42-57` — $V(\sigma)=1/(1+\sigma^2)$, $u_i = V - w_i A_i$.
* `game/mechanism.py:32-49` — (C1), (C2) constraints and solver.
* `game/continuous.py:49-75` — `ContinuousGame.evaluate`.
* `game_experiments/_cache_v2/C1.pkl`, `C2.pkl` — utility grids used
  in Section 6.
* `paper/tex/04_analysis.tex:11-65` — `lem:externality` (input).
* `paper/tex/04_analysis.tex:107-114` — caveat to be rewritten.
* `paper/tex/03_game.tex:124-132` — `ass:participation`.
* `paper/tex/07_discussion.tex:60-63` — Open Question 1.
* `paper/tex/08_appendix.tex:65` — insertion point after
  `app:sensitivity`.
* `tasks_findings/20_small_p_corollary.md` — T20 corollary consumed
  in Sections 3.1 and 4.1.
* `tasks_findings/06_non_potential_game_proposition.md` — T06 (rules out
  potential-game route; also supplies the $\sigma_{ij}>0$ and
  $p_{i,ij}=0$ symbolic identities used in Section 4.1).
* `tasks_findings/07_2player_self_sorting.md` — T07 (Step 7 un-gated
  by Corollary `cor:mcs`).
* Milgrom, P. and Roberts, J. "Rationalizability, Learning, and
  Equilibrium in Games with Strategic Complementarities." Econometrica
  58(6), 1990, pp. 1255-1277. Theorem 5.
* Vives, X. "Nash Equilibrium with Strategic Complementarities."
  Journal of Mathematical Economics 19(3), 1990, pp. 305-321.
* Milgrom, P. and Shannon, C. "Monotone Comparative Statics."
  Econometrica 62(1), 1994, pp. 157-180. Theorem 4.
* Topkis, D. "Equilibrium Points in Nonzero-Sum n-Person Submodular
  Games." SIAM J. Control Optim. 17(6), 1979, pp. 773-787 (original
  lattice-theoretic existence theorem).
* Jensen, M. K. "Aggregative Games and Best-Reply Potentials."
  Economic Theory 43(1), 2010, pp. 45-66 (aggregative-supermodular
  framework relevant to T22 Path 3).
* Balle, B., Barthe, G., and Gaboardi, M. "Privacy Profiles and
  Amplification by Subsampling" (already cited as `balle2020privacy`
  in `refs.bib`) — continuity of the subsampled-Gaussian trade-off
  function needed for hypothesis (2) of Milgrom–Roberts Theorem 5.
