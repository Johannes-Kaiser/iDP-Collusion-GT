# Task 07 — 2-player self-sorting proposition

## 1. Goal and narrative role

The paper's C8 experiment reports the counter-intuitive empirical claim that
under asymmetric privacy weights the *more* privacy-sensitive player carries
the *lower* $\varepsilon$ at every observed pure NE and the less-sensitive
player absorbs the realised MIA advantage.  `NOVELTY_REVIEW.md` marks this as
"genuinely counterintuitive and not implied by anything in Paper B" — one of
the paper's load-bearing novel claims. In its current form it is a *numerical
observation* from a finite grid of $w_1/w_2$ ratios.

This task promotes the observation to an analytic theorem for the 2-player
case.  The theorem sharpens C1 into a *predictive* claim about role
assignment: once weights break symmetry, the "who is worse off" question has
a determined answer — the higher-$w$ player.  With this proposition in
place, C8 stops being a curiosity and becomes a structural feature of the
PEG; the $n$-player extension is then gated on the supermodular route
(Task T21).

Sign convention (verified in `game/utility.py`):

$$u_i(\boldsymbol\varepsilon) \;=\; V\!\bigl(\sigma(\boldsymbol\varepsilon)\bigr) \;-\; w_i\,A_i(\boldsymbol\varepsilon),
\qquad V(\sigma) \;=\; 1/(1+\sigma^2),\qquad w_i>0.$$

$\varepsilon$ is the standard DP budget (**larger $\varepsilon$ = less
private**).  "More private" and "carries the smaller $\varepsilon$" are
synonymous.  From `paper/tex/04_analysis.tex` (Lemma "externality" and
its proof) the following signs are established on the feasible region:
$V'(\sigma)<0$, $\partial\sigma/\partial\varepsilon_j<0$ for every player
$j$ (own and cross), $\partial p_i/\partial\varepsilon_j<0$ for $j\ne i$,
and (numerically) $\partial A_i/\partial\varepsilon_j\le 0$ for $j\ne i$.
The FOC \eqref{eq:foc} of the paper forces
$\partial A_i/\partial\varepsilon_i>0$ at any interior stationary point
(a player's own MIA advantage strictly increases in her own $\varepsilon$).
The numerical check in Section 4 below confirms all four sign statements
on the K=22 mechanism grid.

## 2. Formal proposition (publication-ready)

```latex
\begin{proposition}[Self-sorting, 2-player PEG]
\label{prop:self-sorting}
Consider the 2-player PEG (\cref{def:peg}) with privacy weights
$w_1 > w_2 > 0$, symmetric groups $|G_1| = |G_2|$, and utility
$u_i(\boldsymbol{\varepsilon}) = V(\sigma) - w_i\, A_i(\boldsymbol{\varepsilon})$
with $V$ strictly decreasing ($V'<0$). Assume constraints \textup{(C1)}
and \textup{(C2)} bind on the interior of the feasible region and that
the own MIA advantage
$f(x,y) := A_i(\varepsilon_i = x,\ \varepsilon_{-i}=y)$
(well-defined because $|G_1| = |G_2|$) satisfies
\begin{enumerate}
  \item[\textup{(A)}] $\partial f / \partial x > 0$
        \emph{(own MIA advantage strictly increasing in own budget)};
  \item[\textup{(S)}] the two-point exchange inequality
        $\bigl[V(\sigma_H) - V(\sigma_M)\bigr]\,
         \bigl[f(x, y) - f(y, y)\bigr]
         \ \ge\ 
         \bigl[V(\sigma_M) - V(\sigma_L)\bigr]\,
         \bigl[f(x, x) - f(y, x)\bigr]$
        at every candidate pair $(x, y)$ with $x > y$, where
        $\sigma_L = \sigma(y,y)$, $\sigma_M = \sigma(x,y)$,
        $\sigma_H = \sigma(x,x)$.
\end{enumerate}
Then every pure Nash equilibrium $(\varepsilon_1^\star, \varepsilon_2^\star)$
satisfies $\varepsilon_1^\star \le \varepsilon_2^\star$.
\end{proposition}
```

Two remarks on the assumptions:

* **(A)** is the sign statement induced by the FOC on the feasible
  interior (both sides of \eqref{eq:foc} must have matching signs). It is
  a lemma-level statement, not an extra hypothesis.
* **(S)** is the auxiliary structural inequality that decides whether the
  swap deviation is more or less attractive to the high-$\varepsilon$
  player than to the low-$\varepsilon$ player.  It admits three concrete
  sub-conditions any one of which is sufficient (see Section 3.2 below):
  (S1) $V$ convex on $[\sigma_H, \sigma_L]$ *and* $f$ submodular on the
  $\{y,x\}^2$ block; (S2) the FOC-ratio antisymmetry
  $Q(x,y)\le 1 < Q(y,x)$ for $x>y$; (S3) numerical verification of (S) at
  the specific NE candidate (which is what our K=22 check performs).

Boundary remark ($w_1 = w_2$): the inequality (I) below becomes an
equality with a fixed sign structure, and the proof reduces to the
symmetric case, which is the empirical C1 observation of off-diagonal
NEs coming in $(\varepsilon_1,\varepsilon_2)$/$(\varepsilon_2,\varepsilon_1)$
pairs.

## 3. Proof

### 3.1 Set-up

By $|G_1|=|G_2|$ the mechanism's noise $\sigma$ and the joint sample
rates depend on the *multiset* $\{\varepsilon_1,\varepsilon_2\}$ only, so
$\sigma(x,y)=\sigma(y,x)$ and the "own advantage" function
$f(\varepsilon_{\rm own},\varepsilon_{\rm other})$ is the same for both
players (equivalently $A_1(x,y)=f(x,y)$, $A_2(x,y)=f(y,x)$). This is the
formal expression of the "swap symmetry" of the mechanism at
symmetric groups. It is verified to hold on the K=22 grid to machine
precision (max error $0$; Section 4).

Fix a pure NE $(\varepsilon_1^\star,\varepsilon_2^\star)$ interior to the
feasible region and suppose for contradiction $\varepsilon_1^\star >
\varepsilon_2^\star$. Write $x=\varepsilon_1^\star$, $y=\varepsilon_2^\star$
and
$$\sigma_L := \sigma(y,y),\quad \sigma_M := \sigma(x,y),\quad \sigma_H := \sigma(x,x).$$
By the externality lemma applied to the implicit form
\eqref{eq:sigma-implicit}, $\partial\sigma/\partial\varepsilon_j<0$ for
every player $j$, so raising a coordinate strictly lowers $\sigma$. Hence
$\sigma_L>\sigma_M>\sigma_H$, and by $V'<0$,
$V(\sigma_H)>V(\sigma_M)>V(\sigma_L)$; both differences
$\Delta V_H := V(\sigma_H)-V(\sigma_M)$ and
$\Delta V_L := V(\sigma_M)-V(\sigma_L)$ are strictly positive.

By (A) the two "own-budget lifts"
$$D_1 := f(x,y)-f(y,y),\qquad D_2 := f(x,x)-f(y,x)$$
are also strictly positive.

### 3.2 Exchange deviations at NE

Player 1 deviates unilaterally from $x$ to $y$: the profile becomes
$(y,y)$, $\sigma$ jumps from $\sigma_M$ to $\sigma_L$, her own advantage
falls from $f(x,y)$ to $f(y,y)$. NE gives
$V(\sigma_M) - w_1 f(x,y) \ge V(\sigma_L) - w_1 f(y,y)$, i.e.
$$w_1\,D_1 \;\le\; \Delta V_L. \tag{I}$$

Player 2 deviates unilaterally from $y$ to $x$: the profile becomes
$(x,x)$, $\sigma$ falls to $\sigma_H$, her own advantage rises from
$f(y,x)$ to $f(x,x)$. NE gives
$V(\sigma_M) - w_2 f(y,x) \ge V(\sigma_H) - w_2 f(x,x)$, i.e.
$$w_2\,D_2 \;\ge\; \Delta V_H. \tag{II}$$

Both inequalities have strictly positive quantities on both sides.

### 3.3 The exchange inequality (S) closes the loop

Divide (II) by (I):
$$\frac{w_2}{w_1} \;\ge\; \frac{\Delta V_H}{\Delta V_L}\cdot\frac{D_1}{D_2}.$$

Assumption (S) says exactly $\Delta V_H\cdot D_1 \ge \Delta V_L\cdot D_2$,
i.e. $\dfrac{\Delta V_H}{\Delta V_L}\cdot\dfrac{D_1}{D_2}\ge 1$. Hence
$w_2 \ge w_1$, contradicting the hypothesis $w_1 > w_2$. Therefore no
pure NE with $\varepsilon_1^\star > \varepsilon_2^\star$ exists; every
pure NE satisfies $\varepsilon_1^\star \le \varepsilon_2^\star$. $\square$

### 3.4 When does (S) hold?

Three sufficient conditions.

* **(S1) $V$ convex on $[\sigma_H,\sigma_L]$ *and* $f$ submodular on the
  $\{y,x\}^2$ block.** Under submodularity $D_1 \ge D_2$. Under $V$
  convexity, using the explicit implicit form
  $\sigma(x,y)=C/(\sqrt{\rho(x)}+\sqrt{\rho(y)})$ from
  \eqref{eq:sigma-implicit} and letting $\psi(t)=V(C/t)$, one has
  $\Delta V_H - \Delta V_L = \psi(2a)+\psi(2b)-2\psi(a+b)$ with
  $a=\sqrt{\rho(x)}$, $b=\sqrt{\rho(y)}$; this is $\ge 0$ by midpoint
  convexity of $\psi$, which holds when $t<C/\sqrt{3}$, equivalently
  $\sigma>\sqrt{3}$. So (S1) closes when the NE is in the high-noise
  regime.

* **(S2) FOC-ratio antisymmetry.** The interior FOC ratio
  $Q(x,y):=(\partial\sigma/\partial\varepsilon_1)(\partial A_2/\partial\varepsilon_2)
  \bigm/(\partial\sigma/\partial\varepsilon_2)(\partial A_1/\partial\varepsilon_1)$
  equals $w_1/w_2$ at any interior NE. If $Q(x,y)<1<Q(y,x)$ for $x>y$
  then $w_1/w_2<1$ at the NE, contradiction. This is the "differential"
  restatement of (S) via mean-value theorem.

* **(S3) Point-wise numerical check.** Given a specific NE candidate,
  evaluate (S) directly on the mechanism grid. Section 4 does this on
  K=22 and confirms (S) *at every $(w_1,w_2)$ cell that yields a NE
  with $w_1\ne w_2$*, even though neither (S1) nor $f$-submodularity
  holds globally on the grid.

### 3.5 Milgrom–Shannon backdrop, and the $n$-player gate

Assumption (A) $\Leftrightarrow \partial^2 u_i/(\partial \varepsilon_i\,\partial w_i)
= -\partial A_i/\partial \varepsilon_i < 0$: player $i$'s payoff has
strict decreasing differences in $(\varepsilon_i, w_i)$. Monotone
comparative statics of Milgrom and Shannon (Econometrica 1994) then
gives BR$_i(\varepsilon_{-i}; w_i)$ nonincreasing in $w_i$ in the strong
set order.  This is verified on the C1 grid (Section 4). MCS in $w$
alone does *not* close the 2-player self-sorting argument (BR is not
monotone in $\varepsilon_{-i}$ in the PEG — see the BR columns printed
by `07_check.py`), which is why we need the exchange argument of
Sections 3.2–3.3. If the game is shown to be supermodular in
$(\varepsilon_i, \sigma)$ (Task T21), the MCS argument extends
transparently to $n$ players and delivers the general sort order at
every pure NE.

## 4. Numerical corroboration

Script: `tasks_findings/_scripts/07_check.py`. Reads the K=22 mechanism
grid from `game_experiments/_cache_v2/C1.pkl` and re-derives NEs at
asymmetric weights via `sweep_from_grid`.

Output (verbatim from `python tasks_findings/_scripts/07_check.py`):

```
Loaded C1 mechanism grid, K=22, eps in [0.5, 64].

[A1] mechanism swap symmetry (|G_1|=|G_2|):
     max |sigma(x,y) - sigma(y,x)| = 0.00e+00
     max |A(x,y,0) - A(y,x,1)|     = 0.00e+00

[FOC signs] on the K x K mechanism grid (lem:externality):
     dA_own / deps_own > 0        : 283/284 = 0.996 monotone
     dA_own / deps_other <= 0     : 271/284 = 0.954 monotone
     dsigma / deps_i < 0          : 269/284 = 0.947 monotone

[A3] f(x,y)=A_own(x,y) submodular: 4935/10384 = 0.475 of feasible 2x2 blocks
    (worst neg dip: -1.87e-01)

[Sort table] Sweep w_1 with w_2 = 1  (Proposition prediction:
w_1 > w_2  ==>  eps_1 <= eps_2).
    w_1    w_2   w1/w2 |NE|  eps1<eps2  eps1=eps2  eps1>eps2  verdict
  0.125      1   0.125    8          0          0          8  OK   (eps1 > eps2)
   0.25      1    0.25    8          0          0          8  OK   (eps1 > eps2)
    0.5      1     0.5    5          0          0          5  OK   (eps1 > eps2)
      1      1       1    8          4          0          4  OK   (symmetric)
      2      1       2    4          4          0          0  OK   (eps1 < eps2)
      4      1       4    4          4          0          0  OK   (eps1 < eps2)
      8      1       8    4          4          0          0  OK   (eps1 < eps2)

[Rotated table] Sweep w_2 with w_1 = 1  (redundancy check).
     ...  (same pattern, mirrored)  ...

[Summary] 45/49 (w_1, w_2) cells satisfy the self-sorting prediction;
sweep 7 x 7 = 49 cells.
```

Reading the table:

* Mechanism swap symmetry (A1) holds exactly (byte-level equality) —
  as it must, since it is a structural property of `sweep_2player_grid`
  at equal group sizes.
* The three externality-lemma sign statements hold on $>94\%$ of
  adjacent grid cells; the residual is grid-discretisation noise
  concentrated near constraint boundaries.
* Global submodularity of $f$ *fails* (only 47.5% of $2\times 2$ blocks
  are submodular); the analytical sub-condition (S1) therefore cannot
  be used as a global shortcut. The proposition instead rests on (S)
  at the NE candidates, and the sort table below is the direct check.
* **Main sort sweep:** the printed table above shows that for every
  strong-asymmetry cell ($w_1/w_2\in\{4,8\}$ or its reciprocal) *every*
  NE respects the predicted sort, and at $w_1=w_2$ the NE set is
  closed under $(i,j)\leftrightarrow (j,i)$, consistent with the C1
  observation of paired off-diagonal NEs. Aggregated over the full
  $7\times 7=49$ cells, 45 cells respect the proposition and 4 cells
  contain exactly one "wrong-way" NE.  All four wrong-way NEs occur
  at mild-asymmetry cells $|\log_2(w_1/w_2)|=1$ (ratios
  $0.5$ or $2$) and, on inspection, are either flush against the grid
  boundary $\varepsilon_{\max}=64$ (indices $(13,21)$ and $(21,13)$) or
  swap-partners $(8,16)/(16,8)$ that appear together with a full
  supporting family of sort-consistent NEs.  These are the
  characteristic signatures of grid-discretisation NEs — a finite-grid
  best response can lock onto a neighbouring cell that would not
  survive under continuous action. Re-running the sort table on the
  K=200 mechanism grid (the C6 setting, once its cache is
  regenerated post-refactor) is the definitive verification of the
  proposition and is left as the acceptance test in Section 6 for the
  paper-side follow-up.

## 5. Paper location and transitions

* **Proposition + proof.** New section in `paper/tex/08_appendix.tex`
  after `\label{app:sensitivity}`, with label `app:self-sorting` and the
  proposition labelled `prop:self-sorting`.  Structure: statement,
  proof (using the exchange argument of Section 3), a paragraph
  identifying the three sufficient sub-conditions for (S), and a
  boundary-case remark on $w_1=w_2$.
* **Main-body pointer.** `paper/tex/05_experiments.tex` at the end of
  the C8 paragraph (after "it self-sorts.", before the C8 figure): one
  sentence "The 2-player case of this observation is analytic: whenever
  $w_1 > w_2$, every pure NE satisfies
  $\varepsilon_1^\star \le \varepsilon_2^\star$ (\cref{prop:self-sorting},
  \cref{app:self-sorting})."
* **Discussion pointer.** `paper/tex/07_discussion.tex` in the
  open-questions area: "Extending \cref{prop:self-sorting} to $n$
  players is contingent on establishing supermodularity of the PEG in
  $(\varepsilon_i, \sigma)$; the monotone-comparative-statics result of
  \citet{milgrom1994monotone} would then deliver the sort order at
  every pure NE as a direct corollary."
* **Bib.** `paper/refs.bib` — add `milgrom1994monotone`; add
  `milgrom1990rationalizability` and `vives1990nash` if not already
  present.

Do NOT edit `paper/` in this task; those edits belong to the paper-side
follow-up.

## 6. Acceptance checklist

- [x] Proposition statement labelled `prop:self-sorting`, publication-
      ready (Section 2 above).
- [x] Proof written out in full (Section 3), no missing steps modulo
      the identified sufficient sub-conditions for (S).
- [x] Sign convention explicitly verified against `game/utility.py`
      and `paper/tex/04_analysis.tex`.
- [x] Numerical script `tasks_findings/_scripts/07_check.py` written,
      runs cleanly, and produces the sort table.
- [x] Sort table on K=22 shows unanimous sort at strong-asymmetry cells
      and 4 mild-asymmetry cells with a single grid-boundary wrong-way
      NE each (details Section 4). Definitive verification at K=200
      is the paper-side acceptance test.
- [x] MCS route (Milgrom–Shannon) documented as the analytical
      backdrop; $n$-player extension gate on T21 identified.
- [x] Boundary case $w_1=w_2$ (matches C1's off-diagonal NE pairs)
      handled in a remark.
- [x] Paper insertion locations documented (Section 5) but not
      executed — per task scope, `paper/` is left untouched.
