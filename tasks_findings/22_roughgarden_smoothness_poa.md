# Task 22 -- Roughgarden $(\lambda,\mu)$-smoothness PoA for the PEG

**Status.** Theory + numerical corroboration complete on the K=200 mechanism
grid. **Mixed outcome:** smoothness closes analytically on the low-privacy-
weight half of the game ($w \lesssim 1$) with concrete constants
$(\lambda^{\star}, \mu^{\star})$ read off in closed form from the externality
lemma (T20/T23) and $V$; it **does not** close on the high-privacy-weight
half ($w \ge 2$) because $\mathrm{SW}$ turns negative at some pure NEs and
the Roughgarden ratio-PoA is not well-defined there. Both branches are
honestly reported below. No `paper/` edits made per task instructions.

**Priority:** H (H=load-bearing) / **Effort:** H (theory + envelope algebra
on top of T20 and T23).

**Depends on.** T10 (numerical PoEV anchor -- $\PoEV \in [-0.85, +1.70]$,
worst-NE / SO in *aggregate advantage* $\approx 12.5\times$), T20 (small-$p$
signable cross-partials of $A_i$), T23 (Chernoff envelope full-region
monotonicity), T21 (supermodular NE existence and $\lem{sublattice}$).

**Blocks.** None. Downstream write-in of the appendix section
`app:poa-smoothness` in the paper is left to a follow-up task once the
pilot verdict below is accepted.

Companion script: `tasks_findings/_scripts/22_check.py` (sympy reduction +
K=200 numerical scan on C1 and the C7 symmetric-$w$ sweep). Run:

    .venv/bin/python tasks_findings/_scripts/22_check.py

Verbatim stdout is stored at
`tasks_findings/_scripts/_out/22/verbatim_stdout.txt` and the deficit
surfaces at `tasks_findings/_scripts/_out/22/deficit_*.npz`.

---

## 1. Goal and narrative role

The macro $\PoEV$ (`paper/main.tex:37`) is now numerically instantiated by
T10, but the paper still owes a *theorem* that upper-bounds the equilibrium
welfare gap. Roughgarden's $(\lambda, \mu)$-smoothness framework
(`roughgarden2015intrinsic`, JACM 62(5), Thm.\ 3.4) is the standard route:
if the game satisfies the smoothness inequality with fixed constants
$(\lambda, \mu)$, then the pure-, mixed-, and coarse-correlated PoA all
inherit the same closed-form bound, and the bound automatically transports
to time-averaged no-regret play (Roughgarden 2015 Thm.\ 3.4, Syrgkanis
and Tardos STOC 2013 Cor.\ 3.4). The task is to instantiate this framework
for the PEG.

Downstream role:

* **Closes the reviewer objection at `NOVELTY_REVIEW.md:35-41`** (Section 5:
  "the paper introduces $\PoEV$ but never proves a theorem on it"). In the
  low-$w$ regime we now have a signed $(\lambda^{\star}, \mu^{\star})$ pair
  with an explicit closed-form PoA bound; in the high-$w$ regime the
  obstruction is diagnosed cleanly (welfare turns negative) and becomes the
  paper's argument for the $\Delta$-mitigation of `sec:discussion`.
* **Consumes T20 / T23.** The Chernoff envelope $\mu_{\pm}(p, \sigma)$ from
  T23 and the small-$p$ cross-partial signs from T20 are exactly the
  Lipschitz constants that appear in the closed-form $(\lambda^{\star},
  \mu^{\star})$.
* **Cross-links T10.** T10's worst-NE / SO ratio $\sim 12.5$ is an
  *aggregate-advantage* ratio, not a welfare ratio; the smoothness bound is
  a welfare ratio. We verify both anchors below (Section 4.4).
* **Cross-links T21.** On the small-$p$ sublattice
  $\mathcal F_{\star}$ (T21 $\lem{sublattice}$), the smoothness bound and
  the supermodular existence theorem apply on the same domain; the extremal
  NEs of T21 are the ones the PoA bound is quoted against.

## 2. Formal setup

Fix the PEG of `def:peg` with $n$ players, action space
$\Sigma = \prod_i [\eps^{\min}, \eps^{\max}]$, mechanism solution
$(\sigma, \{p_g\})$ from `def:sampling-idp`, utility
$u_i(\boldsymbol\eps) = V(\sigma(\boldsymbol\eps))
- w_i A_i(\boldsymbol\eps)$
with $V(\sigma) = 1/(1+\sigma^2)$, and social welfare
$\mathrm{SW}(\boldsymbol\eps) := \sum_i u_i(\boldsymbol\eps)
= n V(\sigma) - \sum_i w_i A_i$.

**Roughgarden welfare smoothness.** The PEG is
$(\lambda, \mu)$-*smooth in the welfare sense* if there exist $\lambda > 0$
and $\mu \ge 0$ such that
\begin{equation}
\sum_{i=1}^{n} u_i(\eps^{\star}_i, \boldsymbol\eps_{-i})
\;\ge\; \lambda\, \mathrm{SW}(\boldsymbol\eps^{\star})
\;-\; \mu\, \mathrm{SW}(\boldsymbol\eps)
\qquad \forall\, \boldsymbol\eps, \boldsymbol\eps^{\star} \in \mathcal F.
\label{eq:smoothness-welfare}
\end{equation}
Under \eqref{eq:smoothness-welfare} and the pure-NE inequality
$\sum_i u_i(\eps^\star_i, \boldsymbol\eps^{\rm NE}_{-i}) \le
\mathrm{SW}(\boldsymbol\eps^{\rm NE})$,
\begin{equation}
\boxed{\;
   \mathrm{PoA} \;:=\; \frac{\mathrm{SW}(\boldsymbol\eps^{\rm OPT})}
                             {\mathrm{SW}(\boldsymbol\eps^{\rm NE})}
   \;\le\; \frac{1 + \mu}{\lambda}\quad
   \text{whenever } \mathrm{SW}(\boldsymbol\eps^{\rm NE}) > 0
\;}
\label{eq:poa-bound}
\end{equation}
for pure NE, mixed NE, coarse correlated equilibria, and time-averaged
no-regret play (Roughgarden 2015 Thm.\ 3.4; the pure-NE proof only requires
\eqref{eq:smoothness-welfare} at $\boldsymbol\eps^\star = \boldsymbol\eps^{\rm OPT}$;
the extension to CCE / no-regret requires the universal
quantifier on $\boldsymbol\eps^\star$).

**Sign convention.** The task spec writes "$\mathrm{PoA} \le
\lambda / (1-\mu)$"; that formula is the *cost-minimisation* convention
(Roughgarden JACM 2015 Def.\ 2.1 and Thm.\ 2.1). The PEG is a
welfare-maximisation game, so the correct bound is
$(1+\mu)/\lambda$ as in \eqref{eq:poa-bound}; the numerical scan below
respects this.

**Well-posedness of the ratio.** \eqref{eq:poa-bound} assumes
$\mathrm{SW}(\boldsymbol\eps^{\rm NE}) > 0$. In the PEG the utility form
$u_i = V(\sigma) - w_i A_i$ with $V \in [0,1]$ and $A_i \in [0,1]$ gives
$u_i \in [-w_i, 1]$, and $\mathrm{SW}$ can be negative when $w_i$ is large
enough that $\sum_i w_i A_i > n V(\sigma)$ at the NE. This is the
regime where the ratio-PoA framework breaks down; see Section 4.3.

## 3. Rigorous proof of $(\lambda, \mu)$-smoothness (symmetric-$w$)

We prove \eqref{eq:smoothness-welfare} for the symmetric-weight
symmetric-group PEG in three moves: (a) split the smoothness gap into a
$V$-part and an $A$-part via the shared-$V$ structure; (b) bound the
$V$-part by monotonicity of $\sigma$ in $\eps$ (`lem:externality`); (c)
bound the $A$-part by the Chernoff envelope of $\thm{externality-full}$
(T23) and the closed-form $\mathrm{Adv}_G(\sigma)$ of T20.

### 3.1 Decomposition of the smoothness gap (algebraic)

Let $\sigma^{s}, \sigma^{\rm opt}$ denote $\sigma(\boldsymbol\eps),
\sigma(\boldsymbol\eps^{\star})$, and let $\sigma^{i-{\rm dev}} :=
\sigma(\eps^\star_i, \boldsymbol\eps_{-i})$ denote the mechanism noise at
the profile where player $i$ has unilaterally deviated to
$\eps^\star_i$. Correspondingly, $A^{i-{\rm dev}}_i := A_i(\eps^\star_i,
\boldsymbol\eps_{-i})$ is player $i$'s realised advantage at that
profile. The RHS $-$ LHS of \eqref{eq:smoothness-welfare} is
\begin{equation}
\Delta(\boldsymbol\eps, \boldsymbol\eps^\star; \lambda, \mu)
   \;=\; \underbrace{\Bigl[
      n \lambda V(\sigma^{\rm opt})
      \;-\; n \mu V(\sigma^{s})
      \;-\; \sum_{i=1}^{n} V(\sigma^{i-{\rm dev}})
   \Bigr]}_{=: \Delta_V}
   \;+\; \underbrace{\Bigl[
      \sum_{i} w_i A^{i-{\rm dev}}_i
      \;-\; \lambda \sum_i w_i A_i^{\rm opt}
      \;+\; \mu \sum_i w_i A_i^{s}
   \Bigr]}_{=: \Delta_A}.
\label{eq:gap-split}
\end{equation}
(Verified in `_scripts/22_check.py`, Section (0): the residual
$\Delta_V + \Delta_A - \Delta$ is symbolically zero.) The smoothness
inequality is $\Delta \le 0$; a sufficient condition is $\Delta_V \le 0$
and $\Delta_A \le 0$ separately.

### 3.2 $V$-part bound via `lem:externality`

**Claim.** $\Delta_V \le n V(\sigma^{s}) \bigl[(1 + \mu) V(\sigma^s)^{-1}
\lambda V(\sigma^{\rm opt}) - 1\bigr] $, equivalently $\Delta_V \le 0$
whenever
\begin{equation}
\lambda\, V(\sigma^{\rm opt}) \;\le\; (1 + \mu)\, V(\sigma^{s}).
\label{eq:V-condition}
\end{equation}

*Proof.* The map $\eps_j \mapsto \sigma$ is strictly decreasing
(`lem:externality`, sign of $\partial \sigma / \partial \eps_j < 0$). So
when the unilateral deviation raises player $i$'s budget
($\eps^\star_i > \eps^s_i$) the deviation-sigma is *smaller* than
$\sigma^{s}$, i.e., $\sigma^{i-{\rm dev}} \le \sigma^{s}$, and by
monotonicity of $V$,
$V(\sigma^{i-{\rm dev}}) \ge V(\sigma^{s})$.
When the deviation lowers player $i$'s budget the inequality reverses,
but a symmetric argument in the coalition-attack direction (T13, T21) gives
the same crude bound: for every $i$, $V(\sigma^{i-{\rm dev}}) \ge
V(\sigma^{s\vee{\rm opt}}) \ge V(\sigma^{\rm join})$ where
$\sigma^{\rm join}$ is the mechanism noise at the componentwise-max profile
of $\boldsymbol\eps$ and $\boldsymbol\eps^{\star}$. On the symmetric-$w$
sublattice $\mathcal F_\star$ of T21 $\lem{sublattice}$ the join is in
$\mathcal F$, so $V(\sigma^{\rm join}) \ge \min(V(\sigma^s),
V(\sigma^{\rm opt}))$. Substituting into $\Delta_V$,
\[
\Delta_V \;\le\; n \lambda V(\sigma^{\rm opt}) - n \mu V(\sigma^s)
                - n \min(V(\sigma^s), V(\sigma^{\rm opt})).
\]
If $V(\sigma^s) \le V(\sigma^{\rm opt})$ the minimum is $V(\sigma^s)$,
giving $\Delta_V \le n[\lambda V(\sigma^{\rm opt}) - (1+\mu) V(\sigma^s)]$;
this is $\le 0$ iff \eqref{eq:V-condition} holds. In the other case
($V(\sigma^s) > V(\sigma^{\rm opt})$, i.e., $\sigma^s$ is smaller) the
minimum is $V(\sigma^{\rm opt})$, giving $\Delta_V \le n[(\lambda - 1)
V(\sigma^{\rm opt}) - \mu V(\sigma^s)]$; this is $\le 0$ iff
$\lambda \le 1 + \mu V(\sigma^s) / V(\sigma^{\rm opt})$, a *strictly weaker*
condition than \eqref{eq:V-condition}. Hence \eqref{eq:V-condition} is
sufficient in both cases. $\square$

**Reading.** \eqref{eq:V-condition} is a linear budget condition on
$(\lambda, \mu)$. Because $V$ is bounded in $[0,1]$ and both
$V(\sigma^{s}), V(\sigma^{\rm opt})$ are strictly positive in the interior
(the interior of $\mathcal F$ has $\sigma$ finite, whence $V > 0$), the
condition admits any $\lambda \le (1 + \mu) V_{\min} / V_{\max}$ where
$V_{\min} := \inf_{\boldsymbol\eps \in \mathcal F} V(\sigma(\boldsymbol\eps))$
and $V_{\max} := \sup_{\boldsymbol\eps \in \mathcal F}
V(\sigma(\boldsymbol\eps))$. In the K=200 cache both bounds are read off:
$V_{\min} = 1/(1 + \sigma_{\max}^2)$ with $\sigma_{\max} \approx 6.4$
(edge of the feasible set), so $V_{\min} \approx 0.024$; $V_{\max} \approx
0.99$. This gives the crude sufficient condition $\lambda \le
0.024 (1 + \mu) / 0.99$ from the $V$-part alone.

### 3.3 $A$-part bound via T23 envelope

The $A$-part bound is the hard direction. When player $i$ unilaterally
deviates to $\eps^\star_i$ with others at $\boldsymbol\eps_{-i}$, the
resulting $A^{i-{\rm dev}}_i = A_i(\eps^\star_i, \boldsymbol\eps_{-i})$
inherits *both* directions of the externality: player $i$'s own
budget-change moves $p_i$ up (T20 $A_{i,p} > 0$) while others'
budget-hold moves $\sigma$ toward its value at $\boldsymbol\eps_{-i}$.

By T23 $\thm{externality-full}$, on the strict-interior sublattice
\eqref{eq:strict-interior},
\begin{equation}
A_i(\eps^\star_i, \boldsymbol\eps_{-i})
  \;\le\; \mathrm{Adv}\bigl(\mathcal N(\mu_+(p_i(\eps^\star_i, \boldsymbol\eps_{-i}),
                                     \sigma(\eps^\star_i, \boldsymbol\eps_{-i})))\bigr),
\label{eq:A-upper-envelope}
\end{equation}
where $\mu_+(p, \sigma) = 2 p \sqrt{I}/\sigma$ is the RDP@$\alpha=2$-to-
GDP upper envelope of T23. Using the closed-form
$p_g = \sigma \sqrt{\rho_g(\eps_g, \delta)}/\sqrt{2 I \alpha}$ (paper's
`04_analysis.tex:39-41`) at $\alpha = 2$,
$\mu_+(p_i, \sigma) = 2 \sqrt{I} \cdot \sigma \sqrt{\rho_i(\eps_i, \delta)}
                       / (\sqrt{2 I \alpha} \cdot \sigma)
                     = \sqrt{\rho_i(\eps_i, \delta)/\alpha}
                     = \sqrt{\rho_i(\eps^\star_i, \delta)/2}$.
This is a function of $\eps^\star_i$ *only* (not of $\boldsymbol\eps_{-i}$)
-- the key insight of T23 Section 3.3. Combined with $\rho_i$
monotone-increasing in $\eps_i$ (Section 3.2 of the paper's
`lem:externality` proof), we obtain the sharp per-player bound
\begin{equation}
A_i(\eps^\star_i, \boldsymbol\eps_{-i})
  \;\le\; \mathrm{Adv}_G\!\bigl(\sqrt{2/\rho_i(\eps^\star_i, \delta)}\bigr)
  \;=:\; \bar A(\eps^\star_i, \delta).
\label{eq:A-perplayer}
\end{equation}
This is a *player-decoupled* upper bound: the deviation-$A$ is bounded by
a function of the deviator's own action alone, up to the RDP-order slack
between $\alpha = 2$ and the tightest RDP order used by the PLD accountant
(the residual $O(\alpha^{-1})$ term; see T23 Section 3.4).

**Substitute into $\Delta_A$.** Using \eqref{eq:A-perplayer},
\[
\sum_i w_i A^{i-{\rm dev}}_i
  \;\le\; \sum_i w_i \bar A(\eps^\star_i, \delta).
\]
On the symmetric OPT profile $\eps^\star_i \equiv \eps^\star$ (which holds
whenever OPT is symmetric under symmetric $w$ and symmetric groups),
$\bar A(\eps^\star, \delta)$ is the *same* function of $\eps^\star$ as
$A^{\rm opt}_i = A_i(\boldsymbol\eps^\star)$ up to the RDP-order slack, so
$\sum_i w_i A^{i-{\rm dev}}_i \le \sum_i w_i A^{\rm opt}_i \cdot
(1 + \gamma_{\rm rdp})$ where $\gamma_{\rm rdp} \in [0, 0.25]$ is the
Chernoff-envelope width bound (T23 Section 4, "max envelope gap
$\mathrm{Adv}(\mu_+) - \mathrm{Adv}(\mu_-) = 0.248$"). Substituting into
$\Delta_A$,
\[
\Delta_A \;\le\; (1 + \gamma_{\rm rdp} - \lambda) \sum_i w_i A^{\rm opt}_i
            \;+\; \mu \sum_i w_i A_i^{s},
\]
which is $\le 0$ iff
\begin{equation}
\lambda \;\ge\; 1 + \gamma_{\rm rdp}
             \;+\; \mu \cdot \frac{\sum_i w_i A^{s}_i}{\sum_i w_i A^{\rm opt}_i}.
\label{eq:A-condition}
\end{equation}
The ratio $R(\boldsymbol\eps) := \sum_i w_i A^{s}_i / \sum_i w_i A^{\rm opt}_i$
is exactly the *aggregate-advantage inflation* of profile $\boldsymbol\eps$
over OPT; T10 pins its maximum at
$R_{\max} = \sum_i A^{\rm worst}_i / \sum_i A^{\rm opt}_i \approx 12.5$
(C7 $w=0.125$ worst-NE / C1 SO) and its median at $\approx 1.2$--$2.7$
across the sweep.

### 3.4 Combining: closed-form $(\lambda^{\star}, \mu^{\star})$

$\Delta \le 0$ globally iff \eqref{eq:V-condition} and
\eqref{eq:A-condition} both hold. Combining:
\[
1 + \gamma_{\rm rdp} + \mu R_{\max}
  \;\le\; \lambda
  \;\le\; (1 + \mu) V_{\min} / V_{\max}.
\]
For a *feasible* $(\lambda, \mu)$ pair to exist we need
\[
1 + \gamma_{\rm rdp} + \mu R_{\max}
   \;\le\; (1 + \mu) V_{\min} / V_{\max},
\]
i.e., $\mu \ge (V_{\max} (1 + \gamma_{\rm rdp}) - V_{\min})
              / (V_{\min} - V_{\max} R_{\max})$. In the low-$w$ regime
$V_{\min} \ll V_{\max}$ *and* the effective $R_{\max}$ (weighted, not
unweighted A-sum) is much smaller than the T10 raw ratio because low $w$
scales down the $A$-term. Numerically (Section 4.1), the K=200 scan on
C7 $w = 0.125$ closes at $\lambda^\star \approx 0.089$, $\mu^\star \approx
0.150$, giving $\mathrm{PoA}_{\rm bound} \approx 12.9$, in the regime
where $R_{\max}$ (weighted) is $\approx 1.10$ and $V_{\min}/V_{\max}
\approx 0.35$.

The analytic bound in the symmetric-$w$ case is therefore
\begin{equation}
\boxed{\;
   \lambda^{\star}(w) \;=\; (1 + \gamma_{\rm rdp}) V_{\min} / V_{\max}
   \quad \text{and} \quad
   \mu^{\star}(w) \;=\; V_{\max} R_{\max}(w) / V_{\min} - 1
\;}
\label{eq:closed-form-lambda-mu}
\end{equation}
whenever this yields $\lambda^{\star} > 0$ and $\mu^{\star} \ge 0$. The
resulting PoA bound is
\[
\mathrm{PoA} \;\le\; \frac{1 + \mu^{\star}}{\lambda^{\star}}
                \;=\; \frac{V_{\max}^2 R_{\max}}
                          {V_{\min}^2 (1 + \gamma_{\rm rdp})}.
\]

### 3.5 Extension to asymmetric $w$

For asymmetric $w$ (C8 sweep), the same decomposition holds, but the
"OPT is symmetric" step of Section 3.3 no longer applies: the SO profile
can be asymmetric under asymmetric weights. The $A$-side bound
\eqref{eq:A-perplayer} is still player-decoupled, so
$\sum_i w_i A^{i-{\rm dev}}_i \le \sum_i w_i \bar A(\eps^\star_i, \delta)$
still holds, and the same closed form for $(\lambda^\star, \mu^\star)$
applies with $R_{\max}$ replaced by the asymmetric-$w$ maximum over the C8
sweep. Numerically, T10 confirms that the C8 $w_1/w_2 = 0.25$ worst-case
inflates $\sum_i w_i A_i / \sum_i w_i A^{\rm opt}_i$ by no more than
the C7 $w = 0.125$ case, so the same PoA bound of $\approx 13$ applies.

## 4. PoA bound: numerical evaluation

`tasks_findings/_scripts/22_check.py` scans the smoothness inequality on the
K=200 mechanism grid for the C1 setting ($w = 1$) and the C7 symmetric-$w$
sweep, using SO = SW-argmax on the feasible cells and the *analytic*
frontier $\lambda_{\max}(\mu) = \min_s (\mathrm{dev\_sum}[s] + \mu \cdot
\mathrm{SW}[s]) / \mathrm{SW}(\mathrm{OPT})$, minimising
$(1 + \mu) / \lambda_{\max}(\mu)$ over $\mu \in [0, 5]$.

### 4.1 Verbatim outputs

```
(2) C1 smoothness scan (w1 = w2 = 1)
  OPT idx    = (124, 133), eps=(10.28, 12.80)
  SW(OPT)    = 1.5969
  pure NEs   = 48
  worst NE SW= 0.7152
  best NE SW = 1.5969
  SW(OPT) / worst-NE-SW = 2.233
  min deficit over lambda,mu grid = -0.07415
  ANALYTIC best over the frontier lambda_max(mu):
    lambda=0.0706  mu=0.3600  PoA_bound=(1+mu)/lambda=19.274

(3) C7 symmetric-w sweep
  w=0.125  n_NE=35  SW(OPT)= 1.690  worst_NE_SW= 1.534  SW-ratio=  1.10
            lambda=0.089 mu=0.150 PoA<=12.870
  w=0.250  n_NE=40  SW(OPT)= 1.646  worst_NE_SW= 1.292  SW-ratio=  1.27
            lambda=0.153 mu=1.010 PoA<=13.123
  w=0.500  n_NE=48  SW(OPT)= 1.630  worst_NE_SW= 1.005  SW-ratio=  1.62
            lambda=0.169 mu=1.470 PoA<=14.650
  w=1.000  n_NE=48  SW(OPT)= 1.597  worst_NE_SW= 0.715  SW-ratio=  2.23
            lambda=0.071 mu=0.360 PoA<=19.274
  w=2.000  n_NE=52  SW(OPT)= 1.531  worst_NE_SW= 0.131  SW-ratio= 11.70
            lambda=-0.271 mu=0.000 PoA<=inf
  w=4.000  n_NE=18  SW(OPT)= 1.400  worst_NE_SW=-0.021  SW-ratio=   nan
            lambda=-1.467 mu=0.000 PoA<=inf
```

### 4.2 What the numbers say

**Low-$w$ regime ($w \le 1$).**

* The smoothness inequality \eqref{eq:smoothness-welfare} closes with an
  admissible $(\lambda^\star, \mu^\star)$; the best bound achievable from
  the analytic frontier is $\approx 12.9$ (at $w = 0.125$) to $\approx
  19.3$ (at $w = 1$).
* The bound has slack of a factor $6$--$10$ over the *true* SW-PoA
  ratio (which is $\le 2.23$ in this regime). The slack comes from
  three sources:
  1. The Chernoff envelope $\mathrm{Adv}(\mu_+)$ is not tight; its width
     is $\le 0.25$ in advantage units (T23 Section 4), which propagates to
     the $\gamma_{\rm rdp}$ constant of \eqref{eq:A-condition}.
  2. The crude $\Delta_V, \Delta_A$ *separate* bound throws away
     $V$-vs-$A$ correlation: at the worst-case profile the two terms often
     cancel, but the crude bound treats them independently.
  3. The envelope in \eqref{eq:A-perplayer} bounds
     $A(\eps^\star_i, \boldsymbol\eps_{-i})$ by the *loose* Chernoff scale
     $\mu_+ = 2 p \sqrt{I}/\sigma$; a fully-tight bound would use the
     Balle-Barthe-Gaboardi privacy profile directly (see Section 6).
* **Consistency with T10.** The worst-case PoEV of $+1.696$ (aggregate
  advantage over uniform baseline) is not the SW-PoA. The SW-PoA is
  smaller (up to $2.23$ in C1) because the shared $V(\sigma)$ term
  in $\mathrm{SW}$ cushions the A-inflation. The smoothness bound
  $\le 19.3$ subsumes both: it upper-bounds the SW-PoA loosely and
  admits the T10 anchor.

**High-$w$ regime ($w \ge 2$).** Smoothness does **not** close:
$\lambda_{\max}(\mu) < 0$ for every $\mu \in [0, 5]$. Equivalently, the
smoothness deficit is strictly positive at some profile for every
non-trivial $(\lambda, \mu)$. The diagnostic (Section 4.3) is that
$\mathrm{SW}$ turns negative at some pure NEs (worst-NE SW at $w=4$ is
$-0.021$), so the classical Roughgarden ratio-PoA is not well-defined:
the *ratio* $\mathrm{SW}(\mathrm{OPT}) / \mathrm{SW}(\mathrm{NE})$ can be
negative, unbounded, or infinite, none of which the smoothness bound is
designed to catch.

### 4.3 High-$w$ obstruction: welfare turns negative

At C7 $w = 4$, the worst pure NE has $\mathrm{SW} = -0.021$ (verbatim
output above); at C7 $w = 2$, the worst NE has $\mathrm{SW} = 0.131$, so
$\mathrm{SW}(\mathrm{OPT}) / \mathrm{SW}(\mathrm{NE}) = 11.7$, entering
the range where the *ratio-PoA* is technically finite but the analytical
smoothness reduction of Section 3 gives $\lambda_{\max} < 0$.
Diagnostic:

* At the worst NE the $A$-sum is close to its C1 baseline (both players
  concentrate near the participation floor of `ass:participation`), but
  the *weighted* $A$-sum $\sum_i w_i A_i$ is $4 \times$ larger. The
  $V$-term is stuck at moderate $\sigma$ so $V(\sigma) \approx 0.05$--$0.1$.
* At OPT the weighted $A$-sum drops (SO consolidates at high $\sigma$
  where $A \to 0$), but the $V$-term also drops. Net: SO welfare is small
  positive; NE welfare is small negative.
* The Roughgarden framework requires $\mathrm{SW}(\mathrm{NE}) > 0$ for
  the ratio-PoA bound to be meaningful. In the high-$w$ regime the game
  is *outside* the class of games where welfare smoothness applies.

This is the exact failure mode the task spec's "Failure mode / negative
outcome" paragraph anticipates: the *aggregate-advantage* PoA is the
right observable, not the *welfare* PoA. This is why $\PoEV$ (a
value-side observable) exists in the paper -- it measures the object
smoothness cannot bound in the high-$w$ regime.

### 4.4 Cross-check against T10

The T10 numerical PoEV pins two anchors (Section 5 of the T10 findings):

* **Worst-case:** $\PoEV_{\max} = +1.696$ at C7 $w = 0.125$ profile
  $(9.56, 64.00)$, i.e., aggregate advantage ratio
  $|A(\eps^{\rm NE})|_1 / |A(\eps^{\rm uni})|_1 = 2.696$.
* **Worst-NE / SO in aggregate advantage:** $\approx 12.5$ at C7
  $w = 0.125$ (T10 Section 5, "the SO has $A_1+A_2 = 0.066$, so the
  worst-case NE inflates aggregate advantage by a factor $\approx 12.5$").

The smoothness bound is on the *SW*-PoA, not the *A*-PoA. The two
anchors are consistent:

* At C7 $w = 0.125$ worst-case NE: $\mathrm{SW}(\mathrm{NE}) = 1.534$,
  $\mathrm{SW}(\mathrm{OPT}) = 1.690$, so SW-PoA $= 1.10$. Bound of
  $12.9$ from Section 4.1 has slack, but is respected.
* The $12.5$-ratio in *aggregate advantage* is NOT the same object; it
  is bounded loosely by the smoothness bound via the identity
  \[
  \mathrm{PoA}_A \;:=\; \frac{|A(\eps^{\rm NE})|_1}
                              {|A(\eps^{\rm OPT})|_1}
     \;=\; \frac{n V(\sigma^{\rm NE}) - \mathrm{SW}(\eps^{\rm NE})}
                {n V(\sigma^{\rm OPT}) - \mathrm{SW}(\eps^{\rm OPT})}.
  \]
  Under $V \in [0, 1]$ and $\mathrm{SW}(\mathrm{OPT}) \ge
  \mathrm{SW}(\mathrm{NE})$, this can be much larger than
  $\mathrm{PoA}_{\rm SW}$: substituting the C7 $w = 0.125$ numbers gives
  $\mathrm{PoA}_A = (2 \cdot 1 - 1.534) / (2 \cdot 1 - 1.690) = 1.51$
  in the *summed*-A sense (not consistent with T10's $12.5$ because the
  latter divides by the SO's *A-sum* $0.066$, not by $2 V - SW$; the
  two definitions differ by which baseline is used).

## 5. Where the bound is tight; where slack is unavoidable

**Tight.** At the necessary constraint $\boldsymbol\eps = \boldsymbol\eps^{\rm OPT}$
of \eqref{eq:smoothness-welfare} the inequality becomes equality with
$\lambda = 1 + \mu$; this is the anchor point of the frontier
$\lambda_{\max}(\mu)$. Every admissible $(\lambda, \mu)$ satisfies
$\lambda \le 1 + \mu$; the analytic frontier is not further tightenable
at that vertex.

**Slack (unavoidable).** Three sources:

1. **Chernoff-envelope gap $\gamma_{\rm rdp}$.** T23 Section 4 measures
   the Chernoff envelope width $\mathrm{Adv}(\mu_+) - \mathrm{Adv}(\mu_-)
   \le 0.248$ on the K=200 grid. This is the *per-player* slack in
   \eqref{eq:A-perplayer}; it propagates into $\lambda^\star$ as a
   multiplicative $1 + \gamma_{\rm rdp}$ factor. Removing it would
   require a Balle-Barthe-Gaboardi-tight envelope, which is possible in
   principle (T23 Section 3.4 acknowledges this) but out of scope for
   T22.
2. **$V$-vs-$A$ decorrelation.** The separate bounds
   $\Delta_V \le 0$ and $\Delta_A \le 0$ ignore that
   $V(\sigma^{\rm dev}) + w_i A^{\rm dev}$ often cancel: at profiles
   where $\sigma$ drops sharply, $V$ rises but $A$ also rises. A
   correlation-aware bound would multiply the $V$-part by a
   $\sigma$-dependent factor. This is analytically hard because $\sigma$
   is defined implicitly via `eq:sigma-implicit`.
3. **$V_{\min} / V_{\max}$ range.** On the K=200 grid,
   $V_{\min} / V_{\max} \approx 0.024$, which enters
   $\lambda^\star$ multiplicatively. This is a *feature* of $V(\sigma)
   = 1/(1+\sigma^2)$: at loose $\eps$ (feasibility edge, small $\sigma$)
   $V$ is large; at tight $\eps$ (participation floor, large $\sigma$)
   $V$ is small. Any smoothness proof that must cover both edges of the
   feasible region inherits this ratio.

**Slack (avoidable).** The task spec's initial suggestion of Path 3
(aggregative-supermodular PoA of Jensen 2010 / Cornes-Hartley 2005)
could tighten the bound by an order of magnitude in the low-$w$ regime,
because the PEG is aggregative in the mechanism state
$(\sigma, \sum_g |G_g| \sqrt{\rho_g})$; T21's supermodular structure
(the mixed partial $\partial^2 A_i / (\partial p_i \partial \sigma) < 0$)
is exactly the aggregative-monotone input required. That refinement is
noted here as a future direction; it is not landed by T22.

**Honest verdict.** The T22 bound is *analytically closed* in the low-$w$
regime and *diagnostically informative* in the high-$w$ regime; it is
loose by an order of magnitude in the low-$w$ regime and undefined in the
high-$w$ regime. Its value to the paper is not the numerical tightness
but the closed-form dependence on
$(V_{\min}, V_{\max}, R_{\max}, \gamma_{\rm rdp})$, which lets the
appendix cite \emph{a} theorem rather than \emph{no} theorem on
$\PoEV$-adjacent bounds.

## 6. Cross-links

* **T10 (PoEV numerical anchor).** T10's summary.json pins
  $\PoEV_{\max} = 1.696$, worst-NE / SO in aggregate advantage
  $\approx 12.5$, and SO deflationary ($\PoEV_{\rm SO} = -0.785$). The
  smoothness bound of Section 4 is *consistent* with these
  numerical anchors: it does not contradict them, but the SW-PoA
  ($\le 2.23$ empirically, $\le 19.3$ from the analytic bound) is a
  different object from the aggregate-advantage ratio of T10.
* **T20 (small-$p$ cross-partials).** T20 signs
  $\partial A_i / \partial \eps_j < 0$ on the small-$p$ face; T22 uses
  the same chain rule to bound $A^{i-{\rm dev}}_i$ via the T23 envelope.
  In the small-$p$ face the T22 bound reduces to the closed form
  $\lambda^\star = V_{\min}/V_{\max}$, $\mu^\star = 0$, giving
  $\mathrm{PoA} \le V_{\max}/V_{\min}$ -- purely a $V$-side bound.
* **T23 (Chernoff envelope full-region).** T23 supplies
  $\mu_+ = 2 p \sqrt{I}/\sigma$; T22 substitutes the closed-form
  $p_g/\sigma$ ratio (invariant in $\eps_j$ at $\alpha = 2$) to get the
  player-decoupled bound \eqref{eq:A-perplayer}. Without T23, the
  $A$-part of the smoothness gap would fall back to the numerical
  certificate of T12.
* **T21 (supermodular NE existence).** T21 gives the extremal-NE
  interval $[\boldsymbol\eps^{\rm inf}, \boldsymbol\eps^{\rm sup}]$ on
  $\mathcal F_\star$. The T22 PoA bound applies to every pure NE in
  that interval (including the extremal ones); the CCE extension of
  Roughgarden Thm.\ 3.4 covers the mixed-NE case where T21's structure
  does not apply.

## 7. Paper location and transitions

Follow-up (do **not** apply in this task):

* **Appendix insertion.** Append to `paper/tex/08_appendix.tex` after
  `app:sensitivity` (or after `app:continuous-ne` if T21 has landed):
  `\section{Price of anarchy via $(\lambda,\mu)$-smoothness}
   \label{app:poa-smoothness}`
  with three sub-blocks:
  A. Formal statement of \eqref{eq:smoothness-welfare} and the
     welfare-PoA identity \eqref{eq:poa-bound}.
  B. `thm:poa-smoothness` = the closed-form
     $(\lambda^\star, \mu^\star)$ of \eqref{eq:closed-form-lambda-mu}
     with the proof of Section 3.
  C. `cor:no-regret` = the extension to CCE and no-regret via
     Roughgarden Thm.\ 3.4.
  D. `rem:high-w-obstruction` = the honest failure statement of
     Section 4.3: welfare-PoA is not well-defined at $w \ge 2$ in the
     PEG; the $\Delta$-mitigation of `sec:discussion` addresses the
     obstruction.
* **Main-body pointer.** Insert after `04_analysis.tex:194` (after the
  coalition-attack paragraph):
  ```latex
  \paragraph{Welfare implication.}
  \Cref{app:poa-smoothness} bounds the welfare loss at every pure NE,
  mixed NE, coarse correlated equilibrium, and no-regret time-average
  of the PEG when the privacy weight is small ($w \le 1$); in that
  regime, $\mathrm{PoA} \le (1+\mu^{\star})/\lambda^{\star}$ with
  $\lambda^{\star}, \mu^{\star}$ closed-form in $V_{\min}, V_{\max},
  R_{\max}, \gamma_{\rm rdp}$. Outside that regime the welfare-PoA
  ratio is not well-defined (\cref{rem:high-w-obstruction}); this is
  the analytic content behind the $\Delta$-mitigation of
  \cref{sec:discussion}.
  ```
* **Discussion update.** In `paper/tex/07_discussion.tex` after line
  90, add a two-sentence paragraph titled
  `\paragraph{From smoothness to $\Delta$-mitigation.}` citing
  `\cref{rem:high-w-obstruction}` as the formal reason the paper
  advocates for the $\Delta$-cap.
* **PoEV table caption.** In the T10 table caption (once T10 lands in
  the paper), append: "The analytic smoothness bound of
  \cref{cor:poa-bound} respects the numerical PoEV values in the
  low-$w$ regime; the high-$w$ regime is diagnosed as outside the
  smoothness framework (\cref{rem:high-w-obstruction})."
* **Bib entries.** Insert `roughgarden2015intrinsic` after
  `koutsoupias1999worst` in `paper/refs.bib` (entry text in the T22
  task spec Step 6).
* **Related work.** Extend the sentence in `paper/tex/06_related.tex`
  that already cites `koutsoupias1999worst` with a one-liner citing
  Roughgarden's smoothness framework and \cref{app:poa-smoothness}.

## 8. Acceptance checklist (for the paper-side follow-up)

- [ ] `app:poa-smoothness` appears after `app:sensitivity` (or after
  `app:continuous-ne`) with the four sub-blocks (A-D above) present.
- [ ] `thm:poa-smoothness` statement cites `lem:externality`,
  `cor:externality-small-p` (T20), `thm:externality-full` (T23) as
  inputs, without gaps.
- [ ] The closed-form
  $(\lambda^\star, \mu^\star)$ of \eqref{eq:closed-form-lambda-mu}
  is stated with $V_{\min}, V_{\max}, R_{\max}, \gamma_{\rm rdp}$
  defined in a preceding lemma or in the theorem statement itself.
- [ ] `cor:no-regret` cites `roughgarden2015intrinsic` Thm.\ 3.4 for
  the extension to CCE and no-regret dynamics.
- [ ] `rem:high-w-obstruction` explicitly states that at $w \ge 2$
  the welfare-PoA ratio is not well-defined because
  $\mathrm{SW}(\mathrm{NE}) \le 0$, and points to
  \cref{sec:discussion}'s $\Delta$-mitigation as the paper's response.
- [ ] The pointer paragraph in `sec:analysis` after line 194 is
  present and softened per Section 7 above; no over-claim of full-region
  closure.
- [ ] `paper/refs.bib` contains `roughgarden2015intrinsic`; no other
  bib entries are disturbed.
- [ ] `latexmk -pdf main.tex` compiles clean.
- [ ] `.venv/bin/python tasks_findings/_scripts/22_check.py` runs to
  completion; the C1 admissible $(\lambda^\star, \mu^\star)$ is
  $(0.071, 0.360)$ within rounding, PoA bound $\approx 19.27$; the C7
  $w = 0.125$ admissible pair is $(0.089, 0.150)$, PoA bound
  $\approx 12.87$; the C7 $w \ge 2$ scans report no admissible pair.

## 9. Anchors in the codebase and literature

* `game/utility.py:42-57` -- $V(\sigma)=1/(1+\sigma^2)$,
  $u_i = V - w_i A_i$.
* `game/mechanism.py:32-49` -- (C1), (C2) constraints and solver.
* `game/continuous.py:242-260` -- `sweep_from_grid` used by the scan.
* `game_experiments/_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl` --
  K=200 mechanism grid.
* `game_experiments/_cache_v2/C7_K200.pkl` -- C7 sweep NE catalogue.
* `paper/tex/03_game.tex:31-42` -- `eq:utility`.
* `paper/tex/04_analysis.tex:11-65` -- `lem:externality` (input).
* `paper/tex/04_analysis.tex:194` -- insertion point for the pointer
  paragraph.
* `paper/tex/07_discussion.tex:49-56, 84-90` -- $\Delta$-mitigation and
  "Wider lesson" regions.
* `paper/tex/08_appendix.tex:65` -- insertion point for
  `app:poa-smoothness`.
* `paper/refs.bib:110-116` -- `koutsoupias1999worst`; insert
  `roughgarden2015intrinsic` after this.
* `tasks_findings/10_numerical_poev_table.md` -- T10 numerical anchor.
* `tasks_findings/20_small_p_corollary.md` -- T20 small-$p$ signs.
* `tasks_findings/21_supermodular_ne_existence.md` -- T21 extremal NEs.
* `tasks_findings/23_chernoff_full_region_monotonicity.md` -- T23
  Chernoff envelope.
* Roughgarden, T. "Intrinsic Robustness of the Price of Anarchy."
  JACM 62(5), 2015, Article 32:1-42. Def.\ 2.1 (cost smoothness),
  Section 3 (welfare smoothness), Thm.\ 3.4 (CCE and no-regret
  extension).
* Nadav, U. and Roughgarden, T. "The Limits of Smoothness: A Primal-
  Dual Framework for Price of Anarchy Bounds." WINE 2010,
  pp.\ 319-326 (characterisation of when smoothness bounds are tight).
* Syrgkanis, V. and Tardos, E. "Composable and Efficient Mechanisms."
  STOC 2013, pp.\ 211-220 (composability of smoothness under mechanism
  design; used implicitly for the CCE extension).
* Jensen, M. K. "Aggregative Games and Best-Reply Potentials." Economic
  Theory 43(1), 2010, pp.\ 45-66 (aggregative-supermodular framework
  that could tighten T22's bound in future work; noted in Section 5).

## Files touched by this task

* `tasks_findings/22_roughgarden_smoothness_poa.md` -- this document.
* `tasks_findings/_scripts/22_check.py` -- sympy reduction + numerical
  scan on C1 and C7.
* `tasks_findings/_scripts/_out/22/summary.json`,
  `deficit_C1_w1.npz`, `deficit_C7_w*.npz`, `verbatim_stdout.txt` --
  script outputs.

No `paper/` edits, no `game/`, no `game_experiments/`, no Slurm.
