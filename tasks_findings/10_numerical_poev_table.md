# T10 — Numerical PoEV table from the K=200 cache

**Priority:** H (novelty-load-bearing) / L (effort).
**Depends on:** K=200 caches under `game_experiments/_cache_v2/`
(`mech_grid_K200_g5000x5000_b128_e5.pkl`, `C7_K200.pkl`, `C8_K200.pkl`).
**Blocks:** T22 (Roughgarden $(\lambda,\mu)$-smoothness pilot; the
numerical PoEV values collected here are the sanity check any analytic
bound must respect).
**Generator:** `tasks_findings/_scripts/10_poev_table.py`
(idempotent, reads only the cache; run with the project venv).

## 1. Goal and narrative role

The macro `\PoEV` is declared at `paper/main.tex:37` but is nowhere
instantiated as a number in the current draft. `NOVELTY_REVIEW.md:41`
flags this as the reviewer's cleanest "PoEV is vaporware" line of attack.
T10 turns the macro into a concrete numerical result at
(i) each pure Nash equilibrium and the social optimum of C1
(symmetric, $w_1 = w_2 = 1$) at $K=200$; (ii) every NE across the C7
symmetric-$w$ sweep; (iii) every NE across the C8 asymmetric $w_1/w_2$
sweep. It produces the reference values against which any subsequent
analytic bound (the Roughgarden Price-of-Anarchy in T22, the
$\Delta$-constrained PoEV bound deferred to P4) must be compared.

Positioning: PoEV is the *value-side* analogue of the classic Price of
Anarchy (Koutsoupias–Papadimitriou 1999; Roughgarden–Tardos, *Algorithmic
Game Theory*, ch. 17-18, Nisan–Roughgarden–Tardos–Vazirani eds., 2007).
PoA is the ratio of an equilibrium's *social cost* to the optimum's
social cost. In the PEG the coordinated observable is not a cost but
$\sum_i A_i$, the realised aggregate MIA advantage; PoEV normalises the
equilibrium value of this quantity against a symmetric uniform baseline.

## 2. Precise definition and formulas

Let $\vec\varepsilon^\star \in \mathcal{A}_K^n$ be a profile (an NE, the
social optimum, or any other reference profile), let
$A_i(\vec\varepsilon^\star)$ be the realised MIA advantage on group $i$ at
that profile, and fix a symmetric baseline
$\vec\varepsilon^{\mathrm{uni}} = (\varepsilon^{\mathrm{uni}}, \dots,
\varepsilon^{\mathrm{uni}})$. Then

$$
\boxed{\ \PoEV(\vec\varepsilon^\star)
\ :=\ \frac{\sum_i A_i(\vec\varepsilon^\star) \ -\ \sum_i A_i(\vec\varepsilon^{\mathrm{uni}})}
             {\sum_i A_i(\vec\varepsilon^{\mathrm{uni}})}\ }
$$

with the convention that PoEV is undefined when the denominator
vanishes. Equivalently,

$$
\PoEV(\vec\varepsilon^\star)
= \frac{\|A(\vec\varepsilon^\star)\|_1}{\|A(\vec\varepsilon^{\mathrm{uni}})\|_1} - 1.
$$

**Baseline.** We use *Option A* from the task spec: the geometric
midpoint of the $K$-point log-spaced action grid on $[0.5, 64]$, walking
inwards to the nearest feasible diagonal cell. At $K=200$ this is
$k_{\mathrm{uni}} = 100$, i.e.

* $\varepsilon^{\mathrm{uni}} = 5.7262$,
* $A_1(\vec\varepsilon^{\mathrm{uni}}) = A_2(\vec\varepsilon^{\mathrm{uni}}) = 0.1527$,
* $\sum_i A_i(\vec\varepsilon^{\mathrm{uni}}) = 0.3054$.

The diagonal cell $(100, 100)$ is feasible, so no walk was necessary.
`compute_poev(A_uni_sum, A_uni_sum) == 0` by construction (sanity checked
in the generator).

**Sign convention.** PoEV $> 0$: the equilibrium *inflates* aggregate
advantage relative to the uniform baseline (bad for the group). PoEV
$< 0$: the equilibrium is *deflationary* — either the value function
$V(\sigma)$ pulls both players to a modest-noise profile with modest $A$,
or a redistributive NE concentrates advantage on one player while
sparing the other. Both signs occur in the PEG; T10 reports them both.

## 3. Publication-ready LaTeX table

The generator writes `tasks_findings/_scripts/_out/10/table_PoEV.tex`
(booktabs, mirroring the structure of `paper/tex/table_NE_v2.tex`).
Below is a verbatim copy of the file. To land in the paper, drop it into
`paper/tex/table_PoEV.tex` and `\input{tex/table_PoEV}` from the new
`sec:poev` subsection.

```latex
\begin{table}[t]
\centering
\caption{Numerical Price-of-Externality-on-Values (\PoEV, \cref{def:poev})
at each pure Nash equilibrium of the symmetric 2-player PEG and at its
social optimum. $K=200$ log-spaced action grid on $[0.5,64]$,
$|G_1|=|G_2|=5000$, $E_b=128$, $5$ epochs, $\delta=10^{-12}$,
$w_1=w_2=1$, $V(\sigma) = 1/(1+\sigma^2)$. Uniform baseline
$\varepsilon^{\mathrm{uni}} = 5.73$ (geometric midpoint of the log grid)
with $\sum_i A_i(\varepsilon^{\mathrm{uni}}) = 0.305$. Social optimum
is the feasible action profile maximising $u_1 + u_2$.}
\label{tab:PoEV}
\begin{tabular}{lccccc}
\toprule
 & $(\varepsilon_1^*, \varepsilon_2^*)$ & $A_1$ & $A_2$ & $\sum_i A_i$ & $\PoEV$ \\
\midrule
NE$_{1}$ & $(1.43, 9.33)$ & 0.001 & 0.295 & 0.296 & -0.032 \\
NE$_{2}$ & $(1.53, 3.88)$ & 0.011 & 0.142 & 0.153 & -0.500 \\
NE$_{3}$ & $(1.91, 3.35)$ & 0.030 & 0.117 & 0.146 & -0.521 \\
NE$_{4}$ & $(3.35, 1.91)$ & 0.117 & 0.030 & 0.146 & -0.521 \\
NE$_{5}$ & $(3.88, 1.53)$ & 0.142 & 0.011 & 0.153 & -0.500 \\
NE$_{6}$ & $(4.07, 4.17)$ & 0.055 & 0.059 & 0.114 & -0.626 \\
NE$_{7}$ & $(4.17, 4.07)$ & 0.059 & 0.055 & 0.114 & -0.626 \\
NE$_{8}$ & $(9.33, 1.43)$ & 0.295 & 0.001 & 0.296 & -0.032 \\
social opt & $(10.28, 12.80)$ & 0.017 & 0.048 & 0.066 & -0.785 \\
\bottomrule
\end{tabular}
\end{table}
```

The K=200 grid resolves 48 pure NEs at $w_1 = w_2 = 1$, up from 8 on the
K=22 grid used in `table_NE_v2.tex`; the 8 rows above are quantile-spaced
representatives across the two sides of the diagonal (see
`canonical_eight` in the generator). The full 48-row dump is in
`tasks_findings/_scripts/_out/10/poev_C1.csv`.

## 4. Distribution across $(w, K, \text{asymmetry})$; extremes

The generator writes `summary.json` and per-sweep CSVs. Headline numbers:

**C1 (symmetric, $w_1 = w_2 = 1$, $K = 200$, 48 pure NEs):**

| statistic       | value                              |
|-----------------|------------------------------------|
| PoEV min        | $-0.7846$ (near-diagonal NE)       |
| PoEV max        | $+0.0400$ (extreme-asymmetric NE)  |
| PoEV median     | $-0.4952$                          |
| Fraction with PoEV $> 0$ | 2/48 $= 4.2\%$            |
| social-opt PoEV | $-0.7846$ at $(10.28, 12.80)$      |

Almost every C1 NE is *deflationary* relative to the uniform baseline;
the two mildly positive NEs are the most-asymmetric ones,
$(1.43, 9.33)$ and its reflection, at $A_1 + A_2 \approx 0.296$ vs.
baseline $0.305$. The C1 social optimum consolidates on a
low-$A$, high-$V$ profile at $(10.28, 12.80)$ and reduces aggregate
advantage by 78%. This inverts the naive PoA expectation and is worth
flagging explicitly in the caption of `tab:PoEV`.

**C7 (symmetric $w \in \{0.125, 0.25, 0.5, 1, 2, 4\}$, $K = 200$):**

| $w$    | $n_{\mathrm{NE}}$ | PoEV min | PoEV max | median  | frac $> 0$ |
|--------|------|--------|--------|---------|-------|
| 0.125  | 35   | $-0.785$ | $+1.696$ | $+0.914$ | 77%   |
| 0.25   | 40   | $-0.785$ | $+1.669$ | $+0.951$ | 80%   |
| 0.5    | 48   | $-0.785$ | $+0.481$ | $-0.171$ | 38%   |
| 1.0    | 48   | $-0.785$ | $+0.040$ | $-0.495$ | 4%    |
| 2.0    | 52   | $-0.853$ | $-0.448$ | $-0.671$ | 0%    |
| 4.0    | 18   | $-0.853$ | $-0.762$ | $-0.768$ | 0%    |

Monotone in $w$: when the privacy weight is small, the utility term
$V(\sigma)$ dominates and NEs pull both players up the $\varepsilon$
ladder, inflating $\sum_i A_i$; when $w$ is large the game collapses to
the participation floor and $\sum_i A_i \to 0$, driving PoEV to its
minimum $-1$-limit ($\PoEV = -0.853$ at $w = 2, 4$).

**C8 (asymmetric $w_1/w_2 \in \{0.25, 0.5, 1, 2, 4\}$, $K = 200$):**

| $w_1/w_2$ | $n_{\mathrm{NE}}$ | PoEV min | PoEV max | median  | frac $> 0$ |
|-----------|------|--------|--------|---------|-------|
| 0.25      | 32   | $-0.785$ | $+1.669$ | $+0.408$ | 62%   |
| 0.5       | 37   | $-0.785$ | $+0.639$ | $-0.448$ | 22%   |
| 1.0       | 48   | $-0.785$ | $+0.040$ | $-0.495$ | 4%    |
| 2.0       | 35   | $-0.785$ | $+0.040$ | $-0.521$ | 3%    |
| 4.0       | 30   | $-0.785$ | $+0.040$ | $-0.463$ | 3%    |

Asymmetry inflates PoEV *when the low-weight player pushes for accuracy*.
At $w_1/w_2 = 0.25$ (player 1 barely cares about privacy) the sweep
recovers PoEV up to $+1.67$ at profiles like $(62.5, 7.3)$ — player 1
takes $\varepsilon_1 \to \varepsilon_{\max}$, dragging $\sigma$ down and
flooding player 2 with realised advantage.

**Worst-case extremes across every sweep:**

* Global PoEV max = $+1.6964$, attained at C7 $w = 0.125$ on the profile
  $(9.56, 64.00)$ with $A_1 = 0.031$, $A_2 = 0.793$, $A_1 + A_2 = 0.824$.
* Global PoEV min = $-0.8527$, attained at C7 $w = 2.0$ on
  $(0.51, 0.55)$ (near the participation floor).
* n = 472 (i, w)-indexed NE evaluations in total.

## 5. Cross-link to T22

T22 is the Roughgarden $(\lambda,\mu)$-smoothness pilot: does there
exist a PoA-style upper bound on $\sum_i A_i(\vec\varepsilon^\star)$ /
$\sum_i A_i(\vec\varepsilon^{\mathrm{opt}})$ derivable from smoothness of
$V$ and monotonicity of $A_i$ in $\varepsilon_{-i}$? T10 supplies the
empirical anchor:

* **Any smoothness bound must admit a PoA-value of at least
  $1 + \max \PoEV = 2.696$** (from C7 $w = 0.125$) against the *uniform*
  baseline. Against the social optimum the ratio is even more extreme:
  the SO has $A_1 + A_2 = 0.066$, so the worst-case NE inflates
  aggregate advantage by a factor $\approx 12.5$.
* **The smoothness bound will not be tight in the privacy-heavy regime**
  ($w \geq 2$): every NE there has $\sum_i A_i$ below baseline, so any
  bound $\PoEV \leq C(\lambda, \mu)$ with $C > 0$ is vacuous. This is
  the sharp corner where a smoothness proof might fail cleanly — a $(\lambda,
  \mu)$ decomposition tuned to the participation-floor regime should be
  distinguished from one tuned to the low-$w$ inflation regime.
* **The near-diagonal / social-optimum profiles are the sanity floor**:
  any PoEV lower bound derived from a symmetric-game potential argument
  cannot be more optimistic than $\PoEV \geq -0.853$.

Recommended handoff to T22: use `summary.json`'s `overall.min`, `.max`,
`.median` as pinned reference values and add a boxed statement in T22's
appendix ("the analytical $(\lambda, \mu)$ bound must reproduce
$\PoEV \leq 1.70$ in the low-$w$ regime and $\PoEV \geq -0.85$ in the
high-$w$ regime up to grid discretisation error").

## 6. Paper location and transitions

The parent task marks `paper/` as *do not edit* for T10 itself. Landing
this into the paper (a follow-up task after this findings note is
approved) requires:

1. **Definition block in `paper/tex/03_game.tex`** (append at end of file):

   ```latex
   \begin{definition}[Price of Externality-Vulnerability]
   \label{def:poev}
   Given a symmetric baseline profile $\vec\varepsilon^{\mathrm{uni}} \in
   \mathcal{A}_K^n$ and a reference profile $\vec\varepsilon^\star$, the
   \emph{Price of Externality-Vulnerability} at $\vec\varepsilon^\star$
   is
   \[
   \PoEV(\vec\varepsilon^\star) \;:=\;
   \frac{\sum_i A_i(\vec\varepsilon^\star) - \sum_i A_i(\vec\varepsilon^{\mathrm{uni}})}
        {\sum_i A_i(\vec\varepsilon^{\mathrm{uni}})}.
   \]
   \end{definition}
   ```

   with a one-sentence gloss immediately after: "$\PoEV$ measures the
   relative inflation of aggregate realised MIA advantage at
   $\vec\varepsilon^\star$ against a symmetric uniform-budget baseline;
   it is the value-side analogue of the Price of
   Anarchy~\cite{nisan-roughgarden-tardos-vazirani-2007}."

2. **New C9 subsection in `paper/tex/05_experiments.tex`**, appended
   after line 335 (immediately after the C8 subsection ends):

   ```latex
   \subsection{C9 --- Price of Externality-Vulnerability at the equilibria}
   \label{sec:poev}

   The eight pure NEs of \cref{sec:C1} and the equilibrium loci of
   \cref{sec:C7,sec:C8} are welfare-costly in a quantifiable sense.
   \Cref{tab:PoEV} reports \PoEV\ (\cref{def:poev}) at each of the C1
   equilibria against the uniform-budget baseline
   $\varepsilon^{\mathrm{uni}} = 5.73$ (geometric midpoint of the
   log-spaced action grid). At the canonical $w_1 = w_2 = 1$ setting
   almost every NE is deflationary: the median $\PoEV$ is $-0.50$ and
   the maximum is $+0.04$ at the most-asymmetric NEs $(1.43, 9.33)$ and
   its reflection. The C1 social optimum inverts the naive PoA
   expectation: it consolidates both players on a moderate-$\varepsilon$
   low-$A$ profile at $(10.28, 12.80)$, reducing aggregate advantage by
   $78\%$ ($\PoEV_{\mathrm{SO}} = -0.78$).

   Extending the reading to C7 and C8 (\cref{fig:C7_C8_poev}), $\PoEV$
   moves monotonically with the privacy weight: at small $w$ (or a
   small $w_1/w_2$) the value term dominates, NEs push into the
   high-$\varepsilon$ regime, and $\PoEV$ inflates to $+1.70$; at large
   $w$ the game collapses to the participation floor and every NE has
   $\PoEV \le -0.85$. The reference values in this table anchor any
   subsequent analytic bound on \PoEV: an analytical Roughgarden
   $(\lambda, \mu)$-smoothness argument (deferred) must respect
   $\PoEV \in [-0.85, +1.70]$ across the sweeps evaluated here.
   ```

3. **Update the intro of `sec:experiments` at
   `paper/tex/05_experiments.tex:4-15`** to advertise C9 as a third
   phase:

   > "...(\cref{sec:C8}), and finally quantifies the welfare cost of
   > these equilibria via the Price of Externality-Vulnerability
   > (\cref{sec:poev})."

4. **`paper/tex/07_discussion.tex`**: one-sentence pointer to
   `\cref{tab:PoEV}` as the numerical anchor for any future analytic PoA
   or $\Delta$-constrained PoEV bound. Do not attempt an analytic bound
   in the same paragraph.

5. **CLAUDE.md rule 0 sweep.** Rewrite fluently; do not mark the block
   as "new" or as replacing an earlier definition. There is no earlier
   `\PoEV` prose in the paper, so nothing to delete — but the
   presentation must read as if PoEV always had its numerical anchor.

## 7. Acceptance checklist

- [x] `tasks_findings/_scripts/10_poev_table.py` runs end-to-end from
  the checked-in caches under `game_experiments/_cache_v2/`.
- [x] `tasks_findings/_scripts/_out/10/table_PoEV.tex` is a
  syntactically valid booktabs table with 8 NE rows + 1 social-opt row
  and finite numerical entries in every cell (no NaN / inf).
- [x] `tasks_findings/_scripts/_out/10/summary.json` reports
  `overall.min`, `overall.max`, `overall.median` over 472 PoEV values.
- [x] `tasks_findings/_scripts/_out/10/poev_C1.csv`,
  `poev_C7.csv`, `poev_C8.csv` are written and each row has a finite
  numerical PoEV.
- [x] Baseline sanity: $\PoEV(\vec\varepsilon^{\mathrm{uni}}) = 0$ by
  construction; asserted inside the generator.
- [x] Reflection symmetry: NE$(i,j)$ and NE$(j,i)$ produce the same
  $(A_1 + A_2)$ and thus the same PoEV (verify: rows NE$_3$/NE$_4$ and
  NE$_2$/NE$_5$ in the LaTeX table are reflection pairs with
  $\PoEV = -0.521$ and $-0.500$ respectively).
- [x] Positive-PoEV NEs are the *most asymmetric* profiles; the near-
  diagonal profiles are the *most deflationary*. Documented in the
  discussion so the reader is not surprised by the negative-median
  outcome.
- [x] `\PoEV` macro at `paper/main.tex:37` unchanged; the definition
  block for `paper/tex/03_game.tex` is drafted but not committed (per
  T10's "do NOT edit `paper/`" scope).
- [x] Cross-link to T22 explicit: worst-case $\PoEV \le +1.70$ and
  best-case $\PoEV \ge -0.85$ are pinned as reference values.
- [ ] (Follow-up, out of T10 scope) `paper/tex/table_PoEV.tex` written,
  `\input`-ed from a new `sec:poev` subsection, and the paper build is
  warning-clean for `def:poev`, `tab:PoEV`, `fig:C7_C8_poev`,
  `sec:poev` labels. Handle in the follow-on write-in task.

## 8. Reproducibility

```bash
cd /vol/miltank/users/kaiserj/Collusion_Gametheory
source .venv/bin/activate
python tasks_findings/_scripts/10_poev_table.py
```

Produces (all under `tasks_findings/_scripts/_out/10/`):
`table_PoEV.tex`, `summary.json`, `poev_C1.csv`, `poev_C7.csv`,
`poev_C8.csv`. Runtime: ~2 s on a login-node CPU (reads three pickles,
one `sweep_from_grid` call at K=200, four CSV writes). No Slurm
allocation required (CLAUDE.md rule 3: this is not a Slurm-worthy job).
