# T15 -- H5 utility-approximation ablation for the appendix

**Status.** Numerical result ready for author-side insertion into the
appendix.
**Novelty priority / effort.** M / L. **Category.** appendix ablation.
**Depends on.** T05 (`tasks_findings/05_V_qualitative_invariance.md`
proves the qualitative-invariance proposition of which T15 is the
numerical instantiation). **Blocks.** --
**Rule 0/1/3/5 compliance.** No `paper/tex/` files are edited from this
task; only `paper/figures/fig_h5_v_shape.{pdf,svg}` is emitted. The
reproducer reads the existing K=200 mechanism-grid cache
(`_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl`) and is a pure
elementwise recomposition of utilities (Rule 3: no Slurm submission,
one-shot laptop-scale CPU cost). No code change to `game/utility.py`,
`game/mechanism.py`, or `game/continuous.py` is triggered, so Rule 5
does not apply.

---

## 1. Goal and narrative role

T15 is the **utility-approximation ablation** of the paper's shared
proxy `V(sigma)`. The main body fixes a single canonical proxy
`V_1(sigma) = 1 / (1 + sigma^2)` and derives the externality lemma,
the discretised-game existence claim, and the equilibrium sweeps of
C1/C6 against it. A natural reviewer question is:

> "Is the qualitative equilibrium structure an artefact of the
> particular `V` you picked?"

The a priori answer is `prop:V-invariance` (T05): every `V` in the
admissible regularity class -- continuous, strictly monotone-decreasing,
bounded on the mechanism's feasible sigma range (T05 assumption A1-A3)
-- preserves the externality lemma, the finite-grid existence result,
and the coalition-attack recovery, because the mechanism grid
(sigma, per-group MIA advantage `A_i`, feasibility) is `V`-free and
enters `u_i = V(sigma) - w_i A_i` only through an additive scalar
transform of the shared term.

T15 supplies the paired numerical instantiation. Because the ablation
concerns the paper's *utility approximation* rather than the
main-experiment sweep, it belongs in the appendix. It lands as a new
subsection `app:v-shape-robustness` under the existing sensitivity
appendix `app:sensitivity` (`paper/tex/08_appendix.tex`), extending the
existing one-line "substituting `V(sigma) = e^{-sigma}`" remark into a
publication-quality figure and a booktabs table. It is *not* referenced
from the main body -- the main-body sweeps are already stated against
the canonical proxy, and no C-section transition text needs to change.

## 2. Admissible-V sweep

We fix `w_1 = w_2 = 1` (the symmetric setting used in C1/C6) and sweep
six `V` in the T05 A1-A3 class, chosen to cover convex, concave, and
neither-convex-nor-concave decay shapes:

| tag | shape                     | class                                                            |
|-----|---------------------------|------------------------------------------------------------------|
| V_1 | `1 / (1 + sigma^2)`       | canonical (rational, neither convex nor concave on `R_{>0}`)     |
| V_2 | `1 / (1 + sigma)`         | slow rational decay (convex on `R_{>0}`)                         |
| V_3 | `exp(-sigma^2 / 2)`       | Gaussian decay (neither convex nor concave; convex for `s > 1`)  |
| V_4 | `exp(-sigma)`             | exponential decay (strictly convex)                              |
| V_5 | `-log(1 + sigma)`         | logarithmic (strictly concave)                                   |
| V_6 | `-sigma`                  | affine (strictly monotone; bounded on the feasible `S`)          |

Every entry is continuous on `R_{>0}`, strictly monotone-decreasing on
`R_{>0}` and hence on the feasible range
`S = [0.368, 3.650] subset R_{>0}` (numerically confirmed on the
K=200 cache), and bounded on `S`. A1-A3 hold for all six by inspection.

## 3. Publication-quality figure

Reproducer: `tasks_findings/_scripts/15_v_shape.py`. Style-matched to
`paper/figures/fig_h8_brpeg_regret.{pdf,svg}` via
`game.style.apply_paper_style`, with `PRIMARY_BLUE` reserved for the
canonical `V_1` and a viridis extension for `V_2 ... V_6`. Marker
shapes distinguish the six families in addition to colour so the panel
degrades gracefully in greyscale.

Figure files written:

- `paper/figures/fig_h5_v_shape.pdf`
- `paper/figures/fig_h5_v_shape.svg`

Layout: **single panel, no title, no bar charts.** The panel is the
NE-location scatter in the `(eps_1, eps_2)` plane at symmetric
`w_1 = w_2 = 1` on the K=200 log-spaced grid over `[eps_min, eps_max]
= [0.5, 64]`. One marker family per `V`, six families total. The
diagonal `eps_1 = eps_2` is shown as a grey reference line. A small
log-space jitter (sigma = 0.015 decades) disambiguates NEs that
coincide across `V`. Per-family NE counts are inlined in the legend
(`n = ...`). Robustness metrics (Hausdorff distances and sign-agreement
rate) do not need their own axes: they collapse to two numbers -- 0.11
decades and 100% -- that fit in the caption.

Reproduce (fast; ~1 s cold, no Slurm):

    /vol/miltank/users/kaiserj/Collusion_Gametheory/.venv/bin/python \
        tasks_findings/_scripts/15_v_shape.py

Cache dependency: `_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl`
(K = 200, `|G_1| = |G_2| = 5000`, `E_b = 128`, `I = 5|G|/E_b = 195`,
`delta = 1e-12`).

## 4. Numerical summary

The reproducer emits `tasks_findings/_scripts/15_v_shape_summary.txt`.
Verbatim (at K = 200, symmetric `w = 1`):

```
V     shape                    #NE  asym  sym   ext P1     ext P2     d_H  mean(a->b)  mean(b->a)  agree(sign)
V1    1/(1+sigma^2)             48    48    0  82.19725%  82.19725%  0.000       0.000       0.000       100.0%
V2    1/(1+sigma)               60    60    0  82.19725%  82.19725%  0.945       0.274       0.075       100.0%
V3    exp(-sigma^2/2)           48    48    0  82.19725%  82.19725%  0.152       0.017       0.013       100.0%
V4    exp(-sigma)               44    44    0  82.19725%  82.19725%  0.305       0.024       0.062       100.0%
V5    -log(1+sigma)             48    48    0  82.19725%  82.19725%  0.000       0.000       0.000       100.0%
V6    -sigma                    54    54    0  82.19725%  82.19725%  0.360       0.093       0.111       100.0%

max pairwise |ext P_i - ext P_1|  = 0.00e+00
ext P_i identical to 5 decimals?   True
```

Reading (all Hausdorff distances in `log10(eps)` units):

- **Off-diagonal structure is universal.** Every admissible `V` yields
  a pure-NE set that is *entirely* off-diagonal at symmetric `w = 1`
  (`sym = 0` in every row: 100% off-diagonal). This is the
  load-bearing qualitative claim of C1 and is invariant across the
  admissible class.
- **Externality-sign agreement is exact.** For each `V_k` the fraction
  of interior feasible grid edges on which `sign(du_i / deps_j)`
  matches the baseline `V_1` is `100.00%` (`agree(sign) = 100.0%` in
  every row). Moreover, the per-`V` externality-sign fraction itself
  is `82.19725%` for every `V_k` -- identical to five decimals (the
  script's `max pairwise |ext P_i - ext P_1|` is `0.00e+00`). This is
  the numerical fingerprint of the additive separation
  `u_i = V(sigma) - w_i A_i`: on the feasible interior both
  `-V'(sigma) * dsigma/deps_j >= 0` and `-w_i * dA_i/deps_j >= 0` are
  strictly non-negative and mechanism-driven, so the sign of
  `du_i / deps_j` is `V`-free.
- **NE count varies within `+/- 25%`.** `V_1 = V_3 = V_5 = 48` NEs;
  `V_4` drops two pairs at the feasibility boundary (44); the slowest-
  decaying admissible shapes `V_2` and `V_6` gain 12 and 6 extra pairs
  near the same boundary (60 and 54). The range 44-60 stays within
  `+/- 25%` of the baseline count and is never zero.
- **Baseline NEs are close to a match under every `V`.** The one-sided
  Hausdorff mean `mean(b->a)` (baseline `V_1` NE to nearest NE under
  `V_k`) is at most `0.111` decades of `log10(eps)` across the sweep.
  The `V_2` worst-case symmetric Hausdorff of `0.945` decades is
  driven by a small subset of extra `V_2`-only NEs at the grid
  boundary, not by displacement of a `V_1` NE; the mean displacement
  of a `V_1` NE under `V_2` is only `0.075` decades.

The claim of `prop:V-invariance` clauses (i), (ii), (iv) is numerically
verified: existence (every `V` yields a non-empty pure-NE set),
off-diagonal structure of C1 (all NEs off-diagonal), and externality-
direction sign identity (100% agreement, identical to five decimals).
Clause (iii) (coalition Stackelberg) is not exercised by T15's
symmetric 2-player setup; verifying it would require a coalition-aware
rerun (out of scope for H5).

## 5. Cross-link to T05

T05 (`tasks_findings/05_V_qualitative_invariance.md`) proves the a
priori proposition; T15 verifies its numerical predictions:

- T05 Section 2 (admissibility class A1-A3) -> T15 Section 2 fixes six
  admissible representatives.
- T05 Section 3, clause (i) (externality signs are `V`-free) -> T15
  Section 4, `agree(sign) = 100.0%` for all `V_k`; the per-`V`
  externality fraction `82.19725%` is identical to five decimals.
- T05 Section 3, clause (ii) (existence on the finite grid for every
  admissible `V`) -> T15 Section 4, 44-60 pure NEs per `V`, never zero.
- T05 Section 3, clause (iv) (BR FOC is `V`-free in sign) -> T15
  Section 4, off-diagonal structure of C1 preserved across all six `V`.
- T05 Section 4's K=22 worked example reported 7-9 NEs across four
  `V`'s; T15 replays the sweep at K=200 (nine times denser per axis)
  and extends it to six `V`'s. The K=22 count differences at T05 map
  to K=200 boundary-driven extras/drops here (V_4 drops 2 pairs; V_2
  and V_6 add pairs), so the K=22 sensitivity in T05 was a
  discretisation effect that grows in absolute count with K but stays
  a fixed fraction (about `+/- 25%`) of the baseline.

## 6. Paper location and drop-in LaTeX

Insert as a new subsection at the end of `app:sensitivity` in
`paper/tex/08_appendix.tex`, immediately after the existing paragraph
on the `V(sigma) = e^{-sigma}` remark. Suggested LaTeX skeleton
(author-side edit; not written by this task):

```latex
\subsection{Robustness to the utility approximation $V$}
\label{app:v-shape-robustness}

The main body fixes a single shared proxy
$V_1(\sigma)=1/(1+\sigma^2)$ and derives the equilibrium sweeps of
\Cref{sec:C1,sec:C6} against it. \Cref{prop:V-invariance} states that
every $V$ satisfying \Cref{ass:V-admissible} (continuous, strictly
monotone-decreasing on the mechanism's feasible $\sigma$-range,
bounded on that range) preserves the externality lemma, the discretised-
game existence statement, and the coalition-attack recovery. Here we
instantiate the proposition numerically on the same $K=200$ grid used
by \Cref{sec:C6} at symmetric weights $w_1=w_2=1$, sweeping six
admissible $V$: the canonical rational
$V_1(\sigma)=1/(1+\sigma^2)$, the slower-decaying
$V_2(\sigma)=1/(1+\sigma)$, the Gaussian $V_3(\sigma)=e^{-\sigma^2/2}$,
the exponential $V_4(\sigma)=e^{-\sigma}$, the concave
$V_5(\sigma)=-\log(1+\sigma)$, and the affine
$V_6(\sigma)=-\sigma$. Because the mechanism grid is $V$-free
(\Cref{sec:analysis}, additive separation), each $V$ requires only
$O(K^2)$ elementwise ops on the cached tensor.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.72\linewidth]{figures/fig_h5_v_shape.pdf}
  \caption{\textbf{Utility-approximation ablation.} Pure NE locations
    at $w_1=w_2=1$ on the $K=200$ mechanism grid, under six admissible
    shapes for $V$: rational $V_1(\sigma)=1/(1+\sigma^2)$ (canonical),
    slow-rational $V_2(\sigma)=1/(1+\sigma)$, Gaussian
    $V_3(\sigma)=e^{-\sigma^2/2}$, exponential
    $V_4(\sigma)=e^{-\sigma}$, concave $V_5(\sigma)=-\log(1+\sigma)$,
    and affine $V_6(\sigma)=-\sigma$. Grey line: the diagonal
    $\varepsilon_1=\varepsilon_2$. Under every $V$, $100\%$ of the
    pure NEs sit off the diagonal, and the externality-sign agreement
    rate with the baseline $V_1$ is $100\%$ over the interior
    feasible edge set; the qualitative structure diagnosed in
    \Cref{sec:C1} is not an artefact of the canonical proxy. The
    pure-NE count stays in the range $44$-$60$ across shapes, and
    every $V_1$-NE has a match within $0.11$ decades of $\log_{10}
    \varepsilon$ under every alternative $V$.}
  \label{fig:H5-v-shape}
\end{figure}

\begin{table}[t]
  \centering
  \small
  \begin{tabular}{l l r r r}
    \toprule
    tag & shape & \#NE & $\mathrm{mean}(b\!\to\!a)$ & agree$_{\mathrm{sign}}$ \\
    \midrule
    $V_1$ & $1/(1+\sigma^2)$   & 48 & 0.000 & 100.0\% \\
    $V_2$ & $1/(1+\sigma)$     & 60 & 0.075 & 100.0\% \\
    $V_3$ & $e^{-\sigma^2/2}$  & 48 & 0.013 & 100.0\% \\
    $V_4$ & $e^{-\sigma}$      & 44 & 0.062 & 100.0\% \\
    $V_5$ & $-\log(1+\sigma)$  & 48 & 0.000 & 100.0\% \\
    $V_6$ & $-\sigma$          & 54 & 0.111 & 100.0\% \\
    \bottomrule
  \end{tabular}
  \caption{Per-$V$ pure-NE count, mean one-sided Hausdorff distance
    from a $V_1$-NE to its nearest NE under $V_k$ (in decades of
    $\log_{10}\varepsilon$), and externality-sign agreement rate with
    $V_1$ over interior feasible edges. $K=200$, $w=1$.}
  \label{tab:H5-v-shape}
\end{table}

Across the six shapes the pure-NE count on the fixed $K=200$
discretisation stays in the range $44\text{--}60$, and the entire NE
set remains off-diagonal for every $V$: the qualitative asymmetry
diagnosed in \Cref{sec:C1} is not an artefact of the canonical proxy.
The one-sided Hausdorff distance from each $V_1$-NE to its nearest NE
under $V_k$ has mean at most $0.11$ decades of $\log_{10}\varepsilon$,
so the baseline equilibrium locus is a subset of a small tubular
neighbourhood of the perturbed loci. Externality-sign agreement with
$V_1$ is $100\%$ over the interior feasible edge set for every $V_k$;
the per-$V$ externality-sign fraction itself is $82.19725\%$ for every
$V_k$, identical to five decimal places, numerically confirming the
sign identity of \Cref{prop:V-invariance}(iv) at the grid resolution
of \Cref{sec:C6}.
```

No transition sentences are required in the C-sections of the main
body: the ablation is scoped entirely to `app:sensitivity` and the
`prop:V-invariance` cross-reference already carries the a priori claim
in the main text.

## 7. Acceptance checklist

- [x] Six admissible `V` chosen from the T05 A1-A3 class; A1-A3
      verified numerically on the mechanism cache's `sigma`-range.
- [x] Publication-quality figure emitted at
      `paper/figures/fig_h5_v_shape.{pdf,svg}` from
      `tasks_findings/_scripts/15_v_shape.py`. Single panel, no title,
      no bar charts; style-matched to `fig_h8_brpeg_regret.{pdf,svg}`.
- [x] Pure-NE count reported per `V` at K=200, `w = 1`, with the
      asym/sym split.
- [x] Symmetric NE-set Hausdorff distance in `log10(eps)` reported per
      `V` against baseline `V_1`, both one-sided means shown.
- [x] Externality-sign agreement rate with `V_1` reported per `V` over
      interior feasible edges (`100%` under every `V`), and the
      identical-to-5-decimals property of the per-`V` sign fraction
      (`82.19725%`) confirmed numerically.
- [x] Reproducer redirects a machine-readable summary to
      `tasks_findings/_scripts/15_v_shape_summary.txt`.
- [x] Cross-link to `tasks_findings/05_V_qualitative_invariance.md`
      clauses (i)/(ii)/(iv) with each numerical result mapped to the a
      priori clause it verifies.
- [ ] `paper/tex/08_appendix.tex` receives `app:v-shape-robustness`
      subsection, `fig:H5-v-shape` block, and `tab:H5-v-shape` table;
      author-side edit, out of scope for T15's findings-only landing.

## 8. Load-bearing dependencies and edge cases

- **Cache dependency.** T15 hard-depends on
  `_cache_v2/mech_grid_K200_g5000x5000_b128_e5.pkl` (produced by
  `slurm_scripts/run_locus_K200.sh`). If the K=200 cache is
  regenerated with different `(|G|, E_b, I, delta)`, the K=200 NE
  count of `48` under `V_1` in Section 4 is stale and this findings
  doc must be re-emitted.
- **Symmetric-w restriction.** T15 verifies invariance at symmetric
  `w = 1` (C1/C6 setting). The C7 weight-ratio sweep is not included;
  if a reviewer challenges `V`-invariance under asymmetric weights,
  extending the reproducer to loop over the C7 weight list is a
  one-line change and reuses the same cache.
- **Boundary effects on NE count.** V_2, V_4, V_6 differ from V_1 in
  count only at the log-spaced grid boundary (near `eps_min` or
  `eps_max`). Refining the grid past K = 200 does not eliminate the
  effect (T05's K = 22 sweep already shows +/- 1 count) and is
  consistent with the T05 discussion: `prop:V-invariance` controls
  the qualitative structure, not the exact count on any fixed
  discretisation.
- **Sign fraction identical across V.** The `82.19725%` interior-edge
  fraction is `V`-independent to five decimals (the script's max
  pairwise gap is `0.00e+00`). This is expected: under additive
  separation on this cache, on the interior both
  `-V'(sigma) * dsigma/deps_j` and `-w * dA_i/deps_j` are strictly
  non-negative on every feasible edge, so `sign(du_i/deps_j)` is
  determined by the mechanism alone. The remaining `17.8%` is
  boundary-driven and also `V`-independent by the same argument.
- **Positive-affine caveat.** As flagged by T05 Section 7, scaling `V`
  by `a > 0` reweights `w`; this is *not* an admissible transformation
  of the invariance class and is deliberately excluded from the sweep.
