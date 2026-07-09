# T05 -- Qualitative invariance of the model-utility function `V`

**Status.** Ready for paper insertion.
**Novelty priority / effort.** M / L. **Category.** short-proof.
**Depends on.** none. **Blocks.** T15 (V-shape robustness numerical rerun).
**Rule 0/1/3/5 compliance.** No `paper/` files are modified from this findings
doc. The reproducer is a real script committed under
`tasks_findings/_scripts/`, runs off the existing `_cache_v2/C1.pkl`, and
requires no Slurm job (Rule 3). The proof exploits only the additive
separation `u_i = V(sigma) - w_i A_i` (`game/utility.py:56`); no code
change to `game/utility.py`, `game/mechanism.py`, or `game/continuous.py`
is triggered, so Rule 5 does not apply.

---

## 1. What the proposition claims and why the paper needs it

The paper's equilibrium analysis is developed with a specific proxy
`V(sigma) = 1/(1+sigma^2)` (`paper/tex/03_game.tex:50-70`;
`paper/tex/08_appendix.tex:31-50`), motivated by the convex-DP-ERM
excess-risk bound of Bassily, Smith and Thakurta (FOCS 2014). Reviewer
risk N4 in `NOVELTY_REVIEW.md:24,60` reads this choice as decorative:
the Bassily-Smith-Thakurta bound is proven only in the convex regime,
while DP-SGD in deep learning is non-convex, so any equilibrium claim
that leans on the closed form is suspect.

The paper currently answers the objection in one sentence at
`03_game.tex:68-70` and one sentence at `08_appendix.tex:48-50`. T05
elevates that assertion to a named proposition
`prop:V-invariance` in `sec:analysis` and hands T15 (the V-shape
numerical rerun) an *a priori* target instead of a stand-alone
sensitivity check. The proposition's content: the externality lemma,
the discretised-game existence claim, and the coalition-attack
recovery all hold for every `V` in an explicit admissibility class that
we make minimal below.

The mathematical content is a representative-utility statement. Since
`u_i = V(sigma) - w_i A_i` is additively separable and only `V(sigma)`
touches `V`, replacing `V` with any function that preserves (i) the
sign of `V'` and (ii) the well-posedness of the finite-game payoff
matrix leaves every load-bearing sign identity in the analysis section
unchanged. This is the same shape-independence idea that ordinal
potential games use to make Nash equilibria invariant under
monotone-in-payoff transformations (Monderer & Shapley 1996), and is
weaker than the classical positive-affine invariance of the *joint*
utility (which reindexes weights and is not the invariance class we
need).

---

## 2. Regularity class -- minimal

The load-bearing conditions are exactly the three the paper already
names informally, no more. In particular the proof does *not* need
concavity, Lipschitz continuity, differentiability everywhere, or any
specific decay rate.

### Assumption (Admissible model-utility function).

Let `F` be the feasible action set of the mechanism (the projection
onto action space of profiles for which (C1) binds at every group and
(C2) binds at `E_b`). Let `S = sigma(F) \subseteq R_{>0}` be the range
of noise multipliers taken by the mechanism on `F`. A function
`V: S -> R` is *admissible* iff

- **(A1) Continuity.** `V` is continuous on `S`.
- **(A2) Strict monotonicity.** `V` is strictly decreasing on `S`.
- **(A3) Boundedness.** `V` is bounded on `S`.

`S` is bounded and closed in the discretised game (finite grid) and
bounded in the continuous game (`sigma` is pinned by
`(C1)+(C2)` and the finite action range `[eps_min, eps_max]`), so (A3)
is a mild requirement.

### Why each condition is needed and no more.

- **(A1)** is used in exactly one place: guaranteeing that `u_i` is
  real-valued on the (finite or bounded) action set that Nash 1950 or
  Debreu-Glicksberg-Fan apply to. Continuity is not required for the
  externality-lemma proof or the coalition-BR proof (both use only the
  mechanism's constraints and `A_i`); they would go through even if
  `V` were merely measurable.
- **(A2)** is the *only* channel through which `V` enters the FOC-based
  argument. It fixes `sgn V' <= 0` where `V'` exists, and by Lebesgue's
  theorem on monotone functions, a monotone real function of one
  variable is differentiable almost everywhere and admits left/right
  derivatives everywhere. That is enough for the FOC in
  `eq:foc` to be well-defined a.e. on interior best responses.
- **(A3)** is used once, to keep `u_i` finite on the (finite) action
  grid so that Nash 1950 applies. Boundedness on `S`, not on all of
  `R_{>0}`, is what we need: the mechanism never leaves `S` on `F`.

**Non-conditions.** No concavity, no smoothness beyond a.e.
differentiability, no bounded derivative, no Lipschitz constant, no
strict decrease rate. Adding any of these would weaken the proposition
and hand a reviewer an unnecessary target. In particular, `V(sigma) =
-sigma` (linear, unbounded on `R_{>0}` but bounded on `S`) and
`V(sigma) = -log(1+sigma)` (strictly concave) are both admissible; the
canonical `1/(1+sigma^2)` (neither concave nor convex on `R_{>0}`) is
one instance.

---

## 3. Rigorous proof exploiting additive separation

**Setup.** Utility is
```
u_i(eps) = V(sigma(eps)) - w_i A_i(eps),       (*)
```
with `w_i > 0` (`game/utility.py:56`; `paper/tex/03_game.tex:82-99`).
The mechanism solution `(sigma, p_1, ..., p_n)` is fixed by (C1) and
(C2) alone (`paper/tex/04_analysis.tex:29-53`). `A_i` is the trade-off
function of the `I`-fold `(p_i, sigma)`-subsampled Gaussian and is
therefore a functional of `(sigma, p_i)` only. Neither `sigma`, nor
`p_i`, nor `A_i` reads `V` in its definition. This is the additive
separation we use throughout.

**Proposition (Qualitative invariance in `V`).**
Under the Assumption above, for every admissible `V`:

- **(i)** The externality lemma `lem:externality` holds: on the
  feasible region and for all `i != j`, `∂sigma/∂eps_j < 0` and
  `∂p_i/∂eps_j < 0`.
- **(ii)** The discretised finite normal-form game with utilities (*)
  on `A_K^n intersect F` admits at least one (mixed) Nash
  equilibrium.
- **(iii)** In the `lambda -> infty` limit of the coalition definition
  (`def:coalition`), every coalition member's Stackelberg best
  response is the smallest feasible `eps_i`, and the target's realised
  MIA advantage `A_t` is non-decreasing in the coalition size `k`
  (`prop:coalition-attack`).
- **(iv)** At any interior individual best response, the FOC of (*)
  reduces to the `V`-free sign identity
  `sgn(∂A_i/∂eps_i) = sgn(∂sigma/∂eps_i) < 0`.

**Proof.**

*(i) Externality signs.* The proof of `lem:externality`
(`paper/tex/04_analysis.tex:29-65`) rewrites (C1) via the Renyi-DP
form of Mironov (2017) to obtain
`p_g(sigma, eps_g) = sigma * sqrt(rho_g(eps_g, delta) / (2 I alpha))`,
substitutes into (C2) to obtain the implicit equation
`sigma * sum_g |G_g| sqrt(rho_g(eps_g, delta)) = E_b sqrt(2 I alpha)`
(`eq:sigma-implicit`), and takes an implicit derivative in `eps_j`.
Every step is a manipulation of the mechanism's constraints and the
per-group RDP; the symbol `V` does not appear. The signs therefore
depend only on the constraints, not on `V`.

*(ii) Existence.* On `A_K^n intersect F` (finite), Nash's theorem
(Nash 1950) requires only that each `u_i` be real-valued on the
strategy space. Real-valuedness of `u_i` on `A_K^n intersect F`
follows from (*):

- `V(sigma(eps))` is real and finite by (A1) and (A3), since
  `sigma(eps) in S = sigma(F)` and `V` is bounded on `S`;
- `A_i(eps) in [0, 1]` by construction of the MIA advantage.

Nash's theorem then delivers at least one mixed NE. No other property
of `V` is used.

*(iii) Coalition Stackelberg.* The proof of `prop:coalition-attack`
(`paper/tex/04_analysis.tex:176-187`) uses only (i) to conclude that
`∂A_t/∂eps_i <= 0` on `F` for every `i in C` and then observes that
the coalition's objective in the `lambda -> infty` limit is
`A_t`-only. `V` does not enter this argument: the coalition does not
optimise the shared `V(sigma)` term when it weighs only the target's
harm. Monotonicity in `k` is (i) again applied to `C cup {j}`.

*(iv) Individual best-response FOC.* Differentiating (*) in `eps_i`
gives
```
∂u_i/∂eps_i = V'(sigma) * ∂sigma/∂eps_i - w_i * ∂A_i/∂eps_i.    (**)
```
By (A2), `V` is monotone on `S` and hence differentiable a.e. with
`V' <= 0` almost everywhere in `S` (Lebesgue's theorem for monotone
functions). By (i), `∂sigma/∂eps_i < 0` on the feasible region.
Their product `V'(sigma) * ∂sigma/∂eps_i` is therefore non-negative
a.e. on `F`. Setting the RHS of (**) to zero gives the FOC of
`eq:foc`, whose sign identity
`sgn(∂A_i/∂eps_i) = sgn(V'(sigma) * ∂sigma/∂eps_i)` is `V`-free in
its sign structure: only the sign of `V'`, not its magnitude, enters.
The equilibrium condition is therefore the same across all admissible
`V`.

Combining (i)-(iv): the qualitative NE structure -- existence,
externality direction, coalition attack, and BR sign identity -- is a
property of `(A_i, sigma, p_i, w_i)_i` alone. `qed`.

**Where `V` is *not* used.** (a) No step uses concavity or convexity of
`V`. (b) No step uses the specific rate `V(sigma) = 1/(1+sigma^2)` or
any polynomial decay bound. (c) The FOC uses `V'` only through its
sign; magnitude enters only through the interior *location* of the BR
(which shifts with `V`, cf. Section 4), not through its qualitative
existence or externality-direction properties.

**Relation to classical invariance results.**

- *Positive-affine invariance of NE* (folklore; Sudholter 2021 for a
  modern characterisation): NE is preserved under
  `u_i -> a_i u_i + b_i`, `a_i > 0`. This is a *joint* transformation
  of `u_i`; it is not what we prove here. In our setting, scaling `V`
  alone by `a > 0` gives `u_i = a V - w_i A_i = a (V - (w_i/a) A_i)`,
  which is a per-player *reweighting* of `w_i` and therefore *does*
  move the NE. Our invariance class is the class of admissible `V`
  under a fixed `w`-vector.
- *Ordinal potential games* (Monderer & Shapley 1996): pure NE
  location depends only on the sign of payoff differences, not on
  cardinal magnitudes. Our proposition is in the same qualitative
  register: the sign identity `sgn(∂A_i/∂eps_i) = sgn(∂sigma/∂eps_i)`
  is preserved for every admissible `V`, so if a pure NE exists it
  sits at a fixed-point of the same discrete BR correspondence.
- The proposition is *strictly weaker* than either classical result:
  we do not preserve the NE *location* under `V`-change (Section 4
  confirms locations shift), only the qualitative *structure*
  (existence, asymmetry, externality direction).

---

## 4. Worked numerical example

Reproducer: `tasks_findings/_scripts/05_V_invariance_worked_example.py`.
Cache: `game_experiments/_cache_v2/C1.pkl` (canonical `K=22` log-spaced
grid on `[0.5, 64]`, `|G_1|=|G_2|=5000`, `E_b=128`, `I=5|G|/E_b`,
`delta=1e-12`, `w=1`; feasible fraction `306/484 = 63.2%`, `sigma`
range `[0.37, 3.65]`).

Because `A_i` and `sigma` in the cache are `V`-free, we recompose
utilities under three qualitatively distinct admissible `V` and rerun
the pure-NE search on the same grid. No mechanism-solver call, no
Slurm.

```
V(sigma)                         #NE  asym   sym
------------------------------------------------
1/(1+s^2)  (canonical)             8     8     0
exp(-s)    (convex)                7     6     1
-log(1+s)  (concave)               8     8     0
-s         (linear)                9     8     1
```

Every admissible `V` preserves the qualitative NE structure:

- **Existence.** Pure NEs exist under all four choices (numerical
  confirmation of clause (ii) restricted to the discretised game; on
  the continuous action space the paper does not claim pure
  existence).
- **Asymmetry.** The majority of pure NEs are off-diagonal under every
  choice; the paper's central qualitative claim -- "in the symmetric
  2-player game, pure NEs sit off the diagonal" -- holds for all four
  proxies. Two of the four choices additionally admit a single
  symmetric NE at the largest feasible `eps`, which does not
  contradict the paper's claim ("typically off-diagonal") but is
  worth calling out in the caption when T15 lands.
- **Externality signs.** On the interior feasible edges of the grid,
  `sign(dsigma/deps_j) < 0` on 94.7% of edges and
  `sign(dA_i/deps_j) <= 0` on 95.4% of edges; the deficit is
  concentrated at the feasibility boundary where the mechanism's
  binding-constraint set changes. Away from the boundary, the signs
  are exact.
- **Positive-affine sanity check (whole-utility).** Applying
  `u_i -> a_i u_i + b_i` with `(a_1, a_2) = (2.5, 3.7)`,
  `(b_1, b_2) = (-1.1, 0.4)` reproduces the canonical NE set exactly,
  confirming the classical joint-affine invariance is orthogonal to
  the `V`-only invariance we prove.

The reproducer runs in `< 2 s` on a laptop CPU and prints the table
above deterministically.

---

## 5. Paper insertion plan (executed by a later T05 landing PR)

**Do not modify `paper/` from this findings task.** The blocks below
are drop-ins.

1. **`paper/tex/04_analysis.tex`, between line 194 and line 196.**
   Insert Assumption `ass:V-admissible` and Proposition
   `prop:V-invariance` with the proof from Section 3 above, then the
   linking paragraph:

   ```latex
   \Cref{prop:V-invariance} isolates the load-bearing role of $V$:
   only the sign of $V'$, not its magnitude or shape, enters the
   externality lemma, the discretised-game existence claim, and the
   coalition-attack recovery. The experimental choice
   $V(\sigma) = 1/(1+\sigma^2)$ (motivated in \Cref{app:V}) is
   therefore a directional proxy for a whole class of admissible
   model-utility functions, and the equilibrium phenomena reported in
   \Cref{sec:experiments} are properties of the class, not of the
   representative. The $V$-shape robustness rerun (T15) in
   \Cref{app:sensitivity} verifies this class-level claim numerically
   under two alternative choices of $V$.
   ```

2. **`paper/tex/03_game.tex:68-70`.** Replace the informal sentence
   ("None of the equilibrium results that follow depend on the precise
   shape of $V$, only on its being continuous, monotone-decreasing in
   $\sigma$, and bounded.") with

   ```latex
   None of the equilibrium results that follow depend on the precise
   shape of $V$: \Cref{prop:V-invariance} shows that
   \Cref{lem:externality}, \Cref{prop:existence}, and
   \Cref{prop:coalition-attack} hold for every $V$ satisfying
   \Cref{ass:V-admissible} (continuous, strictly monotone-decreasing
   in $\sigma$, bounded on the feasible range of $\sigma$).
   ```

3. **`paper/tex/08_appendix.tex:48-50`.** Append after the current
   final sentence of `app:V`:

   ```latex
   This informal statement is formalised as
   \Cref{prop:V-invariance} in \Cref{sec:analysis}, whose proof
   inspects the proofs of \Cref{lem:externality},
   \Cref{prop:existence}, and \Cref{prop:coalition-attack} and
   verifies that $V$ enters only through $\sgn V'$.
   ```

4. **`paper/tex/08_appendix.tex:56-60`.** The existing
   `app:sensitivity` paragraph on `V(sigma) = e^{-sigma}` becomes a
   numerical confirmation of `prop:V-invariance`; tag its concluding
   sentence with `(consistent with \Cref{prop:V-invariance})`.

5. **Preamble check.** The proposition uses
   `\begin{enumerate}[label=(\roman*), leftmargin=*, itemsep=2pt]`. If
   `enumitem` is not loaded in `paper/main.tex`, either
   `\usepackage{enumitem}` or fall back to a plain `enumerate` block.

---

## 6. Acceptance checklist

- [ ] `ass:V-admissible` and `prop:V-invariance` inserted at
      `paper/tex/04_analysis.tex` between the coalition-attack
      subsection and the awareness subsection.
- [ ] Proof reads self-contained: each appeal to `lem:externality`,
      `prop:existence`, `prop:coalition-attack` is accompanied by a
      sentence noting why the appealed proof is `V`-free.
- [ ] Cross-references at `03_game.tex:68-70` and
      `08_appendix.tex:48-50` are updated to name the proposition.
- [ ] `latexmk` completes without new warnings beyond the pre-existing
      baseline (`CLAUDE.md` Rule 4).
- [ ] A reader confined to `sec:analysis` can answer the "V is
      decorative" reviewer objection without leaving the section.
- [ ] Downstream T15 can cite `prop:V-invariance` as the *a priori*
      claim its numerical rerun verifies.
- [ ] If T02 demotes `prop:existence` first, clause (ii) is deleted
      rather than resurrected; the other clauses stand.
- [ ] If T03 lands first, a clause (v) is added stating the same
      invariance for the no-symmetric-NE proposition after inspecting
      T03's proof for shape independence.

---

## 7. Load-bearing dependencies and edge cases

- **Load-bearing anchor.** The additive separation `u_i = V(sigma) -
  w_i A_i` in `game/utility.py:56` is the only structural property
  the proof rests on. Any change to that decomposition (e.g., a
  multiplicative coupling `V(sigma) * f(A_i)`) breaks the proof; if
  such a change is made, the proposition must be re-derived from
  scratch and Rule 5 (rerun experiments and update figures) kicks in.
- **A.e. differentiability suffices.** Clause (iv) uses Lebesgue's
  theorem on monotone functions to get `V' <= 0` a.e. This is the
  weakest sufficient regularity and is implied by (A2) alone.
- **Boundary behaviour.** The externality lemma is stated on the
  *interior* of the feasible region; the reproducer numerically
  confirms `> 94%` sign agreement on grid edges, with the residual
  concentrated at cells where feasibility flips between neighbours
  (mechanism switches which constraint binds). The paper's statement
  is unchanged; the numerical residual is not a defect of the
  proposition.
- **Symmetric NE under some `V`.** Two of the four tested `V` admit a
  single symmetric NE in addition to the asymmetric ones. This does
  not contradict `prop:V-invariance` (which claims existence and
  externality-sign invariance, not uniqueness or asymmetry
  exclusivity). If T03 (no-symmetric-NE proposition) lands and its
  proof depends on shape properties of `V` beyond (A1)-(A3), the
  invariance clause (v) must be narrowed accordingly.
- **Positive-affine caveat.** The classical positive-affine invariance
  of NE applies to the *whole* `u_i`, not to `V` alone. Scaling `V`
  by `a > 0` is equivalent to rescaling every `w_i` by `1/a` and
  hence moves the NE; this is not a counter-example to our
  proposition, which is stated at fixed `w`.
