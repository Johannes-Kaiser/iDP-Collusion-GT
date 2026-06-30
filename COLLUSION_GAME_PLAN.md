# A Game-Theoretic Analysis of Collusion in Individual Differential Privacy

**Working title:** *The Privacy Externality Game: Strategic Budget Choices in Sampling-Based Individual Differential Privacy*

This document is the working plan. Section 1 lists the decisions that need confirmation. Sections 2–4 set up the game, the actors, and the experiments. Section 5 contains five rounds of self-critique that re-shaped the plan. Section 6 is the execution plan.

---

## 1. Decisions to surface to the user

The following decisions materially shape the paper. The defaults below are what I will adopt if the user does not override them; the AskUserQuestion tool will be invoked to confirm the most consequential ones before execution.

| # | Decision | Default | Alternatives |
|---|----------|---------|--------------|
| D1 | Number of strategic actors | **LOCKED: n-player groups** (with 2-player as worked example) |
| D2 | Action-space type | **discrete** ε grid `{ε_min,…,ε_max}` (e.g. `{1, 2, 4, 8, 16, 32}` at δ=1e-12); continuous-action sensitivity in appendix |
| D3 | What enters player utility | **LOCKED: two regimes side-by-side** — *naïve* (perceived cost = nominal `h(εᵢ)`) vs *aware* (perceived cost = real `A_i(ε⃗)`). A shared model-utility term `V(σ)` is included in both for the cooperation-incentive baseline. |
| D4 | Attacker model in collusion variant | Stackelberg coalition with optional utility-floor for plausible deniability |
| D5 | (εᵢ,δᵢ,Δ)-iDP mitigation analysis | **LOCKED: deferred to future work**; mentioned in discussion only |
| D6 | Empirical training vs purely theoretical advantage | **purely theoretical** — analytical MIA advantage via PLD/PRV of subsampled Gaussian |
| D7 | Datasets / scenarios for the figures | (i) symmetric n=5, equal sizes 10k each, batch 128, 5 epochs; (ii) federated silo n=5 with 4 colluders + 1 target; (iii) credit-card-like 50k/32/50 to ground in Kaiser et al. |
| D8 | Paper venue / format | **LOCKED: IEEEtran two-column (SaTML / S&P)** |
| D9 | LaTeX layout | `paper/main.tex`, `paper/refs.bib`, `paper/figures/` |

---

## 2. The game

### 2.1 Players, actions, and the mechanism

* **Players.** A finite set `N = {1, …, n}`. Each player `i` contributes a group of `|G_i|` data points to a joint dataset of size `M = Σ|G_i|`. Some players may be coalition members and play jointly.
* **Action.** Each player picks an individual ε-budget `εᵢ ∈ A`, where `A = {ε⁽¹⁾, …, ε⁽K⁾} ⊂ [ε_min, ε_max]`. A common δ is fixed system-wide.
* **Mechanism.** Given the action profile `ε⃗`, the sampling-based iDP mechanism (Boenisch et al. 2023) solves for per-group sampling rates `p_g(ε⃗)` and a single noise multiplier `σ(ε⃗)` satisfying, for `I` iterations:
  
  **(C1)**  RDP constraint per group: `ρ_g ≤ I · 2 p_g² α / σ²`, converted to `(ε_g, δ)`-DP.
  
  **(C2)**  Expected batch size: `Σ_g (|G_g|/M) · p_g = E_b / M`.

* **Excess vulnerability.** For the resulting subsampled Gaussian mechanism, the per-group MIA advantage `A_g(ε⃗)` is computed from the privacy profile of the `I`-fold composition `SubGauss(p_g, σ)^∘I` as `A_g = max_α (1 − α − f_g(α))` where `f_g` is the trade-off function.

Note that **A_g depends on the whole ε⃗, not just ε_g**: this is the externality and is the key formal object that distinguishes the game from Pejó's.

### 2.2 Utility

Let `V(σ)` be a globally shared model-quality proxy (monotone decreasing in σ — more noise, less utility). Two awareness regimes parameterize player `i`:

* **Aware player:** `u_i(ε⃗) = B_i · V(σ(ε⃗)) − C_i · A_i(ε⃗)`
* **Naïve player:** `u_i(ε⃗) = B_i · V(σ(ε⃗)) − D_i · h(εᵢ)` where `h(εᵢ)` is a nominal-privacy cost (decreasing in privacy, e.g. `h(ε) = ε / ε_max`).

A *mixed* player carries both `C_i ≥ 0` and `D_i ≥ 0`. This distinction is novel and central: it explains *why* the attack from Kaiser et al. is invisible — naïve players do not include the externality in their utility.

### 2.3 Variants

| Variant | Description |
|---------|-------------|
| **PEG-honest** | All players play their own utility; look for Nash Equilibrium (NE). |
| **PEG-coalition** | Subset `C ⊂ N` plays a joint utility `u_C(ε⃗) = Σ_{i∈C} u_i − λ · A_t(ε⃗)` for a chosen target `t ∉ C`. λ ≥ 0 weights the harm-the-target term. Look for Stackelberg equilibrium with `C` as leader. |
| **PEG-stackelberg-central** | A central model trainer ("data curator") commits to an `ε⃗` first; players accept/decline. |
| **PEG-Δ-constrained** | Action set restricted to profiles satisfying `(εᵢ, δ, Δ)`-iDP (Kaiser et al.). Solve same equilibria under the constraint. |

### 2.4 New analogues of Pejó's concepts

* **Vulnerability function `Ψ_i(ε⃗) := A_i(ε⃗)`** — the analogue of Pejó's `Φ` but for *privacy risk* instead of accuracy loss.
* **Price of Excess Vulnerability (PoEV).** For a NE `ε⃗*`,
  
  `PoEV(ε⃗*) := ( Σᵢ A_i(ε⃗*) − Σᵢ A_i(ε⃗_uniform) ) / Σᵢ A_i(ε⃗_uniform)`
  
  the relative MIA-advantage inflation at the NE vs. a uniform-budget baseline.

* **Price of Mitigation.** Loss of accuracy `V(σ)` at the equilibrium of the Δ-constrained game vs. the unconstrained NE.

---

## 3. Theoretical analysis to prove

1. **Privacy-Externality Lemma.** Under the sampling-based-iDP constraints, `∂A_i / ∂ε_j` has a known sign: for any `j ≠ i`, lowering `ε_j` (more privacy for j) weakly *increases* `A_i` (worse for i). Proof: implicit differentiation of (C1) and (C2). This is the formal statement that the externality is monotone.

2. **NE existence (honest PEG, discrete actions).** Trivial: finite normal-form game ⇒ mixed-strategy NE exists (Nash 1950); pure NE exists in many parameter regimes (verified empirically and via potential-game arguments when possible).

3. **Coalition best response.** In `PEG-coalition` with `λ → ∞`, the coalition's BR is `εᵢ = ε_min` for all `i ∈ C`. This recovers the Kaiser et al. attack as a one-step Stackelberg solution.

4. **Comparative statics in coalition size.** `A_target(ε_min·𝟙_C, ε⃗_others)` is monotone increasing in `|C|`. Provable from (C2) and the SubGauss-trade-off.

5. **Convergence of BR dynamics.** In simulations, check whether best-response iteration converges (and to a pure NE). For Pejó-style proofs, check if the potential-game condition `∂_{ε_i}∂_{ε_j} u_i = ∂_{ε_i}∂_{ε_j} u_j` holds (likely not, in general, but might for symmetric subgames).

---

## 4. Experiments (theoretical / numerical)

All experiments are *analytical evaluations of the game* — no model training. Each yields a figure/table for the paper.

| # | Experiment | What it shows | Figure |
|---|------------|---------------|--------|
| **E1** | 2-player honest PEG, sweep (B_i, C_i) | NE-budget heatmap; analogue of Pejó's Table 5 with externality | Heatmap of equilibrium ε* per player |
| **E2** | 2-player asymmetric group sizes | NE shift as |G_1|/|G_2| changes | Curves: NE-ε vs size-ratio |
| **E3** | n-player honest PEG (n ∈ {3,5,10}) | Equilibrium ε vs n, equal sizes | Plot |
| **E4** | k-coalition Stackelberg (k ∈ {1,…,n-1}) | A_target(NE) vs k | Spaghetti line, mirror Fig. 6 of Kaiser et al. |
| **E5** | Coalition with utility-floor | Pareto front of (target-vuln gain) vs (own utility loss) | Pareto figure |
| **E6** | Naïve vs aware regime comparison | Equilibrium gap between awareness levels = "the hidden attack surface" | Bar chart |
| **E7** | Δ-constrained PEG | Δ-bound on action set ⇒ NE shifts; Price of Mitigation | Curves vs Δ |

---

## 5. Five rounds of self-critique

### Round 1 — *Is the formal object well-defined?*
The function `A_i(ε⃗)` requires a single noise multiplier σ to make the constraint binding. If the binary search returns multiple solutions (or none), the game is ill-defined. **Fix:** restrict to ε⃗ in the *feasible set* `F` where the binary search succeeds; characterize `F` (it is connected when group sizes are bounded away from zero and ε-grid is bounded). All experiments restrict to `F`. Add to the paper as Assumption 1.

### Round 2 — *Is this novel beyond Pejó+Kaiser?*
Pejó's externality is purely *accuracy-side* (your privacy hurts only your own accuracy). The externality we introduce is *privacy-side* (your privacy hurts others' privacy). That is structurally different and yields opposite incentive geometry: in Pejó, "race to bottom" (no privacy) was always feasible; here, choosing low ε is **attractive both for selfish privacy AND as an offensive weapon**. **Fix:** make this point central in the introduction and label the new equilibrium concept "Privacy Externality Equilibrium" to mark the conceptual departure.

### Round 3 — *Is the action space realistic?*
Real ε's that data subjects choose are bounded above by `εᵢ^max` (their contract). But Kaiser et al. assume a discrete set. **Fix:** use a discrete grid `{0.5, 1, 2, 4, 8, 16, 32, 64}` for the main figures, and run a continuous-action sensitivity check (interpolated grid) for one figure in the appendix. Also: the action ε⃗ = ε_min⃗ may exit the feasibility set — we constrain to F.

### Round 4 — *Are the proofs reachable?*
The Privacy-Externality Lemma requires implicit differentiation of a binary-search solution — formally it is the derivative of σ(ε⃗) and p(ε⃗) implicitly defined by (C1)+(C2). **Fix:** prove it for the *continuous-α* RDP form (so σ and p are differentiable), then argue the discrete-α numeric version is monotone by continuity. This is a clean appendix-length proof. The other items are either trivial (NE existence in finite game) or empirical claims.

### Round 5 — *Scope creep?*
Tempted to add: (a) repeated game with learning dynamics, (b) Bayesian game with type uncertainty, (c) signalling games where players hide their ε. **Fix:** push (a)-(c) to "Future Work" in conclusion. The paper sticks to one-shot, perfect-information, normal-form game theory. This is enough to make the central contribution land cleanly.

**Cross-cutting fix from all 5 rounds:** the paper's central claim is *not* "we found a new attack" — Kaiser et al. already did. Our claim is *"the attack arises rationally as the equilibrium of a natural game; consequently, even non-malicious players in iDP impose privacy externalities on others, and this changes how we should design contracts"*. This is what every section needs to support.

---

## 6. Execution sequence

1. **Set up LaTeX environment** under `paper/` (IEEEtran, `main.tex`, `bib/refs.bib`, `figures/`).
2. **Build the game engine** under `game/` as standalone Python (no opacus dependency for the released code path; the opacus binary search is reimplemented in 30 lines for portability and is verified to agree with opacus_new on a small grid):
   * `game/mechanism.py` — solve `(σ, p_g)` from `(ε⃗, δ, E_b, I, |G_g|)`.
   * `game/advantage.py` — MIA advantage from `(p, σ, I)` via PLD / PRV accountant (use `dp_accounting` from `differential-privacy/`) and from the analytical RDP-bound as a sanity check.
   * `game/utility.py` — utility functions and player types.
   * `game/solve.py` — best-response iteration, NE finder, Stackelberg solver.
3. **Run experiments E1–E7**, saving figures to `paper/figures/`.
4. **Write the paper** section-by-section, getting refs from WebSearch as needed.
5. **Compile** to PDF; verify it stands alone (no code references, no prior-knowledge assumption).

---

## 7. Paper outline (final)

1. **Introduction.** iDP promises individual control; sampling-based realisation breaks the promise by creating privacy externalities. We model this with a game; we show the externality is the rational equilibrium, not just a vulnerability.
2. **Background.** DP, iDP (sampling-based), privacy profiles, trade-off functions, MIA advantage, basic game theory.
3. **The Privacy-Externality Game.** Formal definition. Vulnerability function Ψ. Player types and awareness regimes. PoEV.
4. **Equilibrium analysis.** Privacy-Externality Lemma. NE existence. Coalition Stackelberg. Comparative statics in coalition size.
5. **Numerical evaluation.** E1–E7 with figures and tables.
6. **Mitigation under (εᵢ, δ, Δ)-iDP.** The constrained game; Price of Mitigation.
7. **Related work.** Pejó et al. 2018; Kaiser et al. 2026; Boenisch et al. 2023; Kaissis et al. 2024 (Δ-divergence); FL incentive games (Tu et al. 2025; FedPCS); private NE seeking (Wang et al. 2024).
8. **Discussion & limitations.** One-shot vs repeated; perfect info; central authority enforcement of action space.
9. **Conclusion.**

---

*End of plan. Beginning execution.*
