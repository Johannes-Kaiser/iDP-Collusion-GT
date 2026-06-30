1. The bridge
In the CoL game of PoP, each player n chooses a privacy parameter p_n ∈ [0,1] (mapped from ε), and
u_n(p₁, p₂) = B_n · b(θ_n, Φ_n(p₁, p₂)) − C_n · c(p_n).
The benefit term Φ_n is interdependent — collaborative accuracy depends on both players' parameters — but the privacy-loss term c(p_n) is separable: your leakage depends only on your own choice. This separability was a modelling convenience, not a fact.
Kaiser et al. show it is also false for sampling-based iDP. To hold the expected batch size fixed, individual sampling rates are coupled through a shared noise multiplier, so each sample's actual privacy profile — and therefore its excess MIA vulnerability beyond the calibration point — is a function of the entire budget distribution, not of its own (ε_i, δ_i) alone. The nominal guarantee is unchanged; the realised worst-case advantage is not. Their Appendix A makes the stronger point that this coupling survives any informed choice of a shared σ (min/max/mean over the budgets), not just the fixed-batch-size formulation.
Counterintuitively, the direction of the externality is unusual: a player who picks weak privacy (high ε) becomes more exposed when the others become more protective (lower ε), especially if the weakly-private party is in the minority. In PoP coordinates, that is ∂c_excess(p_target)/∂p_other > 0 in that regime — others increasing protection raises your loss. This is precisely an interdependent-privacy effect, which is why Gergely's framing is the natural theoretical home for the whole line.
The program below is: (A) put this coupling into the utility, (B) figure out how to measure it cheaply enough to play the game, (C) scale to N players / collusion, (D) add a leader / information structure.
2. Direction A — Interdependent privacy-loss utility (two players)
Replace the separable cost with a coupled one and decompose it into the contracted part and the excess:
u_n(p) = B_n · b(θ_n, Φ_n(p)) − C_n · c_n(p_n; p_{−n}), c_n(p_n; p_{−n}) = c̄(p_n) + γ_n · χ_n(p_n; p_{−n}).
Here c̄(p_n) is PoP's original nominal loss (separable, monotone, the contracted (ε_i, δ_i) guarantee) and χ_n is the excess vulnerability — the realised or worst-case MIA advantage beyond the calibration point. All the new coupling lives in χ. Under the (ε_i, δ_i, Δ̄)-iDP contract, χ_n ≤ Δ̄, so the contract is exactly the device that caps the new externality. That tie-in is worth stating up front: the Δ̄ bound is what could keep the game well-behaved.
The theoretical questions are concrete and largely reuse the PoP machinery:
Potential structure. PoP's existence proof (Th. 4 / Cor. 1) relied on the potential condition ∂{p₁}∂{p₂} u₁ = ∂{p₁}∂{p₂} u₂. With a separable cost, ∂{p₁}∂{p₂} c(p_n) = 0 and the cost dropped out. With c_n(p_n; p_{−n}) it does not: the condition picks up the terms −C_n · ∂{p₁}∂{p₂} χ_n. Re-derive whether a potential still exists, and whether Δ̄-boundedness plus regularity of χ is enough to recover it.
Player types and thresholds. Lemmas 1–2 and the α_n, β_n thresholds assumed c monotone decreasing in own p. χ is not monotone in own p and depends on p_{−n}, so the "concerned / unconcerned" characterisation and the best-response map need re-derivation.
Train-alone equilibrium. Verify whether (p₁*, p₂*) = (1, 1) is still always a NE. Intuitively max protection (sampling rate → 0) gives ≈0 benefit and bounded χ, so it likely survives — but it should be confirmed rather than assumed.
3. Direction B — Estimating the coupled cost (the hard part)
PoP already had to approximate Φ_n empirically (self-division), because computing it exactly needs the joint dataset. The new model adds a second function, χ, that is far more expensive to evaluate: a faithful empirical χ needs a LiRA-style MIA per budget configuration. Kaiser's own attack evaluation cost m×b×n = 1000×4×64 = 256,000 shadow models for a single setting. Naively "playing the game" by simulating each strategy profile with an MIA is therefore out of reach.
Three ways out, in increasing fidelity / cost:
Analytical advantage as the cost. The adversarial advantage and privacy profiles of the subsampled Gaussian mechanism are computed from trade-off functions with no model training (Kaiser Figs. 2–3). Define χ from this analytical worst case. It is cheap, distribution-aware, and conservative — the right basis for a first, fully simulatable game.
Δ-divergence as the cost. Δ̄↔ over a set of mechanisms is computable in negligible time (Kaissis et al.). Playing the game directly on the Δ̄ bound makes χ both cheap and contractually meaningful, and aligns the game with the proposed (ε_i, δ_i, Δ̄) contract.
Surrogate calibrated by sparse MIA. Interpolate the cost surface over a coarse (p₁, p₂) grid (as PoP did for Φ), and calibrate it against a handful of real LiRA runs to quantify the worst-case-vs-realised gap.
Recommendation: build the model on (1)/(2) first, treat (3) as validation. The distinction between the worst-case cost (analytical / Δ̄, what a contract should bound) and the realised cost (MIA, what actually happens) is itself a result worth reporting — how large is the gap, and which one should the contract be written against?
4. Direction C — N players, silos, and collusion
Kaiser's collusion attack is inherently N-player and maps cleanly onto FL silos: one player ↦ one silo controlling the budgets of all its samples. Note this is richer than PoP's single-p-per-player — a silo controls a sub-distribution of budgets, not a scalar.
Two natural game classes:
Benign N-player. Each silo cares only about its own (accuracy, privacy). Study how the equilibrium budget distribution shapes everyone's excess risk, and define a Price-of-Anarchy analogue — a "Price of Collusion" / interdependent-privacy externality — measuring the excess risk selfish budget choices impose on the minority.
Adversarial coalition. A subset C colludes against target j. Add a malice term to colluders' utility:
u_i = B_i·b(θ_i, Φ_i(p)) − C_i·c_i(p_i; p_{−i}) + D_i·χ_j(p_j; p_{−j}), with D_i ≥ 0.
D_i = 0 recovers the benign game; D_i > 0 rewards inflating the target's excess loss. The collusion attack is then the corner where colluders drive their own ε down (p → 1) to maximise χ_j — and the natural deterrent is endogenous: doing so degrades model accuracy (their own benefit) and leaves a utility "fingerprint," exactly Kaiser's utility-trade-off observation. The game therefore predicts when collusion is individually rational and how large a coalition is needed (their 20%-of-samples threshold is a useful anchor).
5. Direction D — Stackelberg / sequential / information structure
PoP is a simultaneous, symmetric-information game. Two extensions:
Stackelberg / who declares first. A leader (central coordinator or trainer) commits to a budget-assignment rule or a Δ̄ contract; followers (data holders) respond. Kaiser's Budget Manipulation Attack is exactly the malicious-leader case; the benign case is a coordinator designing the rule. Study the subgame-perfect equilibrium.
Mechanism design / collusion resistance. Treat the choice of shared σ aggregation (Kaiser's Table III — min / max / mean, each with a different threat profile) and the (ε_i, δ_i, Δ̄) contract as design levers. Can the coordinator pick a rule that is collusion-resistant, or is sensitivity-based iDP / public-data padding required when the Δ̄ slack is exceeded?
Information. Symmetric vs private budgets/weights (Bayesian game). This connects directly to PoP's own stated future work on making the weights B, C private.
6. Modelling primitives to fix before coding
A short list of shared decisions, so the experiments and the theory stay compatible:
Player granularity: individual sample vs silo (FL). Pick silo for the FL story; a player then chooses a budget distribution, not a scalar p.
Cost basis: analytical advantage vs Δ̄-bound vs empirical MIA — start with analytical/Δ̄ (Section 3).
Action space: continuous ε / p vs a discrete budget menu (discretisation is what PoP used for BR dynamics).
Coupling assumption: keep the fixed-expected-batch-size setting (Kaiser's main analysis), but record that Appendix A shows the coupling persists for any informed shared σ, with min/mean/max giving distinct games.
7. Concrete next steps and suggested division of labour
Johannes — initial experiments during the holiday window (perfect / full information). Two-player, analytical-advantage cost over a (p₁, p₂) grid using the sampling-based iDP trade-off functions — no MIA training needed. Reproduce the interdependence as a cost surface, then run BR dynamics (as in PoP §7.3) to locate equilibria and check whether a non-trivial pure-strategy NE exists numerically. If quick, also compute Δ̄ over the same grid to compare the two cost bases. Reuse Kaiser's public iDP/LiRA code for later validation runs.
Gergely. Recast χ as an interdependent-privacy externality; lead the N-player / coalitional formalisation (Section 4) and the Price-of-Collusion definition.
Balázs. Overall game structure and the NE-existence re-derivation under the coupled cost (Section 2); mapping back onto the PoP results and contract framing.
Joint, later. Calibrate the analytical surface against a few empirical LiRA runs to quantify the worst-case-vs-realised gap (Section 3, item 3).
8. Open theoretical questions
Does a pure-strategy NE survive the coupled cost, and does Δ̄-boundedness of χ suffice for existence?
Is the modified game still a potential game? If so, what is the potential, and how does −C_n·∂{p₁}∂{p₂}χ_n enter the condition?
What is the sign and monotonicity of χ in p_{−n} across composition regimes (minority vs majority weak-privacy)?
What is the right player granularity for FL (sample vs silo), and how does it change the equilibria?
Worst-case (analytical / Δ̄) vs realised (MIA) cost — which should the contract be written against, and how large is the gap in practice?
Is the (ε_i, δ_i, Δ̄) contract enough as a collusion-resistant mechanism, or are sensitivity-based iDP / public-data padding needed when the slack is exceeded?


