# T16 -- H8 BR-PEG regret curve (redesigned instance)

**Prerequisite landed:** T09 (`09_br_peg_sharpen_and_promote.md`). This
task consumes T09's sharpened proposition (`prop:bayesian-not-min`) as
its formal anchor and produces the empirical regret trace that turns the
proposition's qualitative "eventually converges to the interior argmax"
clause into a quantitative statement about *how fast* and *at what
price*.

**Do not edit `paper/` in this task.** The deliverables here are the
figure, the plotting script, the tabulated regret summary, and the LaTeX
snippets ready for hand-off to a T16-executor task that will paste them
into `paper/tex/07_discussion.tex` (T09's promoted BR-PEG subsection).

**Why the instance was redesigned.** The original 3-type demo
(Theta = {w = 0.5, 1.0, 2.0}) has such a type-separating log-Gaussian
likelihood that M's posterior collapses in one round, making the regret
curve a trivial step function: identification is free because the setup
is too clean, and a reviewer could object that the "constant regret"
claim is an artefact of the instance rather than a property of the
policy. The redesigned instance introduces two honest sources of
ambiguity: (i) an *overlapping* 5-type space in which the low-w types
{0.6, 0.8, 1.0} induce the *same* best-response eps_H at M's
uniform-prior arm and differ only by 0.06-0.13 in u_H, and (ii)
log-Gaussian observation noise of width sigma_obs = 0.15 on M's
observations of (eps_H, u_H) -- the honest model of what M actually sees
in a real deployment (H's action and utility are estimated from finite
proxies). Under this instance identification takes ~10 rounds and the
learning curve is real.

---

## 1. Goal and narrative role of H8

### 1.1 What we claim

The Stackelberg-learning template of Letchford-Conitzer-Munagala 2009,
Blum-Haghtalab-Procaccia 2014, and Balcan-Blum-Haghtalab-Procaccia 2015
delivers an O(sqrt(T)) regret bound for a leader who learns an
unknown-type follower on a finite action set. **BR-PEG with a finite
type space and a correctly-specified full-support prior does strictly
better: its cumulative regret against the full-information Stackelberg
oracle is bounded** -- it plateaus at R ~= 1.3 (a finite identification
cost of ~10 confused rounds at 0.138 regret per round) and is
horizon-independent thereafter. Every alternative that lacks the
Bayesian machinery pays for it: a stationary uniform-belief M and a
mis-specified-prior M pay regret linear in T, and a prior-free EXP3
bandit is the worst of the four over practical horizons.

### 1.2 Why H8 matters

The T09 promoted proposition ends with a qualitative "eventually":
`eps_M_greedy(t) = eps_M_star(theta_H_true)` for all sufficiently large
t. Without a quantitative anchor on *when* eventually kicks in and *how
much regret it costs*, a reviewer can still object that "the theoretical
merging result is asymptotic; you have not shown the finite-round
behaviour is favourable, so the practical attack story is unsupported".
H8 closes that hole with a per-round trace on an instance where
identification is genuinely hard (overlapping types + observation
noise), and it prices the two natural failure modes (no prior; wrong
prior).

Specifically, H8 shows (T = 60, 40 seeds, mean +/- 1 SE):

- BR-PEG with the correct prior identifies theta_H_true in ~10 rounds
  and its cumulative regret plateaus: R(10) = 0.77, R(60) = 1.31 --
  bounded, consistent with (and sharper than) the O(sqrt(T)) template
  rate;
- the uniform-belief constant M pays exactly linear regret
  (0.138/round, R(60) = 8.28);
- BR-PEG with a mis-specified prior (support excludes the truth)
  converges to the nearest wrong type, locks onto that type's greedy
  arm, and pays linear regret at asymptotically the uniform-belief
  slope (R(60) = 8.42): once the truth is outside the support, learning
  buys nothing;
- EXP3 (no prior) is sub-linear asymptotically but its forced
  exploration makes it the worst of the four over this horizon
  (R(60) = 13.2): the Bayesian prior is worth an order of magnitude of
  regret against the black-box bandit alternative.

### 1.3 Positioning against the Stackelberg-learning literature

Stackelberg-learning bounds are proven for a leader who does not know
the follower's type but observes the follower's action each round.
Under a finite Theta with a full-support prior, posterior consistency
(Kalai-Lehrer merging) makes the identification phase finite even when
single observations are ambiguous, so BR-PEG realises a bounded-regret
regime that is strictly inside the O(sqrt(T)) template rate of
Balcan-Blum-Haghtalab-Procaccia. The point of H8 is not to falsify the
Stackelberg-learning template; it is to identify the finite-Theta
sub-case in which our leader-follower coupling delivers a strictly
better rate -- and to show, via the mis-specified-prior arm of the
experiment, that the bounded-regret regime is *earned by the prior*,
not by the coupling alone.

---

## 2. Protocol

- **Game:** BR-PEG (see T09 section 1 for the formal statement).
- **Instance** (redesigned; produced by
  `tasks_findings/_scripts/16_regret_plot.py`):
  - Type set Theta = {w in {0.6, 0.8, 1.0, 1.4, 2.0}}, n_H = 5000.
  - True type theta_H_true has w_H^star = 0.6 (utility-leaning). Chosen
    because (i) its Stackelberg-oracle arm (eps_M = 3.17) differs from
    the uniform-prior argmax arm (eps_M = 1.78), so a policy that never
    learns pays linear regret; and (ii) at the uniform-prior arm the
    three low-w types {0.6, 0.8, 1.0} induce the same eps_H = 10.375
    best response and differ only by ~0.06-0.13 in u_H -- below the
    observation noise -- so identifying the truth takes multiple rounds.
  - eps_H action grid: log-spaced over [0.5, 64], K = 9.
  - eps_M arm set: log-spaced over [1, 32], K = 7 (kept small so EXP3
    has a fair chance).
  - Mechanism config: expected_batch = 128, steps = 5 * 2n / batch,
    delta = 1e-12.
  - Observation model: M observes eps_H with multiplicative log-Gaussian
    noise and u_H with additive Gaussian noise, both width
    sigma_obs = 0.15; the posterior update uses the correctly-specified
    likelihood.
- **Horizons:** continuous curve to T = 60; markers at
  T in {10, 20, 40, 60}.
- **Seeds:** 40 independent seeds; mean +/- 1 SE bands.
- **Metrics.**
  - *Single-round regret* r_t := A_H^oracle - A_H^policy(t).
  - *Cumulative regret* R(T) := sum_{t=1..T} r_t.
- **Policies and baselines**
  - *Stackelberg oracle* (reference, R = 0): full-info M plays the
    constant eps_M^oracle = argmax_{eps_M} A_H(eps_M,
    BR(eps_M | theta_H_true)) = 3.175, A_H^oracle = 0.4540.
  - *BR-PEG greedy, correct prior*: uniform prior on the full Theta
    (includes the truth).
  - *BR-PEG greedy, mis-specified prior*: uniform prior on
    Theta \ {theta_H_true} (support excludes the truth).
  - *EXP3* (Auer et al. 2002): anytime variant with
    eta_t = sqrt(log K / (tK)) and exploration mixing
    gamma_t = min(1, sqrt(K/t)), over the same 7 eps_M arms, no prior.
  - *Uniform-belief constant M*: stationary argmax of
    E_{theta ~ U(Theta)} A_H, i.e. eps_M = 1.782; realises
    A_H = 0.3159 per round against the truth.
- **Lookup cache:** `_cache_v2/H8_regret_lookup.pkl` (per-(type, arm)
  best responses; ~7 min to build cold, then all four policies are pure
  numpy).

---

## 3. Real regret curve

Produced by `tasks_findings/_scripts/16_regret_plot.py`; tabulated in
`tasks_findings/_scripts/16_regret_summary.txt`.

Mean cumulative regret R(T) +/- 1 SE over 40 seeds:

```
   T    BR-PEG(correct)    BR-PEG(mis)       EXP3          uniform
  10    0.773 +/- 0.068   1.517 +/- 0.016   2.805 +/- 0.079   1.380 +/- 0.000
  20    1.073 +/- 0.148   2.897 +/- 0.016   5.212 +/- 0.132   2.761 +/- 0.000
  40    1.239 +/- 0.213   5.658 +/- 0.016   9.457 +/- 0.264   5.522 +/- 0.000
  60    1.308 +/- 0.256   8.419 +/- 0.016  13.205 +/- 0.361   8.283 +/- 0.000
```

Read-outs:

- The correct-prior curve is concave and plateaus: the increment from
  T = 40 to T = 60 is 0.07 (0.0035/round and falling). The plateau
  height ~1.3 corresponds to ~9.5 effective confused rounds at the
  uniform-arm regret of 0.138/round.
- The mis-specified curve is linear from ~t = 5 onwards with slope
  0.138 (the uniform-belief slope), offset slightly above the uniform
  line by an early phase in which the wrong posterior briefly plays the
  even-worse eps_M = 1 arm (0.236/round).
- EXP3 is the worst of the four at every T <= 60.
- The uniform-belief line is exactly linear by construction
  (0.1381/round).

Ambiguity check at the oracle arm (why identification is multi-round):
at M's initial (uniform-prior) arm eps_M = 1.782, types {0.6, 0.8, 1.0}
all best-respond with eps_H = 10.375 and differ only in
u_H = {0.396, 0.333, 0.269} -- separations of 0.4-0.8 noise standard
deviations per round -- so the posterior needs several rounds of
evidence before it puts > 60% mass on the truth, which is the threshold
at which the greedy arm switches from eps_M = 1.78 to the oracle arm
eps_M = 3.17.

---

## 4. Publication-quality figure

- `paper/figures/fig_h8_brpeg_regret.pdf`
- `paper/figures/fig_h8_brpeg_regret.svg`

Single panel (no title, style-matched to `fig_h5_v_shape` via
`game.style.apply_paper_style`): cumulative regret R(T) vs horizon T
for the four policies, mean over 40 seeds with +/- 1 SE bands.
BR-PEG(correct) solid blue; BR-PEG(mis-specified) dashed blue; EXP3
solid orange; uniform-belief constant grey.

LaTeX caption (drop into `paper/tex/07_discussion.tex` alongside the
figure environment inserted by the T09 promotion):

```latex
\begin{figure}[!t]
\centering
\includegraphics[width=0.95\linewidth]{figures/fig_h8_brpeg_regret.pdf}
\caption{\textbf{BR-PEG regret against the full-information Stackelberg
oracle.} Cumulative regret $R(T) = \sum_{t=1}^{T} (A_H^\star -
A_H^{\pi}(t))$, mean $\pm$ 1 SE over 40 seeds, on an instance with five
overlapping types ($w_H \in \{0.6, 0.8, 1.0, 1.4, 2.0\}$, truth
$w_H^\star = 0.6$) and observation noise $\sigma_{\mathrm{obs}} = 0.15$
on M's view of $(\eps_H, u_H)$. The posterior-greedy Bayesian M with a
correctly-specified prior (solid blue) identifies $\theta_H^\star$ in
roughly ten rounds and its regret plateaus at $R \approx 1.3$,
horizon-independent thereafter. The stationary uniform-belief M (grey)
pays exactly linear regret; the mis-specified prior whose support
excludes the truth (dashed blue) locks onto the nearest wrong type and
pays the same linear rate---once the truth is outside the support,
learning buys nothing. A prior-free EXP3 bandit over the same
$\eps_M$ arms (orange) is sub-linear asymptotically but worst over this
horizon: the Bayesian prior is what turns the leader--follower coupling
into a bounded-regret attack.}
\label{fig:brpeg-regret}
\end{figure}
```

---

## 5. Comparison table

All against the same Stackelberg oracle (eps_M^oracle = 3.175,
A_H^oracle = 0.4540, truth w_H^star = 0.6):

| Policy                  | eps_M behaviour              | R(T=10) | R(T=20) | R(T=40) | R(T=60) | Rate       |
|-------------------------|------------------------------|---------|---------|---------|---------|------------|
| Stackelberg oracle      | 3.175 (constant)             | 0.000   | 0.000   | 0.000   | 0.000   | O(0)       |
| **BR-PEG (correct)**    | 1.78 -> 3.17 after ~10 rds   | 0.773   | 1.073   | 1.239   | 1.308   | **bounded**|
| BR-PEG (mis-specified)  | locks onto wrong type's arm  | 1.517   | 2.897   | 5.658   | 8.419   | O(T)       |
| uniform-belief          | 1.782 (constant)             | 1.380   | 2.761   | 5.522   | 8.283   | O(T)       |
| EXP3 (no prior)         | randomised over 7 arms       | 2.805   | 5.212   | 9.457   | 13.205  | sub-linear, worst at T<=60 |

---

## 6. Paper location and transitions

**Primary location:** after T09's promoted
`\subsection{Repeated Bayesian variant (BR-PEG)}` in
`paper/tex/07_discussion.tex`, insert the H8 figure environment and the
`Regret against the full-information optimum.` paragraph *after* the
`Numerical realisation and scope.` paragraph.

Insertion snippet (ready for the T16-executor):

```latex
\paragraph{Regret against the full-information optimum.}
To price the identification phase, we simulate an instance in which
learning is genuinely hard: five overlapping types
($w_H \in \{0.6, 0.8, 1.0, 1.4, 2.0\}$, truth $w_H^\star = 0.6$) whose
best responses coincide at M's initial arm, and observation noise
$\sigma_{\mathrm{obs}} = 0.15$ on M's view of $(\eps_H, u_H)$. Writing
$R(T) := \sum_{t=1}^{T} (A_H^\star - A_H^{\pi}(t))$ for cumulative
regret against the full-information Stackelberg optimum,
\cref{fig:brpeg-regret} shows that the posterior-greedy Bayesian M
identifies $\theta_H^\star$ in roughly ten rounds and its regret
plateaus at $R \approx 1.3$, horizon-independent thereafter---inside
the $O(\sqrt{T})$ rate that the Stackelberg-learning template of
\citet{balcan2015commitment} guarantees for tabular leader--follower
games. The comparison policies price the Bayesian machinery: a
stationary uniform-belief M pays exactly linear regret; a
mis-specified prior whose support excludes the truth converges to the
nearest wrong type and pays the same linear rate, so the bounded-regret
regime is earned by the prior, not by the coupling alone; and a
prior-free EXP3 bandit over the same $\eps_M$ arms, while sub-linear
asymptotically, is the worst of the four over practical horizons.
```

**Cross-references produced:** `\label{fig:brpeg-regret}` on the
figure; `\cref{fig:brpeg-regret}` from the new paragraph.

**Cross-references to add:** the T09 scope paragraph in
`paper/tex/07_discussion.tex` currently hedges "a regret rate ...
provides only a constant-regret upper bound". Concrete rewrite:

- Before: "Second, a regret rate in the style of
  \citet{balcan2015commitment, blum2014learning} that quantifies the
  number of rounds until $\eps_M^{\mathrm{greedy}}(t) = \eps_M^\star$;
  the finite-type Kalai--Lehrer setup used here delivers one-round
  identification and so provides only a constant-regret upper bound."
- After: "Second, a regret rate in the style of
  \citet{balcan2015commitment, blum2014learning}; empirically the
  finite-type setup delivers a bounded identification cost
  (\cref{fig:brpeg-regret}), and the matching finite-round lower bound
  is open."

Note the promoted subsection's `Numerical realisation and scope.`
paragraph quotes the *clean* 3-type demo of `\cref{app:bayesian}`
(one-round collapse, $\eps_M^\star \approx 3.08$, $A_H \approx 0.454$).
That demo remains correct as the appendix's illustration of
\cref{prop:bayesian-not-min}; the H8 figure complements it with the
noisy-overlapping instance. The T16-executor should make sure the prose
distinguishes the two instances rather than conflating them.

**Limitations to keep in scope:** the bounded-regret result depends on
the finite type space and the correctly-specified observation
likelihood. A dense (continuous) Theta, or an H who adversarially
strategises over the observation sequence she reveals, would push the
attack back to the O(sqrt(T)) regime of \citet{balcan2015commitment}.
The mis-specified-prior arm of the experiment already exhibits the
failure mode inside the finite-Theta world: support mis-specification
converts bounded regret into linear regret.

---

## 7. Acceptance checklist

- [x] Regret curves computed from the real mechanism lookup
  (`_cache_v2/H8_regret_lookup.pkl`, built from `H_best_response` on
  the actual DP accountant); simulation layer is pure numpy over that
  lookup; no synthetic trajectories.
- [x] Instance is non-trivial: uniform-prior arm differs from the
  oracle arm, and the truth is only identifiable through
  sub-noise-level u_H differences (multi-round learning).
- [x] Cumulative regret at T = 10, 20, 40, 60 tabulated for all four
  policies (section 5); 40 seeds, +/- 1 SE.
- [x] Figure `paper/figures/fig_h8_brpeg_regret.pdf` and `.svg`
  produced by `tasks_findings/_scripts/16_regret_plot.py`,
  style-matched to the paper (apply_paper_style, single panel, no
  title).
- [x] LaTeX caption and insertion paragraph ready in fenced blocks
  (sections 4 and 6), referencing `\cref{fig:brpeg-regret}`.
- [x] Regret sign check: R(T) >= 0 for every T and every policy.
- [x] Rate qualifier: BR-PEG(correct) bounded << O(sqrt T) template
  <= EXP3's asymptotics, and O(T) for uniform-belief and
  mis-specified-prior policies (sections 1.1, 5).

Follow-ups (deliberately out of scope for this task):

- Dense-Theta BR-PEG (continuous type space) with a proper O(sqrt T)
  regret sweep; falls into the "strategising-H / dense-Theta" scope
  clause of T09.
- A sweep over sigma_obs to trace how the plateau height scales with
  observation noise (the plateau is the identification cost, which
  should grow like sigma_obs^2 through the per-round KL between
  neighbouring types).
