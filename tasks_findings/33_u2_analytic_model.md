# The exact analytic model of $u_2(\varepsilon_2 \mid \varepsilon_1)$

The complete mathematical description of player 2's conditional utility
in the symmetric two-player PEG instance. Everything below is exact; no
asymptotics, no accountant approximations, no discretisation.

## 0. Instance constants

| symbol | value | meaning |
|---|---|---|
| $\lvert G_1\rvert=\lvert G_2\rvert$ | $5000$ | group sizes |
| $B$ | $128$ | expected batch size (fixed) |
| $T$ | $390$ | composition steps |
| $\delta$ | $10^{-12}$ | contract failure probability |
| $\bar p$ | $10^{-5}$ | participation floor |
| $\lambda_2$ | $1$ | privacy weight |

## 1. The mechanism and its exact privacy profile

One step of the mechanism, as seen by one record of a player sampled at
rate $p$ under noise scale $\sigma$ (Poisson subsampling, remove
adjacency, sensitivity $1$), is the pair of output distributions

$$
P_{\sigma,p} \;=\; (1-p)\,\mathcal N(0,\sigma^2) + p\,\mathcal N(1,\sigma^2),
\qquad
Q_{\sigma} \;=\; \mathcal N(0,\sigma^2).
$$

The privacy loss of an output $x$ is

$$
\ell_{\sigma,p}(x) \;=\;
\log\frac{\mathrm dP_{\sigma,p}}{\mathrm dQ_\sigma}(x)
\;=\; \log\!\Bigl[(1-p) + p\,e^{(2x-1)/(2\sigma^2)}\Bigr],
$$

and over the $T$-fold composition the loss is the i.i.d. sum
$L_T = \sum_{t=1}^{T}\ell_{\sigma,p}(X_t)$. The composed mechanism's
privacy profile — the exact $(\varepsilon,\delta)$ trade-off — is the
hockey-stick divergence of the product measures:

$$
\boxed{\;
\delta_{\sigma,p}(\varepsilon)
\;=\; \Pr_{X\sim P_{\sigma,p}^{\otimes T}}\!\bigl[L_T > \varepsilon\bigr]
\;-\; e^{\varepsilon}\,
      \Pr_{X\sim Q_{\sigma}^{\otimes T}}\!\bigl[L_T > \varepsilon\bigr]
\;=\; \mathbb E_{Q_\sigma^{\otimes T}}
      \bigl[(e^{L_T}-e^{\varepsilon})_+\bigr].
\;}
$$

Exact monotonicity, all strict on the relevant domain:
$\delta_{\sigma,p}(\varepsilon)$ is decreasing and convex in
$\varepsilon$, decreasing in $\sigma$, increasing in $p$. Its
$\varepsilon$-derivative has the closed form

$$
\frac{\partial \delta_{\sigma,p}}{\partial\varepsilon}
= -\,e^{\varepsilon}\Pr_{Q_\sigma^{\otimes T}}\!\bigl[L_T>\varepsilon\bigr].
$$

Define the **exact budget function** as the partial inverse in
$\varepsilon$:

$$
\mathcal E(\sigma,p) \;:=\; \text{the unique } \varepsilon
\text{ with } \delta_{\sigma,p}(\varepsilon) = \delta .
$$

$\mathcal E$ is real-analytic, strictly decreasing in $\sigma$, strictly
increasing in $p$.

## 2. The mechanism map

Given $(\varepsilon_1,\varepsilon_2)$, the mechanism
$(\sigma^*, p_1^*, p_2^*)$ is the unique solution of

$$
\mathcal E(\sigma, p_i) = \varepsilon_i \quad (i=1,2),
\qquad
\lvert G_1\rvert\,p_1 + \lvert G_2\rvert\,p_2 = B,
\qquad
p_i \ge \bar p .
$$

*Uniqueness.* Each budget equation inverts to
$p_i = \varphi(\sigma,\varepsilon_i)$ — the rate at which noise $\sigma$
exhausts exactly $\varepsilon_i$ — with $\varphi$ strictly increasing in
$\sigma$ (more noise supports more sampling at the same budget). The
batch equation
$\sum_i \lvert G_i\rvert\,\varphi(\sigma,\varepsilon_i)=B$
therefore has exactly one root $\sigma^*$; the rates follow. If the
solution violates a floor $p_i \ge \bar p$, the profile is infeasible and
$u_2$ is undefined there.

## 3. The utility

$$
\boxed{\;
u_2(\varepsilon_2\mid\varepsilon_1)
\;=\; \frac{1}{1+\sigma^{*2}}
\;-\;\lambda_2\,\mathrm{Adv}\bigl(\sigma^*,p_2^*\bigr)
\;}
$$

with the exact membership advantage: the total-variation distance of the
composed worst-case pair, equivalently the profile of Section 1 at
$\varepsilon=0$,

$$
\mathrm{Adv}(\sigma,p)
\;=\; \mathrm{TV}\!\bigl(P_{\sigma,p}^{\otimes T},\,Q_\sigma^{\otimes T}\bigr)
\;=\; \delta_{\sigma,p}(0)
\;=\; \Pr_{P^{\otimes T}}[L_T>0]-\Pr_{Q^{\otimes T}}[L_T>0].
$$

This is the advantage of the Bayes-optimal likelihood-ratio test on the
mechanism's outputs — the realised (level-2) vulnerability, as opposed to
the contracted ceiling $A_{\varepsilon,\delta}$.

$\varepsilon_1$ appears in $u_2$ **only** through $(\sigma^*, p_2^*)$;
the mechanism map is the externality's entire transmission channel.

## 4. Domain of the slice

Given $\varepsilon_1$, the feasible interval of $\varepsilon_2$ is
$[\underline\varepsilon(\varepsilon_1),\,\bar\varepsilon(\varepsilon_1)]$,
with both endpoints determined by the same exact objects:

- **Lower endpoint** ($p_2$ at the floor): set $p_2=\bar p$; the batch
  pins $p_1 = B/\lvert G_1\rvert - \bar p$; the noise
  $\bar\sigma(\varepsilon_1)$ solves
  $\mathcal E(\sigma, p_1)=\varepsilon_1$ alone; then
  $\underline\varepsilon(\varepsilon_1)
  =\mathcal E\bigl(\bar\sigma(\varepsilon_1),\bar p\bigr)$.
- **Upper endpoint**: the mirror statement with the players exchanged.

This is the participation frontier; on the open interval the slice
$\varepsilon_2 \mapsto u_2(\varepsilon_2\mid\varepsilon_1)$ is
real-analytic.

## 5. Exact comparative statics along the slice

Away from the endpoints, implicit differentiation of the system in
Section 2 gives, with $F_i := \mathcal E(\cdot\,,\cdot)$ evaluated at
$(\sigma^*,p_i^*)$,

$$
\frac{\partial \sigma^*}{\partial \varepsilon_2}
= \frac{\lvert G_2\rvert\big/\partial_p F_2}
       {\lvert G_1\rvert\,\dfrac{\partial_\sigma F_1}{\partial_p F_1}
      + \lvert G_2\rvert\,\dfrac{\partial_\sigma F_2}{\partial_p F_2}}
\;<\;0,
\qquad
\frac{\partial p_1^*}{\partial \varepsilon_2}
= -\,\frac{\partial_\sigma F_1}{\partial_p F_1}\,
   \frac{\partial \sigma^*}{\partial \varepsilon_2}\;<\;0,
$$

$$
\frac{\partial p_2^*}{\partial \varepsilon_2}
= \frac{1}{\partial_p F_2}
  \Bigl(1-\partial_\sigma F_2\,
        \frac{\partial \sigma^*}{\partial \varepsilon_2}\Bigr)\;>\;0 .
$$

Loosening $\varepsilon_2$ thins the noise, grows player 2's share of the
fixed batch, and shrinks player 1's share so that player 1's contract
still binds. The slice derivative of the utility — the exact first-order
condition whose zero set is $\mathrm{BR}_2$ — is

$$
\frac{\mathrm d u_2}{\mathrm d \varepsilon_2}
= V'(\sigma^*)\,\frac{\partial\sigma^*}{\partial\varepsilon_2}
-\lambda_2\!\left(
   \partial_\sigma\mathrm{Adv}\cdot
     \frac{\partial\sigma^*}{\partial\varepsilon_2}
 + \partial_p\mathrm{Adv}\cdot
     \frac{\partial p_2^*}{\partial\varepsilon_2}\right),
\qquad V'(\sigma) = \frac{-2\sigma}{(1+\sigma^2)^2},
$$

where the partials of $\mathcal E$ and $\mathrm{Adv}$ are obtained by
differentiating the expectations of Section 1 under the integral sign
(dominated convergence applies; the integrands are Gaussian mixtures).
The first term is the utility gain of thinner noise and saturates
($V$ is bounded); the two advantage terms are exposure costs and grow;
their crossing pattern generates the two local maxima of the slice — the
tight branch and the loose branch — and the trough between them.

## 6. Where analyticity ends

The entire model reduces, without loss, to **one scalar transcendental
object**: the distribution of the i.i.d. sum $L_T$ (a $T$-fold
convolution of the pushforward of a Gaussian mixture through
$\ell_{\sigma,p}$). Every quantity above — $\delta_{\sigma,p}$,
$\mathcal E$, $\mathrm{Adv}$, the mechanism map, $u_2$, its derivatives —
is an explicit finite composition of that object with elementary
functions, integrals, and one implicit inversion whose uniqueness is
guaranteed by strict monotonicity. No closed elementary form for the
convolution is known for $T>1$; this is the single point at which the
model must be evaluated rather than expanded.
