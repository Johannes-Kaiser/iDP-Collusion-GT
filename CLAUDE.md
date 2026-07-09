# Working with this repository

This repo contains the code, experiments, and paper for the **Privacy
Externality Game** — a game-theoretic analysis of sampling-based individual
differential privacy. The companion paper is
`2601.12922v1.pdf` (Kaiser et al., SaTML 2026).

## Rule 0 — the paper is a *final artefact*, not a log

**The paper (`paper/main.tex` and its `tex/` sub-files) must read as a
self-contained, first-time presentation of the work.** It is not a journal;
it does not track the history of how the work evolved.

That means:

- **Drop outdated material aggressively.** If a section, subsection,
  paragraph, sentence, footnote, or figure caption describes a version of
  the work that no longer holds, rewrite or delete it. Do *not* leave
  legacy content in place "for reference" and do *not* add sentences that
  compare the current formulation to a superseded one.
- **Never write "we used to...", "previously...", "an earlier draft...",
  "we now simplify...", "the two-weight formulation collapses to...", or
  anything of that shape.** The reader of the final paper does not know
  and does not care about the intermediate drafts. Present the current
  formulation as if it were the only one that ever existed.
- **Overwrite, don't append.** If a rewrite makes an equation or a plot
  obsolete, replace it. If a new experiment supersedes an old one, delete
  the old one. If a numerical value changes, propagate the change
  everywhere and delete the old value; do not leave both in the text.
- **The same rule applies to figure files, table files, and captions.**
  Rename figure files if their axis or content has changed; rename table
  labels if their meaning has changed. Do not maintain "v1" / "v2"
  siblings — replace.
- **Docstrings and code comments follow the same rule** when they show up
  in the paper's exposition path (module-level docstrings, config
  comments the paper cites, etc.). Working notes that only Claude or the
  author read can be more forgiving, but should still be pruned when
  they get stale.

Concretely: after any refactor that changes the mathematical formulation,
sweep the paper end-to-end and remove every trace of the old
formulation, including axis labels, figure filenames, notation, and prose
transitions that only made sense in the old formulation.

## Rule 1 — style consistency with the companion paper

The paper is meant to read as a matched pair with
`2601.12922v1.pdf`. Voice and structural conventions are extracted and
enforced (opening promise-vs-reality frame, first-person plural, signposted
transitions, single per-idea metaphor, no throat-clearing, contribution
bullets as 2–3-sentence statements). When rewriting, keep the two papers
aligned; when in doubt, imitate the companion paper's paragraph structure.

## Rule 2 — figures and caches

- `game_experiments/_cache_v2/` holds pickled experiment results. When you
  regenerate a figure, delete its cache first.
- Figures live in `paper/figures/`. Both `.pdf` and `.svg` are produced;
  the paper includes the `.pdf`.
- If a figure filename changes (e.g. because its parameter axis is
  renamed), update every `\includegraphics{...}` reference in the paper
  and delete the stale figure files.

## Rule 3 — running jobs (Slurm, GPUs, budgets)

All non-trivial compute goes through Slurm; do not launch long-running
work on the login node.

- **Prefer CPU.** The bulk of the game-theoretic experiments
  (mechanism-solver, PLD-accounting, best-response sweeps, equilibrium
  search) are CPU-bound. Do not request a GPU for these. Also use Slurm for CPU jobs though.
- **Allocate a GPU only when the job actually trains a neural network** —
  DP-SGD shadow-model training, membership-inference experiments,
  anything that touches PyTorch/Opacus on real data. When a GPU is
  needed, request the smallest one that fits (`--gres=gpu:1`), not a
  full node.
- **Be tight on the budget.** Set `--time`, `--mem`, and `--cpus-per-task`
  to what the job actually needs, not to the queue maximum. Overshooting
  wastes shared resources; undershooting is cheap to correct with a
  requeue.
- **Existing Slurm scripts** live in `slurm_scripts/`; use them as
  templates. When you write a new one, mirror their structure (module
  loads, venv activation, cache paths) rather than inventing your own.
- **Do not run experiments interactively from the login node** just
  because they "look short." If it needs more than a couple of minutes
  of CPU, submit it.

## Rule 4 — the LaTeX build

`cd paper && pdflatex -interaction=nonstopmode main.tex` produces
`main.pdf`. The build is expected to be warning-clean at the level of
compiler errors (a few `hyperref` PDF-string warnings on math-in-headings
are pre-existing and inconsequential). If a build introduces new
warnings or errors, resolve them before finishing the task.

## Rule 5 — code changes that alter equilibria

Numerical experiments that feed the paper's figures and tables are
sensitive to the utility form, the mechanism-solver tolerances, and the
grid parameters. Any change to `game/utility.py`, `game/mechanism.py`,
`game/continuous.py`, or the grid/parameter defaults in the experiment
scripts requires re-running the affected experiments and updating the
figures, tables, and any numerical values quoted in the prose. Do not
leave the paper referring to values that the code no longer produces.

## Rule 6 — what to keep out of the paper

- The paper is anonymous for submission (`\author{Anonymous Submission}`).
  Do not add author names, affiliations, or acknowledgements to the main
  file unless asked.
- Do not add "we thank...", "future work section titled...", or
  self-references to unpublished manuscripts unless the user explicitly
  asks.
- Do not add TODO comments, `XXX` markers, or planning notes to the LaTeX
  sources. Keep any working notes in scratch files, not in `paper/`.

## Rule 7 — terminology for the privacy budget

Never use the word **"currency"** to describe `(eps, delta)` or a
privacy budget — not in the paper, captions, comments, or docstrings.

`(eps, delta)` is a **budget**. It provides the *formal quantification of
a record's information leakage*, and it serves as the *basis of the
contract* between a data owner / contributor and the party (instance)
that processes the data. Frame it in exactly these terms.

Cite the framing properly: `dwork2006calibrating` for `(eps, delta)`
quantifying leakage; `boenisch2023have` and `kaiser2026your` for stating
iDP contracts in `(eps, delta)` form (they account internally in Rényi
DP, `mironov2017renyi` / `abadi2016deep`, but write the contract in
`(eps, delta)`); `cummings2024advancing` for `(eps, delta)` being the
form in which guarantees are deployed and communicated in practice. Do
not overclaim human-interpretability of `(eps, delta)` beyond what these
sources support.

## Concrete: the current utility

The player utility is

    u_i(eps) = V(sigma(eps)) - lambda_i * A_i(eps)

with a single per-player privacy weight. **Paper notation** (main2.tex):
the weight is `lambda_i` (introduced with the Westin heterogeneity
evidence in the iDP background), the advantage is `A_{PLD,i}` (macro
`\Apld{i}`), and the coalition malice weight is `kappa`. **Code
notation** (`game/utility.py`, experiment scripts) keeps `w_i` / `A_i`;
do not rename code symbols without also regenerating every figure that
prints them.
If you see references to `B_i`, `C_i`, `B/C`, or `benefit-to-cost` anywhere
(paper, code, docstrings, comments, captions, filenames), those are stale
and should be removed. Do not add explanations of how the single-weight
form relates to a hypothetical two-weight form: it doesn't need one.
