# Hero schematic + repository hygiene after the main2 rewrite

**Priority:** L  **Effort:** L  **Category:** paper polish / cleanup

## Hero figure
`HERO_FIGURE_BRIEF.md` specifies a TikZ schematic (`paper/tex/figs/hero_peg.tex`)
of the externality pipeline (budgets -> shared sigma -> sampling rates ->
realised advantage). It was never built. main2 currently opens with prose
only; a page-1 schematic would strengthen the intro. Build it per the brief
and include in `tex2/01_intro.tex`.

## Hygiene (do only after main2 is accepted as the canonical paper)
- Stale on disk, referenced by nothing: `paper/figures/fig_E1..E6_*`
  (superseded grid-based series), `paper/tex/table_NE.tex` (K=22 v1),
  `paper/tex/table_NE_v2.tex` (K=22), `fig_C6_locus_K44` remnants if any.
- `paper/figures/fig_C9_federated_measurement_ceiling.*` is SYNTHETIC data
  with a "measurement" name — delete or regenerate from the real T25 run
  (see task 02).
- The old `paper/tex/` tree and `main.tex` remain the source of
  `main_old.pdf`; decide whether to keep them buildable or archive them.
- `game_experiments/_cache_H1/n2_adult/` vs task-spec name
  `n2_credit_card`: the H1 run used Adult; keep names consistent in any
  future rerun.
