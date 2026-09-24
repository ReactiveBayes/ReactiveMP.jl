# legacy/v6

The ReactiveMP 6.5 rule system and every node not yet ported to the new packages, kept for
reference while porting. Nothing here is loaded, tested or formatted: `test/runtests.jl` and
`scripts/formatter.jl` both skip `legacy/`.

The layout mirrors where each file used to live, so `legacy/v6/src/rules/gcv/y.jl` was
`src/rules/gcv/y.jl`. Since Phase 5 closed (step 9), it holds only what Phase 6 ports from, or
deletes after skimming:

- `src/rules/`, `src/nodes/predefined/` — the unported nodes' rules and definitions:
  Autoregressive, ConjugateAR, BIFM and its helper, the Pólya nodes, the transitions, Flow,
  and SoftDot (Delta left in Phase 6 step 2, and GaussianCoupling, Probit and GCV in step 3, all
  ported);
  `rules/mv_normal_mean_precision/marginals.jl` holds MvNormalMeanPrecision's two marginal
  rules for BIFM's `TerminalProdArgument`;
- `src/helpers/algebra/` — the permutation matrix, Flow's, and the standard-basis vector and the
  companion matrix, which AR uses. `common.jl` left in Phase 6 step 1: `mul_trace`,
  `rank1update`, `negate_inplace!` and `mul_inplace!` are Standard's, and the rest had no user;
- `test/` — the tests of all of the above.

Gone in Phase 6: `src/approximations/`, whose methods step 1 ported to
`MessagePassingRulesApproximations` or deleted, and whose `cvi_projection.jl` step 2 ported to the
Delta package with its extension; `ext/`; and `src/fixes.jl`, which nothing used any more.

Gone in Phase 5 step 9: the v6 engine and rule-system files (`rule.jl`, `nodes/{nodes,dependencies,
clusters}.jl`, `score/`), which the new packages replace and git and the 6.5.0 release keep; v6's
rule fallbacks and `StandaloneDistributionNode`, which are not carried over; and the stale
include lists. Before that, each Phase 5 step deleted what it ported. `INVENTORY.md` says where each entity goes. The
behaviour the ports must reproduce is ReactiveMP 6.5.0's, which `compat/v6-comparison` runs.
