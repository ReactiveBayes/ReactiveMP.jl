# legacy/v6

The ReactiveMP 6.5 rule system and every node not yet ported to the new packages, kept for
reference while porting. Nothing here was loaded, tested or formatted: `test/runtests.jl` and
`scripts/formatter.jl` both skip `legacy/`.

**It is empty.** DiscreteTransition, the last node, left in Phase 6 step 9, ported. Phase 6 step 10
deletes this directory. Before it, Delta left in step 2, GaussianCoupling, Probit and GCV in
step 3, Autoregressive, ConjugateAR and SoftDot in step 4, ContinuousTransition in step 5, the
Pólya nodes in step 6, BIFM and its helper in step 7, and Flow in step 8. MvNormalMeanPrecision's
two marginal rules for BIFM's `TerminalProdArgument` left in step 7 unported: only the free energy
of a BIFM model reached them, and that is not supported.

Gone in Phase 6 besides the nodes: `src/helpers/algebra/`, whose permutation matrix left with Flow
in step 8, whose standard-basis vector and companion matrix left with AR in step 4, and whose
`common.jl` step 1 split between Standard (`mul_trace`, `rank1update`, `negate_inplace!`,
`mul_inplace!`) and deletion; `src/approximations/`, whose methods step 1 ported to
`MessagePassingRulesApproximations` or deleted, and whose `cvi_projection.jl` step 2 ported to the
Delta package with its extension; `ext/`; and `src/fixes.jl`, which nothing used any more.

Gone in Phase 5 step 9: the v6 engine and rule-system files (`rule.jl`, `nodes/{nodes,dependencies,
clusters}.jl`, `score/`), which the new packages replace and git and the 6.5.0 release keep; v6's
rule fallbacks and `StandaloneDistributionNode`, which are not carried over; and the stale
include lists. Before that, each Phase 5 step deleted what it ported. `INVENTORY.md` says where
each entity went. The behaviour the ports reproduce is ReactiveMP 6.5.0's, which
`compat/v6-comparison` runs.
