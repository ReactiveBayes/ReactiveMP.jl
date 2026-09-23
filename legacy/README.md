# legacy/v6

The ReactiveMP 6.5 rule system and every node not yet ported to the new packages, kept for
reference while porting. Nothing here is loaded, tested or formatted: `test/runtests.jl` and
`scripts/formatter.jl` both skip `legacy/`.

The layout mirrors where each file used to live, so `legacy/v6/src/rules/beta/out.jl` was
`src/rules/beta/out.jl`:

- `src/rule.jl` — `@rule`, `@marginalrule`, `@call_rule`, `@test_rules` and the rule errors;
- `src/rules/`, `src/nodes/predefined/` — the v6 rules and node definitions;
- `src/nodes/nodes.jl`, `dependencies.jl`, `clusters.jl`, `src/score/{node,score}.jl` — the v6 engine files as
  they were before step 4 rewrote them: `@node`, its traits, the `Require*` dependencies,
  `@average_energy`;
- `src/approximations/`, `src/helpers/algebra/`, `src/fixes.jl`, `ext/` — the approximation
  methods, the algebra helpers, the `ForwardDiff` hot-fix from `src/fixes.jl` (which stays in
  the package, empty) and the extensions they needed;
- `test/` — the tests of all of the above.

Phase 5 ports each node from here into `lib/StandardMessagePassingRules` (or its sibling
packages), and deletes what it ported. `INVENTORY.md` says where each entity goes. The
behaviour the ports must reproduce is ReactiveMP 6.5.0's, which `compat/v6-comparison` runs.
