# `lib/` — GCVMessagePassingRules, until it is registered

The message passing rule packages each have a repository of their own,
`ReactiveBayes/<Package>.jl`, and are registered in General: the standard nodes,
[StandardMessagePassingRules](https://github.com/ReactiveBayes/StandardMessagePassingRules.jl),
and one package per remaining node, Delta, GaussianCoupling, Autoregressive, SoftDot,
ContinuousTransition, Pólya (GPL-3), BIFM, Flow and DiscreteTransition. So are the rule system
they are written with,
[MessagePassingRulesBase](https://github.com/ReactiveBayes/MessagePassingRulesBase.jl), the
numerics they share,
[MessagePassingRulesApproximations](https://github.com/ReactiveBayes/MessagePassingRulesApproximations.jl),
and their test tooling,
[MessagePassingRulesTestUtils](https://github.com/ReactiveBayes/MessagePassingRulesTestUtils.jl).
A new rule package starts from
[MessagePassingRulesTemplate.jl](https://github.com/ReactiveBayes/MessagePassingRulesTemplate.jl).

This directory holds one package, `GCVMessagePassingRules`: the GCV node, its
`ExponentialLinearQuadratic`, and the normal nodes' rules for it. It is split into
[ReactiveBayes/GCVMessagePassingRules.jl](https://github.com/ReactiveBayes/GCVMessagePassingRules.jl)
and registering; the copy here is frozen, a change goes to its repository, and `lib/` is
deleted once it is registered. Its siblings come from General.

```bash
make test-gcv     # its suite, with test_args as the root's `make test`
make docs-gcv     # its site, into lib/GCVMessagePassingRules/docs/build
```

Items tagged `:slow` are skipped unless `TEST_ALL=true`, which CI sets. No `Manifest.toml` under
`lib/` is committed.
