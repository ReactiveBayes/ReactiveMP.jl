# [The Delta node](@id packages-delta)

```@docs
DeltaMessagePassingRules
```

The Delta node represents `z = f(x₁, …, xₙ)` for a deterministic function `f`, given when the
node is created. A deterministic node's clusters are its output and the joint over its inputs,
so the node takes no factorisation. Its rules run under its own algorithm, [`DeltaApproximation`](@ref), which
names the approximation method and, optionally, a known inverse of `f`. An input connected to a
constant or to data is folded into the node function, so the rules see only the random inputs
and reach the function as `getnodefn(ctx.node, Target(:out))`.

The node takes two Gaussian methods from `MessagePassingRulesApproximations`: `Unscented()`, the
unscented transform, and `Linearization()`, the first-order expansion at the inputs' means. Both
run the same rules, which differ only in how the method pushes the inputs' moments through `f`.

```julia
f(x, y) = x^2 + y
node = factornode(DeltaFn{typeof(f)}, [(:out, z), ((:in, 1), x), ((:in, 2), y)]; nodefn = f)
activate!(node, FactorNodeActivationOptions(; algorithm = DeltaApproximation(; method = Unscented())))
```

```@docs
DeltaFn
DeltaApproximation
is_delta_node_compatible
DeltaMessagePassingRules.delta_method_hint
DeltaMessagePassingRules.UnknownInverse
DeltaMessagePassingRules.KnownInverse
DeltaMessagePassingRules.approximate_normal
DeltaMessagePassingRules.forward_statistics
```

## Projection

`CVIProjection` samples the inputs, pushes the samples through `f`, and projects onto an
exponential family with ExponentialFamilyProjection, so it handles functions and families the
Gaussian methods cannot. Its rules are in a package extension, loaded with
`using ExponentialFamilyProjection`; until then the node does not accept it, and the error says
so. The samples come from the rule context's generator, which the engine owns, and the joint rule
keeps its result as the next call's proposal, so the method carries state and its rules are
impure.

```@docs
CVIProjection
CVISamplingStrategy
FullSampling
MeanBased
ProposalDistributionContainer
DeltaMessagePassingRules.get_kth_in_form
```
