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

```julia
f(x, y) = x^2 + y
node = factornode(DeltaFn{typeof(f)}, [(:out, z), ((:in, 1), x), ((:in, 2), y)]; nodefn = f)
activate!(node, FactorNodeActivationOptions(; algorithm = DeltaApproximation(; method = Unscented())))
```

```@docs
DeltaFn
DeltaApproximation
is_delta_node_compatible
DeltaMessagePassingRules.UnknownInverse
DeltaMessagePassingRules.KnownInverse
DeltaMessagePassingRules.approximate_normal
```
