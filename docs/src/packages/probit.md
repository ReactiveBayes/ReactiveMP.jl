# [Probit](@id packages-probit)

```@docs
ProbitMessagePassingRules
```

`Probit` observes a binary `out`, or its probability, through `Φ(in)`. Its algorithm,
[`ProbitEP`](@ref), is expectation propagation: the rule towards `in` reads the message on `in`
itself, and the node declares `NormalMeanPrecision(0.0, 100.0)` as that message's start, which
the engine sets where the model sets none. Under `DefaultAlgorithm()` the node runs plain
rules, which follow the factorisation and do not read their own edge.

```julia
node = factornode(Probit, [(:out, y), (:in, x)])
activate!(node, FactorNodeActivationOptions())           # ProbitEP(), the node's own
```

```@docs
Probit
ProbitEP
```
