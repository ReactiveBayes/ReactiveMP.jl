# [Getting started](@id getting-started)

This page builds a small model by hand and runs inference on it, with the engine alone: what
RxInfer's `@model` and `infer` do for you. The model is a latent `x` with a normal prior, observed
through normal noise of known variance:

```math
x \sim \mathcal{N}(0, 10), \qquad y \mid x \sim \mathcal{N}(x, 1).
```

Its posterior is normal, with precision ``1/10 + 1`` and mean ``y / (1/10 + 1)``, and its evidence
is ``p(y) = \mathcal{N}(y \mid 0, 11)``: belief propagation computes both exactly, and the Bethe
free energy is ``-\log p(y)``.

## Packages

The engine defines no node. `NormalMeanVariance` and its rules come from
[`StandardMessagePassingRules`](https://reactivebayes.github.io/StandardMessagePassingRules.jl/dev/);
loading it is enough for the engine to find them. Rocket.jl provides `subscribe!`.

```@example getting-started
using ReactiveMP, StandardMessagePassingRules, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, get_stream_of_marginals
nothing # hide
```

## 1. Variables

A [`randomvar`](@ref) is inferred, a [`datavar`](@ref) receives observations, and a
[`constvar`](@ref) holds a fixed value (see [Variables](@ref lib-variables)).

```@example getting-started
x = randomvar(label = :x)
y = datavar(label = :y)
prior_mean, prior_var, noise_var = constvar(0.0), constvar(10.0), constvar(1.0)
nothing # hide
```

## 2. Factor nodes

A [`factornode`](@ref) connects a node type to its variables, interface by interface. Without a
factorisation, a node's variables form one cluster, and its rules are those of belief
propagation (see [Factor nodes](@ref lib-node)).

```@example getting-started
prior = factornode(NormalMeanVariance, [(:out, x), (:μ, prior_mean), (:v, prior_var)])
likelihood = factornode(NormalMeanVariance, [(:out, y), (:μ, x), (:v, noise_var)])
nothing # hide
```

## 3. Activation

Activation wires the streams, variables first, then nodes (see
[Inference lifecycle](@ref concepts-inference-lifecycle)). Each takes its options:
[`RandomVariableActivationOptions`](@ref), [`DataVariableActivationOptions`](@ref) and
[`ReactiveMP.FactorNodeActivationOptions`](@ref), here asking the nodes to track log scales (see
[Activation options](@ref lib-activation-options)). A constant needs none.

```@example getting-started
activate!(x, RandomVariableActivationOptions())
activate!(y, DataVariableActivationOptions())
for node in (prior, likelihood)
    activate!(node, FactorNodeActivationOptions(; logscales = true))
end
```

## 4. Subscribing

Nothing is computed until someone listens. Subscribe to the posterior of `x`, and to the free
energy, [`bethe_free_energy`](@ref), which takes every node and every variable, the constants
included.

```@example getting-started
posteriors = Marginal[]
energies = Float64[]
subscriptions = [
    subscribe!(get_stream_of_marginals(x), (q) -> push!(posteriors, q)),
    subscribe!(bethe_free_energy(Float64, (prior, likelihood), (x, y, prior_mean, prior_var, noise_var)), (f) -> push!(energies, f)),
]
nothing # hide
```

## 5. Observing

[`new_observation!`](@ref) gives `y` a value, which propagates through the graph: the likelihood
computes its message towards `x`, `x` multiplies it with the prior's, and the posterior and the
free energy update.

```@example getting-started
new_observation!(y, 2.0)
last(posteriors)
```

The posterior is the closed form, and its log scale (see [Log scales](@ref lib-logscale)) is the
log evidence, as is minus the free energy:

```@example getting-started
q = last(posteriors)
(mean(q) ≈ 2.0 / 1.1, var(q) ≈ 1 / 1.1, getlogscale(q) ≈ logpdf(Normal(0, sqrt(11)), 2.0), last(energies) ≈ -logpdf(Normal(0, sqrt(11)), 2.0))
```

Each new observation updates them again, which is how the engine serves streaming data:

```@example getting-started
new_observation!(y, 1.0)
mean(last(posteriors)), length(energies)
```

When the run is over, unsubscribe:

```@example getting-started
foreach(unsubscribe!, subscriptions)
```

## Next

- A variational model runs the same way, with a [`factornode`](@ref) factorisation such as
  `((:out,), (:μ,), (:v,))`, initial marginals set with [`ReactiveMP.set_initial_marginal!`](@ref),
  and the data given once per iteration.
- [Callbacks](@ref lib-callbacks) show every rule the engine runs.
- The rules come from the [rule packages](@ref ecosystem); how rules are defined is the subject of
  [`MessagePassingRulesBase`](https://reactivebayes.github.io/MessagePassingRulesBase.jl/dev/).
