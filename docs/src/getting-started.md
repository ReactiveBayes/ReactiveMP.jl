# [Getting started](@id getting-started)

This page builds a small model by hand and runs inference on it with the engine alone. It does
what RxInfer's `@model` and `infer` do for you. The model is a latent `x` with a normal prior,
observed through normal noise of known variance:

```math
x \sim \mathcal{N}(0, 10), \qquad y \mid x \sim \mathcal{N}(x, 1).
```

Its posterior is normal, with precision ``1/10 + 1`` and mean ``y / (1/10 + 1)``. Its evidence
is ``p(y) = \mathcal{N}(y \mid 0, 11)``. [Belief propagation](@extref MessagePassingRulesBase glossary-belief-propagation)
computes both exactly, and the [Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy)
of the graph equals ``-\log p(y)``.

## Packages and the node

The engine knows no [factor node](@extref MessagePassingRulesBase glossary-factor-node) until
one is declared. Nodes and their [rules](@extref MessagePassingRulesBase glossary-rule) come
from rule packages, which declare them with
[`MessagePassingRulesBase`](@extref MessagePassingRulesBase MessagePassingRulesBase). This site
declares one small node of its own, `Gaussian`, a normal density with a known variance, in a file
that every page includes ([The example node](@ref example-node) shows it). Rocket.jl provides
`subscribe!`.

```@example getting-started
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, get_stream_of_marginals

include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
nothing # hide
```

The declaration names the node's [interfaces](@extref MessagePassingRulesBase glossary-interface):
`out`, the output, `μ`, the mean, and `v`, the variance.

```@example getting-started
MessagePassingRulesBase.nodespec(Gaussian)
```

Both factors of the model are this node: the prior is ``\mathcal{N}(x \mid 0, 10)`` and the
likelihood ``\mathcal{N}(y \mid x, 1)``.

## 1. Variables

A [`randomvar`](@ref) is inferred. A [`datavar`](@ref) receives observations. A [`constvar`](@ref)
holds a fixed value, such as a known mean or variance (see [Variables](@ref lib-variables)).

```@example getting-started
x = randomvar(label = :x)
y = datavar(label = :y)
prior_mean, prior_var, noise_var = constvar(0.0), constvar(10.0), constvar(1.0)
nothing # hide
```

## 2. Factor nodes

A [`factornode`](@ref) connects a node type to its variables, one `(interface, variable)` pair
per edge (see [Factor nodes](@ref lib-node)).

```@example getting-started
prior = factornode(Gaussian, [(:out, x), (:μ, prior_mean), (:v, prior_var)])
likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, noise_var)])
```

The likelihood lists its edges, the variable on each and its kind, and its
[clusters](@extref MessagePassingRulesBase glossary-cluster). Without a
[factorisation](@extref MessagePassingRulesBase glossary-factorisation), all of a node's
interfaces form one cluster, `(out, μ, v)`, and the engine runs the node's belief propagation
rules.

## 3. Activation

Activation wires the streams, variables first, then nodes (see
[Inference lifecycle](@ref concepts-inference-lifecycle)). Each takes its options:
[`RandomVariableActivationOptions`](@ref), [`DataVariableActivationOptions`](@ref) and
[`ReactiveMP.FactorNodeActivationOptions`](@ref). Here the nodes track
[log scales](@extref MessagePassingRulesBase glossary-log-scale) (see
[Activation options](@ref lib-activation-options)). A constant needs no activation.

```@example getting-started
activate!(x, RandomVariableActivationOptions())
activate!(y, DataVariableActivationOptions())
for node in (prior, likelihood)
    activate!(node, FactorNodeActivationOptions(; logscales = true))
end
```

## 4. Subscribing

The engine computes nothing until something listens. Subscribe to the posterior of `x`, and to
the free energy. [`bethe_free_energy`](@ref) takes every node and every variable of the graph,
the constants included.

```@example getting-started
posteriors = Marginal[]
energies = Float64[]
variables = (x, y, prior_mean, prior_var, noise_var)
subscriptions = [
    subscribe!(get_stream_of_marginals(x), (q) -> push!(posteriors, q)),
    subscribe!(bethe_free_energy(Float64, (prior, likelihood), variables), (f) -> push!(energies, f)),
]
nothing # hide
```

## 5. Observing

[`new_observation!`](@ref) gives `y` a value, which propagates through the graph. The likelihood
computes its [message](@extref MessagePassingRulesBase glossary-message) towards `x`. The
variable `x` multiplies it with the prior's message, and the posterior and the free energy
update.

```@example getting-started
new_observation!(y, 2.0)
q = last(posteriors)
```

The posterior is in its natural form: weighted mean ``y = 2`` and precision ``1.1``. It matches
the closed form, mean ``2 / 1.1`` and variance ``1 / 1.1``:

```@example getting-started
mean(q) ≈ 2.0 / 1.1, var(q) ≈ 1 / 1.1
```

The posterior's log scale (see [Log scales](@ref lib-logscale)) is the log evidence, and the free
energy is minus the log evidence:

```@example getting-started
log_evidence = logpdf(Normal(0, sqrt(11)), 2.0)
getlogscale(q) ≈ log_evidence, last(energies) ≈ -log_evidence
```

Each new observation updates them again. This is how the engine serves streaming data:

```@example getting-started
new_observation!(y, 1.0)
mean(last(posteriors)), length(energies)
```

When the run is over, unsubscribe:

```@example getting-started
foreach(unsubscribe!, subscriptions)
```

## Next

- A variational model runs the same way. Give [`factornode`](@ref) a factorisation, such as
  `((:out,), (:μ,), (:v,))`, and the engine runs the node's variational rules;
  [Message passing](@ref concepts-message-passing-demo) runs this graph both ways. A loopy or
  variational graph starts from initial marginals, set with
  [`ReactiveMP.set_initial_marginal!`](@ref), and takes its data once per iteration.
- [Callbacks](@ref lib-callbacks) show every rule the engine runs.
- [The example node](@ref example-node) shows how `Gaussian` and its rules are declared.
  [Your first node](@extref MessagePassingRulesBase tutorial-first-node) builds such a node step
  by step.
- Models with the standard distributions use the nodes of
  [`StandardMessagePassingRules`](@extref StandardMessagePassingRules StandardMessagePassingRules)
  and the other [rule packages](@ref ecosystem).
  [RxInfer](https://reactivebayes.github.io/RxInfer.jl/stable/) specifies such models with
  `@model` and runs them on this engine.
