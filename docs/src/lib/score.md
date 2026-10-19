# [Free energy](@id lib-score)

The engine computes the [Bethe free energy](@extref MessagePassingRulesBase glossary-bethe-free-energy),
the objective that message passing minimises. It is a sum of local terms, one per factor node
and one per variable, and the engine updates each term as the marginals it depends on change.

```@setup score
using ReactiveMP, MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions, Rocket
import ReactiveMP: activate!, FactorNodeActivationOptions, MessageProductContext, get_stream_of_marginals, set_initial_marginal!
include(joinpath(pkgdir(ReactiveMP), "docs", "nodes.jl"))
```

The examples use the `Gaussian` node of [The example node](@ref example-node).

## [The Bethe free energy](@id lib-score-bethe)

```math
\mathcal{F}[q] = \underbrace{\sum_f \langle -\log f \rangle_{q_f}}_{\text{average energies}} - \underbrace{\sum_f H[q_f]}_{\text{factor entropies}} + \underbrace{\sum_x (d_x - 1)\, H[q_x]}_{\text{variable entropies}}
```

The first two sums run over the factor nodes, with ``q_f`` the local marginal of a node's
clusters. The last runs over the variables, with ``d_x`` the degree of a variable, its number of
connected nodes, and ``q_x`` its marginal. The free energy bounds minus the log evidence from
above, ``\mathcal{F}[q] \geq -\log p(y)``. Belief propagation in a tree reaches the bound.

## [Computing the free energy](@id lib-score-bethe-stream)

[`bethe_free_energy`](@ref) returns the free energy of an activated graph as a stream. It takes
every node and every variable of the graph, the data and the constants included, whose point
entropies cancel those in the nodes' terms. Subscribe to it before you give the data:

```@example score
x, y = randomvar(label = :x), datavar(label = :y)
constants = (constvar(0.0), constvar(10.0), constvar(1.0))
prior = factornode(Gaussian, [(:out, x), (:μ, constants[1]), (:v, constants[2])])
likelihood = factornode(Gaussian, [(:out, y), (:μ, x), (:v, constants[3])])
activate!(x, RandomVariableActivationOptions())
activate!(y, DataVariableActivationOptions())
foreach(n -> activate!(n, FactorNodeActivationOptions()), (prior, likelihood))

energies = Float64[]
subscription = subscribe!(bethe_free_energy(Float64, (prior, likelihood), (x, y, constants...)), (f) -> push!(energies, f))
new_observation!(y, 2.0)
last(energies), -logpdf(Normal(0.0, sqrt(11.0)), 2.0)
```

The graph is a tree, inferred exactly by belief propagation, so the free energy is minus the log
evidence. The stream emits once every term has a value, and again whenever one of them updates.

In a variational graph, you give the data once per iteration, and the stream emits once per
iteration. The chain below has a [mean-field](@extref MessagePassingRulesBase glossary-mean-field)
node between `x1` and `x2`:

```@example score
x1, x2, y2 = randomvar(label = :x1), randomvar(label = :x2), datavar(label = :y2)
chain_constants = (constvar(0.0), constvar(10.0), constvar(1.0), constvar(1.0))
chain = (
    factornode(Gaussian, [(:out, x1), (:μ, chain_constants[1]), (:v, chain_constants[2])]),
    factornode(Gaussian, [(:out, x2), (:μ, x1), (:v, chain_constants[3])], ((:out,), (:μ,), (:v,))),
    factornode(Gaussian, [(:out, y2), (:μ, x2), (:v, chain_constants[4])]),
)
foreach(v -> activate!(v, RandomVariableActivationOptions()), (x1, x2))
activate!(y2, DataVariableActivationOptions())
foreach(n -> activate!(n, FactorNodeActivationOptions()), chain)
set_initial_marginal!(x1, NormalMeanVariance(0.0, 1.0))

chain_energies = Float64[]
chain_subscription = subscribe!(bethe_free_energy(Float64, chain, (x1, x2, y2, chain_constants...)), (f) -> push!(chain_energies, f))
for iteration in 1:20
    new_observation!(y2, 2.0)
end
length(chain_energies), last(chain_energies), -logpdf(Normal(0.0, sqrt(12.0)), 2.0)
```

The free energy converges to a value above minus the log evidence. The gap is the
Kullback–Leibler divergence from the mean-field posterior ``q(x_1) q(x_2)`` to the exact one.

The keyword `algorithm` gives the algorithm each node runs under, as a function of the node, so
that each node contributes the average energy its rules are consistent with. Its default,
`nothing`, is every node's default algorithm. The average energies run as the node's rules do,
with the rule context and the diagnostics the node was activated with.

```@docs
bethe_free_energy
```

## [Tracing the free energy](@id lib-score-tracing)

Each term reports itself to [callbacks](@ref lib-callbacks): a factor node's to the node's, as
[`ReactiveMP.AfterFactorBoundFreeEnergyEvent`](@ref), with its average energy, its entropies and
the average energy that ran; a variable's to those of its marginal's context, as
[`ReactiveMP.AfterVariableBoundEntropyEvent`](@ref). The first graph again, traced:

```@example score
terms = []
record = (event) -> push!(terms, event)
callbacks = (after_factor_bound_free_energy = record, after_variable_bound_entropy = record)

xt, yt = randomvar(label = :x), datavar(label = :y)
traced_constants = (constvar(0.0), constvar(10.0), constvar(1.0))
traced = (
    factornode(Gaussian, [(:out, xt), (:μ, traced_constants[1]), (:v, traced_constants[2])]),
    factornode(Gaussian, [(:out, yt), (:μ, xt), (:v, traced_constants[3])]),
)
activate!(xt, RandomVariableActivationOptions(nothing, MessageProductContext(), MessageProductContext(; callbacks)))
activate!(yt, DataVariableActivationOptions())
foreach(n -> activate!(n, FactorNodeActivationOptions(; callbacks)), traced)

traced_energies = Float64[]
traced_subscription = subscribe!(bethe_free_energy(Float64, traced, (xt, yt, traced_constants...)), (f) -> push!(traced_energies, f))
new_observation!(yt, 2.0)
foreach(event -> println(sprint(show, event; context = :compact => true)), terms)
```

A term is a `BayesBase.CountingReal`: the entropy of a point mass is infinite, and the term counts
those apart from its value. The free energy is the terms' sum less one point entropy for each
connection of a data or constant variable, here four, which cancel the infinities:

```@example score
float(sum(event.result for event in terms) - BayesBase.CountingReal(Float64, 4)), last(traced_energies)
```

## [Score types](@id lib-score-types)

A factor node's average energy, the ``\langle -\log f \rangle_q`` term, belongs to its rule
package. The package declares it with
[`@define_average_energy`](@extref MessagePassingRulesBase.@define_average_energy) next to the
node's rules, and the engine finds it with
[`find_average_energy`](@extref MessagePassingRulesBase.find_average_energy). The engine combines
it with the entropies into these contributions, each computed by [`score`](@ref):

| Type | Represents | Where used |
|------|-----------|-----------|
| [`DifferentialEntropy`](@ref) | ``-\int q \log q``, the entropy of a marginal | factor and variable nodes |
| [`FactorBoundFreeEnergy`](@ref) | a factor node's term: its average energy less its clusters' entropies | factor nodes |
| [`VariableBoundEntropy`](@ref) | a variable's term, its entropy weighted by its degree less one | variable nodes |

```@example score
q = Marginal[]
marginal_subscription = subscribe!(get_stream_of_marginals(x), (marginal) -> push!(q, marginal))
score(DifferentialEntropy(), last(q)), entropy(NormalMeanVariance(2.0 / 1.1, 1 / 1.1))
```

A node without an average energy for its clusters' marginals makes the free energy an error that
names the node, rather than a wrong number. The node below has a rule towards `μ` and no
average energy:

```@example score
struct Shifted end

@define_factor_node(node = Shifted, type = Stochastic, interfaces = [:out, :μ])

@define_message_update_rule(
    node = Shifted, target = :μ, args = (q[:out]::PointMass,),
    body = (args) -> NormalMeanVariance(mean(args.q[:out]) - 1, 1.0),
)

xs, ys, cs = randomvar(label = :x), datavar(label = :y), (constvar(0.0), constvar(10.0))
nodes = (
    factornode(Gaussian, [(:out, xs), (:μ, cs[1]), (:v, cs[2])]),
    factornode(Shifted, [(:out, ys), (:μ, xs)], ((:out,), (:μ,))),
)
activate!(xs, RandomVariableActivationOptions())
activate!(ys, DataVariableActivationOptions())
foreach(n -> activate!(n, FactorNodeActivationOptions()), nodes)
failing = subscribe!(bethe_free_energy(Float64, nodes, (xs, ys, cs...)), (f) -> println(f))

try
    new_observation!(ys, 2.0)
catch err
    showerror(stdout, err)
end
```

```@docs
score
FactorBoundFreeEnergy
VariableBoundEntropy
DifferentialEntropy
```
