# [Free energy](@id lib-score)

ReactiveMP.jl computes the **Bethe free energy** as its variational objective during inference. The free energy decomposes into local contributions from each factor node and each variable node, which are accumulated reactively as messages update.

## [The Bethe free energy](@id lib-score-bethe)

The Bethe free energy approximates the negative log-evidence of the model:

```math
\mathcal{F}_{\text{Bethe}}[q] = \underbrace{\sum_f \langle -\log f \rangle_{q_f}}_{\text{average energy}} - \underbrace{\sum_f H[q_f]}_{\text{factor entropies}} + \underbrace{\sum_x (d_x - 1)\, H[q_x]}_{\text{variable entropies}}
```

where:
- The sum over `f` runs over all factor nodes, with `q_f` the local marginal over the factor's variables.
- The sum over `x` runs over all variable nodes, with `d_x` the degree (number of connected factors) and `q_x` the marginal of that variable.

ReactiveMP.jl computes each term reactively: whenever a marginal changes, the local contribution is recomputed and can be accumulated by subscribing to the score streams.

## [Computing the free energy](@id lib-score-bethe-stream)

[`bethe_free_energy`](@ref) returns the free energy of an activated graph as a stream, which
emits once every component has a value and again whenever one of them updates. It takes every
node and every variable of the graph, the data and the constants included, whose point entropies
cancel those of the nodes' terms:

```julia
energies = Float64[]
subscribe!(bethe_free_energy(Float64, nodes, variables; algorithm = (node) -> nothing), (f) -> push!(energies, f))
```

`algorithm(node)` is the algorithm each node runs under, as given at activation, so the
energy each node contributes is the one its rules are consistent with; `nothing` is the node's
default. [Getting started](@ref getting-started) computes it for a model whose free energy is
minus its log evidence.

The average energies run with the engine's context, [`ReactiveMP.node_context`](@ref)`(node)`,
without the services of the activation option `context` and without the diagnostics.

```@docs
bethe_free_energy
```

## [Score types](@id lib-score-types)

A factor node's average energy, the `⟨-log f⟩_q` term, is its rule package's: it is declared
with [`@define_average_energy`](@extref MessagePassingRulesBase.@define_average_energy) next to
the node's rules, and found by the engine with
[`find_average_energy`](@extref MessagePassingRulesBase.find_average_energy). The engine combines it with the entropies into these contributions:

| Type | Represents | Where used |
|------|-----------|-----------|
| [`DifferentialEntropy`](@ref) | `-∫ q log q`, the entropy of a marginal | factor and variable nodes |
| [`FactorBoundFreeEnergy`](@ref) | a factor node's local contribution: its average energy less its clusters' entropies | factor nodes |
| [`VariableBoundEntropy`](@ref) | a variable's contribution, its entropy weighted by its degree less one | variable nodes |

A node without an average energy for its clusters' marginals makes the free energy an error
that names the node, rather than a wrong number.

```@docs
score
FactorBoundFreeEnergy
VariableBoundEntropy
DifferentialEntropy
```
